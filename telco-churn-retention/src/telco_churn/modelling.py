"""Modelação: modelos base, rede neuronal, ensembles, avaliação e priorização P(churn) × CLTV.

Protocolo (descrito em docs/modelling.md):
    1. Todos os modelos são Pipelines (pré-processador + classificador), por isso a preparação
       dos dados é ajustada dentro de cada fold e não há fuga de informação.
    2. Os hiperparâmetros são escolhidos com RandomizedSearchCV (StratifiedKFold, 5 folds),
       só no treino.
    3. Todos os modelos, base e ensembles, são comparados na mesma validação cruzada, com
       nove métricas. O modelo final é o de maior ROC AUC em CV.
    4. O holdout (20%) é usado uma única vez, no fim, ao lado do Churn Score IBM.

A descrição de cada modelo está em docs/models.md e no dicionário MODEL_INFO abaixo.
"""
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform
from sklearn import metrics
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import ARTIFACTS_DIR, DATA_PROCESSED, RANDOM_STATE
from .preprocessing import build_preprocessors

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
THRESHOLD = 0.5   # limiar por omissão das métricas que precisam de uma decisão sim/não

# --- Modelos ----------------------------------------------------------------------------
# Modelos base: (pré-processador, estimador, espaço de procura do RandomizedSearchCV)
#   "linear" = imputação + escala + one-hot  -> modelos sensíveis à escala (LR, kNN, MLP)
#   "trees"  = imputação + one-hot, sem escala -> modelos de árvores
BASE_MODELS = {
    "Logistic Regression": (
        "linear", LogisticRegression(max_iter=3000),
        {"clf__C": loguniform(1e-3, 1e2)}),
    "Decision Tree": (
        "trees", DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 4, 5, 6, 8, 10], "clf__min_samples_leaf": [5, 10, 20, 50],
         "clf__criterion": ["gini", "entropy"]}),
    "k-Nearest Neighbors": (
        "linear", KNeighborsClassifier(),
        {"clf__n_neighbors": [15, 25, 35, 50, 75, 100], "clf__weights": ["uniform", "distance"]}),
    "Neural Network (MLP)": (
        "linear", MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=20, random_state=RANDOM_STATE),
        {"clf__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
         "clf__activation": ["relu", "tanh"], "clf__alpha": loguniform(1e-5, 1e-1),
         "clf__learning_rate_init": loguniform(1e-4, 1e-2)}),
    "Random Forest": (
        "trees", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        {"clf__max_depth": [None, 6, 10, 16], "clf__min_samples_leaf": [1, 3, 5, 10],
         "clf__max_features": ["sqrt", 0.3, 0.5]}),
    "Extra Trees": (
        "trees", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE),
        {"clf__max_depth": [None, 8, 12, 20], "clf__min_samples_leaf": [1, 3, 5, 10],
         "clf__max_features": ["sqrt", 0.3, 0.5]}),
    "Gradient Boosting": (
        "trees", HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__learning_rate": loguniform(0.02, 0.2), "clf__max_iter": [150, 300, 500],
         "clf__max_leaf_nodes": [7, 15, 31], "clf__min_samples_leaf": [10, 20, 50],
         "clf__l2_regularization": loguniform(1e-3, 10)}),
}
BASELINE = "Baseline (taxa de churn)"
ENSEMBLES = ["Soft Voting", "Stacking"]

# Membros dos ensembles: modelos fortes e de famílias diferentes (linear, rede neuronal,
# bagging, boosting). A árvore isolada e o kNN ficam de fora: são mais fracos e já estão
# representados, com vantagem, pelas florestas. A escolha é fixa à partida, e não feita a
# olhar para os resultados, para que a validação cruzada continue a ser um juiz justo.
ENSEMBLE_MEMBERS = ["Logistic Regression", "Neural Network (MLP)", "Random Forest",
                    "Extra Trees", "Gradient Boosting"]

# Descrição curta de cada modelo (tabelas do notebook). Detalhe completo em docs/models.md.
MODEL_INFO = pd.DataFrame([
    (BASELINE, "referência", "—",
     "Dá a todos os clientes a taxa de churn do treino. Qualquer modelo útil tem de a bater."),
    ("Logistic Regression", "linear", "linear",
     "Soma ponderada das variáveis passada por uma sigmoide. Simples, estável e interpretável."),
    ("Decision Tree", "árvore", "trees",
     "Sequência de perguntas sim/não sobre as variáveis. Fácil de explicar, mas instável."),
    ("k-Nearest Neighbors", "distância", "linear",
     "Prevê com a taxa de churn dos k clientes mais parecidos no treino."),
    ("Neural Network (MLP)", "rede neuronal", "linear",
     "Camadas de neurónios com ativação não linear; generaliza a regressão logística."),
    ("Random Forest", "ensemble: bagging", "trees",
     "Média de centenas de árvores, cada uma treinada numa amostra diferente dos dados."),
    ("Extra Trees", "ensemble: bagging", "trees",
     "Como o Random Forest, mas com cortes aleatórios: mais diversidade, menos variância."),
    ("Gradient Boosting", "ensemble: boosting", "trees",
     "Árvores pequenas em sequência, cada uma a corrigir os erros das anteriores."),
    ("Soft Voting", "ensemble: voting", "ambos",
     "Média das probabilidades dos membros do ensemble."),
    ("Stacking", "ensemble: stacking", "ambos",
     "Uma regressão logística aprende a melhor combinação das probabilidades dos membros."),
], columns=["modelo", "familia", "pre_processador", "como_funciona"]).set_index("modelo")


# --- Métricas ---------------------------------------------------------------------------
# Nome curto -> (descrição, maior é melhor?)
METRICS = {
    "auc": ("ROC AUC: capacidade de ordenar churners acima de não-churners", True),
    "pr_auc": ("PR AUC (average precision): ordenação com foco nos churners", True),
    "brier": ("Brier: erro quadrático médio das probabilidades (calibração)", False),
    "log_loss": ("Log loss: penaliza probabilidades confiantes e erradas", False),
    "accuracy": ("Accuracy: % de clientes bem classificados (limiar 0,5)", True),
    "balanced_accuracy": ("Balanced accuracy: média do recall em cada classe (limiar 0,5)", True),
    "precision": ("Precision: % de sinalizados que saíram de facto (limiar 0,5)", True),
    "recall": ("Recall: % de churners que foram sinalizados (limiar 0,5)", True),
    "f1": ("F1: média harmónica de precision e recall (limiar 0,5)", True),
}
LOWER_IS_BETTER = [m for m, (_, higher) in METRICS.items() if not higher]


def classification_metrics(y, p, threshold: float = THRESHOLD) -> dict:
    """As nove métricas para um vetor de probabilidades P(churn)."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    flagged = p >= threshold
    return {
        "auc": metrics.roc_auc_score(y, p),
        "pr_auc": metrics.average_precision_score(y, p),
        "brier": metrics.brier_score_loss(y, p),
        "log_loss": metrics.log_loss(y, p),
        "accuracy": metrics.accuracy_score(y, flagged),
        "balanced_accuracy": metrics.balanced_accuracy_score(y, flagged),
        "precision": metrics.precision_score(y, flagged, zero_division=0),
        "recall": metrics.recall_score(y, flagged),
        "f1": metrics.f1_score(y, flagged, zero_division=0),
    }


def score_metrics(y, p) -> dict:
    """Mantida por compatibilidade: as nove métricas (inclui auc, pr_auc e brier)."""
    return classification_metrics(y, p)


def metrics_glossary() -> pd.DataFrame:
    """Tabela com o significado de cada métrica e o sentido em que é melhor."""
    return pd.DataFrame(
        [(m, d, "maior" if h else "menor") for m, (d, h) in METRICS.items()],
        columns=["metrica", "significado", "melhor_quando"]).set_index("metrica")


# --- Dados ---------------------------------------------------------------------------
def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Lê train/test/joined e a configuração exportada pelo notebook 01.

    keep_default_na=False preserva o literal 'None' (Offer, Internet Type); só "" é missing.
    City não está em train/test: é recuperada da base unificada para o teste com TargetEncoder.
    """
    read = dict(keep_default_na=False, na_values=[""])
    train = pd.read_csv(DATA_PROCESSED / "train.csv", **read)
    test = pd.read_csv(DATA_PROCESSED / "test.csv", **read)
    joined = pd.read_csv(DATA_PROCESSED / "joined_para_scoring.csv", **read)
    city = pd.read_csv(DATA_PROCESSED / "telco_merged_full.csv", usecols=["Customer ID", "City"],
                       **read)
    train, test = (d.merge(city, on="Customer ID", how="left", validate="one_to_one")
                   for d in (train, test))
    cfg = joblib.load(ARTIFACTS_DIR / "preprocessing_config.joblib")
    return train, test, joined, cfg


# --- Construção dos modelos ----------------------------------------------------------
def make_preprocessors(num: list[str], cat: list[str], te: list[str] = ()) -> dict:
    """Pré-processadores do notebook 01 + TargetEncoder opcional (ex.: City).

    O TargetEncoder fica dentro do Pipeline: é ajustado só nos dados de treino de cada fold.
    """
    preps = build_preprocessors(num, cat)
    if te:
        for p in preps.values():
            p.transformers.append(("te", TargetEncoder(target_type="binary",
                                                       random_state=RANDOM_STATE), list(te)))
    return preps


def _pipeline(prep, clf) -> Pipeline:
    return Pipeline([("prep", clone(prep)), ("clf", clone(clf))])


def tune_base_models(X: pd.DataFrame, y: pd.Series, num: list[str], cat: list[str],
                     n_iter: int = 20) -> tuple[dict, pd.DataFrame]:
    """RandomizedSearchCV (StratifiedKFold, AUC) para cada modelo base — só no treino."""
    preps = make_preprocessors(num, cat)
    params, rows = {}, []
    for name, (kind, clf, space) in BASE_MODELS.items():
        search = RandomizedSearchCV(_pipeline(preps[kind], clf), space, n_iter=n_iter,
                                    scoring="roc_auc", cv=CV, n_jobs=-1,
                                    random_state=RANDOM_STATE)
        search.fit(X, y)
        params[name] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        rows.append({"modelo": name, "cv_auc": search.best_score_,
                     "melhores_parametros": params[name]})
    return params, pd.DataFrame(rows).set_index("modelo")


def build_models(num: list[str], cat: list[str], params: dict, te: list[str] = (),
                 baseline: bool = True) -> dict:
    """Linha de base + modelos base afinados + dois ensembles (soft voting e stacking).

    Os ensembles usam os modelos de ENSEMBLE_MEMBERS. O stacking aprende os pesos com
    previsões out-of-fold (cv=5), para não aprender com previsões feitas sobre o seu treino.
    """
    preps = make_preprocessors(num, cat, te)
    base = {name: _pipeline(preps[kind], clone(clf).set_params(**params.get(name, {})))
            for name, (kind, clf, _) in BASE_MODELS.items()}
    members = [(n.lower().replace(" ", "_").replace("(", "").replace(")", ""), clone(base[n]))
               for n in ENSEMBLE_MEMBERS]
    ensembles = {
        "Soft Voting": VotingClassifier(members, voting="soft", n_jobs=-1),
        "Stacking": StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000),
                                       cv=CV, stack_method="predict_proba", n_jobs=-1),
    }
    first = {BASELINE: _pipeline(preps["trees"], DummyClassifier(strategy="prior"))} if baseline else {}
    return first | base | ensembles


# --- Avaliação em validação cruzada --------------------------------------------------
def _fit_fold(model, X, y, tr, va):
    """Treina num fold e devolve P(churn) no treino e na validação desse fold."""
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    return tr, va, m.predict_proba(X.iloc[tr])[:, 1], m.predict_proba(X.iloc[va])[:, 1]


def cv_evaluate(models: dict, X: pd.DataFrame, y: pd.Series,
                threshold: float = THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """Avalia todos os modelos no mesmo 5-fold estratificado.

    Devolve:
        table: uma linha por modelo, com a média e o desvio-padrão de cada métrica nos folds
               de validação (ex.: auc, auc_std) e a média nos folds de treino (ex.: auc_train).
               A diferença auc_train - auc mede o sobreajuste.
        oof:   para cada modelo, as probabilidades out-of-fold (cada cliente do treino
               recebe a previsão do modelo que não o viu). Servem para as curvas ROC/PR,
               matrizes de confusão e análise de limiar, sem tocar no holdout.
    """
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    folds = list(CV.split(X, y))
    rows, oof = [], {}
    for name, model in models.items():
        runs = Parallel(n_jobs=-1)(delayed(_fit_fold)(model, X, y, tr, va) for tr, va in folds)
        p_oof = np.zeros(len(y))
        val_scores, train_scores = [], []
        for tr, va, p_tr, p_va in runs:
            p_oof[va] = p_va
            val_scores.append(classification_metrics(y.iloc[va], p_va, threshold))
            train_scores.append(classification_metrics(y.iloc[tr], p_tr, threshold))
        val, trn = pd.DataFrame(val_scores), pd.DataFrame(train_scores)
        row = {"modelo": name}
        for m in METRICS:
            row[m] = val[m].mean()
            row[f"{m}_std"] = val[m].std()
            row[f"{m}_train"] = trn[m].mean()
        rows.append(row)
        oof[name] = p_oof
    table = pd.DataFrame(rows).set_index("modelo").sort_values("auc", ascending=False)
    return table, oof


def cv_compare(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Versão resumida de cv_evaluate: AUC (± desvio), PR-AUC e Brier em 5-fold."""
    table, _ = cv_evaluate(models, X, y)
    return table[["auc", "auc_std", "pr_auc", "brier"]]


def metrics_view(table: pd.DataFrame, kind: str = "cv") -> pd.DataFrame:
    """Só as nove métricas, de CV (kind='cv') ou de treino (kind='train')."""
    cols = list(METRICS) if kind == "cv" else [f"{m}_train" for m in METRICS]
    out = table[cols].copy()
    out.columns = list(METRICS)
    return out


def select_best(table: pd.DataFrame, metric: str = "auc") -> str:
    """Modelo com melhor valor médio em CV (a linha de base não é candidata)."""
    return table.drop(index=BASELINE, errors="ignore")[metric].idxmax()


def tied_with_best(table: pd.DataFrame, best: str, metric: str = "auc") -> list[str]:
    """Modelos cuja diferença para o melhor é menor do que o desvio-padrão do melhor em CV.

    Na prática estão empatados: a diferença está dentro do ruído entre folds.
    """
    t = table.drop(index=BASELINE, errors="ignore")
    gap = t.loc[best, metric] - t[metric]
    return list(t.index[(gap < t.loc[best, f"{metric}_std"]) & (t.index != best)])


def prediction_correlation(oof: dict, models: list[str] | None = None) -> pd.DataFrame:
    """Correlação entre as probabilidades OOF dos modelos (diversidade para os ensembles)."""
    models = models or [m for m in oof if m != BASELINE]
    return pd.DataFrame({m: oof[m] for m in models}).corr()


def threshold_table(y, p, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """Precision, recall, F1 e % de clientes sinalizados em vários limiares de decisão."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in thresholds:
        f = p >= t
        rows.append({"limiar": t, "precision": metrics.precision_score(y, f, zero_division=0),
                     "recall": metrics.recall_score(y, f),
                     "f1": metrics.f1_score(y, f, zero_division=0),
                     "pct_sinalizados": f.mean()})
    return pd.DataFrame(rows).set_index("limiar")


def best_f1_threshold(y, p) -> float:
    """Limiar que maximiza o F1 (referência; a escolha final depende dos custos de negócio)."""
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [metrics.f1_score(y, np.asarray(p) >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


# --- Avaliação final -------------------------------------------------------------------
def holdout_report(probas: dict, y: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Nove métricas no holdout para cada modelo e para o Churn Score IBM (0–100 -> 0–1)."""
    rows = {name: classification_metrics(y, p) for name, p in probas.items()}
    rows["Churn Score IBM (benchmark)"] = classification_metrics(y, benchmark / 100)
    return pd.DataFrame(rows).T.sort_values("auc", ascending=False)


# --- Priorização -------------------------------------------------------------------------
def priority_list(df: pd.DataFrame, proba: np.ndarray, id_col: str = "Customer ID",
                  cltv_col: str = "CLTV") -> pd.DataFrame:
    """Ordena clientes por perda esperada = P(churn) × CLTV."""
    out = df[[id_col, cltv_col]].copy()
    out["p_churn"] = proba
    out["perda_esperada"] = out["p_churn"] * out[cltv_col]
    out = out.sort_values("perda_esperada", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["decil"] = pd.qcut(out["rank"], 10, labels=range(1, 11)).astype(int)
    return out


def capture_curve(y: pd.Series, cltv: pd.Series, orderings: dict) -> pd.DataFrame:
    """% do CLTV dos clientes que saíram captado ao contactar os top-k% de cada ordenação."""
    y, cltv = np.asarray(y), np.asarray(cltv, dtype=float)
    lost = cltv * y
    frac = np.arange(1, len(y) + 1) / len(y)
    curves = {"% clientes contactados": frac}
    for name, score in orderings.items():
        order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
        curves[name] = np.cumsum(lost[order]) / lost.sum()
    return pd.DataFrame(curves)


def capture_at(curve: pd.DataFrame, cuts=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Resumo da curva de captura em pontos de corte (ex.: orçamento para 20% da base)."""
    x = curve["% clientes contactados"].to_numpy()
    idx = [np.searchsorted(x, c) for c in cuts]
    out = curve.drop(columns="% clientes contactados").iloc[idx].T
    out.columns = [f"top {c:.0%}" for c in cuts]
    return out
