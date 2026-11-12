"""Modelação: modelos base, rede neuronal, ensembles, avaliação e priorização P(churn) × CLTV.

Protocolo (descrito em docs/modelling.md):
    1. Todos os modelos são Pipelines (pré-processador + classificador), por isso a preparação
       dos dados é ajustada dentro de cada fold e não há fuga de informação.
    2. Os hiperparâmetros são escolhidos com RandomizedSearchCV (StratifiedKFold, 5 folds),
       só no treino.
    3. Todos os modelos, base e ensembles, são comparados na mesma validação cruzada, com
       nove métricas, ao lado de três referências: a taxa de churn (Dummy), uma regra de negócio
       sem ML e o Churn Score IBM (avaliado nos mesmos folds).
    4. Regra de parcimónia: entre o modelo de maior AUC e os empatados com ele (diferença menor
       do que um desvio-padrão entre folds), escolhe-se o mais simples (COMPLEXITY_ORDER).
    5. O holdout (20%) é usado uma única vez, no fim, ao lado das mesmas referências.

A descrição de cada modelo está em docs/models.md e no dicionário MODEL_INFO abaixo.
"""
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
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
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)   # limiares testados nas curvas e no melhor F1

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
RULE_BASELINE = "Regra de negócio (pontos)"
BENCHMARK = "Churn Score IBM (benchmark)"
REFERENCES = [BASELINE, RULE_BASELINE, BENCHMARK]   # comparam-se, mas nunca são escolhidas

# Regra de parcimónia: do mais simples (e mais fácil de explicar) para o mais complexo.
# Entre modelos empatados com o melhor em CV, escolhe-se o que aparece primeiro nesta lista.
COMPLEXITY_ORDER = ["Logistic Regression", "Decision Tree", "k-Nearest Neighbors",
                    "Random Forest", "Extra Trees", "Gradient Boosting", "Neural Network (MLP)",
                    "Soft Voting", "Stacking"]
ENSEMBLES = ["Soft Voting", "Stacking"]

# Membros dos ensembles: modelos fortes e de famílias diferentes (linear, rede neuronal,
# bagging, boosting). A árvore isolada e o kNN ficam de fora: são mais fracos e já estão
# representados, com vantagem, pelas florestas. A escolha é fixa à partida, e não feita a
# olhar para os resultados, para que a validação cruzada continue a ser um juiz justo.
# O sklearn pede um nome curto para cada membro (sem espaços), por isso guardamos os dois.
ENSEMBLE_MEMBERS = {
    "Logistic Regression": "lr",
    "Neural Network (MLP)": "mlp",
    "Random Forest": "rf",
    "Extra Trees": "et",
    "Gradient Boosting": "gb",
}

# Descrição curta de cada modelo (tabelas do notebook). Detalhe completo em docs/models.md.
MODEL_INFO = pd.DataFrame([
    (BASELINE, "referência", "—",
     "Dá a todos os clientes a taxa de churn do treino. Qualquer modelo útil tem de a bater."),
    (RULE_BASELINE, "referência (sem ML)", "—",
     "1 ponto por fator de risco: Month-to-Month, tenure ≤ 12 meses, fibra ótica (0 a 3)."),
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
SUMMARY_COLS = ["auc", "auc_std", "pr_auc", "brier"]   # resumo usado nas tabelas de variantes


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


# --- Regra de negócio (linha de base sem ML) ----------------------------------------------
class BusinessRuleScore(ClassifierMixin, BaseEstimator):
    """Regra simples, sem ML: 1 ponto por cada fator de risco, de 0 a 3 pontos.

    Fatores fixados à partida a partir da EDA (associação mais forte com o churn no treino):
    contrato Month-to-Month, tenure até 12 meses e internet por fibra ótica.
    fit() só regista a taxa de churn de cada nº de pontos no treino, para a regra dar uma
    probabilidade (precisa-se dela para o Brier e o log loss; o AUC só depende dos pontos).
    """

    def points(self, X: pd.DataFrame) -> np.ndarray:
        return ((X["Contract"] == "Month-to-Month").astype(int)
                + (X["Tenure in Months"] <= 12).astype(int)
                + (X["Internet Type"] == "Fiber Optic").astype(int)).to_numpy()

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        rates = pd.Series(y).groupby(self.points(X)).mean()
        self.rate_ = rates.reindex(range(4)).fillna(y.mean()).to_numpy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.rate_[self.points(X)]
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)


# --- Construção dos modelos ----------------------------------------------------------
def make_preprocessors(num: list[str], cat: list[str], te: list[str] = ()) -> dict:
    """Pré-processadores do notebook 01 + TargetEncoder opcional (ex.: City).

    O TargetEncoder fica dentro do Pipeline: é ajustado só nos dados de treino de cada fold.
    Recebe o mesmo StratifiedKFold (CV) do resto do projeto, para dar sempre o mesmo resultado.
    """
    preps = build_preprocessors(num, cat)
    if te:
        for p in preps.values():
            p.transformers.append(("te", TargetEncoder(target_type="binary", cv=CV), list(te)))
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
    """Linhas de base (Dummy e regra de negócio) + modelos afinados + dois ensembles.

    Os ensembles usam os modelos de ENSEMBLE_MEMBERS. O stacking aprende os pesos com
    previsões out-of-fold (cv=5), para não aprender com previsões feitas sobre o seu treino.
    """
    preps = make_preprocessors(num, cat, te)
    base = {name: _pipeline(preps[kind], clone(clf).set_params(**params.get(name, {})))
            for name, (kind, clf, _) in BASE_MODELS.items()}
    members = []
    for name, short_name in ENSEMBLE_MEMBERS.items():
        members.append((short_name, clone(base[name])))
    ensembles = {
        "Soft Voting": VotingClassifier(members, voting="soft", n_jobs=-1),
        "Stacking": StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000),
                                       cv=CV, stack_method="predict_proba", n_jobs=-1),
    }
    first = {}
    if baseline:
        first = {BASELINE: _pipeline(preps["trees"], DummyClassifier(strategy="prior")),
                 RULE_BASELINE: BusinessRuleScore()}
    return first | base | ensembles


# --- Avaliação em validação cruzada --------------------------------------------------
def _fit_fold(model, X, y, tr, va):
    """Treina num fold e devolve P(churn) no treino e na validação desse fold."""
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    return tr, va, m.predict_proba(X.iloc[tr])[:, 1], m.predict_proba(X.iloc[va])[:, 1]


def _summary_row(name: str, val_scores: list, train_scores: list) -> dict:
    """Média e desvio-padrão por métrica nos folds de validação, e média nos de treino."""
    val, trn = pd.DataFrame(val_scores), pd.DataFrame(train_scores)
    row = {"modelo": name}
    for m in METRICS:
        row[m] = val[m].mean()
        row[f"{m}_std"] = val[m].std()
        row[f"{m}_train"] = trn[m].mean()
    return row


def cv_evaluate(models: dict, X: pd.DataFrame, y: pd.Series, threshold: float = THRESHOLD,
                benchmarks: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Avalia todos os modelos no mesmo 5-fold estratificado.

    benchmarks: {nome: score em [0, 1]} de referências que não se treinam (ex.: o Churn Score
    IBM / 100). São avaliadas nos mesmos folds, para comparar com os modelos de forma justa.

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
        rows.append(_summary_row(name, val_scores, train_scores))
        oof[name] = p_oof
    for name, score in (benchmarks or {}).items():
        score = np.asarray(score, dtype=float)
        rows.append(_summary_row(
            name, [classification_metrics(y.iloc[va], score[va], threshold) for _, va in folds],
            [classification_metrics(y.iloc[tr], score[tr], threshold) for tr, _ in folds]))
        oof[name] = score
    table = pd.DataFrame(rows).set_index("modelo").sort_values("auc", ascending=False)
    return table, oof


def metrics_view(table: pd.DataFrame, kind: str = "cv") -> pd.DataFrame:
    """Só as nove métricas, de CV (kind='cv') ou de treino (kind='train')."""
    cols = list(METRICS) if kind == "cv" else [f"{m}_train" for m in METRICS]
    out = table[cols].copy()
    out.columns = list(METRICS)
    return out


def select_best(table: pd.DataFrame, metric: str = "auc") -> str:
    """Modelo com melhor valor médio em CV (as referências não são candidatas)."""
    return table.drop(index=REFERENCES, errors="ignore")[metric].idxmax()


def tied_with_best(table: pd.DataFrame, best: str, metric: str = "auc") -> list[str]:
    """Modelos cuja diferença para o melhor é menor do que o desvio-padrão do melhor em CV.

    Na prática estão empatados: a diferença está dentro do ruído entre folds.
    """
    t = table.drop(index=REFERENCES, errors="ignore")
    gap = t.loc[best, metric] - t[metric]
    return list(t.index[(gap < t.loc[best, f"{metric}_std"]) & (t.index != best)])


def select_parsimonious(table: pd.DataFrame, metric: str = "auc") -> tuple[str, pd.DataFrame, str]:
    """Regra de parcimónia: o mais simples entre o melhor e os empatados com ele.

    Devolve (modelo escolhido, tabela da decisão, justificação em texto).
    """
    best = select_best(table, metric)
    candidates = [best] + tied_with_best(table, best, metric)
    chosen = min(candidates, key=COMPLEXITY_ORDER.index)
    decision = table.drop(index=REFERENCES, errors="ignore")[[metric, f"{metric}_std"]].copy()
    decision["ordem_complexidade"] = [COMPLEXITY_ORDER.index(m) + 1 for m in decision.index]
    decision["empatado_com_melhor"] = decision.index.isin(candidates)
    decision["maior_auc"] = decision.index == best
    decision["escolhido"] = decision.index == chosen
    if chosen == best:
        reason = (f"{best} tem o maior {metric.upper()} em CV e é o mais simples entre os "
                  f"empatados ({', '.join(candidates)}).")
    else:
        reason = (f"{best} tem o maior {metric.upper()} em CV, mas {chosen} está empatado "
                  f"(diferença < 1 desvio-padrão) e é mais simples: escolhe-se {chosen}.")
    return chosen, decision.sort_values(metric, ascending=False), reason


def prediction_correlation(oof: dict, models: list[str] | None = None) -> pd.DataFrame:
    """Correlação entre as probabilidades OOF dos modelos (diversidade para os ensembles)."""
    models = models or [m for m in oof if m not in REFERENCES]
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
    f1 = threshold_table(y, p, THRESHOLD_GRID)["f1"].to_numpy()
    return float(THRESHOLD_GRID[int(np.argmax(f1))])


def capacity_thresholds(p, shares=(0.10, 0.20)) -> dict:
    """Limiar de P(churn) que sinaliza exatamente os top x% de clientes (capacidade da equipa)."""
    p = np.asarray(p, dtype=float)
    return {share: float(np.quantile(p, 1 - share)) for share in shares}


def decision_table(y, p, thresholds: dict) -> pd.DataFrame:
    """Precision, recall, F1 e % sinalizados para limiares de decisão com nome.

    thresholds: {critério: limiar}, ex.: {"0,5": 0.5, "melhor F1": 0.37, "top 10%": 0.81}.
    """
    t = threshold_table(y, p, tuple(thresholds.values()))
    t.insert(0, "criterio", list(thresholds))
    return t.reset_index().set_index("criterio")


# Duas leituras diferentes de P(churn), mostradas lado a lado na app e no lote:
#   nível de risco      -> absoluto: "quão provável é sair?" (o modelo está calibrado, por isso
#                          P ≈ taxa de churn observada; ver 13c_niveis_risco_holdout.csv)
#   prioridade contacto -> relativo: "cabe na capacidade da equipa?" (top 10% / 20% da base)
RISK_LEVELS = {"médio": 0.30, "alto": 0.60}   # baixo < 30% ≤ médio < 60% ≤ alto
RISK_ORDER = ["baixo", "médio", "alto"]
PRIORITY_ORDER = ["top 10%", "top 20%", "fora da capacidade"]


def risk_level(p, cuts: dict = RISK_LEVELS) -> np.ndarray:
    """Nível de risco absoluto: baixo (< 30%), médio (30–60%) ou alto (≥ 60%)."""
    p = np.asarray(p, dtype=float)
    return np.select([p >= cuts["alto"], p >= cuts["médio"]], ["alto", "médio"], default="baixo")


def contact_priority(p, capacity_cuts: dict) -> np.ndarray:
    """Prioridade de contacto pela capacidade da equipa: 'top 10%', 'top 20%' da base ou fora.

    capacity_cuts: {"top 10%": limiar, "top 20%": limiar}, os quantis de P(churn) fixados nas
    previsões out-of-fold do treino (capacity_thresholds), guardados no final_model.joblib.
    """
    p = np.asarray(p, dtype=float)
    return np.select([p >= capacity_cuts["top 10%"], p >= capacity_cuts["top 20%"]],
                     ["top 10%", "top 20%"], default="fora da capacidade")


def risk_level_table(y, p, cuts: dict = RISK_LEVELS) -> pd.DataFrame:
    """Clientes, % da base e taxa de churn observada em cada nível de risco."""
    t = (pd.DataFrame({"nivel_risco": risk_level(p, cuts), "y": np.asarray(y)})
         .groupby("nivel_risco")["y"].agg(clientes="size", churn_real="mean")
         .reindex(RISK_ORDER).fillna(0))
    t["pct_base"] = t["clientes"] / t["clientes"].sum()
    t["p_media"] = pd.Series(np.asarray(p)).groupby(risk_level(p, cuts)).mean().reindex(RISK_ORDER)
    return t


# --- Decisões e treino final ----------------------------------------------------------
def include_city(cv_table: pd.DataFrame, best: str, city_cv: pd.DataFrame) -> tuple[float, bool]:
    """Ganho de AUC em CV ao juntar City e se compensa incluí-la.

    Só compensa se o ganho for maior do que o desvio-padrão do melhor modelo entre folds;
    abaixo disso, a diferença está dentro do ruído.
    """
    gain = city_cv["auc"].iloc[0] - cv_table.loc[best, "auc"]
    return gain, bool(gain > cv_table.loc[best, "auc_std"])


def fit_predict(models: dict, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame) -> dict:
    """Treina cada modelo no treino completo e devolve P(churn) no teste, por modelo."""
    return {name: m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            for name, m in models.items()}


def save_bundle(path, model, name: str, features: list[str], target: str, best_params: dict,
                cv: dict, holdout: dict, **extra) -> None:
    """Guarda o modelo e o que é preciso para o voltar a usar (joblib).

    Chaves: model, name, features, target, best_params, cv, holdout e as de `extra`
    (ex.: population). A app e os testes leem este dicionário.
    """
    joblib.dump({"model": model, "name": name, "features": features, "target": target,
                 "best_params": best_params, "cv": cv, "holdout": holdout, **extra}, path)


# --- Avaliação final -------------------------------------------------------------------
def holdout_report(probas: dict, y: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Nove métricas no holdout para cada modelo e para o Churn Score IBM (0–100 -> 0–1)."""
    rows = {name: classification_metrics(y, p) for name, p in probas.items()}
    rows[BENCHMARK] = classification_metrics(y, benchmark / 100)
    return pd.DataFrame(rows).T.sort_values("auc", ascending=False)


def paired_bootstrap_auc_diff(y, p_model, p_reference, n_boot: int = 2000,
                              random_state: int = RANDOM_STATE, level: float = 0.95) -> dict:
    """Intervalo de confiança (bootstrap emparelhado) para AUC(modelo) − AUC(referência).

    Em cada reamostra sorteiam-se clientes com reposição e calculam-se as duas AUC nos MESMOS
    clientes, por isso a variação da amostra afeta os dois scores da mesma forma. O intervalo
    é o percentil (2,5% e 97,5% para 95%). Se incluir zero, a diferença não é
    estatisticamente distinguível de zero neste holdout. É um reporte: não decide nada.
    """
    y, a, b = np.asarray(y), np.asarray(p_model, dtype=float), np.asarray(p_reference, dtype=float)
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():   # reamostra só com uma classe: AUC indefinida
            continue
        diffs.append(metrics.roc_auc_score(y[i], a[i]) - metrics.roc_auc_score(y[i], b[i]))
    diffs = np.array(diffs)
    tail = (1 - level) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    return {"auc_modelo": metrics.roc_auc_score(y, a), "auc_referencia": metrics.roc_auc_score(y, b),
            "diferenca": metrics.roc_auc_score(y, a) - metrics.roc_auc_score(y, b),
            "ic_inferior": low, "ic_superior": high, "nivel": level,
            "reamostras": len(diffs), "pct_reamostras_modelo_melhor": float(np.mean(diffs > 0)),
            "ic_inclui_zero": bool(low <= 0 <= high), "random_state": random_state}


def comparison_table(rows: list) -> pd.DataFrame:
    """Métricas de vários modelos, cada um no conjunto de clientes indicado.

    rows: lista de (modelo, conjunto, y, P(churn)). Serve para comparar modelos treinados em
    populações diferentes: só faz sentido comparar linhas com o mesmo conjunto.
    """
    table = []
    for model_name, subset_name, y, p in rows:
        row = {"modelo": model_name, "conjunto": subset_name,
               "clientes": len(y), "taxa_churn": np.mean(y)}
        row.update(classification_metrics(y, p))
        table.append(row)
    return pd.DataFrame(table).set_index(["conjunto", "modelo"])


def permutation_ranking(model, X: pd.DataFrame, y) -> pd.Series:
    """Queda média do AUC quando se baralha cada coluna (5 repetições), da maior para a menor.

    Valores negativos (baralhar não piorou o modelo) passam a 0.
    """
    pi = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=0,
                                n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=X.columns, name="queda_auc")
    return importance.clip(lower=0).sort_values(ascending=False)


# --- Priorização -------------------------------------------------------------------------
def priority_list(df: pd.DataFrame, proba: np.ndarray, id_col: str = "Customer ID",
                  cltv_col: str = "CLTV") -> pd.DataFrame:
    """Ordena clientes por perda esperada = P(churn) × CLTV."""
    out = df[[id_col, cltv_col]].copy()
    out["p_churn"] = proba
    out["perda_esperada"] = out["p_churn"] * out[cltv_col]
    out = out.sort_values("perda_esperada", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    # Decil 1 = 10% com maior perda esperada (com menos de 10 clientes, um grupo por cliente)
    out["decil"] = pd.qcut(out["rank"], min(10, len(out)), labels=False).astype(int) + 1
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
