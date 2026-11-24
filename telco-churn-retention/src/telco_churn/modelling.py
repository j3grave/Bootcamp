"""Modelação: modelos, avaliação em validação cruzada, holdout e priorização P(churn) × CLTV.

O protocolo e a descrição de cada modelo estão no README.md ("Fase 2 — Modelação").
"""
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import ARTIFACTS_DIR, DATA_PROCESSED, RANDOM_STATE
from .preprocessing import build_preprocessors

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
THRESHOLD = 0.5            # limiar por omissão das métricas que precisam de uma decisão sim/não
CAPACITY_SHARE = 0.20      # a equipa de retenção contacta, no máximo, 20% dos clientes
LR_MARGIN = 0.01           # margem de AUC para preferirmos a regressão logística (ver select_model)

# --- Modelos ----------------------------------------------------------------------------
# Modelo: (pré-processador, estimador, grelha do GridSearchCV)
#   "linear" = imputação + escala + one-hot -> regressão logística
#   "trees"  = imputação + one-hot, sem escala -> árvores
BASE_MODELS = {
    "Logistic Regression": (
        "linear", LogisticRegression(max_iter=3000),
        {"clf__C": [0.1, 1, 10]}),
    "Decision Tree": (
        "trees", DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [20, 50]}),
    "Random Forest": (
        "trees", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        {"clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.3]}),
    "Gradient Boosting": (
        "trees", HistGradientBoostingClassifier(max_iter=500, random_state=RANDOM_STATE),
        {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15]}),
}
BASELINE = "Baseline (taxa de churn)"
RULE_BASELINE = "Regra de negócio (pontos)"
BENCHMARK = "Churn Score IBM (benchmark)"
REFERENCES = [BASELINE, RULE_BASELINE, BENCHMARK]   # comparam-se, mas nunca são escolhidas

# Descrição curta de cada modelo (tabela do notebook 02)
MODEL_INFO = pd.DataFrame([
    (BASELINE, "referência", "Dá a todos os clientes a taxa de churn do treino."),
    (RULE_BASELINE, "referência (sem ML)",
     "1 ponto por fator de risco: Month-to-Month, tenure ≤ 12 meses, fibra ótica (0 a 3)."),
    ("Logistic Regression", "linear", "Soma ponderada das variáveis passada por uma sigmoide."),
    ("Decision Tree", "árvore", "Sequência de perguntas sim/não sobre as variáveis."),
    ("Random Forest", "ensemble: bagging", "Média de 300 árvores, cada uma treinada numa amostra."),
    ("Gradient Boosting", "ensemble: boosting",
     "Árvores pequenas em sequência, cada uma corrige os erros das anteriores."),
], columns=["modelo", "familia", "como_funciona"]).set_index("modelo")


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
SUMMARY_COLS = ["auc", "auc_std", "pr_auc", "brier"]   # resumo usado na tabela de variantes


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
    """Lê train/test/joined e a configuração da fase 1 (keep_default_na=False preserva 'None')."""
    read = dict(keep_default_na=False, na_values=[""])
    train = pd.read_csv(DATA_PROCESSED / "train.csv", **read)
    test = pd.read_csv(DATA_PROCESSED / "test.csv", **read)
    joined = pd.read_csv(DATA_PROCESSED / "joined_para_scoring.csv", **read)
    cfg = joblib.load(ARTIFACTS_DIR / "preprocessing_config.joblib")
    return train, test, joined, cfg


# --- Regra de negócio (linha de base sem ML) ----------------------------------------------
class BusinessRuleScore(ClassifierMixin, BaseEstimator):
    """Regra sem ML: 1 ponto por fator de risco (Month-to-Month, tenure ≤ 12, fibra), de 0 a 3.

    fit() só regista a taxa de churn por nº de pontos, para dar uma probabilidade.
    """

    def points(self, X: pd.DataFrame) -> np.ndarray:
        """Nº de fatores de risco de cada cliente (0 a 3)."""
        return ((X["Contract"] == "Month-to-Month").astype(int)
                + (X["Tenure in Months"] <= 12).astype(int)
                + (X["Internet Type"] == "Fiber Optic").astype(int)).to_numpy()

    def fit(self, X: pd.DataFrame, y):
        """Guarda a taxa de churn observada para cada nº de pontos."""
        y = np.asarray(y)
        rates = pd.Series(y).groupby(self.points(X)).mean()
        self.rate_ = rates.reindex(range(4)).fillna(y.mean()).to_numpy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(churn) = taxa de churn do treino com o mesmo nº de pontos."""
        p = self.rate_[self.points(X)]
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Churn previsto (1) quando P(churn) ≥ 0,5."""
        return (self.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)


# --- Construção dos modelos ----------------------------------------------------------
def _pipeline(prep, clf) -> Pipeline:
    """Pré-processador e classificador num só Pipeline, ajustado dentro de cada fold."""
    return Pipeline([("prep", clone(prep)), ("clf", clone(clf))])


def tune_base_models(X: pd.DataFrame, y: pd.Series, num: list[str],
                     cat: list[str]) -> tuple[dict, pd.DataFrame]:
    """GridSearchCV (5 folds estratificados, ROC AUC) para cada modelo, só no treino.

    Devolve os melhores hiperparâmetros por modelo e uma tabela com o AUC de cada um.
    """
    preps = build_preprocessors(num, cat)
    params, rows = {}, []
    for name, (kind, clf, grid) in BASE_MODELS.items():
        search = GridSearchCV(_pipeline(preps[kind], clf), grid, scoring="roc_auc", cv=CV,
                              n_jobs=-1)
        search.fit(X, y)
        params[name] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        rows.append({"modelo": name, "cv_auc": search.best_score_,
                     "melhores_parametros": params[name]})
    return params, pd.DataFrame(rows).set_index("modelo")


def build_models(num: list[str], cat: list[str], params: dict, baseline: bool = True) -> dict:
    """Modelos com os hiperparâmetros escolhidos e, se baseline=True, as duas linhas de base."""
    preps = build_preprocessors(num, cat)
    models = {}
    if baseline:
        models = {BASELINE: _pipeline(preps["trees"], DummyClassifier(strategy="prior")),
                  RULE_BASELINE: BusinessRuleScore()}
    for name, (kind, clf, _) in BASE_MODELS.items():
        models[name] = _pipeline(preps[kind], clone(clf).set_params(**params.get(name, {})))
    return models


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


def cv_evaluate(models: dict, X: pd.DataFrame, y: pd.Series,
                benchmarks: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Avalia todos os modelos no mesmo 5-fold estratificado.

    benchmarks: {nome: score em [0, 1]} que não se treinam (o Churn Score IBM / 100).
    Devolve (tabela, oof): a tabela tem a média e o desvio-padrão de cada métrica em validação
    e a média no treino (auc_train − auc = sobreajuste); oof tem as P(churn) out-of-fold.
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
            val_scores.append(classification_metrics(y.iloc[va], p_va))
            train_scores.append(classification_metrics(y.iloc[tr], p_tr))
        rows.append(_summary_row(name, val_scores, train_scores))
        oof[name] = p_oof
    for name, score in (benchmarks or {}).items():
        score = np.asarray(score, dtype=float)
        rows.append(_summary_row(
            name, [classification_metrics(y.iloc[va], score[va]) for _, va in folds],
            [classification_metrics(y.iloc[tr], score[tr]) for tr, _ in folds]))
        oof[name] = score
    table = pd.DataFrame(rows).set_index("modelo").sort_values("auc", ascending=False)
    return table, oof


def metrics_view(table: pd.DataFrame, kind: str = "cv") -> pd.DataFrame:
    """Só as nove métricas, de CV (kind='cv') ou de treino (kind='train')."""
    cols = list(METRICS) if kind == "cv" else [f"{m}_train" for m in METRICS]
    out = table[cols].copy()
    out.columns = list(METRICS)
    return out


def select_model(table: pd.DataFrame) -> tuple[str, str]:
    """Escolhe o modelo final e devolve (nome, justificação).

    Decidimos escolher o maior ROC AUC em CV. Se a regressão logística ficar a menos de
    LR_MARGIN do melhor, escolhemos a regressão logística, por ser mais fácil de explicar.
    """
    candidates = table.drop(index=REFERENCES, errors="ignore")["auc"]
    best = candidates.idxmax()
    gap = candidates[best] - candidates["Logistic Regression"]
    if best != "Logistic Regression" and gap < LR_MARGIN:
        return "Logistic Regression", (
            f"{best} tem o maior AUC em CV, mas a Logistic Regression fica a {gap:.3f} "
            f"(menos de {LR_MARGIN}) e é mais fácil de explicar.")
    return best, (f"{best} tem o maior AUC em CV ({candidates[best]:.3f}); a Logistic "
                  f"Regression fica a {gap:.3f}, acima da margem de {LR_MARGIN}.")


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


def capacity_threshold(p, share: float = CAPACITY_SHARE) -> float:
    """P(churn) a partir da qual um cliente está no top `share` (capacidade da equipa)."""
    return float(np.quantile(np.asarray(p, dtype=float), 1 - share))


def decision_table(y, p, thresholds: dict) -> pd.DataFrame:
    """Precision, recall, F1 e % sinalizados para limiares com nome.

    thresholds: {critério: limiar}, ex.: {"0,5": 0.5, "top 20%": 0.44}.
    """
    t = threshold_table(y, p, tuple(thresholds.values()))
    t.insert(0, "criterio", list(thresholds))
    return t.reset_index().set_index("criterio")


def in_top_capacity(p, cut: float) -> np.ndarray:
    """True para os clientes dentro da capacidade de contacto (P(churn) ≥ corte do top 20%)."""
    return np.asarray(p, dtype=float) >= cut


def calibration_table(y, p) -> pd.DataFrame:
    """Por faixa de P(churn) (< 30%, 30–60%, ≥ 60%): nº de clientes, P média e churn real.

    Se as duas últimas colunas forem parecidas, P(churn) pode ler-se como uma percentagem real.
    """
    faixa = pd.cut(np.asarray(p, dtype=float), bins=[0, 0.3, 0.6, np.inf], right=False,
                   labels=["< 30%", "30–60%", "≥ 60%"])
    t = (pd.DataFrame({"faixa_P": faixa, "p": np.asarray(p, dtype=float), "y": np.asarray(y)})
         .groupby("faixa_P", observed=False)
         .agg(clientes=("y", "size"), p_media=("p", "mean"), churn_real=("y", "mean")))
    t["pct_base"] = t["clientes"] / t["clientes"].sum()
    return t


# --- Treino final e holdout -----------------------------------------------------------
def fit_predict(models: dict, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame) -> dict:
    """Treina cada modelo no treino completo e devolve P(churn) no teste, por modelo."""
    return {name: m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            for name, m in models.items()}


def save_bundle(path, model, name: str, features: list[str], target: str, best_params: dict,
                cv: dict, holdout: dict, **extra) -> None:
    """Guarda o modelo e o que a app e os testes precisam para o usar (dicionário joblib)."""
    joblib.dump({"model": model, "name": name, "features": features, "target": target,
                 "best_params": best_params, "cv": cv, "holdout": holdout, **extra}, path)


def holdout_report(probas: dict, y: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Nove métricas no holdout para cada modelo e para o Churn Score IBM (0–100 -> 0–1)."""
    rows = {name: classification_metrics(y, p) for name, p in probas.items()}
    rows[BENCHMARK] = classification_metrics(y, benchmark / 100)
    return pd.DataFrame(rows).T.sort_values("auc", ascending=False)


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
    """Ordena os clientes por perda esperada = P(churn) × CLTV."""
    out = df[[id_col, cltv_col]].copy()
    out["p_churn"] = proba
    out["perda_esperada"] = out["p_churn"] * out[cltv_col]
    out = out.sort_values("perda_esperada", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    # Decil 1 = 10% com maior perda esperada (com menos de 10 clientes, um grupo por cliente)
    out["decil"] = pd.qcut(out["rank"], min(10, len(out)), labels=False).astype(int) + 1
    return out


def capture_curve(y: pd.Series, cltv: pd.Series, orderings: dict) -> pd.DataFrame:
    """% do CLTV dos clientes que saíram apanhado ao contactar os top-k% de cada ordenação."""
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
