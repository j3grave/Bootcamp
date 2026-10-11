"""Modelação ensemble: tuning em CV, ensembles, holdout final e priorização P(churn) × CLTV."""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from .config import ARTIFACTS_DIR, DATA_PROCESSED, RANDOM_STATE
from .preprocessing import build_preprocessors

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}

# Modelos base: (pré-processador, estimador, espaço de procura para o RandomizedSearchCV)
BASE_MODELS = {
    "Logistic Regression": (
        "linear", LogisticRegression(max_iter=3000),
        {"clf__C": loguniform(1e-3, 1e2)}),
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
ENSEMBLES = ["Soft Voting", "Stacking"]


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


def build_models(num: list[str], cat: list[str], params: dict, te: list[str] = ()) -> dict:
    """Modelos base afinados + dois ensembles (soft voting e stacking com meta-modelo logístico)."""
    preps = make_preprocessors(num, cat, te)
    base = {name: _pipeline(preps[kind], clone(clf).set_params(**params.get(name, {})))
            for name, (kind, clf, _) in BASE_MODELS.items()}
    estimators = [(n.lower().replace(" ", "_"), m) for n, m in base.items()]
    ensembles = {
        "Soft Voting": VotingClassifier(estimators, voting="soft", n_jobs=-1),
        "Stacking": StackingClassifier(estimators, final_estimator=LogisticRegression(max_iter=1000),
                                       cv=CV, stack_method="predict_proba", n_jobs=-1),
    }
    return base | ensembles


def cv_compare(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """AUC, PR-AUC e Brier em 5-fold estratificado (média ± desvio-padrão)."""
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, scoring=SCORING, cv=CV, n_jobs=-1)
        rows.append({"modelo": name,
                     "auc": res["test_auc"].mean(), "auc_std": res["test_auc"].std(),
                     "pr_auc": res["test_pr_auc"].mean(),
                     "brier": -res["test_brier"].mean()})
    return pd.DataFrame(rows).set_index("modelo").sort_values("auc", ascending=False)


# --- Avaliação final -------------------------------------------------------------------
def score_metrics(y: pd.Series, p: np.ndarray) -> dict:
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def holdout_report(probas: dict, y: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Métricas no holdout para cada modelo e para o Churn Score IBM (0–100 -> 0–1)."""
    rows = {name: score_metrics(y, p) for name, p in probas.items()}
    rows["Churn Score IBM (benchmark)"] = score_metrics(y, benchmark / 100)
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
