"""Estatísticas descritivas e medidas de associação."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .config import RANDOM_STATE


def numeric_profile(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """describe() + assimetria, % de zeros e % de outliers pela regra IQR (Tukey)."""
    desc = data[cols].describe().T
    q1, q3 = data[cols].quantile(0.25), data[cols].quantile(0.75)
    iqr = q3 - q1
    outliers = (data[cols] < q1 - 1.5 * iqr) | (data[cols] > q3 + 1.5 * iqr)
    desc["skew"] = data[cols].skew()
    desc["pct_zeros"] = (data[cols] == 0).mean()
    desc["pct_outliers_IQR"] = outliers.mean()
    return desc


def value_dictionary(data: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    """Uma linha por (variável, categoria): nº clientes, % e taxa de churn."""
    frames = []
    for c in cols:
        g = data.groupby(c, observed=True)[target].agg(n="size", churn_rate="mean").reset_index()
        g = g.rename(columns={c: "categoria"})
        g.insert(0, "variavel", c)
        g["pct"] = g["n"] / g["n"].sum()
        frames.append(g[["variavel", "categoria", "n", "pct", "churn_rate"]])
    return pd.concat(frames, ignore_index=True)


def categorical_profile(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cardinalidade, categorias, moda e peso da moda por variável."""
    return pd.DataFrame({
        "n_categorias": data[cols].nunique(),
        "categorias": [", ".join(map(str, sorted(data[c].unique()))) for c in cols],
        "moda": [data[c].mode().iloc[0] for c in cols],
        "freq_moda": [data[c].value_counts(normalize=True).iloc[0] for c in cols],
    }, index=cols).sort_values("n_categorias")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V com correção de enviesamento (Bergsma, 2013). 0 = nada, 1 = perfeita."""
    ct = pd.crosstab(x, y).to_numpy()
    n = ct.sum()
    expected = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - expected) ** 2 / expected).sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1))))


def cramers_ranking(data: pd.DataFrame, cols: list[str], target: str) -> pd.Series:
    return pd.Series({c: cramers_v(data[c], data[target]) for c in cols},
                     name="cramers_v").sort_values(ascending=False)


def mutual_information(data: pd.DataFrame, num: list[str], cat: list[str],
                       target: str) -> pd.Series:
    """Mutual Information de cada variável com o target (num + cat)."""
    X = data[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("category").cat.codes
    mi = mutual_info_classif(X, data[target], discrete_features=[c in cat for c in X.columns],
                             random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns, name="mutual_info").sort_values(ascending=False)


def high_correlations(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pares de variáveis com |rho| >= threshold (triângulo inferior, sem diagonal)."""
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    pairs = corr.where(mask).stack().rename("rho").reset_index()
    pairs.columns = ["var_1", "var_2", "rho"]
    return pairs[pairs["rho"].abs() >= threshold].sort_values("rho", key=abs, ascending=False)
