"""Estatísticas descritivas e medidas de associação."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


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
    """Cramér's V: força da associação entre duas variáveis categóricas (0 = nada, 1 = total)."""
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))


def cramers_ranking(data: pd.DataFrame, cols: list[str], target: str) -> pd.Series:
    """Cramér's V de cada variável categórica com o target, do maior para o menor."""
    return pd.Series({c: cramers_v(data[c], data[target]) for c in cols},
                     name="cramers_v").sort_values(ascending=False)


def high_correlations(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pares de variáveis com |rho| >= threshold (triângulo inferior, sem diagonal)."""
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    pairs = corr.where(mask).stack().rename("rho").reset_index()
    pairs.columns = ["var_1", "var_2", "rho"]
    return pairs[pairs["rho"].abs() >= threshold].sort_values("rho", key=abs, ascending=False)
