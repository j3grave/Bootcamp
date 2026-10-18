"""Carregamento, validação e união dos ficheiros do dataset Telco."""
import numpy as np
import pandas as pd

from .config import DATA_RAW, FILES

CUSTOMER_TABLES = ("demographics", "location", "services", "status")
REPORTING_COLS = ["Count", "Quarter"]   # campos de reporting, repetidos e não informativos


def load_raw(data_dir=DATA_RAW) -> dict[str, pd.DataFrame]:
    """Lê os 5 ficheiros. keep_default_na=False preserva o literal 'None'."""
    return {name: pd.read_excel(data_dir / f, keep_default_na=False) for name, f in FILES.items()}


def validate_keys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Verifica unicidade das chaves, igualdade dos IDs e cobertura do join por Zip Code."""
    rows = []
    for name in CUSTOMER_TABLES:
        d = raw[name]
        rows.append((name, "Customer ID", len(d), d["Customer ID"].nunique()))
    p = raw["population"]
    rows.append(("population", "Zip Code", len(p), p["Zip Code"].nunique()))
    out = pd.DataFrame(rows, columns=["tabela", "chave", "linhas", "valores_unicos"])
    out["chave_unica"] = out["linhas"] == out["valores_unicos"]

    id_sets = [set(raw[n]["Customer ID"]) for n in CUSTOMER_TABLES]
    same_ids = all(s == id_sets[0] for s in id_sets)
    coverage = raw["location"]["Zip Code"].isin(p["Zip Code"]).mean()

    assert out["chave_unica"].all(), "Existem chaves duplicadas"
    assert same_ids, "Conjuntos de Customer ID diferentes entre ficheiros"
    out.attrs.update(same_ids=same_ids, zip_coverage=coverage)
    return out


def merge_datasets(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une as tabelas numa base com 1 linha por cliente."""
    clean = {n: d.drop(columns=[c for c in REPORTING_COLS if c in d.columns])
             for n, d in raw.items()}
    df = (
        clean["demographics"]
        .merge(clean["location"], on="Customer ID", how="inner", validate="one_to_one")
        .merge(clean["services"], on="Customer ID", how="inner", validate="one_to_one")
        .merge(clean["status"], on="Customer ID", how="inner", validate="one_to_one")
        .merge(clean["population"].drop(columns="ID"), on="Zip Code", how="left",
               validate="many_to_one")
    )
    assert len(df) == len(raw["status"]), "O merge alterou o número de clientes"
    assert df["Customer ID"].is_unique
    return df


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, cardinalidade, missing e strings vazias por coluna."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "n_empty_str": (df == "").sum(),
    }).sort_values("n_unique")


def derivation_checks(df: pd.DataFrame) -> pd.Series:
    """Confirma as derivações exatas documentadas no guia (esperado: 100%)."""
    total_rev = (df["Total Charges"] - df["Total Refunds"]
                 + df["Total Extra Data Charges"] + df["Total Long Distance Charges"])
    return pd.Series({
        "Under 30 == (Age < 30)": ((df["Age"] < 30) == (df["Under 30"] == "Yes")).mean(),
        "Senior Citizen == (Age >= 65)": ((df["Age"] >= 65) == (df["Senior Citizen"] == "Yes")).mean(),
        "Dependents == (Nº Dependents > 0)":
            ((df["Number of Dependents"] > 0) == (df["Dependents"] == "Yes")).mean(),
        "Referred a Friend == (Nº Referrals > 0)":
            ((df["Number of Referrals"] > 0) == (df["Referred a Friend"] == "Yes")).mean(),
        "Internet Type 'None' <=> Internet Service 'No'":
            ((df["Internet Type"] == "None") == (df["Internet Service"] == "No")).mean(),
        "Total Revenue = Charges − Refunds + ExtraData + LongDist":
            np.isclose(df["Total Revenue"], total_rev, atol=0.01).mean(),
    }, name="pct_concordancia")
