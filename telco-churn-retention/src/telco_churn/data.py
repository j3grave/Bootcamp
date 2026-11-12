"""Carregamento, validação e união dos ficheiros do dataset Telco.

A validação (contratos de dados em schemas.py) corre por omissão ao carregar e ao unir: se uma
regra com severidade 'erro' falhar, a pipeline pára com DataValidationError.
"""
import numpy as np
import pandas as pd

from . import schemas
from .config import DATA_RAW, FILES

CUSTOMER_TABLES = ("demographics", "location", "services", "status")
REPORTING_COLS = ["Count", "Quarter"]   # campos de reporting, repetidos e não informativos


def load_raw(data_dir=DATA_RAW, validate: bool = True) -> dict[str, pd.DataFrame]:
    """Lê os 5 ficheiros. keep_default_na=False preserva o literal 'None'.

    validate=True aplica os contratos dos ficheiros brutos (schemas.validate_raw) e começa um
    relatório novo em outputs/tables/00_validacao_dados.csv.
    """
    raw = {name: pd.read_excel(data_dir / f, keep_default_na=False) for name, f in FILES.items()}
    if validate:
        schemas.enforce(schemas.validate_raw(raw), fresh=True)
    return raw


def validate_keys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabela das chaves (linhas vs valores únicos), igualdade dos IDs e cobertura por Zip Code.

    Resumo para o notebook 01; as mesmas regras são contratos em schemas.validate_raw.
    """
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

    out.attrs.update(same_ids=same_ids, zip_coverage=coverage)
    return out


def merge_datasets(raw: dict[str, pd.DataFrame], validate: bool = True) -> pd.DataFrame:
    """Une as tabelas numa base com 1 linha por cliente.

    validate=True aplica as regras de negócio e as estatísticas da base unida
    (schemas.validate_merged) e acrescenta-as ao relatório.
    """
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
    if validate:
        schemas.enforce(schemas.validate_merged(df))
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
    """% de concordância das derivações exatas do guia (resumo para o notebook 01).

    São contratos obrigatórios a 100% em schemas.validate_merged.
    """
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
