"""Fixtures partilhadas: os dados são lidos uma única vez por sessão de testes.

Os testes não escrevem em outputs/: usam as funções schemas.validate_* (que só devolvem o
relatório) e, quando precisam de enforce(), redirecionam o relatório para uma pasta temporária.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telco_churn import data, features, preprocessing  # noqa: E402
from telco_churn.config import TARGET  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    """Os 5 ficheiros brutos, sem validação (os testes validam explicitamente)."""
    return data.load_raw(validate=False)


@pytest.fixture(scope="session")
def merged(raw):
    return data.merge_datasets(raw, validate=False)


@pytest.fixture(scope="session")
def split(merged):
    """Treino/teste exatamente como em scripts/run_pipeline.py."""
    df_model, _ = features.split_population(merged)
    num, cat, _ = features.feature_roles(df_model)
    df_model = features.add_features(df_model)
    f_num, f_cat = num + features.NEW_NUM, cat + features.NEW_CAT
    X_tr, X_te, y_tr, y_te, i_tr, i_te = preprocessing.stratified_split(df_model, f_num + f_cat,
                                                                        TARGET)
    return {"train": df_model.loc[i_tr], "test": df_model.loc[i_te], "num": f_num, "cat": f_cat,
            "features": f_num + f_cat}


@pytest.fixture
def report_tmp(tmp_path, monkeypatch):
    """Redireciona o relatório de validação para uma pasta temporária."""
    from telco_churn import schemas
    path = tmp_path / "00_validacao_dados.csv"
    monkeypatch.setattr(schemas, "REPORT_PATH", path)
    return path
