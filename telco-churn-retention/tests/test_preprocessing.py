"""Saída do pré-processamento: split limpo, sem leakage e sem NaN no fim."""
import numpy as np
import pandas as pd

from telco_churn import features, preprocessing, schemas
from telco_churn.config import TARGET

FORBIDDEN = features.LEAKAGE_COLS + features.MODEL_DERIVED_COLS + features.SUSPECT_COLS


def test_treino_e_teste_disjuntos(split):
    assert not set(split["train"]["Customer ID"]) & set(split["test"]["Customer ID"])


def test_split_estratificado(split):
    gap = abs(split["train"][TARGET].mean() - split["test"][TARGET].mean())
    assert gap < 0.01


def test_sem_leakage_nas_features(split):
    assert not set(split["features"]) & set(FORBIDDEN)
    assert TARGET not in split["features"]
    assert "Customer ID" not in split["features"]


def test_sem_nan_depois_do_preprocessador(split):
    preps = preprocessing.build_preprocessors(split["num"], split["cat"])
    for prep in preps.values():
        prep.fit(split["train"][split["features"]])
        for part in ("train", "test"):
            assert not np.isnan(prep.transform(split[part][split["features"]])).any()


def test_validate_split_passa(split):
    prep = preprocessing.build_preprocessors(split["num"], split["cat"])["trees"]
    prep.fit(split["train"][split["features"]])
    report = schemas.validate_split(split["train"], split["test"], split["features"], prep,
                                    FORBIDDEN)
    assert (report["resultado"] == "passa").all(), report


def test_validate_split_apanha_leakage(split):
    prep = preprocessing.build_preprocessors(split["num"], split["cat"])["trees"]
    prep.fit(split["train"][split["features"]])
    report = schemas.validate_split(split["train"], split["test"],
                                    split["features"] + ["Churn Score"], prep, FORBIDDEN)
    assert report.loc[report["regra"].str.contains("leakage"), "resultado"].eq("falha").all()


def test_add_features_nao_usa_estatisticas_do_dataset(merged):
    """Linha a linha: o resultado de um cliente não depende dos outros clientes."""
    full = features.add_features(merged)
    shuffled = merged.sample(frac=1, random_state=1)
    for subset in (merged.iloc[:50], shuffled.iloc[:50], merged.iloc[[7]]):
        part = features.add_features(subset)
        pd.testing.assert_frame_equal(part, full.loc[subset.index])
