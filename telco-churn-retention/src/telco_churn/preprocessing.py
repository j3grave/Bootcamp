"""Split estratificado, transformadores sklearn e exportação de artefactos."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTIFACTS_DIR, DATA_PROCESSED, RANDOM_STATE, TEST_SIZE


def stratified_split(df: pd.DataFrame, features: list[str], target: str):
    """Holdout estratificado ao nível do cliente. Devolve X/y e índices."""
    return train_test_split(df[features], df[target], df.index, test_size=TEST_SIZE,
                            stratify=df[target], random_state=RANDOM_STATE)


def build_preprocessors(num: list[str], cat: list[str]) -> dict[str, ColumnTransformer]:
    """Dois pré-processadores: 'linear' (com escala) e 'trees' (sem escala)."""
    ohe = dict(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    linear = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(**ohe))]), cat),
    ], verbose_feature_names_out=False)
    trees = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(**ohe), cat),
    ], verbose_feature_names_out=False)
    return {"linear": linear, "trees": trees}


def export_artifacts(train: pd.DataFrame, test: pd.DataFrame, joined: pd.DataFrame,
                     merged: pd.DataFrame, config: dict) -> list:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        DATA_PROCESSED / "train.csv": train,
        DATA_PROCESSED / "test.csv": test,
        DATA_PROCESSED / "joined_para_scoring.csv": joined,
        DATA_PROCESSED / "telco_merged_full.csv": merged,
    }
    for p, d in paths.items():
        d.to_csv(p, index=False)
    cfg_path = ARTIFACTS_DIR / "preprocessing_config.joblib"
    joblib.dump(config, cfg_path)
    return list(paths) + [cfg_path]
