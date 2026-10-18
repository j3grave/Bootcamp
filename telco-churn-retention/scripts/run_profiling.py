"""Gera os 3 relatórios ydata-profiling em outputs/profiling/ (sem notebook).

Uso, a partir da raiz do repositório:
    python scripts/run_profiling.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from telco_churn import data, features, profiling
from telco_churn.config import TARGET


def main() -> None:
    df = data.merge_datasets(data.load_raw())
    df_model, _ = features.split_population(df)
    num, cat, _ = features.feature_roles(df_model)
    df_model = features.add_features(df_model)
    feats = num + features.NEW_NUM + cat + features.NEW_CAT
    final_cols = ["Customer ID", "CLTV", "Churn Score", "Satisfaction Score"] + feats + [TARGET]

    _, p1 = profiling.profile_dataset(df, "Telco Churn — Base unida (51 colunas)", "01_base_unida",
                                      "União dos 5 ficheiros: 7.043 clientes, antes da catalogação.")
    _, p2 = profiling.profile_dataset(df_model[final_cols], "Telco Churn — Dataset final", "02_dataset_final",
                                      "População de modelação: 6.589 clientes, 41 colunas.")
    _, p3 = profiling.profile_by_target(df_model, TARGET, feats, "03_stayed_vs_churned")
    for p in (p1, p2, p3):
        print("✔", p)


if __name__ == "__main__":
    main()
