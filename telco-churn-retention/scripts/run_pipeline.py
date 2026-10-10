"""Corre o pipeline completo sem notebook (gráficos guardados, não mostrados).

Uso, a partir da raiz do repositório:
    python scripts/run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import roc_auc_score

from telco_churn import analysis, data, features, plots, preprocessing
from telco_churn.config import RANDOM_STATE, TAB_DIR, TARGET, ensure_dirs
from telco_churn.style import apply_theme


def main() -> None:
    ensure_dirs()
    apply_theme()

    raw = data.load_raw()
    data.validate_keys(raw)
    df = data.merge_datasets(raw)
    data.data_quality(df).to_csv(TAB_DIR / "01_qualidade_dados.csv")
    print(data.derivation_checks(df).to_string(), "\n")

    df_model, df_joined = features.split_population(df)
    rate = df_model[TARGET].mean()
    num, cat, roles = features.feature_roles(df_model)
    roles.to_csv(TAB_DIR / "02_papeis_variaveis.csv", index=False)
    plots.plot_target(df_model[TARGET], show=False)

    analysis.numeric_profile(df_model, num).to_csv(TAB_DIR / "03_perfil_numericas.csv")
    for c in num:
        plots.plot_numeric(df_model, c, TARGET, show=False)
    plots.plot_correlation(df_model[num + [TARGET]].corr(method="spearman"), show=False)

    analysis.categorical_profile(df_model, cat).to_csv(TAB_DIR / "04_perfil_categoricas.csv")
    analysis.value_dictionary(df_model, cat, TARGET).to_csv(
        TAB_DIR / "05_dicionario_categorias.csv", index=False)
    plots.plot_categorical_overview(df_model, cat, show=False)
    cramers = analysis.cramers_ranking(df_model, cat, TARGET)
    cramers.to_csv(TAB_DIR / "06_cramers_v.csv")
    for c in cat:
        plots.plot_categorical_detail(df_model, c, TARGET, rate, cramers[c], show=False)

    mi = analysis.mutual_information(df_model, num, cat, TARGET)
    mi.to_csv(TAB_DIR / "07_mutual_information.csv")
    plots.plot_leakage_checks(df_model, TARGET, show=False)

    df_model = features.add_features(df_model)
    df_joined = features.add_features(df_joined)
    f_num, f_cat = num + features.NEW_NUM, cat + features.NEW_CAT

    X_tr, X_te, y_tr, y_te, i_tr, i_te = preprocessing.stratified_split(
        df_model, f_num + f_cat, TARGET)
    prep = preprocessing.build_preprocessors(f_num, f_cat)
    prep["trees"].fit(X_tr)

    cols = ["Customer ID", "CLTV", "Churn Score", "Satisfaction Score"] + f_num + f_cat + [TARGET]
    preprocessing.export_artifacts(
        df_model.loc[i_tr, cols], df_model.loc[i_te, cols], df_joined, df,
        {"preprocessors": prep, "features_num": f_num, "features_cat": f_cat,
         "target": TARGET, "random_state": RANDOM_STATE})

    auc = roc_auc_score(y_te, df_model.loc[i_te, "Churn Score"])
    print(f"Clientes modelação: {len(df_model):,} | churn {rate:.1%}")
    print(f"Treino {X_tr.shape} | Teste {X_te.shape}")
    print(f"AUC benchmark (Churn Score IBM, holdout): {auc:.3f}")
    print("Concluído. Ver outputs/ e data/processed/.")


if __name__ == "__main__":
    main()
