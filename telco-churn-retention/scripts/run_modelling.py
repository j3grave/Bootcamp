"""Corre a fase de modelação completa sem notebook (gráficos guardados, não mostrados).

Faz o mesmo que notebooks/02_modelling_eda.ipynb, pela mesma ordem e com as mesmas funções
de src/telco_churn. Precisa dos ficheiros gerados pelo pré-processamento
(scripts/run_pipeline.py ou notebook 01).

Uso, a partir da raiz do repositório (telco-churn-retention):
    python scripts/run_modelling.py              # versão completa (N_ITER = 20)
    python scripts/run_modelling.py --n-iter 5   # tuning mais curto
    python scripts/run_modelling.py --rapido     # teste rápido: N_ITER = 3, sem variantes

Também é chamado por scripts/run_all.py.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd

from telco_churn import features, model_plots, modelling, plots, retention
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED, TAB_DIR, ensure_dirs
from telco_churn.style import apply_theme


# Parâmetros de negócio — PRESSUPOSTOS, NÃO DADOS (iguais aos do notebook 02, secção 11.2).
#   taxa_sucesso   : % dos churners contactados que ficam por causa do contacto
#   custo_contacto : custo por cliente contactado (contacto + incentivo), na unidade do CLTV
CENARIOS = {
    "Pessimista": {"taxa_sucesso": 0.10, "custo_contacto": 150},
    "Base":       {"taxa_sucesso": 0.20, "custo_contacto": 100},
    "Otimista":   {"taxa_sucesso": 0.30, "custo_contacto": 50},
}


def step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-iter", type=int, default=20,
                        help="combinações testadas por modelo no RandomizedSearchCV")
    parser.add_argument("--rapido", action="store_true",
                        help="teste rápido: n_iter=3 e sem variantes Satisfaction/City")
    args = parser.parse_args()
    n_iter = 3 if args.rapido else args.n_iter

    ensure_dirs()
    apply_theme()
    if not (DATA_PROCESSED / "train.csv").exists():
        sys.exit("Falta data/processed/train.csv: correr primeiro scripts/run_pipeline.py")

    # 2 · Dados
    train, test, joined, cfg = modelling.load_processed()
    NUM, CAT, TARGET = cfg["features_num"], cfg["features_cat"], cfg["target"]
    FEATURES = NUM + CAT
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    step(f"Treino {X_train.shape} | Teste {X_test.shape} | Joined {len(joined)}")

    # 3 · Tuning dos modelos base
    step(f"Tuning de {len(modelling.BASE_MODELS)} modelos base (n_iter={n_iter})...")
    best_params, tuning = modelling.tune_base_models(X_train, y_train, NUM, CAT, n_iter=n_iter)
    tuning.to_csv(TAB_DIR / "10_tuning_modelos_base.csv")

    # 4 · Comparação em CV: modelos base, rede neuronal e ensembles
    step("Validação cruzada de todos os modelos e ensembles...")
    models = modelling.build_models(NUM, CAT, best_params)
    cv_table, oof = modelling.cv_evaluate(models, X_train, y_train)
    cv_table.to_csv(TAB_DIR / "11_cv_modelos.csv")
    BEST = modelling.select_best(cv_table)
    auc_ibm_train = modelling.classification_metrics(y_train, train["Churn Score"] / 100)["auc"]

    corr = modelling.prediction_correlation(oof)
    corr.to_csv(TAB_DIR / "11c_correlacao_modelos.csv")
    model_plots.plot_diversity(corr, show=False)
    model_plots.plot_cv_comparison(cv_table, benchmark_auc=auc_ibm_train, show=False)
    view = modelling.metrics_view(cv_table)
    std = cv_table[[f"{m}_std" for m in modelling.METRICS]]
    std.columns = list(modelling.METRICS)
    model_plots.plot_metrics_panel(view, std=std, train=modelling.metrics_view(cv_table, "train"),
                                   best=BEST, show=False)
    model_plots.plot_metrics_heatmap(view, show=False)
    model_plots.plot_overfitting(cv_table.drop(index=modelling.BASELINE), show=False)
    oof_ibm = oof | {"Churn Score IBM": train["Churn Score"].to_numpy() / 100}
    model_plots.plot_roc_pr(y_train, oof_ibm, best=BEST, show=False)
    model_plots.plot_confusion_matrices(y_train, oof, show=False)
    t_f1 = modelling.best_f1_threshold(y_train, oof[BEST])
    modelling.threshold_table(y_train, oof[BEST]).to_csv(TAB_DIR / "11b_limiares_modelo_final.csv")
    model_plots.plot_threshold(y_train, oof[BEST], BEST, marker=t_f1, show=False)
    mlp = models["Neural Network (MLP)"].fit(X_train, y_train).named_steps["clf"]
    model_plots.plot_mlp_training(mlp, show=False)
    step(f"Modelo selecionado (maior AUC em CV): {BEST} ({cv_table.loc[BEST, 'auc']:.4f})")
    tied = modelling.tied_with_best(cv_table, BEST)
    if tied:
        step(f"Empatados com o selecionado (diferença < 1 desvio-padrão): {', '.join(tied)}")

    # 5–6 · Variantes: Satisfaction Score e City
    NUM_SAT, FEATURES_SAT = NUM + ["Satisfaction Score"], NUM + ["Satisfaction Score"] + CAT
    modelo_com = modelling.build_models(NUM_SAT, CAT, best_params)[BEST]
    USE_CITY, FEATURES_CITY, city_model = False, FEATURES + ["City"], None
    if not args.rapido:
        step("Variantes: com Satisfaction Score e com City (TargetEncoder)...")
        cv_com = modelling.cv_evaluate({BEST + " — com Satisfaction": modelo_com},
                                       train[FEATURES_SAT], y_train)[0][modelling.SUMMARY_COLS]
        city_model = modelling.build_models(NUM, CAT, best_params, te=["City"])[BEST]
        city_cv = modelling.cv_evaluate({BEST + " + City (TargetEncoder)": city_model},
                                        train[FEATURES_CITY], y_train)[0][modelling.SUMMARY_COLS]
        sem = cv_table.loc[[BEST], modelling.SUMMARY_COLS].rename(
            index={BEST: BEST + " — sem Satisfaction"})
        pd.concat([sem, cv_com, city_cv]).to_csv(TAB_DIR / "12_variantes_satisfaction_city.csv")
        gain = city_cv["auc"].iloc[0] - cv_table.loc[BEST, "auc"]
        USE_CITY = gain > cv_table.loc[BEST, "auc_std"]
        step(f"Ganho com City: {gain:+.4f} -> incluir City: {USE_CITY}")

    # 7 · Holdout (uma única vez)
    step("Avaliação final no holdout...")
    final_model = city_model if USE_CITY else models[BEST]
    FINAL_FEATURES = FEATURES_CITY if USE_CITY else FEATURES
    probas = {}
    for name, model in models.items():
        probas[name] = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    if USE_CITY:
        probas[BEST] = final_model.fit(train[FINAL_FEATURES], y_train).predict_proba(
            test[FINAL_FEATURES])[:, 1]
    modelo_com.fit(train[FEATURES_SAT], y_train)
    probas[BEST + " + Satisfaction (leakage)"] = modelo_com.predict_proba(test[FEATURES_SAT])[:, 1]
    holdout = modelling.holdout_report(probas, y_test, test["Churn Score"])
    holdout.to_csv(TAB_DIR / "13_holdout.csv")
    p_test = probas[BEST]
    model_plots.plot_roc_calibration(y_test, {
        BEST: p_test, "Churn Score IBM": test["Churn Score"] / 100,
        "Logistic Regression": probas["Logistic Regression"],
        "Neural Network (MLP)": probas["Neural Network (MLP)"]}, show=False)
    model_plots.plot_metrics_panel(
        holdout.drop(index=modelling.BASELINE, errors="ignore"), best=BEST,
        name="24b_painel_metricas_holdout", title="Nove métricas por modelo — holdout (20%)",
        show=False)

    # 8 · Interpretação
    step("Permutation importance...")
    importance = modelling.permutation_ranking(final_model, test[FINAL_FEATURES], y_test)
    importance.to_frame("queda_auc").to_csv(TAB_DIR / "14_importancia_permutacao.csv")
    plots.plot_ranking(importance.head(20), f"Permutation importance — {BEST} (queda de AUC, holdout)",
                       "27_importancia_permutacao", highlight=set(CAT + ["City"]), show=False)

    # 9 · Modelo de priorização: só clientes estabelecidos (tenure > 3), mesmo protocolo
    train_est = train[features.is_established(train)]
    test_est = test[features.is_established(test)]
    X_train_est, y_train_est = train_est[FEATURES], train_est[TARGET]
    X_test_est, y_test_est = test_est[FEATURES], test_est[TARGET]
    step(f"Modelo de priorização: tuning e CV em {len(train_est)} clientes estabelecidos...")
    best_params_est, tuning_est = modelling.tune_base_models(X_train_est, y_train_est, NUM, CAT,
                                                            n_iter=n_iter)
    tuning_est.to_csv(TAB_DIR / "16_tuning_estabelecidos.csv")
    models_est = modelling.build_models(NUM, CAT, best_params_est)
    cv_est, _ = modelling.cv_evaluate(models_est, X_train_est, y_train_est)
    cv_est.to_csv(TAB_DIR / "16b_cv_modelos_estabelecidos.csv")
    auc_ibm_train_est = modelling.classification_metrics(
        y_train_est, train_est["Churn Score"] / 100)["auc"]
    model_plots.plot_cv_comparison(cv_est, benchmark_auc=auc_ibm_train_est,
                                   name="28_cv_modelos_estabelecidos",
                                   title="AUC em validação cruzada — clientes estabelecidos (tenure > 3)",
                                   show=False)
    BEST_EST = modelling.select_best(cv_est)
    step(f"Modelo de priorização: {BEST_EST} ({cv_est.loc[BEST_EST, 'auc']:.4f})")

    priority_model = models_est[BEST_EST].fit(X_train_est, y_train_est)
    p_test_est = priority_model.predict_proba(X_test_est)[:, 1]
    est_mask = features.is_established(test).to_numpy()
    comparison = modelling.comparison_table([
        ("Modelo core: " + BEST, "holdout completo", y_test, p_test),
        ("Churn Score IBM", "holdout completo", y_test, test["Churn Score"] / 100),
        ("Modelo core: " + BEST, "holdout tenure > 3", y_test[est_mask], p_test[est_mask]),
        ("Modelo de priorização: " + BEST_EST, "holdout tenure > 3", y_test_est, p_test_est),
        ("Churn Score IBM", "holdout tenure > 3", y_test_est, test_est["Churn Score"] / 100),
    ])
    comparison.to_csv(TAB_DIR / "17_comparacao_core_priorizacao.csv")

    importance_est = modelling.permutation_ranking(priority_model, X_test_est, y_test_est)
    importance_est.to_frame("queda_auc").to_csv(TAB_DIR / "18_importancia_priorizacao.csv")
    plots.plot_ranking(importance_est.head(20),
                       f"Permutation importance — {BEST_EST} (tenure > 3, holdout)",
                       "29_importancia_priorizacao", highlight=set(CAT), show=False)

    # 10 · Priorização dos clientes estabelecidos
    orderings = {
        "Modelo: P × CLTV": p_test_est * test_est["CLTV"],
        "Modelo: só P(churn)": p_test_est,
        "IBM: Churn Score × CLTV": test_est["Churn Score"] * test_est["CLTV"],
        "IBM: só Churn Score": test_est["Churn Score"],
    }
    curve = modelling.capture_curve(y_test_est, test_est["CLTV"], orderings)
    capture = modelling.capture_at(curve)
    capture.to_csv(TAB_DIR / "15_captura_valor.csv")
    model_plots.plot_capture_curve(curve, show=False)
    orderings_compared = retention.compare_orderings(
        y_test_est, test_est["CLTV"], score_p=p_test_est,
        score_value=p_test_est * test_est["CLTV"])
    orderings_compared.to_csv(TAB_DIR / "15b_P_vs_PxCLTV.csv")

    # Custo-benefício: os pressupostos estão em CENARIOS, no topo do ficheiro
    scenarios = retention.scenario_table(y_test_est, p_test_est, test_est["CLTV"], CENARIOS)
    scenarios.to_csv(TAB_DIR / "20_cenarios_custo_beneficio.csv")
    value_curves = {}
    for name, params in CENARIOS.items():
        value_curves[name] = retention.net_value_curve(
            y_test_est, test_est["CLTV"], p_test_est * test_est["CLTV"],
            params["taxa_sucesso"], params["custo_contacto"])
    model_plots.plot_net_value(value_curves, show=False)
    sensitivity = retention.sensitivity_grid(p_test_est, test_est["CLTV"],
                                             success_rates=[0.05, 0.10, 0.20, 0.30, 0.40],
                                             costs=[25, 50, 100, 150, 200])
    sensitivity.to_csv(TAB_DIR / "20b_sensibilidade_pct_contactar.csv")
    prio_test = modelling.priority_list(test_est, p_test_est).merge(
        test_est[["Customer ID", TARGET]], on="Customer ID")
    model_plots.plot_priority_map(prio_test, 200, show=False)
    prio_test = prio_test.merge(test_est[["Customer ID"] + retention.ACTION_COLS], on="Customer ID")
    prio_test["acao_sugerida"] = retention.suggest_action(prio_test)
    n_campanha = int(scenarios.loc["Base", "N_regra_valor_esperado"])
    actions = retention.actions_summary(prio_test, n_campanha, TARGET)
    actions.to_csv(TAB_DIR / "21_acoes_sugeridas.csv")

    # 11 · Clientes novos: regra de onboarding (sem P(churn) para os Joined)
    merged = pd.read_csv(DATA_PROCESSED / "telco_merged_full.csv", keep_default_na=False,
                         na_values=[""])
    novos = merged[~features.is_established(merged)].copy()
    novos["saiu"] = (novos["Customer Status"] == "Churned").astype(int)
    novos["onboarding"] = retention.onboarding_segment(novos)
    regra = novos.groupby("onboarding")["saiu"].agg(clientes="size", taxa_saida="mean")
    regra["joined_no_segmento"] = novos[novos["Customer Status"] == "Joined"].groupby(
        "onboarding").size()
    regra = regra.loc[retention.ONBOARDING_ORDER]
    regra.to_csv(TAB_DIR / "19_regra_onboarding.csv")
    onboarding_joined = retention.onboarding_list(joined)

    # 12 · Exportação
    joblib.dump({"model": final_model, "name": BEST, "features": FINAL_FEATURES,
                 "target": TARGET, "best_params": best_params,
                 "holdout": holdout.loc[BEST].to_dict()}, ARTIFACTS_DIR / "final_model.joblib")
    joblib.dump({"model": priority_model, "name": BEST_EST, "features": FEATURES,
                 "target": TARGET, "population": "Churned + Stayed com tenure > 3",
                 "best_params": best_params_est,
                 "holdout": comparison.loc[("holdout tenure > 3",
                                            "Modelo de priorização: " + BEST_EST)].to_dict()},
                ARTIFACTS_DIR / "priority_model.joblib")
    prio_test.to_csv(DATA_PROCESSED / "prioridade_holdout.csv", index=False)
    onboarding_joined.to_csv(DATA_PROCESSED / "onboarding_joined.csv", index=False)

    # 13 · Resumo
    ibm = holdout.loc["Churn Score IBM (benchmark)"]
    print("\n=== Resumo ===")
    print(f"Modelo final: {BEST}")
    print(f"  Holdout  AUC {holdout.loc[BEST, 'auc']:.3f} | PR-AUC {holdout.loc[BEST, 'pr_auc']:.3f}"
          f" | Brier {holdout.loc[BEST, 'brier']:.3f}")
    print(f"  IBM      AUC {ibm['auc']:.3f} | PR-AUC {ibm['pr_auc']:.3f} | Brier {ibm['brier']:.3f}")
    print(f"  Rede neuronal (MLP) no holdout: AUC {holdout.loc['Neural Network (MLP)', 'auc']:.3f}")
    print(f"Modelo de priorização (tenure > 3): {BEST_EST}")
    print(comparison[["clientes", "auc", "pr_auc", "brier"]].round(3).to_string())
    # Comparar sempre ordenações do mesmo tipo: P × CLTV com P × CLTV, só P com só P
    print("Top 20% dos clientes estabelecidos (% do CLTV perdido coberto):")
    print(f"  P × CLTV: modelo {capture.loc['Modelo: P × CLTV', 'top 20%']:.1%}"
          f" | IBM {capture.loc['IBM: Churn Score × CLTV', 'top 20%']:.1%}")
    print(f"  Só P:     modelo {capture.loc['Modelo: só P(churn)', 'top 20%']:.1%}"
          f" | IBM {capture.loc['IBM: só Churn Score', 'top 20%']:.1%}")
    print("P vs P × CLTV (modelo de priorização):")
    print(orderings_compared.round(3).to_string())
    print("Custo-benefício (pressupostos ilustrativos):")
    print(scenarios.round(2).to_string())
    print(f"Ações sugeridas no top {n_campanha} (cenário Base) — hipóteses a testar:")
    print(actions.round(2).to_string())
    print("Regra de onboarding (clientes novos):")
    print(regra.round(2).to_string())
    print("Ver outputs/figures, outputs/tables, outputs/artifacts e data/processed.")


if __name__ == "__main__":
    main()
