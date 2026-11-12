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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from telco_churn import features, model_plots, modelling, plots, retention, scoring
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED, TAB_DIR, ensure_dirs
from telco_churn.style import apply_theme


def step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Etapas ---------------------------------------------------------------------------------
# Cada etapa recebe `d` (dados e listas de colunas) e devolve o que as seguintes precisam.

def load_data() -> SimpleNamespace:
    """Treino, teste e Joined exportados pelo pré-processamento, com as listas de colunas."""
    train, test, joined, cfg = modelling.load_processed()
    num, cat, target = cfg["features_num"], cfg["features_cat"], cfg["target"]
    return SimpleNamespace(train=train, test=test, joined=joined, NUM=num, CAT=cat,
                           TARGET=target, FEATURES=num + cat)


def stage_cv(d, n_iter: int):
    """Tuning dos modelos base, CV de todos os modelos e ensembles, gráficos de comparação."""
    X, y = d.train[d.FEATURES], d.train[d.TARGET]
    step(f"Tuning de {len(modelling.BASE_MODELS)} modelos base (n_iter={n_iter})...")
    best_params, tuning = modelling.tune_base_models(X, y, d.NUM, d.CAT, n_iter=n_iter)
    tuning.to_csv(TAB_DIR / "10_tuning_modelos_base.csv")

    step("Validação cruzada de todos os modelos, ensembles e referências...")
    models = modelling.build_models(d.NUM, d.CAT, best_params)
    ibm = {modelling.BENCHMARK: d.train["Churn Score"].to_numpy() / 100}
    cv_table, oof = modelling.cv_evaluate(models, X, y, benchmarks=ibm)
    cv_table.to_csv(TAB_DIR / "11_cv_modelos.csv")
    best, decision, reason = modelling.select_parsimonious(cv_table)
    decision.to_csv(TAB_DIR / "11d_selecao_modelo.csv")
    step(f"Seleção (regra de parcimónia): {reason}")
    thresholds = cv_plots(d, models, cv_table, oof, best)
    return best_params, models, cv_table, best, reason, thresholds


def cv_plots(d, models, cv_table, oof, best) -> dict:
    """Gráficos e tabelas da comparação em CV; devolve os limiares de decisão (out-of-fold)."""
    X, y = d.train[d.FEATURES], d.train[d.TARGET]
    corr = modelling.prediction_correlation(oof)
    corr.to_csv(TAB_DIR / "11c_correlacao_modelos.csv")
    model_plots.plot_diversity(corr, show=False)
    model_plots.plot_cv_comparison(cv_table, show=False)
    view = modelling.metrics_view(cv_table)
    std = cv_table[[f"{m}_std" for m in modelling.METRICS]]
    std.columns = list(modelling.METRICS)
    model_plots.plot_metrics_panel(view, std=std, train=modelling.metrics_view(cv_table, "train"),
                                   best=best, show=False)
    model_plots.plot_metrics_heatmap(view, show=False)
    model_plots.plot_overfitting(cv_table.drop(index=modelling.REFERENCES), show=False)
    model_plots.plot_roc_pr(y, oof, best=best, show=False)
    model_plots.plot_confusion_matrices(y, oof, show=False)
    t_f1 = modelling.best_f1_threshold(y, oof[best])
    modelling.threshold_table(y, oof[best]).to_csv(TAB_DIR / "11b_limiares_modelo_final.csv")
    model_plots.plot_threshold(y, oof[best], best, marker=t_f1, show=False)
    # Limiares de negócio: capacidade da equipa de retenção (top 10% / 20% dos clientes)
    cuts = modelling.capacity_thresholds(oof[best])
    thresholds = {"0,5 (por omissão)": 0.5, "melhor F1": t_f1,
                  "capacidade top 10%": cuts[0.10], "capacidade top 20%": cuts[0.20]}
    modelling.decision_table(y, oof[best], thresholds).to_csv(
        TAB_DIR / "11e_limiares_negocio_cv.csv")
    mlp = models["Neural Network (MLP)"].fit(X, y).named_steps["clf"]
    model_plots.plot_mlp_training(mlp, show=False)
    return thresholds


def stage_variants(d, best_params, cv_table, best, rapido: bool):
    """Variantes do melhor modelo: com Satisfaction Score (leakage) e com City (TargetEncoder).

    Devolve o modelo com Satisfaction, o modelo com City (ou None) e se City entra no final.
    """
    y = d.train[d.TARGET]
    features_sat = d.NUM + ["Satisfaction Score"] + d.CAT
    model_sat = modelling.build_models(d.NUM + ["Satisfaction Score"], d.CAT, best_params)[best]
    if rapido:
        return model_sat, None, False
    step("Variantes: com Satisfaction Score e com City (TargetEncoder)...")
    cv_sat, _ = modelling.cv_evaluate({best + " — com Satisfaction": model_sat},
                                      d.train[features_sat], y)
    city_model = modelling.build_models(d.NUM, d.CAT, best_params, te=["City"])[best]
    city_cv, _ = modelling.cv_evaluate({best + " + City (TargetEncoder)": city_model},
                                       d.train[d.FEATURES + ["City"]], y)
    without = cv_table.loc[[best], modelling.SUMMARY_COLS].rename(
        index={best: best + " — sem Satisfaction"})
    variants = pd.concat([without, cv_sat[modelling.SUMMARY_COLS],
                          city_cv[modelling.SUMMARY_COLS]])
    variants.to_csv(TAB_DIR / "12_variantes_satisfaction_city.csv")
    gain, use_city = modelling.include_city(cv_table, best, city_cv)
    step(f"Ganho com City: {gain:+.4f} -> incluir City: {use_city}")
    return model_sat, city_model, use_city


def stage_holdout(d, models, best, model_sat, city_model, use_city, thresholds):
    """Avaliação final no holdout (uma única vez) e importância por permutação."""
    step("Avaliação final no holdout...")
    y_train, y_test = d.train[d.TARGET], d.test[d.TARGET]
    final_model = city_model if use_city else models[best]
    final_features = d.FEATURES + ["City"] if use_city else d.FEATURES
    probas = modelling.fit_predict(models, d.train[d.FEATURES], y_train, d.test[d.FEATURES])
    if use_city:
        probas[best] = modelling.fit_predict({best: final_model}, d.train[final_features],
                                             y_train, d.test[final_features])[best]
    features_sat = d.NUM + ["Satisfaction Score"] + d.CAT
    probas[best + " + Satisfaction (leakage)"] = modelling.fit_predict(
        {"sat": model_sat}, d.train[features_sat], y_train, d.test[features_sat])["sat"]
    holdout = modelling.holdout_report(probas, y_test, d.test["Churn Score"])
    holdout.to_csv(TAB_DIR / "13_holdout.csv")
    p_test = probas[best]
    # Os mesmos limiares, fixados na CV, aplicados uma vez ao holdout
    modelling.decision_table(y_test, p_test, thresholds).to_csv(
        TAB_DIR / "13b_limiares_negocio_holdout.csv")
    # Reporte (não decide nada): IC 95% da diferença de AUC face ao IBM, bootstrap emparelhado
    boot = modelling.paired_bootstrap_auc_diff(y_test, p_test, d.test["Churn Score"] / 100)
    pd.DataFrame([boot], index=[f"{best} − {modelling.BENCHMARK}"]).to_csv(
        TAB_DIR / "13d_bootstrap_auc_vs_ibm.csv")
    step(f"AUC modelo − IBM: {boot['diferenca']:+.4f}, IC 95% [{boot['ic_inferior']:+.4f}, "
         f"{boot['ic_superior']:+.4f}] (inclui zero: {boot['ic_inclui_zero']})")
    # Níveis de risco absolutos (30% / 60%): a taxa de churn real confirma a leitura de P
    modelling.risk_level_table(y_test, p_test).to_csv(TAB_DIR / "13c_niveis_risco_holdout.csv")
    model_plots.plot_roc_calibration(y_test, {
        best: p_test, "Churn Score IBM": d.test["Churn Score"] / 100,
        "Logistic Regression": probas["Logistic Regression"],
        "Neural Network (MLP)": probas["Neural Network (MLP)"]}, show=False)
    model_plots.plot_metrics_panel(
        holdout.drop(index=modelling.BASELINE, errors="ignore"), best=best,
        name="24b_painel_metricas_holdout", title="Nove métricas por modelo — holdout (20%)",
        show=False)

    step("Permutation importance...")
    importance = modelling.permutation_ranking(final_model, d.test[final_features], y_test)
    importance.to_frame("queda_auc").to_csv(TAB_DIR / "14_importancia_permutacao.csv")
    plots.plot_ranking(importance.head(20), f"Permutation importance — {best} (queda de AUC, holdout)",
                       "27_importancia_permutacao", highlight=set(d.CAT + ["City"]), show=False)
    return final_model, final_features, holdout, p_test


def stage_priority_model(d, n_iter: int, best, p_test):
    """Modelo de priorização: só clientes estabelecidos (tenure > 3), mesmo protocolo."""
    train_est = d.train[features.is_established(d.train)]
    test_est = d.test[features.is_established(d.test)]
    X_train_est, y_train_est = train_est[d.FEATURES], train_est[d.TARGET]
    X_test_est, y_test_est = test_est[d.FEATURES], test_est[d.TARGET]
    step(f"Modelo de priorização: tuning e CV em {len(train_est)} clientes estabelecidos...")
    best_params_est, tuning_est = modelling.tune_base_models(X_train_est, y_train_est, d.NUM,
                                                            d.CAT, n_iter=n_iter)
    tuning_est.to_csv(TAB_DIR / "16_tuning_estabelecidos.csv")
    models_est = modelling.build_models(d.NUM, d.CAT, best_params_est)
    ibm_est = {modelling.BENCHMARK: train_est["Churn Score"].to_numpy() / 100}
    cv_est, oof_est = modelling.cv_evaluate(models_est, X_train_est, y_train_est,
                                            benchmarks=ibm_est)
    cv_est.to_csv(TAB_DIR / "16b_cv_modelos_estabelecidos.csv")
    model_plots.plot_cv_comparison(cv_est, name="28_cv_modelos_estabelecidos",
                                   title="AUC em validação cruzada — clientes estabelecidos (tenure > 3)",
                                   show=False)
    best_est, decision_est, reason_est = modelling.select_parsimonious(cv_est)
    decision_est.to_csv(TAB_DIR / "16c_selecao_modelo_estabelecidos.csv")
    step(f"Modelo de priorização: {reason_est}")

    priority_model = models_est[best_est].fit(X_train_est, y_train_est)
    p_test_est = priority_model.predict_proba(X_test_est)[:, 1]

    # Capacidade da equipa medida só nos clientes estabelecidos: os clientes novos (P ≈ 1 no
    # modelo core) já não ocupam o top 10% / 20%. Limiares fixados no OOF, aplicados ao holdout.
    cuts_est = modelling.capacity_thresholds(oof_est[best_est])
    thresholds_est = {"0,5 (por omissão)": 0.5,
                      "capacidade top 10%": cuts_est[0.10], "capacidade top 20%": cuts_est[0.20]}
    modelling.decision_table(y_train_est, oof_est[best_est], thresholds_est).to_csv(
        TAB_DIR / "16d_limiares_capacidade_estabelecidos_cv.csv")
    modelling.decision_table(y_test_est, p_test_est, thresholds_est).to_csv(
        TAB_DIR / "17b_limiares_capacidade_estabelecidos_holdout.csv")
    y_test = d.test[d.TARGET]
    est_mask = features.is_established(d.test).to_numpy()
    comparison = modelling.comparison_table([
        ("Modelo core: " + best, "holdout completo", y_test, p_test),
        ("Churn Score IBM", "holdout completo", y_test, d.test["Churn Score"] / 100),
        ("Modelo core: " + best, "holdout tenure > 3", y_test[est_mask], p_test[est_mask]),
        ("Modelo de priorização: " + best_est, "holdout tenure > 3", y_test_est, p_test_est),
        ("Churn Score IBM", "holdout tenure > 3", y_test_est, test_est["Churn Score"] / 100),
    ])
    comparison.to_csv(TAB_DIR / "17_comparacao_core_priorizacao.csv")

    importance_est = modelling.permutation_ranking(priority_model, X_test_est, y_test_est)
    importance_est.to_frame("queda_auc").to_csv(TAB_DIR / "18_importancia_priorizacao.csv")
    plots.plot_ranking(importance_est.head(20),
                       f"Permutation importance — {best_est} (tenure > 3, holdout)",
                       "29_importancia_priorizacao", highlight=set(d.CAT), show=False)
    inputs = [c for c in d.FEATURES if c not in features.NEW_NUM + features.NEW_CAT]
    bundle = dict(model=priority_model, name=best_est, best_params=best_params_est,
                  cv=cv_est.loc[best_est].to_dict(), selection=reason_est,
                  holdout=comparison.loc[("holdout tenure > 3",
                                          "Modelo de priorização: " + best_est)].to_dict(),
                  benchmark_cv=cv_est.loc[modelling.BENCHMARK].to_dict(),
                  thresholds=thresholds_est, risk_levels=modelling.RISK_LEVELS,
                  capacity_cuts={"top 10%": cuts_est[0.10], "top 20%": cuts_est[0.20]},
                  form=scoring.form_spec(train_est, inputs),
                  cltv_median=float(train_est["CLTV"].median()))
    return test_est, p_test_est, comparison, bundle


def stage_retention(d, test_est, p_test_est):
    """Priorização dos estabelecidos: captura de valor, custo-benefício e ações sugeridas."""
    y_est, cltv = test_est[d.TARGET], test_est["CLTV"]
    orderings = {
        "Modelo: P × CLTV": p_test_est * cltv,
        "Modelo: só P(churn)": p_test_est,
        "IBM: Churn Score × CLTV": test_est["Churn Score"] * cltv,
        "IBM: só Churn Score": test_est["Churn Score"],
    }
    curve = modelling.capture_curve(y_est, cltv, orderings)
    capture = modelling.capture_at(curve)
    capture.to_csv(TAB_DIR / "15_captura_valor.csv")
    model_plots.plot_capture_curve(curve, show=False)
    orderings_compared = retention.compare_orderings(y_est, cltv, score_p=p_test_est,
                                                     score_value=p_test_est * cltv)
    orderings_compared.to_csv(TAB_DIR / "15b_P_vs_PxCLTV.csv")

    # Custo-benefício: os pressupostos estão em retention.CENARIOS
    scenarios = retention.scenario_table(y_est, p_test_est, cltv, retention.CENARIOS)
    scenarios.to_csv(TAB_DIR / "20_cenarios_custo_beneficio.csv")
    value_curves = {name: retention.net_value_curve(y_est, cltv, p_test_est * cltv,
                                                    params["taxa_sucesso"], params["custo_contacto"])
                    for name, params in retention.CENARIOS.items()}
    model_plots.plot_net_value(value_curves, show=False)
    sensitivity = retention.sensitivity_grid(p_test_est, cltv,
                                             success_rates=[0.05, 0.10, 0.20, 0.30, 0.40],
                                             costs=[25, 50, 100, 150, 200])
    sensitivity.to_csv(TAB_DIR / "20b_sensibilidade_pct_contactar.csv")
    prio_test = modelling.priority_list(test_est, p_test_est).merge(
        test_est[["Customer ID", d.TARGET]], on="Customer ID")
    model_plots.plot_priority_map(prio_test, 200, show=False)
    prio_test = prio_test.merge(test_est[["Customer ID"] + retention.ACTION_COLS], on="Customer ID")
    prio_test["acao_sugerida"] = retention.suggest_action(prio_test)
    n_campaign = int(scenarios.loc["Base", "N_regra_valor_esperado"])
    actions = retention.actions_summary(prio_test, n_campaign, d.TARGET)
    actions.to_csv(TAB_DIR / "21_acoes_sugeridas.csv")
    return SimpleNamespace(capture=capture, orderings_compared=orderings_compared,
                           scenarios=scenarios, prio_test=prio_test, n_campaign=n_campaign,
                           actions=actions)


def stage_onboarding(d):
    """Clientes novos: regra de onboarding (sem P(churn) para os Joined)."""
    merged = pd.read_csv(DATA_PROCESSED / "telco_merged_full.csv", keep_default_na=False,
                         na_values=[""])
    new = merged[~features.is_established(merged)].copy()
    new["saiu"] = (new["Customer Status"] == "Churned").astype(int)
    new["onboarding"] = retention.onboarding_segment(new)
    rule = retention.onboarding_rule(new)
    rule.to_csv(TAB_DIR / "19_regra_onboarding.csv")
    return rule, retention.onboarding_list(d.joined)


def print_summary(best, reason, holdout, priority_bundle, comparison, r, rule) -> None:
    ibm = holdout.loc[modelling.BENCHMARK]
    rule_row = holdout.loc[modelling.RULE_BASELINE]
    print("\n=== Resumo ===")
    print(f"Modelo final: {best} — {reason}")
    print(f"  Holdout  AUC {holdout.loc[best, 'auc']:.3f} | PR-AUC {holdout.loc[best, 'pr_auc']:.3f}"
          f" | Brier {holdout.loc[best, 'brier']:.3f}")
    print(f"  IBM      AUC {ibm['auc']:.3f} | PR-AUC {ibm['pr_auc']:.3f} | Brier {ibm['brier']:.3f}")
    print(f"  Regra    AUC {rule_row['auc']:.3f} | PR-AUC {rule_row['pr_auc']:.3f}"
          f" | Brier {rule_row['brier']:.3f}")
    print(f"  Rede neuronal (MLP) no holdout: AUC {holdout.loc['Neural Network (MLP)', 'auc']:.3f}")
    print(f"Modelo de priorização (tenure > 3): {priority_bundle['name']}")
    print(comparison[["clientes", "auc", "pr_auc", "brier"]].round(3).to_string())
    # Comparar sempre ordenações do mesmo tipo: P × CLTV com P × CLTV, só P com só P
    print("Top 20% dos clientes estabelecidos (% do CLTV perdido coberto):")
    print(f"  P × CLTV: modelo {r.capture.loc['Modelo: P × CLTV', 'top 20%']:.1%}"
          f" | IBM {r.capture.loc['IBM: Churn Score × CLTV', 'top 20%']:.1%}")
    print(f"  Só P:     modelo {r.capture.loc['Modelo: só P(churn)', 'top 20%']:.1%}"
          f" | IBM {r.capture.loc['IBM: só Churn Score', 'top 20%']:.1%}")
    print("P vs P × CLTV (modelo de priorização):")
    print(r.orderings_compared.round(3).to_string())
    print("Custo-benefício (pressupostos ilustrativos):")
    print(r.scenarios.round(2).to_string())
    print(f"Ações sugeridas no top {r.n_campaign} (cenário Base) — hipóteses a testar:")
    print(r.actions.round(2).to_string())
    print("Regra de onboarding (clientes novos):")
    print(rule.round(2).to_string())
    print("Ver outputs/figures, outputs/tables, outputs/artifacts e data/processed.")


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

    d = load_data()
    step(f"Treino {d.train[d.FEATURES].shape} | Teste {d.test[d.FEATURES].shape}"
         f" | Joined {len(d.joined)}")
    best_params, models, cv_table, best, reason, thresholds = stage_cv(d, n_iter)
    model_sat, city_model, use_city = stage_variants(d, best_params, cv_table, best, args.rapido)
    final_model, final_features, holdout, p_test = stage_holdout(d, models, best, model_sat,
                                                                 city_model, use_city, thresholds)
    test_est, p_test_est, comparison, priority_bundle = stage_priority_model(d, n_iter, best,
                                                                             p_test)
    r = stage_retention(d, test_est, p_test_est)
    rule, onboarding_joined = stage_onboarding(d)

    # Exportação
    modelling.save_bundle(ARTIFACTS_DIR / "final_model.joblib", final_model, best, final_features,
                          d.TARGET, best_params, cv=cv_table.loc[best].to_dict(),
                          holdout=holdout.loc[best].to_dict(), selection=reason,
                          benchmark_cv=cv_table.loc[modelling.BENCHMARK].to_dict(),
                          benchmark_holdout=holdout.loc[modelling.BENCHMARK].to_dict(),
                          form=scoring.form_spec(d.train, [c for c in final_features
                                                           if c not in features.NEW_NUM
                                                           + features.NEW_CAT]),
                          cltv_median=float(d.train["CLTV"].median()),
                          thresholds=thresholds,
                          risk_levels=modelling.RISK_LEVELS,
                          capacity_cuts={"top 10%": thresholds["capacidade top 10%"],
                                         "top 20%": thresholds["capacidade top 20%"]})
    modelling.save_bundle(ARTIFACTS_DIR / "priority_model.joblib", features=d.FEATURES,
                          target=d.TARGET, population="Churned + Stayed com tenure > 3",
                          **priority_bundle)
    r.prio_test.to_csv(DATA_PROCESSED / "prioridade_holdout.csv", index=False)
    onboarding_joined.to_csv(DATA_PROCESSED / "onboarding_joined.csv", index=False)

    print_summary(best, reason, holdout, priority_bundle, comparison, r, rule)


if __name__ == "__main__":
    main()
