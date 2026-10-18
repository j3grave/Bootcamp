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
from sklearn.inspection import permutation_importance

from telco_churn import model_plots, modelling, plots
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED, TAB_DIR, ensure_dirs
from telco_churn.style import apply_theme


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
    auc_ibm_train = modelling.score_metrics(y_train, train["Churn Score"] / 100)["auc"]

    corr = modelling.prediction_correlation(oof)
    corr.to_csv(TAB_DIR / "11c_correlacao_modelos.csv")
    model_plots.plot_diversity(corr, show=False)
    model_plots.plot_cv_comparison(cv_table, benchmark_auc=auc_ibm_train, show=False)
    view = modelling.metrics_view(cv_table)
    std = cv_table[[f"{m}_std" for m in modelling.METRICS]].set_axis(list(modelling.METRICS),
                                                                     axis=1)
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
        cv_com = modelling.cv_compare({BEST + " — com Satisfaction": modelo_com},
                                      train[FEATURES_SAT], y_train)
        city_model = modelling.build_models(NUM, CAT, best_params, te=["City"])[BEST]
        city_cv = modelling.cv_compare({BEST + " + City (TargetEncoder)": city_model},
                                       train[FEATURES_CITY], y_train)
        sem = cv_table.loc[[BEST], ["auc", "auc_std", "pr_auc", "brier"]].rename(
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
    pi = permutation_importance(final_model, test[FINAL_FEATURES], y_test, scoring="roc_auc",
                                n_repeats=5, random_state=0, n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=FINAL_FEATURES).clip(lower=0)
    importance = importance.sort_values(ascending=False)
    importance.to_frame("queda_auc").to_csv(TAB_DIR / "14_importancia_permutacao.csv")
    plots.plot_ranking(importance.head(20), f"Permutation importance — {BEST} (queda de AUC, holdout)",
                       "27_importancia_permutacao", highlight=set(CAT + ["City"]), show=False)

    # 9 · Priorização
    orderings = {
        BEST + ": P × CLTV": p_test * test["CLTV"], BEST + ": só P(churn)": p_test,
        "IBM: Churn Score × CLTV": test["Churn Score"] * test["CLTV"],
        "IBM: só Churn Score": test["Churn Score"],
    }
    curve = modelling.capture_curve(y_test, test["CLTV"], orderings)
    capture = modelling.capture_at(curve)
    capture.to_csv(TAB_DIR / "15_captura_valor.csv")
    model_plots.plot_capture_curve(curve, show=False)
    prio_test = modelling.priority_list(test, p_test).merge(test[["Customer ID", TARGET]],
                                                            on="Customer ID")
    model_plots.plot_priority_map(prio_test, 200, show=False)
    p_joined = final_model.predict_proba(joined[FINAL_FEATURES])[:, 1]
    prio_joined = modelling.priority_list(joined, p_joined)

    # 10 · Exportação
    joblib.dump({"model": final_model, "name": BEST, "features": FINAL_FEATURES,
                 "target": TARGET, "best_params": best_params,
                 "holdout": holdout.loc[BEST].to_dict()}, ARTIFACTS_DIR / "final_model.joblib")
    prio_test.to_csv(DATA_PROCESSED / "prioridade_holdout.csv", index=False)
    prio_joined.to_csv(DATA_PROCESSED / "prioridade_joined.csv", index=False)

    # 11 · Resumo
    ibm = holdout.loc["Churn Score IBM (benchmark)"]
    print("\n=== Resumo ===")
    print(f"Modelo final: {BEST}")
    print(f"  Holdout  AUC {holdout.loc[BEST, 'auc']:.3f} | PR-AUC {holdout.loc[BEST, 'pr_auc']:.3f}"
          f" | Brier {holdout.loc[BEST, 'brier']:.3f}")
    print(f"  IBM      AUC {ibm['auc']:.3f} | PR-AUC {ibm['pr_auc']:.3f} | Brier {ibm['brier']:.3f}")
    print(f"  Rede neuronal (MLP) no holdout: AUC {holdout.loc['Neural Network (MLP)', 'auc']:.3f}")
    print(f"  Top 20% por P × CLTV capta {capture.loc[BEST + ': P × CLTV', 'top 20%']:.1%}"
          f" do CLTV perdido (IBM: {capture.loc['IBM: só Churn Score', 'top 20%']:.1%})")
    print("Ver outputs/figures, outputs/tables, outputs/artifacts e data/processed.")


if __name__ == "__main__":
    main()
