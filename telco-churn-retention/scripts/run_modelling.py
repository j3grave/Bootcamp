"""Fase 2 sem notebook: faz o mesmo que notebooks/02_modelling_eda.ipynb.

Precisa dos ficheiros da fase 1 (run_preprocessing.py ou notebook 01). Os gráficos são guardados.

Uso, a partir da raiz do projeto:
    python scripts/run_modelling.py            # completo
    python scripts/run_modelling.py --rapido   # sem a variante com Satisfaction Score
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
    """Mensagem de progresso com a hora."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Etapas ---------------------------------------------------------------------------------
# Cada etapa recebe `d` (dados e listas de colunas) e devolve o que as seguintes precisam.

def load_data() -> SimpleNamespace:
    """Treino, teste e Joined exportados pelo pré-processamento, com as listas de colunas."""
    train, test, joined, cfg = modelling.load_processed()
    num, cat, target = cfg["features_num"], cfg["features_cat"], cfg["target"]
    return SimpleNamespace(train=train, test=test, joined=joined, NUM=num, CAT=cat,
                           TARGET=target, FEATURES=num + cat)


def stage_cv(d):
    """Tuning, validação cruzada de todos os modelos e escolha do modelo final."""
    X, y = d.train[d.FEATURES], d.train[d.TARGET]
    step(f"GridSearchCV de {len(modelling.BASE_MODELS)} modelos...")
    best_params, tuning = modelling.tune_base_models(X, y, d.NUM, d.CAT)
    tuning.to_csv(TAB_DIR / "10_tuning_modelos.csv")

    step("Validação cruzada dos modelos e das referências...")
    models = modelling.build_models(d.NUM, d.CAT, best_params)
    ibm = {modelling.BENCHMARK: d.train["Churn Score"].to_numpy() / 100}
    cv_table, oof = modelling.cv_evaluate(models, X, y, benchmarks=ibm)
    cv_table.to_csv(TAB_DIR / "11_cv_modelos.csv")
    best, reason = modelling.select_model(cv_table)
    pd.Series({"modelo_escolhido": best, "justificacao": reason}).to_csv(
        TAB_DIR / "11d_selecao_modelo.csv", header=False)
    step(f"Modelo escolhido: {reason}")

    # Capacidade da equipa: P(churn) do top 20% dos clientes estabelecidos, fixada nas
    # previsões out-of-fold do treino e aplicada depois ao holdout e à app
    est = features.is_established(d.train).to_numpy()
    cut = modelling.capacity_threshold(oof[best][est])
    thresholds = {"0,5 (por omissão)": 0.5, "top 20% dos estabelecidos": cut}
    modelling.decision_table(y[est], oof[best][est], thresholds).to_csv(
        TAB_DIR / "11e_limiares_cv.csv")
    cv_plots(d, cv_table, oof, best, cut)
    return best_params, models, cv_table, best, reason, thresholds


def cv_plots(d, cv_table, oof, best, cut) -> None:
    """Gráficos e tabelas da comparação em validação cruzada."""
    y = d.train[d.TARGET]
    model_plots.plot_cv_comparison(cv_table, show=False)
    std = cv_table[[f"{m}_std" for m in modelling.METRICS]]
    std.columns = list(modelling.METRICS)
    model_plots.plot_metrics_panel(modelling.metrics_view(cv_table), std=std,
                                   train=modelling.metrics_view(cv_table, "train"),
                                   best=best, show=False)
    model_plots.plot_overfitting(cv_table.drop(index=modelling.REFERENCES), show=False)
    model_plots.plot_roc_pr(y, oof, best=best, show=False)
    model_plots.plot_confusion_matrices(y, oof, show=False)
    modelling.threshold_table(y, oof[best]).to_csv(TAB_DIR / "11b_limiares_modelo_final.csv")
    model_plots.plot_threshold(y, oof[best], best, marker=cut, show=False)


def stage_satisfaction(d, best_params, cv_table, best, rapido: bool):
    """Variante do modelo escolhido com Satisfaction Score, para medir o leakage."""
    features_sat = d.NUM + ["Satisfaction Score"] + d.CAT
    model_sat = modelling.build_models(d.NUM + ["Satisfaction Score"], d.CAT, best_params,
                                       baseline=False)[best]
    if rapido:
        return model_sat
    step("Variante com Satisfaction Score...")
    cv_sat, _ = modelling.cv_evaluate({best + " — com Satisfaction": model_sat},
                                      d.train[features_sat], d.train[d.TARGET])
    without = cv_table.loc[[best], modelling.SUMMARY_COLS].rename(
        index={best: best + " — sem Satisfaction"})
    pd.concat([without, cv_sat[modelling.SUMMARY_COLS]]).to_csv(
        TAB_DIR / "12_variante_satisfaction.csv")
    return model_sat


def stage_holdout(d, models, best, model_sat, thresholds):
    """Avaliação final no holdout (uma única vez), também só nos clientes estabelecidos."""
    step("Avaliação final no holdout...")
    y_train, y_test = d.train[d.TARGET], d.test[d.TARGET]
    probas = modelling.fit_predict(models, d.train[d.FEATURES], y_train, d.test[d.FEATURES])
    features_sat = d.NUM + ["Satisfaction Score"] + d.CAT
    probas[best + " + Satisfaction (leakage)"] = modelling.fit_predict(
        {"sat": model_sat}, d.train[features_sat], y_train, d.test[features_sat])["sat"]
    holdout = modelling.holdout_report(probas, y_test, d.test["Churn Score"])
    holdout.to_csv(TAB_DIR / "13_holdout.csv")
    p_test = probas[best]
    modelling.calibration_table(y_test, p_test).to_csv(TAB_DIR / "13c_calibracao_holdout.csv")

    # Clientes estabelecidos (tenure > 3): os que entram na lista de prioridade
    est = features.is_established(d.test).to_numpy()
    holdout_est = modelling.holdout_report({best: p_test[est]}, y_test[est],
                                           d.test.loc[est, "Churn Score"])
    holdout_est.to_csv(TAB_DIR / "13d_holdout_estabelecidos.csv")
    modelling.decision_table(y_test[est], p_test[est], thresholds).to_csv(
        TAB_DIR / "13b_limiares_holdout_estabelecidos.csv")
    modelling.calibration_table(y_test[est], p_test[est]).to_csv(
        TAB_DIR / "13e_calibracao_estabelecidos.csv")

    model_plots.plot_roc_calibration(y_test, {
        best: p_test, "Churn Score IBM": d.test["Churn Score"] / 100,
        "Logistic Regression": probas["Logistic Regression"]}, show=False)
    model_plots.plot_metrics_panel(
        holdout.drop(index=modelling.BASELINE, errors="ignore"), best=best,
        name="24b_painel_metricas_holdout", title="Nove métricas por modelo — holdout (20%)",
        show=False)

    step("Permutation importance...")
    importance = modelling.permutation_ranking(models[best], d.test[d.FEATURES], y_test)
    importance.to_frame("queda_auc").to_csv(TAB_DIR / "14_importancia_permutacao.csv")
    plots.plot_ranking(importance.head(20), f"Permutation importance — {best} (queda de AUC, holdout)",
                       "27_importancia_permutacao", highlight=set(d.CAT), show=False)
    return holdout, holdout_est, p_test


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
    prio_test = modelling.priority_list(test_est, p_test_est).merge(
        test_est[["Customer ID", d.TARGET]], on="Customer ID")
    model_plots.plot_priority_map(prio_test, 200, show=False)
    prio_test = prio_test.merge(test_est[["Customer ID"] + retention.ACTION_COLS], on="Customer ID")
    prio_test["acao_sugerida"] = retention.suggest_action(prio_test)
    n_campaign = int(scenarios.loc["Base", "N_a_contactar"])
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


def print_summary(best, reason, holdout, holdout_est, r, rule) -> None:
    """Resumo dos resultados no terminal."""
    print("\n=== Resumo ===")
    print(f"Modelo final: {reason}")
    for name, table in (("Todos os clientes", holdout), ("Só estabelecidos (tenure > 3)", holdout_est)):
        print(f"{name} (holdout):")
        for model in (best, modelling.BENCHMARK, modelling.RULE_BASELINE):
            if model in table.index:
                m = table.loc[model]
                print(f"  {model:<32} AUC {m['auc']:.3f} | PR-AUC {m['pr_auc']:.3f}"
                      f" | Brier {m['brier']:.3f}")
    # Comparar sempre ordenações do mesmo tipo: P × CLTV com P × CLTV, só P com só P
    print("Top 20% dos clientes estabelecidos (% do CLTV perdido coberto):")
    print(f"  P × CLTV: modelo {r.capture.loc['Modelo: P × CLTV', 'top 20%']:.1%}"
          f" | IBM {r.capture.loc['IBM: Churn Score × CLTV', 'top 20%']:.1%}")
    print(f"  Só P:     modelo {r.capture.loc['Modelo: só P(churn)', 'top 20%']:.1%}"
          f" | IBM {r.capture.loc['IBM: só Churn Score', 'top 20%']:.1%}")
    print("Custo-benefício (pressupostos ilustrativos):")
    print(r.scenarios.round(2).to_string())
    print(f"Ações sugeridas no top {r.n_campaign} (cenário Base), hipóteses a testar:")
    print(r.actions.round(2).to_string())
    print("Regra de onboarding (clientes novos):")
    print(rule.round(2).to_string())
    print("Ver outputs/figures, outputs/tables, outputs/artifacts e data/processed.")


def main() -> None:
    """Corre a fase 2 completa e exporta o modelo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rapido", action="store_true",
                        help="teste rápido: sem a variante com Satisfaction Score")
    args = parser.parse_args()

    ensure_dirs()
    apply_theme()
    if not (DATA_PROCESSED / "train.csv").exists():
        sys.exit("Falta data/processed/train.csv: correr primeiro scripts/run_preprocessing.py")

    d = load_data()
    step(f"Treino {d.train[d.FEATURES].shape} | Teste {d.test[d.FEATURES].shape}"
         f" | Joined {len(d.joined)}")
    best_params, models, cv_table, best, reason, thresholds = stage_cv(d)
    model_sat = stage_satisfaction(d, best_params, cv_table, best, args.rapido)
    holdout, holdout_est, p_test = stage_holdout(d, models, best, model_sat, thresholds)
    est = features.is_established(d.test).to_numpy()
    r = stage_retention(d, d.test[est], p_test[est])
    rule, onboarding_joined = stage_onboarding(d)

    # Exportação: o modelo e o que a app e os testes precisam
    inputs = [c for c in d.FEATURES if c not in features.NEW_NUM + features.NEW_CAT]
    modelling.save_bundle(ARTIFACTS_DIR / "final_model.joblib", models[best], best, d.FEATURES,
                          d.TARGET, best_params[best], cv=cv_table.loc[best].to_dict(),
                          holdout=holdout.loc[best].to_dict(), selection=reason,
                          benchmark_cv=cv_table.loc[modelling.BENCHMARK].to_dict(),
                          benchmark_holdout=holdout.loc[modelling.BENCHMARK].to_dict(),
                          holdout_estabelecidos=holdout_est.loc[best].to_dict(),
                          benchmark_holdout_estabelecidos=holdout_est.loc[
                              modelling.BENCHMARK].to_dict(),
                          thresholds=thresholds,
                          capacity_cut=thresholds["top 20% dos estabelecidos"],
                          form=scoring.form_spec(d.train, inputs),
                          cltv_median=float(d.train["CLTV"].median()))
    r.prio_test.to_csv(DATA_PROCESSED / "prioridade_holdout.csv", index=False)
    onboarding_joined.to_csv(DATA_PROCESSED / "onboarding_joined.csv", index=False)

    print_summary(best, reason, holdout, holdout_est, r, rule)


if __name__ == "__main__":
    main()
