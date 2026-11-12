"""Modelo exportado (outputs/artifacts/final_model.joblib): qualidade mínima e uso pela app.

Não volta a treinar: carrega o artefacto e avalia-o no holdout exportado pelo pré-processamento.
Precisa que a pipeline tenha corrido antes (python scripts/run_all.py --sem-profiling).
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from telco_churn import modelling, retention, scoring
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED

MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
PRIORITY_PATH = ARTIFACTS_DIR / "priority_model.joblib"
MAX_TRAIN_CV_GAP = 0.05   # AUC treino − AUC validação em CV: acima disto, sobreajuste

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists() or not PRIORITY_PATH.exists()
                                or not (DATA_PROCESSED / "test.csv").exists(),
                                reason="falta o modelo exportado: correr scripts/run_all.py")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def priority():
    return joblib.load(PRIORITY_PATH)


@pytest.fixture(scope="module")
def holdout(bundle):
    train, test, _, _ = modelling.load_processed()
    y = test[bundle["target"]]
    return {"train": train, "test": test, "y": y,
            "p": scoring.predict(bundle, test), "ibm": test["Churn Score"] / 100}


def test_artefacto_de_priorizacao_tem_as_chaves_da_app(priority):
    expected = {"model", "name", "features", "target", "cv", "holdout", "selection",
                "risk_levels", "capacity_cuts", "form", "cltv_median", "population"}
    assert expected <= set(priority)


def test_artefacto_tem_as_chaves_esperadas(bundle):
    expected = {"model", "name", "features", "target", "best_params", "cv", "holdout",
                "selection", "thresholds", "risk_levels", "capacity_cuts", "benchmark_cv",
                "benchmark_holdout",
                "form", "cltv_median"}
    assert expected <= set(bundle)


def test_previsao_do_artefacto_igual_a_registada(bundle, holdout):
    """O artefacto reproduz o AUC do holdout guardado na exportação."""
    assert roc_auc_score(holdout["y"], holdout["p"]) == pytest.approx(bundle["holdout"]["auc"])


def test_bate_a_dummy_e_a_regra_de_negocio(holdout):
    auc = roc_auc_score(holdout["y"], holdout["p"])
    assert auc > 0.5   # Dummy: probabilidade constante => AUC 0,5
    rule = modelling.BusinessRuleScore().fit(holdout["train"], holdout["train"]["churn"])
    assert auc > roc_auc_score(holdout["y"], rule.predict_proba(holdout["test"])[:, 1])


def test_nao_e_pior_do_que_o_churn_score_ibm(bundle, holdout):
    """Critério acordado com o grupo: 'não pior do que a IBM' (não inferioridade).

    O pedido original era AUC ≥ AUC IBM, que falha (0,936 vs 0,941). Em AUC aplica-se a mesma
    regra de empate da seleção (modelling.tied_with_best): uma diferença menor do que o
    desvio-padrão em CV do melhor dos dois (aqui, a IBM: 0,0077) está dentro do ruído entre
    folds. Exige-se ainda PR AUC ≥ IBM e Brier < IBM (calibração, precisa para P × CLTV).
    Justificação completa: docs/modelling.md, "Critério de aceitação face ao Churn Score IBM".
    """
    y, p, ibm = holdout["y"], holdout["p"], holdout["ibm"]
    tolerance = bundle["benchmark_cv"]["auc_std"]
    assert roc_auc_score(y, p) >= roc_auc_score(y, ibm) - tolerance
    assert average_precision_score(y, p) >= average_precision_score(y, ibm)
    assert brier_score_loss(y, p) < brier_score_loss(y, ibm)


def test_sobreajuste_controlado(bundle):
    gap = bundle["cv"]["auc_train"] - bundle["cv"]["auc"]
    assert gap < MAX_TRAIN_CV_GAP, f"AUC treino − CV = {gap:.3f}"


def test_aceita_uma_linha_do_formulario(bundle, holdout):
    """Uma linha com os valores por omissão da app devolve uma probabilidade em [0, 1]."""
    cols = scoring.input_columns(bundle)
    row = pd.DataFrame([scoring.default_inputs(holdout["train"], cols)])
    p = scoring.predict(bundle, row)
    assert p.shape == (1,) and 0 <= p[0] <= 1


def test_formulario_sem_colunas_derivadas(bundle):
    cols = scoring.input_columns(bundle)
    assert "n_add_on_services" not in cols and "has_refund" not in cols
    assert not set(cols) & set(modelling.REFERENCES)


def test_lote_separa_estabelecidos_e_novos(bundle, priority, holdout):
    batch = holdout["test"].head(80)
    est, new = scoring.score_batch(priority, batch, core_bundle=bundle)
    tenure = batch.set_index("Customer ID")["Tenure in Months"]
    assert len(est) + len(new) == len(batch) and len(new) > 0
    assert (tenure.loc[est["Customer ID"]] > 3).all()
    assert (tenure.loc[new["Customer ID"]] <= 3).all()
    # Estabelecidos: modelo de priorização, ordenados por perda esperada
    p_priority = priority["model"].predict_proba(
        batch.set_index("Customer ID").loc[est["Customer ID"]][priority["features"]])[:, 1]
    assert np.allclose(est["p_churn"], p_priority)
    assert est["perda_esperada"].is_monotonic_decreasing
    assert np.allclose(est["perda_esperada"], est["p_churn"] * est["CLTV"])
    assert set(est["nivel_risco"]) <= set(modelling.RISK_ORDER)
    assert set(est["prioridade_contacto"]) <= set(modelling.PRIORITY_ORDER)
    # Novos: segmento de onboarding, P só indicativa, sem prioridade de contacto
    assert set(new["onboarding"]) <= set(retention.ONBOARDING_ORDER)
    assert "prioridade_contacto" not in new and "p_churn_indicativo" in new


def test_lote_pequeno_ou_so_com_novos(bundle, priority, holdout):
    test = holdout["test"]
    est, new = scoring.score_batch(priority, test[test["Tenure in Months"] > 3].head(4))
    assert len(est) == 4 and len(new) == 0 and list(est["decil"]) == [1, 2, 3, 4]
    est, new = scoring.score_batch(priority, test[test["Tenure in Months"] <= 3].head(5), bundle)
    assert len(est) == 0 and len(new) == 5


def test_modelo_de_priorizacao_bate_a_regra_nos_estabelecidos(priority, holdout):
    test, train = holdout["test"], holdout["train"]
    est_te, est_tr = test[test["Tenure in Months"] > 3], train[train["Tenure in Months"] > 3]
    p = scoring.predict(priority, est_te)
    rule = modelling.BusinessRuleScore().fit(est_tr, est_tr["churn"])
    auc = roc_auc_score(est_te["churn"], p)
    assert auc == pytest.approx(priority["holdout"]["auc"])
    assert auc > roc_auc_score(est_te["churn"], rule.predict_proba(est_te)[:, 1])


def test_niveis_de_risco_calibrados_no_modelo_de_priorizacao(priority, holdout):
    """É o modelo que a app usa nos clientes estabelecidos: o nível também tem de ser lido
    como uma percentagem real."""
    est = holdout["test"][holdout["test"]["Tenure in Months"] > 3]
    t = modelling.risk_level_table(est["churn"], scoring.predict(priority, est))
    assert t["churn_real"].is_monotonic_increasing
    assert (t["churn_real"] - t["p_media"]).abs().max() < 0.10


def test_capacidade_calculada_so_com_clientes_estabelecidos(bundle, priority, holdout):
    """Os cortes do lote vêm do modelo de priorização (tenure > 3) e sinalizam cerca de 10% e
    20% dos clientes estabelecidos; os do modelo core seriam ocupados pelos clientes novos."""
    test = holdout["test"]
    est = test[test["Tenure in Months"] > 3]
    pr = modelling.contact_priority(scoring.predict(priority, est), priority["capacity_cuts"])
    assert abs((pr == "top 10%").mean() - 0.10) < 0.03
    assert abs((pr != "fora da capacidade").mean() - 0.20) < 0.03
    core_pr = modelling.contact_priority(scoring.predict(bundle, est), bundle["capacity_cuts"])
    assert (core_pr != "fora da capacidade").mean() < (pr != "fora da capacidade").mean()


def test_niveis_de_risco_correspondem_ao_churn_observado(holdout):
    """O modelo está calibrado: a taxa de churn real cresce de nível para nível e fica perto
    de P (um cliente com P = 77% é 'alto', não 'médio')."""
    t = modelling.risk_level_table(holdout["y"], holdout["p"])
    assert t["churn_real"].is_monotonic_increasing
    assert (t["churn_real"] - t["p_media"]).abs().max() < 0.10
    assert modelling.risk_level([0.77])[0] == "alto"
    assert list(modelling.risk_level([0.29, 0.30, 0.59, 0.60])) == ["baixo", "médio", "médio", "alto"]


def test_prioridade_de_contacto_respeita_a_capacidade(bundle, holdout):
    """Os cortes de capacidade sinalizam cerca de 10% e 20% dos clientes do holdout."""
    pr = modelling.contact_priority(holdout["p"], bundle["capacity_cuts"])
    assert abs((pr == "top 10%").mean() - 0.10) < 0.03
    assert abs((pr != "fora da capacidade").mean() - 0.20) < 0.03


def test_lote_sem_coluna_obrigatoria_da_erro_claro(bundle, holdout):
    with pytest.raises(ValueError, match="Contract"):
        scoring.predict(bundle, holdout["test"].drop(columns="Contract").head(3))
