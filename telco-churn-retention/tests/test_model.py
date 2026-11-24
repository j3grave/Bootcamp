"""Modelo exportado: qualidade mínima e uso pela app, sem retreinar.

Precisa da pipeline corrida antes (python scripts/run_pipeline.py --sem-profiling).
"""
import joblib
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from telco_churn import modelling, retention, scoring
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED

MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
MAX_TRAIN_CV_GAP = 0.05   # AUC treino − AUC validação em CV: acima disto, sobreajuste

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists() or not (DATA_PROCESSED / "test.csv").exists(),
                                reason="falta o modelo exportado: correr scripts/run_pipeline.py")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def holdout(bundle):
    train, test, _, _ = modelling.load_processed()
    est = test["Tenure in Months"] > 3
    return {"train": train, "test": test, "y": test[bundle["target"]],
            "p": scoring.predict(bundle, test), "est": est}


def test_artefacto_tem_as_chaves_da_app(bundle):
    expected = {"model", "name", "features", "target", "best_params", "cv", "holdout",
                "selection", "benchmark_cv", "benchmark_holdout", "holdout_estabelecidos",
                "benchmark_holdout_estabelecidos", "thresholds", "capacity_cut", "form",
                "cltv_median"}
    assert expected <= set(bundle)


def test_previsao_do_artefacto_igual_a_registada(bundle, holdout):
    """O artefacto reproduz o AUC do holdout guardado na exportação."""
    assert roc_auc_score(holdout["y"], holdout["p"]) == pytest.approx(bundle["holdout"]["auc"])


def test_bate_a_dummy_e_a_regra_de_negocio(holdout):
    auc = roc_auc_score(holdout["y"], holdout["p"])
    assert auc > 0.5   # Dummy: probabilidade constante => AUC 0,5
    rule = modelling.BusinessRuleScore().fit(holdout["train"], holdout["train"]["churn"])
    assert auc > roc_auc_score(holdout["y"], rule.predict_proba(holdout["test"])[:, 1])


def test_sobreajuste_controlado(bundle):
    gap = bundle["cv"]["auc_train"] - bundle["cv"]["auc"]
    assert gap < MAX_TRAIN_CV_GAP, f"AUC treino − CV = {gap:.3f}"


def test_mais_bem_calibrado_do_que_a_ibm_nos_estabelecidos(holdout):
    """P(churn) × CLTV precisa de probabilidades realistas nos clientes que vão para a lista."""
    est, test = holdout["est"], holdout["test"]
    y = holdout["y"][est]
    assert brier_score_loss(y, holdout["p"][est]) < brier_score_loss(
        y, test.loc[est, "Churn Score"] / 100)


def test_corte_de_capacidade_sinaliza_cerca_de_20_por_cento(bundle, holdout):
    """O corte do top 20%, fixado no treino, sinaliza cerca de 20% dos estabelecidos do holdout."""
    flagged = modelling.in_top_capacity(holdout["p"][holdout["est"]], bundle["capacity_cut"])
    assert abs(flagged.mean() - 0.20) < 0.03


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


def test_lote_separa_estabelecidos_e_novos(bundle, holdout):
    batch = holdout["test"].head(80)
    est, new = scoring.score_batch(bundle, batch)
    tenure = batch.set_index("Customer ID")["Tenure in Months"]
    assert len(est) + len(new) == len(batch) and len(new) > 0
    assert (tenure.loc[est["Customer ID"]] > 3).all()
    assert (tenure.loc[new["Customer ID"]] <= 3).all()
    # Estabelecidos: ordenados por perda esperada
    assert est["perda_esperada"].is_monotonic_decreasing
    assert (est["perda_esperada"] - est["p_churn"] * est["CLTV"]).abs().max() < 1e-9
    # Novos: segmento de onboarding, P só indicativa
    assert set(new["onboarding"]) <= set(retention.ONBOARDING_ORDER)
    assert "top_20_risco" not in new and "p_churn_indicativo" in new


def test_lote_pequeno_ou_so_com_novos(bundle, holdout):
    test = holdout["test"]
    est, new = scoring.score_batch(bundle, test[test["Tenure in Months"] > 3].head(4))
    assert len(est) == 4 and len(new) == 0 and list(est["decil"]) == [1, 2, 3, 4]
    est, new = scoring.score_batch(bundle, test[test["Tenure in Months"] <= 3].head(5))
    assert len(est) == 0 and len(new) == 5


def test_lote_sem_coluna_obrigatoria_da_erro_claro(bundle, holdout):
    with pytest.raises(ValueError, match="Contract"):
        scoring.predict(bundle, holdout["test"].drop(columns="Contract").head(3))
