"""Bootstrap emparelhado da diferença de AUC (modelling.paired_bootstrap_auc_diff).

Testa o método, não o resultado face ao IBM: o intervalo é um reporte e não decide o deploy.
"""
import numpy as np

from telco_churn.modelling import paired_bootstrap_auc_diff

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 600)
GOOD = np.clip(Y * 0.5 + rng.random(600) * 0.6, 0, 1)   # separa bem as classes
NOISE = rng.random(600)                                 # ao acaso


def test_scores_iguais_dao_diferenca_zero():
    r = paired_bootstrap_auc_diff(Y, GOOD, GOOD, n_boot=200)
    assert r["diferenca"] == 0 and r["ic_inferior"] == 0 and r["ic_superior"] == 0
    assert r["ic_inclui_zero"]


def test_modelo_claramente_melhor_nao_inclui_zero():
    r = paired_bootstrap_auc_diff(Y, GOOD, NOISE, n_boot=500)
    assert r["ic_inferior"] > 0 and not r["ic_inclui_zero"]
    assert r["ic_inferior"] <= r["diferenca"] <= r["ic_superior"]


def test_reprodutivel_com_random_state_fixo():
    a = paired_bootstrap_auc_diff(Y, GOOD, NOISE, n_boot=300, random_state=7)
    b = paired_bootstrap_auc_diff(Y, GOOD, NOISE, n_boot=300, random_state=7)
    assert a == b
