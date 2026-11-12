"""Decisão do desconto (scoring.discount_decision): coerente nos casos limite.

Regra: dar o desconto se P(churn) × eficácia × CLTV > desconto mensal × meses.
"""
import pytest

from telco_churn.scoring import discount_decision, is_new_customer


def test_sem_risco_nunca_compensa():
    d = discount_decision(p_churn=0, cltv=5000, discount=10, months=6, effectiveness=0.3)
    assert d["ganho_esperado"] == 0 and not d["dar_desconto"]


def test_risco_total_compensa_quando_o_ganho_supera_o_custo():
    d = discount_decision(p_churn=1, cltv=5000, discount=10, months=6, effectiveness=0.3)
    assert d["ganho_esperado"] == pytest.approx(1500)
    assert d["custo"] == 60 and d["dar_desconto"]


def test_risco_total_nao_compensa_com_desconto_caro():
    d = discount_decision(p_churn=1, cltv=1000, discount=100, months=12, effectiveness=0.3)
    assert d["valor_liquido"] == pytest.approx(300 - 1200)
    assert not d["dar_desconto"]


def test_custo_zero_compensa_se_houver_algum_risco():
    assert discount_decision(0.2, 3000, discount=0, months=6, effectiveness=0.3)["dar_desconto"]
    assert not discount_decision(0, 3000, discount=0, months=6, effectiveness=0.3)["dar_desconto"]


def test_eficacia_zero_nunca_compensa():
    assert not discount_decision(0.9, 6000, discount=5, months=3, effectiveness=0)["dar_desconto"]


def test_empate_nao_da_desconto():
    """Ganho igual ao custo: não há valor líquido, não se dá."""
    d = discount_decision(p_churn=0.5, cltv=1000, discount=50, months=3, effectiveness=0.3)
    assert d["valor_liquido"] == pytest.approx(0) and not d["dar_desconto"]


@pytest.mark.parametrize("kwargs", [
    dict(p_churn=1.2, cltv=1000, discount=10, months=3, effectiveness=0.3),
    dict(p_churn=0.5, cltv=1000, discount=10, months=3, effectiveness=-0.1),
    dict(p_churn=0.5, cltv=-1, discount=10, months=3, effectiveness=0.3),
    dict(p_churn=0.5, cltv=1000, discount=-5, months=3, effectiveness=0.3),
])
def test_valores_invalidos_dao_erro(kwargs):
    with pytest.raises(ValueError):
        discount_decision(**kwargs)


def test_aviso_de_cliente_novo():
    assert list(is_new_customer([1, 3, 4, 40])) == [True, True, False, False]
