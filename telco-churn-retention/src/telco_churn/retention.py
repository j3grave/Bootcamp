"""Decisões de retenção: regra de onboarding para clientes novos.

As regras deste módulo são hipóteses de negócio tiradas da EDA, não efeitos causais
demonstrados. Antes de serem adotadas devem ser testadas com um grupo de controlo.
"""
import numpy as np
import pandas as pd

ONBOARDING_ORDER = ["Onboarding prioritário", "Onboarding standard", "Sem ação específica"]


def onboarding_segment(df: pd.DataFrame) -> pd.Series:
    """Segmento de onboarding de um cliente novo (tenure <= 3), a partir de duas variáveis.

    Nos clientes novos, a saída concentra-se nos contratos Month-to-Month e, dentro destes,
    na fibra ótica (notebook 02, secção 12). As condições são testadas por ordem: o primeiro
    segmento cujas condições se verificam é o atribuído.
    """
    month_to_month = df["Contract"] == "Month-to-Month"
    fiber = df["Internet Type"] == "Fiber Optic"
    segment = np.select(
        [month_to_month & fiber, month_to_month],
        ["Onboarding prioritário", "Onboarding standard"],
        default="Sem ação específica",
    )
    return pd.Series(segment, index=df.index, name="onboarding")


def onboarding_list(joined: pd.DataFrame) -> pd.DataFrame:
    """Lista dos Joined por segmento de onboarding; dentro de cada segmento, maior CLTV primeiro."""
    out = joined[["Customer ID", "Contract", "Internet Type", "CLTV"]].copy()
    out["onboarding"] = onboarding_segment(joined)
    # Ordenar pela ordem dos segmentos (e não pela ordem alfabética)
    out["ordem_segmento"] = out["onboarding"].map({s: i for i, s in enumerate(ONBOARDING_ORDER)})
    out = out.sort_values(["ordem_segmento", "CLTV"], ascending=[True, False])
    return out.drop(columns="ordem_segmento").reset_index(drop=True)


# --- Priorização dos clientes estabelecidos ------------------------------------------------
def top_k_positions(score, k: int) -> np.ndarray:
    """Posições dos k clientes com maior score."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return order[:k]


def compare_orderings(y, cltv, score_p, score_value, cuts=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Compara duas listas de contacto do mesmo tamanho: por P(churn) e por P(churn) × CLTV.

    Para cada corte (ex.: contactar 20% dos clientes) mostra quantos clientes as duas listas
    têm em comum, que % dos churners cada uma apanha e que % do CLTV perdido cada uma cobre.
    """
    y = np.asarray(y)
    cltv = np.asarray(cltv, dtype=float)
    lost_value = cltv * y          # CLTV de quem saiu de facto
    rows = []
    for cut in cuts:
        k = int(round(cut * len(y)))
        top_p = top_k_positions(score_p, k)
        top_value = top_k_positions(score_value, k)
        rows.append({
            "corte": f"top {cut:.0%}",
            "clientes_contactados": k,
            "em_comum": len(set(top_p) & set(top_value)) / k,
            "churners_apanhados_P": y[top_p].sum() / y.sum(),
            "churners_apanhados_PxCLTV": y[top_value].sum() / y.sum(),
            "CLTV_perdido_coberto_P": lost_value[top_p].sum() / lost_value.sum(),
            "CLTV_perdido_coberto_PxCLTV": lost_value[top_value].sum() / lost_value.sum(),
        })
    return pd.DataFrame(rows).set_index("corte")


# --- Quantos clientes contactar: custo-benefício ------------------------------------------
# Pressupostos (não são dados do dataset; definidos no notebook 02, secção 11.2):
#   taxa_sucesso    = % dos churners contactados que ficam por causa do contacto
#   custo_contacto  = custo de contactar um cliente, incluindo o incentivo oferecido,
#                     na mesma unidade do CLTV. Paga-se por todos os contactados, saiam ou não.
def net_value_curve(y, cltv, score, success_rate: float, cost_per_contact: float) -> pd.DataFrame:
    """Valor líquido observado ao contactar os N primeiros da lista, para N = 1, 2, ...

    valor salvo = taxa_sucesso × CLTV dos churners contactados (y = churn real do holdout)
    valor líquido = valor salvo − N × custo_contacto
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    lost_value = np.asarray(cltv, dtype=float)[order] * np.asarray(y)[order]
    n_contacted = np.arange(1, len(order) + 1)
    saved = success_rate * np.cumsum(lost_value)
    cost = n_contacted * cost_per_contact
    return pd.DataFrame({"clientes_contactados": n_contacted, "valor_salvo": saved,
                         "custo": cost, "valor_liquido": saved - cost})


def expected_contacts(p, cltv, success_rate: float, cost_per_contact: float) -> int:
    """Nº de clientes a contactar pela regra do valor esperado (não usa o churn real).

    Contactar um cliente compensa se P(churn) × CLTV × taxa_sucesso > custo_contacto.
    Com a lista ordenada por P × CLTV, são exatamente os primeiros N da lista.
    """
    expected_gain = np.asarray(p) * np.asarray(cltv, dtype=float) * success_rate
    return int((expected_gain > cost_per_contact).sum())


def scenario_table(y, p, cltv, scenarios: dict) -> pd.DataFrame:
    """N ótimo por cenário: pela regra do valor esperado e pelo valor observado no holdout."""
    rows = []
    for name, params in scenarios.items():
        success_rate, cost = params["taxa_sucesso"], params["custo_contacto"]
        curve = net_value_curve(y, cltv, np.asarray(p) * np.asarray(cltv), success_rate, cost)
        n_expected = expected_contacts(p, cltv, success_rate, cost)
        value_with_rule = 0.0
        if n_expected > 0:
            value_with_rule = curve["valor_liquido"].iloc[n_expected - 1]
        best = curve["valor_liquido"].idxmax()
        rows.append({
            "cenario": name,
            "taxa_sucesso": success_rate,
            "custo_contacto": cost,
            "N_regra_valor_esperado": n_expected,
            "pct_base_regra": n_expected / len(curve),
            "valor_liquido_com_N_regra": value_with_rule,
            "N_otimo_observado": int(curve.loc[best, "clientes_contactados"]),
            "valor_liquido_maximo_observado": curve.loc[best, "valor_liquido"],
        })
    return pd.DataFrame(rows).set_index("cenario")


def sensitivity_grid(p, cltv, success_rates, costs) -> pd.DataFrame:
    """% da base a contactar (regra do valor esperado) para cada taxa de sucesso × custo."""
    grid = pd.DataFrame(index=pd.Index(success_rates, name="taxa_sucesso"),
                        columns=pd.Index(costs, name="custo_contacto"), dtype=float)
    for success_rate in success_rates:
        for cost in costs:
            grid.loc[success_rate, cost] = expected_contacts(p, cltv, success_rate, cost) / len(p)
    return grid
