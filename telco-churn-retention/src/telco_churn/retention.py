"""Decisões de retenção: priorização, custo-benefício, ações e regra de onboarding.

As regras são hipóteses tiradas da EDA, a testar com grupo de controlo antes de adotar.
"""
import numpy as np
import pandas as pd

ONBOARDING_ORDER = ["Onboarding prioritário", "Onboarding standard", "Sem ação específica"]


def onboarding_segment(df: pd.DataFrame) -> pd.Series:
    """Segmento de onboarding de um cliente novo (tenure <= 3).

    A saída concentra-se no Month-to-Month e, dentro deste, na fibra. Vale o primeiro segmento
    cujas condições se verificam.
    """
    month_to_month = df["Contract"] == "Month-to-Month"
    fiber = df["Internet Type"] == "Fiber Optic"
    segment = np.select(
        [month_to_month & fiber, month_to_month],
        ["Onboarding prioritário", "Onboarding standard"],
        default="Sem ação específica",
    )
    return pd.Series(segment, index=df.index, name="onboarding")


def onboarding_rule(new: pd.DataFrame) -> pd.DataFrame:
    """Taxa de saída dos clientes novos por segmento de onboarding e nº de Joined em cada um.

    new: clientes com tenure <= 3, com as colunas 'onboarding' (onboarding_segment), 'saiu'
    (1 = Churned) e 'Customer Status'.
    """
    rule = new.groupby("onboarding")["saiu"].agg(clientes="size", taxa_saida="mean")
    rule["joined_no_segmento"] = new[new["Customer Status"] == "Joined"].groupby(
        "onboarding").size()
    return rule.loc[ONBOARDING_ORDER]


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
    """Compara as listas por P(churn) e por P(churn) × CLTV, do mesmo tamanho.

    Por corte: clientes em comum e % de churners e de CLTV perdido que cada lista apanha.
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
# Pressupostos de negócio, não dados: substituir pelos valores reais da empresa.
# taxa_sucesso = % dos churners contactados que ficam por causa do contacto;
# custo_contacto = custo por cliente contactado, com o incentivo, na unidade do CLTV.
CENARIOS = {
    "Pessimista": {"taxa_sucesso": 0.10, "custo_contacto": 150},
    "Base":       {"taxa_sucesso": 0.20, "custo_contacto": 100},
    "Otimista":   {"taxa_sucesso": 0.30, "custo_contacto": 50},
}


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
    """Por cenário: nº de clientes a contactar pela regra do valor esperado e o valor líquido
    que essa decisão teria dado no holdout."""
    rows = []
    for name, params in scenarios.items():
        success_rate, cost = params["taxa_sucesso"], params["custo_contacto"]
        curve = net_value_curve(y, cltv, np.asarray(p) * np.asarray(cltv), success_rate, cost)
        n_expected = expected_contacts(p, cltv, success_rate, cost)
        value_with_rule = curve["valor_liquido"].iloc[n_expected - 1] if n_expected else 0.0
        rows.append({
            "cenario": name,
            "taxa_sucesso": success_rate,
            "custo_contacto": cost,
            "N_a_contactar": n_expected,
            "pct_base": n_expected / len(curve),
            "valor_liquido_no_holdout": value_with_rule,
        })
    return pd.DataFrame(rows).set_index("cenario")


# --- Ações sugeridas para a lista de intervenção -------------------------------------------
# Hipóteses de negócio a testar, não efeitos causais. Usam só variáveis conhecidas antes do
# churn (Churn Reason / Churn Category não podem ser usadas: só existem depois da saída).
ACTION_ORDER = ["Proposta de contrato anual", "Oferta de suporte técnico e segurança",
                "Revisão de tarifário antes da renovação"]
ACTION_COLS = ["Contract", "Internet Type", "Online Security", "Premium Tech Support"]


def suggest_action(df: pd.DataFrame) -> pd.Series:
    """Ação sugerida para cada cliente da lista; as condições são testadas por ordem.

    1. Month-to-Month: o churn concentra-se nos contratos mensais -> propor contrato anual.
    2. Fibra sem Online Security nem Premium Tech Support: churn mais alto sem estes serviços
       -> oferecer suporte técnico e segurança.
    3. Restantes (já com contrato de 1 ou 2 anos) -> rever o tarifário antes da renovação.
    """
    month_to_month = df["Contract"] == "Month-to-Month"
    fiber_without_support = ((df["Internet Type"] == "Fiber Optic")
                             & (df["Online Security"] == "No")
                             & (df["Premium Tech Support"] == "No"))
    action = np.select([month_to_month, fiber_without_support], ACTION_ORDER[:2],
                       default=ACTION_ORDER[2])
    return pd.Series(action, index=df.index, name="acao_sugerida")


def actions_summary(prio: pd.DataFrame, top_n: int, target: str) -> pd.DataFrame:
    """Top-N da lista por ação sugerida: nº de clientes, perda esperada e churn real (holdout)."""
    top = prio[prio["rank"] <= top_n]
    summary = top.groupby("acao_sugerida").agg(
        clientes=("Customer ID", "size"),
        perda_esperada=("perda_esperada", "sum"),
        churn_real=(target, "mean"),
    )
    return summary.reindex(ACTION_ORDER).dropna(how="all")
