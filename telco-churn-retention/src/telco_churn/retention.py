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
