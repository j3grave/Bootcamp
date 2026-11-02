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
