"""Usar o modelo exportado: um cliente (formulário da app), um lote (CSV) e a decisão do desconto.

Este módulo não depende do Gradio: a app (app/app.py) só trata da interface e chama estas
funções, que são testadas em tests/test_model.py e tests/test_app.py.
"""
import numpy as np
import pandas as pd

from .features import NEW_CAT, NEW_CUSTOMER_MAX_TENURE, NEW_NUM, add_features
from .modelling import contact_priority, priority_list, risk_level

ID_COL, CLTV_COL = "Customer ID", "CLTV"


def input_columns(bundle: dict) -> list[str]:
    """Variáveis que o utilizador indica: as features do modelo sem as derivadas."""
    return [c for c in bundle["features"] if c not in NEW_NUM + NEW_CAT]


def default_inputs(train: pd.DataFrame, columns: list[str]) -> dict:
    """Valor por omissão de cada variável: mediana nas numéricas, moda nas categóricas."""
    defaults = {}
    for c in columns:
        if pd.api.types.is_numeric_dtype(train[c]):
            defaults[c] = float(train[c].median())
        else:
            defaults[c] = train[c].mode().iloc[0]
    return defaults


def form_spec(train: pd.DataFrame, columns: list[str]) -> dict:
    """Especificação do formulário da app, tirada do treino e guardada no artefacto.

    Numéricas: valor por omissão (mediana), mínimo, máximo e se são inteiras.
    Categóricas: valor por omissão (moda) e opções possíveis.
    Assim a app não precisa dos dados de treino.
    """
    defaults = default_inputs(train, columns)
    spec = {}
    for c in columns:
        if pd.api.types.is_numeric_dtype(train[c]):
            spec[c] = {"default": defaults[c], "min": float(train[c].min()),
                       "max": float(train[c].max()),
                       "integer": bool(pd.api.types.is_integer_dtype(train[c]))}
        else:
            spec[c] = {"default": defaults[c], "options": sorted(train[c].astype(str).unique())}
    return spec


def predict(bundle: dict, customers: pd.DataFrame) -> np.ndarray:
    """P(churn) de cada cliente. As features derivadas são calculadas aqui (add_features)."""
    missing = [c for c in input_columns(bundle) if c not in customers.columns]
    if missing:
        raise ValueError(f"Faltam colunas: {', '.join(missing)}")
    X = add_features(customers)[bundle["features"]]
    return bundle["model"].predict_proba(X)[:, 1]


def is_new_customer(tenure) -> np.ndarray:
    """Clientes com poucos meses de casa: o modelo extrapola (ver docs/modelling.md)."""
    return np.asarray(tenure) <= NEW_CUSTOMER_MAX_TENURE


def discount_decision(p_churn: float, cltv: float, discount: float, months: float,
                      effectiveness: float) -> dict:
    """Vale a pena oferecer o desconto? Dar se P(churn) × eficácia × CLTV > desconto × meses.

    p_churn:       probabilidade de saída prevista pelo modelo (0 a 1)
    cltv:          valor do cliente (mesma unidade monetária do desconto)
    discount:      desconto mensal oferecido
    months:        duração do desconto, em meses
    effectiveness: % de clientes em risco que ficam por causa do desconto (0 a 1). É um
                   PRESSUPOSTO: o modelo prevê o risco, não o efeito causal do desconto.
    """
    if not 0 <= p_churn <= 1 or not 0 <= effectiveness <= 1:
        raise ValueError("P(churn) e eficácia têm de estar entre 0 e 1")
    if min(cltv, discount, months) < 0:
        raise ValueError("CLTV, desconto e duração não podem ser negativos")
    expected_gain = p_churn * effectiveness * cltv
    cost = discount * months
    return {"ganho_esperado": expected_gain, "custo": cost, "valor_liquido": expected_gain - cost,
            "dar_desconto": bool(expected_gain > cost)}


def local_factors(bundle: dict, customer: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Principais fatores de um cliente: quanto muda P(churn) se cada variável tivesse o valor
    típico (mediana/moda do treino), mantendo as restantes.

    Positivo = o valor deste cliente aumenta o risco face ao típico. É uma explicação local e
    simples (sem SHAP), e mostra associação, não causa.
    """
    base = predict(bundle, customer)[0]
    variants = []
    for col, s in bundle["form"].items():
        row = customer.copy()
        row[col] = s["default"]
        variants.append(row)
    p_typical = predict(bundle, pd.concat(variants, ignore_index=True))
    out = pd.DataFrame({"variavel": list(bundle["form"]),
                        "valor_cliente": [customer[c].iloc[0] for c in bundle["form"]],
                        "valor_tipico": [s["default"] for s in bundle["form"].values()],
                        "efeito_em_P": base - p_typical})
    out = out[out["efeito_em_P"].abs() > 1e-6]   # variáveis em que o cliente já é típico saem
    return out.reindex(out["efeito_em_P"].abs().sort_values(ascending=False).index).head(top)


def score_batch(bundle: dict, customers: pd.DataFrame) -> pd.DataFrame:
    """Lista de prioridade de um lote: ordenada por perda esperada = P(churn) × CLTV.

    Acrescenta o nível de risco (absoluto), a prioridade de contacto (capacidade da equipa)
    e um aviso para clientes novos (tenure baixo).
    """
    for col in (ID_COL, CLTV_COL):
        if col not in customers.columns:
            raise ValueError(f"O ficheiro tem de ter a coluna '{col}'")
    customers = customers.reset_index(drop=True)
    p = predict(bundle, customers)
    out = priority_list(customers, p)
    extra = customers.set_index(ID_COL).loc[out[ID_COL]]
    out["nivel_risco"] = risk_level(out["p_churn"], bundle["risk_levels"])
    out["prioridade_contacto"] = contact_priority(out["p_churn"], bundle["capacity_cuts"])
    out["cliente_novo"] = is_new_customer(extra["Tenure in Months"].to_numpy())
    return out
