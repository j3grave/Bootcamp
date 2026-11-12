"""População de modelação, papéis das variáveis e feature engineering."""
import pandas as pd

from .config import TARGET

ID_COLS = ["Customer ID"]

# Colunas de negócio guardadas em train/test ao lado das features (não entram no modelo)
BUSINESS_COLS = ["Customer ID", "CLTV", "Churn Score", "Satisfaction Score"]

# Codificam o resultado ou só existem depois do churn -> nunca entram no modelo
LEAKAGE_COLS = ["Customer Status", "Churn Label", "Churn Value", "Churn Category", "Churn Reason"]

# Saídas de modelos/fórmulas existentes: Churn Score = benchmark; CLTV = priorização
MODEL_DERIVED_COLS = ["Churn Score", "CLTV"]

# Scores 1–2 => 100% churn: suspeita de registo no contexto da saída
SUSPECT_COLS = ["Satisfaction Score"]

# Constantes, duplicadas, derivações exatas ou identificadores de alta cardinalidade
REDUNDANT_COLS = [
    "Country", "State", "Lat Long", "Under 30", "Senior Citizen", "Dependents",
    "Referred a Friend", "Total Revenue", "Zip Code", "City",
]

ADD_ONS = ["Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support",
           "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"]

NEW_NUM = ["n_add_on_services", "avg_hist_charge", "charge_delta"]
NEW_CAT = ["has_refund"]

# Os Joined têm todos 1 a 3 meses de casa. Um cliente novo que ficou é rotulado Joined (e sai
# da população core); um cliente novo que saiu é rotulado Churned (e fica). Por isso, dentro de
# Churned + Stayed, tenure <= 3 significa quase sempre churn: é um efeito da forma como os
# estados foram definidos, não um padrão de comportamento.
NEW_CUSTOMER_MAX_TENURE = 3


def split_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa Churned/Stayed (modelação) de Joined (scoring) e cria o target binário."""
    df_joined = df[df["Customer Status"] == "Joined"].copy()
    df_model = df[df["Customer Status"].isin(["Churned", "Stayed"])].copy()
    df_model[TARGET] = (df_model["Customer Status"] == "Churned").astype(int)
    return df_model, df_joined


def is_established(df: pd.DataFrame) -> pd.Series:
    """True para clientes com mais de NEW_CUSTOMER_MAX_TENURE meses de casa."""
    return df["Tenure in Months"] > NEW_CUSTOMER_MAX_TENURE


def tenure_status_table(df: pd.DataFrame) -> pd.DataFrame:
    """Nº de clientes por faixa de tenure e Customer Status, com a taxa de churn core.

    A taxa usa só Churned e Stayed, como o modelo core. Mostra o artefacto dos clientes novos.
    """
    faixa = is_established(df).map({False: f"1–{NEW_CUSTOMER_MAX_TENURE} meses",
                                     True: f"{NEW_CUSTOMER_MAX_TENURE + 1} ou mais meses"})
    table = pd.crosstab(faixa.rename("tenure"), df["Customer Status"])
    table["taxa_churn_core"] = table["Churned"] / (table["Churned"] + table["Stayed"])
    return table


def feature_roles(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    """Devolve (numéricas, categóricas, tabela de papéis)."""
    excluded = {
        "id": ID_COLS,
        "leakage": LEAKAGE_COLS,
        "derivada de modelo (benchmark)": MODEL_DERIVED_COLS,
        "suspeita de leakage": SUSPECT_COLS,
        "redundante": REDUNDANT_COLS,
    }
    exclude = {c for cols in excluded.values() for c in cols} | {TARGET}
    num = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    cat = [c for c in df.columns if c not in exclude and c not in num]
    roles = excluded | {"numérica": num, "categórica": cat}
    table = pd.DataFrame([(c, papel) for papel, cols in roles.items() for c in cols],
                         columns=["variavel", "papel"])
    return num, cat, table


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Features calculadas linha a linha (sem estatísticas do dataset => sem fuga)."""
    out = data.copy()
    out["n_add_on_services"] = (out[ADD_ONS] == "Yes").sum(axis=1)
    out["has_refund"] = (out["Total Refunds"] > 0).map({True: "Yes", False: "No"})
    out["avg_hist_charge"] = out["Total Charges"] / out["Tenure in Months"].clip(lower=1)
    out["charge_delta"] = out["Monthly Charge"] - out["avg_hist_charge"]
    return out
