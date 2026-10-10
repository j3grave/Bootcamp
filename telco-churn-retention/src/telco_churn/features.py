"""População de modelação, papéis das variáveis e feature engineering."""
import pandas as pd

from .config import TARGET

ID_COLS = ["Customer ID"]

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


def split_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa Churned/Stayed (modelação) de Joined (scoring) e cria o target binário."""
    df_joined = df[df["Customer Status"] == "Joined"].copy()
    df_model = df[df["Customer Status"].isin(["Churned", "Stayed"])].copy()
    df_model[TARGET] = (df_model["Customer Status"] == "Churned").astype(int)
    return df_model, df_joined


def feature_roles(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    """Devolve (numéricas, categóricas, tabela de papéis)."""
    exclude = set(ID_COLS + LEAKAGE_COLS + MODEL_DERIVED_COLS + SUSPECT_COLS
                  + REDUNDANT_COLS + [TARGET])
    num = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    cat = [c for c in df.columns if c not in exclude and c not in num]
    roles = pd.DataFrame(
        [(c, "id") for c in ID_COLS]
        + [(c, "leakage") for c in LEAKAGE_COLS]
        + [(c, "derivada de modelo (benchmark)") for c in MODEL_DERIVED_COLS]
        + [(c, "suspeita de leakage") for c in SUSPECT_COLS]
        + [(c, "redundante") for c in REDUNDANT_COLS]
        + [(c, "numérica") for c in num]
        + [(c, "categórica") for c in cat],
        columns=["variavel", "papel"],
    )
    return num, cat, roles


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Features calculadas linha a linha (sem estatísticas do dataset => sem fuga)."""
    out = data.copy()
    out["n_add_on_services"] = (out[ADD_ONS] == "Yes").sum(axis=1)
    out["has_refund"] = (out["Total Refunds"] > 0).map({True: "Yes", False: "No"})
    out["avg_hist_charge"] = out["Total Charges"] / out["Tenure in Months"].clip(lower=1)
    out["charge_delta"] = out["Monthly Charge"] - out["avg_hist_charge"]
    return out
