"""Contratos de dados: o que os ficheiros têm de cumprir para a pipeline poder continuar.

Cada regra é um esquema pandera pequeno com nome, camada e severidade:
    erro  -> a pipeline pára (DataValidationError) com a lista das regras que falharam;
    aviso -> regista-se no relatório e a pipeline continua.

Camadas: os 5 ficheiros brutos, a base unida e o resultado do pré-processamento (train/test).
O relatório fica em outputs/tables/00_validacao_dados.csv (regra, camada, severidade,
resultado, linhas_a_falhar). As regras estão documentadas no README ("Data Contracts").

Os intervalos numéricos são plausíveis, não os mínimos/máximos observados: deixam margem para
dados novos, mas apanham erros de registo (ex.: idade negativa, Satisfaction Score = 9).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .config import TAB_DIR, TARGET

ERRO, AVISO = "erro", "aviso"
REPORT_PATH = TAB_DIR / "00_validacao_dados.csv"

# Valores de referência das estatísticas (aviso, não erro)
EXPECTED_ROWS = 7043          # clientes no extrato IBM (Q3)
ROWS_TOLERANCE = 0.05         # ± 5% de linhas
EXPECTED_CHURN_RATE = 0.284   # Churned / (Churned + Stayed) no extrato de referência
CHURN_RATE_TOLERANCE = 0.03   # ± 3 pontos percentuais
MAX_MISSING_PCT = 0.05        # % máxima de valores em falta por coluna
EXPECTED_EMPTY = ["Churn Category", "Churn Reason"]   # vazias por definição nos não-Churned

YES_NO = ["Yes", "No"]
ADD_ONS = ["Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support",
           "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"]


class DataValidationError(ValueError):
    """Um ou mais contratos de dados com severidade 'erro' falharam."""


@dataclass(frozen=True)
class Rule:
    name: str
    layer: str
    severity: str
    count_failures: Callable[[pd.DataFrame], int]   # nº de linhas que violam a regra


def _pandera(schema: pa.DataFrameSchema) -> Callable[[pd.DataFrame], int]:
    """Transforma um esquema pandera numa função que conta as linhas que o violam."""
    def count(df: pd.DataFrame) -> int:
        try:
            schema.validate(df, lazy=True)
            return 0
        except pa.errors.SchemaErrors as e:
            idx = e.failure_cases["index"].dropna()
            return int(idx.nunique()) if len(idx) else len(df)   # sem índice = tabela inteira
    return count


def _row_rule(check: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], int]:
    """Regra linha a linha: `check` devolve True nas linhas que cumprem."""
    return _pandera(pa.DataFrameSchema(checks=pa.Check(check)))


def _column(col: str, *checks, unique: bool = False) -> Callable[[pd.DataFrame], int]:
    return _pandera(pa.DataFrameSchema({col: pa.Column(checks=list(checks), unique=unique)}))


# --- 1. Ficheiros brutos: colunas e tipos ----------------------------------------------------
# strict=True: colunas a mais ou a menos também são erro. str = texto, int/float = números.
RAW_COLUMNS = {
    "demographics": {
        "Customer ID": str, "Count": int, "Gender": str, "Age": int, "Under 30": str,
        "Senior Citizen": str, "Married": str, "Dependents": str, "Number of Dependents": int,
    },
    "location": {
        "Customer ID": str, "Count": int, "Country": str, "State": str, "City": str,
        "Zip Code": int, "Lat Long": str, "Latitude": float, "Longitude": float,
    },
    "population": {"ID": int, "Zip Code": int, "Population": int},
    "services": {
        "Customer ID": str, "Count": int, "Quarter": str, "Referred a Friend": str,
        "Number of Referrals": int, "Tenure in Months": int, "Offer": str, "Phone Service": str,
        "Avg Monthly Long Distance Charges": float, "Multiple Lines": str,
        "Internet Service": str, "Internet Type": str, "Avg Monthly GB Download": int,
        **{c: str for c in ADD_ONS}, "Contract": str, "Paperless Billing": str,
        "Payment Method": str, "Monthly Charge": float, "Total Charges": float,
        "Total Refunds": float, "Total Extra Data Charges": int,
        "Total Long Distance Charges": float, "Total Revenue": float,
    },
    "status": {
        "Customer ID": str, "Count": int, "Quarter": str, "Satisfaction Score": int,
        "Customer Status": str, "Churn Label": str, "Churn Value": int, "Churn Score": int,
        "CLTV": int, "Churn Category": str, "Churn Reason": str,
    },
}

# --- 2. Domínios: valores permitidos e intervalos plausíveis ---------------------------------
# (ficheiro, coluna, check). Observado nos dados (nov. 2026): Age 19–80, Tenure 1–72,
# Monthly Charge 18,25–118,75, Total Charges 18,80–8 684,80, Churn Score 5–96.
DOMAINS = [
    ("demographics", "Gender", pa.Check.isin(["Male", "Female"])),
    ("demographics", "Age", pa.Check.in_range(18, 100)),
    ("demographics", "Number of Dependents", pa.Check.in_range(0, 15)),
    *[("demographics", c, pa.Check.isin(YES_NO))
      for c in ["Under 30", "Senior Citizen", "Married", "Dependents"]],
    ("location", "Country", pa.Check.isin(["United States"])),
    ("location", "State", pa.Check.isin(["California"])),
    ("location", "Zip Code", pa.Check.in_range(90001, 96162)),       # ZIP codes da Califórnia
    ("location", "Latitude", pa.Check.in_range(32.5, 42.1)),
    ("location", "Longitude", pa.Check.in_range(-124.5, -114.1)),
    ("population", "Population", pa.Check.gt(0)),
    ("services", "Tenure in Months", pa.Check.in_range(1, 120)),
    ("services", "Number of Referrals", pa.Check.in_range(0, 50)),
    ("services", "Offer", pa.Check.isin(["None", "Offer A", "Offer B", "Offer C", "Offer D",
                                         "Offer E"])),
    ("services", "Internet Type", pa.Check.isin(["None", "DSL", "Cable", "Fiber Optic"])),
    ("services", "Contract", pa.Check.isin(["Month-to-Month", "One Year", "Two Year"])),
    ("services", "Payment Method", pa.Check.isin(["Bank Withdrawal", "Credit Card",
                                                  "Mailed Check"])),
    *[("services", c, pa.Check.isin(YES_NO))
      for c in ["Referred a Friend", "Phone Service", "Multiple Lines", "Internet Service",
                "Paperless Billing", *ADD_ONS]],
    ("services", "Monthly Charge", pa.Check.in_range(0, 300)),
    ("services", "Total Charges", pa.Check.in_range(0, 50_000)),
    ("services", "Total Refunds", pa.Check.ge(0)),
    ("services", "Total Extra Data Charges", pa.Check.ge(0)),
    ("services", "Total Long Distance Charges", pa.Check.ge(0)),
    ("services", "Avg Monthly Long Distance Charges", pa.Check.in_range(0, 100)),
    ("services", "Avg Monthly GB Download", pa.Check.in_range(0, 500)),
    ("status", "Satisfaction Score", pa.Check.isin([1, 2, 3, 4, 5])),
    ("status", "Churn Score", pa.Check.in_range(0, 100)),
    ("status", "CLTV", pa.Check.gt(0)),
    ("status", "Customer Status", pa.Check.isin(["Churned", "Stayed", "Joined"])),
    ("status", "Churn Label", pa.Check.isin(YES_NO)),
    ("status", "Churn Value", pa.Check.isin([0, 1])),
    ("status", "Churn Category", pa.Check.isin(["", "Attitude", "Competitor",
                                                "Dissatisfaction", "Other", "Price"])),
]


def _describe(col: str, check: pa.Check) -> str:
    """Nome legível da regra (ex.: 'Age ∈ [18, 100]', 'Gender ∈ {Male, Female}')."""
    s = check.statistics
    if check.name == "isin":
        return f"{col} ∈ {{{', '.join(repr(v) if v == '' else str(v) for v in s['allowed_values'])}}}"
    if check.name == "in_range":
        return f"{col} ∈ [{s['min_value']}, {s['max_value']}]"
    symbol = {"greater_than": ">", "greater_than_or_equal_to": "≥"}[check.name]
    return f"{col} {symbol} {s['min_value']}"


def raw_rules() -> list[Rule]:
    rules = [Rule(f"colunas e tipos ({len(cols)} colunas)", table, ERRO,
                  _pandera(pa.DataFrameSchema({c: pa.Column(t) for c, t in cols.items()},
                                              strict=True)))
             for table, cols in RAW_COLUMNS.items()]
    rules += [Rule("Customer ID único", t, ERRO, _column("Customer ID", unique=True))
              for t in ("demographics", "location", "services", "status")]
    rules.append(Rule("Zip Code único", "population", ERRO, _column("Zip Code", unique=True)))
    rules += [Rule(_describe(col, check), table, ERRO, _column(col, check))
              for table, col, check in DOMAINS]
    return rules


def validate_raw(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Valida os 5 ficheiros brutos e as relações entre eles. Devolve o relatório."""
    rows = [_evaluate(rule, raw[rule.layer]) for rule in raw_rules()]

    # Relações entre ficheiros (não cabem num esquema de um só ficheiro)
    ids = raw["status"]["Customer ID"]
    for table in ("demographics", "location", "services"):
        other = raw[table]["Customer ID"]
        n = int((~other.isin(ids)).sum() + (~ids.isin(other)).sum())
        rows.append(_row("mesmos Customer ID que status", table, ERRO, n))
    missing_zip = ~raw["location"]["Zip Code"].isin(raw["population"]["Zip Code"])
    rows.append(_row("todos os Zip Code de location existem em population", "location", ERRO,
                     int(missing_zip.sum())))
    return pd.DataFrame(rows)


# --- 3. Base unida: regras de negócio e estatísticas -----------------------------------------
def _is(df, col, value):
    return df[col] == value


def merged_rules() -> list[Rule]:
    churned = lambda d: _is(d, "Customer Status", "Churned")   # noqa: E731
    no_internet = lambda d: _is(d, "Internet Service", "No")   # noqa: E731
    no_phone = lambda d: _is(d, "Phone Service", "No")         # noqa: E731
    total_revenue = lambda d: (d["Total Charges"] - d["Total Refunds"]   # noqa: E731
                               + d["Total Extra Data Charges"] + d["Total Long Distance Charges"])
    business = {
        "Churn Label = Yes ⇔ Customer Status = Churned":
            lambda d: _is(d, "Churn Label", "Yes") == churned(d),
        "Churn Value = 1 ⇔ Customer Status = Churned":
            lambda d: (d["Churn Value"] == 1) == churned(d),
        "Churn Category preenchida ⇔ Churned": lambda d: (d["Churn Category"] != "") == churned(d),
        "Churn Reason preenchida ⇔ Churned": lambda d: (d["Churn Reason"] != "") == churned(d),
        "Internet Service = No ⇒ add-ons de internet = No":
            lambda d: ~no_internet(d) | (d[ADD_ONS] == "No").all(axis=1),
        "Internet Service = No ⇔ Internet Type = None":
            lambda d: no_internet(d) == _is(d, "Internet Type", "None"),
        "Internet Service = No ⇒ Avg Monthly GB Download = 0":
            lambda d: ~no_internet(d) | (d["Avg Monthly GB Download"] == 0),
        "Phone Service = No ⇒ Multiple Lines = No":
            lambda d: ~no_phone(d) | _is(d, "Multiple Lines", "No"),
        "Phone Service = No ⇒ Avg Monthly Long Distance Charges = 0":
            lambda d: ~no_phone(d) | (d["Avg Monthly Long Distance Charges"] == 0),
        # Derivações exatas (antes em data.derivation_checks, agora obrigatórias a 100%)
        "Under 30 = Yes ⇔ Age < 30": lambda d: (d["Age"] < 30) == _is(d, "Under 30", "Yes"),
        "Senior Citizen = Yes ⇔ Age ≥ 65":
            lambda d: (d["Age"] >= 65) == _is(d, "Senior Citizen", "Yes"),
        "Dependents = Yes ⇔ Number of Dependents > 0":
            lambda d: (d["Number of Dependents"] > 0) == _is(d, "Dependents", "Yes"),
        "Referred a Friend = Yes ⇔ Number of Referrals > 0":
            lambda d: (d["Number of Referrals"] > 0) == _is(d, "Referred a Friend", "Yes"),
        "Total Revenue = Charges − Refunds + Extra Data + Long Distance":
            lambda d: pd.Series(np.isclose(d["Total Revenue"], total_revenue(d), atol=0.01),
                                index=d.index),
    }
    rules = [Rule(name, "base unida", ERRO, _row_rule(check)) for name, check in business.items()]
    # Propriedade deste extrato, não contrato: os Joined são clientes com 1 a 3 meses de casa
    rules.append(Rule("Customer Status = Joined ⇒ Tenure in Months ≤ 3", "base unida", AVISO,
                      _row_rule(lambda d: ~_is(d, "Customer Status", "Joined")
                                | (d["Tenure in Months"] <= 3))))
    return rules


def validate_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Regras de negócio e estatísticas da base unida (1 linha por cliente)."""
    rows = [_evaluate(rule, df) for rule in merged_rules()]

    n = len(df)
    off = abs(n - EXPECTED_ROWS) > ROWS_TOLERANCE * EXPECTED_ROWS
    rows.append(_row(f"nº de clientes ≈ {EXPECTED_ROWS} (± {ROWS_TOLERANCE:.0%}) — "
                     f"observado {n}", "base unida", AVISO, n if off else 0))
    model = df[df["Customer Status"].isin(["Churned", "Stayed"])]
    rate = (model["Customer Status"] == "Churned").mean()
    off = abs(rate - EXPECTED_CHURN_RATE) > CHURN_RATE_TOLERANCE
    rows.append(_row(f"taxa de churn ≈ {EXPECTED_CHURN_RATE:.1%} (± {CHURN_RATE_TOLERANCE * 100:.0f} p.p.) "
                     f"— observada {rate:.1%}", "base unida", AVISO, len(model) if off else 0))
    missing = (df.isna() | (df == "")).mean().drop(EXPECTED_EMPTY, errors="ignore")
    too_many = missing[missing > MAX_MISSING_PCT]
    rows.append(_row(f"% de missing ≤ {MAX_MISSING_PCT:.0%} por coluna"
                     + (f" — acima: {', '.join(too_many.index)}" if len(too_many) else ""),
                     "base unida", AVISO, int((df[too_many.index].isna()
                                               | (df[too_many.index] == "")).any(axis=1).sum())))
    return pd.DataFrame(rows)


# --- 4. Saída do pré-processamento ------------------------------------------------------------
def validate_split(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                   preprocessor, forbidden: list[str]) -> pd.DataFrame:
    """Treino/teste disjuntos e estratificados, sem leakage nas features, sem NaN no fim.

    preprocessor: o ColumnTransformer já ajustado no treino.
    forbidden: colunas que nunca podem ser features (leakage, derivadas de modelo, suspeitas).
    """
    layer = "pré-processamento"
    common = set(train["Customer ID"]) & set(test["Customer ID"])
    gap = abs(train[TARGET].mean() - test[TARGET].mean())
    leaked = [c for c in features if c in forbidden]
    n_nan = sum(int(np.isnan(preprocessor.transform(d[features])).any(axis=1).sum())
                for d in (train, test))
    return pd.DataFrame([
        _row("treino e teste sem Customer ID em comum", layer, ERRO, len(common)),
        _row(f"taxa de churn semelhante em treino e teste (|Δ| ≤ 1 p.p.) — Δ = {gap:.2%}",
             layer, ERRO, len(test) if gap > 0.01 else 0),
        _row("nenhuma coluna de leakage / derivada de modelo / suspeita nas features"
             + (f" — encontradas: {', '.join(leaked)}" if leaked else ""), layer, ERRO, len(leaked)),
        _row("zero NaN depois do pré-processador", layer, ERRO, n_nan),
    ])


# --- Relatório e aplicação --------------------------------------------------------------------
def _row(name: str, layer: str, severity: str, n_failures: int) -> dict:
    return {"regra": name, "camada": layer, "severidade": severity,
            "resultado": "passa" if n_failures == 0 else "falha", "linhas_a_falhar": n_failures}


def _evaluate(rule: Rule, df: pd.DataFrame) -> dict:
    return _row(rule.name, rule.layer, rule.severity, rule.count_failures(df))


def enforce(report: pd.DataFrame, fresh: bool = False) -> pd.DataFrame:
    """Guarda o relatório, mostra os avisos e pára (DataValidationError) se houver erros.

    fresh=True começa um relatório novo (primeira camada); caso contrário acrescenta ao
    existente, substituindo as linhas das mesmas camadas.
    """
    if not fresh and REPORT_PATH.exists():
        old = pd.read_csv(REPORT_PATH)
        report = pd.concat([old[~old["camada"].isin(report["camada"])], report],
                           ignore_index=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    failed = report[report["resultado"] == "falha"]
    for _, r in failed[failed["severidade"] == AVISO].iterrows():
        print(f"AVISO [{r['camada']}] {r['regra']}: {r['linhas_a_falhar']} linha(s)")
    errors = failed[failed["severidade"] == ERRO]
    if len(errors):
        lines = [f"  - [{r['camada']}] {r['regra']}: {r['linhas_a_falhar']} linha(s) a falhar"
                 for _, r in errors.iterrows()]
        raise DataValidationError(
            f"Validação dos dados falhou: {len(errors)} regra(s) com severidade 'erro'.\n"
            + "\n".join(lines) + f"\nRelatório completo: {REPORT_PATH}")
    return report
