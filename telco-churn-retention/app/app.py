"""App de retenção de clientes (Gradio): risco de churn e decisão de desconto.

Dois separadores:
  • Cliente individual — formulário com as variáveis originais do modelo; devolve P(churn),
    nível de risco, prioridade de contacto, principais fatores e a recomendação de desconto.
  • Lote — upload de um CSV; devolve a lista ordenada por P(churn) × CLTV, para descarregar.

Modelo: final_model.joblib. No Hugging Face é descarregado do Model repo indicado na variável
de ambiente MODEL_REPO; localmente usa outputs/artifacts/final_model.joblib.
Correr localmente, a partir da raiz do projeto:  python app/app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
# O pacote telco_churn vem copiado para app/ no deploy; localmente usa-se o de src/
sys.path.insert(0, str(HERE if (HERE / "telco_churn").exists() else HERE.parent / "src"))

from telco_churn import modelling, scoring  # noqa: E402

BLUE, GREEN, BLACK = "#3777FF", "#0EE071", "#000000"
LOCAL_MODEL = HERE.parent / "outputs" / "artifacts" / "final_model.joblib"


def load_bundle() -> dict:
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        # Model repo público: não precisa de token. Se for privado, o deploy cria o secret
        # HF_TOKEN no Space e ele é usado aqui.
        path = hf_hub_download(repo_id=repo, filename="final_model.joblib",
                               token=os.environ.get("HF_TOKEN") or None)
        return joblib.load(path)
    for path in (HERE / "final_model.joblib", LOCAL_MODEL):   # ao lado da app, ou o do projeto
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("Sem modelo: definir MODEL_REPO ou correr scripts/run_all.py")


BUNDLE = load_bundle()
FORM = BUNDLE["form"]
COLUMNS = list(FORM)
RISK_LEVELS = BUNDLE["risk_levels"]
CAPACITY_CUTS = BUNDLE["capacity_cuts"]

# Ordem e grupos do formulário (as variáveis que não estão aqui vão para "Outros")
GROUPS = {
    "Perfil": ["Gender", "Age", "Married", "Number of Dependents", "Number of Referrals"],
    "Contrato e faturação": ["Tenure in Months", "Contract", "Offer", "Payment Method",
                             "Paperless Billing", "Monthly Charge", "Total Charges",
                             "Total Refunds", "Total Extra Data Charges",
                             "Total Long Distance Charges"],
    "Serviços": ["Phone Service", "Multiple Lines", "Avg Monthly Long Distance Charges",
                 "Internet Service", "Internet Type", "Avg Monthly GB Download",
                 "Online Security", "Online Backup", "Device Protection Plan",
                 "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music",
                 "Unlimited Data"],
    "Localização": ["Latitude", "Longitude", "Population"],
}
GROUPS = {g: [c for c in cols if c in FORM] for g, cols in GROUPS.items()}
GROUPS["Outros"] = [c for c in COLUMNS if not any(c in g for g in GROUPS.values())]
ORDER = [c for cols in GROUPS.values() for c in cols]   # ordem dos campos no formulário


def _component(col: str):
    s = FORM[col]
    if "options" in s:
        return gr.Dropdown(choices=s["options"], value=s["default"], label=col)
    return gr.Number(value=round(s["default"], 2), label=col, minimum=0 if s["min"] >= 0 else None,
                     precision=0 if s["integer"] else 2)


def _customer(values) -> pd.DataFrame:
    row = dict(zip(ORDER, values))
    for col in ORDER:
        if "options" not in FORM[col] and row[col] is None:
            row[col] = FORM[col]["default"]   # campo apagado: volta ao valor típico
    return pd.DataFrame([row])[COLUMNS]


def _eur(x: float) -> str:
    """Valor em euros à portuguesa: espaço nos milhares, sem casas decimais (ex.: 4 500 €)."""
    return f"{x:,.0f} €".replace(",", "\u00a0")


def _pct(x: float, decimals: int = 1) -> str:
    """Percentagem com vírgula decimal (ex.: 4,4%)."""
    return f"{x:.{decimals}%}".replace(".", ",")


def _metrics(m: dict) -> str:
    return ", ".join(f"{k} {m[c]:.3f}".replace(".", ",")
                     for k, c in (("AUC", "auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier")))


def _risk_text(level: str) -> str:
    lo, hi = _pct(RISK_LEVELS["médio"], 0), _pct(RISK_LEVELS["alto"], 0)
    return {"alto": f"**alto** (P ≥ {hi})", "médio": f"**médio** (P entre {lo} e {hi})",
            "baixo": f"**baixo** (P < {lo})"}[level]


def _priority_text(priority: str) -> str:
    return {"top 10%": "**top 10% da base** — dentro da capacidade, mesmo com só 10% de contactos",
            "top 20%": "**top 20% da base** — dentro da capacidade se a equipa contactar 20%",
            "fora da capacidade": "**fora do top 20%** — só com capacidade acima de 20% da base"
            }[priority]


def assess(cltv, discount, months, effectiveness, *values):
    """Separador 'Cliente individual': risco, fatores e decisão de desconto."""
    customer = _customer(values)
    p = float(scoring.predict(BUNDLE, customer)[0])
    level = modelling.risk_level([p], RISK_LEVELS)[0]
    priority = modelling.contact_priority([p], CAPACITY_CUTS)[0]
    lines = [f"## P(churn) = {_pct(p)}",
             f"Nível de risco: {_risk_text(level)}",
             f"Prioridade de contacto: {_priority_text(priority)}"]
    new_customer = bool(scoring.is_new_customer(customer["Tenure in Months"])[0])
    if new_customer:
        lines.append("⚠️ **Cliente novo (tenure ≤ 3 meses).** No treino, quase todos os clientes "
                     "com tão poucos meses saíram, por um efeito da forma como os estados foram "
                     "registados. Para estes clientes o modelo extrapola: usar a regra de "
                     "onboarding em vez deste valor.")

    d = scoring.discount_decision(p, float(cltv), float(discount), float(months),
                                  float(effectiveness))
    verdict = "✅ **Oferecer o desconto.**" if d["dar_desconto"] else "❌ **Não oferecer o desconto.**"
    if new_customer:
        verdict += " _(indicativo: cliente novo — seguir a regra de onboarding)_"
    decision = (
        f"{verdict}\n\n"
        f"- Ganho esperado = P(churn) × eficácia × CLTV = {_pct(p)} × {_pct(effectiveness, 0)} × "
        f"{_eur(cltv)} = **{_eur(d['ganho_esperado'])}**\n"
        f"- Custo = desconto × meses = {_eur(discount)} × {months:.0f} = **{_eur(d['custo'])}**\n"
        f"- Valor líquido esperado: **{'+' if d['valor_liquido'] >= 0 else '−'}"
        f"{_eur(abs(d['valor_liquido']))}**\n\n"
        + ("Em média, o desconto recupera mais valor do que custa." if d["dar_desconto"] else
           "O risco ou o valor do cliente não chegam para pagar o desconto: "
           "procure uma ação mais barata ou reduza o desconto.")
    )
    factors = scoring.local_factors(BUNDLE, customer).assign(
        efeito_em_P=lambda t: t["efeito_em_P"].map(lambda v: ("+" if v >= 0 else "−") + _pct(abs(v))))
    factors.columns = ["Variável", "Valor do cliente", "Valor típico", "Efeito em P(churn)"]
    return "\n\n".join(lines), decision, factors


def score_file(file):
    """Separador 'Lote': lista de prioridade a partir de um CSV."""
    if file is None:
        raise gr.Error("Escolha primeiro um ficheiro CSV.")
    try:
        customers = pd.read_csv(file, keep_default_na=False, na_values=[""])
        out = scoring.score_batch(BUNDLE, customers)
    except ValueError as e:
        raise gr.Error(str(e))
    path = Path(tempfile.mkdtemp()) / "lista_prioridade.csv"
    out.to_csv(path, index=False)
    shown = out.head(50).assign(p_churn=lambda t: t["p_churn"].round(3),
                                perda_esperada=lambda t: t["perda_esperada"].round(0))
    summary = (f"{len(out)} clientes pontuados · {int((out['nivel_risco'] == 'alto').sum())} "
               f"com risco alto (P ≥ {_pct(RISK_LEVELS['alto'], 0)}) · "
               f"{int((out['prioridade_contacto'] != 'fora da capacidade').sum())} no top 20% "
               f"de prioridade · {int(out['cliente_novo'].sum())} clientes novos (ver aviso).")
    return summary, shown, gr.DownloadButton(value=str(path), visible=True)


NOTE = (
    "**Pressuposto:** a eficácia do desconto não vem dos dados. O modelo prevê o **risco** de "
    "saída, não o efeito causal do desconto. Antes de generalizar, testar com um grupo de "
    "controlo e substituir a eficácia assumida pela observada."
)

THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
    button_primary_background_fill=BLUE, button_primary_background_fill_hover=GREEN,
    button_primary_text_color="#FFFFFF", body_background_fill_dark=BLACK,
    block_title_text_color=BLUE, slider_color=GREEN)
CSS = f"""
h1, h2 {{ color: {BLUE}; }}
.gradio-container {{ max-width: 1200px !important; }}
.nota {{ border-left: 4px solid {GREEN}; padding-left: 12px; }}
"""

with gr.Blocks(title="Retenção de clientes — Telco") as demo:
    gr.Markdown(
        f"# Retenção de clientes — risco de churn\n"
        f"Modelo **{BUNDLE['name']}** · holdout: {_metrics(BUNDLE['holdout'])} "
        f"(Churn Score IBM: {_metrics(BUNDLE['benchmark_holdout'])}). "
        "Dados sintéticos IBM (Telco, Califórnia).")
    with gr.Tab("Cliente individual"):
        inputs = []
        with gr.Row():
            with gr.Column(scale=3):
                for group, cols in GROUPS.items():
                    if not cols:
                        continue
                    with gr.Accordion(group, open=group in ("Perfil", "Contrato e faturação")):
                        with gr.Row():
                            inputs += [_component(c) for c in cols]
            with gr.Column(scale=2):
                gr.Markdown("### Decisão de desconto")
                cltv = gr.Number(value=round(BUNDLE["cltv_median"]), label="CLTV do cliente (€)",
                                 minimum=0, info="Por omissão, a mediana do treino")
                discount = gr.Number(value=10, label="Desconto mensal (€)", minimum=0)
                months = gr.Number(value=6, label="Duração do desconto (meses)", minimum=0,
                                   precision=0)
                effectiveness = gr.Slider(0, 1, value=0.3, step=0.05,
                                          label="Eficácia assumida do desconto",
                                          info="% dos clientes em risco que ficam por causa dele")
                run = gr.Button("Calcular", variant="primary")
                risk = gr.Markdown()
                decision = gr.Markdown()
                gr.Markdown(NOTE, elem_classes="nota")
        factors = gr.Dataframe(label="Principais fatores deste cliente (P(churn) se a variável "
                                     "tivesse o valor típico; mostra associação, não causa)",
                               interactive=False)
        run.click(assess, inputs=[cltv, discount, months, effectiveness, *inputs],
                  outputs=[risk, decision, factors])

    with gr.Tab("Lote"):
        gr.Markdown(
            "Carregue um CSV com uma linha por cliente e as colunas `Customer ID`, `CLTV` e as "
            f"{len(COLUMNS)} variáveis do formulário (por exemplo, `data/processed/test.csv`). "
            "Devolve a lista ordenada por **perda esperada = P(churn) × CLTV**, com o nível de "
            "risco (baixo < 30%, médio 30–60%, alto ≥ 60%), a prioridade de contacto (top 10% / "
            "top 20% da base) e o aviso de cliente novo.")
        upload = gr.File(label="CSV de clientes", file_types=[".csv"])
        go = gr.Button("Pontuar lote", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Primeiros 50 da lista", interactive=False)
        download = gr.DownloadButton("Descarregar lista completa (CSV)", visible=False)
        go.click(score_file, inputs=upload, outputs=[summary, table, download])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
