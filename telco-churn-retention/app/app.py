"""App de retenção de clientes (Gradio): risco de churn e decisão de desconto.

Dois separadores:
  • Cliente individual — formulário com as variáveis originais do modelo; devolve P(churn),
    segmento de risco, principais fatores do cliente e a recomendação de desconto.
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
        return joblib.load(hf_hub_download(repo_id=repo, filename="final_model.joblib"))
    for path in (HERE / "final_model.joblib", LOCAL_MODEL):   # ao lado da app, ou o do projeto
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("Sem modelo: definir MODEL_REPO ou correr scripts/run_all.py")


BUNDLE = load_bundle()
FORM = BUNDLE["form"]
COLUMNS = list(FORM)
CUTS = BUNDLE["risk_cuts"]

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


def _segment_text(segment: str) -> str:
    return {"alto": "**alto** — está no top 10% de risco",
            "médio": "**médio** — está entre o top 10% e o top 20% de risco",
            "baixo": "**baixo** — fora do top 20% de risco"}[segment]


def assess(cltv, discount, months, effectiveness, *values):
    """Separador 'Cliente individual': risco, fatores e decisão de desconto."""
    customer = _customer(values)
    p = float(scoring.predict(BUNDLE, customer)[0])
    segment = modelling.risk_segment([p], CUTS)[0]
    lines = [f"## P(churn) = {p:.1%}", f"Segmento de risco: {_segment_text(segment)}"]
    if scoring.is_new_customer(customer["Tenure in Months"])[0]:
        lines.append("⚠️ **Cliente novo (tenure ≤ 3 meses).** No treino, quase todos os clientes "
                     "com tão poucos meses saíram, por um efeito da forma como os estados foram "
                     "registados. Para estes clientes o modelo extrapola: usar a regra de "
                     "onboarding em vez deste valor.")

    d = scoring.discount_decision(p, float(cltv), float(discount), float(months),
                                  float(effectiveness))
    verdict = "✅ **Oferecer o desconto.**" if d["dar_desconto"] else "❌ **Não oferecer o desconto.**"
    decision = (
        f"{verdict}\n\n"
        f"- Ganho esperado = P(churn) × eficácia × CLTV = {p:.1%} × {effectiveness:.0%} × "
        f"{cltv:,.0f} € = **{d['ganho_esperado']:,.0f} €**\n"
        f"- Custo = desconto × meses = {discount:,.0f} € × {months:.0f} = **{d['custo']:,.0f} €**\n"
        f"- Valor líquido esperado: **{d['valor_liquido']:+,.0f} €**\n\n"
        + ("Em média, o desconto recupera mais valor do que custa." if d["dar_desconto"] else
           "O risco ou o valor do cliente não chegam para pagar o desconto: "
           "procure uma ação mais barata ou reduza o desconto.")
    )
    factors = scoring.local_factors(BUNDLE, customer).assign(
        efeito_em_P=lambda t: t["efeito_em_P"].map(lambda v: f"{v:+.1%}"))
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
    summary = (f"{len(out):,} clientes pontuados · {int((out['segmento_risco'] == 'alto').sum())} "
               f"em risco alto · {int(out['cliente_novo'].sum())} clientes novos (ver aviso).")
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
        f"Modelo **{BUNDLE['name']}** · holdout: AUC {BUNDLE['holdout']['auc']:.3f}, "
        f"PR-AUC {BUNDLE['holdout']['pr_auc']:.3f}, Brier {BUNDLE['holdout']['brier']:.3f} "
        f"(Churn Score IBM: AUC {BUNDLE['benchmark_cv']['auc']:.3f} em CV). "
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
            "Devolve a lista ordenada por **perda esperada = P(churn) × CLTV**, com o segmento "
            "de risco e o aviso de cliente novo.")
        upload = gr.File(label="CSV de clientes", file_types=[".csv"])
        go = gr.Button("Pontuar lote", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Primeiros 50 da lista", interactive=False)
        download = gr.DownloadButton("Descarregar lista completa (CSV)", visible=False)
        go.click(score_file, inputs=upload, outputs=[summary, table, download])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
