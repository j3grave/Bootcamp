"""App Gradio de retenção: risco de churn e decisão de desconto.

Cliente individual: P(churn), se está no top 20% de risco, fatores e decisão de desconto.
Lote: CSV com duas listas, clientes estabelecidos (por P(churn) × CLTV) e clientes novos
(segmento de onboarding; a P(churn) é só indicativa).
O modelo vem do Model repo em MODEL_REPO ou, localmente, de outputs/artifacts/.

Correr a partir da raiz do projeto: python app/app.py
"""
import os
import sys
import tempfile
from pathlib import Path

# O ZeroGPU só arranca com pelo menos uma função @spaces.GPU, mesmo sem usar a GPU.
# O pacote `spaces` só existe nos Spaces; localmente este bloco não faz nada.
try:
    import spaces   # tem de vir antes de pacotes que usem CUDA
except ImportError:
    spaces = None
if spaces is not None:
    # Tudo corre em CPU, sem gastar a quota de ZeroGPU dos visitantes.
    spaces.disable_gradio_auto_wrap()

    @spaces.GPU(duration=1)
    def _zerogpu_startup_check():
        """Só existe para o ZeroGPU arrancar; nunca é chamada pela app."""
        return None

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
# O pacote telco_churn vem copiado para app/ no deploy; localmente usa-se o de src/
sys.path.insert(0, str(HERE if (HERE / "telco_churn").exists() else HERE.parent / "src"))

from telco_churn import retention, scoring  # noqa: E402

BLUE, GREEN, BLACK = "#3777FF", "#0EE071", "#000000"
LOCAL_ARTIFACTS = HERE.parent / "outputs" / "artifacts"


def load_bundle(filename: str) -> dict:
    """Carrega o modelo do Hugging Face (MODEL_REPO) ou, localmente, de outputs/artifacts/."""
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        # Público dispensa token; se for privado, o deploy cria o secret HF_TOKEN no Space.
        path = hf_hub_download(repo_id=repo, filename=filename,
                               token=os.environ.get("HF_TOKEN") or None)
        return joblib.load(path)
    for path in (HERE / filename, LOCAL_ARTIFACTS / filename):   # ao lado da app, ou do projeto
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"Sem {filename}: definir MODEL_REPO ou correr scripts/run_pipeline.py")


MODEL = load_bundle("final_model.joblib")
FORM = MODEL["form"]
COLUMNS = list(FORM)
CAPACITY_CUT = MODEL["capacity_cut"]   # P(churn) do top 20% dos clientes estabelecidos

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
ORDER = [c for cols in GROUPS.values() for c in cols]


def _component(col: str):
    """Campo do formulário: lista de opções nas categóricas, número nas numéricas."""
    s = FORM[col]
    if "options" in s:
        return gr.Dropdown(choices=s["options"], value=s["default"], label=col)
    return gr.Number(value=round(s["default"], 2), label=col, minimum=0 if s["min"] >= 0 else None,
                     precision=0 if s["integer"] else 2)


def _customer(values) -> pd.DataFrame:
    """Uma linha com os valores do formulário; um campo apagado volta ao valor típico."""
    row = dict(zip(ORDER, values))
    for col in ORDER:
        if "options" not in FORM[col] and row[col] is None:
            row[col] = FORM[col]["default"]
    return pd.DataFrame([row])[COLUMNS]


def _eur(x: float) -> str:
    """Valor em euros à portuguesa: espaço nos milhares, sem casas decimais (ex.: 4 500 €)."""
    return f"{x:,.0f} €".replace(",", "\u00a0")


def _pct(x: float, decimals: int = 1) -> str:
    """Percentagem com vírgula decimal (ex.: 4,4%)."""
    return f"{x:.{decimals}%}".replace(".", ",")


def _metrics(m: dict) -> str:
    """AUC, PR-AUC e Brier num texto curto."""
    return ", ".join(f"{k} {m[c]:.3f}".replace(".", ",")
                     for k, c in (("AUC", "auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier")))


def _capacity_text(p: float) -> str:
    """Diz se o cliente está dentro da capacidade de contacto (top 20% de risco)."""
    if p >= CAPACITY_CUT:
        return f"**dentro do top 20% de risco** dos clientes estabelecidos (P ≥ {_pct(CAPACITY_CUT)})"
    return f"**fora do top 20% de risco** dos clientes estabelecidos (P < {_pct(CAPACITY_CUT)})"


def assess(cltv, discount, months, effectiveness, *values):
    """Separador 'Cliente individual': risco, fatores e decisão de desconto."""
    customer = _customer(values)
    new_customer = bool(scoring.is_new_customer(customer["Tenure in Months"])[0])
    p = float(scoring.predict(MODEL, customer)[0])
    if new_customer:
        segment = retention.onboarding_segment(customer).iloc[0]
        lines = [f"## P(churn) = {_pct(p)} _(indicativo)_",
                 "⚠️ **Cliente novo (tenure ≤ 3 meses).** No treino, quase todos os clientes "
                 "com tão poucos meses saíram, por um efeito da forma como os estados foram "
                 "registados. Para estes clientes o modelo extrapola: o valor acima não deve "
                 "ser usado para priorizar.",
                 f"Segmento de onboarding: **{segment}** (regra de onboarding: contrato e tipo "
                 "de internet). Os clientes novos não entram na prioridade de contacto."]
    else:
        lines = [f"## P(churn) = {_pct(p)}", f"Prioridade de contacto: {_capacity_text(p)}"]

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
    factors = scoring.local_factors(MODEL, customer).assign(
        efeito_em_P=lambda t: t["efeito_em_P"].map(lambda v: ("+" if v >= 0 else "−") + _pct(abs(v))))
    factors.columns = ["Variável", "Valor do cliente", "Valor típico", "Efeito em P(churn)"]
    return "\n\n".join(lines), decision, factors


def score_file(file):
    """Separador 'Lote': estabelecidos (por P × CLTV) e novos (onboarding), à parte."""
    if file is None:
        raise gr.Error("Escolha primeiro um ficheiro CSV.")
    try:
        customers = pd.read_csv(file, keep_default_na=False, na_values=[""])
        est, new = scoring.score_batch(MODEL, customers)
    except ValueError as e:
        raise gr.Error(str(e))
    folder = Path(tempfile.mkdtemp())
    est_path, new_path = folder / "lista_prioridade_estabelecidos.csv", folder / "clientes_novos.csv"
    est.to_csv(est_path, index=False)
    new.to_csv(new_path, index=False)
    shown_est = est.head(50).assign(p_churn=lambda t: t["p_churn"].astype(float).round(3),
                                    perda_esperada=lambda t: t["perda_esperada"].astype(float).round(0))
    shown_new = new.round({"p_churn_indicativo": 3})
    summary = (f"**{len(est)} clientes estabelecidos**, ordenados por P(churn) × CLTV · "
               f"{int(est['top_20_risco'].sum())} no top 20% de risco "
               f"(P ≥ {_pct(CAPACITY_CUT)}).  \n"
               f"**{len(new)} clientes novos** (tenure ≤ 3) numa lista à parte, por segmento de "
               "onboarding: o modelo extrapola nestes clientes.")
    return (summary, shown_est, shown_new, gr.DownloadButton(value=str(est_path), visible=True),
            gr.DownloadButton(value=str(new_path), visible=True))


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
        f"Modelo **{MODEL['name']}**. Holdout, clientes estabelecidos (tenure > 3): "
        f"{_metrics(MODEL['holdout_estabelecidos'])}; Churn Score IBM nos mesmos clientes: "
        f"{_metrics(MODEL['benchmark_holdout_estabelecidos'])}. "
        "Dados sintéticos IBM (Telco, Califórnia).\n\n"
        "ℹ️ Usamos este modelo e não o Churn Score da IBM porque dá probabilidades calibradas "
        "(Brier mais baixo), de que P(churn) × CLTV e a decisão de desconto precisam, e porque "
        "usa só variáveis conhecidas antes da saída.")
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
                cltv = gr.Number(value=round(MODEL["cltv_median"]), label="CLTV do cliente (€)",
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
            "Devolve duas listas:\n\n"
            "- **Clientes estabelecidos** (tenure > 3), ordenados por **perda esperada = "
            "P(churn) × CLTV** e marcados quando estão no top 20% de risco;\n"
            "- **Clientes novos** (tenure ≤ 3), à parte, por segmento de onboarding. A P(churn) "
            "aparece só como valor indicativo: nestes clientes o modelo extrapola.")
        upload = gr.File(label="CSV de clientes", file_types=[".csv"])
        go = gr.Button("Pontuar lote", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Clientes estabelecidos — primeiros 50 da lista de prioridade",
                             interactive=False)
        download = gr.DownloadButton("Descarregar lista de prioridade (CSV)", visible=False)
        new_table = gr.Dataframe(label="⚠️ Clientes novos (tenure ≤ 3) — regra de onboarding; "
                                       "P(churn) só indicativa (extrapolação)", interactive=False)
        download_new = gr.DownloadButton("Descarregar clientes novos (CSV)", visible=False)
        go.click(score_file, inputs=upload,
                 outputs=[summary, table, new_table, download, download_new])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
