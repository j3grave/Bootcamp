"""Relatórios HTML de exploração com ydata-profiling, na identidade visual do projeto.

Três relatórios:
  1. profile_dataset      -> perfil de uma tabela (ex.: base unida com todas as colunas)
  2. profile_by_target    -> comparação Stayed vs Churned lado a lado (azul vs verde)
  3. brand_html           -> aplica a paleta (navbar preta, títulos azuis, acentos verdes)

Requer: pip install ydata-profiling "setuptools<81"
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import OUTPUTS
from .style import BLACK, BLUE, GREEN

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ydata_profiling")

PROFILE_DIR = OUTPUTS / "profiling"
WHITE = "#FFFFFF"

# --- Colormaps com nome, porque o ydata-profiling recebe o cmap como texto ----------
CMAP_DIV = "telco_blue_white_green"      # correlações (-1 azul ... +1 verde)
CMAP_MISSING = "telco_white_blue"        # matriz/heatmap de missing
for _name, _colors in ((CMAP_DIV, [BLUE, WHITE, GREEN]), (CMAP_MISSING, [WHITE, BLUE])):
    if _name not in mpl.colormaps:
        mpl.colormaps.register(LinearSegmentedColormap.from_list(_name, _colors))

# Cores para barras de frequência das categóricas (até 10 categorias)
CAT_COLORS = [BLUE, GREEN, "#7FA6FF", "#6FEDAA", "#1F4FBF", "#08A653",
              "#B3CAFF", "#A8F5CB", "#0B2E80", "#05703A"]

# Descrições curtas que aparecem em cada variável do relatório
DESCRIPTIONS = {
    "Customer ID": "Identificador único do cliente (chave).",
    "Gender": "Género do cliente.",
    "Age": "Idade em anos no fim do trimestre.",
    "Married": "Se o cliente é casado.",
    "Number of Dependents": "Nº de dependentes que vivem com o cliente.",
    "Latitude": "Latitude da residência principal.",
    "Longitude": "Longitude da residência principal.",
    "Population": "População estimada da área do ZIP code.",
    "City": "Cidade de residência (1.106 categorias; excluída do modelo base).",
    "Number of Referrals": "Nº de referências feitas pelo cliente.",
    "Tenure in Months": "Meses de relação com a empresa.",
    "Offer": "Última oferta aceite ('None' = sem oferta, valor real).",
    "Phone Service": "Subscreve telefone fixo.",
    "Avg Monthly Long Distance Charges": "Média mensal de encargos de longa distância.",
    "Multiple Lines": "Subscreve várias linhas.",
    "Internet Service": "Subscreve internet.",
    "Internet Type": "Tecnologia de internet ('None' = sem internet).",
    "Avg Monthly GB Download": "Média mensal de GB descarregados.",
    "Online Security": "Serviço adicional de segurança online.",
    "Online Backup": "Serviço adicional de backup online.",
    "Device Protection Plan": "Plano de proteção de equipamento.",
    "Premium Tech Support": "Suporte técnico premium.",
    "Streaming TV": "Streaming de TV de terceiros.",
    "Streaming Movies": "Streaming de filmes de terceiros.",
    "Streaming Music": "Streaming de música de terceiros.",
    "Unlimited Data": "Dados ilimitados.",
    "Contract": "Tipo de contrato.",
    "Paperless Billing": "Fatura eletrónica.",
    "Payment Method": "Método de pagamento.",
    "Monthly Charge": "Fatura mensal atual.",
    "Total Charges": "Encargos acumulados.",
    "Total Refunds": "Reembolsos acumulados.",
    "Total Extra Data Charges": "Encargos acumulados por dados extra.",
    "Total Long Distance Charges": "Encargos acumulados de longa distância.",
    "Satisfaction Score": "Satisfação 1–5. Suspeita de leakage.",
    "Churn Score": "Score IBM 0–100. Benchmark, não é feature.",
    "CLTV": "Valor de vida do cliente. Usado na priorização.",
    "n_add_on_services": "Feature criada: nº de serviços adicionais (0–8).",
    "has_refund": "Feature criada: teve algum reembolso.",
    "avg_hist_charge": "Feature criada: Total Charges ÷ Tenure.",
    "charge_delta": "Feature criada: Monthly Charge − fatura média histórica.",
    "churn": "Target: 1 = Churned, 0 = Stayed.",
}

# CSS injetado no HTML final (Bootstrap 5 usado pelo ydata-profiling)
BRAND_CSS = f"""
<style id="telco-brand">
:root {{ --bs-primary:{BLUE}; --bs-primary-rgb:55,119,255; --bs-link-color:{BLUE};
         --bs-link-color-rgb:55,119,255; --bs-link-hover-color:#1F4FBF; }}
.navbar {{ background:{BLACK} !important; border-bottom:3px solid {GREEN}; }}
.navbar .navbar-brand, .navbar .nav-link, .navbar .dropdown-toggle {{ color:{WHITE} !important; }}
.navbar .nav-link:hover, .navbar .nav-link:focus {{ color:{GREEN} !important; }}
.navbar-toggler {{ border-color:{GREEN}; }}
h1.section-name, .section-name {{ color:{BLUE}; border-bottom:2px solid {GREEN}; padding-bottom:6px; }}
.variable .variable-name, .variable .anchor {{ color:{BLUE} !important; }}
.nav-tabs .nav-link.active, .nav-pills .nav-link.active {{ background:{BLUE} !important; color:{WHITE} !important;
         border-color:{BLUE} !important; }}
.nav-tabs .nav-link, .nav-pills .nav-link {{ color:{BLUE}; }}
.badge.text-bg-info, .badge.bg-info {{ background:{GREEN} !important; color:{BLACK} !important; }}
.badge.text-bg-primary, .badge.bg-primary {{ background:{BLUE} !important; color:{WHITE} !important; }}
.btn-primary, .btn-info {{ background:{BLUE} !important; border-color:{BLUE} !important; color:{WHITE} !important; }}
.btn-light:hover {{ border-color:{GREEN}; }}
.dropdown-item:active, .dropdown-item.active {{ background:{BLUE}; }}
.table thead th {{ border-bottom:2px solid {BLUE}; }}
a {{ color:{BLUE}; }} a:hover {{ color:#1F4FBF; }}
footer, .footer {{ border-top:3px solid {GREEN}; }}
</style>
"""


def _config(title: str, dataset_description: str, minimal: bool, columns) -> dict:
    """Configuração comum: paleta, cmaps, descrições e opções de desempenho."""
    return dict(
        title=title,
        minimal=minimal,
        progress_bar=True,
        explorative=not minimal,
        dataset={"description": dataset_description,
                 "creator": "EDSB 2026 · Projeto D · Grupo EDSB26_G1",
                 "copyright_holder": "NOVA IMS"},
        variables={"descriptions": {c: d for c, d in DESCRIPTIONS.items() if c in columns}},
        show_variable_description=True,
        html={"style": {"primary_colors": [BLUE, GREEN, BLACK], "full_width": True},
              "navbar_show": True},
        plot={"correlation": {"cmap": CMAP_DIV, "bad": "#9AA0A6"},
              "missing": {"cmap": CMAP_MISSING},
              "cat_freq": {"show": True, "type": "bar", "max_unique": 10, "colors": CAT_COLORS},
              "dpi": 200},
        # As interações (scatter de todos os pares) são pesadas: limitamos ao essencial
        interactions={"continuous": True,
                      "targets": [c for c in ("Tenure in Months", "Monthly Charge", "churn") if c in columns]},
        correlations={"auto": {"calculate": True}, "pearson": {"calculate": False},
                      "spearman": {"calculate": True}, "kendall": {"calculate": False},
                      "phi_k": {"calculate": False}, "cramers": {"calculate": True}},
        samples={"head": 5, "tail": 5, "random": 0},
    )


def brand_html(path: Path) -> Path:
    """Injeta o CSS da paleta no HTML gerado.

    O ydata-profiling minifica o HTML e omite </head>, por isso o CSS é inserido
    antes de <body> (ou no início do ficheiro, como último recurso).
    """
    html = path.read_text(encoding="utf-8")
    if 'id="telco-brand"' in html:
        return path
    if "</head>" in html:
        html = html.replace("</head>", BRAND_CSS + "</head>", 1)
    elif "<body" in html:
        i = html.index("<body")
        html = html[:i] + BRAND_CSS + html[i:]
    else:
        html = BRAND_CSS + html
    path.write_text(html, encoding="utf-8")
    return path


def profile_dataset(df: pd.DataFrame, title: str, file_name: str, description: str = "",
                    minimal: bool = False):
    """Gera um relatório ydata-profiling de um DataFrame e grava-o em outputs/profiling/."""
    from ydata_profiling import ProfileReport

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    report = ProfileReport(df, **_config(title, description, minimal, df.columns))
    out = PROFILE_DIR / f"{file_name}.html"
    report.to_file(out)
    brand_html(out)
    return report, out


def profile_by_target(df: pd.DataFrame, target: str, columns: list[str], file_name: str = "03_stayed_vs_churned",
                      minimal: bool = False):
    """Compara Stayed (azul) vs Churned (verde) no mesmo relatório, variável a variável."""
    from ydata_profiling import ProfileReport, compare

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in columns if c != target]
    desc = "Comparação entre clientes que ficaram (Stayed) e que saíram (Churned)."
    stayed = ProfileReport(df.loc[df[target] == 0, cols], **_config("Stayed", desc, minimal, cols))
    churned = ProfileReport(df.loc[df[target] == 1, cols], **_config("Churned", desc, minimal, cols))
    report = compare([stayed, churned])
    report.config.title = "Telco Churn — Stayed vs Churned"
    report.config.html.style.primary_colors = [BLUE, GREEN]
    out = PROFILE_DIR / f"{file_name}.html"
    report.to_file(out)
    brand_html(out)
    return report, out
