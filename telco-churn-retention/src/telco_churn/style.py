"""Identidade visual do projeto: paleta, tema matplotlib/seaborn e tabelas."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from .config import FIG_DIR

# --- Paleta oficial -----------------------------------------------------------
BLUE = "#3777FF"    # Azul Digital — destaque principal
GREEN = "#0EE071"   # Verde Néon / Data Green — acento
BLACK = "#000000"   # Preto Profundo — fundo

# --- Tema: "light" (fundo branco) ou "dark" (fundo preto) ------------------------
THEME = "light"

WHITE = "#FFFFFF"
if THEME == "light":
    BG, INK, GREY, GRID = WHITE, BLACK, "#5F6368", "#E6E8EC"
else:
    BG, INK, GREY, GRID = BLACK, WHITE, "#9AA0A6", "#262626"

TARGET_COLORS = {0: BLUE, 1: GREEN}          # 0 = Stayed, 1 = Churned
TARGET_LABELS = {0: "Stayed", 1: "Churned"}

# Mapa divergente azul -> fundo -> verde (correlações)
CMAP_DIVERGING = LinearSegmentedColormap.from_list("blue_bg_green", [BLUE, BG, GREEN])
# Mapa sequencial fundo -> azul -> verde (intensidades)
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("bg_blue_green", [BG, BLUE, GREEN])

RC = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "legend.facecolor": BG, "legend.edgecolor": GRID, "legend.labelcolor": INK,
    "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 140,
    "axes.prop_cycle": cycler(color=[BLUE, GREEN, INK, GREY]),
    "font.family": "DejaVu Sans",
    "patch.edgecolor": BG, "patch.linewidth": 0.0,
}


def apply_theme() -> None:
    """Aplica o tema escuro com a paleta do projeto a todos os gráficos."""
    sns.set_theme(context="notebook", style="whitegrid" if THEME == "light" else "darkgrid", rc=RC)
    plt.rcParams.update(RC)


def save_fig(fig, name: str, show: bool = True) -> None:
    """Guarda a figura em outputs/figures/<name>.png e mostra-a."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def style_table(df: pd.DataFrame, precision: int = 3, highlight: str | None = None):
    """Devolve um Styler com a paleta do projeto (cabeçalho azul, fundo do tema).

    highlight: nome de uma coluna numérica para destacar com barras verdes.
    """
    styler = (
        df.style.format(precision=precision, na_rep="—")
        .set_table_styles([
            {"selector": "th", "props": [("background-color", BLUE), ("color", WHITE),
                                         ("font-weight", "bold"), ("border", f"1px solid {BG}")]},
            {"selector": "td", "props": [("background-color", BG), ("color", INK),
                                         ("border", f"1px solid {GRID}")]},
            {"selector": "caption", "props": [("color", BLUE), ("font-weight", "bold"),
                                              ("caption-side", "top")]},
        ])
    )
    if highlight is not None and highlight in df.columns:
        styler = styler.bar(subset=[highlight], color=GREEN, vmin=0)
    return styler
