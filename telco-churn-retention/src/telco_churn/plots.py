"""Gráficos da EDA com a paleta do projeto (fundo preto, azul e verde)."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .style import (BLACK, BLUE, CMAP_DIVERGING, GREEN, GREY, TARGET_COLORS, TARGET_LABELS,
                    WHITE, save_fig)


def _slug(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def _order_categories(s: pd.Series) -> list:
    """Ordem natural para Yes/No e contratos; restantes por frequência."""
    cats = list(s.value_counts().index)
    if set(cats) == {"Yes", "No"}:
        return ["Yes", "No"]
    contract = ["Month-to-Month", "One Year", "Two Year"]
    if set(cats) == set(contract):
        return contract
    return cats


# --- Target ----------------------------------------------------------------------
def plot_target(y: pd.Series, show: bool = True):
    counts = y.map(TARGET_LABELS).value_counts().reindex(["Stayed", "Churned"])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(counts.index, counts.values, color=[BLUE, GREEN], width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v / counts.sum():.1%}",
                ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Distribuição do target (população de modelação)")
    ax.set_ylabel("Nº clientes")
    save_fig(fig, "06_target", show)


# --- Numéricas ---------------------------------------------------------------------
def plot_numeric(df: pd.DataFrame, col: str, target: str, show: bool = True, prefix="08"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for t in (0, 1):
        sns.histplot(df.loc[df[target] == t, col], bins=30, stat="density", element="step",
                     fill=True, alpha=0.45, linewidth=1.4, color=TARGET_COLORS[t],
                     label=TARGET_LABELS[t], ax=axes[0])
    axes[0].set_title(f"{col} — distribuição por target")
    axes[0].legend()
    sns.boxplot(data=df, x=target, y=col, hue=target, palette=TARGET_COLORS, legend=False,
                ax=axes[1], linecolor=WHITE, flierprops={"markerfacecolor": GREY,
                                                         "markeredgecolor": GREY, "markersize": 2})
    axes[1].set_xticks([0, 1], ["Stayed", "Churned"])
    axes[1].set_xlabel("")
    axes[1].set_title("Boxplot")
    save_fig(fig, f"{prefix}_num_{_slug(col)}", show)


def plot_correlation(corr: pd.DataFrame, show: bool = True):
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=CMAP_DIVERGING, center=0, vmin=-1, vmax=1, annot=True,
                fmt=".2f", annot_kws={"size": 7, "color": WHITE}, linewidths=0.5,
                linecolor="#111111", cbar_kws={"shrink": 0.7}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlação de Spearman — numéricas + target")
    save_fig(fig, "08_correlacao_spearman", show)


# --- Categóricas -------------------------------------------------------------------
def plot_categorical_overview(df: pd.DataFrame, cols: list[str], ncols: int = 4,
                              show: bool = True):
    """Painel com a composição de TODAS as categóricas: o que contém cada variável."""
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        order = _order_categories(df[col])
        vc = df[col].value_counts().reindex(order)
        pct = vc / vc.sum()
        ax.barh([str(i) for i in vc.index], pct.values, color=BLUE, height=0.6)
        ax.invert_yaxis()
        for i, (p, n) in enumerate(zip(pct.values, vc.values)):
            ax.text(p + 0.01, i, f"{p:.0%}  (n={n:,})", va="center", color=WHITE, fontsize=7)
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(col, color=GREEN)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Composição das variáveis categóricas — categorias e peso de cada uma",
                 color=WHITE, fontweight="bold", y=1.0)
    save_fig(fig, "09_cat_overview", show)


def plot_categorical_detail(df: pd.DataFrame, col: str, target: str, overall_rate: float,
                            v: float, show: bool = True):
    """Para uma variável: composição (n, %) + proporção Stayed/Churned por categoria."""
    order = _order_categories(df[col])
    vc = df[col].value_counts().reindex(order)
    share = (pd.crosstab(df[col], df[target], normalize="index")
             .reindex(order)[[0, 1]])
    labels = [str(i) for i in order]
    h = max(2.6, 0.45 * len(order) + 1.3)

    fig, axes = plt.subplots(1, 2, figsize=(11, h), gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) composição
    axes[0].barh(labels, vc.values, color=BLUE, height=0.6)
    axes[0].invert_yaxis()
    for i, n in enumerate(vc.values):
        axes[0].text(n, i, f"  {n:,} ({n / vc.sum():.1%})", va="center", color=WHITE, fontsize=8)
    axes[0].set_xlim(0, vc.max() * 1.45)
    axes[0].set_title(f"{col} — o que contém (nº e % de clientes)")
    axes[0].set_xlabel("Nº clientes")

    # (b) proporção Stayed vs Churned (barra 100% empilhada)
    axes[1].barh(labels, share[0], color=BLUE, height=0.6, label="Stayed")
    axes[1].barh(labels, share[1], left=share[0], color=GREEN, height=0.6, label="Churned")
    axes[1].invert_yaxis()
    for i, r in enumerate(share[1].values):
        if r >= 0.18:   # rótulo dentro do segmento verde
            axes[1].text(0.985, i, f"churn {r:.1%}", va="center", ha="right", color=BLACK,
                         fontsize=8, fontweight="bold")
        else:           # segmento estreito: rótulo no azul, junto à fronteira
            axes[1].text(1 - r - 0.012, i, f"churn {r:.1%}", va="center", ha="right",
                         color=WHITE, fontsize=8, fontweight="bold")
    axes[1].axvline(1 - overall_rate, ls="--", lw=1.2, color=WHITE,
                    label=f"churn médio {overall_rate:.1%}")
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_yticklabels([])
    axes[1].set_title(f"Stayed vs Churned por categoria   |   Cramér's V = {v:.2f}")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 if len(order) < 4 else -0.1),
                   ncol=3, frameon=False)
    save_fig(fig, f"09_cat_{_slug(col)}", show)


# --- Rankings e verificações -------------------------------------------------------
def plot_ranking(s: pd.Series, title: str, name: str, highlight: set | None = None,
                 xlabel: str = "", show: bool = True):
    """Barras horizontais ordenadas. Itens em `highlight` a verde, restantes a azul."""
    highlight = highlight or set()
    colors = [GREEN if i in highlight else BLUE for i in s.index]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=colors, height=0.65)
    ax.invert_yaxis()
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.3f}", va="center", color=WHITE, fontsize=7)
    ax.set_xlim(0, s.max() * 1.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save_fig(fig, name, show)


def plot_leakage_checks(df: pd.DataFrame, target: str, show: bool = True):
    sat = df.groupby("Satisfaction Score")[target].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    bars = axes[0].bar(sat.index.astype(str), sat.values,
                       color=[GREEN if v == 1 else BLUE for v in sat.values])
    for b, v in zip(bars, sat.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom",
                     color=WHITE, fontsize=8)
    axes[0].set_ylim(0, 1.15)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("Taxa de churn por Satisfaction Score")
    axes[0].set_xlabel("Satisfaction Score")
    for t in (0, 1):
        sns.histplot(df.loc[df[target] == t, "Churn Score"], bins=40, stat="density",
                     element="step", fill=True, alpha=0.45, linewidth=1.4, color=TARGET_COLORS[t],
                     label=TARGET_LABELS[t], ax=axes[1])
    axes[1].legend()
    axes[1].set_title("Churn Score IBM (benchmark) por target")
    save_fig(fig, "10_leakage_checks", show)


# --- Modelação (notebook 02) ---------------------------------------------------------
def plot_cv_comparison(cv: pd.DataFrame, benchmark_auc: float | None = None, name: str = "21_cv_modelos",
                       title: str = "AUC em validação cruzada (5-fold, treino)", show: bool = True):
    """AUC médio ± desvio-padrão por modelo. Ensembles a verde; benchmark a tracejado."""
    cv = cv.sort_values("auc")
    colors = [GREEN if i in ("Soft Voting", "Stacking") else BLUE for i in cv.index]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(cv) + 1.2))
    ax.barh(cv.index, cv["auc"], xerr=cv.get("auc_std"), color=colors, height=0.6,
            error_kw={"ecolor": WHITE, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(cv["auc"]):
        ax.text(v, i, f"  {v:.3f}", va="center", color=WHITE, fontsize=8)
    lo = cv["auc"].min()
    if benchmark_auc is not None:
        ax.axvline(benchmark_auc, color=GREY, linestyle="--", linewidth=1)
        ax.text(benchmark_auc, len(cv) - 0.4, " IBM", color=GREY, fontsize=8)
        lo = min(lo, benchmark_auc)
    ax.set_xlim(lo - 0.03, 1.0)
    ax.set_title(title)
    ax.set_xlabel("ROC AUC")
    save_fig(fig, name, show)


def plot_roc_calibration(y: pd.Series, probas: dict, show: bool = True):
    """Curvas ROC (esq.) e de calibração (dir.) no holdout."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, roc_curve

    palette = [GREEN, BLUE, WHITE, GREY, "#7FA7FF", "#7FF0B5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (name, p), c in zip(probas.items(), palette):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=c, linewidth=1.6, label=f"{name} ({roc_auc_score(y, p):.3f})")
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1].plot(mean, frac, marker="o", markersize=3, color=c, linewidth=1.4, label=name)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
        ax.legend(loc="lower right")
    axes[0].set(title="Curva ROC — holdout", xlabel="Falsos positivos", ylabel="Verdadeiros positivos")
    axes[1].set(title="Calibração — holdout", xlabel="P(churn) prevista", ylabel="Taxa de churn observada")
    save_fig(fig, "24_roc_calibracao", show)


def plot_capture_curve(curve: pd.DataFrame, show: bool = True):
    """% do CLTV perdido (clientes que saíram) captado vs % de clientes contactados."""
    x = curve["% clientes contactados"]
    palette = [GREEN, BLUE, WHITE, GREY]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, c in zip(curve.columns.drop("% clientes contactados"), palette):
        ax.plot(x, curve[col], color=c, linewidth=1.8, label=col)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="Aleatório")
    for ax_fmt in (ax.xaxis, ax.yaxis):
        ax_fmt.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Curva de captura de valor — holdout")
    ax.set_xlabel("% de clientes contactados (por ordem de prioridade)")
    ax.set_ylabel("% do CLTV dos churners captado")
    ax.legend(loc="lower right")
    save_fig(fig, "25_captura_valor", show)


def plot_priority_map(prio: pd.DataFrame, top_n: int, name: str = "26_mapa_prioridade",
                      show: bool = True):
    """P(churn) vs CLTV; os top-N por perda esperada a verde."""
    top = prio["rank"] <= top_n
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(prio.loc[~top, "p_churn"], prio.loc[~top, "CLTV"], s=6, color=BLUE, alpha=0.45,
               label="Restantes")
    ax.scatter(prio.loc[top, "p_churn"], prio.loc[top, "CLTV"], s=10, color=GREEN, alpha=0.9,
               label=f"Top {top_n} — P(churn) × CLTV")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Mapa de prioridade: risco × valor")
    ax.set_xlabel("P(churn)")
    ax.set_ylabel("CLTV")
    ax.legend(loc="upper left")
    save_fig(fig, name, show)
