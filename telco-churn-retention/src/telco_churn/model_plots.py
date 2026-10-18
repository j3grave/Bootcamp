"""Gráficos da modelação (notebook 02), com a paleta e o tema de style.py.

Estão separados de plots.py (gráficos da EDA) para que alterações a uma fase não apaguem,
por engano, os gráficos da outra. Todos guardam a figura em outputs/figures/ via save_fig.

Convenção de cores:
    azul  = modelos individuais        verde = ensembles e modelo final
    cinzento = linha de base           preto/tinta = benchmark IBM e linhas de referência
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from .modelling import BASELINE, ENSEMBLES, LOWER_IS_BETTER, METRICS
from .style import BG, BLUE, GREEN, GREY, GRID, INK, save_fig

LIGHT_BLUE = "#AFC8FF"
DARK_BLUE = "#0B2E80"
CMAP_BLUE = LinearSegmentedColormap.from_list("bg_blue", [BG, BLUE])

METRIC_TITLES = {
    "auc": "ROC AUC", "pr_auc": "PR AUC", "brier": "Brier", "log_loss": "Log loss",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision", "recall": "Recall", "f1": "F1",
}


def _color(model: str, best: str | None = None) -> str:
    if model == BASELINE:
        return GREY
    if model == best or model in ENSEMBLES or model.startswith(tuple(ENSEMBLES)):
        return GREEN
    return BLUE


# --- Comparação geral --------------------------------------------------------------------
def plot_cv_comparison(cv: pd.DataFrame, benchmark_auc: float | None = None,
                       name: str = "21_cv_modelos",
                       title: str = "AUC em validação cruzada (5-fold, treino)", show: bool = True):
    """AUC médio ± desvio-padrão por modelo. Ensembles a verde; benchmark a tracejado."""
    cv = cv.drop(index=BASELINE, errors="ignore").sort_values("auc")
    colors = [_color(i) for i in cv.index]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(cv) + 1.2))
    ax.barh(cv.index, cv["auc"], xerr=cv.get("auc_std"), color=colors, height=0.6,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    errs = cv["auc_std"] if "auc_std" in cv else pd.Series(0, index=cv.index)
    for i, (v, e) in enumerate(zip(cv["auc"], errs)):
        ax.text(v + e, i, f"  {v:.3f}", va="center", color=INK, fontsize=8)
    lo = cv["auc"].min()
    if benchmark_auc is not None:
        ax.axvline(benchmark_auc, color=INK, linestyle="--", linewidth=1)
        ax.text(benchmark_auc, len(cv) - 0.4, " IBM", color=INK, fontsize=8)
        lo = min(lo, benchmark_auc)
    ax.set_xlim(lo - 0.03, 1.0)
    ax.set_title(title)
    ax.set_xlabel("ROC AUC")
    save_fig(fig, name, show)


def plot_metrics_panel(view: pd.DataFrame, std: pd.DataFrame | None = None,
                       train: pd.DataFrame | None = None, best: str | None = None,
                       name: str = "22_painel_metricas_cv",
                       title: str = "Nove métricas por modelo — validação cruzada (5-fold)",
                       show: bool = True):
    """Um painel por métrica e uma barra por modelo.

    view:  valores a mostrar (modelos x métricas), por ex. médias em CV ou o holdout.
    std:   desvio-padrão entre folds (barra de erro), opcional.
    train: valores no treino, opcional; aparecem como um losango sobre a barra. Quando o
           losango fica muito à direita da barra (ou à esquerda, no log loss e no Brier),
           o modelo está a decorar o treino.
    """
    order = list(view.sort_values("auc", ascending=False).index)
    fig, axes = plt.subplots(3, 3, figsize=(15, 0.36 * len(order) * 3 + 2.5), sharey=True)
    for ax, m in zip(axes.ravel(), METRICS):
        vals = view.loc[order, m].astype(float)
        err = std.loc[order, m].astype(float) if std is not None else None
        ax.barh(order, vals, xerr=err, color=[_color(i, best) for i in order], height=0.62,
                error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        if train is not None:
            ax.scatter(train.loc[order, m].astype(float), order, marker="D", s=18,
                       facecolor=BG, edgecolor=INK, linewidth=1, zorder=3)
        top = max(vals.max(), train[m].max() if train is not None else 0)
        xmax = top * 1.28 if m in LOWER_IS_BETTER else 1.12
        for i, v in enumerate(vals):
            if v > 0.3 * xmax:   # valor dentro da barra, para não colidir com o losango
                ax.text(v - xmax * 0.015, i, f"{v:.3f}", va="center", ha="right",
                        color="#FFFFFF", fontsize=7, fontweight="bold")
            else:
                ax.text(v + xmax * 0.015, i, f"{v:.3f}", va="center", color=INK, fontsize=7)
        ax.set_xlim(0, xmax)
        better = "menor" if m in LOWER_IS_BETTER else "maior"
        ax.set_title(f"{METRIC_TITLES[m]}  ·  {better} é melhor", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    axes[0, 0].invert_yaxis()
    note = ("barra = média em CV (± desvio-padrão)  ·  losango = treino" if train is not None
            else "")
    fig.suptitle(title + (f"\n{note}" if note else ""), color=INK, fontweight="bold", y=1.0)
    save_fig(fig, name, show)


def plot_metrics_heatmap(view: pd.DataFrame, name: str = "22b_heatmap_metricas_cv",
                         title: str = "Métricas por modelo — validação cruzada",
                         show: bool = True):
    """Grelha modelo × métrica. A cor é a posição do modelo em cada métrica (azul = melhor),
    já invertida no log loss e no Brier; o número é o valor real."""
    view = view.sort_values("auc", ascending=False).astype(float)
    rank = view.rank(pct=True)
    rank[LOWER_IS_BETTER] = view[LOWER_IS_BETTER].rank(pct=True, ascending=False)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(view) + 1.6))
    ax.imshow(rank.values, cmap=CMAP_BLUE, aspect="auto", vmin=0, vmax=1.05)
    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            txt_color = "#FFFFFF" if rank.values[i, j] > 0.7 else INK
            ax.text(j, i, f"{view.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color=txt_color)
    ax.set_xticks(range(view.shape[1]), [METRIC_TITLES[m] for m in view.columns], fontsize=8)
    ax.set_yticks(range(view.shape[0]), view.index, fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.set_title(title + "\n(cor = posição em cada métrica; azul mais forte = melhor)", pad=28)
    save_fig(fig, name, show)


def plot_overfitting(table: pd.DataFrame, metric: str = "auc", name: str = "23_sobreajuste",
                     show: bool = True):
    """Valor no treino vs em CV para cada modelo. Quanto maior a distância, mais sobreajuste."""
    t = table.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.hlines(y, t[metric], t[f"{metric}_train"], color=GRID, linewidth=3, zorder=1)
    ax.scatter(t[f"{metric}_train"], y, s=55, facecolor=BG, edgecolor=BLUE, linewidth=1.5,
               label="treino", zorder=2)
    ax.scatter(t[metric], y, s=55, color=BLUE, label="validação cruzada", zorder=3)
    for i, (cv, tr) in enumerate(zip(t[metric], t[f"{metric}_train"])):
        ax.text(min(cv, tr) - 0.005, i, f"{cv:.3f} ", ha="right", va="center", fontsize=7,
                color=INK)
        ax.text(max(cv, tr) + 0.005, i, f"Δ {tr - cv:+.3f}", ha="left", va="center",
                fontsize=7, color=GREEN if tr - cv > 0.03 else GREY)
    ax.set_yticks(y, t.index, fontsize=8)
    ax.set_xlim(max(0.45, t[metric].min() - 0.08), 1.06)
    ax.set_xlabel(METRIC_TITLES[metric])
    ax.set_title(f"Sobreajuste: {METRIC_TITLES[metric]} no treino vs em validação cruzada")
    ax.legend(loc="lower left")
    save_fig(fig, name, show)


# --- Curvas e matrizes -------------------------------------------------------------------
def plot_roc_pr(y, probas: dict, best: str | None = None, name: str = "23b_roc_pr_cv",
                title: str = "Curvas ROC e Precision-Recall — previsões out-of-fold",
                show: bool = True):
    """ROC e PR de todos os modelos. Modelo final a verde e mais grosso; benchmark a preto."""
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    others = [m for m in probas if m not in (BASELINE, best) and "IBM" not in m]
    shades = LinearSegmentedColormap.from_list("ramp", [DARK_BLUE, BLUE, LIGHT_BLUE, "#7FD9A8"])
    palette = {m: shades(i / max(len(others) - 1, 1)) for i, m in enumerate(others)}
    for model, p in probas.items():
        if model == BASELINE:
            continue
        p = np.asarray(p, dtype=float)
        is_best, is_ibm = model == best, "IBM" in model
        color = GREEN if is_best else INK if is_ibm else palette[model]
        kw = dict(color=color, linewidth=2.6 if is_best else 1.2,
                  linestyle="--" if is_ibm else "-", zorder=3 if is_best else 2)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{model} ({roc_auc_score(y, p):.3f})", **kw)
        axes[1].plot(rec, prec, **kw)
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    axes[1].axhline(y.mean(), color=GREY, linestyle=":", linewidth=1)
    axes[1].text(0.01, y.mean() + 0.01, f"taxa de churn {y.mean():.1%}", color=GREY, fontsize=7)
    axes[0].set(title="ROC", xlabel="Falsos positivos (fiéis sinalizados)",
                ylabel="Verdadeiros positivos (churners apanhados)")
    axes[1].set(title="Precision-Recall", xlabel="Recall (churners apanhados)",
                ylabel="Precision (sinalizados que saíram)", ylim=(0, 1.02))
    axes[0].legend(loc="lower right", fontsize=7, title="ROC AUC", title_fontsize=7)
    fig.suptitle(title, color=INK, fontweight="bold")
    save_fig(fig, name, show)


def plot_confusion_matrices(y, probas: dict, threshold: float = 0.5, ncols: int = 3,
                            name: str = "23c_matrizes_confusao_cv",
                            title: str = "Matrizes de confusão — previsões out-of-fold",
                            show: bool = True):
    """Uma matriz por modelo: nº de clientes e % da linha (classe real)."""
    models = [m for m in probas if m != BASELINE]
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    labels = ["Stayed", "Churned"]
    for ax, model in zip(axes, models):
        cm = confusion_matrix(y, np.asarray(probas[model]) >= threshold)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=CMAP_BLUE, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}\n{share[r, c]:.0%}", ha="center", va="center",
                        fontsize=9, color="#FFFFFF" if share[r, c] > 0.6 else INK)
        ax.set_xticks([0, 1], labels, fontsize=8)
        ax.set_yticks([0, 1], labels, fontsize=8)
        ax.set_xlabel("previsto", fontsize=8)
        ax.set_ylabel("real", fontsize=8)
        ax.set_title(model, fontsize=9)
        ax.grid(False)
    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.suptitle(f"{title}, limiar {threshold:.2f}  (canto inferior esquerdo = churners não detetados)",
                 color=INK, fontweight="bold", y=1.0)
    save_fig(fig, name, show)


def plot_threshold(y, p, model: str, marker: float | None = None,
                   name: str = "23d_limiar_modelo_final", show: bool = True):
    """Precision, recall, F1 e % de clientes sinalizados em função do limiar de decisão."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0.05, 0.95, 91)
    prec = [precision_score(y, p >= t, zero_division=0) for t in grid]
    rec = [recall_score(y, p >= t) for t in grid]
    f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
    flagged = [(p >= t).mean() for t in grid]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(grid, prec, color=BLUE, linewidth=2, label="precision")
    ax.plot(grid, rec, color=GREEN, linewidth=2, label="recall")
    ax.plot(grid, f1, color=INK, linewidth=2, label="F1")
    ax.plot(grid, flagged, color=GREY, linewidth=1.5, linestyle=":", label="% clientes sinalizados")
    if marker is not None:
        ax.axvline(marker, color=INK, linestyle="--", linewidth=1)
        ax.text(marker, 1.01, f" limiar {marker:.2f}", color=INK, fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Limiar de decisão",
           title=f"{model}: métricas por limiar (out-of-fold)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    save_fig(fig, name, show)


def plot_diversity(corr: pd.DataFrame, name: str = "20_diversidade_modelos", show: bool = True):
    """Correlação entre as probabilidades dos modelos. Valores baixos = erros diferentes."""
    fig, ax = plt.subplots(figsize=(0.75 * len(corr) + 3, 0.6 * len(corr) + 2))
    vmin = max(0.0, float(np.floor((corr.values.min() - 0.05) * 20) / 20))
    ax.imshow(corr.values, cmap=CMAP_BLUE, vmin=vmin, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#FFFFFF" if (v - vmin) / (1 - vmin) > 0.6 else INK)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.grid(False)
    ax.set_title("Diversidade entre modelos: correlação das probabilidades out-of-fold")
    save_fig(fig, name, show)


def plot_mlp_training(mlp, name: str = "23e_rede_neuronal_treino", show: bool = True):
    """Curva de aprendizagem da rede neuronal: perda no treino e accuracy na validação interna.

    mlp: o MLPClassifier já treinado (ex.: pipeline.named_steps['clf']). Com early_stopping,
    o scikit-learn guarda a loss em cada época (loss_curve_) e a accuracy na validação interna
    (validation_scores_). O treino pára quando esta deixa de melhorar.
    """
    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    epochs = np.arange(1, len(mlp.loss_curve_) + 1)
    ax1.plot(epochs, mlp.loss_curve_, color=BLUE, linewidth=2, label="log loss (treino)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Log loss (treino)", color=BLUE)
    if getattr(mlp, "validation_scores_", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(epochs, mlp.validation_scores_, color=GREEN, linewidth=2,
                 label="accuracy (validação interna)")
        ax2.set_ylabel("Accuracy (validação interna)", color=GREEN)
        ax2.grid(False)
        best = int(np.argmax(mlp.validation_scores_)) + 1
        ax1.axvline(best, color=INK, linestyle="--", linewidth=1)
        ax1.text(best, ax1.get_ylim()[1], f" melhor época: {best}", va="top", fontsize=8)
    ax1.set_title("Rede neuronal (MLP): evolução do treino com early stopping")
    save_fig(fig, name, show)


# --- Holdout e priorização -----------------------------------------------------------------
def plot_roc_calibration(y: pd.Series, probas: dict, show: bool = True):
    """Curvas ROC (esq.) e de calibração (dir.) no holdout."""
    palette = [GREEN, INK, BLUE, GREY, LIGHT_BLUE, "#7FF0B5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (name, p), c in zip(probas.items(), palette):
        fpr, tpr, _ = roc_curve(y, p)
        style = "--" if "IBM" in name else "-"
        axes[0].plot(fpr, tpr, color=c, linewidth=1.6, linestyle=style,
                     label=f"{name} ({roc_auc_score(y, p):.3f})")
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1].plot(mean, frac, marker="o", markersize=3, color=c, linewidth=1.4,
                     linestyle=style, label=name)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set(title="Curva ROC — holdout", xlabel="Falsos positivos", ylabel="Verdadeiros positivos")
    axes[1].set(title="Calibração — holdout", xlabel="P(churn) prevista",
                ylabel="Taxa de churn observada")
    save_fig(fig, "24_roc_calibracao", show)


def plot_capture_curve(curve: pd.DataFrame, show: bool = True):
    """% do CLTV perdido (clientes que saíram) captado vs % de clientes contactados."""
    x = curve["% clientes contactados"]
    palette = [GREEN, BLUE, INK, GREY]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, c in zip(curve.columns.drop("% clientes contactados"), palette):
        ax.plot(x, curve[col], color=c, linewidth=1.8, label=col)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="Aleatório")
    for ax_fmt in (ax.xaxis, ax.yaxis):
        ax_fmt.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Curva de captura de valor — holdout")
    ax.set_xlabel("% de clientes contactados (por ordem de prioridade)")
    ax.set_ylabel("% do CLTV dos churners captado")
    ax.legend(loc="lower right", fontsize=7)
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
