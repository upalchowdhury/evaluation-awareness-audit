"""Publication figures (matplotlib; light-surface print style).

Conventions: fixed categorical color order (never cycled ad hoc), one hue
light->dark for magnitudes (AUC heatmaps), blue<->red with a neutral gray
midpoint for signed causal effects, thin marks, recessive grid, no dual axes.
Every figure is saved as PDF (for LaTeX) + PNG (for quick viewing), and the
numbers behind each figure are saved as CSV next to it (the "table view").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Fixed categorical order (validated palette; do not cycle arbitrary colors).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "ea_seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "ea_div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7c1f1f"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.dpi": 110,
    })


def save(fig, out_base: str | Path, data: pd.DataFrame | None = None) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    if data is not None:
        data.to_csv(str(out_base) + ".csv", index=False)


# --------------------------------------------------------------------------- #
def layer_curves(series: list[dict], ylabel: str, title: str,
                 hline: float | None = None, figsize=(4.6, 3.0)):
    """series: [{label, x, y, lo?, hi?}]; fixed color order; chance line optional."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    for i, s in enumerate(series):
        c = SERIES[i % len(SERIES)]
        ax.plot(s["x"], s["y"], color=c, label=s["label"])
        if "lo" in s and s.get("lo") is not None:
            ax.fill_between(s["x"], s["lo"], s["hi"], color=c, alpha=0.15, linewidth=0)
    if hline is not None:
        ax.axhline(hline, color=BASELINE, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if len(series) > 1:
        ax.legend(fontsize=8)
    return fig


def heatmap(mat: np.ndarray, xlabels, ylabels, title: str, cbar_label: str,
            kind: str = "seq", center: float | None = None,
            vmin=None, vmax=None, figsize=(5.6, 3.4), annotate: bool = False):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "div":
        m = np.nanmax(np.abs(mat - (center or 0.0))) or 1.0
        norm = TwoSlopeNorm(vcenter=center or 0.0,
                            vmin=(center or 0.0) - m, vmax=(center or 0.0) + m)
        im = ax.imshow(mat, aspect="auto", cmap=DIV_CMAP, norm=norm,
                       interpolation="nearest")
    else:
        im = ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
    ax.grid(False)
    step_x = max(1, len(xlabels) // 12)
    ax.set_xticks(range(0, len(xlabels), step_x),
                  [str(x) for x in xlabels[::step_x]], rotation=0)
    step_y = max(1, len(ylabels) // 16)
    ax.set_yticks(range(0, len(ylabels), step_y), [str(y) for y in ylabels[::step_y]])
    if annotate and mat.size <= 64:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)
    ax.set_title(title, loc="left")
    return fig


def dose_response(df: pd.DataFrame, x: str, y: str, group: str,
                  ylabel: str, title: str, lo: str | None = None, hi: str | None = None,
                  figsize=(4.6, 3.0)):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    for i, (g, sub) in enumerate(df.groupby(group, sort=False)):
        sub = sub.sort_values(x)
        c = SERIES[i % len(SERIES)]
        ax.plot(sub[x], sub[y], marker="o", markersize=4, color=c, label=str(g))
        if lo and hi and lo in sub:
            ax.fill_between(sub[x], sub[lo], sub[hi], color=c, alpha=0.15, linewidth=0)
    ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel(r"steering strength $\alpha$ (multiples of $\sigma$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8)
    return fig


def bars_ci(labels, means, los, his, xlabel: str, title: str,
            colors=None, figsize=(4.6, None)):
    apply_style()
    h = max(1.8, 0.32 * len(labels) + 0.9)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ypos = np.arange(len(labels))[::-1]
    colors = colors or [SERIES[0]] * len(labels)
    err_lo = np.array(means) - np.array(los)
    err_hi = np.array(his) - np.array(means)
    ax.barh(ypos, means, xerr=[err_lo, err_hi], color=colors, height=0.62,
            error_kw=dict(ecolor=MUTED, lw=1))
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return fig


def stage_summary(layers, curves: dict[str, np.ndarray], title: str,
                  figsize=(5.2, 3.2)):
    """The headline figure: normalized decodability vs causal-belief vs behavior."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, y) in enumerate(curves.items()):
        y = np.asarray(y, dtype=float)
        rng = np.nanmax(y) - np.nanmin(y)
        yn = (y - np.nanmin(y)) / (rng if rng > 1e-9 else 1.0)
        ax.plot(layers, yn, color=SERIES[i % len(SERIES)], label=label)
        if np.any(~np.isnan(yn)):
            peak = int(np.nanargmax(yn))
            ax.plot([layers[peak]], [yn[peak]], "o", color=SERIES[i % len(SERIES)],
                    markersize=5)
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized effect (min-max per curve)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8)
    return fig


def attribution_panel(attr_df: pd.DataFrame, verify_df: pd.DataFrame | None,
                      top_n: int, title: str, figsize=(7.6, 3.2)):
    apply_style()
    top = attr_df.head(top_n).iloc[::-1]
    ncols = 2 if verify_df is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    axes = np.atleast_1d(axes)
    labels = [f"L{r.layer}H{r.head}" if r.type == "head" else f"MLP{r.layer}"
              for r in top.itertuples()]
    colors = [SERIES[0] if t == "head" else SERIES[1] for t in top["type"]]
    axes[0].barh(range(len(top)), top["attr"], color=colors, height=0.62)
    axes[0].set_yticks(range(len(top)), labels, fontsize=7)
    axes[0].axvline(0, color=BASELINE, linewidth=0.8)
    axes[0].set_xlabel("attribution (belief metric)")
    axes[0].set_title("E1: attribution screening", loc="left")
    if verify_df is not None:
        v = verify_df[verify_df.node != "JOINT"].set_index("node")
        vv = [v.loc[l, "mean_delta"] if l in v.index else np.nan for l in labels]
        axes[1].barh(range(len(labels)), vv, color=colors, height=0.62)
        axes[1].set_yticks(range(len(labels)), [""] * len(labels))
        axes[1].axvline(0, color=BASELINE, linewidth=0.8)
        axes[1].set_xlabel(r"causal $\Delta$belief when patched")
        axes[1].set_title("E2: causal verification", loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig
