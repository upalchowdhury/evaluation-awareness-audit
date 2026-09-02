#!/usr/bin/env python
"""Figures for paper/main_v2.tex from results_v2/<model>/.

  python scripts/regen_paper_figs_v2.py --results results_v2 --model qwen15b
Writes paper/figures/pub2_*.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C_MAIN, C_CTRL, C_MUTED, C_ALT, C_NEUT = "#16697A", "#C2612C", "#8B95A1", "#4C956C", "#7B5EA7"


def fig_raw_belief(res: Path, out: Path) -> None:
    """Per-prompt stated belief under the two framings: the raw data."""
    v = pd.read_csv(res / "exp1" / "verbal.csv")
    v = v[v.split == "test"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    rng = np.random.default_rng(0)
    for i, (cls, color, label) in enumerate((("deploy", C_CTRL, "deployment framing"),
                                             ("eval", C_MAIN, "evaluation framing"))):
        y = v[v.cls == cls].belief.to_numpy()
        x = i + rng.uniform(-0.28, 0.28, size=len(y))
        ax.scatter(x, y, s=6, alpha=.45, color=color, edgecolors="none", label=label)
        ax.hlines(y.mean(), i - .35, i + .35, color=color, lw=2)
    ax.axhline(0, color=C_MUTED, lw=.7, ls=":")
    ax.set_xticks([0, 1], ["deployment", "evaluation"])
    ax.set_ylabel("stated belief (logits; + = 'a test')")
    ax.set_title("What the model says, per prompt")
    fig.tight_layout()
    fig.savefig(out / "pub2_raw_belief.pdf", bbox_inches="tight"); fig.savefig(out / "pub2_raw_belief.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_patch_layers(res: Path, out: Path) -> None:
    df = pd.read_csv(res / "exp2" / "agg_patch.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for direction, color, style, label in (
            ("deploy_to_eval", C_MAIN, "-", "matched source (deploy, same task)"),
            ("unpaired_control", C_CTRL, "--", "unrelated source (deploy, other task)")):
        sub = df[df.direction == direction].sort_values("layer")
        ax.plot(sub.layer, sub["mean"], style, color=color, lw=1.8, label=label)
        ax.fill_between(sub.layer, sub.lo, sub.hi, color=color, alpha=.15, lw=0)
    ax.axhline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ stated belief (logits)")
    ax.set_title("Patching the shared suffix")
    ax.legend(fontsize=6.7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pub2_patch_layers.pdf", bbox_inches="tight"); fig.savefig(out / "pub2_patch_layers.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_cells(res: Path, out: Path) -> None:
    s7 = json.loads((res / "exp7" / "summary.json").read_text())
    dl = s7["deep_layer"]
    ag = pd.read_csv(res / "exp7" / "span_patch_agg.csv")
    deep = ag[ag.layer == dl]
    order = [("readout", "paired", "deploy / same task"),
             ("readout", "unpaired", "deploy / other task"),
             ("readout", "sameclass", "eval / same task"),
             ("readout", "unpaired_eval", "eval / other task"),
             ("task", "paired", "task tokens only, deploy / same"),
             ("task", "unpaired", "task tokens only, deploy / other")]
    labels, means, los, his, colors = [], [], [], [], []
    for region, src, lab in order:
        r = deep[(deep.region == region) & (deep.source == src)]
        if not len(r):
            continue
        labels.append(lab)
        means.append(r["mean"].iloc[0])
        los.append(r["mean"].iloc[0] - r.lo.iloc[0])
        his.append(r.hi.iloc[0] - r["mean"].iloc[0])
        colors.append(C_MUTED if "task tokens" in lab else (C_CTRL if lab.startswith("deploy") else C_MAIN))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, means, xerr=[los, his], color=colors, height=.62, error_kw=dict(lw=.9, capsize=2))
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.axvline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel(r"$\Delta$ stated belief (logits)")
    ax.set_title(f"Source cells at layer {dl}")
    fig.tight_layout()
    fig.savefig(out / "pub2_cells.pdf", bbox_inches="tight"); fig.savefig(out / "pub2_cells.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_dose(res: Path, out: Path) -> None:
    df = pd.read_csv(res / "exp2" / "dose_belief.csv")
    st = df[df.layer_tag == "steer"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for cond, framing, color, label in (
            ("d", "eval", C_MAIN, "evaluation direction, eval-framed prompts"),
            ("d", "deploy", C_ALT, "evaluation direction, deploy-framed prompts"),
            ("random", "eval", C_MUTED, "mean of 3 random directions")):
        sub = st[(st.condition == cond) & (st.framing == framing)].groupby("alpha").belief.mean()
        if not len(sub):
            continue
        ax.plot(sub.index, sub.values, "-o", color=color, lw=1.6, ms=2.6, label=label)
    ax.axvline(0, color=C_MUTED, lw=.6, ls=":")
    ax.set_xlabel(r"steering dose $\alpha$ (multiples of $\sigma$)")
    ax.set_ylabel("stated belief (logits)")
    L = int(st.layer.iloc[0])
    ax.set_title(f"Dose-response at the steering-peak layer ({L})")
    ax.legend(fontsize=6.4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pub2_dose.pdf", bbox_inches="tight"); fig.savefig(out / "pub2_dose.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_register(res: Path, out: Path) -> None:
    s8 = json.loads((res / "exp8" / "summary.json").read_text())
    sw = s8["swings"]
    named = [("d", "evaluation direction $d$"),
             ("style", "in-corpus style"),
             ("style_neutral", "neutral formality"),
             ("d_perp_style", "$d$ minus in-corpus style"),
             ("d_perp_style_neutral", "$d$ minus neutral formality")]
    rand = sorted((abs(v["mean"]) for k, v in sw.items() if k.startswith("random_")), reverse=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = [lab for k, lab in named if k in sw]
    means = [sw[k]["mean"] for k, _ in named if k in sw]
    los = [sw[k]["mean"] - sw[k]["lo"] for k, _ in named if k in sw]
    his = [sw[k]["hi"] - sw[k]["mean"] for k, _ in named if k in sw]
    colors = [C_MAIN, C_CTRL, C_NEUT, C_MAIN, C_MAIN][:len(labels)]
    y = list(range(len(labels)))[::-1]
    ax.barh(y, means, xerr=[los, his], color=colors, height=.6, error_kw=dict(lw=.9, capsize=2))
    ax.set_yticks(y, labels, fontsize=7.5)
    if rand:
        hi = float(np.percentile(rand, 95))
        ax.axvspan(-hi, hi, color=C_MUTED, alpha=.18, lw=0,
                   label=f"5-95% of |swing|, {len(rand)} random directions")
        ax.legend(fontsize=6.4, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.28), ncol=1)
    ax.axvline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel(r"belief swing across $\pm4\sigma$ (logits)")
    ax.set_title(f"Matched-norm steering at layer {s8['layer']}")
    fig.tight_layout()
    fig.savefig(out / "pub2_register.pdf", bbox_inches="tight"); fig.savefig(out / "pub2_register.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen15b")
    ap.add_argument("--results", default="results_v2")
    a = ap.parse_args()
    res = ROOT / a.results / a.model
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_raw_belief, fig_patch_layers, fig_cells, fig_dose, fig_register):
        try:
            fn(res, out)
            print(f"  {fn.__name__} ok")
        except Exception as exc:
            print(f"  {fn.__name__} FAILED: {exc}")


if __name__ == "__main__":
    main()
