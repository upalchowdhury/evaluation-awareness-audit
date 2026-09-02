#!/usr/bin/env python
"""Regenerate the paper's figures from stored result CSVs with publication
titles (no internal experiment codenames like "A:", "F1", "G2").

  python scripts/regen_paper_figs.py --model qwen15b
Writes paper/figures/pub_*.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})
C_MAIN, C_CTRL, C_MUTED, C_ALT = "#16697A", "#C2612C", "#8B95A1", "#4C956C"


def fig_patch_layers(res: Path, out: Path) -> None:
    df = pd.read_csv(res / "exp2" / "agg_patch.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for direction, color, style, label in (
            ("deploy_to_eval", C_MAIN, "-", "matched source (deploy, same content)"),
            ("unpaired_control", C_CTRL, "--", "unpaired source (deploy, other content)")):
        sub = df[df.direction == direction].sort_values("layer")
        ax.plot(sub.layer, sub["mean"], style, color=color, lw=1.8, label=label)
        ax.fill_between(sub.layer, sub.lo, sub.hi, color=color, alpha=.15, lw=0)
    ax.axhline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ verbalized belief (logits)")
    ax.set_title("Full-suffix patching: the control tracks the effect")
    ax.legend(fontsize=6.7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pub_patch_layers.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cells(res: Path, out: Path) -> None:
    s7 = json.loads((res / "exp7" / "summary.json").read_text())
    dl = s7["deep_layer"]
    ag = pd.read_csv(res / "exp7" / "span_patch_agg.csv")
    deep = ag[ag.layer == dl]
    order = [("readout", "paired", "deploy / same content"),
             ("readout", "unpaired", "deploy / other content"),
             ("readout", "sameclass", "eval / same content"),
             ("readout", "unpaired_eval", "eval / other content"),
             ("task", "paired", "task span, deploy / same"),
             ("task", "unpaired", "task span, deploy / other")]
    labels, means, los, his, colors = [], [], [], [], []
    for region, src, lab in order:
        r = deep[(deep.region == region) & (deep.source == src)]
        if not len(r):
            continue
        labels.append(lab)
        means.append(r["mean"].iloc[0])
        los.append(r["mean"].iloc[0] - r.lo.iloc[0])
        his.append(r.hi.iloc[0] - r["mean"].iloc[0])
        colors.append(C_CTRL if "deploy" in lab and "task" not in lab
                      else (C_MUTED if "task" in lab else C_MAIN))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = range(len(labels))[::-1]
    ax.barh(list(y), means, xerr=[los, his], color=colors, height=.62,
            error_kw=dict(lw=.9, capsize=2))
    ax.set_yticks(list(y), labels, fontsize=7.2)
    ax.axvline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel(r"$\Delta$ verbalized belief (logits)")
    ax.set_title(f"Source cells at layer {dl}: class moves it, content does not")
    fig.tight_layout()
    fig.savefig(out / "pub_cells.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_dose(res: Path, out: Path) -> None:
    df = pd.read_csv(res / "exp2" / "dose_belief.csv")
    st = df[df.layer_tag == "steer"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for cond, framing, color, label in (
            ("d", "eval", C_MAIN, "awareness direction (eval framing)"),
            ("d", "deploy", C_ALT, "awareness direction (deploy framing)"),
            ("random", "eval", C_MUTED, "random directions (mean)")):
        sub = (st[(st.condition == cond) & (st.framing == framing)]
               .groupby("alpha").belief.mean())
        if not len(sub):
            continue
        ax.plot(sub.index, sub.values, "-o", color=color, lw=1.6, ms=2.6,
                label=label)
    ax.axvline(0, color=C_MUTED, lw=.6, ls=":")
    ax.set_xlabel(r"steering dose $\alpha$ (multiples of $\sigma$)")
    ax.set_ylabel("verbalized belief (logits)")
    L = int(st.layer.iloc[0])
    ax.set_title(f"Steering dose-response at the locked layer ({L})")
    ax.legend(fontsize=6.7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pub_dose.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_register(res: Path, out: Path) -> None:
    s8 = json.loads((res / "exp8" / "summary.json").read_text())
    sw = s8["swings"]
    named = [("d", "awareness $d$"), ("d_perp_style", r"$d$ ⊥ style"),
             ("style", "style"), ("style_perp_d", "style ⊥ $d$")]
    rand = sorted((abs(v["mean"]) for k, v in sw.items()
                   if k.startswith("random_")), reverse=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = [lab for k, lab in named if k in sw]
    means = [sw[k]["mean"] for k, _ in named if k in sw]
    los = [sw[k]["mean"] - sw[k]["lo"] for k, _ in named if k in sw]
    his = [sw[k]["hi"] - sw[k]["mean"] for k, _ in named if k in sw]
    y = range(len(labels))[::-1]
    ax.barh(list(y), means, xerr=[los, his], color=C_MAIN, height=.6,
            error_kw=dict(lw=.9, capsize=2))
    ax.set_yticks(list(y), labels, fontsize=8)
    if rand:
        import numpy as np
        mu = float(np.mean(rand))
        lo, hi = (float(np.percentile(rand, 5)), float(np.percentile(rand, 95)))
        ax.axvspan(-hi, hi, color=C_CTRL, alpha=.10, lw=0,
                   label=f"random null, 5-95% of |swing| (n={len(rand)})")
        ax.axvline(mu, color=C_CTRL, lw=1.0, ls="--", label=f"random-null mean |swing| ({mu:.2f})")
        ax.axvline(-mu, color=C_CTRL, lw=1.0, ls="--")
        ax.legend(fontsize=6.4, frameon=False, loc="lower right")
    ax.axvline(0, color=C_MUTED, lw=.7)
    ax.set_xlabel(r"belief swing across $\pm4\sigma$ (logits)")
    ax.set_title(f"Orthogonalized steering at layer {s8['layer']}")
    fig.tight_layout()
    fig.savefig(out / "pub_register.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen15b")
    a = ap.parse_args()
    res = ROOT / "results" / a.model
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_patch_layers, fig_cells, fig_dose, fig_register):
        try:
            fn(res, out)
            print(f"  {fn.__name__} ok")
        except Exception as exc:
            print(f"  {fn.__name__} FAILED: {exc}")


if __name__ == "__main__":
    main()
