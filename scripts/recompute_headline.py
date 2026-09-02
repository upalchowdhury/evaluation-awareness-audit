#!/usr/bin/env python
"""Recompute the paper's headline numbers from raw per-item CSVs, independently
of scripts/make_paper_assets.py -- so a human can verify them by hand.

  python scripts/recompute_headline.py --model qwen15b

Prints, from raw records only (no summary.json used):
  1. the source 2x2 cells, class / content main effects, and the interaction
     at the readout span (exp7/span_patch.csv)
  2. the paired-minus-unpaired contrast at the same span
  3. the positional class effects (exp10/span_direction_patch.csv)
  4. the locked steering estimate (exp2/steer_belief.csv, estimation half)
All CIs: content-clustered bootstrap, 2000 resamples, seed 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def cluster_ci(values, clusters, n_boot=2000, seed=0):
    """Mean and 95% CI by resampling whole clusters (content items)."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        means.append(values[idx].mean())
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen15b")
    ap.add_argument("--results", default="results",
                    help="results dir (results = v1 readout, results_v2 = second-turn readout)")
    a = ap.parse_args()
    res = ROOT / a.results / a.model

    # ---- 1-2. the 2x2 at the readout span, max-effect layer ----
    sp = pd.read_csv(res / "exp7" / "span_patch.csv")
    layer = json.loads((res / "exp7" / "summary.json").read_text())["deep_layer"]
    rr = sp[(sp.layer == layer) & (sp.region == "readout")]
    pv = rr.pivot_table(index=["pair", "content_id"], columns="source",
                        values="delta").reset_index()
    c = pv.content_id.to_numpy()
    print(f"=== {a.model}: source 2x2 at readout span, layer {layer} "
          f"({pv.content_id.nunique()} content clusters) ===")
    for src in ("sameclass", "paired", "unpaired_eval", "unpaired"):
        print(f"  cell {src:<14} {fmt(cluster_ci(pv[src], c))}")
    cls = (pv.paired + pv.unpaired) / 2 - (pv.sameclass + pv.unpaired_eval) / 2
    cnt = (pv.unpaired + pv.unpaired_eval) / 2 - (pv.paired + pv.sameclass) / 2
    # halved, so it sits on the same scale as the two main effects (each of
    # which averages two simple effects); the paper quotes the halved version
    ixn = ((pv.paired - pv.sameclass) - (pv.unpaired - pv.unpaired_eval)) / 2
    std = pv.paired - pv.unpaired
    print(f"  class main effect        {fmt(cluster_ci(cls, c))}")
    print(f"  content main effect      {fmt(cluster_ci(cnt, c))}")
    print(f"  class x content interact {fmt(cluster_ci(ixn, c))}  (halved: main-effect scale)")
    print(f"  paired - unpaired        {fmt(cluster_ci(std, c))}   <- the contrast I first used")

    # ---- 3. positional class effects ----
    p10 = res / "exp10" / "span_direction_patch.csv"
    if p10.exists():
        d10 = pd.read_csv(p10)
        l10 = json.loads((res / "exp10" / "summary.json").read_text())["deep_layer"]
        print(f"\n=== positional class effects, layer {l10} ===")
        for dc in ("eval", "deploy"):
            for region in ("task", "question", "final"):
                sub = d10[(d10.dst_cls == dc) & (d10.layer == l10) & (d10.region == region)]
                pvx = sub.pivot_table(index="content_id", columns="source",
                                      values="delta").dropna()
                eff = ((pvx.othercls_samecont + pvx.othercls_diffcont) / 2
                       - (pvx.samecls_samecont + pvx.samecls_diffcont) / 2)
                print(f"  dst={dc:<6} {region:<9} {fmt(cluster_ci(eff, pvx.index))}")

    # ---- 4. locked steering estimate ----
    st = pd.read_csv(res / "exp2" / "steer_belief.csv")
    uniq = sorted(set(pd.read_csv(res / "exp2" / "patch_belief.csv").content_id))
    disc, est = set(uniq[::2]), set(uniq[1::2])
    e = st[st.framing == "eval"]
    sel = (e[e.content_id.isin(disc)].groupby("layer").delta.mean().abs()
           + st[(st.framing == "deploy") & st.content_id.isin(disc)]
           .groupby("layer").delta.mean().abs())
    L = int(sel.idxmax())
    hold = e[(e.layer == L) & e.content_id.isin(est)]
    print(f"\n=== locked steering: layer {L} chosen on {len(disc)} discovery clusters, "
          f"estimated on {hold.content_id.nunique()} held-out clusters ===")
    print(f"  eval framing, -4 sigma   {fmt(cluster_ci(hold.delta, hold.content_id))}")


if __name__ == "__main__":
    main()
