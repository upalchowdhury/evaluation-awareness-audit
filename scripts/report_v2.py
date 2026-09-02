#!/usr/bin/env python
"""One-screen digest of the v2 numbers per model, for writing verdict
sentences from the data rather than from memory.

  python scripts/report_v2.py [--results results_v2] [--models a,b,c]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def ci(d, nd=2):
    if not d or d.get("mean") is None:
        return "   --"
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--models", default="qwen05b,qwen15b,qwen7b,phi3mini,olmo7b")
    a = ap.parse_args()
    res = ROOT / a.results
    for sh in a.models.split(","):
        d = res / sh
        if not (d / "exp1" / "summary.json").exists():
            print(f"===== {sh}: no exp1 yet")
            continue
        s1, s2, s3 = (_load(d / f"exp{i}" / "summary.json") for i in (1, 2, 3))
        s7, s8, s10, s11 = (_load(d / f"exp{i}" / "summary.json") for i in (7, 8, 10, 11))
        gap = s1.get("natural_gap") or {}
        val = s1.get("readout_validity") or {}
        g = gap.get("mean")
        print(f"===== {sh}  ({_load(d / 'dataset' / 'manifest.json').get('model', sh)})")
        curve = pd.read_csv(d / "exp1" / "probe_curve.csv").sort_values("layer") if (d / "exp1" / "probe_curve.csv").exists() else None
        if curve is not None:
            print(f"  probe AUC: layer0 {curve.test_auc.iloc[0]:.3f}  best {curve.test_auc.max():.3f} (L{int(curve.loc[curve.test_auc.idxmax(), 'layer'])})"
                  f"  | verbal AUC {s1.get('verbal_auc', float('nan')):.3f}  probe-vs-verbal rho {s1.get('repr_verbal_spearman', float('nan')):.2f}")
        print(f"  readout: greedy Yes/No {100 * val.get('greedy_yesno_frac', float('nan')):.0f}%  mass {val.get('yesno_mass_mean', float('nan')):.3f}"
              f"  | natural gap {ci(gap)}  eval mean {gap.get('eval_mean', float('nan')):+.2f} deploy mean {gap.get('deploy_mean', float('nan')):+.2f}"
              f"  score>0: eval {100 * gap.get('frac_positive_eval', float('nan')):.0f}% deploy {100 * gap.get('frac_positive_deploy', float('nan')):.0f}%")
        lk = s2.get("locked") or {}
        se, sd = lk.get("steer_eval") or {}, lk.get("steer_deploy") or {}
        if se:
            print(f"  locked steering L{se['layer']}: eval {ci(se)} ({abs(se['mean']) / abs(g):.2f} gaps)"
                  f" | deploy {ci(sd)} ({abs(sd['mean']) / abs(g):.2f} gaps)"
                  f" | raw patch max {s2.get('max_patch_effect', float('nan')):+.2f} unpaired max {s2.get('max_unpaired_effect', float('nan')):.2f}")
        if s8:
            sw = s8.get("swings") or {}
            p4 = (s8.get("percentiles_all") or {}).get("alpha4") or {}
            p1 = (s8.get("percentiles_all") or {}).get("alpha1") or {}
            rp = s8.get("random_p95") or {}
            def _sw(k):
                v = sw.get(k)
                return f"{v['mean']:+.2f}" if v else "--"
            print(f"  register L{s8.get('layer')}: cos(d,style) {s8.get('cos_d_style_at_layer', float('nan')):.2f}"
                  f"  cos(d,neutral) {s8.get('cos_d_style_neutral_at_layer', float('nan')):.2f}"
                  f"  | swings@4: d {_sw('d')} (pct {p4.get('d')}) style {_sw('style')} (pct {p4.get('style')})"
                  f" neutral {_sw('style_neutral')} (pct {p4.get('style_neutral')}) d-neutral {_sw('d_perp_style_neutral')} (pct {p4.get('d_perp_style_neutral')})"
                  f"  | random p95 @4 {rp.get('alpha4', float('nan')):.2f} @1 {rp.get('alpha1', float('nan')):.2f}"
                  f"  | small-dose pct d {p1.get('d')} neutral {p1.get('style_neutral')}"
                  f"  | frac d surviving neutral removal {s8.get('frac_d_surviving_neutral_removal')}")
        if s10:
            for dc in ("eval", "deploy"):
                parts = [f"{r} {ci(s10.get(f'{dc}_{r}_class'))}" for r in ("task", "question", "final", "readout")]
                print(f"  positional ({dc} dst, L{s10.get('deep_layer')}): " + "; ".join(parts))
            ef = s10.get("eval_final_class") or {}
            if ef.get("mean") is not None and g:
                print(f"    final-position class effect = {abs(ef['mean']) / abs(g):.2f} gaps; interaction {ci(s10.get('eval_final_interaction'))}")
        if s11:
            for dc in ("eval", "deploy"):
                print(f"  matched ({dc} dst): unmatched {ci(s11.get(f'unmatched_{dc}_deep_final_class'))}"
                      f"  shrink-only {ci(s11.get(f'constructed_min_{dc}_deep_final_class'))}"
                      f"  shrink readout {ci(s11.get(f'constructed_min_{dc}_deep_readout_class'))}")
        if s3:
            print(f"  sandbag gap {ci(s3.get('sandbag_gap_paired'))}  incentive {ci(s3.get('incentive_effect_paired'))}  cells {s3.get('baseline_cells')}")
    praw = ROOT / "paper" / "tables" / "position_v2_raw.csv"
    if praw.exists():
        pr = pd.read_csv(praw)
        print("===== positional cells (from paper/tables/position_v2_raw.csv):",
              f"{len(pr)} cells, {(pr.p < 0.05).sum()} uncorrected, {(pr.holm < 0.05).sum()} Holm")
        print(pr[["sh", "dst", "region", "mean", "lo", "hi", "p", "holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
