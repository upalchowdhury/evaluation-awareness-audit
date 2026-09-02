#!/usr/bin/env python
"""One-screen digest of a completed run: every quantity the Results section quotes.

  python scripts/report.py --model qwen15b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    r = ROOT / a.results / a.model
    s = {i: load(r / f"exp{i}" / "summary.json") for i in range(1, 7)}

    def head(t):
        print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")

    head(f"R1 representation  [{a.model}]")
    print(json.dumps(s[1], indent=2, default=str))
    p = r / "exp1" / "probe_curve.csv"
    if p.exists():
        c = pd.read_csv(p)
        print(f"\ntest_auc: max {c.test_auc.max():.3f} @L{int(c.test_auc.idxmax())} | "
              f"min {c.test_auc.min():.3f} @L{int(c.test_auc.idxmin())} | "
              f"last {c.test_auc.iloc[-1]:.3f}")
    p = r / "exp1" / "generalization.csv"
    if p.exists():
        g = pd.read_csv(p)
        off = g[g.train_family != g.test_family]
        print(f"off-diagonal transfer AUC: mean {off.auc.mean():.3f}, "
              f"min {off.auc.min():.3f} "
              f"({off.loc[off.auc.idxmin(), 'train_family']} -> "
              f"{off.loc[off.auc.idxmin(), 'test_family']})")

    head("R2 causal belief")
    print(json.dumps(s[2], indent=2, default=str))
    p = r / "exp2" / "agg_patch.csv"
    if p.exists():
        d = pd.read_csv(p)
        for direction, sub in d.groupby("direction"):
            i = sub["mean"].abs().idxmax()
            print(f"  {direction:>18}: peak {sub.loc[i, 'mean']:+.3f} "
                  f"[{sub.loc[i, 'lo']:+.3f},{sub.loc[i, 'hi']:+.3f}] @L{int(sub.loc[i, 'layer'])}")
    p = r / "exp2" / "dose_belief.csv"
    if p.exists():
        print("\ndose-response (belief vs alpha):")
        print(pd.read_csv(p).to_string(index=False))

    head("R3 behavior")
    print(json.dumps(s[3], indent=2, default=str))
    for f in ("sandbag_baseline.csv", "sandbag_controls.csv", "sandbag_ablate.csv",
              "sycophancy.csv", "refusal.csv", "selfreport.csv"):
        p = r / "exp3" / f
        if p.exists():
            print(f"\n-- {f}")
            print(pd.read_csv(p).to_string(index=False))
    p = r / "exp3" / "sandbag_steer.csv"
    if p.exists():
        st = pd.read_csv(p)
        print("\n-- sandbag_steer: p_correct swing (alpha_max - alpha_min) per cell")
        for (l, fr, sy), sub in st.groupby(["layer", "framing", "system"]):
            sub = sub.sort_values("alpha")
            print(f"  L{l} {fr}/{sy}: {sub.p_correct.iloc[-1] - sub.p_correct.iloc[0]:+.4f} "
                  f"(base {sub[sub.alpha == 0].p_correct.iloc[0]:.4f})")

    head("R4 stages")
    print(json.dumps(s[4], indent=2, default=str))

    head("R5 positions")
    print(json.dumps(s[5], indent=2, default=str))

    head("R6 circuit")
    print(json.dumps(s[6], indent=2, default=str))
    p = r / "exp6" / "verify.csv"
    if p.exists():
        print(pd.read_csv(p).to_string(index=False))
    p = r / "exp6" / "alignment.csv"
    if p.exists():
        print("\n-- write/read scores")
        print(pd.read_csv(p).to_string(index=False))


if __name__ == "__main__":
    main()
