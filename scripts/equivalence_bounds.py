#!/usr/bin/env python
"""Equivalence (minimum-effect-size) bounds for the exp10 content and
interaction "nulls".

Reviewer point: a CI overlapping zero is not evidence of a negligible effect.
This script quantifies how large a content main effect / interaction the exp10
data can actually rule out, TOST-style: the smallest delta such that the 90%
content-clustered bootstrap CI of the effect lies inside [-delta, +delta]
(equivalently delta = max(|lo90|, |hi90|); a 90% CI inside the bound is the
usual two-one-sided-tests criterion at alpha = 0.05).

  python scripts/equivalence_bounds.py

Per model (deep layer, from exp10/summary.json), per dst_cls, at the `final`
region (primary) and `readout` (secondary), reports:
  (a) the equivalence bound for the content main effect and the interaction,
  (b) that bound as a fraction of |class main effect| in the same cell
      (>= 1 means the null is uninformative at the scale of the class effect),
  (c) the class main effect itself (95% CI) for reference.
Effects and clustering are computed exactly as in exp10_span_directions.py
(per-content cell means, evalaware.stats.bootstrap_mean_ci, seed 0).
Writes results/equivalence.json and prints a compact table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalaware.stats import bootstrap_mean_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["qwen05b", "qwen15b", "qwen7b", "phi3mini", "olmo7b"]
REGIONS = ["final", "readout"]  # primary, secondary
SEED = 0
NEED = {"samecls_samecont", "othercls_samecont",
        "samecls_diffcont", "othercls_diffcont"}


def cell_rows(model: str) -> list[dict]:
    res = ROOT / "results" / model / "exp10"
    df = pd.read_csv(res / "span_direction_patch.csv")
    deep = int(json.loads((res / "summary.json").read_text())["deep_layer"])
    piv = df[df.layer == deep].pivot_table(
        index=["dst_cls", "region", "content_id"],
        columns="source", values="delta").reset_index()
    out = []
    for dc in ("eval", "deploy"):
        for region in REGIONS:
            sub = piv[(piv.dst_cls == dc) & (piv.region == region)]
            if not len(sub) or not NEED <= set(sub.columns):
                continue
            # same per-content contrasts as exp10's factorial analysis
            cls_ = ((sub.othercls_samecont + sub.othercls_diffcont) / 2
                    - (sub.samecls_samecont + sub.samecls_diffcont) / 2).dropna()
            cnt = ((sub.samecls_diffcont + sub.othercls_diffcont) / 2
                   - (sub.samecls_samecont + sub.othercls_samecont) / 2).dropna()
            # halved: same scale as the (averaged) main effects
            ixn = (((sub.othercls_samecont - sub.samecls_samecont)
                    - (sub.othercls_diffcont - sub.samecls_diffcont)) / 2).dropna()

            def _ci(vals, alpha):
                return bootstrap_mean_ci(
                    vals.to_numpy(),
                    clusters=sub.content_id.loc[vals.index].to_numpy(),
                    seed=SEED, alpha=alpha)

            cm, cl, ch = _ci(cls_, 0.05)          # class ME, 95% CI (reference)
            row = dict(model=model, dst_cls=dc, region=region, layer=deep,
                       n_contents=int(sub.content_id.nunique()),
                       class_mean=cm, class_lo95=cl, class_hi95=ch)
            for name, vals in (("content", cnt), ("interaction", ixn)):
                m, lo, hi = _ci(vals, 0.10)        # 90% CI -> TOST at 0.05
                bound = max(abs(lo), abs(hi))
                frac = bound / abs(cm) if abs(cm) > 1e-12 else float("nan")
                row.update({f"{name}_mean": m, f"{name}_lo90": lo,
                            f"{name}_hi90": hi, f"{name}_equiv_bound": bound,
                            f"{name}_bound_over_class": frac})
            out.append(row)
    return out


def main() -> None:
    rows = []
    for model in MODELS:
        try:
            rows += cell_rows(model)
        except FileNotFoundError as e:
            print(f"skip {model}: {e}", file=sys.stderr)
    (ROOT / "results" / "equivalence.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    print(f"wrote results/equivalence.json ({len(rows)} rows)\n")

    hdr = (f"{'model':<9} {'dst':<7} {'region':<8} {'class ME [95%]':<24} "
           f"{'content bnd':>11} {'(x class)':>9} "
           f"{'ixn bnd':>9} {'(x class)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cls_s = (f"{r['class_mean']:+.2f} "
                 f"[{r['class_lo95']:+.2f},{r['class_hi95']:+.2f}]")
        print(f"{r['model']:<9} {r['dst_cls']:<7} {r['region']:<8} {cls_s:<24} "
              f"{r['content_equiv_bound']:>11.3f} "
              f"{r['content_bound_over_class']:>9.2f} "
              f"{r['interaction_equiv_bound']:>9.3f} "
              f"{r['interaction_bound_over_class']:>9.2f}")
    bad = [r for r in rows
           if not np.isnan(r["content_bound_over_class"])
           and max(r["content_bound_over_class"],
                   r["interaction_bound_over_class"]) >= 1.0]
    if bad:
        print(f"\nnote: {len(bad)}/{len(rows)} cells have an equivalence bound "
              f">= |class ME| -- in those cells the null is uninformative at "
              f"the scale of the class effect.")


if __name__ == "__main__":
    main()
