#!/usr/bin/env python
"""Print randomly sampled raw items for manual human verification.

  python scripts/sample_audit.py --model qwen15b --n 5 --seed 7

For each sampled content item this prints: the rendered eval and deploy
prompts (exact model input), the token-span boundaries used by span
experiments, the recorded belief scores, one 2x2 patch record, and one
steering record --- so a human can check, end to end, that the numbers in
the paper trace to real prompts and sensible alignments.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen15b")
    ap.add_argument("--results", default="results",
                    help="results dir (results = v1 readout, results_v2 = second-turn readout)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = ROOT / a.results / a.model
    rng = random.Random(a.seed)

    corpus = [json.loads(l) for l in open(res / "dataset" / "corpus.jsonl")]
    test = [r for r in corpus if r["split"] == "test" and not r["special"]]
    by_content: dict = {}
    for r in test:
        by_content.setdefault(r["content_id"], []).append(r)
    picks = rng.sample(sorted(by_content), min(a.n, len(by_content)))

    verbal = pd.read_csv(res / "exp1" / "verbal.csv")
    try:
        sp = pd.read_csv(res / "exp7" / "span_patch.csv")
    except FileNotFoundError:
        sp = None
    try:
        st = pd.read_csv(res / "exp2" / "steer_belief.csv")
    except FileNotFoundError:
        st = None

    for cid in picks:
        recs = by_content[cid]
        ev = next((r for r in recs if r["cls"] == "eval"), None)
        dp = next((r for r in recs if r["cls"] == "deploy"), None)
        print("=" * 78)
        print(f"CONTENT {cid}  ({len(recs)} renderings in test split)")
        for tag, r in (("EVAL", ev), ("DEPLOY", dp)):
            if r is None:
                continue
            print(f"\n--- {tag} rendering (pid {r['pid']}, family {r['family']}) ---")
            print(r["user_text"][:600])
            vb = verbal[verbal.pid == r["pid"]]
            if len(vb):
                print(f"[belief score {vb.belief.iloc[0]:+.3f} | "
                      f"probe projection {vb.probe_proj.iloc[0]:+.3f}]")
        if sp is not None:
            rows = sp[sp.content_id == cid]
            if len(rows):
                r0 = rows.iloc[0]
                print(f"\n[2x2 patch sample] layer {r0.layer} region {r0.region} "
                      f"source {r0.source}: delta {r0.delta:+.3f} "
                      f"(patched last {r0.a}..{r0.b} tokens from end)")
        if st is not None:
            rows = st[st.content_id == cid]
            if len(rows):
                r0 = rows.iloc[0]
                print(f"[steering sample] layer {r0.layer} framing {r0.framing} "
                      f"alpha {r0.alpha}: delta {r0.delta:+.3f}")
        print()


if __name__ == "__main__":
    main()
