"""exp0: build and freeze the corpus + behavior suites (no model needed)."""
from __future__ import annotations

import json

from ..data.build import build_behavior_suites, build_corpus, save_corpus
from ..utils import log
from .common import Ctx


def run(ctx: Ctx) -> None:
    d = ctx.dir("dataset")
    records = build_corpus(ctx.cfg)
    save_corpus(records, d / "corpus.jsonl")

    n = len(records)
    n_train = sum(r["split"] == "train" for r in records)
    n_test = sum(r["split"] == "test" for r in records)
    n_special = sum(r["split"] == "special" for r in records)
    log.info(f"corpus: {n} prompts (train {n_train} / test {n_test} / special {n_special})")

    suites = build_behavior_suites(ctx.cfg)
    with open(d / "behaviors.json", "w") as f:
        json.dump(suites, f, indent=2)
    log.info("behavior suites: " + ", ".join(f"{k}={len(v)}" for k, v in suites.items()))

    ctx.write_manifest("dataset", dict(n_prompts=n, n_train=n_train, n_test=n_test,
                                       n_special=n_special))
