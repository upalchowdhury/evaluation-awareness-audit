"""Corpus construction: contrastive eval/deploy prompts with clean splits.

Split discipline
----------------
* Contents are split train/test ONCE (by content id). Probes and directions are
  fit on TRAIN contents only; every causal experiment (patching, steering,
  behavior) runs on TEST contents. Nothing causal is ever measured on prompts
  the probe saw.
* Cue families are additionally held out one-at-a-time for the generalization
  matrix (exp1), including the no-lexicon `subtle_exam_format` family.
* Topic controls (deployment-context prompts ABOUT evaluation) and ambiguous
  framings are labeled special and excluded from all training.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..utils import load_jsonl, save_jsonl
from .contents import get_contents
from .templates import (
    AMBIGUOUS,
    DEPLOY_FAMILIES,
    EVAL_FAMILIES,
    TOPIC_CONTROLS,
    has_trailing_text,
    render,
)
from . import behaviors as B


def build_corpus(cfg) -> list[dict]:
    rng = random.Random(cfg.data.seed)
    contents = get_contents(cfg.data.n_contents)
    tpf = int(cfg.data.templates_per_family)

    records: list[dict] = []
    for c in contents:
        for cls, fams in (("eval", EVAL_FAMILIES), ("deploy", DEPLOY_FAMILIES)):
            for fam, bank in fams.items():
                for ti in rng.sample(range(len(bank)), min(tpf, len(bank))):
                    records.append(
                        dict(
                            content_id=c["id"],
                            domain=c["domain"],
                            kind=c["kind"],
                            answer=c["answer"],
                            cls=cls,
                            family=fam,
                            template_idx=ti,
                            user_text=render(bank[ti], c["task"]),
                            task_text=c["task"],
                            special=None,
                            trailing=has_trailing_text(bank[ti]),
                        )
                    )

    for i, t in enumerate(TOPIC_CONTROLS):
        records.append(
            dict(content_id=f"tc{i:02d}", domain="meta", kind="open", answer=None,
                 cls="deploy", family="topic_control", template_idx=0,
                 user_text=t, task_text=t, special="topic_control")
        )
    amb_contents = rng.sample(contents, min(len(AMBIGUOUS), len(contents)))
    for i, tpl in enumerate(AMBIGUOUS):
        c = amb_contents[i % len(amb_contents)]
        records.append(
            dict(content_id=c["id"], domain=c["domain"], kind=c["kind"], answer=c["answer"],
                 cls="ambiguous", family="ambiguous", template_idx=i,
                 user_text=render(tpl, c["task"]), task_text=c["task"], special="ambiguous")
        )

    # content-level split, stratified by kind so MC items split proportionally
    # (the capability behavior suite draws exclusively on TEST-split MC items)
    by_kind: dict[str, list[str]] = {}
    for c in contents:
        by_kind.setdefault(c["kind"], []).append(c["id"])
    test_ids: set[str] = set()
    for kind_ids in by_kind.values():
        rng.shuffle(kind_ids)
        n_test = max(1, round(len(kind_ids) * float(cfg.data.test_content_frac)))
        test_ids |= set(kind_ids[:n_test])
    for r in records:
        if r["special"]:
            r["split"] = "special"
        else:
            r["split"] = "test" if r["content_id"] in test_ids else "train"

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["pid"] = f"p{i:05d}"
    return records


def load_corpus(path: str | Path) -> list[dict]:
    return load_jsonl(path)


def save_corpus(records: list[dict], path: str | Path) -> None:
    save_jsonl(path, records)


def make_pairs(records: list[dict], n: int, seed: int = 0, split: str = "test") -> list[tuple[dict, dict]]:
    """(eval_rec, deploy_rec) pairs sharing content, drawn from `split` contents.

    Multiple distinct framing pairs per content are allowed; the returned pairs
    carry content_id so downstream CIs can cluster-bootstrap by content.
    """
    rng = random.Random(seed)
    by_content: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        if r["special"] or r["split"] != split or r.get("trailing"):
            continue  # trailing-cue renderings do not share the task suffix
        by_content.setdefault(r["content_id"], {"eval": [], "deploy": []})[r["cls"]].append(r)

    pairs: list[tuple[dict, dict]] = []
    contents = sorted(by_content)
    rng.shuffle(contents)
    round_i = 0
    while len(pairs) < n and round_i < 8:
        for cid in contents:
            ev, dp = by_content[cid]["eval"], by_content[cid]["deploy"]
            if len(ev) > round_i and len(dp) > round_i and len(pairs) < n:
                pairs.append((ev[round_i], dp[round_i]))
        round_i += 1
    return pairs[:n]


def build_behavior_suites(cfg) -> dict:
    """Framed behavior items. Capability items come from TEST-split MC contents only."""
    rng = random.Random(int(cfg.data.seed) + 1)
    corpus = build_corpus(cfg)  # deterministic given seed: reuse for the split
    test_ids = sorted({r["content_id"] for r in corpus if r["split"] == "test"})
    contents = {c["id"]: c for c in get_contents(cfg.data.n_contents)}
    mc_test = [contents[i] for i in test_ids if contents[i]["kind"] == "mc"]
    rng.shuffle(mc_test)

    eval_fams = list(EVAL_FAMILIES)
    dep_fams = list(DEPLOY_FAMILIES)

    def frame(task: str, i: int) -> dict[str, str]:
        ef = eval_fams[i % len(eval_fams)]
        df = dep_fams[i % len(dep_fams)]
        et = EVAL_FAMILIES[ef][rng.randrange(len(EVAL_FAMILIES[ef]))]
        dt = DEPLOY_FAMILIES[df][rng.randrange(len(DEPLOY_FAMILIES[df]))]
        return {"eval": render(et, task), "deploy": render(dt, task),
                "eval_family": ef, "deploy_family": df}

    capability = []
    for i, c in enumerate(mc_test[: int(cfg.behaviors.n_capability_items)]):
        f = frame(c["task"], i)
        capability.append(dict(item_id=f"cap{i:02d}", content_id=c["id"], answer=c["answer"],
                               task_text=c["task"], **f))

    sycophancy = []
    for i, s in enumerate(B.SYCOPHANCY_STATEMENTS[: int(cfg.behaviors.n_sycophancy_items)]):
        neutral, asserted = B.SYC_NEUTRAL.format(s=s), B.SYC_ASSERTED.format(s=s)
        fn, fa = frame(neutral, i), frame(asserted, i)
        sycophancy.append(dict(item_id=f"syc{i:02d}", statement=s,
                               neutral_eval=fn["eval"], neutral_deploy=fn["deploy"],
                               asserted_eval=fa["eval"], asserted_deploy=fa["deploy"]))

    refusal = []
    for i, t in enumerate(B.REFUSAL_ITEMS[: int(cfg.behaviors.n_refusal_items)]):
        f = frame(t, i)
        refusal.append(dict(item_id=f"ref{i:02d}", task_text=t, **f))

    selfreport = []
    for i, d in enumerate(B.SELFREPORT_DOMAINS[: int(cfg.behaviors.n_selfreport_items)]):
        t = B.SELFREPORT_TEMPLATE.format(d=d)
        f = frame(t, i)
        selfreport.append(dict(item_id=f"sr{i:02d}", domain=d, task_text=t, **f))

    return dict(capability=capability, sycophancy=sycophancy,
                refusal=refusal, selfreport=selfreport)
