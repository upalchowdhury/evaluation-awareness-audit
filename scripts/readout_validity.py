#!/usr/bin/env python
"""Instrument check: does the belief readout actually read a Yes/No answer?

The belief score is logsumexp(Yes variants) - logsumexp(No variants) over the
first answer token. That number is only meaningful if the model's first token
is (nearly always) a Yes/No variant. If most probability mass is elsewhere
(e.g. "I", "As", "This", "\n"), the score is a difference between two tokens
the model never emits, and any "the report does not move" finding is an
instrument failure rather than a fact about the model.

For each model: 20 eval + 20 deploy held-out prompts, both polarities.
Reports P(Yes)+P(No) mass, the share of prompts whose greedy first token is a
Yes/No variant, the top-5 first tokens, and the belief score.

  python scripts/readout_validity.py qwen15b
Writes results/<short>/readout_validity.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalaware.config import load_config  # noqa: E402
from evalaware.data.templates import BELIEF_SUFFIX_A, BELIEF_SUFFIX_B  # noqa: E402
from evalaware.model_io import (  # noqa: E402
    chat_wrap,
    final_logits,
    first_token_variant_ids,
    load_model,
)
from evalaware.utils import load_jsonl  # noqa: E402


def main() -> None:
    short = sys.argv[1]
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cfg = load_config("configs/default.yaml", f"configs/models/{short}.yaml")
    model = load_model(cfg.model)
    tok = model.tokenizer
    corpus = load_jsonl(f"results/{short}/dataset/corpus.jsonl")
    test = [r for r in corpus if r["split"] == "test" and not r["special"]]
    rng = random.Random(0)
    ev = rng.sample([r for r in test if r["cls"] == "eval"], n_per)
    dp = rng.sample([r for r in test if r["cls"] == "deploy"], n_per)
    yes = first_token_variant_ids(tok, ["Yes"])
    no = first_token_variant_ids(tok, ["No"])
    print(f"[{short}] yes ids {yes} -> {[tok.decode([i]) for i in yes]}")
    print(f"[{short}] no  ids {no} -> {[tok.decode([i]) for i in no]}")
    yesno = set(yes) | set(no)

    out: dict = dict(model=cfg.model.name, short=short, yes_ids=yes, no_ids=no)
    for pol, suf in (("A", BELIEF_SUFFIX_A), ("B", BELIEF_SUFFIX_B)):
        for cls, recs in (("eval", ev), ("deploy", dp)):
            texts = [chat_wrap(model, r["user_text"] + suf) for r in recs]
            lg = final_logits(model, texts, batch_size=4)
            p = torch.softmax(lg, dim=-1)
            m_yes = p[:, yes].sum(-1)
            m_no = p[:, no].sum(-1)
            mass = m_yes + m_no
            greedy = p.argmax(-1)
            greedy_is_yesno = torch.tensor([int(g) in yesno for g in greedy]).float()
            top = p.topk(5, dim=-1)
            score = (torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1))
            greedy_counts = Counter(tok.decode([int(g)]) for g in greedy)
            key = f"pol{pol}_{cls}"
            out[key] = dict(
                mass_yes_plus_no_mean=float(mass.mean()),
                mass_yes_plus_no_min=float(mass.min()),
                mass_yes_plus_no_median=float(mass.median()),
                p_yes_given_yesno_mean=float((m_yes / mass.clamp_min(1e-9)).mean()),
                greedy_is_yesno_frac=float(greedy_is_yesno.mean()),
                greedy_token_counts=dict(greedy_counts),
                score_mean=float(score.mean()),
                score_std=float(score.std()),
                examples=[
                    dict(
                        family=r["family"],
                        top5=[(tok.decode([int(i)]), round(float(v), 4))
                              for v, i in zip(top.values[j], top.indices[j])],
                        score=round(float(score[j]), 3),
                    )
                    for j, r in enumerate(recs[:3])
                ],
            )
            print(f"[{short}] polarity {pol} {cls:6s}: mass(Yes+No) mean {mass.mean():.3f} "
                  f"min {mass.min():.3f} | greedy is Yes/No {greedy_is_yesno.mean():.2f} "
                  f"| P(Yes|Yes/No) {float((m_yes / mass.clamp_min(1e-9)).mean()):.2f} "
                  f"| score {score.mean():+.2f} +/- {score.std():.2f} "
                  f"| greedy {dict(greedy_counts)}")
    path = Path(f"results/{short}/readout_validity.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"[{short}] wrote {path}")


if __name__ == "__main__":
    main()
