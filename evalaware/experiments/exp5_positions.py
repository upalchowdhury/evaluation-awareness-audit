"""exp5 (Experiment D): WHERE in the prompt is the belief constructed?

Two maps:
  1. decodability grid (from exp1): mass-mean probe AUC at (layer x position)
  2. causal span map: project d_hat OUT of the residual stream at one
     (layer, span) at a time - cue span / task span / readout span - and
     measure the belief drop. Shows which token region the belief signal is
     read from, causally rather than correlationally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .. import plotting as P
from ..data.build import make_pairs
from ..data.templates import BELIEF_READOUT_MARKER
from ..interventions import ablate_span_hook, resid_name, tokens_for
from ..model_io import chat_wrap
from ..readouts import belief_from_logits, belief_prompt
from ..stats import bootstrap_mean_ci
from ..utils import find_span, log, save_json, timer
from .common import Ctx, sweep_layers

SPANS = ["cue", "task", "readout"]


def token_spans(model, wrapped: str, user_text: str, task_text: str) -> dict | None:
    """Map cue/task/readout character regions to token index ranges."""
    enc = model.tokenizer(wrapped, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    cs_user = find_span(wrapped, user_text[: min(48, len(user_text))])
    cs_task = find_span(wrapped, task_text)
    cs_read = find_span(wrapped, BELIEF_READOUT_MARKER)
    if not (cs_user and cs_task and cs_read):
        return None

    def tok_range(c0: int, c1: int) -> tuple[int, int]:
        idx = [i for i, (a, b) in enumerate(offsets) if b > c0 and a < c1]
        return (idx[0], idx[-1] + 1) if idx else (0, 0)

    spans = dict(
        cue=tok_range(cs_user[0], cs_task[0]),
        task=tok_range(cs_task[0], cs_read[0]),
        readout=tok_range(cs_read[0], len(wrapped)),
    )
    if any(e <= s for s, e in spans.values()):
        return None
    return spans


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp5")
    dirs, sigma, s1 = ctx.directions()
    layers = sweep_layers(ctx)
    seed = int(cfg.data.seed)

    n = min(int(cfg.interventions.n_belief_pairs), 40)
    pairs = make_pairs(ctx.corpus(), n, seed=seed + 5, split="test")

    prepared = []
    for e, _ in pairs:
        wrapped = belief_prompt(model, e["user_text"], "A")
        spans = token_spans(model, wrapped, e["user_text"], e["task_text"])
        if spans is None:
            continue
        prepared.append((e, wrapped, spans))
    log.info(f"exp5: {len(prepared)}/{len(pairs)} prompts with clean span maps")

    rows = []
    with timer("causal span map"):
        for e, wrapped, spans in prepared:
            ids = tokens_for(model, wrapped)
            with torch.no_grad():
                base = belief_from_logits(model, model(ids)[:, -1].float().cpu())[0]
            for l in layers:
                for span in SPANS:
                    s, t = spans[span]
                    hook = ablate_span_hook(dirs[l], s, t)
                    with torch.no_grad(), model.hooks(fwd_hooks=[(resid_name(l), hook)]):
                        lg = model(ids)[:, -1].float().cpu()
                    delta = belief_from_logits(model, lg)[0] - base
                    rows.append(dict(layer=l, span=span, content_id=e["content_id"],
                                     delta=float(delta)))
    df = pd.DataFrame(rows)
    df.to_csv(d / "span_ablation.csv", index=False)

    agg = []
    for (l, span), sub in df.groupby(["layer", "span"]):
        m, lo, hi = bootstrap_mean_ci(sub.delta, clusters=sub.content_id, seed=seed)
        agg.append(dict(layer=l, span=span, mean=m, lo=lo, hi=hi))
    agg = pd.DataFrame(agg)
    agg.to_csv(d / "span_ablation_agg.csv", index=False)

    mat = np.full((len(layers), len(SPANS)), np.nan)
    for i, l in enumerate(layers):
        for j, span in enumerate(SPANS):
            v = agg[(agg.layer == l) & (agg.span == span)]["mean"]
            if len(v):
                mat[i, j] = v.iloc[0]
    P.save(
        P.heatmap(mat, SPANS, [str(l) for l in layers],
                  "D: belief change when d is ablated at (layer x span)",
                  r"$\Delta$ belief", kind="div", center=0.0, figsize=(4.6, 3.6)),
        d / "fig_span_ablation", agg,
    )

    P.save(
        P.layer_curves(
            [dict(label=span,
                  x=agg[agg.span == span].sort_values("layer").layer,
                  y=agg[agg.span == span].sort_values("layer")["mean"],
                  lo=agg[agg.span == span].sort_values("layer").lo,
                  hi=agg[agg.span == span].sort_values("layer").hi)
             for span in SPANS],
            ylabel=r"$\Delta$ belief (ablate d at span)", hline=0.0,
            title="D: which token region carries the signal, by layer",
        ),
        d / "fig_span_curves", agg,
    )

    strongest = agg.loc[agg["mean"].abs().idxmax()] if len(agg) else None
    # Per-span peaks: the staged account predicts the CUE span matters early
    # (the belief is being constructed there) and the READOUT span late (it is
    # being consumed there). Reported so R5 is a claim, not a figure caption.
    per_span = {}
    for span in SPANS:
        sub = agg[agg.span == span]
        if not len(sub):
            continue
        row = sub.loc[sub["mean"].abs().idxmax()]
        per_span[str(span)] = dict(peak_layer=int(row.layer), delta=float(row["mean"]))
    save_json(d / "summary.json", dict(
        n_prompts=len(prepared),
        strongest=None if strongest is None else dict(
            layer=int(strongest.layer), span=str(strongest.span),
            delta=float(strongest["mean"])),
        per_span=per_span,
    ))
    ctx.write_manifest("exp5")
