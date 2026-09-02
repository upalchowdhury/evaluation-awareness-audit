"""exp9 (Experiment H): the control confound is concept-general.

If the class-carrying-control flaw of Exp. F is a fact about paired-prompt
patching rather than about evaluation awareness, it must reproduce on an
unrelated concept. Concept: REGISTER (formal vs casual), entirely within
deployment-class prompts, with a formality readout question. Destination: a
formal rendering of a content item. Source 2x2 (class x content):

    sameclass       formal,  same content
    paired          casual,  same content     (class flipped)
    unpaired_eval   formal,  diff content     (content flipped)   [label kept
    unpaired        casual,  diff content     (both flipped)       for reuse]

Predictions if the confound is general: the readout-span class (formality)
main effect is nonzero; the content main effect is ~0; and the standard
paired-minus-unpaired estimator cancels the class effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data.templates import DEPLOY_FAMILIES
from ..interventions import resid_name, tokens_for
from ..model_io import chat_wrap
from ..readouts import _ids
from ..stats import bootstrap_mean_ci
from ..utils import log, save_json, timer
from .common import Ctx
from .exp5_positions import token_spans
from .exp7_causal_anatomy import _cache_resid, _common_suffix, patch_end_range_hook

FORMAL_FAMILIES = ["production_metadata", "task_context"]
CASUAL_FAMILIES = ["casual_user", "personal_stakes"]

# Mirrors BELIEF_SUFFIX_A's structure (token_spans keys on this exact prefix)
FORMALITY_SUFFIX = (
    "\n\nSeparately from everything above, answer with one word, Yes or No: "
    "is the tone of this conversation formal and professional?"
)


def _formality_score(model, ids) -> float:
    yes = _ids(model, "yes", ["Yes"])
    no = _ids(model, "no", ["No"])
    with torch.no_grad():
        row = model(ids)[0, -1].float().cpu()
    return float(torch.logsumexp(row[yes], 0) - torch.logsumexp(row[no], 0))


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp9")
    seed = int(cfg.data.seed)
    L = model.cfg.n_layers

    import json
    s2 = {}
    p2 = ctx.out / "exp2" / "summary.json"
    if p2.exists():
        s2 = json.loads(p2.read_text())
    deep_layer = int(s2.get("patch_top_layers", [L - 3])[0])
    steer_layer = int(s2.get("steer_layer", L // 2))
    layers = sorted({steer_layer, deep_layer})
    log.info(f"exp9: layers {layers} (steer={steer_layer}, deep={deep_layer})")

    # group deploy prompts by content and register
    by_content: dict[str, dict[str, list]] = {}
    for r in ctx.corpus():
        if r["split"] != "test" or r["special"] or r["cls"] != "deploy":
            continue
        reg = ("formal" if r["family"] in FORMAL_FAMILIES
               else "casual" if r["family"] in CASUAL_FAMILIES else None)
        if reg is None:
            continue
        by_content.setdefault(r["content_id"], {"formal": [], "casual": []})[reg].append(r)

    contents = sorted(cid for cid, g in by_content.items()
                      if len(g["formal"]) >= 2 and len(g["casual"]) >= 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(contents)
    n = min(24, len(contents))
    log.info(f"exp9: {n} usable contents (of {len(contents)})")

    rows = []
    with timer("H: register 2x2 patching"):
        for i, cid in enumerate(contents[:n]):
            g = by_content[cid]
            dest = g["formal"][0]
            we = chat_wrap(model, dest["user_text"] + FORMALITY_SUFFIX)
            spans = token_spans(model, we, dest["user_text"], dest["task_text"])
            if spans is None:
                continue
            other = by_content[contents[(i + 1) % n]]
            srcs = dict(
                sameclass=g["formal"][1],
                paired=g["casual"][0],
                unpaired_eval=other["formal"][0],
                unpaired=other["casual"][0],
            )
            ids_e = tokens_for(model, we)
            T = ids_e.shape[1]
            r_ = T - spans["readout"][0]
            base = _formality_score(model, ids_e)
            ok = True
            src_ids = {}
            for name, rec in srcs.items():
                sids = tokens_for(model, chat_wrap(model, rec["user_text"] + FORMALITY_SUFFIX))
                kk = _common_suffix(ids_e, sids)
                if kk <= r_ + 2:
                    ok = False
                    break
                src_ids[name] = (sids, kk)
            if not ok:
                continue
            for name, (sids, kk) in src_ids.items():
                cache = _cache_resid(model, sids, layers)
                for l in layers:
                    for region, (a, b) in dict(full=(kk, 0), readout=(r_, 0),
                                               task=(kk, r_)).items():
                        hk = patch_end_range_hook(cache[l], a, b)
                        with model.hooks(fwd_hooks=[(resid_name(l), hk)]):
                            m = _formality_score(model, ids_e)
                        rows.append(dict(layer=l, region=region, source=name,
                                         content_id=cid, delta=m - base))
                del cache
    df = pd.DataFrame(rows)
    df.to_csv(d / "register_patch.csv", index=False)

    piv = df.pivot_table(index=["layer", "region", "content_id"],
                         columns="source", values="delta").reset_index()
    eff = []
    for (l, region), sub in piv.groupby(["layer", "region"]):
        if not {"paired", "sameclass", "unpaired", "unpaired_eval"} <= set(sub.columns):
            continue
        cls = ((sub.paired + sub.unpaired) / 2
               - (sub.sameclass + sub.unpaired_eval) / 2).dropna()
        cnt = ((sub.unpaired + sub.unpaired_eval) / 2
               - (sub.paired + sub.sameclass) / 2).dropna()
        std = (sub.paired - sub.unpaired).dropna()   # the standard estimator
        for name, vals in (("class", cls), ("content", cnt), ("standard", std)):
            m, lo, hi = bootstrap_mean_ci(vals.to_numpy(),
                                          clusters=sub.content_id.loc[vals.index].to_numpy(),
                                          seed=seed)
            eff.append(dict(layer=l, region=region, effect=name, mean=m, lo=lo, hi=hi))
    eff = pd.DataFrame(eff)
    eff.to_csv(d / "register_patch_effects.csv", index=False)
    log.info("H: register 2x2 effects:\n" + eff.round(3).to_string(index=False))

    deep = eff[(eff.layer == deep_layer) & (eff.region == "readout")]

    def _row(name):
        r0 = deep[deep.effect == name]
        return (dict(mean=float(r0["mean"].iloc[0]), lo=float(r0.lo.iloc[0]),
                     hi=float(r0.hi.iloc[0])) if len(r0) else None)

    save_json(d / "summary.json", dict(
        layers=layers, deep_layer=deep_layer,
        n_contents=int(piv.content_id.nunique()),
        readout_class=_row("class"),
        readout_content=_row("content"),
        readout_standard_estimator=_row("standard"),
    ))
    ctx.write_manifest("exp9")
