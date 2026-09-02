"""exp10 (Experiment I): where does class transfer live, and is it just the
answer state?

Addresses the two sharpest objections to the 2x2 result:

  (a) "You patched while the model was already answering the belief question
      -- this may be answer-state transplantation, not evaluation-context
      processing."  So the patched region is split four ways:
        task      shared tokens BEFORE the belief question begins
        question  the belief-question tokens, EXCLUDING the final position
        final     the final position only (where the Yes/No logit forms)
        readout   question + final (the exp7 readout span, for continuity)
      If class transfer exists in `task`, context transfers before any answer
      is constructed. If it lives only in `question`/`final`, the honest
      conclusion is a verbal-report/readout phenomenon.

  (b) "Deploy sources may simply be FARTHER from an eval destination, so the
      class effect could be a distance effect."  Every patch records the mean
      L2 distance between source and destination residuals over the patched
      span, so class effects can be checked against (and regressed on)
      per-cell distances.

Both destination classes are run (eval destinations with deploy-class
flipped sources, and deploy destinations with eval-class flipped sources),
giving both patch directions. Sources per destination form the full 2x2
relative to the destination's class:
        samecls_samecont   same class,  same content
        othercls_samecont  other class, same content   (the "paired" patch)
        samecls_diffcont   same class,  different content
        othercls_diffcont  other class, different content (the "standard control")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..interventions import resid_name, tokens_for
from ..model_io import chat_wrap
from ..readouts import belief_from_logits, belief_prompt
from ..stats import bootstrap_mean_ci
from ..utils import log, save_json, timer
from .common import Ctx
from .exp5_positions import token_spans
from .exp7_causal_anatomy import _cache_resid, _common_suffix, discovery_patch_layer, patch_end_range_hook


def _group_by_content(corpus):
    by: dict[str, dict[str, list]] = {}
    for r in corpus:
        if r["split"] != "test" or r["special"] or r.get("trailing"):
            continue  # trailing-cue renderings do not share the task suffix
        by.setdefault(r["content_id"], {"eval": [], "deploy": []})[r["cls"]].append(r)
    return by


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp10")
    seed = int(cfg.data.seed)
    L = model.cfg.n_layers

    import json
    s2 = {}
    p2 = ctx.out / "exp2" / "summary.json"
    if p2.exists():
        s2 = json.loads(p2.read_text())
    deep_layer = int(s2.get("patch_top_layers", [L - 3])[0])
    steer_layer = int(s2.get("steer_layer", L // 2))
    lk = (s2.get("locked") or {}).get("steer_eval") or {}
    locked_layer = int(lk["layer"]) if lk.get("layer") is not None else steer_layer
    disc_layer = discovery_patch_layer(ctx)
    layers = sorted({steer_layer, deep_layer, locked_layer}
                    | ({disc_layer} if disc_layer is not None else set()))
    log.info(f"exp10: layers {layers} (steer={steer_layer}, deep={deep_layer}, "
             f"locked={locked_layer})")

    by = _group_by_content(ctx.corpus())
    common_path = ctx.out.parent / "common_contents.json"
    if common_path.exists():
        contents = [c for c in json.loads(common_path.read_text()).get("common", [])
                    if c in by and len(by[c]["eval"]) >= 2 and len(by[c]["deploy"]) >= 2]
        log.info(f"exp10: using common content list ({len(contents)} contents)")
    else:
        contents = sorted(cid for cid, g in by.items()
                          if len(g["eval"]) >= 2 and len(g["deploy"]) >= 2)
        rng = np.random.default_rng(seed + 21)
        rng.shuffle(contents)
    n = len(contents)
    log.info(f"exp10: {n} contents per destination class "
             f"({len(contents)} usable)")

    rows = []
    with timer("I: directional span 2x2"):
        for dst_cls in ("eval", "deploy"):
            other = "deploy" if dst_cls == "eval" else "eval"
            for i, cid in enumerate(contents[:n]):
                g = by[cid]
                dest = g[dst_cls][0]
                we = belief_prompt(model, dest["user_text"], "A")
                spans = token_spans(model, we, dest["user_text"], dest["task_text"])
                if spans is None:
                    continue
                og = by[contents[(i + 1) % n]]
                assert contents[(i + 1) % n] != cid, "different-content source must differ"
                srcs = dict(
                    samecls_samecont=g[dst_cls][1],
                    othercls_samecont=g[other][0],
                    samecls_diffcont=og[dst_cls][0],
                    othercls_diffcont=og[other][0],
                )
                ids_e = tokens_for(model, we)
                T = ids_e.shape[1]
                r_ = T - spans["readout"][0]        # question + final, from end
                with torch.no_grad():
                    base = belief_from_logits(model, model(ids_e)[:, -1].float().cpu())[0]
                src_ids = {}
                ok = True
                for name, rec in srcs.items():
                    sids = tokens_for(model, belief_prompt(model, rec["user_text"], "A"))
                    kk = _common_suffix(ids_e, sids)
                    if kk < r_:
                        ok = False
                        break
                    src_ids[name] = (sids, kk)
                if not ok:
                    continue
                for name, (sids, kk) in src_ids.items():
                    cache = _cache_resid(model, sids, layers)
                    dst_cache = _cache_resid(model, ids_e, layers)
                    regions = dict(task=(kk, r_), question=(r_, 1),
                                   final=(1, 0), readout=(r_, 0))
                    for l in layers:
                        for region, (a, b) in regions.items():
                            hk = patch_end_range_hook(cache[l], a, b)
                            with torch.no_grad(), model.hooks(
                                    fwd_hooks=[(resid_name(l), hk)]):
                                lg = model(ids_e)[:, -1].float().cpu()
                            delta = belief_from_logits(model, lg)[0] - base
                            # mean L2 source-vs-destination distance on the span
                            Ts, Td = cache[l].shape[1], dst_cache[l].shape[1]
                            seg_s = cache[l][0, Ts - a: (Ts - b) if b > 0 else Ts]
                            seg_d = dst_cache[l][0, Td - a: (Td - b) if b > 0 else Td]
                            dist = float((seg_s - seg_d).norm(dim=-1).mean())
                            rows.append(dict(dst_cls=dst_cls, layer=l, region=region,
                                             source=name, content_id=cid,
                                             delta=float(delta), dist=dist,
                                             a=a, b=b, kk=kk))
                    del cache, dst_cache
    df = pd.DataFrame(rows)
    df.to_csv(d / "span_direction_patch.csv", index=False)

    # ---- factorial analysis per (dst_cls, layer, region):
    # class / content main effects + interaction, all with clustered CIs ----
    piv = df.pivot_table(index=["dst_cls", "layer", "region", "content_id"],
                         columns="source", values="delta").reset_index()
    eff = []
    need = {"samecls_samecont", "othercls_samecont", "samecls_diffcont",
            "othercls_diffcont"}
    for (dc, l, region), sub in piv.groupby(["dst_cls", "layer", "region"]):
        if not need <= set(sub.columns):
            continue
        cls_ = ((sub.othercls_samecont + sub.othercls_diffcont) / 2
                - (sub.samecls_samecont + sub.samecls_diffcont) / 2).dropna()
        cnt = ((sub.samecls_diffcont + sub.othercls_diffcont) / 2
               - (sub.samecls_samecont + sub.othercls_samecont) / 2).dropna()
        # interaction halved so it is on the same scale as the main effects
        # (both main effects are averages of two simple effects)
        ixn = (((sub.othercls_samecont - sub.samecls_samecont)
                - (sub.othercls_diffcont - sub.samecls_diffcont)) / 2).dropna()
        std = (sub.othercls_samecont - sub.othercls_diffcont).dropna()
        for name, vals in (("class", cls_), ("content", cnt),
                           ("interaction", ixn), ("standard", std)):
            if not len(vals):
                continue
            m, lo, hi = bootstrap_mean_ci(
                vals.to_numpy(),
                clusters=sub.content_id.loc[vals.index].to_numpy(), seed=seed)
            eff.append(dict(dst_cls=dc, layer=l, region=region, effect=name,
                            mean=m, lo=lo, hi=hi,
                            n_contents=int(sub.content_id.nunique())))
    eff = pd.DataFrame(eff)
    eff.to_csv(d / "span_direction_effects.csv", index=False)

    # ---- distance table per cell (for the distance-matching check) ----
    dist = (df.groupby(["dst_cls", "layer", "region", "source"])
            .dist.agg(["mean", "std"]).reset_index())
    dist.to_csv(d / "cell_distances.csv", index=False)

    # ---- summary: the decisive cells at the deep layer ----
    def _get(dc, region, effect):
        r0 = eff[(eff.dst_cls == dc) & (eff.layer == deep_layer)
                 & (eff.region == region) & (eff.effect == effect)]
        return (dict(mean=float(r0["mean"].iloc[0]), lo=float(r0.lo.iloc[0]),
                     hi=float(r0.hi.iloc[0])) if len(r0) else None)

    summary = dict(layers=layers, deep_layer=deep_layer, locked_layer=locked_layer,
                   n_contents_per_dst=n)
    for dc in ("eval", "deploy"):
        for region in ("task", "question", "final", "readout"):
            for effect in ("class", "content", "interaction", "standard"):
                v = _get(dc, region, effect)
                if v:
                    summary[f"{dc}_{region}_{effect}"] = v
    save_json(d / "summary.json", summary)
    for dc in ("eval", "deploy"):
        parts = []
        for region in ("task", "question", "final"):
            v = summary.get(f"{dc}_{region}_class")
            if v:
                parts.append(f"{region} {v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]")
        log.info(f"I: dst={dc} class effect by region: " + "; ".join(parts))
    ctx.write_manifest("exp10")
