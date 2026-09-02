"""exp7 (Experiment F): anatomy of the readout-span artifact, and its remedy.

Three parts:

  F1  Span-resolved patching. The full-suffix patch confounds two regions:
      the shared TASK span (where eval/deploy content could genuinely
      transfer) and the shared READOUT span (the belief question + template
      tail, where the answer is composed). Patch each region separately,
      each with paired and unpaired sources. Predictions from the artifact
      account: readout-only reproduces the full-suffix effect for BOTH
      sources; task-only has a small unpaired effect, so its
      paired-minus-unpaired difference is an artifact-resistant estimate of
      content transfer -- the "remedy" version of activation patching.

  F2  Artifact scaling (no new forwards). From exp2's per-pair records:
      how does the unpaired effect grow with depth and with the patched
      span length? Gives the diagnostic statistic a practitioner can
      compute before trusting any suffix patch.

  F3  Multi-layer ablation. Single-layer d-ablation cannot exclude
      cross-layer redundancy (exp6/E4's stated limitation). Project d_l out
      at EVERY layer simultaneously, at cumulative prefixes [0..L], and at
      cumulative suffixes [L..end], with per-layer random-direction
      controls, and measure how much of the eval/deploy belief separation
      survives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .. import plotting as P
from ..attribution import _belief_metric
from ..data.build import make_pairs
from ..interventions import ablate_hook, random_direction, resid_name, tokens_for
from ..model_io import chat_wrap
from ..readouts import belief_from_logits, belief_prompt
from ..stats import bootstrap_mean_ci, spearman
from ..utils import log, save_json, timer
from .common import Ctx
from .exp5_positions import token_spans


# --------------------------------------------------------------------------- #
# end-aligned range patching
# --------------------------------------------------------------------------- #
def patch_end_range_hook(src_resid: torch.Tensor, a: int, b: int):
    """Patch positions [-a, -b) (counted from the end; a > b >= 0) with the
    source run's same end-aligned positions. Generalizes patch_last_k_hook."""
    Ts = src_resid.shape[1]
    seg = src_resid[:, Ts - a: (Ts - b) if b > 0 else Ts, :].clone()

    def hook(resid, **_kw):
        T = resid.shape[1]
        if T < a or T <= 1:  # KV-cached generation step or shorter prompt
            return resid
        resid[:, T - a: (T - b) if b > 0 else T, :] = seg.to(
            device=resid.device, dtype=resid.dtype)
        return resid

    return hook


def _cache_resid(model, ids, layers):
    want = {resid_name(l) for l in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(ids, names_filter=lambda n: n in want)
    return {l: cache[resid_name(l)].float().cpu() for l in layers}


def _common_suffix(ids_a, ids_b) -> int:
    la, lb = ids_a.shape[1], ids_b.shape[1]
    k = 0
    while k < min(la, lb) - 1 and ids_a[0, la - 1 - k].item() == ids_b[0, lb - 1 - k].item():
        k += 1
    return k


def discovery_patch_layer(ctx: Ctx) -> int | None:
    """Layer with the largest deploy->eval patch effect on the DISCOVERY half
    of the exp2 contents (even-indexed sorted content ids; the same split
    exp2 uses for its locked estimates). Running the span/2x2 experiments at
    this layer is what makes a properly locked class estimate possible."""
    p = ctx.out / "exp2" / "patch_belief.csv"
    if not p.exists():
        return None
    pb = pd.read_csv(p)
    uniq = sorted(set(pb.content_id))
    disc = set(uniq[::2])
    de = pb[(pb.direction == "deploy_to_eval") & pb.content_id.isin(disc)]
    if not len(de):
        return None
    return int(de.groupby("layer").delta.mean().abs().idxmax())


def _f3_summary(ml: pd.DataFrame, seed: int) -> dict:
    """Robust per-condition stats for the multi-layer ablation.

    The naive `1 - abl/base` ratio of means is ill-conditioned: per-pair
    separations are heterogeneous (sd comparable to the mean), so bootstrap
    ratios explode. Report instead (a) the paired per-pair change in
    separation, mean + clustered CI, in logits; and (b) the AUC of the
    per-pair separations against zero before and after -- a rank statistic
    that asks "are eval prompts still scored more 'eval' than their deploy
    twins?"
    """
    out = {}
    for cname, sub in ml.groupby("condition"):
        d_ = sub.abl_sep - sub.base_sep
        m, lo, hi = bootstrap_mean_ci(d_.to_numpy(),
                                      clusters=sub.content_id.to_numpy(), seed=seed)
        out[str(cname)] = dict(
            base_sep=float(sub.base_sep.mean()),
            abl_sep=float(sub.abl_sep.mean()),
            dsep_mean=float(m), dsep_lo=float(lo), dsep_hi=float(hi),
            auc_base=float((sub.base_sep > 0).mean()),
            auc_abl=float((sub.abl_sep > 0).mean()),
            n=int(len(sub)),
        )
    return out


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp7")
    dirs, sigma, s1 = ctx.directions()
    seed = int(cfg.data.seed)
    L = model.cfg.n_layers

    import json
    s2 = {}
    p2 = ctx.out / "exp2" / "summary.json"
    if p2.exists():
        s2 = json.loads(p2.read_text())
    deep_layer = int(s2.get("patch_top_layers", [L - 3])[0])
    net_layer = int(s2.get("net_layer", max(1, L // 4)))
    steer_layer = int(s2.get("steer_layer", L // 2))
    # the discovery-half-selected steering layer (exp2 locked split), so a
    # locked-layer estimate exists alongside the exploratory max-effect layer
    lk = (s2.get("locked") or {}).get("steer_eval") or {}
    locked_layer = int(lk["layer"]) if lk.get("layer") is not None else steer_layer
    disc_layer = discovery_patch_layer(ctx)
    span_layers = sorted({net_layer, steer_layer, deep_layer, locked_layer}
                         | ({disc_layer} if disc_layer is not None else set()))
    log.info(f"exp7: span-patch layers {span_layers} "
             f"(net={net_layer}, steer={steer_layer}, deep={deep_layer}, "
             f"locked={locked_layer}, discovery-patch={disc_layer})")

    # Fixed content list shared by ALL models (results/common_contents.json,
    # built from tokenizer-only checks before any effect was looked at), so
    # cross-model comparisons use identical content clusters. Falls back to
    # make_pairs if the file is absent.
    common_path = ctx.out.parent / "common_contents.json"
    common = None
    if common_path.exists():
        common = json.loads(common_path.read_text()).get("common")
    corpus_all = ctx.corpus()
    if common:
        byc: dict[str, dict[str, list]] = {}
        for r_ in corpus_all:
            if r_["split"] == "test" and not r_["special"]:
                byc.setdefault(r_["content_id"], {"eval": [], "deploy": []})[r_["cls"]].append(r_)
        pairs = [(byc[c]["eval"][0], byc[c]["deploy"][0]) for c in common if c in byc]
        n = len(pairs)
        log.info(f"exp7: using common content list ({n} contents)")
    else:
        n = min(int(cfg.interventions.n_belief_pairs), 24)
        pairs = make_pairs(corpus_all, 3 * n, seed=seed + 11, split="test")

    # ---------------- F1: span-resolved patching ----------------
    # FOUR sources per destination prompt -- a 2x2 of (source class x content):
    #   sameclass      EVAL,   same content   (both matched)
    #   paired         DEPLOY, same content   (class flipped)
    #   unpaired_eval  EVAL,   diff content   (content flipped)
    #   unpaired       DEPLOY, diff content   (both flipped)
    # The standard "unpaired control" is the both-flipped cell: it carries the
    # source CLASS, so matching it against the paired patch cancels class
    # transfer along with disruption. The class main effect is
    # (paired + unpaired)/2 - (sameclass + unpaired_eval)/2; the content main
    # effect is the transpose. paired - sameclass remains the same-content
    # class contrast.
    by_content: dict[str, list] = {}
    for rec in ctx.corpus():
        if (rec["split"] == "test" and not rec["special"] and rec["cls"] == "eval"
                and not rec.get("trailing")):
            by_content.setdefault(rec["content_id"], []).append(rec)

    rows = []
    used = 0
    with timer("F1 span-resolved patching"):
        for i, (e, p) in enumerate(pairs):
            if used >= n:
                break
            we = belief_prompt(model, e["user_text"], "A")
            wp = belief_prompt(model, p["user_text"], "A")
            spans = token_spans(model, we, e["user_text"], e["task_text"])
            if spans is None:
                continue
            alt = [r_ for r_ in by_content.get(e["content_id"], [])
                   if r_["pid"] != e["pid"]]
            if not alt:
                continue
            ids_e, ids_p = tokens_for(model, we), tokens_for(model, wp)
            ids_s = tokens_for(model, belief_prompt(model, alt[0]["user_text"], "A"))
            q_dep = pairs[(i + 1) % len(pairs)][1]
            q_ev = pairs[(i + 1) % len(pairs)][0]
            assert q_dep["content_id"] != e["content_id"], "unpaired source must differ in content"
            ids_u = tokens_for(model, belief_prompt(model, q_dep["user_text"], "A"))
            ids_ue = tokens_for(model, belief_prompt(model, q_ev["user_text"], "A"))
            k = _common_suffix(ids_e, ids_p)
            ks = _common_suffix(ids_e, ids_s)
            ku = _common_suffix(ids_e, ids_u)
            kue = _common_suffix(ids_e, ids_ue)
            T = ids_e.shape[1]
            r = T - spans["readout"][0]          # readout span length from end
            # the readout span must lie fully inside every source's shared
            # suffix; no extra margin (an earlier +2 margin excluded contents
            # on a tokenizer-dependent basis -- see results/attrition.json)
            if min(k, ks, ku, kue) < r:
                continue
            with torch.no_grad():
                base = belief_from_logits(model, model(ids_e)[:, -1].float().cpu())[0]
            cp = _cache_resid(model, ids_p, span_layers)
            cs = _cache_resid(model, ids_s, span_layers)
            cu = _cache_resid(model, ids_u, span_layers)
            cue_ = _cache_resid(model, ids_ue, span_layers)
            for l in span_layers:
                for src_name, cache, kk in (("paired", cp, k),
                                            ("sameclass", cs, ks),
                                            ("unpaired", cu, ku),
                                            ("unpaired_eval", cue_, kue)):
                    for region, (a, b) in dict(full=(kk, 0), readout=(r, 0),
                                               task=(kk, r)).items():
                        hk = patch_end_range_hook(cache[l], a, b)
                        with torch.no_grad(), model.hooks(
                                fwd_hooks=[(resid_name(l), hk)]):
                            lg = model(ids_e)[:, -1].float().cpu()
                        delta = belief_from_logits(model, lg)[0] - base
                        rows.append(dict(layer=l, region=region, source=src_name,
                                         content_id=e["content_id"], pair=i,
                                         delta=float(delta), a=a, b=b))
            del cp, cs, cu, cue_
            used += 1
    span_df = pd.DataFrame(rows)
    span_df.to_csv(d / "span_patch.csv", index=False)
    log.info(f"F1: {used} pairs")

    agg = []
    for (l, region, src), sub in span_df.groupby(["layer", "region", "source"]):
        m, lo, hi = bootstrap_mean_ci(sub.delta, clusters=sub.content_id, seed=seed)
        agg.append(dict(layer=l, region=region, source=src, mean=m, lo=lo, hi=hi))
    agg = pd.DataFrame(agg)
    agg.to_csv(d / "span_patch_agg.csv", index=False)

    # content estimate: paired - unpaired, per (layer, region), paired per pair
    piv = span_df.pivot_table(index=["layer", "region", "pair", "content_id"],
                              columns="source", values="delta").reset_index()
    # same-content CLASS contrast: deploy-class source minus eval-class source
    # of the SAME content (paired - sameclass). Never fall back to
    # paired - unpaired: that contrast holds class fixed and cannot estimate it.
    if "sameclass" not in piv:
        raise RuntimeError("exp7: sameclass cell missing; the same-content class "
                           "contrast cannot be computed")
    piv["class_samecontent"] = piv.paired - piv.sameclass
    cagg = []
    for (l, region), sub in piv.groupby(["layer", "region"]):
        m, lo, hi = bootstrap_mean_ci(sub.class_samecontent, clusters=sub.content_id, seed=seed)
        cagg.append(dict(layer=l, region=region, mean=m, lo=lo, hi=hi))
    cagg = pd.DataFrame(cagg)
    cagg.to_csv(d / "span_patch_class_samecontent.csv", index=False)

    # class and content main effects of the source 2x2, per (layer, region)
    if "unpaired_eval" in span_df.source.unique():
        eff = []
        for (l, region), sub in piv.groupby(["layer", "region"]):
            if not {"paired", "sameclass", "unpaired", "unpaired_eval"} <= set(sub.columns):
                continue
            cls = ((sub.paired + sub.unpaired) / 2
                   - (sub.sameclass + sub.unpaired_eval) / 2).dropna()
            cnt = ((sub.unpaired + sub.unpaired_eval) / 2
                   - (sub.paired + sub.sameclass) / 2).dropna()
            cids_ = sub.content_id
            for name, vals in (("class", cls), ("content", cnt)):
                m, lo, hi = bootstrap_mean_ci(vals.to_numpy(),
                                              clusters=cids_.loc[vals.index].to_numpy(),
                                              seed=seed)
                eff.append(dict(layer=l, region=region, effect=name,
                                mean=m, lo=lo, hi=hi))
        eff = pd.DataFrame(eff)
        eff.to_csv(d / "span_patch_effects.csv", index=False)
        log.info("F1 source 2x2 main effects:\n" + eff.round(3).to_string(index=False))
    log.info("F1 class-specific transfer (paired - sameclass):\n" + cagg.to_string(index=False))

    # ---------------- F2: artifact scaling from exp2 records ----------------
    f2 = {}
    pb = ctx.out / "exp2" / "patch_belief.csv"
    if pb.exists():
        rec = pd.read_csv(pb)
        un = rec[rec.direction == "unpaired_control"].copy()
        per_layer = un.groupby("layer").delta.apply(lambda x: x.abs().mean())
        f2["depth_spearman"] = spearman(per_layer.index.to_numpy(),
                                        per_layer.to_numpy())
        deep = un[un.layer >= int(2 * L / 3)]
        if deep.k.nunique() > 3:
            f2["span_spearman_deep"] = spearman(deep.k.to_numpy(),
                                                deep.delta.abs().to_numpy())
        per_layer.rename("mean_abs_unpaired").to_csv(d / "artifact_by_depth.csv")
    log.info(f"F2 artifact scaling: {f2}")

    # ---------------- F3: multi-layer ablation ----------------
    ml_pairs = make_pairs(ctx.corpus(), n, seed=seed + 12, split="test")
    texts = [(belief_prompt(model, e["user_text"], "A"),
              belief_prompt(model, p["user_text"], "A"),
              e["content_id"]) for e, p in ml_pairs]
    rand = {l: random_direction(model.cfg.d_model, seed=seed + 500 + l)
            for l in range(L)}
    stride = max(2, L // 7)
    cuts = list(range(stride, L, stride))
    conditions = [("all_d", list(range(L)), "d"), ("all_rand", list(range(L)), "rand")]
    conditions += [(f"prefix_{c}", list(range(0, c)), "d") for c in cuts]
    conditions += [(f"suffix_{c}", list(range(c, L)), "d") for c in cuts]

    mrows = []
    with timer("F3 multi-layer ablation"):
        for we, wp, cid in texts:
            ids_e, ids_p = tokens_for(model, we), tokens_for(model, wp)
            with torch.no_grad():
                sep0 = (_belief_metric(model, model(ids_e)).item()
                        - _belief_metric(model, model(ids_p)).item())
            for cname, layers_, kind in conditions:
                hooks = [(resid_name(l),
                          ablate_hook(dirs[l] if kind == "d" else rand[l]))
                         for l in layers_]
                with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                    sep1 = (_belief_metric(model, model(ids_e)).item()
                            - _belief_metric(model, model(ids_p)).item())
                mrows.append(dict(condition=cname, kind=kind,
                                  n_layers=len(layers_), content_id=cid,
                                  base_sep=sep0, abl_sep=sep1))
    ml = pd.DataFrame(mrows)
    ml.to_csv(d / "multilayer_ablation.csv", index=False)

    msum = _f3_summary(ml, seed)
    ad, ar = msum["all_d"], msum["all_rand"]
    log.info(f"F3 (robust): base AUC {ad['auc_base']:.3f}; all-layer d -> "
             f"AUC {ad['auc_abl']:.3f}, dsep {ad['dsep_mean']:+.3f} "
             f"[{ad['dsep_lo']:+.3f},{ad['dsep_hi']:+.3f}]; all-layer random -> "
             f"AUC {ar['auc_abl']:.3f}, dsep {ar['dsep_mean']:+.3f} "
             f"[{ar['dsep_lo']:+.3f},{ar['dsep_hi']:+.3f}]")

    # ---------------- figures & summary ----------------
    try:
        deep = agg[agg.layer == deep_layer]
        labels, means, los, his = [], [], [], []
        for region in ("full", "task", "readout"):
            for src in ("paired", "unpaired"):
                row = deep[(deep.region == region) & (deep.source == src)]
                if len(row):
                    labels.append(f"{region}/{src}")
                    means.append(row["mean"].iloc[0])
                    los.append(row.lo.iloc[0])
                    his.append(row.hi.iloc[0])
        P.save(P.bars_ci(labels, means, los, his,
                         xlabel=r"$\Delta$ belief",
                         title=f"F1: where the patch effect lives (layer {deep_layer})"),
               d / "fig_span_patch", agg)
    except Exception as exc:  # figure is non-critical
        log.info(f"fig_span_patch skipped: {exc}")

    save_json(d / "summary.json", dict(
        span_layers=span_layers,
        deep_layer=deep_layer,
        locked_layer=locked_layer,
        common_contents=bool(common),
        n_pairs_f1=used,
        artifact_scaling=f2,
        multilayer=msum,
    ))
    ctx.write_manifest("exp7")
