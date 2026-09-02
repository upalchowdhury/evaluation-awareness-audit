"""exp11 (Experiment J): the distance-matched source 2x2.

exp10 answered the distance objection observationally: the inert
same-class/different-content source happens to lie FARTHER from the
destination than the potent other-class/same-content source, so distance
cannot explain the class ordering.  A reviewer can still argue that
activation distance, geometry, or another source-destination property
drives the apparent class effect.  This experiment removes the objection
by construction rather than by observation.

Design.  Same destinations and factorial estimator as exp10 (class x
content source 2x2, both destination classes, content-clustered bootstrap
CIs), re-estimated under four arms that handle residual distance
differently:

  unmatched        first admitted rendering per cell (exp10's sources
                   wherever they pass the suffix filter), spans equalized
                   across cells.  Baseline / bridge to exp10.
  selected         each cell offers up to K=6 candidate renderings per
                   destination; one candidate per cell is chosen to
                   MINIMIZE the |class-condition mean distance gap|
                   (the quantity the class main effect averages over),
                   with the four-cell distance range as tiebreak.  Real
                   activations only; distance matched by selection.
  constructed      each source's per-position residual offset (src - dst)
                   is rescaled to the four-cell MEAN norm at that
                   position: all four cells sit at exactly the same L2
                   distance from the destination at every patched
                   position; only the offset DIRECTION differs by cell.
  constructed_min  same, but rescaled to the four-cell per-position MIN
                   norm: every constructed vector is a convex
                   interpolation dst + g*(src-dst) with g <= 1 -- no
                   offset is ever amplified, so rescaling can only
                   ATTENUATE whatever the offset carries.  Any surviving
                   class effect is a lower bound; this is the
                   conservative primary arm.

PRE-SPECIFIED DECISION RULE (committed before any full-scale run of this
file): the primary confirmatory cells are the class main effect in the
constructed_min arm, FINAL region, at the exp2 maximum-patch-effect layer
(the layer the paper's class-effect tables use), for BOTH destination
classes; success = both CIs exclude zero with exp10's signs.  The
constructed (mean-target) and selected arms, the locked-layer variant,
and the task/readout regions are robustness/descriptive rows.  Everything
else in matched_effects.csv is reported as exploratory.

Honesty instrumentation, per adversarial review of the first draft:
per-row per-position rescale extremes (scale_min/scale_max, clamp count);
achieved distances measured AFTER casting the patched segment to the
model dtype (so bf16 models report the real, not asserted, balance);
a unit-normalized (direction-metric) distance column, since RMSNorm makes
the network partly radial-invariant; per-row source pid + shared-suffix
length for audit; a rescale-footprint table (unmatched minus constructed
delta per cell -- the empirical bound on off-manifold distortion);
per-destination-class attrition counts.

Estimand note for the paper: the constructed arms estimate the effect of
the class-associated offset DIRECTION at standardized per-position L2
dose -- exact selection-based balance among real activations is
geometrically impossible if class is linearly encoded (the other-class
cell has a distance floor), which is why construction is primary and
selection is the on-manifold check.
"""
from __future__ import annotations

from itertools import product

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
from .exp7_causal_anatomy import _cache_resid, _common_suffix, discovery_patch_layer

CELLS = ("samecls_samecont", "othercls_samecont",
         "samecls_diffcont", "othercls_diffcont")
OTHER_IDX, SAME_IDX = (1, 3), (0, 2)
ARMS = ("unmatched", "selected", "constructed", "constructed_min")
K_CAND = 6
EPS = 1e-6


def patch_seg_hook(seg: torch.Tensor, a: int, b: int):
    """Write a precomputed [1, a-b, d] segment into positions [-a, -b)."""
    def hook(resid, **_kw):
        T = resid.shape[1]
        if T < a or T <= 1:  # KV-cached generation step or shorter prompt
            return resid
        resid[:, T - a: (T - b) if b > 0 else T, :] = seg.to(
            device=resid.device, dtype=resid.dtype)
        return resid
    return hook


def _end_slice(t: torch.Tensor, a: int, b: int) -> torch.Tensor:
    T = t.shape[1]
    return t[0, T - a: (T - b) if b > 0 else T]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def _group_by_content(corpus):
    by: dict[str, dict[str, list]] = {}
    for r in corpus:
        if r["split"] != "test" or r["special"] or r.get("trailing"):
            continue  # trailing-cue renderings do not share the task suffix
        by.setdefault(r["content_id"], {"eval": [], "deploy": []})[r["cls"]].append(r)
    return by


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp11")
    seed = int(cfg.data.seed)
    L = model.cfg.n_layers
    mdt = next(model.parameters()).dtype

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
    log.info(f"exp11: layers {layers} (steer={steer_layer}, deep={deep_layer}, "
             f"locked={locked_layer}), model dtype {mdt}")

    by = _group_by_content(ctx.corpus())
    common_path = ctx.out.parent / "common_contents.json"
    if common_path.exists():
        contents = [c for c in json.loads(common_path.read_text()).get("common", [])
                    if c in by and len(by[c]["eval"]) >= 2 and len(by[c]["deploy"]) >= 2]
        log.info(f"exp11: using common content list ({len(contents)} contents)")
    else:
        contents = sorted(cid for cid, g in by.items()
                          if len(g["eval"]) >= 2 and len(g["deploy"]) >= 2)
        rng = np.random.default_rng(seed + 21)  # exp10 parity in the fallback path
        rng.shuffle(contents)
    n = len(contents)

    rows = []
    attrition = dict(no_spans=0, empty_cell=0, no_task_span=0,
                     used_eval=0, used_deploy=0)
    with timer("J: distance-matched 2x2"):
        for dst_cls in ("eval", "deploy"):
            other = "deploy" if dst_cls == "eval" else "eval"
            for i, cid in enumerate(contents):
                g = by[cid]
                og = by[contents[(i + 1) % n]]
                dest = g[dst_cls][0]
                we = belief_prompt(model, dest["user_text"], "A")
                spans = token_spans(model, we, dest["user_text"], dest["task_text"])
                if spans is None:
                    attrition["no_spans"] += 1
                    continue
                ids_e = tokens_for(model, we)
                T = ids_e.shape[1]
                r_ = T - spans["readout"][0]        # question + final, from end

                pools = dict(
                    samecls_samecont=[x for x in g[dst_cls] if x["pid"] != dest["pid"]],
                    othercls_samecont=list(g[other]),
                    samecls_diffcont=list(og[dst_cls]),
                    othercls_diffcont=list(og[other]),
                )
                # admit candidates whose shared suffix covers the readout
                # span (filter first, cap at K_CAND after)
                admitted: dict[str, list] = {}
                ok = True
                for cell in CELLS:
                    cand = []
                    for rec in pools[cell]:
                        sids = tokens_for(
                            model, belief_prompt(model, rec["user_text"], "A"))
                        kk = _common_suffix(ids_e, sids)
                        if kk >= r_:
                            cand.append((rec, sids, kk))
                            if len(cand) == K_CAND:
                                break
                    if not cand:
                        ok = False
                        break
                    admitted[cell] = cand
                if not ok:
                    attrition["empty_cell"] += 1
                    continue

                a_task = min(kk for cand in admitted.values() for _, _, kk in cand)
                # regression guard: every admitted candidate must share the
                # last a_task tokens with the destination verbatim
                for cand in admitted.values():
                    for _, sids, _kk in cand:
                        assert torch.equal(sids[0, -a_task:], ids_e[0, -a_task:])
                regions = dict(final=(1, 0), readout=(r_, 0))
                if a_task > r_:
                    regions["task"] = (a_task, r_)
                else:
                    attrition["no_task_span"] += 1

                with torch.no_grad():
                    base = belief_from_logits(model, model(ids_e)[:, -1].float().cpu())[0]
                dst_cache = _cache_resid(model, ids_e, layers)
                caches = {cell: [(_cache_resid(model, sids, layers), kk, rec["pid"])
                                 for rec, sids, kk in cand]
                          for cell, cand in admitted.items()}

                # distances per (cell, candidate, layer, region)
                dist = {}
                for cell, lst in caches.items():
                    for j, (cache, _kk, _pid) in enumerate(lst):
                        for l in layers:
                            for region, (a, b) in regions.items():
                                seg_s = _end_slice(cache[l], a, b)
                                seg_d = _end_slice(dst_cache[l], a, b)
                                dist[(cell, j, l, region)] = float(
                                    (seg_s - seg_d).norm(dim=-1).mean())

                def _patch(cell, j, l, region, arm, construct_target=None):
                    a, b = regions[region]
                    cache, kk, pid = caches[cell][j]
                    seg_s = _end_slice(cache[l], a, b).clone()
                    seg_d = _end_slice(dst_cache[l], a, b)
                    scale = scale_min = scale_max = 1.0
                    n_clamped = 0
                    if construct_target is not None:
                        off = seg_s - seg_d
                        norms = off.norm(dim=-1, keepdim=True)
                        n_clamped = int((norms <= EPS).sum())
                        ratio = construct_target / norms.clamp_min(EPS)
                        seg_s = seg_d + off * ratio
                        scale = float(ratio.mean())
                        scale_min = float(ratio.min())
                        scale_max = float(ratio.max())
                    # measure the ACHIEVED distance after casting to the model
                    # dtype (bf16 quantization must be measured, not asserted)
                    seg_cast = seg_s.to(mdt).float()
                    hk = patch_seg_hook(seg_cast[None], a, b)
                    with torch.no_grad(), model.hooks(fwd_hooks=[(resid_name(l), hk)]):
                        lg = model(ids_e)[:, -1].float().cpu()
                    delta = belief_from_logits(model, lg)[0] - base
                    dd = float((seg_cast - seg_d).norm(dim=-1).mean())
                    du = float((_unit(seg_cast) - _unit(seg_d)).norm(dim=-1).mean())
                    rows.append(dict(arm=arm, dst_cls=dst_cls, layer=l,
                                     region=region, source=cell, content_id=cid,
                                     delta=float(delta), dist=dd, dist_unit=du,
                                     scale=scale, scale_min=scale_min,
                                     scale_max=scale_max, n_clamped=n_clamped,
                                     pid=pid, kk=kk, a=a, b=b, cand=j))

                for l in layers:
                    for region in regions:
                        # arm 1: unmatched (first admitted candidate per cell)
                        for cell in CELLS:
                            _patch(cell, 0, l, region, "unmatched")
                        # arm 2: selected -- lexicographic: |class-condition
                        # mean distance gap| first, four-cell range tiebreak
                        ks = [len(caches[c]) for c in CELLS]
                        best, best_key = None, (np.inf, np.inf)
                        for combo in product(*(range(k) for k in ks)):
                            ds = [dist[(c, j, l, region)]
                                  for c, j in zip(CELLS, combo)]
                            gap = abs(sum(ds[i] for i in OTHER_IDX) / 2
                                      - sum(ds[i] for i in SAME_IDX) / 2)
                            key = (gap, max(ds) - min(ds))
                            if key < best_key:
                                best, best_key = combo, key
                        for cell, j in zip(CELLS, best):
                            _patch(cell, j, l, region, "selected")
                        # arms 3+4: per-position norm equalization over the
                        # same first candidates as 'unmatched'
                        a, b = regions[region]
                        offs = [(_end_slice(caches[c][0][0][l], a, b)
                                 - _end_slice(dst_cache[l], a, b)) for c in CELLS]
                        nstack = torch.stack([o.norm(dim=-1) for o in offs])
                        t_mean = nstack.mean(0)[:, None]
                        t_min = nstack.min(0).values[:, None]
                        for cell in CELLS:
                            _patch(cell, 0, l, region, "constructed",
                                   construct_target=t_mean)
                            _patch(cell, 0, l, region, "constructed_min",
                                   construct_target=t_min)
                del caches, dst_cache
                attrition[f"used_{dst_cls}"] += 1

    df = pd.DataFrame(rows)
    df.to_csv(d / "matched_patch.csv", index=False)
    log.info(f"exp11: attrition {attrition}")
    if not len(df):
        save_json(d / "summary.json", dict(layers=layers, n_contents=n,
                                           attrition=attrition))
        log.info("exp11: no usable destinations")
        ctx.write_manifest("exp11")
        return

    # ---- factorial effects per (arm, dst_cls, layer, region) ----
    piv = df.pivot_table(index=["arm", "dst_cls", "layer", "region", "content_id"],
                         columns="source", values="delta").reset_index()
    need = set(CELLS)
    eff = []
    for (arm, dc, l, region), sub in piv.groupby(["arm", "dst_cls", "layer", "region"]):
        if not need <= set(sub.columns):
            continue
        cls_ = ((sub.othercls_samecont + sub.othercls_diffcont) / 2
                - (sub.samecls_samecont + sub.samecls_diffcont) / 2).dropna()
        cnt = ((sub.samecls_diffcont + sub.othercls_diffcont) / 2
               - (sub.samecls_samecont + sub.othercls_samecont) / 2).dropna()
        # interaction halved so it is on the same scale as the main effects
        ixn = (((sub.othercls_samecont - sub.samecls_samecont)
                - (sub.othercls_diffcont - sub.samecls_diffcont)) / 2).dropna()
        for name, vals in (("class", cls_), ("content", cnt), ("interaction", ixn)):
            if not len(vals):
                continue
            m, lo, hi = bootstrap_mean_ci(
                vals.to_numpy(),
                clusters=sub.content_id.loc[vals.index].to_numpy(), seed=seed)
            eff.append(dict(arm=arm, dst_cls=dc, layer=l, region=region,
                            effect=name, mean=m, lo=lo, hi=hi,
                            n_contents=int(sub.content_id.nunique())))
    eff = pd.DataFrame(eff)
    eff.to_csv(d / "matched_effects.csv", index=False)

    # ---- balance: achieved class-condition distance gap per arm, in the
    # raw L2 metric and the unit-direction metric ----
    bal = []
    for metric in ("dist", "dist_unit"):
        dpiv = df.pivot_table(index=["arm", "dst_cls", "layer", "region", "content_id"],
                              columns="source", values=metric).reset_index()
        for (arm, dc, l, region), sub in dpiv.groupby(
                ["arm", "dst_cls", "layer", "region"]):
            if not need <= set(sub.columns):
                continue
            gap = ((sub.othercls_samecont + sub.othercls_diffcont) / 2
                   - (sub.samecls_samecont + sub.samecls_diffcont) / 2).dropna()
            m, lo, hi = bootstrap_mean_ci(
                gap.to_numpy(),
                clusters=sub.content_id.loc[gap.index].to_numpy(), seed=seed)
            row = dict(metric=metric, arm=arm, dst_cls=dc, layer=l, region=region,
                       gap_mean=m, gap_lo=lo, gap_hi=hi)
            for cell in CELLS:
                row[f"dist_{cell}"] = float(sub[cell].mean())
            bal.append(row)
    bal = pd.DataFrame(bal)
    bal.to_csv(d / "balance.csv", index=False)

    # ---- rescale footprint: unmatched minus constructed, per cell ----
    # (same candidate, same spans; the empirical bound on what rescaling does)
    fpv = df[df.arm.isin(("unmatched", "constructed", "constructed_min"))].pivot_table(
        index=["dst_cls", "layer", "region", "content_id", "source"],
        columns="arm", values="delta").reset_index()
    fp = []
    for arm in ("constructed", "constructed_min"):
        if arm not in fpv.columns:
            continue
        diff = (fpv["unmatched"] - fpv[arm]).dropna()
        for (dc, l, region, src), sub_idx in fpv.loc[diff.index].groupby(
                ["dst_cls", "layer", "region", "source"]).groups.items():
            vals = diff.loc[sub_idx]
            m, lo, hi = bootstrap_mean_ci(
                vals.to_numpy(),
                clusters=fpv.content_id.loc[sub_idx].to_numpy(), seed=seed)
            fp.append(dict(arm=arm, dst_cls=dc, layer=l, region=region,
                           source=src, mean=m, lo=lo, hi=hi))
    fp = pd.DataFrame(fp)
    fp.to_csv(d / "rescale_footprint.csv", index=False)

    # ---- summary ----
    balL2 = bal[bal.metric == "dist"]

    summary = dict(layers=layers, deep_layer=deep_layer, locked_layer=locked_layer,
                   steer_layer=steer_layer, k_cand=K_CAND,
                   n_contents=n, attrition=attrition, model_dtype=str(mdt))
    for arm in ARMS:
        for dc in ("eval", "deploy"):
            for l_name, l in (("deep", deep_layer), ("locked", locked_layer)):
                for region in ("task", "final", "readout"):
                    q = eff[(eff.arm == arm) & (eff.dst_cls == dc)
                            & (eff.layer == l) & (eff.region == region)
                            & (eff.effect == "class")]
                    if len(q):
                        summary[f"{arm}_{dc}_{l_name}_{region}_class"] = dict(
                            mean=float(q["mean"].iloc[0]), lo=float(q.lo.iloc[0]),
                            hi=float(q.hi.iloc[0]),
                            n=int(q.n_contents.iloc[0]))
                q = balL2[(balL2.arm == arm) & (balL2.dst_cls == dc)
                          & (balL2.layer == l) & (balL2.region == "final")]
                if len(q):
                    summary[f"{arm}_{dc}_{l_name}_final_gap"] = dict(
                        mean=float(q.gap_mean.iloc[0]), lo=float(q.gap_lo.iloc[0]),
                        hi=float(q.gap_hi.iloc[0]))
    for arm in ("constructed", "constructed_min"):
        cs = df[df.arm == arm]
        if len(cs):
            summary[f"{arm}_scale"] = dict(
                mean=float(cs.scale.mean()),
                p50=float(cs.scale.median()),
                p95_max=float(cs.scale_max.quantile(0.95)),
                max=float(cs.scale_max.max()),
                n_clamped=int(cs.n_clamped.sum()))
    summary["construct_scale"] = dict(  # consumer compatibility
        mean=summary.get("constructed_scale", {}).get("mean"),
        mean_abs_log=float(np.abs(np.log(
            df[df.arm == "constructed"].scale.clip(lower=EPS))).mean())
        if len(df[df.arm == "constructed"]) else None)
    save_json(d / "summary.json", summary)
    for arm in ARMS:
        for dc in ("eval", "deploy"):
            v = summary.get(f"{arm}_{dc}_deep_final_class")
            gv = summary.get(f"{arm}_{dc}_deep_final_gap")
            if v:
                log.info(f"J: {arm:15s} dst={dc:6s} final class "
                         f"{v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]"
                         + (f"  dist-gap {gv['mean']:+.2f} "
                            f"[{gv['lo']:+.2f},{gv['hi']:+.2f}]" if gv else ""))
    ctx.write_manifest("exp11")
