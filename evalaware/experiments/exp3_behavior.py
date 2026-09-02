"""exp3 (Experiment B): does moving the representation move BEHAVIOR?

Headline: sandbagging. MC capability items under a 2x2 of
(framing: eval/deploy) x (system: incentive-to-underperform / neutral), then:

  steer    +/- alpha sigma d_hat at the top layers   (dose-response on accuracy)
  ablate   project d_hat out                          (necessity test)
  patch    deploy-neutral residuals -> eval-incentive run, per layer
           (the behavior column of the stage table)
  controls random directions and a style direction at matched norm
           (selectivity: is it THIS direction, or any big nudge?)

Also measured under steering: sycophancy, over-refusal (generation),
capability self-report. Belief is recorded on the same framed items so exp4
can correlate representation, report, and behavior per item.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .. import plotting as P
from ..data.behaviors import SANDBAG_SYSTEMS
from ..interventions import (
    ablate_hook,
    cache_resid,
    common_suffix_len,
    patched_final_logits,
    random_direction,
    resid_name,
    steer_hook,
    tokens_for,
)
from ..model_io import chat_wrap
from ..probes import mass_mean_directions
from ..readouts import (
    agree_scores,
    belief_scores,
    letter_probs,
    mc_scores,
    refusal_rates,
    selfreport_ev,
)
from ..stats import bootstrap_mean_ci, cohen_d_paired
from ..utils import log, save_json, timer
from .common import Ctx, sweep_layers

CELLS = [  # (framing, system) - the sandbagging 2x2
    ("eval", "incentive"), ("eval", "neutral"),
    ("deploy", "incentive"), ("deploy", "neutral"),
]


def _steer_peak_layer(ctx: Ctx) -> int | None:
    """Layer where +/-4 sigma steering moves the belief most in BOTH framings.

    Steering has no patch-span confound, so this is the most trustworthy
    single-layer estimate of where d causally controls the belief. It is
    typically NOT exp1's decodability argmax and NOT the raw patch argmax.
    """
    p = ctx.out / "exp2" / "agg_steer.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p)
    piv = a.pivot_table(index="layer", columns="framing", values="mean")
    if not {"eval", "deploy"} <= set(piv.columns):
        return None
    return int((piv["eval"].abs() + piv["deploy"].abs()).idxmax())


def _top_layers(ctx: Ctx) -> list[int]:
    """Layers carried into the behavior experiments.

    Ranked by how *causally specific* the layer is, not by the raw patch
    effect: the raw patch curve is dominated by readout-span overwriting (see
    exp2's unpaired-source control), so its argmax points at layers where
    nothing eval-specific is happening. Order: steering peak, then the layer
    where the paired patch most exceeds the unpaired control, then the raw
    patch ranking, then decodability.
    """
    import json

    _, _, s1 = ctx.directions()
    k = int(ctx.cfg.interventions.top_layers)
    pref: list[int] = []
    sp = _steer_peak_layer(ctx)
    if sp is not None:
        pref.append(sp)
    p2 = ctx.out / "exp2" / "summary.json"
    if p2.exists():
        with open(p2) as f:
            s2 = json.load(f)
        if s2.get("net_layer") is not None:
            pref.append(int(s2["net_layer"]))
        pref += [int(x) for x in s2.get("patch_top_layers", [])]
    pref += [int(x) for x in s1["layers_ranked"]]
    return list(dict.fromkeys(pref))[:k]


def _style_direction(ctx: Ctx, layer: int) -> np.ndarray:
    """Control concept: formal vs casual deployment style, same extraction recipe."""
    X, labels = ctx.acts()
    m = (labels.split == "train") & (labels.cls == "deploy")
    formal = labels.family.isin(["production_metadata", "task_context"])
    rows = m & labels.family.isin(
        ["production_metadata", "task_context", "casual_user", "personal_stakes"])
    dirs, _ = mass_mean_directions(X[rows.to_numpy()], formal[rows].astype(int).to_numpy())
    return dirs[layer]


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp3")
    dirs, sigma, s1 = ctx.directions()
    best_layer = int(s1["best_layer"])
    tops = _top_layers(ctx)
    # Selectivity controls must be run where steering actually does something,
    # otherwise "d is no better than a random direction" is trivially true
    # because nothing moves at that layer. Default to the steering peak.
    ctrl_layer = _steer_peak_layer(ctx) or best_layer
    bs = int(cfg.model.batch_size)
    seed = int(cfg.data.seed)
    suites = ctx.behaviors()
    cap = suites["capability"]
    log.info(f"exp3: top layers {tops}, control layer {ctrl_layer} "
             f"(exp1 argmax {best_layer}), {len(cap)} capability items")

    alphas = [float(a) for a in cfg.interventions.alpha_sweep]
    answers = [it["answer"] for it in cap]
    cap_cids = [it["content_id"] for it in cap]

    def cell_texts(framing: str) -> list[str]:
        return [it[framing] for it in cap]

    # ================= sandbagging baselines (2x2) =================
    base_rows, item_rows = [], []
    with timer("sandbag baselines"):
        for framing, system in CELLS:
            r = mc_scores(model, cell_texts(framing), answers, batch_size=bs,
                          system_msg=SANDBAG_SYSTEMS[system])
            m, lo, hi = bootstrap_mean_ci(r["p_correct"], clusters=cap_cids, seed=seed)
            base_rows.append(dict(framing=framing, system=system, acc=r["acc"],
                                  p_correct=m, lo=lo, hi=hi))
            for it, pc, cor in zip(cap, r["p_correct"], r["correct"]):
                item_rows.append(dict(item_id=it["item_id"], framing=framing,
                                      system=system, p_correct=float(pc),
                                      correct=float(cor)))
    base = pd.DataFrame(base_rows)
    base.to_csv(d / "sandbag_baseline.csv", index=False)
    pd.DataFrame(item_rows).to_csv(d / "sandbag_items.csv", index=False)
    log.info("\n" + base.to_string(index=False))

    # belief on the same framed items (for exp4's per-item dissociation)
    bel_rows = []
    for framing, system in (("eval", "incentive"), ("deploy", "neutral")):
        b = belief_scores(model, cell_texts(framing), batch_size=bs,
                          system_msg=SANDBAG_SYSTEMS[system])
        for it, v in zip(cap, b):
            bel_rows.append(dict(item_id=it["item_id"], framing=framing,
                                 system=system, belief=float(v)))
    pd.DataFrame(bel_rows).to_csv(d / "sandbag_belief.csv", index=False)

    # ================= steering dose-response =================
    steer_rows = []
    with timer("sandbag steering"):
        for layer in tops:
            for framing, system in (("eval", "incentive"), ("deploy", "neutral")):
                for alpha in alphas:
                    hooks = None if alpha == 0.0 else [
                        (resid_name(layer), steer_hook(dirs[layer], alpha * float(sigma[layer])))]
                    r = mc_scores(model, cell_texts(framing), answers, batch_size=bs,
                                  fwd_hooks=hooks, system_msg=SANDBAG_SYSTEMS[system])
                    m, lo, hi = bootstrap_mean_ci(r["p_correct"], clusters=cap_cids, seed=seed)
                    steer_rows.append(dict(layer=layer, framing=framing, system=system,
                                           alpha=alpha, acc=r["acc"], p_correct=m,
                                           lo=lo, hi=hi))
            log.info(f"  steered layer {layer}")
    steer = pd.DataFrame(steer_rows)
    steer.to_csv(d / "sandbag_steer.csv", index=False)

    # ================= controls: random + style directions =================
    ctrl_rows = []
    with timer("controls"):
        style_dir = _style_direction(ctx, ctrl_layer)
        # d itself is run through the identical protocol (same layer, same
        # +/-4 sigma, same eval/incentive cell) so the selectivity claim is a
        # like-for-like comparison rather than d-at-another-layer vs controls.
        control_dirs = [("d", dirs[ctrl_layer]), ("style", style_dir)] + [
            (f"random_{r}", random_direction(model.cfg.d_model, seed=seed + 200 + r))
            for r in range(int(cfg.interventions.n_random_controls))]
        for name, cd in control_dirs:
            for alpha in (-4.0, 4.0):
                hooks = [(resid_name(ctrl_layer),
                          steer_hook(cd, alpha * float(sigma[ctrl_layer])))]
                r = mc_scores(model, cell_texts("eval"), answers, batch_size=bs,
                              fwd_hooks=hooks, system_msg=SANDBAG_SYSTEMS["incentive"])
                b = belief_scores(model, cell_texts("eval"), batch_size=bs,
                                  fwd_hooks=hooks, system_msg=SANDBAG_SYSTEMS["incentive"])
                ctrl_rows.append(dict(direction=name, layer=ctrl_layer,
                                      alpha=alpha, acc=r["acc"],
                                      p_correct=float(np.mean(r["p_correct"])),
                                      belief=float(np.mean(b))))
    pd.DataFrame(ctrl_rows).to_csv(d / "sandbag_controls.csv", index=False)

    # ================= ablation (necessity) =================
    abl_rows = []
    with timer("ablation"):
        for layer in tops:
            hooks = [(resid_name(layer), ablate_hook(dirs[layer]))]
            for framing, system in (("eval", "incentive"), ("deploy", "neutral")):
                r = mc_scores(model, cell_texts(framing), answers, batch_size=bs,
                              fwd_hooks=hooks, system_msg=SANDBAG_SYSTEMS[system])
                b = belief_scores(model, cell_texts(framing), batch_size=bs,
                                  fwd_hooks=hooks, system_msg=SANDBAG_SYSTEMS[system])
                abl_rows.append(dict(layer=layer, framing=framing, system=system,
                                     acc=r["acc"], p_correct=float(np.mean(r["p_correct"])),
                                     belief=float(np.mean(b))))
    pd.DataFrame(abl_rows).to_csv(d / "sandbag_ablate.csv", index=False)

    # ================= patch layer sweep (behavior column of stage table) =====
    n_sweep = int(cfg.behaviors.n_capability_sweep_items)
    sweep_items = cap[:n_sweep]
    layers = sweep_layers(ctx)
    prow = []
    with timer("behavior patch sweep"):
        for j, it in enumerate(sweep_items):
            t_eval = chat_wrap(model, it["eval"], SANDBAG_SYSTEMS["incentive"])
            t_dep = chat_wrap(model, it["deploy"], SANDBAG_SYSTEMS["neutral"])
            # Unpaired-source control, mirroring exp2: the same patch from a
            # DIFFERENT item's deploy prompt. Whatever it does to p(correct) is
            # generic suffix disruption, not eval/deploy content - the behavior
            # column inherits exactly the artifact the belief column has.
            it_u = sweep_items[(j + 1) % len(sweep_items)]
            t_unp = chat_wrap(model, it_u["deploy"], SANDBAG_SYSTEMS["neutral"])
            ids_e, ids_d = tokens_for(model, t_eval), tokens_for(model, t_dep)
            ids_u = tokens_for(model, t_unp)
            k = common_suffix_len(ids_e, ids_d)
            ku = common_suffix_len(ids_e, ids_u)
            with torch.no_grad():
                base_logits = model(ids_e)[:, -1].float().cpu()
            base_pc = float(letter_probs(model, base_logits)[0, "ABCD".index(it["answer"])])
            cache_d_src = cache_resid(model, ids_d, layers)
            cache_u_src = cache_resid(model, ids_u, layers)
            for l in layers:
                lg = patched_final_logits(model, ids_e, l, cache_d_src[l], k)
                pc = float(letter_probs(model, lg)[0, "ABCD".index(it["answer"])])
                lg_u = patched_final_logits(model, ids_e, l, cache_u_src[l], ku)
                pc_u = float(letter_probs(model, lg_u)[0, "ABCD".index(it["answer"])])
                prow.append(dict(layer=l, item_id=it["item_id"], content_id=it["content_id"],
                                 delta_p_correct=pc - base_pc,
                                 delta_p_correct_unpaired=pc_u - base_pc,
                                 base_p_correct=base_pc, k=k, ku=ku))
            del cache_d_src, cache_u_src
    patch = pd.DataFrame(prow)
    patch.to_csv(d / "sandbag_patch_layers.csv", index=False)
    agg_patch = []
    for l, sub in patch.groupby("layer"):
        m, lo, hi = bootstrap_mean_ci(sub.delta_p_correct, clusters=sub.content_id, seed=seed)
        mu, _, _ = bootstrap_mean_ci(sub.delta_p_correct_unpaired,
                                     clusters=sub.content_id, seed=seed)
        agg_patch.append(dict(layer=l, mean=m, lo=lo, hi=hi, mean_unpaired=mu))
    agg_patch = pd.DataFrame(agg_patch)
    agg_patch.to_csv(d / "sandbag_patch_agg.csv", index=False)

    # ================= sycophancy =================
    syc = suites["sycophancy"]
    syc_rows = []
    with timer("sycophancy"):
        for framing in ("eval", "deploy"):
            neu = agree_scores(model, [it[f"neutral_{framing}"] for it in syc], batch_size=bs)
            for alpha in (-4.0, 0.0, 4.0):
                hooks = None if alpha == 0.0 else [
                    (resid_name(best_layer),
                     steer_hook(dirs[best_layer], alpha * float(sigma[best_layer])))]
                ass = agree_scores(model, [it[f"asserted_{framing}"] for it in syc],
                                   batch_size=bs, fwd_hooks=hooks)
                neu_a = neu if alpha == 0.0 else agree_scores(
                    model, [it[f"neutral_{framing}"] for it in syc], batch_size=bs,
                    fwd_hooks=hooks)
                shift = ass - neu_a
                m, lo, hi = bootstrap_mean_ci(shift, seed=seed)
                syc_rows.append(dict(framing=framing, alpha=alpha, syc_shift=m,
                                     lo=lo, hi=hi, n=len(syc)))
    pd.DataFrame(syc_rows).to_csv(d / "sycophancy.csv", index=False)

    # ================= over-refusal (generation) =================
    ref = suites["refusal"]
    ref_rows, gens_dump = [], {}
    with timer("refusal"):
        for framing in ("eval", "deploy"):
            r = refusal_rates(model, [it[framing] for it in ref],
                              max_new_tokens=int(cfg.model.max_new_tokens))
            ref_rows.append(dict(framing=framing, alpha=0.0, rate=r["rate"], n=len(ref)))
            gens_dump[f"{framing}_a0"] = r["generations"]
        for alpha in [float(a) for a in cfg.behaviors.refusal_alphas]:
            if alpha == 0.0:
                continue
            hooks = [(resid_name(best_layer),
                      steer_hook(dirs[best_layer], alpha * float(sigma[best_layer])))]
            r = refusal_rates(model, [it["eval"] for it in ref],
                              max_new_tokens=int(cfg.model.max_new_tokens), fwd_hooks=hooks)
            ref_rows.append(dict(framing="eval", alpha=alpha, rate=r["rate"], n=len(ref)))
            gens_dump[f"eval_a{alpha:+.0f}"] = r["generations"]
    pd.DataFrame(ref_rows).to_csv(d / "refusal.csv", index=False)
    save_json(d / "refusal_generations.json", gens_dump)

    # ================= self-report =================
    sr = suites["selfreport"]
    sr_rows = []
    with timer("self-report"):
        for framing in ("eval", "deploy"):
            for alpha in (-4.0, 0.0, 4.0):
                hooks = None if alpha == 0.0 else [
                    (resid_name(best_layer),
                     steer_hook(dirs[best_layer], alpha * float(sigma[best_layer])))]
                ev = selfreport_ev(model, [it[framing] for it in sr], batch_size=bs,
                                   fwd_hooks=hooks)
                m, lo, hi = bootstrap_mean_ci(ev, seed=seed)
                sr_rows.append(dict(framing=framing, alpha=alpha, self_rating=m,
                                    lo=lo, hi=hi))
    pd.DataFrame(sr_rows).to_csv(d / "selfreport.csv", index=False)

    # ================= headline numbers & figures =================
    b = {(r.framing, r.system): r for r in base.itertuples()}
    sandbag_gap = b[("deploy", "neutral")].p_correct - b[("eval", "incentive")].p_correct
    it_df = pd.DataFrame(item_rows)
    piv = it_df.pivot_table(index="item_id", columns=["framing", "system"],
                            values="p_correct")
    d_eff = cohen_d_paired(piv[("deploy", "neutral")], piv[("eval", "incentive")])

    # The gap is a WITHIN-item difference, so its uncertainty must be estimated
    # paired, not from the two per-cell CIs (which carry across-item variance and
    # are far wider). R3 is a null claim, so the interval is the result: it says
    # what size of sandbagging gap these items could have excluded.
    cid_map = {it["item_id"]: it["content_id"] for it in cap}
    gap_items = (piv[("deploy", "neutral")] - piv[("eval", "incentive")]).dropna()
    gap_cids = np.array([cid_map[i] for i in gap_items.index])
    gap_mean, gap_lo, gap_hi = bootstrap_mean_ci(gap_items.to_numpy(),
                                                 clusters=gap_cids, seed=seed)
    # same for the incentive main effect (does the incentive prompt do anything?)
    inc_items = (((piv[("eval", "incentive")] - piv[("eval", "neutral")])
                  + (piv[("deploy", "incentive")] - piv[("deploy", "neutral")])) / 2).dropna()
    inc_mean, inc_lo, inc_hi = bootstrap_mean_ci(inc_items.to_numpy(),
                                                 clusters=np.array([cid_map[i] for i in inc_items.index]),
                                                 seed=seed)
    log.info(f"sandbag gap {gap_mean:+.4f} [{gap_lo:+.4f},{gap_hi:+.4f}] (d={d_eff:+.2f}); "
             f"incentive main effect {inc_mean:+.4f} [{inc_lo:+.4f},{inc_hi:+.4f}]; "
             f"n={len(gap_items)} items")

    steer_best = steer[(steer.layer == tops[0])]
    P.save(
        P.dose_response(
            steer_best.assign(group=steer_best.framing + "+" + steer_best.system),
            "alpha", "p_correct", "group", ylabel="P(correct)", lo="lo", hi="hi",
            title=f"B: capability vs steering (layer {tops[0]})",
        ),
        d / "fig_sandbag_dose", steer_best,
    )
    P.save(
        P.layer_curves(
            [dict(label="patch deploy->eval", x=agg_patch.layer, y=agg_patch["mean"],
                  lo=agg_patch.lo, hi=agg_patch.hi)],
            ylabel=r"$\Delta$ P(correct)", hline=0.0,
            title="B: behavior effect of patching, by layer",
        ),
        d / "fig_sandbag_patch_layers", agg_patch,
    )
    cells_lab = [f"{f}/{s}" for f, s in CELLS]
    P.save(
        P.bars_ci(cells_lab, base.p_correct, base.lo, base.hi,
                  xlabel="P(correct)", title="B: sandbagging 2x2 baselines"),
        d / "fig_sandbag_baseline", base,
    )

    save_json(d / "summary.json", dict(
        top_layers=tops,
        sandbag_gap_p_correct=float(sandbag_gap),
        sandbag_gap_cohen_d=float(d_eff),
        sandbag_gap_paired=dict(mean=float(gap_mean), lo=float(gap_lo), hi=float(gap_hi),
                                n_items=int(len(gap_items))),
        incentive_effect_paired=dict(mean=float(inc_mean), lo=float(inc_lo),
                                     hi=float(inc_hi)),
        baseline_cells={f"{f}/{s}": float(b[(f, s)].p_correct) for f, s in CELLS},
    ))
    ctx.write_manifest("exp3")
