"""exp4 (Experiment C): stage localization & the three-way dissociation.

Pure analysis over exp1-exp3 outputs (no model forward passes):

  stage table   per layer: decodability (probe AUC) | causal belief effect
                (patching) | causal belief effect (steering) | behavior effect
                (patching P(correct))
  peaks         where each curve maxes out - the "computational stages" claim
  dissociation  layer-level rank correlations between the curves, and the
                item-level representation / verbalization / behavior triangle
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting as P
from ..stats import spearman
from ..utils import log, save_json
from .common import Ctx


def run(ctx: Ctx) -> None:
    d = ctx.dir("exp4")
    e1, e2, e3 = ctx.out / "exp1", ctx.out / "exp2", ctx.out / "exp3"

    curve = pd.read_csv(e1 / "probe_curve.csv")
    patch_b = pd.read_csv(e2 / "agg_patch.csv")
    steer_b = pd.read_csv(e2 / "agg_steer.csv")
    patch_beh = pd.read_csv(e3 / "sandbag_patch_agg.csv")

    de = (patch_b[patch_b.direction == "deploy_to_eval"]
          .rename(columns={"mean": "patch_belief"})[["layer", "patch_belief"]])
    un = (patch_b[patch_b.direction == "unpaired_control"]
          .rename(columns={"mean": "patch_belief_control"})[["layer", "patch_belief_control"]])
    st = (steer_b[steer_b.framing == "eval"]
          .rename(columns={"mean": "steer_belief"})[["layer", "steer_belief"]])
    bh_cols = ["layer", "patch_behavior"]
    bh = patch_beh.rename(columns={"mean": "patch_behavior"})
    if "mean_unpaired" in bh:
        bh = bh.rename(columns={"mean_unpaired": "patch_behavior_control"})
        bh_cols.append("patch_behavior_control")
    bh = bh[bh_cols]

    stage = (curve[["layer", "test_auc"]]
             .merge(de, on="layer", how="inner")
             .merge(un, on="layer", how="left")
             .merge(st, on="layer", how="inner")
             .merge(bh, on="layer", how="inner"))
    stage["decodability"] = stage.test_auc
    stage["causal_belief"] = stage.patch_belief.abs()
    stage["causal_belief_steer"] = stage.steer_belief.abs()
    stage["behavior"] = stage.patch_behavior.abs()
    if "patch_behavior_control" in stage:
        # same correction as the belief column: subtract the unpaired-source
        # patch so generic suffix disruption doesn't read as a behavior stage
        stage["behavior_net"] = (stage.behavior
                                 - stage.patch_behavior_control.abs())
    # Control-corrected causal belief: the unpaired-source patch measures how
    # much of the effect is generic disruption rather than the eval/deploy
    # contrast, so subtract it. This is the column the staged-computation claim
    # should be read off, and it is what `causal_belief_net` peaks report.
    stage["causal_belief_net"] = (stage.causal_belief
                                  - stage.patch_belief_control.abs().fillna(0.0))
    stage.to_csv(d / "stage_table.csv", index=False)

    peak_cols = ["decodability", "causal_belief", "causal_belief_net",
                 "causal_belief_steer", "behavior"]
    if "behavior_net" in stage:
        peak_cols.append("behavior_net")
    peaks = {c: int(stage.loc[stage[c].idxmax(), "layer"]) for c in peak_cols}
    log.info(f"stage peaks: {peaks}")

    corrs = dict(
        decode_vs_causal_belief=spearman(stage.decodability, stage.causal_belief),
        decode_vs_causal_belief_net=spearman(stage.decodability, stage.causal_belief_net),
        causal_belief_net_vs_behavior=spearman(stage.causal_belief_net, stage.behavior),
        decode_vs_behavior=spearman(stage.decodability, stage.behavior),
        causal_belief_vs_behavior=spearman(stage.causal_belief, stage.behavior),
    )

    # ---- item-level triangle: verbalized belief vs behavior gap ----
    bel = pd.read_csv(e3 / "sandbag_belief.csv")
    items = pd.read_csv(e3 / "sandbag_items.csv")
    pc = items.pivot_table(index="item_id", columns=["framing", "system"], values="p_correct")
    gap = (pc[("deploy", "neutral")] - pc[("eval", "incentive")]).rename("behavior_gap")
    bel_e = bel[(bel.framing == "eval")].set_index("item_id").belief.rename("belief_eval")
    tri = pd.concat([gap, bel_e], axis=1).dropna()
    tri.to_csv(d / "item_triangle.csv")
    corrs["item_belief_vs_behavior_gap"] = spearman(tri.belief_eval, tri.behavior_gap)

    import json
    with open(e1 / "summary.json") as f:
        s1 = json.load(f)
    corrs["item_repr_vs_verbal"] = s1.get("repr_verbal_spearman")

    # ---- figures & LaTeX table ----
    P.save(
        P.stage_summary(
            stage.layer.to_numpy(),
            {
                "decodable (probe AUC)": stage.decodability.to_numpy(),
                "causal on belief (patch, control-corrected)":
                    stage.causal_belief_net.to_numpy(),
                "causal on behavior (patch)": stage.behavior.to_numpy(),
            },
            title="C: where each capacity lives",
        ),
        d / "fig_stages", stage,
    )

    tex = stage[["layer", "decodability", "causal_belief", "causal_belief_net",
                 "causal_belief_steer", "behavior"]].copy()
    tex.columns = ["Layer", "Probe AUC", r"$|\Delta$belief$|$ patch",
                   r"net of control", r"$|\Delta$belief$|$ steer",
                   r"$|\Delta P(\mathrm{corr})|$ patch"]
    # bare tabular (no table/caption wrapper); main.tex supplies the float
    with open(d / "stage_table.tex", "w") as f:
        f.write(tex.to_latex(index=False, float_format="%.3f", escape=False))

    net_max = float(stage.causal_belief_net.max())
    extra = {}
    if "behavior_net" in stage:
        extra = dict(max_behavior_net=float(stage.behavior_net.max()),
                     max_behavior_raw=float(stage.behavior.max()),
                     max_behavior_control=float(stage.patch_behavior_control.abs().max()))
    save_json(d / "summary.json", dict(
        peaks=peaks, correlations=corrs,
        max_causal_belief_net=net_max,
        max_causal_belief_raw=float(stage.causal_belief.max()),
        max_unpaired_control=float(stage.patch_belief_control.abs().max()),
        **extra,
    ))
    ctx.write_manifest("exp4")
