"""exp8 (Experiment G): is the "evaluation-awareness direction" just register?

Motivation: at 7B a matched-norm STYLE direction (mass-mean of formal vs
casual deployment framings -- zero evaluation content) moved the verbalized
belief MORE than d itself. That is the format-sensitivity critique surfacing
at the causal level. This experiment decomposes it.

  G1  Geometry. cos(d_l, style_l) at every layer: how much of d IS the
      register axis, representationally?

  G2  Causal decomposition. At the steering-peak layer, steer with
        d_hat            the awareness direction
        s_hat            the style direction
        d_perp           d orthogonalized to style   (context minus register)
        s_perp           style orthogonalized to d   (register minus context)
        random           matched-norm baseline
      all at identical norm alpha * sigma_l. The fraction of d's belief
      swing that survives orthogonalization to style is the headline: if
      d_perp keeps most of the swing, the direction carries genuine context
      signal beyond register; if it collapses, the causally potent part of
      "evaluation awareness" is largely a register axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting as P
from ..data.build import make_pairs
from ..interventions import random_direction, resid_name, steer_hook
from ..probes import mass_mean_directions
from ..readouts import belief_scores
from ..stats import bootstrap_mean_ci
from ..utils import log, save_json, timer
from .common import Ctx


def _style_dirs_all_layers(ctx: Ctx) -> np.ndarray:
    """Mass-mean formal-vs-casual direction per layer, deploy prompts only,
    train split only (same recipe as exp3's control, all layers at once)."""
    X, labels = ctx.acts()
    m = (labels.split == "train") & (labels.cls == "deploy")
    formal = labels.family.isin(["production_metadata", "task_context"])
    rows = m & labels.family.isin(
        ["production_metadata", "task_context", "casual_user", "personal_stakes"])
    dirs, _ = mass_mean_directions(X[rows.to_numpy()],
                                   formal[rows].astype(int).to_numpy())
    return dirs


def _style_neutral_dirs(ctx: Ctx) -> np.ndarray:
    """Formal-vs-casual direction from NEUTRAL register wrappers (no
    evaluation or deployment cue families involved), train contents only.
    The in-corpus style direction above is built from deploy-side families
    and so is correlated with class by construction; this one is not."""
    from ..activations import cache_final_resid
    from ..data.contents import get_contents
    from ..data.templates import REGISTER_CASUAL, REGISTER_FORMAL, render
    from ..model_io import chat_wrap

    corpus = ctx.corpus()
    train_ids = sorted({r["content_id"] for r in corpus
                        if r["split"] == "train" and not r["special"]})
    contents = {c["id"]: c for c in get_contents(ctx.cfg.data.n_contents)}
    texts, labels = [], []
    for cid in train_ids:
        task = contents[cid]["task"]
        for t in REGISTER_FORMAL:
            texts.append(chat_wrap(ctx.model, render(t, task)))
            labels.append(1)
        for t in REGISTER_CASUAL:
            texts.append(chat_wrap(ctx.model, render(t, task)))
            labels.append(0)
    X = cache_final_resid(ctx.model, texts, batch_size=int(ctx.cfg.model.batch_size))
    dirs, _ = mass_mean_directions(X, np.array(labels))
    return dirs


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp8")
    dirs, sigma, s1 = ctx.directions()
    seed = int(cfg.data.seed)
    L = model.cfg.n_layers

    import json
    s2 = {}
    p2 = ctx.out / "exp2" / "summary.json"
    if p2.exists():
        s2 = json.loads(p2.read_text())
    layer = int(s2.get("steer_layer", s1["best_layer"]))

    # ---------------- G1: geometry ----------------
    style = _style_dirs_all_layers(ctx)
    with timer("neutral register direction"):
        style_n = _style_neutral_dirs(ctx)
    cos_rows = [dict(layer=l, cos=float(_unit(dirs[l]) @ _unit(style[l])),
                     cos_neutral=float(_unit(dirs[l]) @ _unit(style_n[l])),
                     cos_style_neutral=float(_unit(style[l]) @ _unit(style_n[l])))
                for l in range(L)]
    cos_df = pd.DataFrame(cos_rows)
    cos_df.to_csv(d / "cos_d_style.csv", index=False)
    cos_at = float(cos_df[cos_df.layer == layer].cos.iloc[0])
    cos_n_at = float(cos_df[cos_df.layer == layer].cos_neutral.iloc[0])
    log.info(f"G1: cos(d, style) at steering layer {layer}: {cos_at:+.3f} "
             f"(range {cos_df.cos.min():+.3f}..{cos_df.cos.max():+.3f}); "
             f"cos(d, neutral style) {cos_n_at:+.3f}")

    # ---------------- G2: causal decomposition ----------------
    d_hat = _unit(dirs[layer])
    s_hat = _unit(style[layer])
    d_perp = _unit(d_hat - (d_hat @ s_hat) * s_hat)
    s_perp = _unit(s_hat - (s_hat @ d_hat) * d_hat)
    sn_hat = _unit(style_n[layer])
    d_perp_n = _unit(d_hat - (d_hat @ sn_hat) * sn_hat)
    # The random-direction null is a DISTRIBUTION, so it needs many draws:
    # default 100 matched-norm random unit vectors (config:
    # interventions.n_register_randoms), each steered at a small dose
    # (|alpha| = 1, the on-distribution regime, primary) and a large dose
    # (|alpha| = 4, secondary). d's selectivity is reported as its empirical
    # percentile within the |random swing| distribution at each dose.
    n_rand = int(cfg.interventions.get("n_register_randoms", 100))
    rands = [(f"random_{r}", random_direction(model.cfg.d_model, seed=seed + 900 + r))
             for r in range(n_rand)]
    direction_set = [("d", d_hat), ("style", s_hat), ("d_perp_style", d_perp),
                     ("style_perp_d", s_perp), ("style_neutral", sn_hat),
                     ("d_perp_style_neutral", d_perp_n)] + rands
    cos_to_d = {name: float(_unit(v) @ d_hat) for name, v in direction_set}

    n = min(int(cfg.interventions.n_belief_pairs), 40)
    pairs = make_pairs(ctx.corpus(), n, seed=seed + 13, split="test")
    users = [e["user_text"] for e, _ in pairs]
    cids = [e["content_id"] for e, _ in pairs]
    bs = int(cfg.model.batch_size)
    doses = (1.0, 4.0)

    base = belief_scores(model, users, batch_size=bs)
    rows = []
    with timer(f"G2 matched-norm steering ({len(direction_set)} directions)"):
        for di, (name, vec) in enumerate(direction_set):
            for a0 in doses:
                for alpha in (-a0, a0):
                    hooks = [(resid_name(layer),
                              steer_hook(vec, alpha * float(sigma[layer])))]
                    b = belief_scores(model, users, batch_size=bs, fwd_hooks=hooks)
                    for cid, val, delta in zip(cids, b, b - base):
                        rows.append(dict(direction=name, alpha=alpha,
                                         content_id=cid, belief=float(val),
                                         delta=float(delta)))
            if di < 6 or (di - 6) % 20 == 0:
                log.info(f"  steered {name} ({di + 1}/{len(direction_set)})")
    reg = pd.DataFrame(rows)
    reg.to_csv(d / "register_steer.csv", index=False)

    def _swing_table(dose):
        out = {}
        sub0 = reg[reg.alpha.abs() == dose]
        for name, sub in sub0.groupby("direction"):
            hi = sub[sub.alpha > 0].groupby("content_id").delta.mean()
            lo = sub[sub.alpha < 0].groupby("content_id").delta.mean()
            per_item = (hi - lo).dropna()
            m, l_, h_ = bootstrap_mean_ci(per_item.to_numpy(),
                                          clusters=per_item.index.to_numpy(),
                                          seed=seed)
            out[name] = dict(mean=float(m), lo=float(l_), hi=float(h_),
                             cos_d=cos_to_d.get(name))
        return out

    swings_by_dose = {a: _swing_table(a) for a in doses}
    swings = swings_by_dose[4.0]      # backward-compatible: legacy keys use |a|=4

    def _percentile(table):
        rmags = sorted(abs(v["mean"]) for k, v in table.items()
                       if k.startswith("random_"))
        if not rmags:
            return None
        dmag = abs(table["d"]["mean"])
        return float(100.0 * sum(1 for r in rmags if r < dmag) / len(rmags))

    percentiles = {f"alpha{int(a)}": _percentile(swings_by_dose[a]) for a in doses}
    log.info("G2 selectivity percentile of |d| vs random null: "
             + ", ".join(f"|a|={k[5:]}: {v}" for k, v in percentiles.items()))
    rand_means = [v["mean"] for k, v in swings.items() if k.startswith("random_")]
    rand_mu = float(np.mean(rand_means)) if rand_means else None
    rand_sd = float(np.std(rand_means, ddof=1)) if len(rand_means) > 1 else None
    frac = (swings["d_perp_style"]["mean"] / swings["d"]["mean"]
            if abs(swings["d"]["mean"]) > 1e-9 else None)
    frac_n = (swings["d_perp_style_neutral"]["mean"] / swings["d"]["mean"]
              if abs(swings["d"]["mean"]) > 1e-9 else None)
    # projection law: least-squares slope of swing on cos(x, d), through origin
    xs = np.array([v["cos_d"] for v in swings.values() if v["cos_d"] is not None])
    ys = np.array([v["mean"] for v in swings.values() if v["cos_d"] is not None])
    slope = float((xs @ ys) / (xs @ xs)) if len(xs) else None
    resid = ys - slope * xs if slope is not None else np.array([])
    r2 = (float(1 - (resid @ resid) / ((ys - ys.mean()) @ (ys - ys.mean())))
          if len(ys) > 2 and (ys - ys.mean()) @ (ys - ys.mean()) > 1e-12 else None)
    log.info("G2 swings: " + "; ".join(f"{k}: {v['mean']:+.3f}" for k, v in swings.items()))
    log.info(f"G2: frac of d surviving style removal {frac and round(frac, 3)}; "
             f"random null {rand_mu and round(rand_mu, 3)} +/- {rand_sd and round(rand_sd, 3)}; "
             f"projection-law slope {slope and round(slope, 3)} R2 {r2 and round(r2, 3)}")

    # ---------------- figures & summary ----------------
    try:
        order = (["d", "d_perp_style", "style", "style_perp_d",
                  "style_neutral", "d_perp_style_neutral"]
                 + [k for k in swings if k.startswith("random_")])
        labels = [o for o in order if o in swings]
        P.save(P.bars_ci(labels, [swings[o]["mean"] for o in labels],
                         [swings[o]["lo"] for o in labels],
                         [swings[o]["hi"] for o in labels],
                         xlabel=r"belief swing across $\pm4\sigma$ (logits)",
                         title=f"G2: matched-norm steering at layer {layer}"),
               d / "fig_register", reg)
        P.save(P.layer_curves(
            [dict(label="cos(d, style)", x=cos_df.layer, y=cos_df.cos,
                  lo=cos_df.cos, hi=cos_df.cos)],
            ylabel="cosine similarity", hline=0.0,
            title="G1: how much of d is the register axis, by layer"),
            d / "fig_cos_d_style", cos_df)
    except Exception as exc:
        log.info(f"exp8 figures skipped: {exc}")

    small = swings_by_dose[1.0]
    frac_small = (small["d_perp_style"]["mean"] / small["d"]["mean"]
                  if abs(small["d"]["mean"]) > 1e-9 else None)
    frac_n_small = (small["d_perp_style_neutral"]["mean"] / small["d"]["mean"]
                    if abs(small["d"]["mean"]) > 1e-9 else None)

    def _pct_of(table, name):
        rmags = sorted(abs(v["mean"]) for k, v in table.items() if k.startswith("random_"))
        if not rmags or name not in table:
            return None
        return float(100.0 * sum(1 for r in rmags if r < abs(table[name]["mean"])) / len(rmags))

    save_json(d / "summary.json", dict(
        layer=layer,
        cos_d_style_at_layer=cos_at,
        cos_d_style_neutral_at_layer=cos_n_at,
        cos_d_style_max=float(cos_df.cos.abs().max()),
        cos_d_style_neutral_max=float(cos_df.cos_neutral.abs().max()),
        frac_d_surviving_neutral_removal=frac_n,
        frac_d_surviving_neutral_removal_small=frac_n_small,
        percentiles_all={f"alpha{int(a)}": {nm: _pct_of(swings_by_dose[a], nm)
                                             for nm in ("d", "style", "style_neutral",
                                                        "d_perp_style", "d_perp_style_neutral")}
                         for a in doses},
        random_p95={f"alpha{int(a)}": float(np.percentile(
            [abs(v["mean"]) for k, v in swings_by_dose[a].items() if k.startswith("random_")], 95))
            for a in doses},
        swings=swings,
        swings_small_dose={k: v for k, v in small.items()
                           if not k.startswith("random_")},
        frac_d_surviving_style_removal=frac,
        frac_d_surviving_style_removal_small=frac_small,
        random_null=dict(mean=rand_mu, sd=rand_sd, n=len(rand_means)),
        selectivity_percentile=percentiles,
        projection_law=dict(slope=slope, r2=r2),
    ))
    ctx.write_manifest("exp8")
