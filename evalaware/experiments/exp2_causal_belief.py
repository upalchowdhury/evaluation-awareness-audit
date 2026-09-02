"""exp2 (Experiment A): causal control of the model's verbalized belief.

Interventions on eval/deploy prompt PAIRS (same content, different framing),
measuring the belief readout:

  patch d->e   deploy residuals into the eval run (aligned suffix) at layer l:
               does the model stop reporting "this is a test"?
  patch e->d   the reverse: does the deploy run start reporting "test"?
  unpaired     control: source is a DIFFERENT content's deploy run - only the
               shared readout suffix aligns. Distinguishes "moves the context
               signal" from "any suffix swap does this".
  steer        alpha * sigma * d_hat added at layer l (no pairing needed);
               full dose-response at the best layer, incl. a random-direction
               control at matched norm.

All prompts come from TEST-split contents (never probe-trained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .. import plotting as P
from ..data.build import make_pairs
from ..interventions import (
    cache_resid,
    common_suffix_len,
    patched_final_logits,
    random_direction,
    resid_name,
    steer_hook,
    tokens_for,
)
from ..model_io import chat_wrap
from ..readouts import belief_from_logits, belief_prompt, belief_scores
from ..stats import bootstrap_mean_ci
from ..utils import log, save_json, timer
from .common import Ctx, sweep_layers


def _agg(df: pd.DataFrame, by: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        m, lo, hi = bootstrap_mean_ci(sub.delta, clusters=sub.content_id, seed=seed)
        rows.append({**dict(zip(by, key)), "mean": m, "lo": lo, "hi": hi, "n": len(sub)})
    return pd.DataFrame(rows)


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp2")
    dirs, sigma, summary = ctx.directions()
    best_layer = int(summary["best_layer"])
    layers = sweep_layers(ctx)
    seed = int(cfg.data.seed)

    pairs = make_pairs(ctx.corpus(), int(cfg.interventions.n_belief_pairs),
                       seed=seed, split="test")
    log.info(f"exp2: {len(pairs)} pairs, {len(layers)} layers")

    etexts = [belief_prompt(model, e["user_text"], "A") for e, _ in pairs]
    dtexts = [belief_prompt(model, p["user_text"], "A") for _, p in pairs]
    cids = [e["content_id"] for e, _ in pairs]

    # ---------------- patching ----------------
    rows = []
    with timer("patch layer sweep"):
        for i, (etext, dtext) in enumerate(zip(etexts, dtexts)):
            ids_e, ids_d = tokens_for(model, etext), tokens_for(model, dtext)
            k = common_suffix_len(ids_e, ids_d)
            j = (i + 1) % len(pairs)  # unpaired source (different content)
            ids_u = tokens_for(model, dtexts[j])
            ku = common_suffix_len(ids_e, ids_u)
            with torch.no_grad():
                base_e = belief_from_logits(model, model(ids_e)[:, -1].float().cpu())[0]
                base_d = belief_from_logits(model, model(ids_d)[:, -1].float().cpu())[0]
            cache_e = cache_resid(model, ids_e, layers)
            cache_d = cache_resid(model, ids_d, layers)
            cache_u = cache_resid(model, ids_u, layers)
            for l in layers:
                m_de = belief_from_logits(
                    model, patched_final_logits(model, ids_e, l, cache_d[l], k))[0]
                m_ed = belief_from_logits(
                    model, patched_final_logits(model, ids_d, l, cache_e[l], k))[0]
                m_un = belief_from_logits(
                    model, patched_final_logits(model, ids_e, l, cache_u[l], ku))[0]
                rows += [
                    dict(layer=l, pair=i, direction="deploy_to_eval", content_id=cids[i],
                         delta=float(m_de - base_e), baseline=float(base_e), k=k),
                    dict(layer=l, pair=i, direction="eval_to_deploy", content_id=cids[i],
                         delta=float(m_ed - base_d), baseline=float(base_d), k=k),
                    dict(layer=l, pair=i, direction="unpaired_control", content_id=cids[i],
                         delta=float(m_un - base_e), baseline=float(base_e), k=ku),
                ]
            del cache_e, cache_d, cache_u
            if (i + 1) % 10 == 0:
                log.info(f"  patched {i + 1}/{len(pairs)} pairs")
    patch = pd.DataFrame(rows)
    patch.to_csv(d / "patch_belief.csv", index=False)
    agg_patch = _agg(patch, ["layer", "direction"], seed)
    agg_patch.to_csv(d / "agg_patch.csv", index=False)

    # ---------------- steering layer sweep (alpha = +/-4 sigma) ----------------
    e_users = [e["user_text"] for e, _ in pairs]
    d_users = [p["user_text"] for _, p in pairs]
    base_e_belief = belief_scores(model, e_users, batch_size=int(cfg.model.batch_size))
    base_d_belief = belief_scores(model, d_users, batch_size=int(cfg.model.batch_size))
    srows = []
    with timer("steer layer sweep"):
        for l in layers:
            for framing, users, base, alpha in (
                ("eval", e_users, base_e_belief, -4.0),
                ("deploy", d_users, base_d_belief, +4.0),
            ):
                hooks = [(resid_name(l), steer_hook(dirs[l], alpha * float(sigma[l])))]
                b = belief_scores(model, users, batch_size=int(cfg.model.batch_size),
                                  fwd_hooks=hooks)
                for cid, delta in zip(cids, b - base):
                    srows.append(dict(layer=l, framing=framing, alpha=alpha,
                                      content_id=cid, delta=float(delta)))
            log.info(f"  steered layer {l}")
    steer = pd.DataFrame(srows)
    steer.to_csv(d / "steer_belief.csv", index=False)
    agg_steer = _agg(steer, ["layer", "framing"], seed)
    agg_steer.to_csv(d / "agg_steer.csv", index=False)

    # ---------------- dose-response ----------------
    # Two layers are probed. `best_layer` is exp1's decodability argmax, which is
    # the conventional choice but sits on a near-flat AUC plateau. `net_layer` is
    # where the PAIRED patch effect most exceeds the unpaired-source control,
    # i.e. where the causal effect is actually specific to eval/deploy rather
    # than generic suffix overwriting. They are usually different layers.
    de_m = agg_patch[agg_patch.direction == "deploy_to_eval"].set_index("layer")["mean"]
    un_m = agg_patch[agg_patch.direction == "unpaired_control"].set_index("layer")["mean"]
    net = (de_m.abs() - un_m.abs().reindex(de_m.index).fillna(0.0))
    net_layer = int(net.idxmax())
    # The dose-response measures STEERING, so its layer must be chosen from the
    # steering sweep, not the patching sweep - these disagree. We run all three
    # candidate layers, because which one you pick determines what you conclude,
    # and that is itself the point.
    piv_st = agg_steer.pivot_table(index="layer", columns="framing", values="mean")
    steer_layer = (int((piv_st["eval"].abs() + piv_st["deploy"].abs()).idxmax())
                   if {"eval", "deploy"} <= set(piv_st.columns) else best_layer)
    log.info(f"dose layers: best={best_layer} (decodability argmax), "
             f"net={net_layer} (paired-minus-control {net.max():+.3f}), "
             f"steer={steer_layer} (steering sweep peak)")

    # ---- locked estimates: select the layer on a discovery half of the test
    # contents, then estimate the effect (with a clustered bootstrap CI) on the
    # held-out estimation half. Guards the headline numbers against
    # winner's-curse inflation from scanning all layers.
    uniq_cids = sorted(set(cids))
    disc_c, est_c = set(uniq_cids[::2]), set(uniq_cids[1::2])

    de_p = patch[patch.direction == "deploy_to_eval"][
        ["layer", "pair", "content_id", "delta"]]
    un_p = (patch[patch.direction == "unpaired_control"][["layer", "pair", "delta"]]
            .rename(columns={"delta": "delta_un"}))
    net_pairs = de_p.merge(un_p, on=["layer", "pair"])
    net_pairs["net"] = net_pairs.delta - net_pairs.delta_un

    def _lock(df, value_col, score):
        """(locked_layer, mean, lo, hi, n_est) with selection on disc_c only."""
        disc = df[df.content_id.isin(disc_c)]
        sel = score(disc)
        if sel is None or not len(sel):
            return None
        layer_star = int(sel.abs().idxmax())
        est = df[(df.layer == layer_star) & df.content_id.isin(est_c)]
        if not len(est):
            return None
        m, lo, hi = bootstrap_mean_ci(est[value_col].to_numpy(),
                                      clusters=est.content_id.to_numpy(), seed=seed)
        return dict(layer=layer_star, mean=float(m), lo=float(lo), hi=float(hi),
                    n_contents=int(est.content_id.nunique()))

    locked_net = _lock(net_pairs, "net",
                       lambda d_: d_.groupby("layer").net.mean())
    st_e = steer[steer.framing == "eval"][["layer", "content_id", "delta"]]
    st_d = steer[steer.framing == "deploy"][["layer", "content_id", "delta"]]
    joint_score = (lambda d_: (st_e[st_e.content_id.isin(disc_c)].groupby("layer").delta.mean().abs()
                               + st_d[st_d.content_id.isin(disc_c)].groupby("layer").delta.mean().abs()))
    locked_steer_eval = _lock(st_e, "delta", joint_score)
    locked_steer_deploy = _lock(st_d, "delta", joint_score)
    log.info(f"locked: net={locked_net} | steer_eval={locked_steer_eval} | "
             f"steer_deploy={locked_steer_deploy}")

    # Health metrics for large-alpha steering: mean KL of the final-position
    # next-token distribution from its unsteered counterpart, and its entropy.
    # If large alpha simply pushes the model off-distribution, it shows up here
    # (KL exploding, entropy collapsing/exploding) rather than silently
    # contaminating the belief curve.
    from ..model_io import final_logits
    from ..readouts import belief_texts

    def _dose_measure(users, hooks, base_la, base_lb):
        la = final_logits(model, belief_texts(model, users, "A"),
                          fwd_hooks=hooks, batch_size=int(cfg.model.batch_size))
        lb = final_logits(model, belief_texts(model, users, "B"),
                          fwd_hooks=hooks, batch_size=int(cfg.model.batch_size))
        from ..readouts import belief_from_logits as _bfl
        b = _bfl(model, la, lb)
        lp, lq = torch.log_softmax(la, -1), torch.log_softmax(base_la, -1)
        kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
        ent = float(-(lp.exp() * lp).sum(-1).mean())
        return b, kl, ent

    base_logits_a = {"eval": final_logits(model, belief_texts(model, e_users, "A"),
                                          batch_size=int(cfg.model.batch_size)),
                     "deploy": final_logits(model, belief_texts(model, d_users, "A"),
                                            batch_size=int(cfg.model.batch_size))}
    base_logits_b = {"eval": final_logits(model, belief_texts(model, e_users, "B"),
                                          batch_size=int(cfg.model.batch_size)),
                     "deploy": final_logits(model, belief_texts(model, d_users, "B"),
                                            batch_size=int(cfg.model.batch_size))}

    drows = []
    rand_dirs = [random_direction(model.cfg.d_model, seed=seed + 100 + r)
                 for r in range(int(cfg.interventions.n_random_controls))]
    dose_layers = [("best", best_layer)]
    for tag_, l_ in (("net", net_layer), ("steer", steer_layer)):
        if l_ not in [x for _, x in dose_layers]:
            dose_layers.append((tag_, l_))
    with timer("dose-response"):
      for tag, dose_layer in dose_layers:
        for alpha in [float(a) for a in cfg.interventions.alpha_sweep]:
            for framing, users, base in (("eval", e_users, base_e_belief),
                                         ("deploy", d_users, base_d_belief)):
                if alpha == 0.0:
                    b = base
                    lp0 = torch.log_softmax(base_logits_a[framing], -1)
                    kl, ent = 0.0, float(-(lp0.exp() * lp0).sum(-1).mean())
                else:
                    hooks = [(resid_name(dose_layer),
                              steer_hook(dirs[dose_layer], alpha * float(sigma[dose_layer])))]
                    b, kl, ent = _dose_measure(users, hooks, base_logits_a[framing],
                                               base_logits_b[framing])
                for cid, val, delta in zip(cids, b, b - base):
                    drows.append(dict(alpha=alpha, framing=framing, condition="d",
                                      layer=dose_layer, layer_tag=tag,
                                      content_id=cid, belief=float(val), delta=float(delta),
                                      kl=kl, entropy=ent))
            # random-direction control (eval framing only; average over dirs)
            if alpha != 0.0:
                accum = np.zeros(len(e_users))
                kl_acc = ent_acc = 0.0
                for rd in rand_dirs:
                    hooks = [(resid_name(dose_layer),
                              steer_hook(rd, alpha * float(sigma[dose_layer])))]
                    b_r, kl_r, ent_r = _dose_measure(e_users, hooks,
                                                     base_logits_a["eval"],
                                                     base_logits_b["eval"])
                    accum += b_r
                    kl_acc += kl_r
                    ent_acc += ent_r
                b = accum / len(rand_dirs)
                for cid, val, delta in zip(cids, b, b - base_e_belief):
                    drows.append(dict(alpha=alpha, framing="eval", condition="random",
                                      layer=dose_layer, layer_tag=tag,
                                      content_id=cid, belief=float(val), delta=float(delta),
                                      kl=kl_acc / len(rand_dirs),
                                      entropy=ent_acc / len(rand_dirs)))
    dose = pd.DataFrame(drows)
    dose.to_csv(d / "dose_belief.csv", index=False)

    # ---------------- figures ----------------
    def curves(direction_names, df, val="mean"):
        out = []
        for name in direction_names:
            sub = df[df.direction == name] if "direction" in df else df[df.framing == name]
            sub = sub.sort_values("layer")
            out.append(dict(label=name, x=sub.layer, y=sub[val], lo=sub.lo, hi=sub.hi))
        return out

    P.save(
        P.layer_curves(
            curves(["deploy_to_eval", "eval_to_deploy", "unpaired_control"], agg_patch),
            ylabel=r"$\Delta$ belief (patched $-$ baseline)",
            title="A: patching the residual stream moves the belief", hline=0.0,
        ),
        d / "fig_patch_layers", agg_patch,
    )
    P.save(
        P.layer_curves(
            [dict(label=f"{r} framing", x=agg_steer[agg_steer.framing == r].sort_values("layer").layer,
                  y=agg_steer[agg_steer.framing == r].sort_values("layer")["mean"],
                  lo=agg_steer[agg_steer.framing == r].sort_values("layer").lo,
                  hi=agg_steer[agg_steer.framing == r].sort_values("layer").hi)
             for r in ("eval", "deploy")],
            ylabel=r"$\Delta$ belief under $\mp 4\sigma$ steering",
            title="A: steering d by layer", hline=0.0,
        ),
        d / "fig_steer_layers", agg_steer,
    )
    dagg = []
    for (alpha, framing, cond, tag, dl), sub in dose.groupby(
            ["alpha", "framing", "condition", "layer_tag", "layer"]):
        m, lo, hi = bootstrap_mean_ci(sub.belief, clusters=sub.content_id, seed=seed)
        dagg.append(dict(alpha=alpha, layer_tag=tag, layer=dl,
                         group=f"{framing} ({cond})", belief=m, lo=lo, hi=hi))
    dagg = pd.DataFrame(dagg)
    for tag, dl in dose_layers:
        sub = dagg[dagg.layer_tag == tag]
        label = {"best": "decodability argmax",
                 "net": "peak paired-minus-control",
                 "steer": "steering sweep peak"}[tag]
        P.save(
            P.dose_response(sub, "alpha", "belief", "group",
                            ylabel="belief score", lo="lo", hi="hi",
                            title=f"A: dose-response at layer {dl} ({label})"),
            d / ("fig_dose_belief" if tag == "best" else f"fig_dose_belief_{tag}"), sub,
        )

    top = (agg_patch[agg_patch.direction == "deploy_to_eval"]
           .assign(absmean=lambda x: x["mean"].abs())
           .sort_values("absmean", ascending=False))
    net_at = float(net.max())
    save_json(d / "summary.json", dict(
        locked=dict(net=locked_net, steer_eval=locked_steer_eval,
                    steer_deploy=locked_steer_deploy),
        best_layer_exp1=best_layer,
        net_layer=net_layer,
        steer_layer=steer_layer,
        max_net_patch_effect=net_at,
        dose_layers={t: int(l) for t, l in dose_layers},
        patch_top_layers=[int(l) for l in top.layer.head(5)],
        max_patch_effect=float(top["mean"].iloc[0]) if len(top) else None,
        max_unpaired_effect=float(un_m.abs().max()),
        layers=layers,
    ))
    ctx.write_manifest("exp2")
