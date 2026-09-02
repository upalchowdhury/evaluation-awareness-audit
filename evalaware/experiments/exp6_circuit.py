"""exp6 (Experiment E): who creates, carries, and uses d?

E1  attribution patching over every attention head + MLP (screening)
E2  causal verification: patch the top-k nodes individually and jointly
    (deploy source -> eval run); then patch the best 3 during the SANDBAGGING
    task and measure the accuracy change (does the micro-circuit carry the
    behavior, not just the verbalized belief?)
E3  connect components to the direction: write scores (does the component's
    output move along d differently by class?) and read ratios (are its
    Q/V weights unusually aligned with d vs random directions?)

Gradients want fp32: either run the whole pipeline on a small model, or set
`attribution.model_override` to a smaller sibling (its direction d is then
re-extracted for that model from the same corpus).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .. import plotting as P
from ..activations import cache_final_resid
from ..attribution import attribution_patching, verify_nodes, write_read_scores
from ..config import NS
from ..data.behaviors import SANDBAG_SYSTEMS
from ..data.build import make_pairs
from ..interventions import (ablate_hook, common_suffix_len, node_hook_name,
                             patch_node_hook, random_direction, resid_name, tokens_for)
from ..model_io import chat_wrap, load_model
from ..probes import mass_mean_auc, mass_mean_directions
from ..readouts import belief_prompt, letter_probs
from ..utils import log, save_json, timer
from .common import Ctx


def _belief_metric_local(model, ids) -> float:
    from ..attribution import _belief_metric

    return float(_belief_metric(model, model(ids)).item())


def _label_local(nd: dict) -> str:
    from ..attribution import _label

    return _label(nd)


def _same_node(a: dict, b: dict) -> bool:
    return a["type"] == b["type"] and a["layer"] == b["layer"] and a.get("head") == b.get("head")


def _attr_model_and_direction(ctx: Ctx):
    """(model, dirs, best_layer, tag). Re-derives d if a smaller model is used."""
    override = ctx.cfg.attribution.get("model_override")
    if not override or override == ctx.cfg.model.name:
        dirs, sigma, s1 = ctx.directions()
        return ctx.model, dirs, int(s1["best_layer"]), ctx.cfg.model.short

    mcfg = ctx.cfg.model.to_dict()
    mcfg["name"] = override
    model = load_model(NS(mcfg))
    records = [r for r in ctx.corpus() if r["split"] == "train" and not r["special"]]
    rng = np.random.default_rng(int(ctx.cfg.data.seed))
    rng.shuffle(records)
    records = records[:300]
    texts = [chat_wrap(model, r["user_text"]) for r in records]
    y = np.array([1 if r["cls"] == "eval" else 0 for r in records])
    with timer("override-model direction extraction"):
        X = cache_final_resid(model, texts, batch_size=int(ctx.cfg.model.batch_size))
    dirs, _ = mass_mean_directions(X, y)
    half = len(y) // 2
    aucs = [mass_mean_auc(X[:half, l].astype(np.float32), y[:half],
                          X[half:, l].astype(np.float32), y[half:])
            for l in range(X.shape[1])]
    best = int(np.nanargmax(aucs))
    log.info(f"override model best layer {best} (auc={aucs[best]:.3f})")
    return model, dirs, best, "override"


def run(ctx: Ctx) -> None:
    cfg = ctx.cfg
    d = ctx.dir("exp6")
    seed = int(cfg.data.seed)
    model, dirs, best_layer, tag = _attr_model_and_direction(ctx)
    corpus = ctx.corpus()

    # Screening (E1) and causal verification (E2/E4) use DISJOINT content
    # halves, so the verified effects are not the winner's-curse residue of the
    # scan that selected the nodes.
    test_cids = sorted({r["content_id"] for r in corpus
                        if r["split"] == "test" and not r["special"]})
    screen_c, verify_c = set(test_cids[::2]), set(test_cids[1::2])
    log.info(f"content split: {len(screen_c)} screen / {len(verify_c)} verify")

    def belief_pair_texts(n: int, seed_off: int, contents=None) -> list[tuple[str, str]]:
        pairs = make_pairs(corpus, 3 * n, seed=seed + seed_off, split="test")
        if contents is not None:
            pairs = [pq for pq in pairs if pq[0]["content_id"] in contents]
        pairs = pairs[:n]
        return [
            (belief_prompt(model, e["user_text"], "A"),
             belief_prompt(model, p["user_text"], "A"))
            for e, p in pairs
        ]

    # ---------------- E1: attribution screening ----------------
    with timer("E1 attribution patching"):
        attr = attribution_patching(model, belief_pair_texts(int(cfg.attribution.n_pairs), 6,
                                                             contents=screen_c))
    attr.to_csv(d / "attribution.csv", index=False)
    top_k = int(cfg.attribution.top_k)
    nodes = [dict(type=r.type, layer=int(r.layer), head=int(r.head))
             for r in attr.head(top_k).itertuples()]
    log.info("top nodes: " + ", ".join(
        (f"L{n['layer']}H{n['head']}" if n["type"] == "head" else f"MLP{n['layer']}")
        for n in nodes[:10]))

    # ---------------- E2: causal verification ----------------
    with timer("E2 node patching"):
        verify = verify_nodes(model, belief_pair_texts(int(cfg.attribution.n_verify_pairs), 7,
                                                       contents=verify_c),
                              nodes)
    verify.to_csv(d / "verify.csv", index=False)
    meta_rows = {"JOINT", "FULL", "UNPAIRED_JOINT"}
    v = verify[~verify.node.isin(meta_rows)].dropna(subset=["mean_delta"])
    v = v.reindex(v.mean_delta.abs().sort_values(ascending=False).index)
    top3 = [n for n in (v.node.head(3).tolist())]
    joint_delta = float(verify[verify.node == "JOINT"].mean_delta.iloc[0])
    full_delta = float(verify[verify.node == "FULL"].mean_delta.iloc[0])
    unp = verify[verify.node == "UNPAIRED_JOINT"].mean_delta
    unpaired_joint_delta = float(unp.iloc[0]) if len(unp) else None
    # fraction of the full deploy-vs-eval framing effect recovered by patching
    # only the screened nodes (the number R6 quotes)
    joint_pct = joint_delta / full_delta if abs(full_delta) > 1e-9 else float("nan")
    log.info(f"E2: strongest verified nodes {top3}; joint delta {joint_delta:+.3f} "
             f"= {100 * joint_pct:.0f}% of full framing delta {full_delta:+.3f}; "
             f"unpaired-joint {unpaired_joint_delta}")

    # ---------------- E2b: do the top nodes carry the BEHAVIOR? ----------------
    from ..attribution import parse_label

    top3_nodes = [parse_label(l) for l in top3]
    names = sorted({node_hook_name(nd) for nd in top3_nodes})
    cap = ctx.behaviors()["capability"][: int(cfg.attribution.n_behavior_verify_items)]
    brows = []
    with timer("E2b behavior verification"):
        for it in cap:
            t_e = chat_wrap(model, it["eval"], SANDBAG_SYSTEMS["incentive"])
            t_d = chat_wrap(model, it["deploy"], SANDBAG_SYSTEMS["neutral"])
            ids_e, ids_d = tokens_for(model, t_e), tokens_for(model, t_d)
            k = common_suffix_len(ids_e, ids_d)
            if k < 2:
                continue
            with torch.no_grad():
                base_pc = float(letter_probs(model, model(ids_e)[:, -1].float().cpu())
                                [0, "ABCD".index(it["answer"])])
                _, cache_r = model.run_with_cache(ids_d, names_filter=lambda n: n in names)
            src = {n: cache_r[n].float().cpu() for n in names}
            del cache_r
            hooks = [(node_hook_name(nd), patch_node_hook(nd, src[node_hook_name(nd)], k))
                     for nd in top3_nodes]
            with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                pc = float(letter_probs(model, model(ids_e)[:, -1].float().cpu())
                           [0, "ABCD".index(it["answer"])])
            brows.append(dict(item_id=it["item_id"], base_p_correct=base_pc,
                              patched_p_correct=pc, delta=pc - base_pc))
    beh = pd.DataFrame(brows)
    beh.to_csv(d / "behavior_verify.csv", index=False)

    # ---------------- E3: connection to d ----------------
    test_recs = [r for r in corpus if r["split"] == "test" and not r["special"]]
    rng = np.random.default_rng(seed)
    rng.shuffle(test_recs)
    ev = [chat_wrap(model, r["user_text"]) for r in test_recs if r["cls"] == "eval"][:40]
    dp = [chat_wrap(model, r["user_text"]) for r in test_recs if r["cls"] == "deploy"][:40]
    with timer("E3 write/read scores"):
        align = write_read_scores(model, ev, dp, nodes[: min(10, len(nodes))],
                                  dirs[best_layer],
                                  n_random=int(cfg.attribution.n_random_dirs), seed=seed)
    align.to_csv(d / "alignment.csv", index=False)

    # ---------------- E4: is the node effect mediated by d? ----------------
    # The review-proof "circuit" step: patch a node (deploy -> eval) and then
    # project d out of the residual stream at that node's own layer. If the
    # node's belief effect travels along d, it collapses under d-ablation but
    # survives a matched random-direction ablation. Also measures, per layer,
    # how much of the FULL eval/deploy belief separation d mediates. Verify
    # half only; the screen half never touches these estimates.
    a_ok = align.dropna(subset=["write_score"])
    writer_label = (str(a_ok.loc[a_ok.write_score.abs().idxmax(), "node"])
                    if len(a_ok) else None)
    med_nodes = []
    for lbl in ([writer_label] if writer_label else []) + top3:
        nd = parse_label(lbl)
        if not any(_same_node(nd, m) for m in med_nodes):
            med_nodes.append(nd)
    med_layers = sorted({nd["layer"] for nd in med_nodes})

    e4_pairs = belief_pair_texts(int(cfg.attribution.n_verify_pairs), 9,
                                 contents=verify_c)
    rand_dir = random_direction(model.cfg.d_model, seed=seed + 400)
    med_rows = []
    with timer("E4 mediation"):
        for i, (clean_text, corrupt_text) in enumerate(e4_pairs):
            ids_c = tokens_for(model, clean_text)
            ids_r = tokens_for(model, corrupt_text)
            k = common_suffix_len(ids_c, ids_r)
            if k < 2:
                continue
            names = sorted({node_hook_name(nd) for nd in med_nodes})
            with torch.no_grad():
                base_c = _belief_metric_local(model, ids_c)
                base_r = _belief_metric_local(model, ids_r)
                _, cache_r = model.run_with_cache(ids_r, names_filter=lambda n: n in names)
            src = {n: cache_r[n].float().cpu() for n in names}
            del cache_r

            # E4a: how much of the full separation does d mediate, per layer?
            for L in med_layers:
                for dname, vec in (("d", dirs[L]), ("rand", rand_dir)):
                    hk = [(resid_name(L), ablate_hook(vec))]
                    with torch.no_grad(), model.hooks(fwd_hooks=hk):
                        m_c = _belief_metric_local(model, ids_c)
                        m_r = _belief_metric_local(model, ids_r)
                    med_rows.append(dict(kind="full", node="", layer=L, ablate=dname,
                                         pair=i, base_sep=base_c - base_r,
                                         abl_sep=m_c - m_r))

            # E4b: node patch effect, with and without d at the node's layer
            for nd in med_nodes:
                L = nd["layer"]
                lbl = _label_local(nd)
                ph = [(node_hook_name(nd), patch_node_hook(nd, src[node_hook_name(nd)], k))]
                with torch.no_grad(), model.hooks(fwd_hooks=ph):
                    m_p = _belief_metric_local(model, ids_c)
                for dname, vec in (("d", dirs[L]), ("rand", rand_dir)):
                    ah = (resid_name(L), ablate_hook(vec))
                    with torch.no_grad(), model.hooks(fwd_hooks=[ah]):
                        m_ab = _belief_metric_local(model, ids_c)
                    with torch.no_grad(), model.hooks(fwd_hooks=ph + [ah]):
                        m_pab = _belief_metric_local(model, ids_c)
                    med_rows.append(dict(kind="node", node=lbl, layer=L, ablate=dname,
                                         pair=i, delta_patch=m_p - base_c,
                                         delta_patch_abl=m_pab - m_ab))
    med = pd.DataFrame(med_rows)
    med.to_csv(d / "e4_mediation.csv", index=False)

    med_summary = {"full": {}, "nodes": {}}
    if len(med):
        for L, sub in med[med.kind == "full"].groupby("layer"):
            row = {}
            for dname, ss in sub.groupby("ablate"):
                bs, ab = ss.base_sep.mean(), ss.abl_sep.mean()
                row[dname] = dict(base_sep=float(bs), abl_sep=float(ab),
                                  mediated_frac=float(1 - ab / bs) if abs(bs) > 1e-9 else None)
            med_summary["full"][int(L)] = row
        for (lbl, dname), ss in med[med.kind == "node"].groupby(["node", "ablate"]):
            dp, dpa = ss.delta_patch.mean(), ss.delta_patch_abl.mean()
            med_summary["nodes"].setdefault(lbl, {})[dname] = dict(
                delta_patch=float(dp), delta_patch_abl=float(dpa),
                mediated_frac=float(1 - dpa / dp) if abs(dp) > 1e-9 else None)
        log.info("E4 mediation: " + "; ".join(
            f"{lbl}: d-frac {v.get('d', {}).get('mediated_frac')} vs rand "
            f"{v.get('rand', {}).get('mediated_frac')}"
            for lbl, v in med_summary["nodes"].items()))

    # ---------------- figures & summary ----------------
    P.save(P.attribution_panel(attr, verify[~verify.node.isin(meta_rows)], top_n=min(top_k, 16),
                               title=f"E: circuit screening -> verification ({tag})"),
           d / "fig_attribution")
    a = align.dropna(subset=["write_score"]).sort_values("write_score")
    P.save(P.bars_ci(a.node.tolist(), a.write_score.tolist(),
                     a.write_score.tolist(), a.write_score.tolist(),
                     xlabel="write score along d (eval - deploy, in SDs)",
                     title="E3: which components write into d?"),
           d / "fig_write_scores", align)

    save_json(d / "summary.json", dict(
        attribution_model=str(model.cfg.model_name),
        best_layer=best_layer,
        top_nodes=[f"L{n['layer']}H{n['head']}" if n["type"] == "head"
                   else f"MLP{n['layer']}" for n in nodes],
        verified_top3=top3,
        joint_belief_delta=joint_delta,
        unpaired_joint_delta=unpaired_joint_delta,
        full_framing_delta=full_delta,
        joint_pct_of_full=joint_pct,
        behavior_delta_mean=float(beh.delta.mean()) if len(beh) else None,
        strongest_writer=(str(align.loc[align.write_score.abs().idxmax(), "node"])
                          if len(align.dropna(subset=["write_score"])) else None),
        mediation=med_summary,
    ))
    ctx.write_manifest("exp6")
