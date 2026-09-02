"""exp1: representation. Cache activations, probe by layer, extract direction d.

Products
--------
final_resid.npy [n, L, d] + labels.csv    activation cache
probe_curve.csv                            grouped-CV AUC (train) + held-out AUC (test) by layer
generalization.csv                         family x family transfer matrix (best layer)
lofo_subtle.csv                            layer curve for the held-out no-lexicon family
directions.npy / sigma.npy                 mass-mean direction per layer (train split only)
verbal.csv                                 per-prompt belief readout (test + special rows)
grid_auc.npy                               (layer x position) decodability map
summary.json                               best layer, AUCs, permutation p, control rates,
                                           representation-vs-verbalization correlation
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plotting as P
from ..activations import cache_final_resid, cache_grid_resid
from ..data.templates import EVAL_FAMILIES
from ..model_io import chat_wrap
from ..probes import (
    family_generalization_matrix,
    grid_auc,
    leave_family_out_by_layer,
    mass_mean_directions,
    permutation_test,
    probe_cv_by_layer,
    projection_scores,
)
from ..readouts import belief_scores, belief_validity
from ..stats import bootstrap_mean_ci, spearman
from ..utils import log, save_json, timer
from .common import Ctx


def run(ctx: Ctx) -> None:
    cfg, model = ctx.cfg, ctx.model
    d = ctx.dir("exp1")
    records = ctx.corpus()
    labels = pd.DataFrame(
        [{k: r[k] for k in ("pid", "cls", "family", "content_id", "split", "kind")}
         for r in records]
    )
    texts = [chat_wrap(model, r["user_text"]) for r in records]

    # ---------------- activation cache ----------------
    with timer("final-token resid cache"):
        X = cache_final_resid(model, texts, batch_size=int(cfg.model.batch_size))
    np.save(d / "final_resid.npy", X)
    labels.to_csv(d / "labels.csv", index=False)

    is_eval = (labels.cls == "eval").to_numpy()
    is_dep = (labels.cls == "deploy").to_numpy()
    core = (is_eval | is_dep) & (labels.split != "special").to_numpy()
    y = is_eval.astype(int)
    groups = labels.content_id.to_numpy()
    train_rows = core & (labels.split == "train").to_numpy()
    test_rows = core & (labels.split == "test").to_numpy()

    # ---------------- probes by layer ----------------
    with timer("grouped-CV probes"):
        curve = probe_cv_by_layer(
            X[train_rows], y[train_rows], groups[train_rows],
            folds=int(cfg.probes.cv_folds), c_reg=float(cfg.probes.c_reg),
        )

    dirs, sigma = mass_mean_directions(X[train_rows], y[train_rows])
    np.save(d / "directions.npy", dirs)
    np.save(d / "sigma.npy", sigma)

    # held-out AUC of the train-fit direction on test contents
    from sklearn.metrics import roc_auc_score

    proj = projection_scores(X, dirs)  # [n, L]
    test_auc = [
        float(roc_auc_score(y[test_rows], proj[test_rows, l]))
        for l in range(X.shape[1])
    ]
    curve["test_auc"] = test_auc
    curve.to_csv(d / "probe_curve.csv", index=False)

    best_layer = int(curve.test_auc.idxmax())
    ranked = curve.sort_values("test_auc", ascending=False).layer.tolist()
    log.info(f"best layer by held-out direction AUC: {best_layer} "
             f"(auc={curve.test_auc.max():.3f})")

    # ---------------- generalization structure ----------------
    fam = labels.family.to_numpy()
    gen = family_generalization_matrix(
        X, y, fam, best_layer, list(EVAL_FAMILIES), train_rows, test_rows,
        c_reg=float(cfg.probes.c_reg),
    )
    gen.to_csv(d / "generalization.csv", index=False)

    lofo = leave_family_out_by_layer(
        X, y, fam, "subtle_exam_format", train_rows, test_rows,
        c_reg=float(cfg.probes.c_reg),
    )
    lofo.to_csv(d / "lofo_subtle.csv", index=False)

    perm = permutation_test(
        X[core][:, best_layer, :], y[core], groups[core],
        n_perm=int(cfg.probes.n_permutations), seed=int(cfg.data.seed),
    )

    # ---------------- control prompts ----------------
    thresh = 0.5 * (proj[train_rows & is_eval, best_layer].mean()
                    + proj[train_rows & is_dep, best_layer].mean())
    tc_rows = (labels.family == "topic_control").to_numpy()
    amb_rows = (labels.cls == "ambiguous").to_numpy()
    controls = dict(
        threshold=float(thresh),
        eval_rate_test_eval=float((proj[test_rows & is_eval, best_layer] > thresh).mean()),
        eval_rate_test_deploy=float((proj[test_rows & is_dep, best_layer] > thresh).mean()),
        eval_rate_topic_controls=float((proj[tc_rows, best_layer] > thresh).mean()),
        eval_rate_ambiguous=float((proj[amb_rows, best_layer] > thresh).mean()),
    )
    log.info(f"controls: {controls}")

    # ---------------- verbalized belief on test + special rows ----------------
    verbal_rows = np.where(test_rows | tc_rows | amb_rows)[0]
    with timer("belief readout"):
        beliefs = belief_scores(
            model, [records[i]["user_text"] for i in verbal_rows],
            batch_size=int(cfg.model.batch_size),
        )
    # instrument check: does the model actually answer the belief question
    # with Yes/No? (mass on Yes/No variants; greedy token is Yes/No)
    with timer("belief readout validity"):
        val = belief_validity(model, [records[i]["user_text"] for i in verbal_rows],
                              batch_size=int(cfg.model.batch_size))
    verbal = labels.iloc[verbal_rows].copy()
    verbal["belief"] = beliefs
    verbal["probe_proj"] = proj[verbal_rows, best_layer]
    verbal["yesno_mass"] = val["mass"]
    verbal["greedy_yesno"] = val["greedy_yesno"]
    verbal.to_csv(d / "verbal.csv", index=False)

    vt = verbal[verbal.split == "test"]
    verbal_auc = float(roc_auc_score((vt.cls == "eval").astype(int), vt.belief))
    rv_corr = spearman(vt.probe_proj, vt.belief)
    log.info(f"verbalized AUC={verbal_auc:.3f} | probe-vs-verbal spearman={rv_corr:.3f}")
    validity = dict(
        yesno_mass_mean=float(vt.yesno_mass.mean()),
        yesno_mass_min=float(vt.yesno_mass.min()),
        greedy_yesno_frac=float(vt.greedy_yesno.mean()),
        greedy_yesno_frac_eval=float(vt[vt.cls == "eval"].greedy_yesno.mean()),
        greedy_yesno_frac_deploy=float(vt[vt.cls == "deploy"].greedy_yesno.mean()),
    )
    log.info(f"readout validity: {validity}")
    # the natural eval-vs-deploy gap of the readout: per-content mean belief
    # under eval framings minus under deploy framings, clustered CI over
    # contents. Every intervention effect should be read against this.
    per_c = vt.groupby(["content_id", "cls"]).belief.mean().unstack("cls")
    per_c = per_c.dropna(subset=["eval", "deploy"])
    gap_vals = (per_c["eval"] - per_c["deploy"]).to_numpy()
    g_m, g_lo, g_hi = bootstrap_mean_ci(gap_vals, clusters=per_c.index.to_numpy(),
                                        seed=int(cfg.data.seed))
    natural_gap = dict(mean=float(g_m), lo=float(g_lo), hi=float(g_hi),
                       n_contents=int(len(per_c)),
                       eval_mean=float(vt[vt.cls == "eval"].belief.mean()),
                       deploy_mean=float(vt[vt.cls == "deploy"].belief.mean()),
                       frac_positive_eval=float((vt[vt.cls == "eval"].belief > 0).mean()),
                       frac_positive_deploy=float((vt[vt.cls == "deploy"].belief > 0).mean()))
    log.info(f"natural eval-deploy belief gap: {natural_gap}")

    # ---------------- (layer x position) grid ----------------
    gsel = np.where(test_rows)[0]
    rng = np.random.default_rng(int(cfg.data.seed))
    rng.shuffle(gsel)
    gsel = np.sort(gsel[: int(cfg.data.grid_prompts)])
    with timer("grid cache"):
        grid, glens = cache_grid_resid(
            model, [texts[i] for i in gsel], last_k=int(cfg.data.grid_last_k)
        )
    gy = y[gsel]
    gcontents = groups[gsel]
    uniqc = np.unique(gcontents)
    gtrain = np.isin(gcontents, uniqc[: len(uniqc) // 2])
    with timer("grid probes"):
        gauc = grid_auc(grid, gy, gtrain)
    np.save(d / "grid_auc.npy", gauc)
    np.save(d / "grid_lengths.npy", glens)

    # ---------------- figures ----------------
    P.save(
        P.layer_curves(
            [
                dict(label="grouped CV (train)", x=curve.layer, y=curve.cv_auc),
                dict(label="held-out contents (test)", x=curve.layer, y=curve.test_auc),
                dict(label="held-out family: subtle format", x=lofo.layer, y=lofo.auc),
            ],
            ylabel="probe AUC", title="Where is eval/deploy context decodable?", hline=0.5,
        ),
        d / "fig_probe_layers", curve,
    )
    gm = gen.pivot(index="train_family", columns="test_family", values="auc")
    P.save(
        P.heatmap(gm.to_numpy(), list(gm.columns), list(gm.index),
                  "Cross-family generalization (best layer)", "AUC",
                  vmin=0.5, vmax=1.0, annotate=True),
        d / "fig_generalization", gen,
    )
    K = grid.shape[2]
    P.save(
        P.heatmap(gauc, list(range(-K, 0)), list(range(gauc.shape[0])),
                  "Decodability by (layer x token position)", "AUC",
                  vmin=0.5, vmax=1.0, figsize=(6.2, 3.6)),
        d / "fig_grid",
    )
    P.apply_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for c, m, lab in ((P.SERIES[0], vt.cls == "eval", "eval-framed"),
                      (P.SERIES[1], vt.cls == "deploy", "deploy-framed")):
        ax.scatter(vt.probe_proj[m], vt.belief[m], s=8, alpha=0.55, color=c, label=lab,
                   edgecolors="none")
    ax.set_xlabel("internal probe projection")
    ax.set_ylabel("verbalized belief score")
    ax.set_title("Representation vs verbalization", loc="left")
    ax.legend(fontsize=8)
    P.save(fig, d / "fig_repr_vs_verbal", vt)

    save_json(
        d / "summary.json",
        dict(
            best_layer=best_layer,
            layers_ranked=ranked,
            best_test_auc=float(curve.test_auc.max()),
            verbal_auc=verbal_auc,
            repr_verbal_spearman=rv_corr,
            permutation=perm,
            controls=controls,
            readout_validity=validity,
            natural_gap=natural_gap,
        ),
    )
    ctx.write_manifest("exp1")
