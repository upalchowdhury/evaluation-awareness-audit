#!/usr/bin/env python
"""Collect results into the paper: figures, tables, and numbers.tex macros.

Usage:
  python scripts/make_paper_assets.py --model qwen7b
Then:
  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FIGS = {  # macro name -> results-relative pdf
    "FigProbeLayers": "exp1/fig_probe_layers.pdf",
    "FigGeneralization": "exp1/fig_generalization.pdf",
    "FigGrid": "exp1/fig_grid.pdf",
    "FigReprVerbal": "exp1/fig_repr_vs_verbal.pdf",
    "FigPatchLayers": "exp2/fig_patch_layers.pdf",
    "FigSteerLayers": "exp2/fig_steer_layers.pdf",
    "FigDose": "exp2/fig_dose_belief.pdf",
    "FigDoseNet": "exp2/fig_dose_belief_net.pdf",
    "FigDoseSteer": "exp2/fig_dose_belief_steer.pdf",
    "FigSandbagDose": "exp3/fig_sandbag_dose.pdf",
    "FigSandbagBaseline": "exp3/fig_sandbag_baseline.pdf",
    "FigStages": "exp4/fig_stages.pdf",
    "FigSpan": "exp5/fig_span_ablation.pdf",
    "FigAttribution": "exp6/fig_attribution.pdf",
    "FigSpanPatch": "exp7/fig_span_patch.pdf",
    "FigRegister": "exp8/fig_register.pdf",
}


def _placeholder(figdir: Path, name: str) -> str:
    """A visible 'not produced' figure so the paper still compiles when an
    experiment failed or was skipped, instead of dying on a missing file."""
    out = figdir / "missing.pdf"
    if not out.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.text(0.5, 0.5, "figure not produced\n(experiment not run)",
                ha="center", va="center", fontsize=9, color="0.35")
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out.name


def _load(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "TODO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--compare", default="", help="comma-separated result dirs "
                    "for the cross-model replication table, e.g. qwen15b,qwen7b")
    args = ap.parse_args()

    res = ROOT / args.results / args.model
    paper = ROOT / args.paper
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    # ---- figures ----
    macros: dict[str, str] = {}
    for name, rel in FIGS.items():
        src = res / rel
        dst_name = f"{args.model}_{Path(rel).name}"
        if src.exists():
            shutil.copy(src, paper / "figures" / dst_name)
            macros[name] = dst_name
        else:
            macros[name] = _placeholder(paper / "figures", name)
            print(f"  [missing] {src}")

    # ---- tables ----
    st = res / "exp4" / "stage_table.tex"
    if st.exists():
        shutil.copy(st, paper / "tables" / "stage_table.tex")

    # ---- appendix: cue families + behavior suites, generated from source ----
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    try:
        from evalaware.data.behaviors import SANDBAG_SYSTEMS
        from evalaware.data.templates import DEPLOY_FAMILIES, EVAL_FAMILIES
    except Exception as exc:  # pragma: no cover - appendix is optional
        print(f"  [appendix skipped] {exc}")
    else:
        def _tex(t: str) -> str:
            for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                         ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
                         ("}", "\\}"), ("~", "\\textasciitilde{}"),
                         ("^", "\\textasciicircum{}"), ("|", "\\textbar{}"),
                         ("<", "\\textless{}"), (">", "\\textgreater{}")):
                t = t.replace(a, b)
            return t.replace("\n", " / ")

        rows = ["\\begin{tabular}{@{}p{0.22\\linewidth}rp{0.62\\linewidth}@{}}",
                "\\toprule",
                "family & $n$ & example template \\\\", "\\midrule"]
        for cls, fams in (("EVAL", EVAL_FAMILIES), ("DEPLOY", DEPLOY_FAMILIES)):
            rows.append("\\multicolumn{3}{@{}l}{\\emph{%s framings}} \\\\" % cls)
            for fam, tpls in fams.items():
                ex = _tex(tpls[0])
                ex = ex[:150] + ("\\ldots" if len(ex) > 150 else "")
                rows.append(f"\\texttt{{{_tex(fam)}}} & {len(tpls)} & {ex} \\\\")
        rows += ["\\bottomrule", "\\end{tabular}"]
        (paper / "tables" / "cue_families.tex").write_text("\n".join(rows) + "\n")

        sysrows = ["\\begin{tabular}{@{}p{0.14\\linewidth}p{0.80\\linewidth}@{}}",
                   "\\toprule", "system prompt & text \\\\", "\\midrule"]
        for k, v in SANDBAG_SYSTEMS.items():
            sysrows.append(f"\\texttt{{{_tex(k)}}} & {_tex(v)} \\\\")
        sysrows += ["\\bottomrule", "\\end{tabular}"]
        (paper / "tables" / "sandbag_systems.tex").write_text("\n".join(sysrows) + "\n")

    beh_parts = []
    for name, cols in (
        ("sycophancy.csv", ["framing", "alpha", "syc_shift", "lo", "hi"]),
        ("refusal.csv", ["framing", "alpha", "rate", "n"]),
        ("selfreport.csv", ["framing", "alpha", "self_rating", "lo", "hi"]),
    ):
        p = res / "exp3" / name
        if p.exists():
            df = pd.read_csv(p)[[c for c in cols if c in pd.read_csv(p)]]
            df.columns = [c.replace("_", " ") for c in df.columns]  # no math underscores
            beh_parts.append(df.to_latex(index=False, float_format="%.3f", escape=True))
    if beh_parts:
        (paper / "tables" / "behaviors.tex").write_text("\n\\medskip\n".join(beh_parts))

    # ---- numbers ----
    s1 = _load(res / "exp1" / "summary.json")
    s2 = _load(res / "exp2" / "summary.json")
    s3 = _load(res / "exp3" / "summary.json")
    s4 = _load(res / "exp4" / "summary.json")
    s6 = _load(res / "exp6" / "summary.json")
    ds = _load(res / "dataset" / "manifest.json")

    # Leave-one-family-out AUC on the no-lexicon family, reported AT THE PROBE'S
    # OWN BEST LAYER rather than maximized over layers -- taking the max would
    # be a selection over ~28 layers and inflate the headline.
    subtle = subtle_med = "TODO"
    p = res / "exp1" / "lofo_subtle.csv"
    if p.exists():
        lf = pd.read_csv(p)
        bl = s1.get("best_layer")
        at_best = lf[lf.layer == bl].auc if bl is not None else lf.auc
        subtle = fmt(at_best.iloc[0]) if len(at_best) else fmt(lf.auc.median())
        subtle_med = fmt(lf.auc.median())
    patch_ctrl = "TODO"
    p = res / "exp2" / "agg_patch.csv"
    if p.exists():
        df = pd.read_csv(p)
        un = df[df.direction == "unpaired_control"]
        de = df[df.direction == "deploy_to_eval"]
        if len(un) and len(de):
            l_star = int(de.loc[de["mean"].abs().idxmax(), "layer"])
            row = un[un.layer == l_star]
            patch_ctrl = fmt(row["mean"].iloc[0]) if len(row) else "TODO"

    # ---- decodability is a plateau, not a peak: quote its extent honestly ----
    plateau, decode_last, decode_span = "TODO", "TODO", "TODO"
    p = res / "exp1" / "probe_curve.csv"
    if p.exists():
        pc = pd.read_csv(p)
        hi = pc[pc.test_auc >= 0.90].layer
        if len(hi):
            plateau = f"{int(hi.min())}--{int(hi.max())}"
            band = pc[pc.layer.isin(hi)].test_auc
            decode_span = fmt(float(band.max() - band.min()))
        decode_last = fmt(float(pc.sort_values("layer").test_auc.iloc[-1]))

    # ---- cross-family transfer, with the outlier family called out ----
    tr = {}
    p = res / "exp1" / "generalization.csv"
    if p.exists():
        g = pd.read_csv(p)
        off = g[g.train_family != g.test_family]
        if len(off):
            tr["NtransferMean"] = fmt(off.auc.mean())
            worst = off.loc[off.auc.idxmin()]
            # the family that drags transfer down, in both directions
            by_fam = {f: pd.concat([off[off.train_family == f], off[off.test_family == f]]).auc.mean()
                      for f in set(off.train_family) | set(off.test_family)}
            odd = min(by_fam, key=by_fam.get)
            keep = off[(off.train_family != odd) & (off.test_family != odd)]
            tr["NtransferOddFam"] = odd.replace("_", r"\_")
            tr["NtransferOddMean"] = fmt(by_fam[odd])
            tr["NtransferMeanClean"] = fmt(keep.auc.mean()) if len(keep) else "TODO"
            tr["NtransferMinClean"] = fmt(keep.auc.min()) if len(keep) else "TODO"
            tr["NtransferNClean"] = str(len(keep))
            tr["NtransferWorst"] = fmt(worst.auc)

    # ---- steering layer sweep: peak layer and both-framing magnitudes ----
    st_peak = {}
    p = res / "exp2" / "agg_steer.csv"
    if p.exists():
        a = pd.read_csv(p)
        piv = a.pivot_table(index="layer", columns="framing", values="mean")
        if {"eval", "deploy"} <= set(piv.columns):
            # the layer where BOTH framings move most, i.e. the joint peak
            joint = piv["eval"].abs() + piv["deploy"].abs()
            L = int(joint.idxmax())
            st_peak = {"NsteerLayer": str(L),
                       "NsteerEval": fmt(piv.loc[L, "eval"]),
                       "NsteerDeploy": fmt(piv.loc[L, "deploy"]),
                       "NsteerLate": fmt(piv.loc[piv.index >= 2 * piv.index.max() // 3]
                                         .abs().max().max())}

    controls = s1.get("controls", {})
    peaks = s4.get("peaks", {})
    corrs = s4.get("correlations", {})

    # ---- R3 selectivity: matched-norm +/-4 sigma swing at the same layer ----
    # exp3 runs d, a style direction and n random directions through an
    # identical protocol (eval framing + incentive system prompt); the swing is
    # p_correct(alpha=+4) - p_correct(alpha=-4). A direction that is specific to
    # evaluation awareness should move BELIEF much more than p(correct).
    def _swing(df, col):
        hi = df[df.alpha > 0][col]
        lo = df[df.alpha < 0][col]
        if not len(hi) or not len(lo):
            return None
        return float(hi.mean() - lo.mean())

    swings = {f"Nswing{k}{m}": None for k in ("D", "Style", "Rand") for m in ("Acc", "Belief")}
    ctrl_layer_macro = {}
    p = res / "exp3" / "sandbag_controls.csv"
    if p.exists():
        ctrl = pd.read_csv(p)
        ctrl_layer_macro = ({"NctrlLayer": str(int(ctrl.layer.iloc[0]))}
                            if "layer" in ctrl else {})
        for key, mask in (
            ("D", ctrl.direction == "d"),
            ("Style", ctrl.direction == "style"),
            ("Rand", ctrl.direction.str.startswith("random")),
        ):
            sub = ctrl[mask]
            swings[f"Nswing{key}Acc"] = _swing(sub, "p_correct")
            swings[f"Nswing{key}Belief"] = _swing(sub, "belief")

    # ---- R3: the 2x2 cells themselves, and the incentive main effect ----
    cells = {}
    p = res / "exp3" / "sandbag_baseline.csv"
    if p.exists():
        b = pd.read_csv(p).set_index(["framing", "system"])
        def cell(f, sy):
            return b.loc[(f, sy), "p_correct"] if (f, sy) in b.index else None
        ei, en = cell("eval", "incentive"), cell("eval", "neutral")
        di, dn = cell("deploy", "incentive"), cell("deploy", "neutral")
        cells["NcellEI"], cells["NcellEN"] = fmt(ei), fmt(en)
        cells["NcellDI"], cells["NcellDN"] = fmt(di), fmt(dn)
        if None not in (ei, en, di, dn):
            # main effect of the incentive prompt, averaged over framing
            cells["NincentiveEffect"] = fmt(((ei - en) + (di - dn)) / 2)
            # main effect of framing, averaged over system prompt
            cells["NframingEffect"] = fmt(((di - ei) + (dn - en)) / 2)
        row = pd.read_csv(p)
        w = row.assign(width=row.hi - row.lo)
        cells["NcellCIwidth"] = fmt(w.width.mean())

    # ---- R3: framing and incentive MAIN effects from the per-item cells
    # (sandbag_gap_paired is the diagonal deploy/neutral - eval/incentive) ----
    p = res / "exp3" / "sandbag_items.csv"
    if p.exists():
        try:
            from evalaware.stats import bootstrap_mean_ci as _bciM
            it = pd.read_csv(p).pivot_table(index="item_id", columns=["framing", "system"],
                                            values="p_correct")
            ei, en = it[("eval", "incentive")], it[("eval", "neutral")]
            di, dn = it[("deploy", "incentive")], it[("deploy", "neutral")]
            fr = ((di + dn) / 2 - (ei + en) / 2).dropna()
            inc = ((ei + di) / 2 - (en + dn) / 2).dropna()
            for tag, vals in (("Nframing", fr), ("Nincentive", inc)):
                m_, lo_, hi_ = _bciM(vals.to_numpy(), clusters=vals.index.to_numpy(), seed=0)
                cells[f"{tag}ME"], cells[f"{tag}MELo"], cells[f"{tag}MEHi"] = fmt(m_), fmt(lo_), fmt(hi_)
            cells["NsandbagNitemsME"] = str(int(fr.shape[0]))
        except Exception as exc:
            print(f"  [main effects skipped] {exc}")
    for k in ("NframingME", "NframingMELo", "NframingMEHi", "NincentiveME",
              "NincentiveMELo", "NincentiveMEHi", "NsandbagNitemsME"):
        cells.setdefault(k, "TODO")

    # ---- R3 necessity: does ablating d close the sandbagging gap? ----
    ablate_gap = None
    p = res / "exp3" / "sandbag_ablate.csv"
    if p.exists():
        ab = pd.read_csv(p)
        g = []
        for layer, sub in ab.groupby("layer"):
            e = sub[(sub.framing == "eval") & (sub.system == "incentive")].p_correct
            dd = sub[(sub.framing == "deploy") & (sub.system == "neutral")].p_correct
            if len(e) and len(dd):
                g.append(float(dd.iloc[0] - e.iloc[0]))
        if g:
            ablate_gap = float(pd.Series(g).mean())

    # ---- R5: strongest causal span ----
    s5 = _load(res / "exp5" / "summary.json")
    strongest = s5.get("strongest", {})

    # ---- R2: peak causal-belief patch effect and its layer ----
    patch_peak_layer = "TODO"
    p = res / "exp2" / "agg_patch.csv"
    if p.exists():
        de = pd.read_csv(p)
        de = de[de.direction == "deploy_to_eval"]
        if len(de):
            patch_peak_layer = str(int(de.loc[de["mean"].abs().idxmax(), "layer"]))

    med = s6.get("mediation") or {}
    med_best = {"NmedFullMaxLayer": "TODO", "NmedFullMax": "TODO", "NmedFullMaxRand": "TODO"}
    if med.get("full"):
        best_l, best_v, best_r = None, None, None
        for L, v in med["full"].items():
            f = (v.get("d") or {}).get("mediated_frac")
            if f is not None and (best_v is None or f > best_v):
                best_l, best_v = L, f
                best_r = (v.get("rand") or {}).get("mediated_frac")
        if best_l is not None:
            med_best = {"NmedFullMaxLayer": str(best_l),
                        "NmedFullMaxNum": fmt(100 * best_v, 0),
                        "NmedFullMax": fmt(100 * best_v, 0) + r"\%",
                        "NmedFullMaxRand": fmt(100 * (best_r or 0), 0) + r"\%"}
    med_nodes_tex = med_full_tex = "TODO"
    if med.get("nodes"):
        parts = []
        for lbl, v in med["nodes"].items():
            dfrac, rfrac = (v.get("d") or {}).get("mediated_frac"), (v.get("rand") or {}).get("mediated_frac")
            if dfrac is None:
                continue
            parts.append(f"{lbl}: {100*dfrac:.0f}\\% (rand {100*(rfrac or 0):.0f}\\%)")
        med_nodes_tex = "; ".join(parts) or "TODO"
    if med.get("full"):
        parts = []
        for L, v in sorted(med["full"].items(), key=lambda kv: int(kv[0])):
            dfrac = (v.get("d") or {}).get("mediated_frac")
            if dfrac is None:
                continue
            parts.append(f"L{L}: {100*dfrac:.0f}\\%")
        med_full_tex = ", ".join(parts) or "TODO"

    # ---- dose-response at the locked steering layer: shape + health ----
    dose_macros = {}
    p = res / "exp2" / "dose_belief.csv"
    if p.exists():
        dd = pd.read_csv(p)
        if "layer_tag" in dd and (dd.layer_tag == "steer").any():
            st = dd[(dd.layer_tag == "steer") & (dd.condition == "d")]
            ev = st[st.framing == "eval"].groupby("alpha").belief.mean()
            if {1.0, -1.0} <= set(ev.index):
                dose_macros["NdoseSlopeSmall"] = fmt((ev[1.0] - ev[-1.0]) / 2, 2)
            if "kl" in st:
                small = st[st.alpha.abs() <= 1.0]
                dose_macros["NdoseKLsmall"] = fmt(small.kl.max())
                big = st[st.alpha == st.alpha.max()]
                dose_macros["NdoseKLbig"] = fmt(big.kl.mean(), 2)
                rnd = dd[(dd.layer_tag == "steer") & (dd.condition == "random")]
                if len(rnd) and "kl" in rnd:
                    dose_macros["NdoseKLrandBig"] = fmt(
                        rnd[rnd.alpha == rnd.alpha.max()].kl.mean(), 2)
                    rb = rnd.groupby("alpha").belief.mean()
                    dose_macros["NdoseRandSpan"] = fmt(rb.max() - rb.min(), 2)
                dose_macros["NdoseDSpan"] = fmt(ev.max() - ev.min(), 2)

    # ---- data accounting + interaction (reviewer: pseudoreplication) ----
    acct = {}
    try:
        pb = pd.read_csv(res / "exp2" / "patch_belief.csv")
        acct["NpairsExpTwo"] = str(pb[pb.direction == "deploy_to_eval"].pair.nunique()
                                   if "pair" in pb else 60)
        acct["NclustersExpTwo"] = str(pb.content_id.nunique())
    except Exception:
        pass
    try:
        sp = pd.read_csv(res / "exp7" / "span_patch.csv")
        acct["NpairsFone"] = str(sp.pair.nunique())
        acct["NclustersFone"] = str(sp.content_id.nunique())
        # class x content interaction at the deep readout span
        s7l = _load(res / "exp7" / "summary.json")
        rr = sp[(sp.layer == s7l.get("deep_layer")) & (sp.region == "readout")]
        pv = rr.pivot_table(index=["pair", "content_id"], columns="source",
                            values="delta").reset_index()
        if {"paired", "sameclass", "unpaired", "unpaired_eval"} <= set(pv.columns):
            # halved: same scale as the (averaged) main effects
            ixn = (((pv.paired - pv.sameclass)
                    - (pv.unpaired - pv.unpaired_eval)) / 2).dropna()
            from evalaware.stats import bootstrap_mean_ci as _bci2
            mi, li, hi_ = _bci2(ixn.to_numpy(),
                                clusters=pv.content_id.loc[ixn.index].to_numpy(),
                                seed=0)
            acct["NixnMean"] = fmt(mi)
            acct["NixnLo"] = fmt(li)
            acct["NixnHi"] = fmt(hi_)
    except Exception:
        pass
    try:
        rs = pd.read_csv(res / "exp8" / "register_steer.csv")
        acct["NclustersReg"] = str(rs.content_id.nunique())
    except Exception:
        pass
    # exp8 v3 percentile + small-dose macros
    s8b = _load(res / "exp8" / "summary.json")
    pcts = s8b.get("selectivity_percentile") or {}
    if pcts:
        acct["NpctSmall"] = fmt(pcts.get("alpha1"), 0)
        acct["NpctLarge"] = fmt(pcts.get("alpha4"), 0)
    sm = s8b.get("swings_small_dose") or {}
    if sm.get("d"):
        acct["NswDsmall"] = fmt(sm["d"]["mean"])
        acct["NswDsmallLo"] = fmt(sm["d"]["lo"])
        acct["NswDsmallHi"] = fmt(sm["d"]["hi"])
    fs = s8b.get("frac_d_surviving_style_removal_small")
    acct["NfracDperpSmall"] = (fmt(100 * fs, 0) + r"\%") if fs is not None else "TODO"

    # ---- locked-layer / estimation-half class effect for the primary model ----
    lockm = {}
    try:
        s7L = _load(res / "exp7" / "summary.json")
        rawL = pd.read_csv(res / "exp7" / "span_patch.csv")
        pbL = pd.read_csv(res / "exp2" / "patch_belief.csv")
        uniqL = sorted(set(pbL.content_id))
        discL, estL = set(uniqL[::2]), set(uniqL[1::2])
        deL = pbL[(pbL.direction == "deploy_to_eval") & pbL.content_id.isin(discL)]
        lockedL = int(deL.groupby("layer").delta.mean().abs().idxmax())
        subL = rawL[rawL.content_id.isin(estL) & (rawL.layer == lockedL)
                    & (rawL.region == "readout")]
        pvL = subL.pivot_table(index=["pair", "content_id"], columns="source",
                               values="delta").reset_index()
        clsL = ((pvL.paired + pvL.unpaired) / 2
                - (pvL.sameclass + pvL.unpaired_eval) / 2).dropna()
        from evalaware.stats import bootstrap_mean_ci as _bciL
        mL, loL, hiL = _bciL(clsL.to_numpy(),
                             clusters=pvL.content_id.loc[clsL.index].to_numpy(), seed=0)
        lockm = {"NlockClass": fmt(mL), "NlockClassLo": fmt(loL), "NlockClassHi": fmt(hiL),
                 "NlockClassLayer": str(lockedL),
                 "NlockClassN": str(pvL.content_id.nunique())}
    except Exception:
        pass

    # ---- exp10: directional span 2x2 (answer-state localization) ----
    s10 = _load(res / "exp10" / "summary.json")
    ten = {}
    if s10:
        def _ci10(key, nd=2):
            v = s10.get(key)
            if not v:
                return ("TODO",) * 3
            return (fmt(v["mean"], nd), fmt(v["lo"], nd), fmt(v["hi"], nd))
        for dc in ("eval", "deploy"):
            for region in ("task", "question", "final"):
                m_, l_, h_ = _ci10(f"{dc}_{region}_class")
                tag = f"N{dc.capitalize()}{region.capitalize()}"
                ten[tag] = m_
                ten[tag + "Lo"] = l_
                ten[tag + "Hi"] = h_
        m_, l_, h_ = _ci10("eval_final_interaction")
        ten["NixnFinal"], ten["NixnFinalLo"], ten["NixnFinalHi"] = m_, l_, h_
        ten["NclustersTen"] = str(s10.get("n_contents_per_dst", "TODO"))
    try:
        dd = pd.read_csv(res / "exp10" / "cell_distances.csv")
        s10d = _load(res / "exp10" / "summary.json")
        sub = dd[(dd.layer == s10d.get("deep_layer")) & (dd.region == "final")
                 & (dd.dst_cls == "eval")].set_index("source")["mean"]
        ten["NdistPaired"] = fmt(sub.get("othercls_samecont"), 1)
        ten["NdistSameDiff"] = fmt(sub.get("samecls_diffcont"), 1)
    except Exception:
        pass

    # ---- locked analog of exp10's final-position class effects: layer from
    # the exp2 discovery half, estimated on held-out contents only ----
    try:
        m10 = pd.read_csv(res / "exp10" / "span_direction_patch.csv")
        pbP = pd.read_csv(res / "exp2" / "patch_belief.csv")
        uniqP = sorted(set(pbP.content_id))
        discP, estP = set(uniqP[::2]), set(uniqP[1::2])
        deP = pbP[(pbP.direction == "deploy_to_eval")
                  & pbP.content_id.isin(discP)]
        lockP = int(deP.groupby("layer").delta.mean().abs().idxmax())
        from evalaware.stats import bootstrap_mean_ci as _bciP
        for dc, dtag in (("eval", "Eval"), ("deploy", "Deploy")):
            subP = m10[(m10.layer == lockP) & (m10.region == "final")
                       & (m10.dst_cls == dc) & m10.content_id.isin(estP)]
            pvP = subP.pivot_table(index="content_id", columns="source",
                                   values="delta").dropna().reset_index()
            if not len(pvP):
                continue
            clsP = ((pvP.othercls_samecont + pvP.othercls_diffcont) / 2
                    - (pvP.samecls_samecont + pvP.samecls_diffcont) / 2)
            mP, loP, hiP = _bciP(clsP.to_numpy(),
                                 clusters=pvP.content_id.to_numpy(), seed=0)
            tg = f"NlockFinal{dtag}"
            ten[tg], ten[tg + "Lo"], ten[tg + "Hi"] = fmt(mP), fmt(loP), fmt(hiP)
            ten[tg + "N"] = str(pvP.content_id.nunique())
        ten["NlockFinalLayer"] = str(lockP)
    except Exception:
        pass

    # ---- exp11: distance-matched source 2x2 (Experiment J) ----
    s11 = _load(res / "exp11" / "summary.json")
    if s11:
        def _ci11(key, nd=2):
            v = s11.get(key)
            if not v:
                return ("TODO",) * 3
            return (fmt(v["mean"], nd), fmt(v["lo"], nd), fmt(v["hi"], nd))
        for arm, atag in (("unmatched", "Un"), ("selected", "Sel"),
                          ("constructed", "Con"), ("constructed_min", "ConMin")):
            for dc in ("eval", "deploy"):
                for lname, ltag in (("deep", ""), ("locked", "Lk")):
                    m_, l_, h_ = _ci11(f"{arm}_{dc}_{lname}_final_class")
                    tag = f"Nmatch{atag}{dc.capitalize()}Final{ltag}"
                    ten[tag], ten[tag + "Lo"], ten[tag + "Hi"] = m_, l_, h_
            m_, l_, h_ = _ci11(f"{arm}_eval_deep_final_gap")
            tag = f"Nmatch{atag}GapEval"
            ten[tag], ten[tag + "Lo"], ten[tag + "Hi"] = m_, l_, h_
        cs = s11.get("construct_scale") or {}
        ten["NmatchScale"] = fmt(cs.get("mean"), 2)
        att = s11.get("attrition") or {}
        ten["NmatchClusters"] = str(att.get("used_eval")
                                    or (att.get("used", 0) // 2) or "TODO")
        ten["NmatchK"] = str(s11.get("k_cand", "TODO"))

        # locked variant: discovery-half patch-argmax layer (exp2 records),
        # estimated on the held-out content half, per arm at the final position
        try:
            m11 = pd.read_csv(res / "exp11" / "matched_patch.csv")
            pbJ = pd.read_csv(res / "exp2" / "patch_belief.csv")
            uniqJ = sorted(set(pbJ.content_id))
            discJ, estJ = set(uniqJ[::2]), set(uniqJ[1::2])
            deJ = pbJ[(pbJ.direction == "deploy_to_eval")
                      & pbJ.content_id.isin(discJ)]
            lockJ = int(deJ.groupby("layer").delta.mean().abs().idxmax())
            from evalaware.stats import bootstrap_mean_ci as _bciJ
            for arm, atag in (("unmatched", "Un"), ("selected", "Sel"),
                              ("constructed", "Con"),
                              ("constructed_min", "ConMin")):
                for dc, dtag in (("eval", ""), ("deploy", "Dep")):
                    subJ = m11[(m11.arm == arm) & (m11.layer == lockJ)
                               & (m11.region == "final") & (m11.dst_cls == dc)
                               & m11.content_id.isin(estJ)]
                    pvJ = subJ.pivot_table(index="content_id", columns="source",
                                           values="delta").dropna().reset_index()
                    if not len(pvJ):
                        continue
                    clsJ = ((pvJ.othercls_samecont + pvJ.othercls_diffcont) / 2
                            - (pvJ.samecls_samecont + pvJ.samecls_diffcont) / 2)
                    mJ, loJ, hiJ = _bciJ(clsJ.to_numpy(),
                                         clusters=pvJ.content_id.to_numpy(),
                                         seed=0)
                    tg = f"NmatchLock{atag}{dtag}"
                    ten[tg], ten[tg + "Lo"], ten[tg + "Hi"] = (
                        fmt(mJ), fmt(loJ), fmt(hiJ))
                    ten[tg + "N"] = str(pvJ.content_id.nunique())
            ten["NmatchLockLayer"] = str(lockJ)
        except Exception:
            pass

    read_max_v = read_max_q = write_max = "TODO"
    read_extra = {}
    p = res / "exp6" / "alignment.csv"
    if p.exists():
        al = pd.read_csv(p)
        if "read_ratio_v" in al and al.read_ratio_v.notna().any():
            read_max_v = fmt(al.read_ratio_v.max(), 2)
            read_max_q = fmt(al.read_ratio_q.max(), 2)
            rv, rq = al.read_ratio_v.dropna(), al.read_ratio_q.dropna()
            read_extra.update({
                "NreadMinV": fmt(rv.min(), 2), "NreadMinQ": fmt(rq.min(), 2),
                "NreadNheads": str(len(rv)),
                "NreadAboveQ": str(int((rq > 1).sum())),
                "NreadAboveV": str(int((rv > 1).sum())),
            })
        if "write_score" in al and al.write_score.notna().any():
            write_max = fmt(al.write_score.abs().max(), 2)

    # exact checkpoint name (short names like "qwen15b" misread as "Qwen 15B")
    hf_name = str(ds.get("model", args.model))
    model_display = hf_name.split("/")[-1].replace("_", r"\_")

    n = {
        "NModel": model_display,
        "NContents": str(ds.get("config", {}).get("data", {}).get("n_contents", "TODO")),
        "NbestLayer": str(s1.get("best_layer", "TODO")),
        "NprobeAUC": fmt(s1.get("best_test_auc")),
        "NverbalAUC": fmt(s1.get("verbal_auc")),
        "NrvCorr": fmt(s1.get("repr_verbal_spearman")),
        "NpermP": fmt(s1.get("permutation", {}).get("p_value")),
        "NtopicFPR": fmt(controls.get("eval_rate_topic_controls")),
        "NevalRate": fmt(controls.get("eval_rate_test_eval")),
        "NsubtleAUC": subtle,
        "NsubtleMedian": subtle_med,
        "NpatchMax": fmt(s2.get("max_patch_effect")),
        "NpatchCtrl": patch_ctrl,
        "NnetLayer": str(s2.get("net_layer", "TODO")),
        "NlockNetLayer": str((s2.get("locked") or {}).get("net", {}).get("layer", "TODO")
                             if (s2.get("locked") or {}).get("net") else "TODO"),
        "NlockNet": fmt(((s2.get("locked") or {}).get("net") or {}).get("mean")),
        "NlockNetLo": fmt(((s2.get("locked") or {}).get("net") or {}).get("lo")),
        "NlockNetHi": fmt(((s2.get("locked") or {}).get("net") or {}).get("hi")),
        "NlockSteerLayer": str(((s2.get("locked") or {}).get("steer_eval") or {}).get("layer", "TODO")),
        "NlockSteerEval": fmt(((s2.get("locked") or {}).get("steer_eval") or {}).get("mean")),
        "NlockSteerEvalLo": fmt(((s2.get("locked") or {}).get("steer_eval") or {}).get("lo")),
        "NlockSteerEvalHi": fmt(((s2.get("locked") or {}).get("steer_eval") or {}).get("hi")),
        "NlockSteerDeploy": fmt(((s2.get("locked") or {}).get("steer_deploy") or {}).get("mean")),
        "NlockSteerDeployLo": fmt(((s2.get("locked") or {}).get("steer_deploy") or {}).get("lo")),
        "NlockSteerDeployHi": fmt(((s2.get("locked") or {}).get("steer_deploy") or {}).get("hi")),
        "NsteerLayerE": str(s2.get("steer_layer", "TODO")),
        "NnetPatch": fmt(s2.get("max_net_patch_effect")),
        "NunpairedMax": fmt(s2.get("max_unpaired_effect")),
        "NsandbagGap": fmt(s3.get("sandbag_gap_p_correct")),
        "NsandbagD": fmt(s3.get("sandbag_gap_cohen_d"), 2),
        "NsandbagGapLo": fmt(s3.get("sandbag_gap_paired", {}).get("lo")),
        "NsandbagGapHi": fmt(s3.get("sandbag_gap_paired", {}).get("hi")),
        "NsandbagGapPaired": fmt(s3.get("sandbag_gap_paired", {}).get("mean")),
        "NsandbagNitems": str(s3.get("sandbag_gap_paired", {}).get("n_items", "TODO")),
        "NincentiveLo": fmt(s3.get("incentive_effect_paired", {}).get("lo")),
        "NincentiveHi": fmt(s3.get("incentive_effect_paired", {}).get("hi")),
        "NincentivePaired": fmt(s3.get("incentive_effect_paired", {}).get("mean")),
        "NpeakDecode": str(peaks.get("decodability", "TODO")),
        "NdecodePlateau": plateau,
        "NdecodeSpan": decode_span,
        "NdecodeLast": decode_last,
        "NpeakBelief": str(peaks.get("causal_belief", "TODO")),
        "NpeakBeliefNet": str(peaks.get("causal_belief_net", "TODO")),
        "NpeakBeliefSteer": str(peaks.get("causal_belief_steer", "TODO")),
        "NcorrDecodeBelief": fmt(corrs.get("decode_vs_causal_belief_net"), 2),
        "NcorrBeliefBehavior": fmt(corrs.get("causal_belief_net_vs_behavior"), 2),
        "NpeakBehavior": str(peaks.get("behavior", "TODO")),
        "NpeakBehaviorNet": str(peaks.get("behavior_net", "TODO")),
        "NbehaviorRawMax": fmt(s4.get("max_behavior_raw")),
        "NbehaviorNetMax": fmt(s4.get("max_behavior_net")),
        "NbehaviorCtrlMax": fmt(s4.get("max_behavior_control")),
        "NitemCorr": fmt(corrs.get("item_belief_vs_behavior_gap"), 2),
        "NtopNodes": ", ".join(s6.get("top_nodes", [])[:5]) or "TODO",
        "NverifiedNodes": ", ".join(s6.get("verified_top3", [])) or "TODO",
        "NjointDelta": fmt(s6.get("joint_belief_delta")),
        "NunpairedJointE": fmt(s6.get("unpaired_joint_delta")),
        "NjointNetPct": (fmt(100 * (s6["joint_belief_delta"] - s6["unpaired_joint_delta"])
                             / s6["full_framing_delta"], 0) + r"\%"
                         if all(s6.get(k) is not None for k in
                                ("joint_belief_delta", "unpaired_joint_delta",
                                 "full_framing_delta"))
                         and abs(s6["full_framing_delta"]) > 1e-9 else "TODO"),
        "NunpairedJointPct": (fmt(100 * s6["unpaired_joint_delta"]
                                  / s6["full_framing_delta"], 0) + r"\%"
                              if s6.get("unpaired_joint_delta") is not None
                              and abs(s6.get("full_framing_delta") or 0) > 1e-9
                              else "TODO"),
        "NjointPct": (fmt(100 * s6["joint_pct_of_full"], 0) + r"\%"
                      if s6.get("joint_pct_of_full") is not None else "TODO"),
        "NfullDelta": fmt(s6.get("full_framing_delta")),
        "NpatchPeakLayer": patch_peak_layer,
        "NablateGap": fmt(ablate_gap),
        "NspanLayer": str(strongest.get("layer", "TODO")),
        "NspanName": str(strongest.get("span", "TODO")),
        "NspanDelta": fmt(strongest.get("delta")),
        "NspanCueLayer": str(s5.get("per_span", {}).get("cue", {}).get("peak_layer", "TODO")),
        "NspanCueDelta": fmt(s5.get("per_span", {}).get("cue", {}).get("delta")),
        "NspanReadLayer": str(s5.get("per_span", {}).get("readout", {}).get("peak_layer", "TODO")),
        "NspanReadDelta": fmt(s5.get("per_span", {}).get("readout", {}).get("delta")),
        "NspanTaskLayer": str(s5.get("per_span", {}).get("task", {}).get("peak_layer", "TODO")),
        "NspanTaskDelta": fmt(s5.get("per_span", {}).get("task", {}).get("delta")),
        "NbehaviorDeltaE": fmt(s6.get("behavior_delta_mean")),
        "NstrongWriter": str(s6.get("strongest_writer", "TODO")),
        "NmedNodes": med_nodes_tex,
        "NmedFull": med_full_tex,
        **med_best,
        **read_extra,
        "NreadMaxV": read_max_v,
        "NreadMaxQ": read_max_q,
        "NwriteMax": write_max,
    }
    n.update(tr)
    n.update(dose_macros)
    n.update(st_peak)
    n.update(cells)
    n.update({k: (fmt(v) if v is not None else "TODO") for k, v in swings.items()})
    n.update(ctrl_layer_macro)
    n.update(macros)

    # ---- exp7/exp8 macros (artifact anatomy + register decomposition) ----
    s7 = _load(res / "exp7" / "summary.json")
    s8 = _load(res / "exp8" / "summary.json")
    # defaults so every macro exists even when a run predates its experiment
    extra78 = {k: "TODO" for k in (
        "NmlDdsep NmlDdsepLo NmlDdsepHi NmlDaucBase NmlDaucAbl NmlRdsep "
        "NmlRdsepLo NmlRdsepHi NmlRaucAbl NartifactDepthRho NdeepLayerF "
        "NspFullP NspFullU NspReadP NspReadU NspTaskP NspTaskU "
        "NTaskContent NTaskContentLo NTaskContentHi NReadContent "
        "NReadContentLo NReadContentHi NFullContent NFullContentLo "
        "NFullContentHi NcontentLayer NscRead NscTask NscFull "
        "NmeClass NmeClassLo NmeClassHi NmeContent NmeContentLo "
        "NmeContentHi NmeClassMid NmeClassMidLayer NuePatch "
        "NlockClass NlockClassLo NlockClassHi NlockClassLayer NlockClassN "
        "NEvalTask NEvalTaskLo NEvalTaskHi NEvalQuestion NEvalQuestionLo "
        "NEvalQuestionHi NEvalFinal NEvalFinalLo NEvalFinalHi NDeployTask "
        "NDeployTaskLo NDeployTaskHi NDeployQuestion NDeployQuestionLo "
        "NDeployQuestionHi NDeployFinal NDeployFinalLo NDeployFinalHi "
        "NixnFinal NixnFinalLo NixnFinalHi NclustersTen NdistPaired NdistSameDiff "
        "NpairsExpTwo NclustersExpTwo NpairsFone NclustersFone NixnMean "
        "NixnLo NixnHi NclustersReg NpctSmall NpctLarge NswDsmall NswDsmallLo "
        "NswDsmallHi NfracDperpSmall "
        "NmatchUnEvalFinal NmatchUnEvalFinalLo NmatchUnEvalFinalHi NmatchUnEvalFinalLk NmatchUnEvalFinalLkLo NmatchUnEvalFinalLkHi "
        "NmatchUnDeployFinal NmatchUnDeployFinalLo NmatchUnDeployFinalHi NmatchUnDeployFinalLk NmatchUnDeployFinalLkLo NmatchUnDeployFinalLkHi "
        "NmatchUnGapEval NmatchUnGapEvalLo NmatchUnGapEvalHi NmatchSelEvalFinal NmatchSelEvalFinalLo NmatchSelEvalFinalHi "
        "NmatchSelEvalFinalLk NmatchSelEvalFinalLkLo NmatchSelEvalFinalLkHi NmatchSelDeployFinal NmatchSelDeployFinalLo NmatchSelDeployFinalHi "
        "NmatchSelDeployFinalLk NmatchSelDeployFinalLkLo NmatchSelDeployFinalLkHi NmatchSelGapEval NmatchSelGapEvalLo NmatchSelGapEvalHi "
        "NmatchConEvalFinal NmatchConEvalFinalLo NmatchConEvalFinalHi NmatchConEvalFinalLk NmatchConEvalFinalLkLo NmatchConEvalFinalLkHi "
        "NmatchConDeployFinal NmatchConDeployFinalLo NmatchConDeployFinalHi NmatchConDeployFinalLk NmatchConDeployFinalLkLo NmatchConDeployFinalLkHi "
        "NmatchConGapEval NmatchConGapEvalLo NmatchConGapEvalHi NmatchScale NmatchClusters NmatchK "
        "NmatchConMinEvalFinal NmatchConMinEvalFinalLo NmatchConMinEvalFinalHi NmatchConMinEvalFinalLk NmatchConMinEvalFinalLkLo "
        "NmatchConMinEvalFinalLkHi NmatchConMinDeployFinal NmatchConMinDeployFinalLo NmatchConMinDeployFinalHi NmatchConMinDeployFinalLk "
        "NmatchConMinDeployFinalLkLo NmatchConMinDeployFinalLkHi NmatchConMinGapEval NmatchConMinGapEvalLo NmatchConMinGapEvalHi "
        "NmatchLockConMin NmatchLockConMinLo NmatchLockConMinHi NmatchLockConMinN "
        "NmatchLockUnDep NmatchLockUnDepLo NmatchLockUnDepHi NmatchLockUnDepN NmatchLockSel "
        "NmatchLockSelLo NmatchLockSelHi NmatchLockSelN NmatchLockSelDep NmatchLockSelDepLo "
        "NmatchLockSelDepHi NmatchLockSelDepN NmatchLockCon NmatchLockConLo NmatchLockConHi "
        "NmatchLockConN NmatchLockConDep NmatchLockConDepLo NmatchLockConDepHi NmatchLockConDepN "
        "NmatchLockConMin NmatchLockConMinLo NmatchLockConMinHi NmatchLockConMinN NmatchLockConMinDep "
        "NmatchLockConMinDepLo NmatchLockConMinDepHi NmatchLockConMinDepN NlockFinalEval NlockFinalEvalLo "
        "NlockFinalEvalHi NlockFinalEvalN NlockFinalDeploy NlockFinalDeployLo NlockFinalDeployHi "
        "NlockFinalDeployN NlockFinalLayer "
        "NmatchLockUn NmatchLockUnLo NmatchLockUnHi NmatchLockUnN "
        "NmatchLockCon NmatchLockConLo NmatchLockConHi NmatchLockConN "
        "NmatchLockLayer "
        "NrandAbsSwing NrandSwingSd NcosDS NcosDSmax NregLayer NswD "
        "NswDperp NswStyle NswStylePerp NswRand NfracDperp").split()}
    if s7:
        ml = s7.get("multilayer", {})

        def _mlm(cond, key, nd=3):
            return fmt(ml.get(cond, {}).get(key), nd)

        extra78.update({
            "NmlDdsep": _mlm("all_d", "dsep_mean"),
            "NmlDdsepLo": _mlm("all_d", "dsep_lo"), "NmlDdsepHi": _mlm("all_d", "dsep_hi"),
            "NmlDaucBase": _mlm("all_d", "auc_base", 2), "NmlDaucAbl": _mlm("all_d", "auc_abl", 2),
            "NmlRdsep": _mlm("all_rand", "dsep_mean"),
            "NmlRdsepLo": _mlm("all_rand", "dsep_lo"), "NmlRdsepHi": _mlm("all_rand", "dsep_hi"),
            "NmlRaucAbl": _mlm("all_rand", "auc_abl", 2),
            "NartifactDepthRho": fmt(s7.get("artifact_scaling", {}).get("depth_spearman"), 2),
            "NdeepLayerF": str(s7.get("deep_layer", "TODO")),
        })
        pc = res / "exp7" / "span_patch_agg.csv"
        pcc = res / "exp7" / "span_patch_class_samecontent.csv"
        if not pcc.exists():  # v1 results used the old (misleading) file name
            pcc = res / "exp7" / "span_patch_content.csv"
        if pc.exists():
            ag = pd.read_csv(pc)
            dl = s7.get("deep_layer")
            deep = ag[ag.layer == dl]
            def _cell(region, source):
                r = deep[(deep.region == region) & (deep.source == source)]
                return fmt(r["mean"].iloc[0]) if len(r) else "TODO"
            extra78.update({
                "NspFullP": _cell("full", "paired"), "NspFullU": _cell("full", "unpaired"),
                "NspReadP": _cell("readout", "paired"), "NspReadU": _cell("readout", "unpaired"),
                "NspTaskP": _cell("task", "paired"), "NspTaskU": _cell("task", "unpaired"),
            })
        if pcc.exists():
            cg = pd.read_csv(pcc)
            dl = s7.get("deep_layer")
            for region, tag in (("task", "Task"), ("readout", "Read"), ("full", "Full")):
                rr = cg[(cg.region == region) & (cg.layer == dl)]
                if len(rr):
                    extra78.update({
                        f"N{tag}Content": fmt(rr["mean"].iloc[0]),
                        f"N{tag}ContentLo": fmt(rr.lo.iloc[0]),
                        f"N{tag}ContentHi": fmt(rr.hi.iloc[0]),
                    })
            extra78["NcontentLayer"] = str(dl)
        pce = res / "exp7" / "span_patch_effects.csv"
        if pce.exists():
            ef = pd.read_csv(pce)
            dl = s7.get("deep_layer")
            for effect, tag in (("class", "Class"), ("content", "Content")):
                r_ = ef[(ef.layer == dl) & (ef.region == "readout")
                        & (ef.effect == effect)]
                if len(r_):
                    extra78.update({
                        f"Nme{tag}": fmt(r_["mean"].iloc[0]),
                        f"Nme{tag}Lo": fmt(r_.lo.iloc[0]),
                        f"Nme{tag}Hi": fmt(r_.hi.iloc[0]),
                    })
            # class effect at the mid (steering-adjacent) layer for the
            # "consolidates late" sentence
            mids = sorted(ef.layer.unique())
            if len(mids) > 1:
                ml_ = mids[0]
                r_ = ef[(ef.layer == ml_) & (ef.region == "readout")
                        & (ef.effect == "class")]
                if len(r_):
                    extra78["NmeClassMid"] = fmt(r_["mean"].iloc[0])
                    extra78["NmeClassMidLayer"] = str(ml_)
        # same-class source cells at the deep layer (if the run recorded them)
        if pc.exists():
            ag2 = pd.read_csv(pc)
            deep2 = ag2[ag2.layer == s7.get("deep_layer")]
            sc = deep2[(deep2.source == "sameclass")]
            ue = deep2[(deep2.source == "unpaired_eval") & (deep2.region == "readout")]
            if len(ue):
                extra78["NuePatch"] = fmt(ue["mean"].iloc[0])
            for region, tag in (("readout", "Read"), ("task", "Task"), ("full", "Full")):
                r_ = sc[sc.region == region]
                if len(r_):
                    extra78[f"Nsc{tag}"] = fmt(r_["mean"].iloc[0])
    if s8:
        sw = s8.get("swings", {})
        def _sw(k):
            return fmt(sw[k]["mean"]) if k in sw else "TODO"
        fr = s8.get("frac_d_surviving_style_removal")
        rn = s8.get("random_null") or {}
        rmeans = [v["mean"] for k, v in sw.items() if k.startswith("random_")]
        rabs = fmt(float(np.mean([abs(x) for x in rmeans])), 2) if rmeans else "TODO"
        extra78.update({
            "NrandAbsSwing": rabs,
            "NrandSwingSd": fmt(rn.get("sd"), 2),
            "NcosDS": fmt(s8.get("cos_d_style_at_layer"), 2),
            "NcosDSmax": fmt(s8.get("cos_d_style_max"), 2),
            "NregLayer": str(s8.get("layer", "TODO")),
            "NswD": _sw("d"), "NswDperp": _sw("d_perp_style"),
            "NswStyle": _sw("style"), "NswStylePerp": _sw("style_perp_d"),
            "NswRand": (fmt(rn.get("mean")) if rn.get("mean") is not None else "TODO"),
            "NfracDperp": (fmt(100 * fr, 0) + r"\%") if fr is not None else "TODO",
        })
    n.update(extra78)
    n.update({k: v for k, v in acct.items()})
    n.update(ten)
    n.update(lockm)

    def _class_me(sh, which="deep"):
        """Class main effect at the readout span from exp7.
        which='deep'   : max-patch-effect layer, all common contents (exploratory)
        which='locked' : discovery-selected steering layer, ESTIMATION-half
                         contents only (the locked estimate)."""
        s7c = _load(ROOT / args.results / sh / "exp7" / "summary.json")
        praw = ROOT / args.results / sh / "exp7" / "span_patch.csv"
        if not s7c or not praw.exists():
            return None
        raw = pd.read_csv(praw)
        if which == "deep":
            layer = s7c.get("deep_layer")
            sub = raw
        else:
            pb2 = ROOT / args.results / sh / "exp2" / "patch_belief.csv"
            if not pb2.exists():
                return None
            pbd = pd.read_csv(pb2)
            uniq = sorted(set(pbd.content_id))
            disc, est = set(uniq[::2]), set(uniq[1::2])   # exp2's locked split
            de_ = pbd[(pbd.direction == "deploy_to_eval") & pbd.content_id.isin(disc)]
            layer = int(de_.groupby("layer").delta.mean().abs().idxmax())
            if layer not in set(raw.layer.unique()):
                return None                # discovery layer not among exp7's layers
            sub = raw[raw.content_id.isin(est)]
        rr = sub[(sub.layer == layer) & (sub.region == "readout")]
        pv = rr.pivot_table(index=["pair", "content_id"], columns="source",
                            values="delta").reset_index()
        need = {"paired", "sameclass", "unpaired", "unpaired_eval"}
        if not need <= set(pv.columns) or not len(pv):
            return None
        cls = ((pv.paired + pv.unpaired) / 2 - (pv.sameclass + pv.unpaired_eval) / 2).dropna()
        if len(cls) < 3:
            return None
        from evalaware.stats import bootstrap_mean_ci as _bci3
        m_, lo_, hi_ = _bci3(cls.to_numpy(),
                             clusters=pv.content_id.loc[cls.index].to_numpy(), seed=0)
        return f"{m_:+.2f} [{lo_:+.2f},{hi_:+.2f}] (n={pv.content_id.nunique()})"

    # ---- cross-model replication table ----
    if args.compare:
        shorts = [x.strip() for x in args.compare.split(",") if x.strip()]
        crows = []
        for sh in shorts:
            r = ROOT / args.results / sh
            a1, a3, a4, a6 = (_load(r / f"exp{i}" / "summary.json") for i in (1, 3, 4, 6))
            if not a1:
                print(f"  [compare: no results for {sh}]")
                continue
            pk = a4.get("peaks", {})
            man = _load(r / "dataset" / "manifest.json")
            disp = str(man.get("model", sh)).split("/")[-1]
            a2 = _load(r / "exp2" / "summary.json")
            lk = a2.get("locked") or {}
            gp = a3.get("sandbag_gap_paired") or {}

            def _ci(dd):
                if not dd or dd.get("mean") is None:
                    return None
                return f"{dd['mean']:+.2f} [{dd['lo']:+.2f},{dd['hi']:+.2f}]"

            jp = a6.get("joint_pct_of_full")
            up = (None if a6.get("unpaired_joint_delta") is None
                  or not a6.get("full_framing_delta")
                  else a6["unpaired_joint_delta"] / a6["full_framing_delta"])
            crows.append({
                "model": disp,
                "probe AUC": a1.get("best_test_auc"),
                "verbal AUC": a1.get("verbal_auc"),
                "sandbag gap (paired CI)": _ci(gp),
                "locked steering (eval framing)": (
                    f"L{lk['steer_eval']['layer']}: " + _ci(lk["steer_eval"])
                    if lk.get("steer_eval") else None),
                "class effect, max-effect layer (exploratory)": _class_me(sh, "deep"),
                "class effect, locked (disc.-half layer, est.-half contents)": _class_me(sh, "locked"),
            })
        # per-model register-decomposition trend table (exp8)
        grows = []
        for sh in shorts:
            g8 = _load(ROOT / args.results / sh / "exp8" / "summary.json")
            if not g8:
                continue
            man = _load(ROOT / args.results / sh / "dataset" / "manifest.json")
            disp = str(man.get("model", sh)).split("/")[-1]
            sw8 = g8.get("swings", {})
            d_m = (sw8.get("d") or {}).get("mean")
            st_m = (sw8.get("style") or {}).get("mean")
            rn8 = [abs(v["mean"]) for k, v in sw8.items() if k.startswith("random_")]
            def _ci8(k):
                v = sw8.get(k)
                return (f"{v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]"
                        if v else None)
            grows.append({
                "model": disp,
                "cos(d, style)": g8.get("cos_d_style_at_layer"),
                "swing d (CI)": _ci8("d"),
                "swing style (CI)": _ci8("style"),
                "d surviving style removal":
                    g8.get("frac_d_surviving_style_removal"),
                "random |swing| mean":
                    (float(np.mean(rn8)) if rn8 else None),
            })
        if grows:
            gdf = pd.DataFrame(grows)
            (paper / "tables" / "register_scale.tex").write_text(
                gdf.to_latex(index=False, float_format="%.2f", escape=True,
                             na_rep="--"))
            print(f"  register-scale table: {len(grows)} model(s)")

        # estimator-sensitivity table: true class signal vs the standard
        # paired-minus-unpaired estimator, per model, at the deep readout span
        srows = []
        for sh in shorts:
            s7c = _load(ROOT / args.results / sh / "exp7" / "summary.json")
            peff = ROOT / args.results / sh / "exp7" / "span_patch_effects.csv"
            praw = ROOT / args.results / sh / "exp7" / "span_patch.csv"
            if not s7c or not peff.exists() or not praw.exists():
                continue
            dl = s7c.get("deep_layer")
            ef = pd.read_csv(peff)
            cl = ef[(ef.layer == dl) & (ef.region == "readout") & (ef.effect == "class")]
            raw = pd.read_csv(praw)
            rr = raw[(raw.layer == dl) & (raw.region == "readout")]
            pv = rr.pivot_table(index=["pair", "content_id"], columns="source",
                                values="delta").reset_index()
            if not {"paired", "unpaired"} <= set(pv.columns) or not len(cl):
                continue
            stdv = (pv.paired - pv.unpaired).dropna()
            from evalaware.stats import bootstrap_mean_ci as _bci
            m_, lo_, hi_ = _bci(stdv.to_numpy(),
                                clusters=pv.content_id.loc[stdv.index].to_numpy(), seed=0)
            man = _load(ROOT / args.results / sh / "dataset" / "manifest.json")
            disp = str(man.get("model", sh)).split("/")[-1]
            srows.append({
                "model": disp,
                "estimated source-class effect (2x2 CI)":
                    f"{cl['mean'].iloc[0]:+.2f} [{cl.lo.iloc[0]:+.2f},{cl.hi.iloc[0]:+.2f}]",
                "paired-minus-unpaired contrast (CI)":
                    f"{m_:+.2f} [{lo_:+.2f},{hi_:+.2f}]",
            })
        if srows:
            sdf = pd.DataFrame(srows)
            (paper / "tables" / "estimator_sensitivity.tex").write_text(
                sdf.to_latex(index=False, escape=True, na_rep="--"))
            print(f"  estimator-sensitivity table: {len(srows)} model(s)")

        # per-model positional class effects (exp10) -> tables/position_models.tex
        prow = []
        for sh in shorts:
            s10m = _load(ROOT / args.results / sh / "exp10" / "summary.json")
            if not s10m:
                continue
            man = _load(ROOT / args.results / sh / "dataset" / "manifest.json")
            disp = str(man.get("model", sh)).split("/")[-1]
            for dc, dlab in (("eval", "eval dest."), ("deploy", "deploy dest.")):
                cells = {}
                for region, rlab in (("task", "pre-question"), ("question", "question"),
                                     ("final", "final pre-answer")):
                    v = s10m.get(f"{dc}_{region}_class")
                    cells[rlab] = (f"{v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]"
                                   if v else "--")
                prow.append({"model": disp, "destination": dlab, **cells})
        if prow:
            pdf_ = pd.DataFrame(prow)
            (paper / "tables" / "position_models.tex").write_text(
                pdf_.to_latex(index=False, escape=True, na_rep="--"))
            print(f"  position table: {len(prow)} rows")

        # per-model distance-matched class effects (exp11) -> tables/matched_models.tex
        mrow = []
        for sh in shorts:
            s11m = _load(ROOT / args.results / sh / "exp11" / "summary.json")
            if not s11m:
                continue
            man = _load(ROOT / args.results / sh / "dataset" / "manifest.json")
            disp = str(man.get("model", sh)).split("/")[-1]
            for dc, dlab in (("eval", "eval dest."), ("deploy", "deploy dest.")):
                cells = {}
                for arm, alab in (("unmatched", "unmatched"),
                                  ("selected", "distance-selected"),
                                  ("constructed", "distance-constructed"),
                                  ("constructed_min", "shrink-constructed")):
                    v = s11m.get(f"{arm}_{dc}_deep_final_class")
                    cells[alab] = (f"{v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]"
                                   if v else "--")
                # own-locus column: the readout span (belief question + final
                # position) contains every model's locus, so a model whose
                # class effect sits at the question tokens is not shown as an
                # uninformative row of zeros
                vr = s11m.get(f"constructed_min_{dc}_deep_readout_class")
                cells["shrink, readout span"] = (
                    f"{vr['mean']:+.2f} [{vr['lo']:+.2f},{vr['hi']:+.2f}]"
                    if vr else "--")
                mrow.append({"model": disp, "destination": dlab, **cells})
        if mrow:
            mdf_ = pd.DataFrame(mrow)
            (paper / "tables" / "matched_models.tex").write_text(
                mdf_.to_latex(index=False, escape=True, na_rep="--"))
            print(f"  matched table: {len(mrow)} rows")

        # inverse-variance pooled sandbagging gap across checkpoints
        ests = []
        for sh in shorts:
            man_ = _load(ROOT / args.results / sh / "dataset" / "manifest.json")
            if "Qwen" not in str(man_.get("model", "")):
                continue  # the pooled null is a same-family statement
            g = _load(ROOT / args.results / sh / "exp3" / "summary.json").get(
                "sandbag_gap_paired") or {}
            if g.get("mean") is not None and g.get("hi") is not None:
                se = max((g["hi"] - g["lo"]) / (2 * 1.96), 1e-6)
                ests.append((g["mean"], se))
        if len(ests) >= 2:
            w = [1 / se ** 2 for _, se in ests]
            pooled = sum(m * wi for (m, _), wi in zip(ests, w)) / sum(w)
            pooled_se = (1 / sum(w)) ** 0.5
            n["NpooledGap"] = fmt(pooled)
            n["NpooledGapLo"] = fmt(pooled - 1.96 * pooled_se)
            n["NpooledGapHi"] = fmt(pooled + 1.96 * pooled_se)
            n["NpooledK"] = str(len(ests))

        if crows:
            cdf = pd.DataFrame(crows)
            cdf.columns = [c.replace("_", " ") for c in cdf.columns]
            (paper / "tables" / "cross_model.tex").write_text(
                cdf.to_latex(index=False, float_format="%.3f", escape=True, na_rep="--"))
            print(f"  cross-model table: {len(crows)} model(s)")

    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    (paper / "numbers.tex").write_text("% auto-generated - do not edit\n" + "\n".join(lines) + "\n")
    print(f"wrote {paper / 'numbers.tex'} with {len(n)} macros; "
          f"{sum(1 for v in macros.values() if v != 'missing.pdf')}/{len(FIGS)} figures copied")


if __name__ == "__main__":
    main()
