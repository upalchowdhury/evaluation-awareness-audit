#!/usr/bin/env python
"""Extra macros and tables for the plain-language paper (main_v2.tex).

Reads results_v2/<model>/ and writes paper/numbers_v2.tex plus
paper/tables/*_v2.tex. Run AFTER make_paper_assets.py (which writes
numbers.tex from the same results dir):

  python scripts/make_paper_assets.py --results results_v2 --model qwen15b \
      --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b
  python scripts/make_paper_assets_v2.py --results results_v2 --model qwen15b \
      --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b

What is new here, and why:
  * readout validity per model (does the model answer Yes/No at all?)
  * the natural eval-vs-deploy gap of the readout, and every intervention
    effect expressed as a multiple of it (calibration)
  * the random-direction null (p95 of |swing| over 100 matched-norm random
    directions) next to every steering effect
  * the neutral register direction (formal vs casual wrappers with no
    eval/deploy vocabulary) next to the in-corpus style direction
  * Holm-corrected significance markers for the 30 positional cells
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evalaware.stats import bootstrap_mean_ci, holm  # noqa: E402


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def fmt(x, nd=2, sign=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(v):
        return "--"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(d: dict | None, nd=2) -> str:
    if not d or d.get("mean") is None:
        return "--"
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}]"


def _disp(res: Path, short: str) -> str:
    man = _load(res / short / "dataset" / "manifest.json")
    return str(man.get("model", short)).split("/")[-1]


def boot_p(values: np.ndarray, clusters: np.ndarray, iters=4000, seed=0) -> float:
    """Two-sided clustered-bootstrap p-value for mean != 0."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by = {u: values[clusters == u] for u in uniq}
    means = np.empty(iters)
    for i in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = np.concatenate([by[u] for u in pick]).mean()
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(max(p, 1.0 / iters))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen15b")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--compare", default="qwen05b,qwen15b,qwen7b,phi3mini,olmo7b")
    a = ap.parse_args()
    res = ROOT / a.results
    paper = ROOT / a.paper
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    shorts = [s for s in a.compare.split(",") if (res / s / "exp1" / "summary.json").exists()]
    n: dict[str, str] = {k: "--" for k in (
        "VgapMean VgapLo VgapHi VgapEvalMean VgapDeployMean VfracYesEval VfracYesDeploy "
        "VgreedyYesNo VyesnoMass VprobeAUCLzero VprobeAUCbest VprobeBestLayer "
        "VsteerEvalOverGap VsteerDeployOverGap VclassFinalOverGap VcosNeutral VcosStyle "
        "VfracNeutral VfracNeutralSmall VrandPninetyfiveBig VrandPninetyfiveSmall VswDoverGap VrandPninetyfiveOverGap "
        "VposCells VposUncorrected VposHolm").split()}
    for k in ("VselectiveModels VnSelective VinvertedModels VpctDmin VpctDmax VneutralBeatsD "
              "VcosNeutralMin VcosNeutralMax VfracNeutralMin VfracNeutralMax VlocusFinalModels "
              "VlocusQuestionModels VverbalAUCmin VverbalAUCmax VrhoMin VrhoMax VprobeLzeroMin "
              "VprobeLzeroMax VnModels VframingSigModels VnFramingNull VtaskMaxAbs").split():
        n[k] = "--"
    for tag in ("D", "Style", "Neutral", "Dperp", "DperpNeutral"):
        for suf in ("", "Lo", "Hi"):
            n[f"Vsw{tag}{suf}"] = "--"
        n[f"Vpct{tag}big"] = n[f"Vpct{tag}small"] = "--"

    # ------------------------------------------------------------------ primary
    s1 = _load(res / a.model / "exp1" / "summary.json")
    s2 = _load(res / a.model / "exp2" / "summary.json")
    s8 = _load(res / a.model / "exp8" / "summary.json")
    gap = s1.get("natural_gap") or {}
    val = s1.get("readout_validity") or {}
    n["VgapMean"] = fmt(gap.get("mean"), 2, sign=True)
    n["VgapLo"] = fmt(gap.get("lo"), 2, sign=True)
    n["VgapHi"] = fmt(gap.get("hi"), 2, sign=True)
    n["VgapEvalMean"] = fmt(gap.get("eval_mean"), 2, sign=True)
    n["VgapDeployMean"] = fmt(gap.get("deploy_mean"), 2, sign=True)
    n["VfracYesEval"] = fmt(100 * gap.get("frac_positive_eval", np.nan), 0) + r"\%"
    n["VfracYesDeploy"] = fmt(100 * gap.get("frac_positive_deploy", np.nan), 0) + r"\%"
    n["VgreedyYesNo"] = fmt(100 * val.get("greedy_yesno_frac", np.nan), 0) + r"\%"
    n["VyesnoMass"] = fmt(100 * val.get("yesno_mass_mean", np.nan), 0) + r"\%"
    pc = res / a.model / "exp1" / "probe_curve.csv"
    if pc.exists():
        curve = pd.read_csv(pc).sort_values("layer")
        n["VprobeAUCLzero"] = fmt(curve.test_auc.iloc[0], 2)
        n["VprobeAUCbest"] = fmt(curve.test_auc.max(), 3)
        n["VprobeBestLayer"] = str(int(curve.loc[curve.test_auc.idxmax(), "layer"]))
    lk = (s2.get("locked") or {})
    se, sd_ = lk.get("steer_eval") or {}, lk.get("steer_deploy") or {}
    g = gap.get("mean")
    if se.get("mean") is not None and g:
        n["VsteerEvalOverGap"] = fmt(abs(se["mean"]) / abs(g), 2)
    if sd_.get("mean") is not None and g:
        n["VsteerDeployOverGap"] = fmt(abs(sd_["mean"]) / abs(g), 2)
    # class effect over gap (exp10 final position, eval destinations)
    s10 = _load(res / a.model / "exp10" / "summary.json")
    ef = (s10.get("eval_final_class") or {})
    if ef.get("mean") is not None and g:
        n["VclassFinalOverGap"] = fmt(abs(ef["mean"]) / abs(g), 2)
    # register / random null
    sw = s8.get("swings") or {}
    rp = s8.get("random_p95") or {}
    pct = s8.get("percentiles_all") or {}
    n["VcosNeutral"] = fmt(s8.get("cos_d_style_neutral_at_layer"), 2)
    n["VcosStyle"] = fmt(s8.get("cos_d_style_at_layer"), 2)
    n["VfracNeutral"] = (fmt(100 * s8["frac_d_surviving_neutral_removal"], 0) + r"\%"
                         if s8.get("frac_d_surviving_neutral_removal") is not None else "--")
    n["VfracNeutralSmall"] = (fmt(100 * s8["frac_d_surviving_neutral_removal_small"], 0) + r"\%"
                              if s8.get("frac_d_surviving_neutral_removal_small") is not None else "--")
    n["VrandPninetyfiveBig"] = fmt(rp.get("alpha4"), 2)
    n["VrandPninetyfiveSmall"] = fmt(rp.get("alpha1"), 2)
    for nm, tag in (("d", "D"), ("style", "Style"), ("style_neutral", "Neutral"),
                    ("d_perp_style", "Dperp"), ("d_perp_style_neutral", "DperpNeutral")):
        v = sw.get(nm)
        n[f"Vsw{tag}"] = fmt(v["mean"], 2, sign=True) if v else "--"
        n[f"Vsw{tag}Lo"] = fmt(v["lo"], 2, sign=True) if v else "--"
        n[f"Vsw{tag}Hi"] = fmt(v["hi"], 2, sign=True) if v else "--"
        p4 = (pct.get("alpha4") or {}).get(nm)
        p1 = (pct.get("alpha1") or {}).get(nm)
        n[f"Vpct{tag}big"] = fmt(p4, 0) if p4 is not None else "--"
        n[f"Vpct{tag}small"] = fmt(p1, 0) if p1 is not None else "--"
    if g and sw.get("d"):
        n["VswDoverGap"] = fmt(abs(sw["d"]["mean"]) / abs(g), 2)
    if g and rp.get("alpha4") is not None:
        n["VrandPninetyfiveOverGap"] = fmt(rp["alpha4"] / abs(g), 2)

    # ------------------------------------------------------------------ cross-model
    rows = []
    for sh in shorts:
        e1 = _load(res / sh / "exp1" / "summary.json")
        e2 = _load(res / sh / "exp2" / "summary.json")
        e8 = _load(res / sh / "exp8" / "summary.json")
        e10 = _load(res / sh / "exp10" / "summary.json")
        e3 = _load(res / sh / "exp3" / "summary.json")
        gp = e1.get("natural_gap") or {}
        vl = e1.get("readout_validity") or {}
        lkm = (e2.get("locked") or {}).get("steer_eval") or {}
        gm = gp.get("mean")
        curve_p = res / sh / "exp1" / "probe_curve.csv"
        l0 = pd.read_csv(curve_p).sort_values("layer").test_auc.iloc[0] if curve_p.exists() else np.nan
        rnd = (e8.get("random_p95") or {}).get("alpha4")
        swd = (e8.get("swings") or {}).get("d") or {}
        ecls = e10.get("eval_final_class") or {}
        rcls = e10.get("eval_readout_class") or {}
        # framing main effect on accuracy from the per-item 2x2 (exp3)
        fr_ci = "--"
        pit = res / sh / "exp3" / "sandbag_items.csv"
        if pit.exists():
            try:
                it = pd.read_csv(pit).pivot_table(index="item_id", columns=["framing", "system"],
                                                  values="p_correct")
                fr = ((it[("deploy", "incentive")] + it[("deploy", "neutral")]) / 2
                      - (it[("eval", "incentive")] + it[("eval", "neutral")]) / 2).dropna()
                m_, lo_, hi_ = bootstrap_mean_ci(fr.to_numpy(), clusters=fr.index.to_numpy(), seed=0)
                fr_ci = f"{m_:+.2f} [{lo_:+.2f},{hi_:+.2f}]"
            except Exception:
                pass
        rows.append({
            "model": _disp(res, sh),
            "answers Yes/No": fmt(100 * vl.get("greedy_yesno_frac", np.nan), 0) + r"\%",
            "probe AUC (layer 0 / best)": f"{fmt(l0, 2)} / {fmt(e1.get('best_test_auc'), 2)}",
            "verbal AUC": fmt(e1.get("verbal_auc"), 2),
            "probe-vs-answer rho": fmt(e1.get("repr_verbal_spearman"), 2),
            "natural gap": ci(gp),
            "locked steering": (f"L{lkm['layer']}: {ci(lkm)}" if lkm.get("mean") is not None else "--"),
            "steer / gap": fmt(abs(lkm["mean"]) / abs(gm), 2) if (lkm.get("mean") is not None and gm) else "--",
            "random p95 swing": fmt(rnd, 2),
            "class effect, final": ci(ecls),
            "class / gap": fmt(abs(ecls["mean"]) / abs(gm), 2) if (ecls.get("mean") is not None and gm) else "--",
            "framing effect on accuracy": fr_ci,
        })
    if rows:
        df = pd.DataFrame(rows)
        a_cols = ["model", "answers Yes/No", "probe AUC (layer 0 / best)", "verbal AUC",
                  "probe-vs-answer rho", "natural gap"]
        b_cols = ["model", "locked steering", "steer / gap", "random p95 swing",
                  "class effect, final", "class / gap", "framing effect on accuracy"]
        (paper / "tables" / "landscape_v2a.tex").write_text(
            df[a_cols].to_latex(index=False, escape=False, na_rep="--"))
        (paper / "tables" / "landscape_v2b.tex").write_text(
            df[b_cols].to_latex(index=False, escape=False, na_rep="--"))
        print(f"  landscape_v2a/b: {len(rows)} models")

    # ---------------------------------------------------- cross-model verdict macros
    def _short(sh):
        return {"qwen05b": "Qwen-0.5B", "qwen15b": "Qwen-1.5B", "qwen7b": "Qwen-7B",
                "phi3mini": "Phi-3-mini", "olmo7b": "Olmo-3-7B"}.get(sh, sh)
    sel, inv, neutral_beats, pcts, cosn, fracn, locus_final, locus_q, vAUC, rhos, l0s = [], [], [], [], [], [], [], [], [], [], []
    for sh in shorts:
        e1 = _load(res / sh / "exp1" / "summary.json")
        e2 = _load(res / sh / "exp2" / "summary.json")
        e8 = _load(res / sh / "exp8" / "summary.json")
        e10 = _load(res / sh / "exp10" / "summary.json")
        lkm = (e2.get("locked") or {}).get("steer_eval") or {}
        if lkm.get("mean") is not None and lkm["mean"] > 0:
            inv.append(_short(sh))
        p4 = (e8.get("percentiles_all") or {}).get("alpha4") or {}
        if p4.get("d") is not None:
            pcts.append(p4["d"])
            if p4["d"] >= 95:
                sel.append(_short(sh))
        sw8 = e8.get("swings") or {}
        if sw8.get("style_neutral") and sw8.get("d"):
            if abs(sw8["style_neutral"]["mean"]) > abs(sw8["d"]["mean"]) and (p4.get("style_neutral") or 0) >= 95:
                neutral_beats.append(_short(sh))
        if e8.get("cos_d_style_neutral_at_layer") is not None:
            cosn.append(e8["cos_d_style_neutral_at_layer"])
        if e8.get("frac_d_surviving_neutral_removal") is not None:
            fracn.append(e8["frac_d_surviving_neutral_removal"])
        ef, eq = e10.get("eval_final_class") or {}, e10.get("eval_question_class") or {}
        if ef.get("mean") is not None and eq.get("mean") is not None:
            (locus_final if abs(ef["mean"]) >= abs(eq["mean"]) else locus_q).append(_short(sh))
        if e1.get("verbal_auc") is not None:
            vAUC.append(e1["verbal_auc"])
        if e1.get("repr_verbal_spearman") is not None:
            rhos.append(e1["repr_verbal_spearman"])
        cp = res / sh / "exp1" / "probe_curve.csv"
        if cp.exists():
            l0s.append(pd.read_csv(cp).sort_values("layer").test_auc.iloc[0])
    # models whose framing effect on accuracy excludes zero, with the value
    fsig, task_abs = [], []
    for sh in shorts:
        pit = res / sh / "exp3" / "sandbag_items.csv"
        if pit.exists():
            try:
                it = pd.read_csv(pit).pivot_table(index="item_id", columns=["framing", "system"],
                                                  values="p_correct")
                fr = ((it[("deploy", "incentive")] + it[("deploy", "neutral")]) / 2
                      - (it[("eval", "incentive")] + it[("eval", "neutral")]) / 2).dropna()
                m_, lo_, hi_ = bootstrap_mean_ci(fr.to_numpy(), clusters=fr.index.to_numpy(), seed=0)
                if lo_ > 0 or hi_ < 0:
                    fsig.append(f"{_short(sh)} ({m_:+.2f} [{lo_:+.2f}, {hi_:+.2f}])")
            except Exception:
                pass
        e10 = _load(res / sh / "exp10" / "summary.json")
        for dc in ("eval", "deploy"):
            v = e10.get(f"{dc}_task_class") or {}
            if v.get("mean") is not None:
                task_abs.append(abs(v["mean"]))
    n["VframingSigModels"] = ", ".join(fsig) if fsig else "none"
    n["VnFramingNull"] = str(len(shorts) - len(fsig))
    n["VtaskMaxAbs"] = fmt(max(task_abs), 2) if task_abs else "--"

    def _join(xs):
        return ", ".join(xs) if xs else "none"
    n["VselectiveModels"] = _join(sel)
    n["VnSelective"] = str(len(sel))
    n["VinvertedModels"] = _join(inv)
    n["VpctDmin"] = fmt(min(pcts), 0) if pcts else "--"
    n["VpctDmax"] = fmt(max(pcts), 0) if pcts else "--"
    n["VneutralBeatsD"] = _join(neutral_beats)
    n["VcosNeutralMin"] = fmt(min(cosn), 2) if cosn else "--"
    n["VcosNeutralMax"] = fmt(max(cosn), 2) if cosn else "--"
    n["VfracNeutralMin"] = (fmt(100 * min(fracn), 0) + r"\%") if fracn else "--"
    n["VfracNeutralMax"] = (fmt(100 * max(fracn), 0) + r"\%") if fracn else "--"
    n["VlocusFinalModels"] = _join(locus_final)
    n["VlocusQuestionModels"] = _join(locus_q)
    n["VverbalAUCmin"] = fmt(min(vAUC), 2) if vAUC else "--"
    n["VverbalAUCmax"] = fmt(max(vAUC), 2) if vAUC else "--"
    n["VrhoMin"] = fmt(min(rhos), 2) if rhos else "--"
    n["VrhoMax"] = fmt(max(rhos), 2) if rhos else "--"
    n["VprobeLzeroMin"] = fmt(min(l0s), 2) if l0s else "--"
    n["VprobeLzeroMax"] = fmt(max(l0s), 2) if l0s else "--"
    n["VnModels"] = str(len(shorts))

    # register table
    rrows = []
    for sh in shorts:
        e8 = _load(res / sh / "exp8" / "summary.json")
        if not e8:
            continue
        sw8 = e8.get("swings") or {}
        p4 = (e8.get("percentiles_all") or {}).get("alpha4") or {}
        rp8 = (e8.get("random_p95") or {}).get("alpha4")

        def _s(k):
            v = sw8.get(k)
            return f"{v['mean']:+.2f}" if v else "--"

        def _p(k):
            v = p4.get(k)
            return f"{v:.0f}" if v is not None else "--"

        rrows.append({
            "model": _disp(res, sh),
            "cos(d, style)": fmt(e8.get("cos_d_style_at_layer"), 2),
            "cos(d, neutral style)": fmt(e8.get("cos_d_style_neutral_at_layer"), 2),
            "swing d (pct)": f"{_s('d')} ({_p('d')})",
            "swing style (pct)": f"{_s('style')} ({_p('style')})",
            "swing neutral style (pct)": f"{_s('style_neutral')} ({_p('style_neutral')})",
            "d minus neutral style": f"{_s('d_perp_style_neutral')} ({_p('d_perp_style_neutral')})",
            "random p95": fmt(rp8, 2),
            "n random": str((e8.get("random_null") or {}).get("n", "--")),
        })
    if rrows:
        (paper / "tables" / "register_v2.tex").write_text(
            pd.DataFrame(rrows).to_latex(index=False, escape=False, na_rep="--"))
        print(f"  register_v2: {len(rrows)} models")

    # positional table with Holm-corrected markers
    prow, pvals, keys = [], [], []
    for sh in shorts:
        raw_p = res / sh / "exp10" / "span_direction_patch.csv"
        e10 = _load(res / sh / "exp10" / "summary.json")
        if not raw_p.exists() or not e10:
            continue
        raw = pd.read_csv(raw_p)
        dl = e10.get("deep_layer")
        for dc in ("eval", "deploy"):
            for region in ("task", "question", "final"):
                sub = raw[(raw.layer == dl) & (raw.region == region) & (raw.dst_cls == dc)]
                pv = sub.pivot_table(index="content_id", columns="source", values="delta").dropna()
                need = {"othercls_samecont", "othercls_diffcont", "samecls_samecont", "samecls_diffcont"}
                if not need <= set(pv.columns) or not len(pv):
                    continue
                cls = ((pv.othercls_samecont + pv.othercls_diffcont) / 2
                       - (pv.samecls_samecont + pv.samecls_diffcont) / 2).to_numpy()
                m, lo, hi = bootstrap_mean_ci(cls, clusters=pv.index.to_numpy(), seed=0)
                p = boot_p(cls, pv.index.to_numpy())
                keys.append((sh, dc, region))
                pvals.append(p)
                prow.append(dict(sh=sh, model=_disp(res, sh), dst=dc, region=region,
                                 mean=m, lo=lo, hi=hi, p=p))
    if prow:
        adj = np.array(holm(np.array(pvals)))
        pdf = pd.DataFrame(prow)
        pdf["holm"] = adj
        pdf.to_csv(paper / "tables" / "position_v2_raw.csv", index=False)
        out = []
        lab = {"task": "pre-question", "question": "question", "final": "final position"}
        for (sh, model, dc), sub in pdf.groupby(["sh", "model", "dst"], sort=False):
            cells = {}
            for _, r in sub.iterrows():
                mark = r"$^{\dagger}$" if r.holm < 0.05 else (r"$^{*}$" if r.p < 0.05 else "")
                cells[lab[r.region]] = f"{r['mean']:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]{mark}"
            out.append({"model": model, "destination": f"{dc} dest.", **cells})
        (paper / "tables" / "position_v2.tex").write_text(
            pd.DataFrame(out).to_latex(index=False, escape=False, na_rep="--"))
        n["VposCells"] = str(len(pvals))
        n["VposUncorrected"] = str(int((np.array(pvals) < 0.05).sum()))
        n["VposHolm"] = str(int((adj < 0.05).sum()))
        print(f"  position_v2: {len(out)} rows; {n['VposUncorrected']} uncorrected, {n['VposHolm']} Holm")

    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    (paper / "numbers_v2.tex").write_text("% auto-generated - do not edit\n" + "\n".join(lines) + "\n")
    print(f"wrote {paper / 'numbers_v2.tex'} with {len(n)} macros")


if __name__ == "__main__":
    main()
