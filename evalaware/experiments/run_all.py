"""Orchestrator: run any subset of experiments with one model load.

Usage:
  python run.py --config configs/default.yaml --model-config configs/models/qwen7b.yaml
  python run.py --config configs/smoke.yaml                 # tiny end-to-end check
  python run.py ... --exps 0,1,2 --quick                    # subset, shrunken sizes
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ..config import load_config
from ..utils import log, timer
from .common import Ctx

EXPS = {
    "0": ("exp0_dataset", "build corpus + behavior suites"),
    "1": ("exp1_probes", "probes, directions, grids"),
    "2": ("exp2_causal_belief", "Experiment A: patch/steer the belief"),
    "3": ("exp3_behavior", "Experiment B: behavior under intervention"),
    "4": ("exp4_stages", "Experiment C: stage table + dissociation"),
    "5": ("exp5_positions", "Experiment D: token-position causal map"),
    "6": ("exp6_circuit", "Experiment E: attribution -> verification -> d"),
    "7": ("exp7_causal_anatomy", "Experiment F: artifact anatomy + multi-layer ablation"),
    "8": ("exp8_register", "Experiment G: register decomposition of d"),
    "9": ("exp9_generality", "Experiment H: concept-generality of the control confound"),
    "10": ("exp10_span_directions", "Experiment I: directional span 2x2 + distances"),
    "11": ("exp11_distance_matched", "Experiment J: distance-matched source 2x2"),
}


def _parse_exps(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        elif part:
            out.append(part)
    return [e for e in out if e in EXPS]


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-config", default=None,
                    help="configs/models/*.yaml overlay")
    ap.add_argument("--exps", default="0-6", help="e.g. '0-3' or '1,4'")
    ap.add_argument("--quick", action="store_true",
                    help="shrink dataset/sweep sizes (smoke.yaml sizes, same model)")
    args = ap.parse_args(argv)

    # default.yaml is always the base; --config and --model-config overlay it.
    default_cfg = str(Path(args.config).parent / "default.yaml")
    paths = [default_cfg]
    if str(Path(args.config)) != default_cfg:
        paths.append(args.config)
    if args.model_config:
        paths.append(args.model_config)
    cfg = load_config(*paths)

    if args.quick:
        smoke = load_config(Path(args.config).parent / "smoke.yaml").to_dict()
        smoke.pop("model", None)  # keep the chosen model
        merged = cfg.to_dict()
        from ..config import _deep_merge

        cfg = type(cfg)(_deep_merge(merged, smoke))

    ctx = Ctx(cfg)
    todo = _parse_exps(args.exps)
    log.info(f"running experiments {todo} for {cfg.model.name} -> {ctx.out}")

    failed = []
    for e in todo:
        mod_name, desc = EXPS[e]
        log.info(f"=== exp{e}: {desc} ===")
        try:
            import importlib

            mod = importlib.import_module(f"evalaware.experiments.{mod_name}")
            with timer(f"exp{e} total"):
                mod.run(ctx)
        except Exception:
            traceback.print_exc()
            failed.append(e)
            log.info(f"exp{e} FAILED; continuing")
    if failed:
        log.info(f"finished with failures: {failed}")
        return 1
    log.info("all requested experiments finished")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
