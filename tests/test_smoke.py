"""End-to-end smoke test on a tiny model (~10-20 min CPU, faster on MPS).

Run:  pytest tests/test_smoke.py -s
or:   python run.py --config configs/smoke.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_pipeline_end_to_end(tmp_path):
    from evalaware.config import load_config
    from evalaware.experiments.common import Ctx
    from evalaware.experiments import (  # noqa: F401
        exp0_dataset, exp1_probes, exp2_causal_belief, exp3_behavior,
        exp4_stages, exp5_positions, exp6_circuit,
    )

    cfg_d = load_config(ROOT / "configs" / "default.yaml",
                        ROOT / "configs" / "smoke.yaml").to_dict()
    cfg_d["output_dir"] = str(tmp_path / "results")
    from evalaware.config import NS

    ctx = Ctx(NS(cfg_d))

    exp0_dataset.run(ctx)
    corpus = ctx.corpus()
    assert len(corpus) > 20
    assert any(r["split"] == "test" for r in corpus)

    exp1_probes.run(ctx)
    with open(ctx.out / "exp1" / "summary.json") as f:
        s1 = json.load(f)
    assert 0.0 <= s1["best_test_auc"] <= 1.0
    assert (ctx.out / "exp1" / "directions.npy").exists()

    exp2_causal_belief.run(ctx)
    assert (ctx.out / "exp2" / "agg_patch.csv").exists()

    exp3_behavior.run(ctx)
    assert (ctx.out / "exp3" / "sandbag_baseline.csv").exists()

    exp4_stages.run(ctx)
    assert (ctx.out / "exp4" / "stage_table.csv").exists()

    exp5_positions.run(ctx)
    assert (ctx.out / "exp5" / "span_ablation.csv").exists()

    exp6_circuit.run(ctx)
    assert (ctx.out / "exp6" / "attribution.csv").exists()
