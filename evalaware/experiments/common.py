"""Shared experiment context: config, model (lazy), directories, artifacts."""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NS, results_dir
from ..model_io import load_model
from ..utils import load_jsonl, log, save_json, set_seed


class Ctx:
    def __init__(self, cfg: NS):
        self.cfg = cfg
        self._model = None
        self.out: Path = results_dir(cfg)
        set_seed(int(cfg.data.seed))

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.cfg.model)
        return self._model

    def dir(self, name: str) -> Path:
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------- dataset artifacts ----------------
    def corpus(self) -> list[dict]:
        p = self.out / "dataset" / "corpus.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run exp0 first")
        return load_jsonl(p)

    def behaviors(self) -> dict:
        import json

        p = self.out / "dataset" / "behaviors.json"
        with open(p) as f:
            return json.load(f)

    # ---------------- exp1 artifacts ----------------
    def directions(self) -> tuple[np.ndarray, np.ndarray, dict]:
        d = self.out / "exp1"
        dirs = np.load(d / "directions.npy")
        sigma = np.load(d / "sigma.npy")
        import json

        with open(d / "summary.json") as f:
            summary = json.load(f)
        return dirs, sigma, summary

    def acts(self) -> tuple[np.ndarray, pd.DataFrame]:
        d = self.out / "exp1"
        X = np.load(d / "final_resid.npy")
        labels = pd.read_csv(d / "labels.csv")
        return X, labels

    def write_manifest(self, exp: str, extra: dict | None = None) -> None:
        import torch
        import transformer_lens

        try:
            tl_version = transformer_lens.__version__
        except AttributeError:  # TransformerLens >= 3 dropped the module attribute
            from importlib.metadata import PackageNotFoundError, version

            try:
                tl_version = version("transformer-lens")
            except PackageNotFoundError:
                tl_version = "unknown"

        try:
            real_dtype = str(next(self._model.parameters()).dtype) if self._model is not None else None
        except Exception:
            real_dtype = None
        info = dict(
            experiment=exp,
            model=self.cfg.model.name,
            model_dtype=real_dtype,
            time=datetime.now().isoformat(timespec="seconds"),
            python=platform.python_version(),
            torch=torch.__version__,
            transformer_lens=tl_version,
            config=self.cfg.to_dict(),
        )
        if extra:
            info.update(extra)
        save_json(self.dir(exp) / "manifest.json", info)
        log.info(f"[{exp}] manifest written")


def sweep_layers(ctx: Ctx) -> list[int]:
    L = ctx.model.cfg.n_layers
    stride = int(ctx.cfg.interventions.layer_stride)
    layers = list(range(0, L, stride))
    if (L - 1) not in layers:
        layers.append(L - 1)
    return layers
