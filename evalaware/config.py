"""Config loading: YAML files deep-merged left-to-right, dot-access namespace."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class NS:
    """Minimal dot-access wrapper over nested dicts."""

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k: str) -> Any:
        if k.startswith("_"):
            raise AttributeError(k)
        try:
            v = self._d[k]
        except KeyError as e:
            raise AttributeError(f"config key missing: {k!r} (have {sorted(self._d)})") from e
        return NS(v) if isinstance(v, dict) else v

    def get(self, k: str, default: Any = None) -> Any:
        v = self._d.get(k, default)
        return NS(v) if isinstance(v, dict) else v

    def to_dict(self) -> dict:
        return copy.deepcopy(self._d)

    def __repr__(self) -> str:
        return f"NS({json.dumps(self._d, indent=2, default=str)[:2000]})"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path) -> NS:
    """Merge YAML files left-to-right (later files win). First is usually default.yaml."""
    merged: dict = {}
    for p in paths:
        if p is None:
            continue
        with open(p) as f:
            d = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, d)
    if not merged:
        raise ValueError("no config loaded")
    return NS(merged)


def results_dir(cfg: NS) -> Path:
    d = Path(cfg.get("output_dir", "results")) / cfg.model.short
    d.mkdir(parents=True, exist_ok=True)
    return d
