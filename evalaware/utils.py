"""Shared utilities: seeding, logging, IO, small helpers."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

log = logging.getLogger("evalaware")
if not log.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def batched(seq: Sequence, n: int) -> Iterator[list]:
    buf: list = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    log.info(f"{name}: {time.time() - t0:.1f}s")


def find_span(text: str, needle: str) -> tuple[int, int] | None:
    """Character span of `needle` in `text` (first occurrence), or None.

    Falls back to matching a distinctive prefix of the needle, which tolerates
    chat templates that re-wrap whitespace.
    """
    if not needle:
        return None
    i = text.find(needle)
    if i >= 0:
        return (i, i + len(needle))
    head = needle[: min(24, len(needle))].strip()
    if len(head) >= 8:
        i = text.find(head)
        if i >= 0:
            return (i, i + len(needle))
    return None
