"""Residual-stream caching.

Two products:
  * final-token residuals at every layer  -> probes, directions      [n, L, d]
  * end-aligned (layer x position) grids  -> construction maps (exp5) [n, L, K, d]

Stored float16 on CPU/np to keep a 7B run around ~1 GB.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .model_io import batch_tokens, single_tokens, _supports_mask
from .utils import batched


def _resid_filter(name: str) -> bool:
    return name.endswith("hook_resid_post")


def cache_final_resid(model, texts: list[str], batch_size: int = 8) -> np.ndarray:
    """[n, n_layers, d_model] float16: resid_post at the final token, every layer."""
    L = model.cfg.n_layers
    use_mask = _supports_mask(model)
    bs = batch_size if use_mask else 1
    outs = []
    for chunk in tqdm(list(batched(texts, bs)), desc="cache final resid", leave=False):
        ids, mask = batch_tokens(model, chunk)
        with torch.no_grad():
            if use_mask:
                _, cache = model.run_with_cache(ids, attention_mask=mask, names_filter=_resid_filter)
            else:
                _, cache = model.run_with_cache(ids, names_filter=_resid_filter)
        # (L, batch, d) at final position (left padding -> -1 is content-final)
        x = torch.stack([cache[f"blocks.{l}.hook_resid_post"][:, -1, :] for l in range(L)], dim=0)
        outs.append(x.transpose(0, 1).to(torch.float16).cpu())
        del cache
    return torch.cat(outs, dim=0).numpy()


def cache_grid_resid(model, texts: list[str], last_k: int) -> tuple[np.ndarray, np.ndarray]:
    """End-aligned grids, batch=1 (no padding ambiguity).

    Returns (grid [n, L, K, d] float16 with NaN where the prompt is shorter
    than K, lengths [n]).
    """
    L, d = model.cfg.n_layers, model.cfg.d_model
    n = len(texts)
    grid = np.full((n, L, last_k, d), np.nan, dtype=np.float16)
    lengths = np.zeros(n, dtype=np.int64)
    for i, t in enumerate(tqdm(texts, desc="cache grid resid", leave=False)):
        ids = single_tokens(model, t)
        T = ids.shape[1]
        lengths[i] = T
        k = min(last_k, T)
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=_resid_filter)
        x = torch.stack(
            [cache[f"blocks.{l}.hook_resid_post"][0, -k:, :] for l in range(L)], dim=0
        )  # (L, k, d)
        grid[i, :, last_k - k:, :] = x.to(torch.float16).cpu().numpy()
        del cache
    return grid, lengths
