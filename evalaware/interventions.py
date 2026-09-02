"""Causal intervention primitives on the residual stream.

Three families:
  steer   resid += alpha * sigma * d_hat          (alignment-free; works in generation)
  ablate  resid -= (resid . d_hat) d_hat          (removes the direction; span-able)
  patch   resid[-k:] <- other run's resid[-k:]    (paired prompts, end-aligned)

End alignment: eval/deploy prompt pairs share their entire suffix (task +
belief question + chat-template tail) by construction; only the cue prefix
differs. `common_suffix_len` measures the shared suffix in tokens, and patching
swaps exactly that span. During generation with KV caching, later forward
passes see single-token inputs; patch hooks guard on sequence length so the
prompt-pass patch persists through the KV cache while generation steps pass
through untouched.
"""
from __future__ import annotations

import numpy as np
import torch

from .model_io import single_tokens


def resid_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def _as_dir(direction) -> torch.Tensor:
    d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    return d / d.norm().clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Steering / ablation hooks
# --------------------------------------------------------------------------- #
def steer_hook(direction, coeff: float):
    """Add coeff * d_hat at every position (prompt and generated tokens)."""
    d = _as_dir(direction)

    def hook(resid, **_kw):
        dd = d.to(device=resid.device, dtype=resid.dtype)
        return resid + coeff * dd

    return hook


def ablate_hook(direction):
    """Project d_hat out of the residual stream at every position."""
    d = _as_dir(direction)

    def hook(resid, **_kw):
        dd = d.to(device=resid.device, dtype=resid.dtype)
        proj = torch.einsum("bpd,d->bp", resid, dd)
        return resid - proj.unsqueeze(-1) * dd

    return hook


def ablate_span_hook(direction, start: int, end: int):
    """Project d_hat out only at absolute token positions [start, end). batch=1 use."""
    d = _as_dir(direction)

    def hook(resid, **_kw):
        T = resid.shape[1]
        if T <= start:  # generation step or shorter prompt: skip
            return resid
        e = min(end, T)
        dd = d.to(device=resid.device, dtype=resid.dtype)
        seg = resid[:, start:e, :]
        proj = torch.einsum("bpd,d->bp", seg, dd)
        resid[:, start:e, :] = seg - proj.unsqueeze(-1) * dd
        return resid

    return hook


def random_direction(d_model: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d_model)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Activation patching (paired, end-aligned)
# --------------------------------------------------------------------------- #
def common_suffix_len(ids_a: torch.Tensor, ids_b: torch.Tensor) -> int:
    """Length of the shared token suffix of two (1, T) id tensors.

    Capped at min(Ta, Tb) - 1 so position 0 (BOS/template head) is never patched.
    """
    a, b = ids_a[0].tolist(), ids_b[0].tolist()
    m = min(len(a), len(b)) - 1
    k = 0
    while k < m and a[len(a) - 1 - k] == b[len(b) - 1 - k]:
        k += 1
    return k


def patch_last_k_hook(src_resid: torch.Tensor, k: int):
    """Copy the source run's last-k residuals into the current run.

    src_resid: (1, T_src, d) tensor (kept on CPU fp32; cast lazily).
    Applies only when the current pass covers >= k positions (the prompt pass).
    """
    src_tail = src_resid[:, -k:, :].clone()

    def hook(resid, **_kw):
        if resid.shape[1] < k:
            return resid
        resid[:, -k:, :] = src_tail.to(device=resid.device, dtype=resid.dtype)
        return resid

    return hook


def cache_resid(model, ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """resid_post at the given layers for one prompt. CPU fp32, (1, T, d) each."""
    want = {resid_name(l) for l in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(ids, names_filter=lambda n: n in want)
    return {l: cache[resid_name(l)].float().cpu() for l in layers}


def patched_final_logits(
    model,
    ids_dst: torch.Tensor,
    layer: int,
    src_resid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Final-position logits of the destination prompt with resid[-k:] at
    `layer` replaced by the source run's. (1, d_vocab) fp32 cpu."""
    hook = patch_last_k_hook(src_resid, k)
    with torch.no_grad(), model.hooks(fwd_hooks=[(resid_name(layer), hook)]):
        logits = model(ids_dst)
    return logits[:, -1, :].float().cpu()


def tokens_for(model, text: str) -> torch.Tensor:
    return single_tokens(model, text)


# --------------------------------------------------------------------------- #
# Node-level patching (used by Experiment E2)
# --------------------------------------------------------------------------- #
def node_hook_name(node: dict) -> str:
    if node["type"] == "head":
        return f"blocks.{node['layer']}.attn.hook_z"
    if node["type"] == "mlp":
        return f"blocks.{node['layer']}.hook_mlp_out"
    raise ValueError(node)


def patch_node_hook(node: dict, src_act: torch.Tensor, k: int):
    """Patch one component's activation over the last k positions.

    src_act: source-run activation, full shape as cached:
      head: (1, T, n_heads, d_head)   mlp: (1, T, d_model)
    """
    if node["type"] == "head":
        tail = src_act[:, -k:, node["head"], :].clone()

        def hook(act, **_kw):
            if act.shape[1] < k:
                return act
            act[:, -k:, node["head"], :] = tail.to(device=act.device, dtype=act.dtype)
            return act

    else:
        tail = src_act[:, -k:, :].clone()

        def hook(act, **_kw):
            if act.shape[1] < k:
                return act
            act[:, -k:, :] = tail.to(device=act.device, dtype=act.dtype)
            return act

    return hook
