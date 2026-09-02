"""Experiment E: which components create, carry, and use the direction d?

E1  Attribution patching (Nanda 2023; Syed et al. 2023): a first-order
    estimate of every head/MLP's effect on the belief metric, from ONE clean
    forward+backward and one corrupt forward per pair:
        attr(node) ~ sum_{suffix pos} (a_clean - a_corrupt) . dL/da_clean
E2  Causal verification: actually patch the top-|attr| nodes (deploy source ->
    eval run), one at a time and jointly, and measure the belief change.
    Attribution proposes; patching disposes.
E3  Connection to d: does each top component WRITE along d (its output's
    projection onto d differs by class) or READ d (its Q/V weights are
    unusually aligned with d relative to random directions)?

Gradient trick: all parameters are frozen; the embedding output is flagged
requires_grad so autograd builds an activation graph without allocating any
parameter gradients. Run this on fp32 for trustworthy gradients (use
`attribution.model_override` to point at a smaller sibling model if the
primary is fp16).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .interventions import common_suffix_len, node_hook_name, patch_node_hook
from .model_io import single_tokens
from .utils import log


def _metric_ids(model) -> tuple[list[int], list[int]]:
    from .readouts import yes_no_ids

    return yes_no_ids(model)


def _belief_metric(model, logits: torch.Tensor) -> torch.Tensor:
    """Scalar polarity-A belief: logsumexp(Yes) - logsumexp(No) at final pos."""
    yes, no = _metric_ids(model)
    row = logits[0, -1]
    return torch.logsumexp(row[yes], 0) - torch.logsumexp(row[no], 0)


def freeze_params(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# E1: attribution patching
# --------------------------------------------------------------------------- #
def attribution_patching(model, pair_texts: list[tuple[str, str]]) -> pd.DataFrame:
    """pair_texts: (clean_eval_text, corrupt_deploy_text), belief suffix included.

    Returns one row per node {type, layer, head, attr} averaged over pairs.
    """
    freeze_params(model)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    z_names = {f"blocks.{l}.attn.hook_z": l for l in range(L)}
    m_names = {f"blocks.{l}.hook_mlp_out": l for l in range(L)}
    watch = set(z_names) | set(m_names)

    head_attr = np.zeros((L, H))
    mlp_attr = np.zeros(L)
    n_used = 0

    for i, (clean_text, corrupt_text) in enumerate(pair_texts):
        ids_c = single_tokens(model, clean_text)
        ids_r = single_tokens(model, corrupt_text)
        unpaired_text = pair_texts[(i + 1) % len(pair_texts)][1]
        ids_un = single_tokens(model, unpaired_text)
        k = common_suffix_len(ids_c, ids_r)
        ku = common_suffix_len(ids_c, ids_un)
        if k < 2 or ku < 2:
            continue

        with torch.no_grad():
            _, cache_r = model.run_with_cache(ids_r, names_filter=lambda n: n in watch)
        corrupt = {n: cache_r[n][:, -k:].float().cpu() for n in watch}
        del cache_r

        saved: dict[str, torch.Tensor] = {}

        def grad_root(t, hook):
            t.requires_grad_(True)
            return t

        def save(t, hook):
            t.retain_grad()
            saved[hook.name] = t
            return t

        hooks = [("hook_embed", grad_root)] + [(n, save) for n in watch]
        with torch.enable_grad(), model.hooks(fwd_hooks=hooks):
            logits = model(ids_c)
            metric = _belief_metric(model, logits)
            metric.backward()

        for n in watch:
            act = saved[n]
            if act.grad is None:
                continue
            diff = act[:, -k:].detach().float().cpu() - corrupt[n]
            g = act.grad[:, -k:].float().cpu()
            contrib = diff * g
            if n in z_names:  # (1, k, H, d_head)
                head_attr[z_names[n]] += contrib.sum(dim=(0, 1, 3)).numpy()
            else:  # (1, k, d_model)
                mlp_attr[m_names[n]] += contrib.sum().item()
        saved.clear()
        model.zero_grad(set_to_none=True)
        n_used += 1
        log.info(f"attribution pair {n_used}/{len(pair_texts)} done (k={k})")

    if n_used == 0:
        raise RuntimeError("no usable pairs for attribution (shared suffixes too short)")
    head_attr /= n_used
    mlp_attr /= n_used

    rows = [dict(type="head", layer=l, head=h, attr=float(head_attr[l, h]))
            for l in range(L) for h in range(H)]
    rows += [dict(type="mlp", layer=l, head=-1, attr=float(mlp_attr[l])) for l in range(L)]
    df = pd.DataFrame(rows)
    df["abs_attr"] = df["attr"].abs()
    return df.sort_values("abs_attr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# E2: causal verification of top nodes
# --------------------------------------------------------------------------- #
def verify_nodes(
    model,
    pair_texts: list[tuple[str, str]],
    nodes: list[dict],
) -> pd.DataFrame:
    """Patch each node (deploy -> eval run, aligned suffix) and measure the
    belief drop. Also patches all `nodes` jointly ("joint" row).

    Returns rows {node_label, mean_delta, per-pair deltas serialized}.
    """
    names = sorted({node_hook_name(nd) for nd in nodes})
    deltas: dict[str, list[float]] = {_label(nd): [] for nd in nodes}
    deltas["JOINT"] = []
    # UNPAIRED_JOINT: the same joint patch sourced from a DIFFERENT pair's
    # deploy prompt. If it matches JOINT, the nodes are moving the readout
    # generically rather than carrying eval/deploy content (the same artifact
    # exp2's unpaired control exposes at the residual level).
    deltas["UNPAIRED_JOINT"] = []
    # FULL is the reference the joint patch is trying to reproduce: the belief
    # metric on the deploy prompt itself, minus the eval baseline. Reported as a
    # row so the "% of the framing effect" number survives to CSV.
    deltas["FULL"] = []
    baselines: list[float] = []

    for i, (clean_text, corrupt_text) in enumerate(pair_texts):
        ids_c = single_tokens(model, clean_text)
        ids_r = single_tokens(model, corrupt_text)
        unpaired_text = pair_texts[(i + 1) % len(pair_texts)][1]
        ids_un = single_tokens(model, unpaired_text)
        k = common_suffix_len(ids_c, ids_r)
        ku = common_suffix_len(ids_c, ids_un)
        if k < 2 or ku < 2:
            continue
        with torch.no_grad():
            base = _belief_metric(model, model(ids_c)).item()
            deltas["FULL"].append(_belief_metric(model, model(ids_r)).item() - base)
            _, cache_r = model.run_with_cache(ids_r, names_filter=lambda n: n in names)
        src = {n: cache_r[n].float().cpu() for n in names}
        del cache_r
        baselines.append(base)

        for nd in nodes:
            hk = patch_node_hook(nd, src[node_hook_name(nd)], k)
            with torch.no_grad(), model.hooks(fwd_hooks=[(node_hook_name(nd), hk)]):
                m = _belief_metric(model, model(ids_c)).item()
            deltas[_label(nd)].append(m - base)

        joint_hooks = [(node_hook_name(nd), patch_node_hook(nd, src[node_hook_name(nd)], k))
                       for nd in nodes]
        with torch.no_grad(), model.hooks(fwd_hooks=joint_hooks):
            m = _belief_metric(model, model(ids_c)).item()
        deltas["JOINT"].append(m - base)

        with torch.no_grad():
            _, cache_un = model.run_with_cache(ids_un, names_filter=lambda n: n in names)
        src_un = {n: cache_un[n].float().cpu() for n in names}
        del cache_un
        un_hooks = [(node_hook_name(nd), patch_node_hook(nd, src_un[node_hook_name(nd)], ku))
                    for nd in nodes]
        with torch.no_grad(), model.hooks(fwd_hooks=un_hooks):
            m = _belief_metric(model, model(ids_c)).item()
        deltas["UNPAIRED_JOINT"].append(m - base)

    base_mean = float(np.mean(baselines)) if baselines else float("nan")
    rows = []
    for label, ds in deltas.items():
        arr = np.array(ds, dtype=float)
        rows.append(dict(node=label, mean_delta=float(arr.mean()) if len(arr) else np.nan,
                         std_delta=float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
                         n=len(arr), baseline_mean=base_mean))
    out = pd.DataFrame(rows)
    out.attrs["baseline_mean"] = base_mean
    return out


def _label(nd: dict) -> str:
    return f"L{nd['layer']}H{nd['head']}" if nd["type"] == "head" else f"MLP{nd['layer']}"


def parse_label(label: str) -> dict:
    if label.startswith("MLP"):
        return dict(type="mlp", layer=int(label[3:]), head=-1)
    l, h = label[1:].split("H")
    return dict(type="head", layer=int(l), head=int(h))


# --------------------------------------------------------------------------- #
# E3: connection of components to the direction d
# --------------------------------------------------------------------------- #
def write_read_scores(
    model,
    eval_texts: list[str],
    deploy_texts: list[str],
    nodes: list[dict],
    direction: np.ndarray,
    n_random: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """For each node:
      write_score  E_eval[out . d_hat] - E_deploy[out . d_hat] at the final
                   token (out = the component's actual write into the residual
                   stream), normalized by the pooled std of the projection.
      read_ratio_v ||d_hat @ W_V|| relative to random unit directions
      read_ratio_q ||d_hat @ W_Q|| relative to random unit directions
    """
    d_hat = torch.tensor(np.asarray(direction), dtype=torch.float32)
    d_hat = d_hat / d_hat.norm()
    names = sorted({node_hook_name(nd) for nd in nodes})

    def final_writes(texts: list[str]) -> dict[str, list[float]]:
        vals: dict[str, list[float]] = {_label(nd): [] for nd in nodes}
        for t in texts:
            ids = single_tokens(model, t)
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
            for nd in nodes:
                if nd["type"] == "head":
                    z = cache[node_hook_name(nd)][0, -1, nd["head"], :].float().cpu()
                    W_O = model.blocks[nd["layer"]].attn.W_O[nd["head"]].float().cpu()
                    out = z @ W_O
                else:
                    out = cache[node_hook_name(nd)][0, -1, :].float().cpu()
                vals[_label(nd)].append(float(out @ d_hat))
            del cache
        return vals

    ev, dp = final_writes(eval_texts), final_writes(deploy_texts)

    rng = np.random.default_rng(seed)
    rand_dirs = rng.standard_normal((n_random, len(d_hat))).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_t = torch.tensor(rand_dirs)

    n_heads = model.cfg.n_heads
    n_kv = getattr(model.cfg, "n_key_value_heads", None) or n_heads

    rows = []
    for nd in nodes:
        lab = _label(nd)
        e, p = np.array(ev[lab]), np.array(dp[lab])
        pooled = np.concatenate([e, p]).std(ddof=1) + 1e-8
        write_score = float((e.mean() - p.mean()) / pooled)
        if nd["type"] == "head":
            l, h = nd["layer"], nd["head"]
            kv = h * n_kv // n_heads
            W_V = model.blocks[l].attn.W_V
            W_V_h = (W_V[kv] if W_V.shape[0] == n_kv else W_V[h]).float().cpu()
            W_Q_h = model.blocks[l].attn.W_Q[h].float().cpu()
            v_read = float((d_hat @ W_V_h).norm())
            q_read = float((d_hat @ W_Q_h).norm())
            v_rand = float((rand_t @ W_V_h).norm(dim=1).mean())
            q_rand = float((rand_t @ W_Q_h).norm(dim=1).mean())
            rows.append(dict(node=lab, write_score=write_score,
                             read_ratio_v=v_read / (v_rand + 1e-8),
                             read_ratio_q=q_read / (q_rand + 1e-8)))
        else:
            rows.append(dict(node=lab, write_score=write_score,
                             read_ratio_v=np.nan, read_ratio_q=np.nan))
    return pd.DataFrame(rows)
