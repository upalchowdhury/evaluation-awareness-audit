"""Statistics helpers: cluster bootstrap CIs, effect sizes, multiple comparisons."""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values,
    clusters=None,
    iters: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean with a (cluster) bootstrap CI.

    If `clusters` is given (e.g. content ids), whole clusters are resampled,
    which respects the fact that several framings of one content item are not
    independent observations.
    """
    x = np.asarray(values, dtype=float)
    keep = ~np.isnan(x)
    x = x[keep]
    if clusters is not None:
        clusters = np.asarray(clusters)[keep]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx_sets = rng.integers(0, len(x), size=(iters, len(x)))
        means = x[idx_sets].mean(axis=1)
    else:
        c = np.asarray(clusters)
        uniq = np.unique(c)
        by = {u: x[c == u] for u in uniq}
        means = np.empty(iters)
        for i in range(iters):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            means[i] = np.concatenate([by[u] for u in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(x.mean()), float(lo), float(hi))


def paired_diff_ci(a, b, clusters=None, iters: int = 2000, seed: int = 0):
    """Bootstrap CI on mean(a - b) for paired arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bootstrap_mean_ci(a - b, clusters=clusters, iters=iters, seed=seed)


def cohen_d_paired(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else np.nan


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 1e-12 else np.nan
