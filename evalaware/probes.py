"""Linear probing of the eval/deploy distinction, and direction extraction.

Rigor notes
-----------
* Cross-validation is GROUPED BY CONTENT: no probe is scored on a content item
  it saw under any framing (blocks "the probe memorized the question" leaks).
* Steering/patching directions are MASS-MEAN (difference of class means),
  fit on train contents only - simple, reproducible, and the standard choice
  for concept steering. Logistic probes are used for decoding accuracy.
* The permutation test and the (layer x position) grid use the mass-mean
  projection AUC, which is closed-form and cheap enough to permute honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .utils import log


# --------------------------------------------------------------------------- #
# Core fits
# --------------------------------------------------------------------------- #
def _logreg(c_reg: float) -> LogisticRegression:
    return LogisticRegression(C=c_reg, max_iter=3000, class_weight="balanced")


def probe_cv_by_layer(
    X: np.ndarray,  # [n, L, d] float16/32
    y: np.ndarray,  # [n] {0,1}
    groups: np.ndarray,  # [n] content ids
    folds: int = 5,
    c_reg: float = 1.0,
) -> pd.DataFrame:
    """Grouped-CV logistic probe accuracy/AUC per layer."""
    n, L, _ = X.shape
    folds = min(folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=folds)
    rows = []
    for l in range(L):
        Xl = X[:, l, :].astype(np.float32)
        accs, aucs = [], []
        for tr, te in gkf.split(Xl, y, groups):
            sc = StandardScaler().fit(Xl[tr])
            clf = _logreg(c_reg).fit(sc.transform(Xl[tr]), y[tr])
            z = clf.decision_function(sc.transform(Xl[te]))
            accs.append(float(((z > 0).astype(int) == y[te]).mean()))
            if len(np.unique(y[te])) == 2:
                aucs.append(float(roc_auc_score(y[te], z)))
        rows.append(dict(layer=l, cv_acc=float(np.mean(accs)),
                         cv_auc=float(np.mean(aucs)) if aucs else np.nan))
        log.info(f"probe layer {l:>2}: acc={rows[-1]['cv_acc']:.3f} auc={rows[-1]['cv_auc']:.3f}")
    return pd.DataFrame(rows)


def mass_mean_directions(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions mu_eval - mu_deploy per layer, plus projection std.

    Returns (dirs [L, d] float32 unit-norm, sigma [L] std of train projections).
    alpha in steering is expressed in multiples of sigma.
    """
    Xf = X.astype(np.float32)
    mu1 = Xf[y == 1].mean(axis=0)  # [L, d]
    mu0 = Xf[y == 0].mean(axis=0)
    dirs = mu1 - mu0
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    proj = np.einsum("nld,ld->nl", Xf, dirs)
    sigma = proj.std(axis=0)
    return dirs, sigma


def projection_scores(X: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """[n, L] projections of residuals onto per-layer unit directions."""
    return np.einsum("nld,ld->nl", X.astype(np.float32), dirs)


def mass_mean_auc(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    """Closed-form probe: AUC of the train-fit mass-mean projection on test rows.

    X here is [n, d] (single layer/cell)."""
    d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    nrm = np.linalg.norm(d)
    if nrm < 1e-8 or len(np.unique(yte)) < 2:
        return np.nan
    z = Xte @ (d / nrm)
    return float(roc_auc_score(yte, z))


# --------------------------------------------------------------------------- #
# Generalization structure
# --------------------------------------------------------------------------- #
def family_generalization_matrix(
    X: np.ndarray, y: np.ndarray, families: np.ndarray, layer: int,
    eval_families: list[str], train_rows: np.ndarray, test_rows: np.ndarray,
    c_reg: float = 1.0,
) -> pd.DataFrame:
    """Train on eval-family i (+ all deploy) using TRAIN-split rows; test on
    eval-family j (+ deploy) using TEST-split rows.

    Diagonal ~ within-family fit; off-diagonal ~ transfer. Uniformly high
    off-diagonal AUC = one shared representation across surface forms.
    """
    Xl = X[:, layer, :].astype(np.float32)
    dep = y == 0
    rows = []
    for fi in eval_families:
        tr_mask = train_rows & (dep | (families == fi))
        sc = StandardScaler().fit(Xl[tr_mask])
        clf = _logreg(c_reg).fit(sc.transform(Xl[tr_mask]), y[tr_mask])
        for fj in eval_families:
            te_mask = test_rows & (dep | (families == fj))
            if len(np.unique(y[te_mask])) < 2:
                continue
            z = clf.decision_function(sc.transform(Xl[te_mask]))
            rows.append(dict(train_family=fi, test_family=fj,
                             auc=float(roc_auc_score(y[te_mask], z))))
    return pd.DataFrame(rows)


def leave_family_out_by_layer(
    X: np.ndarray, y: np.ndarray, families: np.ndarray, held_out: str,
    train_rows: np.ndarray, test_rows: np.ndarray, c_reg: float = 1.0,
) -> pd.DataFrame:
    """Probe trained WITHOUT `held_out` eval family (train split), tested ONLY
    on that family (+ deploy) in the test split."""
    n, L, _ = X.shape
    dep = y == 0
    tr_mask = train_rows & (dep | ((y == 1) & (families != held_out)))
    te_mask = test_rows & (dep | ((y == 1) & (families == held_out)))
    rows = []
    for l in range(L):
        Xl = X[:, l, :].astype(np.float32)
        sc = StandardScaler().fit(Xl[tr_mask])
        clf = _logreg(c_reg).fit(sc.transform(Xl[tr_mask]), y[tr_mask])
        z = clf.decision_function(sc.transform(Xl[te_mask]))
        rows.append(dict(layer=l, held_out=held_out,
                         auc=float(roc_auc_score(y[te_mask], z))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Permutation test (mass-mean statistic, grouped split preserved)
# --------------------------------------------------------------------------- #
def permutation_test(
    X_layer: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_perm: int = 200, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    half = len(uniq) // 2
    tr_mask = np.isin(groups, uniq[:half])
    Xf = X_layer.astype(np.float32)
    obs = mass_mean_auc(Xf[tr_mask], y[tr_mask], Xf[~tr_mask], y[~tr_mask])
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null.append(mass_mean_auc(Xf[tr_mask], yp[tr_mask], Xf[~tr_mask], yp[~tr_mask]))
    null = np.array(null, dtype=float)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return dict(observed_auc=obs, p_value=p, null_mean=float(np.nanmean(null)),
                null_q95=float(np.nanquantile(null, 0.95)))


# --------------------------------------------------------------------------- #
# (layer x position) grid
# --------------------------------------------------------------------------- #
def grid_auc(
    grid: np.ndarray,  # [n, L, K, d] float16 (NaN-padded)
    y: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Mass-mean AUC for every (layer, position) cell. Returns [L, K]."""
    n, L, K, d = grid.shape
    out = np.full((L, K), np.nan)
    for k in range(K):
        valid = ~np.isnan(grid[:, 0, k, 0])
        tr = valid & train_mask
        te = valid & ~train_mask
        if tr.sum() < 4 or te.sum() < 4:  # need a few of each split to fit+score
            continue
        for l in range(L):
            out[l, k] = mass_mean_auc(
                grid[tr, l, k, :].astype(np.float32), y[tr],
                grid[te, l, k, :].astype(np.float32), y[te],
            )
    return out
