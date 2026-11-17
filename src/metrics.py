"""Accuracy, stability and mitigation functions. All inputs are numpy arrays.

Conventions
  probs : (K, N, 3) label probabilities for K prompts x N abstracts, label order yes,no,maybe
  preds : (K, N) integer predictions in {0,1,2}
  y     : (N,) gold labels in {0,1,2}
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score

LABELS = ["yes", "no", "maybe"]
L2I = {l: i for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------- accuracy
def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


# ---------------------------------------------------------------- stability
def pairwise_disagreement(preds: np.ndarray) -> np.ndarray:
    """Errica et al. (2025) sensitivity per instance: fraction of prompt pairs whose
    predictions differ. preds (K, N) -> (N,). Requires K >= 2."""
    K = preds.shape[0]
    assert K >= 2
    pairs = list(combinations(range(K), 2))
    return np.mean([preds[i] != preds[j] for i, j in pairs], axis=0)


def flip_rate(preds: np.ndarray) -> float:
    """Fraction of instances whose conclusion is not identical under every prompt."""
    return float((preds != preds[0]).any(axis=0).mean())


def fleiss_kappa(preds: np.ndarray, n_cat: int = 3) -> float:
    """Fleiss' kappa treating the K prompts as raters of N items."""
    K, N = preds.shape
    counts = np.stack([(preds == c).sum(0) for c in range(n_cat)], axis=1)  # (N, C)
    p_j = counts.sum(0) / (N * K)
    P_i = ((counts ** 2).sum(1) - K) / (K * (K - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    if np.isclose(P_e, 1.0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def errica_consistency(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Errica et al. (2025) consistency per class: mean over prompts of the fraction of
    class-c instances predicted as c (i.e., per-class recall averaged over prompts)."""
    return {LABELS[c]: float((preds[:, y == c] == c).mean()) for c in range(3) if (y == c).any()}


# ---------------------------------------------------------------- mitigations
def contextual_calibration(probs: np.ndarray, p_cf: np.ndarray) -> np.ndarray:
    """Zhao et al. (2021): W = diag(p_cf)^-1, b = 0; q = normalise(W p).
    probs (..., 3); p_cf (3,) mean content-free label distribution for that prompt."""
    q = probs / np.clip(p_cf, 1e-12, None)
    return q / q.sum(-1, keepdims=True)


def ensemble(probs: np.ndarray) -> np.ndarray:
    """Average label distributions over prompts: (K, N, 3) -> (N, 3)."""
    return probs.mean(0)


# ---------------------------------------------------------------- variance sources
def sample_predictions(probs: np.ndarray, temperature: float, n_samples: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Draw labels from the restricted answer distribution at a temperature.
    probs (N, 3) -> (n_samples, N)."""
    logp = np.log(np.clip(probs, 1e-12, None)) / temperature
    p = np.exp(logp - logp.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    cdf = p.cumsum(-1)
    u = rng.random((n_samples, probs.shape[0], 1))
    return np.minimum((u > cdf[None]).sum(-1), 2)


def two_analyst_agreement(probs: np.ndarray, k: int, n_draws: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Two analysts independently pick k prompts (without replacement, disjoint when
    possible) from the pool and report the ensemble conclusion. Returns per-draw
    agreement rates over abstracts, shape (n_draws,)."""
    K = probs.shape[0]
    out = np.empty(n_draws)
    for d in range(n_draws):
        if 2 * k <= K:
            perm = rng.permutation(K)
            a, b = perm[:k], perm[k:2 * k]
        else:
            a, b = rng.choice(K, k, replace=False), rng.choice(K, k, replace=False)
        pa, pb = probs[a].mean(0).argmax(-1), probs[b].mean(0).argmax(-1)
        out[d] = (pa == pb).mean()
    return out


# ---------------------------------------------------------------- uncertainty
def bootstrap_ci(fn, *arrays, n: int = 2000, seed: int = 2026, axis_n: int = -1,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over abstracts (the last axis of every array)."""
    rng = np.random.default_rng(seed)
    N = arrays[0].shape[axis_n]
    point = fn(*arrays)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        stats[b] = fn(*[np.take(a, idx, axis=axis_n) for a in arrays])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)
