"""
Epistemic Stress Harness — Statistics

Small, seeded, dependency-light inference helpers. numpy is used when it is
installed (much faster); a pure-Python fallback gives the same answers up to
Monte Carlo error. Every function takes a seed so analyses are reproducible.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

try:  # optional accelerator
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: Sequence[float]) -> float:
    """Sample standard deviation (n - 1). 0.0 for fewer than 2 values."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, q in [0, 1]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """
    Standardized mean difference (a - b) with pooled SD.
    None when it is undefined: <2 values in a group, or zero pooled variance
    (including when the means also match).
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = math.sqrt(((na - 1) * sd(a) ** 2 + (nb - 1) * sd(b) ** 2) / (na + nb - 2))
    if pooled == 0:
        return None
    return (mean(a) - mean(b)) / pooled


def bootstrap_ci(
    xs: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05,
    seed: int = 0, use_numpy: Optional[bool] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (float(xs[0]), float(xs[0]))
    fast = (_np is not None) if use_numpy is None else (use_numpy and _np is not None)
    if fast:
        rng = _np.random.default_rng(seed)
        arr = _np.asarray(xs, dtype=float)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        means = arr[idx].mean(axis=1)
        return (float(_np.quantile(means, alpha / 2)), float(_np.quantile(means, 1 - alpha / 2)))
    rng = random.Random(seed)
    n = len(xs)
    means = [mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    return (quantile(means, alpha / 2), quantile(means, 1 - alpha / 2))


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 0, use_numpy: Optional[bool] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), resampling each group."""
    if not a or not b:
        return (float("nan"), float("nan"))
    fast = (_np is not None) if use_numpy is None else (use_numpy and _np is not None)
    if fast:
        rng = _np.random.default_rng(seed)
        aa, bb = _np.asarray(a, float), _np.asarray(b, float)
        ma = aa[rng.integers(0, len(aa), size=(n_boot, len(aa)))].mean(axis=1)
        mb = bb[rng.integers(0, len(bb), size=(n_boot, len(bb)))].mean(axis=1)
        diff = ma - mb
        return (float(_np.quantile(diff, alpha / 2)), float(_np.quantile(diff, 1 - alpha / 2)))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ma = mean([a[rng.randrange(len(a))] for _ in range(len(a))])
        mb = mean([b[rng.randrange(len(b))] for _ in range(len(b))])
        diffs.append(ma - mb)
    return (quantile(diffs, alpha / 2), quantile(diffs, 1 - alpha / 2))


def permutation_test(
    a: Sequence[float], b: Sequence[float], *, n_perm: int = 5000,
    seed: int = 0, use_numpy: Optional[bool] = None,
) -> Optional[float]:
    """
    Two-sided permutation test on the difference of means.
    Returns None if either group is empty. The +1 correction keeps p > 0.
    """
    if not a or not b:
        return None
    observed = abs(mean(a) - mean(b))
    na, nb = len(a), len(b)
    fast = (_np is not None) if use_numpy is None else (use_numpy and _np is not None)
    if fast:
        rng = _np.random.default_rng(seed)
        pooled = _np.asarray(list(a) + list(b), dtype=float)
        total = pooled.sum()
        idx = rng.random((n_perm, na + nb)).argsort(axis=1)[:, :na]
        sa = pooled[idx].sum(axis=1)
        diff = _np.abs(sa / na - (total - sa) / nb)
        extreme = int((diff >= observed - 1e-12).sum())
    else:
        rng = random.Random(seed)
        pooled = list(a) + list(b)
        extreme = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            if abs(mean(pooled[:na]) - mean(pooled[na:])) >= observed - 1e-12:
                extreme += 1
    return (extreme + 1) / (n_perm + 1)


def holm_adjust(pvalues: Dict[object, Optional[float]]) -> Dict[object, Optional[float]]:
    """Holm-Bonferroni adjusted p-values. None values are passed through."""
    items = [(k, p) for k, p in pvalues.items() if p is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[object, Optional[float]] = {k: None for k in pvalues}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[k] = running
    return adjusted


def signflip_test(
    diffs: Sequence[float], *, n_perm: int = 20000, seed: int = 0,
) -> Optional[float]:
    """
    Paired two-sided sign-flip test on the mean of per-unit differences.
    Exact enumeration for up to 14 units, Monte Carlo above that.
    Returns None for an empty input.
    """
    n = len(diffs)
    if n == 0:
        return None
    observed = abs(sum(diffs)) / n
    if observed == 0:
        return 1.0
    if n <= 14:
        extreme = 0
        total = 1 << n
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                s += d if (mask >> i) & 1 else -d
            if abs(s) / n >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) / n >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)
