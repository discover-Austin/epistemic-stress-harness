"""Statistics helpers: correctness, reproducibility, numpy/pure agreement."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import stats


def _gauss(n, mu, seed):
    r = random.Random(seed)
    return [r.gauss(mu, 1) for _ in range(n)]


def test_mean_sd_quantile():
    assert stats.mean([1, 2, 3]) == 2
    assert abs(stats.sd([1, 2, 3]) - 1.0) < 1e-12
    assert stats.sd([5]) == 0.0
    assert stats.quantile([1, 2, 3, 4], 0.5) == 2.5


def test_permutation_test_is_calibrated_under_the_null():
    # Rejection rate at alpha=0.05 over many null datasets should be near 5%.
    rejects = 0
    trials = 300
    for t in range(trials):
        a, b = _gauss(25, 0, 1000 + t), _gauss(25, 0, 5000 + t)
        rejects += stats.permutation_test(a, b, n_perm=300, seed=t) <= 0.05
    assert 0.02 <= rejects / trials <= 0.09, rejects / trials


def test_permutation_test_has_power_against_a_one_sd_shift():
    hits = 0
    trials = 100
    for t in range(trials):
        a, b = _gauss(25, 0, 1000 + t), _gauss(25, 1, 5000 + t)
        hits += stats.permutation_test(a, b, n_perm=300, seed=t) <= 0.05
    assert hits / trials >= 0.9, hits / trials


def test_permutation_is_seed_reproducible():
    a, b = _gauss(30, 0, 1), _gauss(30, 0.3, 2)
    assert stats.permutation_test(a, b, seed=7) == stats.permutation_test(a, b, seed=7)


def test_numpy_and_pure_paths_agree():
    a, b = _gauss(30, 0, 1), _gauss(30, 0.6, 2)
    p1 = stats.permutation_test(a, b, n_perm=6000, seed=1, use_numpy=True)
    p2 = stats.permutation_test(a, b, n_perm=6000, seed=1, use_numpy=False)
    assert abs(p1 - p2) < 0.02
    lo1, hi1 = stats.bootstrap_diff_ci(a, b, seed=1, use_numpy=True)
    lo2, hi2 = stats.bootstrap_diff_ci(a, b, seed=1, use_numpy=False)
    assert abs(lo1 - lo2) < 0.15 and abs(hi1 - hi2) < 0.15


def test_permutation_empty_is_none():
    assert stats.permutation_test([], [1.0]) is None


def test_bootstrap_ci_contains_mean_and_degenerate_cases():
    xs = _gauss(60, 2.0, 3)
    lo, hi = stats.bootstrap_ci(xs, seed=1)
    assert lo < stats.mean(xs) < hi
    assert stats.bootstrap_ci([4.0]) == (4.0, 4.0)


def test_bootstrap_ci_coverage_is_roughly_nominal():
    hits = 0
    trials = 200
    for t in range(trials):
        xs = _gauss(30, 0, 100 + t)
        lo, hi = stats.bootstrap_ci(xs, n_boot=400, seed=t)
        hits += lo <= 0 <= hi
    assert 0.88 <= hits / trials <= 0.99


def test_cohens_d():
    assert abs(stats.cohens_d([1, 2, 3, 4], [3, 4, 5, 6]) + 1.5491933384829668) < 1e-9
    assert stats.cohens_d([1, 1], [1, 1]) is None
    assert stats.cohens_d([1], [2, 3]) is None


def test_holm():
    adj = stats.holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5, "d": None})
    assert abs(adj["a"] - 0.03) < 1e-12
    assert abs(adj["b"] - 0.04) < 1e-12
    assert adj["c"] == 0.5 and adj["d"] is None
    # monotone: adjusted p never decreases down the sorted list
    adj = stats.holm_adjust({"a": 0.001, "b": 0.04, "c": 0.04})
    assert adj["a"] <= adj["b"] <= adj["c"] + 1e-12


def test_signflip_exact():
    assert stats.signflip_test([]) is None
    assert stats.signflip_test([0.0, 0.0]) == 1.0
    assert stats.signflip_test([1.0] * 8) == 2 / 256     # only all+ and all- are as extreme
    p = stats.signflip_test([1, -1, 1, -1, 1, -1])
    assert p == 1.0
