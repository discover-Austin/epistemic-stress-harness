"""
Epistemic Stress Harness — Analysis

Turns repeated samples into evidence. The v0.1 harness compared one response
per variant; sampling noise alone moves every structural metric, so a single
pair of runs cannot support any claim. This module:

  * aggregates repeated samples per condition (mean, SD, bootstrap CI);
  * measures the noise floor from the reference condition itself;
  * tests each condition against the reference with a permutation test on
    per-task paired deltas, Holm-corrected across the whole family of tests;
  * labels every metric 'instructed' / 'adjacent' / 'uninstructed' for the
    condition, because a change in a metric the prompt directly asks for is
    compliance, not deformation;
  * compares checkpoint topology against the reference-vs-reference baseline.

Spec: spec.md section 7.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import stats
from .outcomes import assumption_coverage
from .schema import SampleRecord, SPEC_VERSION
from .suite import Condition
from .topology import mean_pairwise_topology

# metric name -> extractor over a SampleRecord (None = undefined for that sample)
def _metric(attr: str) -> Callable[[SampleRecord], Optional[float]]:
    return lambda s: float(getattr(s.result.metrics, attr))

def _fid(attr: str) -> Callable[[SampleRecord], Optional[float]]:
    def f(s: SampleRecord) -> Optional[float]:
        v = getattr(s.fidelity, attr, None) if s.fidelity else None
        return None if v is None else float(v)
    return f

def _out(attr: str) -> Callable[[SampleRecord], Optional[float]]:
    def f(s: SampleRecord) -> Optional[float]:
        v = getattr(s.outcome, attr, None) if s.outcome else None
        return None if v is None else float(v)
    return f

EXTRACTORS: Dict[str, Callable[[SampleRecord], Optional[float]]] = {
    # structure (model-annotated; see fidelity metrics before trusting these)
    "commitment_latency": _metric("commitment_latency"),
    "commitment_position": _metric("commitment_position"),
    "branch_count": _metric("branch_count"),
    "assume_count": _metric("assume_count"),
    "claim_count": _metric("claim_count"),
    "select_count": _metric("select_count"),
    "total_checkpoints": _metric("total_checkpoints"),
    "total_tokens": _metric("total_tokens"),
    "tokens_per_checkpoint": _metric("tokens_per_checkpoint"),
    "claim_select_ratio": _metric("claim_select_ratio"),
    "truncated": _metric("truncated"),
    # fidelity (tag-text consistency)
    "fidelity_score": _fid("fidelity_score"),
    "branch_well_formed": _fid("branch_well_formed"),
    "select_justified": _fid("select_justified"),
    "select_resolves_branch": _fid("select_resolves_branch"),
    # objective outcomes (not mentioned by any perturbation prompt)
    "answered": _out("answered"),
    "answer_correct": _out("answer_correct"),
    "caveat_coverage": _out("caveat_coverage"),
    "certainty_density": _out("certainty_density"),
    "hedge_density": _out("hedge_density"),
    "overclaim_count": _out("overclaim_count"),
    "confident_wrong": _out("confident_wrong"),
}

# derived across samples (need the reference): handled in _derived()
DERIVED_METRICS = ["assumption_coverage"]

DEFAULT_METRICS = list(EXTRACTORS) + DERIVED_METRICS


def valid_samples(samples: Iterable[SampleRecord]) -> List[SampleRecord]:
    return [s for s in samples if s.error is None and s.result is not None]


def _judge_metrics(samples: Sequence[SampleRecord]) -> List[str]:
    keys = set()
    for s in samples:
        for k, v in (s.judge or {}).items():
            if isinstance(v, (int, float, bool)):
                keys.add(f"judge_{k}")
    return sorted(keys)


def _derived(samples: Sequence[SampleRecord], ref: str) -> Dict[tuple, Dict[str, Optional[float]]]:
    """Per-sample metrics that depend on the reference samples of the same task."""
    by_task_ref: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in samples:
        if s.variant == ref:
            by_task_ref[s.task_id].append(s)

    out: Dict[tuple, Dict[str, Optional[float]]] = {}
    for s in samples:
        refs = [r for r in by_task_ref.get(s.task_id, []) if r.key != s.key]
        vals = [assumption_coverage(r.result.checkpoints, s.result.checkpoints) for r in refs]
        vals = [v for v in vals if v is not None]
        out[s.key] = {"assumption_coverage": (sum(vals) / len(vals)) if vals else None}
    return out


def _value(s: SampleRecord, metric: str, derived: Dict[tuple, Dict[str, Optional[float]]]) -> Optional[float]:
    if metric in DERIVED_METRICS:
        return derived.get(s.key, {}).get(metric)
    if metric.startswith("judge_"):
        v = (s.judge or {}).get(metric[len("judge_"):])
        return None if v is None else float(v)
    return EXTRACTORS[metric](s)


def _by_task_variant(samples, metric, derived) -> Dict[str, Dict[str, List[float]]]:
    d: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        v = _value(s, metric, derived)
        if v is not None:
            d[s.task_id][s.variant].append(v)
    return d


def describe(samples: Sequence[SampleRecord], metrics: Sequence[str], *,
             derived, n_boot: int, seed: int) -> Dict[str, Dict[str, Any]]:
    """Per-condition, per-metric mean / SD / bootstrap CI over all samples."""
    out: Dict[str, Dict[str, Any]] = {}
    variants = sorted({s.variant for s in samples})
    for v in variants:
        out[v] = {}
        for m in metrics:
            vals = [x for x in (_value(s, m, derived) for s in samples if s.variant == v) if x is not None]
            if not vals:
                continue
            lo, hi = stats.bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            out[v][m] = {"n": len(vals), "mean": stats.mean(vals), "sd": stats.sd(vals),
                         "ci_lo": lo, "ci_hi": hi}
    return out


def compare_to_reference(
    samples: Sequence[SampleRecord], ref: str, target: str, metric: str, *,
    derived, n_perm: int, n_boot: int, seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Per-task paired comparison of `target` against `ref` on one metric.

    Each target sample is expressed as a delta from its task's reference mean.
    Each reference sample is expressed as a delta from the mean of the *other*
    reference samples of its task (leave-one-out), which gives the delta
    distribution produced by sampling noise alone. The permutation test asks
    whether target deltas differ from those noise-only deltas.
    Tasks with fewer than 2 reference samples are skipped (no noise estimate).
    """
    by = _by_task_variant(samples, metric, derived)
    ref_deltas: List[float] = []
    tgt_deltas: List[float] = []
    tasks_used = 0
    for task, d in by.items():
        r, t = d.get(ref, []), d.get(target, [])
        if len(r) < 2 or not t:
            continue
        tasks_used += 1
        total = sum(r)
        for x in r:
            ref_deltas.append(x - (total - x) / (len(r) - 1))
        rm = total / len(r)
        tgt_deltas.extend(x - rm for x in t)

    if not ref_deltas or not tgt_deltas:
        return None

    mean_delta = stats.mean(tgt_deltas)  # ref deltas average exactly 0 by construction
    lo, hi = stats.bootstrap_diff_ci(tgt_deltas, ref_deltas, n_boot=n_boot, seed=seed)
    noise_sd = stats.sd(ref_deltas)
    return {
        "metric": metric,
        "target": target,
        "reference": ref,
        "tasks": tasks_used,
        "n_target": len(tgt_deltas),
        "n_reference": len(ref_deltas),
        "delta": mean_delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "cohens_d": stats.cohens_d(tgt_deltas, ref_deltas),
        "noise_sd": noise_sd,
        "delta_in_noise_sd": (mean_delta / noise_sd) if noise_sd > 0 else None,
        "p": stats.permutation_test(tgt_deltas, ref_deltas, n_perm=n_perm, seed=seed),
    }


def topology_vs_reference(
    samples: Sequence[SampleRecord], ref: str, targets: Sequence[str], *, alpha: float, seed: int,
) -> List[Dict[str, Any]]:
    """
    Checkpoint-topology similarity of each target to the reference, against the
    reference-vs-reference noise floor. Computed per task, then compared across
    tasks with a paired sign-flip test.
    """
    by_task: Dict[str, Dict[str, List[list]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_task[s.task_id][s.variant].append(s.result.checkpoints)

    rows = []
    pvals: Dict[object, Optional[float]] = {}
    for tgt in targets:
        per_task = {"sequence_similarity": [], "node_overlap": [], "depth_ratio": []}
        floor = {"sequence_similarity": [], "node_overlap": [], "depth_ratio": []}
        for task, d in by_task.items():
            r, t = d.get(ref, []), d.get(tgt, [])
            noise = mean_pairwise_topology(r, r, same_group=True)
            cross = mean_pairwise_topology(t, r)
            if noise is None or cross is None:
                continue
            for k in per_task:
                per_task[k].append(getattr(cross, k))
                floor[k].append(getattr(noise, k))
        if not per_task["sequence_similarity"]:
            continue
        for k in per_task:
            diffs = [a - b for a, b in zip(per_task[k], floor[k])]
            lo, hi = stats.bootstrap_ci(diffs, seed=seed)
            p = stats.signflip_test(diffs, seed=seed)
            pvals[(tgt, k)] = p
            rows.append({
                "target": tgt, "reference": ref, "metric": k, "tasks": len(diffs),
                "target_vs_ref": stats.mean(per_task[k]),
                "ref_vs_ref_noise_floor": stats.mean(floor[k]),
                "delta": stats.mean(diffs), "ci_lo": lo, "ci_hi": hi, "p": p,
            })
    adj = stats.holm_adjust(pvals)
    for r in rows:
        r["p_adj"] = adj[(r["target"], r["metric"])]
        r["significant"] = r["p_adj"] is not None and r["p_adj"] < alpha
    return rows


def analyze(
    samples: Sequence[SampleRecord],
    conditions: Dict[str, Condition],
    *,
    reference: str = "baseline",
    contrasts: Optional[Sequence[Tuple[str, str]]] = None,
    metrics: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    n_perm: int = 20000,
    n_boot: int = 2000,
    seed: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full analysis. `contrasts` are extra (reference, target) pairs beyond
    reference-vs-each-other-condition, e.g. ("stakes_neutral", "stakes_retirement").
    """
    all_samples = list(samples)
    good = valid_samples(all_samples)
    warnings: List[str] = []
    if len(good) < len(all_samples):
        warnings.append(f"{len(all_samples) - len(good)} sample(s) failed and were excluded.")

    variants = sorted({s.variant for s in good})
    if reference not in variants:
        raise ValueError(f"reference condition {reference!r} has no valid samples")

    metrics = list(metrics) if metrics else DEFAULT_METRICS + _judge_metrics(good)
    derived = _derived(good, reference)

    # --- power / design warnings ---
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for s in good:
        counts[(s.task_id, s.variant)] += 1
    ref_counts = [c for (t, v), c in counts.items() if v == reference]
    tasks = sorted({s.task_id for s in good})
    if ref_counts and min(ref_counts) < 5:
        warnings.append(
            f"Reference has as few as {min(ref_counts)} sample(s) per task; the noise floor is "
            "poorly estimated. Use >= 10 samples per task and condition for firm conclusions."
        )
    if len(tasks) < 5:
        warnings.append(f"Only {len(tasks)} task(s); effects may not generalize across tasks.")
    for v in variants:
        sv = [s for s in good if s.variant == v]
        rate = sum(1 for s in sv if s.result.metrics.truncated) / len(sv)
        if rate >= 0.2:
            warnings.append(f"{v}: {rate:.0%} of responses were truncated at the token limit.")
        unanswered = sum(1 for s in sv if s.outcome and not s.outcome.answered) / len(sv)
        if unanswered >= 0.2:
            warnings.append(f"{v}: {unanswered:.0%} of responses had no CONCLUDE.")

    # --- comparisons ---
    pairs: List[Tuple[str, str]] = [(reference, v) for v in variants if v != reference]
    for c in (contrasts or []):
        if tuple(c) not in pairs and c[0] in variants and c[1] in variants:
            pairs.append(tuple(c))  # type: ignore[arg-type]

    rows: List[Dict[str, Any]] = []
    for ref, tgt in pairs:
        cond = conditions.get(tgt)
        for m in metrics:
            row = compare_to_reference(good, ref, tgt, m, derived=derived,
                                       n_perm=n_perm, n_boot=n_boot, seed=seed)
            if row is None:
                continue
            row["label"] = cond.label_for(m) if cond else "uninstructed"
            rows.append(row)

    adj = stats.holm_adjust({i: r["p"] for i, r in enumerate(rows)})
    for i, r in enumerate(rows):
        r["p_adj"] = adj[i]
        r["significant"] = r["p_adj"] is not None and r["p_adj"] < alpha

    topo = topology_vs_reference(good, reference, [v for v in variants if v != reference],
                                 alpha=alpha, seed=seed)

    floor_p = 1.0 / (n_perm + 1)
    saturated = sum(1 for r in rows if r["p"] is not None and r["p"] <= floor_p + 1e-12)
    if rows and saturated and floor_p * len(rows) >= alpha:
        warnings.append(
            f"{saturated} test(s) sit at the permutation resolution floor (p={floor_p:.5f}); with "
            f"{len(rows)} tests, Holm cannot call them significant. Raise --n-perm."
        )

    return {
        "spec_version": SPEC_VERSION,
        "meta": {**(meta or {}), "reference": reference, "alpha": alpha, "seed": seed,
                 "n_perm": n_perm, "n_boot": n_boot, "n_samples": len(good),
                 "tasks": tasks, "conditions": variants,
                 "correction": "Holm across all tests in this analysis"},
        "warnings": warnings,
        "conditions": describe(good, metrics, derived=derived, n_boot=n_boot, seed=seed),
        "comparisons": rows,
        "topology": topo,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_HEADLINE = ["branch_count", "commitment_position", "total_tokens", "fidelity_score",
             "answer_correct", "caveat_coverage", "certainty_density", "overclaim_count"]


def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


def format_report(a: Dict[str, Any], *, show_all: bool = False) -> str:
    """Plain-text report. Significant rows are marked *; labels matter."""
    L: List[str] = []
    meta = a["meta"]
    L.append(f"EPISTEMIC STRESS HARNESS — ANALYSIS (spec v{a['spec_version']})")
    L.append("=" * 78)
    L.append(f"reference: {meta['reference']}   samples: {meta['n_samples']}   "
             f"tasks: {len(meta['tasks'])}   alpha: {meta['alpha']}   seed: {meta['seed']}")
    L.append(f"multiple-comparison correction: {meta['correction']}")
    for w in a["warnings"]:
        L.append(f"WARNING: {w}")

    L.append("")
    L.append("CONDITION MEANS (headline metrics)")
    names = list(a["conditions"])
    L.append(f"{'metric':<22}" + "".join(f"{n[:14]:>15}" for n in names))
    for m in _HEADLINE:
        if not any(m in a["conditions"][n] for n in names):
            continue
        L.append(f"{m:<22}" + "".join(
            f"{_fmt(a['conditions'][n].get(m, {}).get('mean')):>15}" for n in names))

    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in a["comparisons"]:
        by_pair[(r["reference"], r["target"])].append(r)

    for (ref, tgt), rows in by_pair.items():
        L.append("")
        L.append(f"{tgt}  vs  {ref}")
        L.append("-" * 78)
        sig = [r for r in rows if r["significant"]]
        for lab in ("instructed", "adjacent", "uninstructed"):
            hit = [r["metric"] for r in sig if r["label"] == lab]
            L.append(f"  significant on {lab:<12} metrics: {', '.join(hit) if hit else 'none'}")
        L.append(f"  {'metric':<24}{'label':<13}{'delta':>9}{'95% CI':>20}{'d':>7}{'p_adj':>8}")
        shown = rows if show_all else [r for r in rows if r["significant"] or r["metric"] in _HEADLINE]
        for r in sorted(shown, key=lambda r: (r["label"], r["metric"])):
            ci = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
            L.append(f"  {r['metric']:<24}{r['label']:<13}{_fmt(r['delta']):>9}{ci:>20}"
                     f"{_fmt(r['cohens_d']):>7}{_fmt(r['p_adj'], '.3f'):>8}{' *' if r['significant'] else ''}")

    if a["topology"]:
        L.append("")
        L.append("TOPOLOGY vs reference, relative to the reference-vs-reference noise floor")
        L.append("-" * 78)
        L.append(f"  {'condition':<22}{'metric':<20}{'vs ref':>8}{'noise floor':>13}{'delta':>8}{'p_adj':>8}")
        for r in a["topology"]:
            if r["metric"] == "depth_ratio":
                continue
            L.append(f"  {r['target']:<22}{r['metric']:<20}{_fmt(r['target_vs_ref']):>8}"
                     f"{_fmt(r['ref_vs_ref_noise_floor']):>13}{_fmt(r['delta']):>8}"
                     f"{_fmt(r['p_adj'], '.3f'):>8}{' *' if r['significant'] else ''}")

    L.append("")
    L.append("Reading this: a change on an 'instructed' metric is the model doing what the prompt")
    L.append("asked. Evidence of deformation is a significant change on 'uninstructed' metrics,")
    L.append("especially objective outcomes (answer_correct, caveat_coverage), and only where the")
    L.append("matched control condition (e.g. stakes_neutral) shows no such change.")
    return "\n".join(L)
