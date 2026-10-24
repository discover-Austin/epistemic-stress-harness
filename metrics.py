"""
Epistemic Stress Harness — Metric Computation

Pure functions over checkpoints. No interpretation.
Spec: spec.md section 2.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .extract import parse_checkpoints_detailed
from .schema import Checkpoint, CheckpointType, HarnessResult, Metrics

TokenEstimator = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3. Replace via token_estimator=."""
    return int(len(text.split()) * 1.3)


def commitment_latency(checkpoints: List[Checkpoint]) -> float:
    """
    Fraction of checkpoints before first SELECT or CONCLUDE.

    0.0 = immediate commitment, 1.0 = no commitment found, 0.0 if no checkpoints.
    Depends on how many checkpoints there are; compare commitment_position when
    densities differ across variants.
    """
    if not checkpoints:
        return 0.0

    for i, cp in enumerate(checkpoints):
        if cp["type"] in ("SELECT", "CONCLUDE"):
            return i / len(checkpoints)

    return 1.0


def commitment_position(text: str, checkpoints: List[Checkpoint]) -> float:
    """
    Character-offset fraction of the response at which the first SELECT or
    CONCLUDE appears. 1.0 if none. Unlike commitment_latency this does not
    change just because a variant emits fewer or more tags.
    """
    if not text or not checkpoints:
        return 0.0
    for cp in checkpoints:
        if cp["type"] in ("SELECT", "CONCLUDE"):
            start = cp.get("start")
            if start is None:
                return commitment_latency(checkpoints)
            return start / len(text)
    return 1.0


def count_by_type(checkpoints: List[Checkpoint], cp_type: CheckpointType) -> int:
    """Count checkpoints of a specific type."""
    return sum(1 for cp in checkpoints if cp["type"] == cp_type)


def claim_select_ratio(checkpoints: List[Checkpoint]) -> float:
    """
    claim_count / select_count, and 0.0 when there are no SELECTs (spec 2).

    v0.1 returned claim_count in that case, contradicting the spec and
    conflating "no selects" with "many claims".
    """
    selects = count_by_type(checkpoints, "SELECT")
    if selects == 0:
        return 0.0
    return count_by_type(checkpoints, "CLAIM") / selects


def prose_text(text: str, checkpoints: List[Checkpoint]) -> str:
    """The text with checkpoint tags removed."""
    spans = [(cp["start"], cp["end"]) for cp in checkpoints if "start" in cp]
    if not spans:
        return text
    out, last = [], 0
    for s, e in sorted(spans):
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    return "".join(out)


def tokens_per_checkpoint(text: str, checkpoints: List[Checkpoint],
                          token_estimator: Optional[TokenEstimator] = None) -> float:
    """Average tokens per checkpoint (justification density)."""
    if not checkpoints:
        return 0.0
    est = token_estimator or estimate_tokens
    return est(text) / len(checkpoints)


def compute_metrics(
    text: str,
    checkpoints: List[Checkpoint],
    *,
    malformed_count: int = 0,
    truncated: bool = False,
    reported_output_tokens: Optional[int] = None,
    token_estimator: Optional[TokenEstimator] = None,
) -> Metrics:
    """
    Compute all core metrics from text and its checkpoints.

    If the API reported the true output token count, pass it as
    reported_output_tokens; it replaces the estimate everywhere.
    """
    est = token_estimator or estimate_tokens
    if reported_output_tokens is not None:
        total_tokens, source = int(reported_output_tokens), "reported"
    else:
        total_tokens, source = est(text), "estimate"

    n = len(checkpoints)
    return Metrics(
        commitment_latency=commitment_latency(checkpoints),
        assume_count=count_by_type(checkpoints, "ASSUME"),
        claim_count=count_by_type(checkpoints, "CLAIM"),
        branch_count=count_by_type(checkpoints, "BRANCH"),
        select_count=count_by_type(checkpoints, "SELECT"),
        conclude_count=count_by_type(checkpoints, "CONCLUDE"),
        total_checkpoints=n,
        tokens_per_checkpoint=(total_tokens / n) if n else 0.0,
        claim_select_ratio=claim_select_ratio(checkpoints),
        total_tokens=total_tokens,
        commitment_position=commitment_position(text, checkpoints),
        prose_tokens=est(prose_text(text, checkpoints)),
        truncated=truncated,
        malformed_count=malformed_count,
        tokens_source=source,
    )


def extract_metrics(
    text: str,
    variant: str = "baseline",
    *,
    strict: bool = True,
    truncated: bool = False,
    reported_output_tokens: Optional[int] = None,
    token_estimator: Optional[TokenEstimator] = None,
) -> HarnessResult:
    """
    Main entry point: parse checkpoints and compute all metrics.

    Args:
        text: Response with checkpoint annotations.
        variant: Name of the perturbation variant (a label only).
        strict: Require uppercase tags, as the spec does.
        truncated: True if generation stopped at the token limit.
        reported_output_tokens: True output token count from the API, if known.
        token_estimator: Replacement for the words*1.3 estimate.
    """
    parsed = parse_checkpoints_detailed(text, strict=strict)
    metrics = compute_metrics(
        text,
        parsed.checkpoints,
        malformed_count=parsed.malformed_count,
        truncated=truncated,
        reported_output_tokens=reported_output_tokens,
        token_estimator=token_estimator,
    )
    return HarnessResult(
        variant=variant,
        raw_text=text,
        checkpoints=parsed.checkpoints,
        metrics=metrics,
    )


def explain_metrics(metrics: Metrics) -> Dict[str, str]:
    """Human-readable explanations for metric values."""
    est_note = (
        "the API-reported output token count"
        if metrics.tokens_source == "reported"
        else "a rough estimate (words x 1.3)"
    )
    return {
        "commitment_latency": (
            f"Commitment latency is {metrics.commitment_latency:.2f}, the fraction of checkpoints "
            "before the first SELECT or CONCLUDE. Lower values mean earlier commitment."
        ),
        "commitment_position": (
            f"Commitment position is {metrics.commitment_position:.2f}, the fraction of the response "
            "text before the first SELECT or CONCLUDE."
        ),
        "assume_count": f"ASSUME count is {metrics.assume_count}, the explicit starting assumptions.",
        "claim_count": f"CLAIM count is {metrics.claim_count}, the intermediate assertions.",
        "branch_count": (
            f"Branch count is {metrics.branch_count}, the BRANCH tags recording alternatives. "
            "A count of tags, not proof that alternatives were weighed; see fidelity metrics."
        ),
        "select_count": f"SELECT count is {metrics.select_count}, the explicit choice points.",
        "conclude_count": f"CONCLUDE count is {metrics.conclude_count}, the final conclusions.",
        "total_checkpoints": f"Total checkpoints is {metrics.total_checkpoints}.",
        "tokens_per_checkpoint": (
            f"Tokens per checkpoint is {metrics.tokens_per_checkpoint:.1f}, based on {est_note}. "
            "Higher means more text per checkpoint."
        ),
        "claim_select_ratio": (
            f"CLAIM/SELECT ratio is {metrics.claim_select_ratio:.2f} (0 when there are no SELECTs)."
        ),
        "total_tokens": f"Total tokens is {metrics.total_tokens}, from {est_note}.",
        "truncated": (
            "The response was cut off at the token limit; late checkpoints (often CONCLUDE) may be missing."
            if metrics.truncated else "The response was not truncated."
        ),
        "malformed_count": (
            f"{metrics.malformed_count} tag(s) were unterminated or interrupted by another tag and were dropped."
        ),
    }
