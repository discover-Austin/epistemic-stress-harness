"""
Epistemic Stress Harness — Metric Computation

Pure functions over checkpoints. No interpretation.
Spec: spec.md section 2.
"""

from typing import List, Dict

from .schema import Checkpoint, CheckpointType, Metrics, HarnessResult
from .extract import parse_checkpoints


def commitment_latency(checkpoints: List[Checkpoint]) -> float:
    """
    Fraction of checkpoints before first SELECT or CONCLUDE.

    0.0 = immediate commitment, 1.0 = no commitment found.
    """
    if not checkpoints:
        return 0.0

    for i, cp in enumerate(checkpoints):
        if cp["type"] in ("SELECT", "CONCLUDE"):
            return i / len(checkpoints)

    return 1.0


def count_by_type(checkpoints: List[Checkpoint], cp_type: CheckpointType) -> int:
    """Count checkpoints of a specific type."""
    return sum(1 for cp in checkpoints if cp["type"] == cp_type)


def claim_select_ratio(checkpoints: List[Checkpoint]) -> float:
    """
    Ratio of CLAIM to SELECT nodes.

    Higher = more intermediate reasoning per decision.
    """
    claims = count_by_type(checkpoints, "CLAIM")
    selects = count_by_type(checkpoints, "SELECT")

    if selects == 0:
        return float(claims) if claims > 0 else 0.0

    return claims / selects


def tokens_per_checkpoint(text: str, checkpoints: List[Checkpoint]) -> float:
    """
    Average tokens between checkpoints.

    Token estimate: word_count * 1.3
    """
    if not checkpoints:
        return 0.0

    words = len(text.split())
    tokens = int(words * 1.3)

    return tokens / len(checkpoints)


def compute_metrics(text: str, checkpoints: List[Checkpoint]) -> Metrics:
    """Compute all core metrics from text and its checkpoints."""
    words = len(text.split())
    total_tokens = int(words * 1.3)

    return Metrics(
        commitment_latency=commitment_latency(checkpoints),
        assume_count=count_by_type(checkpoints, "ASSUME"),
        claim_count=count_by_type(checkpoints, "CLAIM"),
        branch_count=count_by_type(checkpoints, "BRANCH"),
        select_count=count_by_type(checkpoints, "SELECT"),
        conclude_count=count_by_type(checkpoints, "CONCLUDE"),
        total_checkpoints=len(checkpoints),
        tokens_per_checkpoint=tokens_per_checkpoint(text, checkpoints),
        claim_select_ratio=claim_select_ratio(checkpoints),
        total_tokens=total_tokens,
    )


def extract_metrics(text: str, variant: str = "baseline") -> HarnessResult:
    """
    Main entry point: parse checkpoints and compute all metrics.

    Args:
        text: Response with checkpoint annotations.
        variant: Name of perturbation variant.

    Returns:
        Complete harness result.
    """
    checkpoints = parse_checkpoints(text)
    metrics = compute_metrics(text, checkpoints)
    return HarnessResult(
        variant=variant,
        raw_text=text,
        checkpoints=checkpoints,
        metrics=metrics,
    )


def explain_metrics(metrics: Metrics) -> Dict[str, str]:
    """Human-readable explanations for metric values."""
    return {
        "commitment_latency": (
            f"Commitment latency is {metrics.commitment_latency:.2f}, the fraction of checkpoints "
            "before the first SELECT or CONCLUDE. Lower values indicate earlier commitment."
        ),
        "assume_count": (
            f"ASSUME count is {metrics.assume_count}, representing explicit starting assumptions."
        ),
        "claim_count": (
            f"CLAIM count is {metrics.claim_count}, capturing intermediate assertions."
        ),
        "branch_count": (
            f"Branch count is {metrics.branch_count}, the number of BRANCH checkpoints that record alternatives."
        ),
        "select_count": (
            f"SELECT count is {metrics.select_count}, the number of explicit choice points."
        ),
        "conclude_count": (
            f"CONCLUDE count is {metrics.conclude_count}, the number of final conclusions."
        ),
        "total_checkpoints": (
            f"Total checkpoints is {metrics.total_checkpoints}, the total number of annotated checkpoints."
        ),
        "tokens_per_checkpoint": (
            f"Tokens per checkpoint is {metrics.tokens_per_checkpoint:.1f}, a rough density estimate "
            "based on an approximate word count multiplied by 1.3. "
            "Higher values mean more justification per checkpoint."
        ),
        "claim_select_ratio": (
            f"CLAIM/SELECT ratio is {metrics.claim_select_ratio:.2f}, comparing intermediate claims to selections. "
            "Higher values imply more justificatory steps per commitment."
        ),
        "total_tokens": (
            f"Total tokens is {metrics.total_tokens}, a rough estimate derived from word count."
        ),
    }
