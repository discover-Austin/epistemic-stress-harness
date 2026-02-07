"""
Epistemic Stress Harness — Topology Comparison

Baseline-relative structural metrics.
Spec: spec.md section 3.
"""

from collections import Counter
from typing import List

from .schema import Checkpoint, CheckpointType, TopologyMetrics


def checkpoint_sequence(checkpoints: List[Checkpoint]) -> List[CheckpointType]:
    """Extract ordered sequence of checkpoint types."""
    return [cp["type"] for cp in checkpoints]


def node_overlap(seq1: List[CheckpointType], seq2: List[CheckpointType]) -> float:
    """
    Jaccard similarity of checkpoint type multisets.

    1.0 = identical type distribution, 0.0 = completely different.
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    count1 = Counter(seq1)
    count2 = Counter(seq2)

    intersection = sum((count1 & count2).values())
    union = sum((count1 | count2).values())

    return intersection / union if union > 0 else 0.0


def sequence_similarity(seq1: List[CheckpointType], seq2: List[CheckpointType]) -> float:
    """
    1 - normalized Levenshtein distance between checkpoint sequences.

    1.0 = identical ordering, 0.0 = completely different.
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    edit_distance = dp[m][n]
    max_length = max(m, n)

    return 1.0 - (edit_distance / max_length) if max_length > 0 else 1.0


def depth_ratio(baseline: List[Checkpoint], variant: List[Checkpoint]) -> float:
    """
    Ratio of reasoning depth (variant / baseline).

    Approximated by total checkpoint count.
    """
    if not baseline:
        return 1.0 if not variant else 0.0

    return len(variant) / len(baseline)


def compare_topology(baseline: List[Checkpoint], variant: List[Checkpoint]) -> TopologyMetrics:
    """
    Compare topology of variant checkpoint sequence against baseline.

    Returns structural preservation metrics.
    """
    seq_baseline = checkpoint_sequence(baseline)
    seq_variant = checkpoint_sequence(variant)

    return TopologyMetrics(
        node_overlap=node_overlap(seq_baseline, seq_variant),
        sequence_similarity=sequence_similarity(seq_baseline, seq_variant),
        depth_ratio=depth_ratio(baseline, variant),
    )
