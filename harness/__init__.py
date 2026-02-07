"""
Epistemic Stress Harness — Core Package

Public API. Import from here.
"""

from .schema import (
    SPEC_VERSION,
    CheckpointType,
    CHECKPOINT_TYPES,
    Checkpoint,
    Metrics,
    TopologyMetrics,
    HarnessResult,
    save_result,
    load_result,
)
from .extract import parse_checkpoints
from .metrics import (
    commitment_latency,
    count_by_type,
    claim_select_ratio,
    tokens_per_checkpoint,
    compute_metrics,
    extract_metrics,
    explain_metrics,
)
from .topology import (
    checkpoint_sequence,
    node_overlap,
    sequence_similarity,
    depth_ratio,
    compare_topology,
)

__all__ = [
    "SPEC_VERSION",
    "CheckpointType",
    "CHECKPOINT_TYPES",
    "Checkpoint",
    "Metrics",
    "TopologyMetrics",
    "HarnessResult",
    "save_result",
    "load_result",
    "parse_checkpoints",
    "commitment_latency",
    "count_by_type",
    "claim_select_ratio",
    "tokens_per_checkpoint",
    "compute_metrics",
    "extract_metrics",
    "explain_metrics",
    "checkpoint_sequence",
    "node_overlap",
    "sequence_similarity",
    "depth_ratio",
    "compare_topology",
]
