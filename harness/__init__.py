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
    FidelityMetrics,
    OutcomeMetrics,
    HarnessResult,
    SampleRecord,
    save_result,
    load_result,
    load_result_as_object,
    write_jsonl,
    read_jsonl,
)
from .extract import parse_checkpoints, parse_checkpoints_detailed, ParseResult
from .metrics import (
    estimate_tokens,
    commitment_latency,
    commitment_position,
    count_by_type,
    claim_select_ratio,
    tokens_per_checkpoint,
    prose_text,
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
    mean_pairwise_topology,
)
from .fidelity import compute_fidelity
from .outcomes import compute_outcomes, assumption_coverage, check_answer
from .suite import Condition, Item, load_perturbations, load_tasks, load_scenarios, suite_contrasts
from .pipeline import build_sample, run_matrix, load_samples
from .analysis import analyze, format_report
from .judge import Judge, judge_samples, blind_text, parse_judge_output
from .controls import run_offline_controls, run_live_controls, format_controls

__all__ = [
    "SPEC_VERSION", "CheckpointType", "CHECKPOINT_TYPES", "Checkpoint", "Metrics",
    "TopologyMetrics", "FidelityMetrics", "OutcomeMetrics", "HarnessResult", "SampleRecord",
    "save_result", "load_result", "load_result_as_object", "write_jsonl", "read_jsonl",
    "parse_checkpoints", "parse_checkpoints_detailed", "ParseResult",
    "estimate_tokens", "commitment_latency", "commitment_position", "count_by_type",
    "claim_select_ratio", "tokens_per_checkpoint", "prose_text", "compute_metrics",
    "extract_metrics", "explain_metrics",
    "checkpoint_sequence", "node_overlap", "sequence_similarity", "depth_ratio",
    "compare_topology", "mean_pairwise_topology",
    "compute_fidelity", "compute_outcomes", "assumption_coverage", "check_answer",
    "Condition", "Item", "load_perturbations", "load_tasks", "load_scenarios", "suite_contrasts",
    "build_sample", "run_matrix", "load_samples", "analyze", "format_report",
    "Judge", "judge_samples", "blind_text", "parse_judge_output",
    "run_offline_controls", "run_live_controls", "format_controls",
]
