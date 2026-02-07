"""
Epistemic Stress Harness — Schema Definitions

Data types and JSON serialization for harness results.
Spec: spec.md section 4.
"""

import json
from typing import TypedDict, Literal, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

SPEC_VERSION = "0.1"

CheckpointType = Literal["ASSUME", "CLAIM", "BRANCH", "SELECT", "CONCLUDE"]

CHECKPOINT_TYPES: List[CheckpointType] = [
    "ASSUME", "CLAIM", "BRANCH", "SELECT", "CONCLUDE"
]


class Checkpoint(TypedDict):
    index: int
    type: CheckpointType
    text: str


@dataclass
class Metrics:
    """Core epistemic integrity metrics (spec section 2)."""
    commitment_latency: float
    assume_count: int
    claim_count: int
    branch_count: int
    select_count: int
    conclude_count: int
    total_checkpoints: int
    tokens_per_checkpoint: float
    claim_select_ratio: float
    total_tokens: int


@dataclass
class TopologyMetrics:
    """Topology comparison metrics, baseline-relative (spec section 3)."""
    node_overlap: float
    sequence_similarity: float
    depth_ratio: float


@dataclass
class HarnessResult:
    """Complete result from a single harness run."""
    variant: str
    raw_text: str
    checkpoints: List[Checkpoint]
    metrics: Metrics


def save_result(result: HarnessResult, filepath: str) -> None:
    """Serialize a HarnessResult to versioned JSON."""
    output = {
        "version": SPEC_VERSION,
        "variant": result.variant,
        "raw_text": result.raw_text,
        "checkpoints": result.checkpoints,
        "metrics": asdict(result.metrics),
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_result(filepath: str) -> Dict[str, Any]:
    """Load a harness result from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
