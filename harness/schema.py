"""
Epistemic Stress Harness — Schema Definitions

Data types and JSON (de)serialization for harness results.
Spec: spec.md section 4.

v0.2 adds, on top of the v0.1 fields (all kept, same names and meaning):
  * Metrics: commitment_position, prose_tokens, truncated, malformed_count,
    tokens_source
  * FidelityMetrics: do the tags describe what the text actually does?
  * OutcomeMetrics: objective, task-level measurements (correctness, caveats,
    certainty language) that do not depend on the model's own tags
  * SampleRecord: one model sample with its provenance, one JSONL line each
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

SPEC_VERSION = "0.2"
SUPPORTED_LOAD_VERSIONS = ("0.1", "0.2")

CheckpointType = Literal["ASSUME", "CLAIM", "BRANCH", "SELECT", "CONCLUDE"]

CHECKPOINT_TYPES: List[CheckpointType] = [
    "ASSUME", "CLAIM", "BRANCH", "SELECT", "CONCLUDE"
]


class _CheckpointBase(TypedDict):
    index: int
    type: CheckpointType
    text: str


class Checkpoint(_CheckpointBase, total=False):
    # Character offsets of the full tag in the source text (v0.2, optional so
    # v0.1 result files still load).
    start: int
    end: int


@dataclass
class Metrics:
    """Core structural metrics (spec section 2). v0.1 fields come first."""
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
    # --- v0.2 ---
    commitment_position: float = 0.0   # char-offset fraction of first SELECT/CONCLUDE
    prose_tokens: int = 0              # tokens outside checkpoint tags
    truncated: bool = False            # generation hit the token limit
    malformed_count: int = 0           # unterminated / interrupted tags
    tokens_source: str = "estimate"    # "estimate" | "reported"


@dataclass
class TopologyMetrics:
    """Topology comparison metrics, reference-relative (spec section 3)."""
    node_overlap: float
    sequence_similarity: float
    depth_ratio: float


@dataclass
class FidelityMetrics:
    """
    Tag-text consistency (spec section 4.2). Each component is a fraction in
    [0, 1], or None when it is undefined for the sample (e.g. no BRANCH tags).
    """
    branch_well_formed: Optional[float] = None
    select_justified: Optional[float] = None
    select_resolves_branch: Optional[float] = None
    nonempty_rate: Optional[float] = None
    fidelity_score: Optional[float] = None


@dataclass
class OutcomeMetrics:
    """
    Objective outcome measurements (spec section 4.3).
    None means the task defines no ground truth for that measurement.
    """
    answered: bool = False                    # a CONCLUDE checkpoint exists
    answer_correct: Optional[bool] = None
    caveat_coverage: Optional[float] = None   # fraction of required caveats present
    certainty_density: float = 0.0            # certainty markers per 100 words
    hedge_density: float = 0.0                # hedging markers per 100 words
    overclaim_count: int = 0                  # universal / absolute claims
    confident_wrong: Optional[bool] = None    # wrong AND stated with certainty


@dataclass
class HarnessResult:
    """Complete result from a single extraction."""
    variant: str
    raw_text: str
    checkpoints: List[Checkpoint]
    metrics: Metrics


@dataclass
class SampleRecord:
    """One model sample plus provenance. Serialized as one JSONL line."""
    task_id: str
    variant: str
    sample_index: int
    runner: str = ""
    prompt: str = ""                      # exact user prompt sent
    system_suffix: Optional[str] = None   # condition suffix appended to the system prompt
    temperature: Optional[float] = None
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    result: Optional[HarnessResult] = None
    fidelity: Optional[FidelityMetrics] = None
    outcome: Optional[OutcomeMetrics] = None
    judge: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (self.task_id, self.variant, self.sample_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": SPEC_VERSION,
            "task_id": self.task_id,
            "variant": self.variant,
            "sample_index": self.sample_index,
            "runner": self.runner,
            "prompt": self.prompt,
            "system_suffix": self.system_suffix,
            "temperature": self.temperature,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "result": _result_to_dict(self.result) if self.result else None,
            "fidelity": asdict(self.fidelity) if self.fidelity else None,
            "outcome": asdict(self.outcome) if self.outcome else None,
            "judge": self.judge,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleRecord":
        return cls(
            task_id=d["task_id"],
            variant=d["variant"],
            sample_index=d["sample_index"],
            runner=d.get("runner", ""),
            prompt=d.get("prompt", ""),
            system_suffix=d.get("system_suffix"),
            temperature=d.get("temperature"),
            stop_reason=d.get("stop_reason"),
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            result=_result_from_dict(d["result"]) if d.get("result") else None,
            fidelity=_dc_from_dict(FidelityMetrics, d["fidelity"]) if d.get("fidelity") else None,
            outcome=_dc_from_dict(OutcomeMetrics, d["outcome"]) if d.get("outcome") else None,
            judge=d.get("judge"),
            error=d.get("error"),
        )


# --------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------

def _dc_from_dict(cls, d: Dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys (forward compat)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _result_to_dict(result: HarnessResult) -> Dict[str, Any]:
    return {
        "variant": result.variant,
        "raw_text": result.raw_text,
        "checkpoints": result.checkpoints,
        "metrics": asdict(result.metrics),
    }


def _result_from_dict(d: Dict[str, Any]) -> HarnessResult:
    return HarnessResult(
        variant=d["variant"],
        raw_text=d["raw_text"],
        checkpoints=d["checkpoints"],
        metrics=_dc_from_dict(Metrics, d["metrics"]),
    )


def save_result(result: HarnessResult, filepath: str) -> None:
    """Serialize a HarnessResult to versioned JSON."""
    output = {"version": SPEC_VERSION, **_result_to_dict(result)}
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def load_result(filepath: str) -> Dict[str, Any]:
    """Load a harness result from JSON. Accepts v0.1 and v0.2 files."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("version")
    if version not in SUPPORTED_LOAD_VERSIONS:
        raise ValueError(
            f"{filepath}: unsupported result version {version!r} "
            f"(supported: {', '.join(SUPPORTED_LOAD_VERSIONS)})"
        )
    return data


def load_result_as_object(filepath: str) -> HarnessResult:
    """Load a result file (v0.1 or v0.2) as a HarnessResult."""
    return _result_from_dict(load_result(filepath))


def write_jsonl(path: Path, record: SampleRecord) -> None:
    """Append one sample to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_jsonl(path: Path) -> List[SampleRecord]:
    """Read all samples from a JSONL file. Skips blank lines."""
    records: List[SampleRecord] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(SampleRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{lineno}: bad sample record ({e})") from e
    return records
