"""
Epistemic Stress Harness - Core Measurement Layer

Extracts checkpoints from annotated responses and computes
epistemic integrity metrics under constraint.
"""

import re
import json
from typing import TypedDict, Literal, List, Dict, Any
from dataclasses import dataclass, asdict


# Schema definitions
CheckpointType = Literal["ASSUME", "CLAIM", "BRANCH", "SELECT", "CONCLUDE"]

class Checkpoint(TypedDict):
    index: int
    type: CheckpointType
    text: str


@dataclass
class Metrics:
    """Core epistemic integrity metrics"""
    # Commitment timing
    commitment_latency: float
    
    # Node counts
    assume_count: int
    claim_count: int
    branch_count: int
    select_count: int
    conclude_count: int
    total_checkpoints: int
    
    # Density measures
    tokens_per_checkpoint: float
    claim_select_ratio: float
    
    # Raw data
    total_tokens: int


@dataclass
class HarnessResult:
    """Complete result from running the harness"""
    variant: str
    raw_text: str
    checkpoints: List[Checkpoint]
    metrics: Metrics


# Checkpoint extraction
def parse_checkpoints(text: str) -> List[Checkpoint]:
    """
    Extract checkpoints from annotated text.
    
    Pattern: [TYPE: content] or [TYPE]
    Types: ASSUME, CLAIM, BRANCH, SELECT, CONCLUDE
    """
    # Find all checkpoints with their positions
    pattern = r'\[(ASSUME|CLAIM|BRANCH|SELECT|CONCLUDE):?\s*([^\]]*)\]'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    checkpoints = []
    for idx, match in enumerate(matches):
        checkpoint_type = match.group(1).upper()
        checkpoint_text = match.group(2).strip()
        
        checkpoints.append({
            "index": idx,
            "type": checkpoint_type,  # type: ignore
            "text": checkpoint_text
        })
    
    return checkpoints


# Metric extractors - pure functions
def commitment_latency(checkpoints: List[Checkpoint]) -> float:
    """
    Fraction of checkpoints before first SELECT or CONCLUDE.
    
    Lower = earlier commitment
    Higher = prolonged exploration
    """
    if not checkpoints:
        return 0.0
    
    for i, cp in enumerate(checkpoints):
        if cp["type"] in ("SELECT", "CONCLUDE"):
            return i / len(checkpoints)
    
    return 1.0  # No commitment found


def count_by_type(checkpoints: List[Checkpoint], cp_type: CheckpointType) -> int:
    """Count checkpoints of specific type"""
    return sum(1 for cp in checkpoints if cp["type"] == cp_type)


def claim_select_ratio(checkpoints: List[Checkpoint]) -> float:
    """
    Ratio of CLAIM to SELECT nodes.
    
    Higher = more intermediate reasoning
    Lower = more decisive jumps
    """
    claims = count_by_type(checkpoints, "CLAIM")
    selects = count_by_type(checkpoints, "SELECT")
    
    if selects == 0:
        return float(claims) if claims > 0 else 0.0
    
    return claims / selects


def tokens_per_checkpoint(text: str, checkpoints: List[Checkpoint]) -> float:
    """
    Average tokens between checkpoints.
    Measures justification density.
    
    Uses rough token estimate (words * 1.3)
    """
    if not checkpoints:
        return 0.0
    
    # Rough token count
    words = len(text.split())
    tokens = int(words * 1.3)
    
    return tokens / len(checkpoints)


# Main extraction
def extract_metrics(text: str, variant: str = "baseline") -> HarnessResult:
    """
    Extract all metrics from annotated text.
    
    Args:
        text: Response with checkpoint annotations
        variant: Name of perturbation variant
        
    Returns:
        Complete harness result with metrics
    """
    checkpoints = parse_checkpoints(text)
    
    # Count tokens (rough estimate)
    words = len(text.split())
    total_tokens = int(words * 1.3)
    
    # Compute metrics
    metrics = Metrics(
        commitment_latency=commitment_latency(checkpoints),
        assume_count=count_by_type(checkpoints, "ASSUME"),
        claim_count=count_by_type(checkpoints, "CLAIM"),
        branch_count=count_by_type(checkpoints, "BRANCH"),
        select_count=count_by_type(checkpoints, "SELECT"),
        conclude_count=count_by_type(checkpoints, "CONCLUDE"),
        total_checkpoints=len(checkpoints),
        tokens_per_checkpoint=tokens_per_checkpoint(text, checkpoints),
        claim_select_ratio=claim_select_ratio(checkpoints),
        total_tokens=total_tokens
    )
    
    return HarnessResult(
        variant=variant,
        raw_text=text,
        checkpoints=checkpoints,
        metrics=metrics
    )


def save_result(result: HarnessResult, filepath: str):
    """Save harness result to JSON"""
    output = {
        "variant": result.variant,
        "raw_text": result.raw_text,
        "checkpoints": result.checkpoints,
        "metrics": asdict(result.metrics)
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def load_result(filepath: str) -> Dict[str, Any]:
    """Load harness result from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Quick test on baseline run
    test_text = """
[ASSUME: nodes may behave arbitrarily (Byzantine faults), network is partially synchronous, safety and liveness are required]

Distributed consensus with Byzantine participants typically requires assumptions about synchrony and a bound on faulty nodes.

[CLAIM: classical results show consensus is possible if faulty nodes < 1/3 of total]

Given this, the system must tolerate arbitrary behavior while maintaining agreement and progress.

[BRANCH: leader-based BFT protocols vs leaderless quorum-based protocols]

Leader-based approaches (e.g., PBFT-style) offer simpler coordination but suffer from leader bottlenecks.

[SELECT: rotating-leader BFT | because: balances coordination simplicity with fault tolerance]

[CLAIM: consensus proceeds in rounds with proposer rotation and quorum voting]

Core components: proposer rotation, proposal broadcast, quorum certificate formation.

[CONCLUDE: implement rotating-leader BFT with quorum certificates, cryptographic authentication, and timeout-based recovery]
"""
    
    result = extract_metrics(test_text, "baseline")
    
    print("Epistemic Stress Harness - Test Run")
    print("=" * 50)
    print(f"Variant: {result.variant}")
    print(f"Total checkpoints: {result.metrics.total_checkpoints}")
    print(f"Commitment latency: {result.metrics.commitment_latency:.2f}")
    print(f"CLAIM/SELECT ratio: {result.metrics.claim_select_ratio:.2f}")
    print(f"Tokens/checkpoint: {result.metrics.tokens_per_checkpoint:.1f}")
    print(f"Branch count: {result.metrics.branch_count}")
    print("\nCheckpoints:")
    for cp in result.checkpoints:
        print(f"  [{cp['index']}] {cp['type']}: {cp['text'][:50]}...")
