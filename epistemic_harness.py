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
class TopologyMetrics:
    """Topology comparison metrics (requires baseline)"""
    node_overlap: float  # Jaccard similarity of checkpoint types
    sequence_similarity: float  # How similar is the ordering
    depth_ratio: float  # Depth relative to baseline


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


# Topology comparison
def checkpoint_sequence(checkpoints: List[Checkpoint]) -> List[CheckpointType]:
    """Extract ordered sequence of checkpoint types"""
    return [cp["type"] for cp in checkpoints]


def node_overlap(seq1: List[CheckpointType], seq2: List[CheckpointType]) -> float:
    """
    Jaccard similarity of checkpoint type multisets.
    
    Measures whether same node types appear, regardless of order.
    1.0 = identical types, 0.0 = completely different
    """
    from collections import Counter
    
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    count1 = Counter(seq1)
    count2 = Counter(seq2)
    
    # Intersection: min count for each type
    intersection = sum((count1 & count2).values())
    # Union: max count for each type
    union = sum((count1 | count2).values())
    
    return intersection / union if union > 0 else 0.0


def sequence_similarity(seq1: List[CheckpointType], seq2: List[CheckpointType]) -> float:
    """
    Normalized edit distance between checkpoint sequences.
    
    Measures whether reasoning follows same order.
    1.0 = identical sequence, 0.0 = completely different
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    # Simple Levenshtein distance
    m, n = len(seq1), len(seq2)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    edit_distance = dp[m][n]
    max_length = max(m, n)
    
    # Normalize: 1 - (distance / max_length)
    return 1.0 - (edit_distance / max_length) if max_length > 0 else 1.0


def depth_ratio(checkpoints1: List[Checkpoint], checkpoints2: List[Checkpoint]) -> float:
    """
    Ratio of reasoning depth.
    
    Depth = longest chain from ASSUME to CONCLUDE
    Simple approximation: total checkpoints
    """
    if not checkpoints1:
        return 1.0 if not checkpoints2 else 0.0
    
    return len(checkpoints2) / len(checkpoints1)


def compare_topology(baseline: List[Checkpoint], variant: List[Checkpoint]) -> TopologyMetrics:
    """
    Compare topology of two checkpoint sequences.
    
    Returns metrics measuring structural preservation.
    """
    seq_baseline = checkpoint_sequence(baseline)
    seq_variant = checkpoint_sequence(variant)
    
    return TopologyMetrics(
        node_overlap=node_overlap(seq_baseline, seq_variant),
        sequence_similarity=sequence_similarity(seq_baseline, seq_variant),
        depth_ratio=depth_ratio(baseline, variant)
    )


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


def explain_metrics(metrics: Metrics) -> Dict[str, str]:
    """Return human-readable explanations for metric values."""
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
            "based on whitespace-separated word count (text.split()) × 1.3. "
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
