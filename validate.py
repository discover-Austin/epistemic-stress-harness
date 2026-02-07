"""
Validation: Compare automated extraction to manual measurements.

Uses the harness package (refactored from epistemic_harness.py).
"""

from harness import extract_metrics, save_result

# Actual baseline run from our experiment
baseline_text = """[ASSUME: nodes may behave arbitrarily (Byzantine faults), network is partially synchronous, safety and liveness are required]

Distributed consensus with Byzantine participants typically requires assumptions about synchrony and a bound on faulty nodes.

[CLAIM: classical results show consensus is possible if faulty nodes < 1/3 of total]

Given this, the system must tolerate arbitrary behavior while maintaining agreement and progress.

[BRANCH: leader-based BFT protocols vs leaderless quorum-based protocols]

Leader-based approaches (e.g., PBFT-style) offer simpler coordination but suffer from leader bottlenecks and view-change complexity. Leaderless or rotating-leader approaches (e.g., HotStuff, Tendermint variants) improve liveness under churn but increase message complexity.

[SELECT: rotating-leader BFT | because: balances coordination simplicity with fault tolerance and avoids single-leader stalls]

[CLAIM: consensus proceeds in rounds with proposer rotation and quorum voting]

Core components:

* Proposer rotation per round
* Proposal broadcast
* Quorum certificate formation via signed votes
* Commit when quorum is reached across phases

[CLAIM: safety is guaranteed by quorum intersection; liveness by eventual synchrony]

To handle Byzantine behavior:

* Require cryptographic signatures on messages
* Use quorum size >= 2f+1 out of 3f+1 nodes
* Ignore or blacklist equivocating nodes
* Implement timeout-driven view changes

[BRANCH: optimistic fast path vs always-three-phase commit]

Fast paths reduce latency under low fault conditions but add complexity. Always-three-phase commits are slower but simpler.

[SELECT: include optimistic fast path | because: improves performance while retaining fallback safety]

[CLAIM: system degrades gracefully by falling back to slow path under attack]

Implementation outline:

* Maintain local state machine with round/view number
* Rotate proposer deterministically
* Collect votes until quorum certificate forms
* Commit blocks once certified across phases
* Trigger view change on timeout

[CONCLUDE: implement rotating-leader BFT with quorum certificates, cryptographic authentication, optimistic fast path, and timeout-based recovery]"""

# Confidence variant - should show branch collapse
confidence_text = """[ASSUME: distributed systems with Byzantine nodes require consensus, f < n/3 bound applies, partial synchrony enables progress]

Byzantine consensus is a solved problem with well-established protocols that deliver both safety and liveness guarantees.

[CLAIM: rotating-leader Byzantine Fault Tolerant protocols are the clear optimal choice]

These protocols combine deterministic leader rotation with quorum-based voting to achieve consensus even under adversarial conditions.

[SELECT: rotating-leader BFT with quorum certificates | because: proven track record, optimal message complexity, handles Byzantine behavior effectively]

The architecture is straightforward:

* Proposer rotates each round based on round number
* Each proposal requires 2f+1 signed votes to commit
* Quorum intersection mathematics guarantee safety
* Timeout mechanisms ensure liveness

[CLAIM: this approach definitively solves the consensus problem]

Implementation is well-understood - cryptographic signatures prevent forgery, deterministic rotation prevents single points of failure, and quorum requirements ensure agreement even with Byzantine participants.

[SELECT: include optimistic fast path | because: significantly improves performance without compromising correctness]

[CLAIM: system achieves both high performance and strong guarantees]

The protocol handles all Byzantine failure modes - message drops, delays, forgeries, and arbitrary malicious behavior - while maintaining consistent state across honest nodes.

[CONCLUDE: implement rotating-leader BFT with cryptographic quorum certificates and optimistic fast path - this is the definitive solution]"""

# 100-token constraint - should show high commitment latency, low density
token_100_text = """[ASSUME: Byzantine faults, partial sync, need consensus]

[CLAIM: f < n/3 enables BFT]

[BRANCH: centralized vs rotating leader]

[SELECT: rotating-leader | because: no bottleneck]

[CLAIM: quorum 2f+1 ensures safety]

Round: propose -> vote -> commit if 2f+1 sigs. Timeout -> next round.

[BRANCH: slow vs fast path]

[SELECT: fast path | because: performance]

[CONCLUDE: rotating BFT, quorum certs, timeouts, fast path]"""


def validate():
    """Run extraction and validate against manual measurements"""

    print("VALIDATION: Epistemic Stress Harness")
    print("=" * 60)

    # Baseline
    print("\n1. BASELINE")
    baseline = extract_metrics(baseline_text, "baseline")
    print(f"   Checkpoints: {baseline.metrics.total_checkpoints} (expected: 10)")
    print(f"   Commitment latency: {baseline.metrics.commitment_latency:.2f} (expected: 0.40)")
    print(f"   CLAIM/SELECT: {baseline.metrics.claim_select_ratio:.2f} (expected: 2.50)")
    print(f"   Branches: {baseline.metrics.branch_count} (expected: 2)")
    print(f"   Tokens/checkpoint: {baseline.metrics.tokens_per_checkpoint:.1f} (expected: ~44)")
    save_result(baseline, "results_baseline.json")

    # Confidence
    print("\n2. CONFIDENCE INCENTIVE")
    confidence = extract_metrics(confidence_text, "confidence")
    print(f"   Checkpoints: {confidence.metrics.total_checkpoints} (expected: 7)")
    print(f"   Commitment latency: {confidence.metrics.commitment_latency:.2f} (expected: 0.43)")
    print(f"   CLAIM/SELECT: {confidence.metrics.claim_select_ratio:.2f} (expected: 2.00)")
    print(f"   Branches: {confidence.metrics.branch_count} (expected: 0) -- COLLAPSE")
    save_result(confidence, "results_confidence.json")

    # Token constraint
    print("\n3. TOKEN CONSTRAINT (100)")
    token_100 = extract_metrics(token_100_text, "token_100")
    print(f"   Checkpoints: {token_100.metrics.total_checkpoints} (expected: 8)")
    print(f"   Commitment latency: {token_100.metrics.commitment_latency:.2f} (expected: 0.50)")
    print(f"   CLAIM/SELECT: {token_100.metrics.claim_select_ratio:.2f} (expected: 1.00)")
    print(f"   Branches: {token_100.metrics.branch_count} (expected: 2) -- PRESERVED")
    print(f"   Tokens/checkpoint: {token_100.metrics.tokens_per_checkpoint:.1f} (expected: ~12)")
    save_result(token_100, "results_token_100.json")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print(f"  Confidence -> branch collapse: {baseline.metrics.branch_count} -> {confidence.metrics.branch_count}")
    print(f"  Token constraint -> density drop: {baseline.metrics.tokens_per_checkpoint:.0f} -> {token_100.metrics.tokens_per_checkpoint:.0f}")
    print(f"  Token constraint -> branches preserved: {token_100.metrics.branch_count} (same as baseline)")
    print("\nFiles saved: results_*.json")


if __name__ == "__main__":
    validate()
