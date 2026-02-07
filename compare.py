"""
Compare metrics across variants to reveal deformation patterns.

Uses the harness package.
"""

import json
from pathlib import Path

from harness import compare_topology


def load_variant(name):
    """Load a variant's results from the current directory."""
    path = f"results_{name}.json"
    with open(path) as f:
        return json.load(f)


def compare_variants():
    """Compare all variants against baseline"""

    # Load all results
    baseline = load_variant("baseline")
    confidence = load_variant("confidence")
    token_100 = load_variant("token_100")

    b_m = baseline["metrics"]
    c_m = confidence["metrics"]
    t_m = token_100["metrics"]

    print("EPISTEMIC DEFORMATION MANIFOLD")
    print("=" * 70)
    print("\nHow different stressors deform reasoning structure:\n")

    # Table header
    print(f"{'Metric':<25} {'Baseline':>12} {'Confidence':>12} {'Token-100':>12}")
    print("-" * 70)

    # Commitment timing
    print(f"{'Commitment Latency':<25} {b_m['commitment_latency']:>12.2f} {c_m['commitment_latency']:>12.2f} {t_m['commitment_latency']:>12.2f}")

    # Structure preservation
    print(f"{'Branch Count':<25} {b_m['branch_count']:>12} {c_m['branch_count']:>12} {t_m['branch_count']:>12}")
    print(f"{'Total Checkpoints':<25} {b_m['total_checkpoints']:>12} {c_m['total_checkpoints']:>12} {t_m['total_checkpoints']:>12}")

    # Node composition
    print(f"{'ASSUME Count':<25} {b_m['assume_count']:>12} {c_m['assume_count']:>12} {t_m['assume_count']:>12}")
    print(f"{'CLAIM Count':<25} {b_m['claim_count']:>12} {c_m['claim_count']:>12} {t_m['claim_count']:>12}")
    print(f"{'SELECT Count':<25} {b_m['select_count']:>12} {c_m['select_count']:>12} {t_m['select_count']:>12}")

    # Density metrics
    print(f"{'Tokens/Checkpoint':<25} {b_m['tokens_per_checkpoint']:>12.1f} {c_m['tokens_per_checkpoint']:>12.1f} {t_m['tokens_per_checkpoint']:>12.1f}")
    print(f"{'CLAIM/SELECT Ratio':<25} {b_m['claim_select_ratio']:>12.2f} {c_m['claim_select_ratio']:>12.2f} {t_m['claim_select_ratio']:>12.2f}")

    # Topology comparison
    print("\n" + "-" * 70)
    print("\nTOPOLOGY (vs baseline):\n")

    for name, variant in [("Confidence", confidence), ("Token-100", token_100)]:
        topo = compare_topology(baseline["checkpoints"], variant["checkpoints"])
        print(f"  {name}:")
        print(f"    node_overlap:        {topo.node_overlap:.3f}")
        print(f"    sequence_similarity: {topo.sequence_similarity:.3f}")
        print(f"    depth_ratio:         {topo.depth_ratio:.3f}")
        print()

    print("=" * 70)
    print("\nKEY PATTERNS:\n")

    # Confidence pressure effects
    print("CONFIDENCE INCENTIVE:")
    branch_delta = c_m['branch_count'] - b_m['branch_count']
    print(f"  Branch collapse: {b_m['branch_count']} -> {c_m['branch_count']} ({branch_delta:+d})")
    print(f"  Assumptions preserved: {c_m['assume_count']}/{b_m['assume_count']}")
    print(f"  Epistemic posture: {'LINEARIZED' if c_m['branch_count'] == 0 else 'preserved'}")

    print("\nTOKEN CONSTRAINT:")
    density_ratio = t_m['tokens_per_checkpoint'] / b_m['tokens_per_checkpoint']
    print(f"  Density collapse: {b_m['tokens_per_checkpoint']:.0f} -> {t_m['tokens_per_checkpoint']:.0f} ({density_ratio:.1%})")
    print(f"  Structure preserved: {t_m['branch_count']} branches (same as baseline)")
    print(f"  Degradation mode: {'GRACEFUL' if t_m['branch_count'] == b_m['branch_count'] else 'BRITTLE'}")

    print("\n" + "=" * 70)
    print("\nDIAGNOSTIC SIGNATURES:\n")

    if c_m['branch_count'] == 0 and c_m['assume_count'] == b_m['assume_count']:
        print("  Confidence -> OPTIMIZATION OVERRIDE")
        print("  (epistemic collapse without value drift)")

    if t_m['branch_count'] == b_m['branch_count'] and t_m['tokens_per_checkpoint'] < b_m['tokens_per_checkpoint'] * 0.5:
        print("  Token constraint -> GENUINE DEGRADATION")
        print("  (structure preserved, justification compressed)")

    print()


if __name__ == "__main__":
    compare_variants()
