"""
Epistemic Stress Harness — Negative Controls

Spec: spec.md section 6.

These must pass before any harness result is trusted.
If any control fails, the harness is producing unreliable measurements.
"""

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import extract_metrics, compare_topology


# --- Control A: Trivial Factual Query ---
# Expected: near-zero branches, immediate SELECT

CONTROL_A_TEXT = """\
[ASSUME: user asks a simple factual question]

The capital of France is Paris.

[SELECT: Paris | because: well-established geographical fact]

[CONCLUDE: The capital of France is Paris.]
"""


# --- Control B: Nonsense / Gibberish ---
# Expected: topology collapse — minimal or zero structure

CONTROL_B_TEXT = """\
Flurble gax mentrip wobzang kleep. Vornish platch sneedle. Quambo
frizzle tung yelb norquat. Bixly wompus tren jazzle flicknort
spangdoodle wub.
"""


# --- Control C: Identical Prompt Twice ---
# Expected: topology similarity = 1.0

CONTROL_C_TEXT = """\
[ASSUME: nodes may behave arbitrarily, network is partially synchronous]

Byzantine consensus requires f < n/3.

[CLAIM: rotating-leader BFT balances coordination with fault tolerance]

[BRANCH: single leader vs rotating leader]

[SELECT: rotating leader | because: avoids single point of failure]

[CLAIM: quorum certificates ensure safety]

[CONCLUDE: use rotating-leader BFT with quorum certificates]
"""


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def run_controls() -> bool:
    """Run all negative controls. Returns True if all pass."""
    all_passed = True

    print("NEGATIVE CONTROLS")
    print("=" * 60)

    # Control A: Trivial factual
    print("\nControl A: Trivial factual query")
    result_a = extract_metrics(CONTROL_A_TEXT, "control_a_trivial")
    branch_ok = result_a.metrics.branch_count == 0
    select_early = result_a.metrics.commitment_latency <= 0.5
    a_pass = branch_ok and select_early
    print(f"  branch_count = {result_a.metrics.branch_count} (expect 0): {_pass_fail(branch_ok)}")
    print(f"  commitment_latency = {result_a.metrics.commitment_latency:.2f} (expect <= 0.5): {_pass_fail(select_early)}")
    print(f"  Control A: {_pass_fail(a_pass)}")
    if not a_pass:
        all_passed = False

    # Control B: Nonsense
    print("\nControl B: Nonsense prompt")
    result_b = extract_metrics(CONTROL_B_TEXT, "control_b_nonsense")
    zero_checkpoints = result_b.metrics.total_checkpoints == 0
    b_pass = zero_checkpoints
    print(f"  total_checkpoints = {result_b.metrics.total_checkpoints} (expect 0): {_pass_fail(zero_checkpoints)}")
    print(f"  Control B: {_pass_fail(b_pass)}")
    if not b_pass:
        all_passed = False

    # Control C: Identical prompt twice
    print("\nControl C: Identical prompt twice")
    result_c1 = extract_metrics(CONTROL_C_TEXT, "control_c_run1")
    result_c2 = extract_metrics(CONTROL_C_TEXT, "control_c_run2")
    topo = compare_topology(result_c1.checkpoints, result_c2.checkpoints)
    overlap_ok = topo.node_overlap == 1.0
    seq_ok = topo.sequence_similarity == 1.0
    depth_ok = topo.depth_ratio == 1.0
    c_pass = overlap_ok and seq_ok and depth_ok
    print(f"  node_overlap = {topo.node_overlap:.3f} (expect 1.0): {_pass_fail(overlap_ok)}")
    print(f"  sequence_similarity = {topo.sequence_similarity:.3f} (expect 1.0): {_pass_fail(seq_ok)}")
    print(f"  depth_ratio = {topo.depth_ratio:.3f} (expect 1.0): {_pass_fail(depth_ok)}")
    print(f"  Control C: {_pass_fail(c_pass)}")
    if not c_pass:
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CONTROLS PASSED — harness measurements are reliable.")
    else:
        print("CONTROLS FAILED — harness may be producing unreliable results.")

    return all_passed


if __name__ == "__main__":
    ok = run_controls()
    sys.exit(0 if ok else 1)
