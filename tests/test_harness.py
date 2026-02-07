"""
Epistemic Stress Harness — Unit Tests

Tests for core extraction, metrics, and topology.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    parse_checkpoints,
    extract_metrics,
    compare_topology,
    commitment_latency,
    count_by_type,
    claim_select_ratio,
    tokens_per_checkpoint,
    node_overlap,
    sequence_similarity,
    checkpoint_sequence,
)


def test_parse_checkpoints_basic():
    text = "[ASSUME: test] some text [CLAIM: mid] [CONCLUDE: end]"
    cps = parse_checkpoints(text)
    assert len(cps) == 3
    assert cps[0]["type"] == "ASSUME"
    assert cps[1]["type"] == "CLAIM"
    assert cps[2]["type"] == "CONCLUDE"
    assert cps[0]["text"] == "test"


def test_parse_checkpoints_case_insensitive():
    text = "[assume: lower] [Claim: mixed] [BRANCH: upper]"
    cps = parse_checkpoints(text)
    assert len(cps) == 3
    assert all(cp["type"].isupper() for cp in cps)


def test_parse_checkpoints_no_content():
    text = "[CONCLUDE]"
    cps = parse_checkpoints(text)
    assert len(cps) == 1
    assert cps[0]["type"] == "CONCLUDE"
    assert cps[0]["text"] == ""


def test_parse_checkpoints_empty():
    assert parse_checkpoints("no checkpoints here") == []
    assert parse_checkpoints("") == []


def test_commitment_latency_early():
    cps = [
        {"index": 0, "type": "SELECT", "text": ""},
        {"index": 1, "type": "CLAIM", "text": ""},
    ]
    assert commitment_latency(cps) == 0.0


def test_commitment_latency_late():
    cps = [
        {"index": 0, "type": "ASSUME", "text": ""},
        {"index": 1, "type": "CLAIM", "text": ""},
        {"index": 2, "type": "BRANCH", "text": ""},
        {"index": 3, "type": "SELECT", "text": ""},
    ]
    assert commitment_latency(cps) == 0.75


def test_commitment_latency_no_commitment():
    cps = [
        {"index": 0, "type": "ASSUME", "text": ""},
        {"index": 1, "type": "CLAIM", "text": ""},
    ]
    assert commitment_latency(cps) == 1.0


def test_commitment_latency_empty():
    assert commitment_latency([]) == 0.0


def test_count_by_type():
    cps = [
        {"index": 0, "type": "CLAIM", "text": ""},
        {"index": 1, "type": "CLAIM", "text": ""},
        {"index": 2, "type": "BRANCH", "text": ""},
    ]
    assert count_by_type(cps, "CLAIM") == 2
    assert count_by_type(cps, "BRANCH") == 1
    assert count_by_type(cps, "ASSUME") == 0


def test_claim_select_ratio_normal():
    cps = [
        {"index": 0, "type": "CLAIM", "text": ""},
        {"index": 1, "type": "CLAIM", "text": ""},
        {"index": 2, "type": "SELECT", "text": ""},
    ]
    assert claim_select_ratio(cps) == 2.0


def test_claim_select_ratio_no_selects():
    cps = [{"index": 0, "type": "CLAIM", "text": ""}]
    assert claim_select_ratio(cps) == 1.0


def test_claim_select_ratio_empty():
    assert claim_select_ratio([]) == 0.0


def test_node_overlap_identical():
    seq = ["ASSUME", "CLAIM", "SELECT"]
    assert node_overlap(seq, seq) == 1.0


def test_node_overlap_disjoint():
    assert node_overlap(["ASSUME"], ["CONCLUDE"]) < 0.5


def test_node_overlap_empty():
    assert node_overlap([], []) == 1.0
    assert node_overlap(["ASSUME"], []) == 0.0


def test_sequence_similarity_identical():
    seq = ["ASSUME", "CLAIM", "BRANCH", "SELECT"]
    assert sequence_similarity(seq, seq) == 1.0


def test_sequence_similarity_different():
    seq1 = ["ASSUME", "CLAIM", "SELECT"]
    seq2 = ["BRANCH", "CONCLUDE"]
    sim = sequence_similarity(seq1, seq2)
    assert 0.0 <= sim <= 1.0


def test_extract_metrics_integration():
    text = """
    [ASSUME: starting point]
    Some reasoning here.
    [CLAIM: intermediate]
    More reasoning.
    [BRANCH: option A vs option B]
    [SELECT: option A | because: better]
    [CONCLUDE: final answer]
    """
    result = extract_metrics(text, "test")
    assert result.variant == "test"
    assert result.metrics.total_checkpoints == 5
    assert result.metrics.assume_count == 1
    assert result.metrics.branch_count == 1
    assert result.metrics.select_count == 1
    assert result.metrics.conclude_count == 1
    assert result.metrics.commitment_latency == 0.6


def test_compare_topology_identical():
    text = "[ASSUME: a] [CLAIM: b] [SELECT: c] [CONCLUDE: d]"
    r1 = extract_metrics(text, "a")
    r2 = extract_metrics(text, "b")
    topo = compare_topology(r1.checkpoints, r2.checkpoints)
    assert topo.node_overlap == 1.0
    assert topo.sequence_similarity == 1.0
    assert topo.depth_ratio == 1.0


def run_all():
    """Run all tests, report results."""
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    print("UNIT TESTS")
    print("=" * 60)
    ok = run_all()
    sys.exit(0 if ok else 1)
