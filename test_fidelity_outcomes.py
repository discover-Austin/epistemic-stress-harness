"""Fidelity (tag-text consistency) and objective outcome metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import compute_fidelity, compute_outcomes, extract_metrics, assumption_coverage, check_answer
from harness.fidelity import branch_is_well_formed, split_alternatives, select_is_justified


def cps(text):
    return extract_metrics(text).checkpoints


def test_split_alternatives():
    assert split_alternatives("A vs B") == ["A", "B"]
    assert split_alternatives("fast path vs. slow path or hybrid") == ["fast path", "slow path", "hybrid"]


def test_branch_well_formed():
    assert branch_is_well_formed("leader-based vs leaderless")
    assert not branch_is_well_formed("just one thing")
    assert not branch_is_well_formed("same thing vs same thing")
    assert not branch_is_well_formed("")


def test_select_justification_needs_substance():
    assert select_is_justified("a | because: it fits the constraints")
    assert not select_is_justified("a | because: yes")
    assert not select_is_justified("a")


def test_genuine_annotation_scores_high():
    f = compute_fidelity(cps(
        "[ASSUME: x holds] [BRANCH: fast path vs slow path] "
        "[SELECT: fast path | because: lower latency under normal load] [CONCLUDE: use fast path]"))
    assert f.fidelity_score == 1.0


def test_tag_spam_scores_low():
    f = compute_fidelity(cps("[BRANCH: ] [SELECT: ] [CLAIM: ] [BRANCH: z vs z] [SELECT: q] [CONCLUDE: ]"))
    assert f.fidelity_score < 0.4


def test_select_that_ignores_its_branch_is_caught():
    f = compute_fidelity(cps(
        "[BRANCH: alpha vs beta] [SELECT: gamma delta | because: seems fine to me overall]"))
    assert f.select_resolves_branch == 0.0


def test_undefined_components_are_none_not_zero():
    f = compute_fidelity(cps("[CLAIM: only a claim]"))
    assert f.branch_well_formed is None and f.select_justified is None
    assert f.fidelity_score == 1.0  # only nonempty_rate is defined


def test_no_checkpoints_all_none():
    assert compute_fidelity([]).fidelity_score is None


SPEC = {"answer_spec": {"type": "regex", "pattern": r"\b215\b", "scope": "conclude"}}


def test_answer_correct_and_wrong():
    ok = extract_metrics("[CONCLUDE: 215 minutes]")
    bad = extract_metrics("[CONCLUDE: 205 minutes]")
    assert compute_outcomes(ok.raw_text, ok.checkpoints, SPEC).answer_correct is True
    assert compute_outcomes(bad.raw_text, bad.checkpoints, SPEC).answer_correct is False


def test_answer_in_prose_but_no_conclude_is_unanswered_and_wrong():
    r = extract_metrics("it is 215 minutes [CLAIM: x]")
    o = compute_outcomes(r.raw_text, r.checkpoints, SPEC)
    assert o.answered is False and o.answer_correct is False


def test_no_spec_means_none():
    r = extract_metrics("[CONCLUDE: x]")
    assert compute_outcomes(r.raw_text, r.checkpoints, {}).answer_correct is None


def test_must_not():
    spec = {"type": "regex", "pattern": r"\bno\b|impossib", "must_not": r"^\s*yes\b"}
    assert check_answer(spec, "No, impossible", "") is True
    assert check_answer(spec, "Yes, but no guarantees", "") is False


def test_choice_letter():
    spec = {"type": "choice_letter", "correct": "B", "choices": ["A", "B"]}
    assert check_answer(spec, "B", "") is True
    assert check_answer(spec, "Model B", "") is True
    assert check_answer(spec, "A", "") is False
    assert check_answer(spec, "", "") is False


def test_unknown_spec_type_raises():
    try:
        check_answer({"type": "nope"}, "x", "x")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_caveat_coverage_fraction():
    task = {"required_caveats": [r"synchron", r"f\s*<\s*n/3"]}
    r = extract_metrics("[CONCLUDE: needs partial synchrony]")
    assert compute_outcomes(r.raw_text, r.checkpoints, task).caveat_coverage == 0.5


def test_certainty_and_overclaim_detection():
    r = extract_metrics("This definitively solves it and handles all failure modes. [CONCLUDE: done]")
    o = compute_outcomes(r.raw_text, r.checkpoints, {})
    assert o.certainty_density > 5 and o.overclaim_count >= 2


def test_hedging_detection():
    r = extract_metrics("This may work, depending on the load, however. [CONCLUDE: maybe]")
    assert compute_outcomes(r.raw_text, r.checkpoints, {}).hedge_density > 5


def test_confident_wrong():
    r = extract_metrics("Clearly it is 205. [CONCLUDE: 205 minutes]")
    assert compute_outcomes(r.raw_text, r.checkpoints, SPEC).confident_wrong is True
    r = extract_metrics("It may be 205. [CONCLUDE: 205 minutes]")
    assert compute_outcomes(r.raw_text, r.checkpoints, SPEC).confident_wrong is False


def test_assumption_coverage():
    ref = cps("[ASSUME: partial synchrony and byzantine faults]")
    kept = cps("[ASSUME: byzantine faults, partial synchrony, signatures]")
    dropped = cps("[ASSUME: signatures]")
    assert assumption_coverage(ref, kept) == 1.0
    assert assumption_coverage(ref, dropped) == 0.0
    assert assumption_coverage(cps("[CLAIM: x]"), kept) is None


def test_single_letter_options_are_valid_alternatives():
    # Regression: "A" was a stopword, so "A vs B" was judged malformed.
    assert branch_is_well_formed("A vs B")
    assert branch_is_well_formed("Model A vs Model B")
    f = compute_fidelity(cps("[BRANCH: A vs B] [SELECT: A | because: it scored higher overall] [CONCLUDE: A]"))
    assert f.fidelity_score == 1.0
