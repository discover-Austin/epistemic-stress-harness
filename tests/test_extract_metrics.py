"""Parser and metric behavior added or fixed in v0.2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    parse_checkpoints, parse_checkpoints_detailed, extract_metrics, prose_text,
    commitment_position, explain_metrics,
)


def test_balanced_brackets_in_content():
    cps = parse_checkpoints("[CLAIM: arr[0] < arr[1]] then [CONCLUDE: done]")
    assert [c["text"] for c in cps] == ["arr[0] < arr[1]", "done"]


def test_code_regions_ignored():
    text = "```\n[CLAIM: fenced]\n```\n`[BRANCH: inline]` [CLAIM: real]\n~~~\n[SELECT: tilde]\n~~~"
    assert [c["text"] for c in parse_checkpoints(text)] == ["real"]


def test_unterminated_fence_masks_to_end():
    assert parse_checkpoints("[CLAIM: a]\n```\n[CLAIM: b]") == parse_checkpoints("[CLAIM: a]")


def test_unterminated_tag_is_malformed_not_swallowing_the_next():
    r = parse_checkpoints_detailed("[CLAIM: cut off [SELECT: y | because: z] [CONCLUDE: ok]")
    assert r.malformed_count == 1
    assert [c["type"] for c in r.checkpoints] == ["SELECT", "CONCLUDE"]


def test_unterminated_at_end_of_text():
    r = parse_checkpoints_detailed("[CLAIM: fine] [SELECT: cut off")
    assert r.malformed_count == 1 and len(r.checkpoints) == 1


def test_type_must_be_followed_by_colon_or_bracket():
    assert parse_checkpoints("[ASSUMED: x] [CLAIMS: y]") == []


def test_offsets_slice_back_to_the_tag():
    text = "intro [CLAIM: x] mid [CONCLUDE]"
    for cp in parse_checkpoints(text):
        assert text[cp["start"]:cp["end"]].startswith("[" + cp["type"])


def test_prose_text_removes_tags():
    text = "a [CLAIM: x] b [CONCLUDE: y] c"
    assert prose_text(text, parse_checkpoints(text)).split() == ["a", "b", "c"]


def test_commitment_position_is_density_independent():
    sparse = "words " * 50 + "[SELECT: a | because: b c d]"
    dense = "[CLAIM: x] " * 10 + "words " * 50 + "[SELECT: a | because: b c d]"
    p1 = extract_metrics(sparse).metrics.commitment_position
    p2 = extract_metrics(dense).metrics.commitment_position
    assert p1 > 0.9 and p2 > 0.5
    # commitment_latency, by contrast, jumps with tag count alone
    assert extract_metrics(sparse).metrics.commitment_latency == 0.0
    assert extract_metrics(dense).metrics.commitment_latency > 0.8


def test_commitment_position_no_commitment_is_one():
    assert extract_metrics("text [CLAIM: x]").metrics.commitment_position == 1.0


def test_reported_tokens_replace_estimate():
    r = extract_metrics("one two three [CLAIM: x]", reported_output_tokens=500)
    assert r.metrics.total_tokens == 500 and r.metrics.tokens_source == "reported"
    assert r.metrics.tokens_per_checkpoint == 500.0


def test_estimate_is_labelled():
    assert extract_metrics("a b c").metrics.tokens_source == "estimate"


def test_truncated_and_malformed_flow_through():
    r = extract_metrics("[CLAIM: ok] [SELECT: cut", truncated=True)
    assert r.metrics.truncated and r.metrics.malformed_count == 1
    assert "cut off" in explain_metrics(r.metrics)["truncated"]


def test_custom_token_estimator():
    r = extract_metrics("a b c d", token_estimator=lambda t: 7)
    assert r.metrics.total_tokens == 7
