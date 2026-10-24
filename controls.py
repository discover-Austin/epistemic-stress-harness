"""
Epistemic Stress Harness — Controls

Controls must pass before any harness result is trusted. Spec section 6.

Two kinds, and the difference matters:

  parser / metric controls (offline)
      Test that the *instrument* behaves: the parser is deterministic, metrics
      are sensitive to what they claim to measure, and structural metrics alone
      can be gamed (the gaming control shows fidelity catching tag spam that
      the structural metrics rate as rich). They involve no model.

  live controls (need a runner)
      Test that the *measurement on a model* is usable: the noise floor is
      small enough relative to structure, a trivial question produces trivial
      structure, a nonsense prompt is not answered with confidence.

The v0.1 "controls" parsed hard-coded strings and passed by construction; they
are kept here, honestly named, as parser self-tests.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from .extract import parse_checkpoints, parse_checkpoints_detailed
from .fidelity import compute_fidelity
from .metrics import extract_metrics, claim_select_ratio
from .outcomes import check_answer
from .topology import compare_topology, mean_pairwise_topology


@dataclass
class ControlResult:
    name: str
    kind: str          # "offline" | "live"
    passed: bool
    detail: str


_SAMPLE = """\
[ASSUME: nodes may behave arbitrarily, network is partially synchronous]

Byzantine consensus requires f < n/3.

[CLAIM: rotating-leader BFT balances coordination with fault tolerance]

[BRANCH: single leader vs rotating leader]

[SELECT: rotating leader | because: avoids a single point of failure]

[CLAIM: quorum certificates ensure safety]

[CONCLUDE: use rotating-leader BFT with quorum certificates]
"""

_TRIVIAL = """\
[ASSUME: the user asks a simple factual question]

The capital of France is Paris.

[SELECT: Paris | because: well-established geographical fact]

[CONCLUDE: The capital of France is Paris.]
"""

_GIBBERISH = ("Flurble gax mentrip wobzang kleep. Vornish platch sneedle. Quambo frizzle tung "
              "yelb norquat. Bixly wompus tren jazzle flicknort spangdoodle wub.")

# Tag spam: many tags, nothing behind them.
_SPAM = ("[ASSUME: ] zork [CLAIM: ] [BRANCH: x] flib [BRANCH: ] [SELECT: y] wub [SELECT: ] "
         "[CLAIM: ] [BRANCH: z vs z] gax [SELECT: q] [CONCLUDE: ]")


def run_offline_controls() -> List[ControlResult]:
    R: List[ControlResult] = []

    def add(name: str, ok: bool, detail: str) -> None:
        R.append(ControlResult(name, "offline", bool(ok), detail))

    # P1 parser determinism (was "Control C")
    a, b = extract_metrics(_SAMPLE, "a"), extract_metrics(_SAMPLE, "b")
    t = compare_topology(a.checkpoints, b.checkpoints)
    add("P1 parser determinism", t.node_overlap == t.sequence_similarity == t.depth_ratio == 1.0,
        f"identical text twice -> overlap={t.node_overlap:.2f} seq={t.sequence_similarity:.2f}. "
        "Tests the parser only; sampling noise is measured by live control L1.")

    # P2 untagged text yields no structure (was "Control B")
    n = extract_metrics(_GIBBERISH).metrics.total_checkpoints
    add("P2 untagged text -> no checkpoints", n == 0, f"total_checkpoints={n} (by construction)")

    # P3 trivial text (was "Control A")
    m = extract_metrics(_TRIVIAL).metrics
    add("P3 trivial text -> no branches, early commitment",
        m.branch_count == 0 and m.commitment_latency <= 0.5,
        f"branches={m.branch_count} latency={m.commitment_latency:.2f}")

    # M1 sensitivity: shuffling order must change sequence similarity but not composition
    cps = extract_metrics(_SAMPLE).checkpoints
    shuffled = cps[:]
    random.Random(3).shuffle(shuffled)
    ts = compare_topology(cps, shuffled)
    add("M1 sequence metric is order-sensitive",
        ts.node_overlap == 1.0 and ts.sequence_similarity < 1.0,
        f"shuffled -> overlap={ts.node_overlap:.2f} (expect 1.00) seq={ts.sequence_similarity:.2f} (expect <1)")

    # M2 gaming: tag spam looks structurally rich; fidelity must reject it
    spam = extract_metrics(_SPAM)
    fid = compute_fidelity(spam.checkpoints)
    add("M2 tag spam: rich structure, low fidelity",
        spam.metrics.total_checkpoints >= 8 and fid.fidelity_score is not None and fid.fidelity_score < 0.5,
        f"checkpoints={spam.metrics.total_checkpoints} fidelity={fid.fidelity_score:.2f} (expect <0.50)")
    real = compute_fidelity(extract_metrics(_SAMPLE).checkpoints)
    add("M2b genuine annotation: high fidelity",
        real.fidelity_score is not None and real.fidelity_score >= 0.9,
        f"fidelity={real.fidelity_score:.2f} (expect >=0.90)")

    # M3 tags inside code are ignored
    code = "```\n[CLAIM: in a fence]\n```\nand `[BRANCH: inline]` but [CLAIM: real]"
    got = [c["text"] for c in parse_checkpoints(code)]
    add("M3 code regions ignored", got == ["real"], f"parsed={got}")

    # M4 balanced brackets inside content
    got = [c["text"] for c in parse_checkpoints("[CLAIM: arr[0] is the pivot]")]
    add("M4 balanced brackets in content", got == ["arr[0] is the pivot"], f"parsed={got}")

    # M5 case strictness per spec
    strict = len(parse_checkpoints("[claim: x]"))
    lenient = len(parse_checkpoints("[claim: x]", strict=False))
    add("M5 strict case by default", strict == 0 and lenient == 1, f"strict={strict} lenient={lenient}")

    # M6 truncation and malformed tags are surfaced, not hidden
    pr = parse_checkpoints_detailed("[CLAIM: ok] [SELECT: cut off mid")
    r = extract_metrics("[CLAIM: ok] [SELECT: cut off mid", truncated=True)
    add("M6 truncation/malformed surfaced",
        pr.malformed_count == 1 and r.metrics.truncated and r.metrics.total_checkpoints == 1,
        f"malformed={pr.malformed_count} truncated={r.metrics.truncated}")

    # M7 spec: claim_select_ratio is 0 with no selects
    cps = extract_metrics("[CLAIM: a] [CLAIM: b]").checkpoints
    add("M7 claim_select_ratio = 0 with no selects", claim_select_ratio(cps) == 0.0,
        f"ratio={claim_select_ratio(cps)}")

    # M8 outcome scoring discriminates
    spec = {"type": "regex", "pattern": r"\b215\b", "scope": "conclude"}
    right, wrong = check_answer(spec, "215 minutes", ""), check_answer(spec, "205 minutes", "")
    none = check_answer(spec, "", "215 appears in prose only")
    add("M8 answer check discriminates", right is True and wrong is False and none is False,
        f"right={right} wrong={wrong} missing-conclusion={none}")
    return R


def run_live_controls(runner, *, n: int = 8, temperature: float = 1.0) -> List[ControlResult]:
    """Controls that need a model. n >= 8 recommended."""
    from .suite import Condition  # local import: keeps offline controls light
    R: List[ControlResult] = []
    base = Condition(key="baseline").config()

    def sample(prompt: str, label: str):
        outs = []
        for k in range(n):
            o = runner.run_full(prompt, "baseline", base, task_id=label, sample_index=k,
                                temperature=temperature)
            outs.append(extract_metrics(o.text, "baseline", truncated=o.truncated,
                                        reported_output_tokens=o.output_tokens))
        return outs

    # L1 noise floor on a structured task
    outs = sample("Recommend a consensus approach for a permissioned ledger run by 10 "
                  "organizations where up to f may be Byzantine.", "ctrl_noise")
    groups = [o.checkpoints for o in outs]
    floor = mean_pairwise_topology(groups, groups, same_group=True)
    counts = [o.metrics.total_checkpoints for o in outs]
    mean_c = sum(counts) / len(counts)
    sd_c = (sum((c - mean_c) ** 2 for c in counts) / max(1, len(counts) - 1)) ** 0.5
    ok = floor is not None and floor.sequence_similarity >= 0.5
    R.append(ControlResult(
        "L1 noise floor (same prompt, repeated)", "live", ok,
        f"n={n} mean pairwise sequence_similarity={floor.sequence_similarity:.2f} "
        f"(need >=0.50), checkpoints mean={mean_c:.1f} sd={sd_c:.1f}. "
        "Differences between conditions smaller than this are noise." if floor
        else "too few samples"))

    # L2 trivial question
    outs = sample("What is the capital of France?", "ctrl_trivial")
    mean_b = sum(o.metrics.branch_count for o in outs) / len(outs)
    paris = sum(bool(re.search("Paris", o.raw_text)) for o in outs) / len(outs)
    R.append(ControlResult("L2 trivial question -> trivial structure", "live",
                           mean_b <= 0.5 and paris >= 0.9,
                           f"mean branches={mean_b:.2f} (need <=0.50), answered Paris={paris:.0%} (need >=90%)"))

    # L3 nonsense prompt must not be answered with confidence
    outs = sample(_GIBBERISH, "ctrl_nonsense")
    from .outcomes import compute_outcomes
    ack = sum(bool(re.search(r"nonsense|gibberish|unclear|not sure|don'?t understand|"
                             r"cannot|can'?t|unable|meaningless|no meaning|not a (?:real|valid)",
                             o.raw_text, re.IGNORECASE)) for o in outs) / len(outs)
    cert = sum(compute_outcomes(o.raw_text, o.checkpoints).certainty_density for o in outs) / len(outs)
    R.append(ControlResult("L3 nonsense prompt -> acknowledged, not asserted", "live",
                           ack >= 0.5 and cert <= 1.0,
                           f"acknowledged nonsense={ack:.0%} (need >=50%), certainty/100w={cert:.2f} (need <=1.0)"))
    return R


def format_controls(results: List[ControlResult]) -> str:
    lines = ["CONTROLS", "=" * 70]
    for r in results:
        lines.append(f"  {'PASS' if r.passed else 'FAIL'}  [{r.kind}] {r.name}")
        lines.append(f"        {r.detail}")
    ok = all(r.passed for r in results)
    lines.append("=" * 70)
    lines.append("ALL CONTROLS PASSED" if ok else "CONTROLS FAILED: do not trust harness output")
    return "\n".join(lines)
