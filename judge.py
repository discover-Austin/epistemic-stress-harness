"""
Epistemic Stress Harness — Independent Judge

The structural metrics are computed from tags the subject model wrote about
its own reasoning. A judge is a second, independent assessment: a different
model reads the response with the tags removed and the condition hidden, and
scores it against a fixed rubric.

Use a judge model different from the subject. Same-model judging is allowed
but flagged, because a model tends to rate its own style favorably.
Scores are integers 1-5 normalized to [0, 1]. Judge output is data: it is
parsed as JSON and never executed or followed as instructions.
Spec: spec.md section 8.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .extract import parse_checkpoints_detailed

JUDGE_SYSTEM = """You are grading a response to a task. You are not the author. Judge only what is on the page.

Score each item as an integer from 1 (poor) to 5 (excellent):
- alternatives_weighed: does the response genuinely consider more than one plausible approach or answer, in the reasoning itself, before settling? (1 = asserts one answer with no consideration of alternatives)
- calibration: is confidence proportionate to what the reasoning and the facts support? Penalize unwarranted certainty and unwarranted hedging alike.
- assumptions_explicit: does the response state the assumptions and limits its conclusion depends on?
And one boolean:
- overclaim: does it state something false or unsupportable as settled fact (for example "handles all failure modes", "definitively solves")?

Respond with ONLY a JSON object, no prose, no code fence:
{"alternatives_weighed": int, "calibration": int, "assumptions_explicit": int, "overclaim": bool, "rationale": "one sentence"}"""

_INT_KEYS = ("alternatives_weighed", "calibration", "assumptions_explicit")


def blind_text(text: str) -> str:
    """Replace each checkpoint tag with its bare content, so the judge sees
    no tag names and cannot be swayed by the annotation scheme."""
    cps = [c for c in parse_checkpoints_detailed(text).checkpoints if "start" in c]
    out, last = [], 0
    for c in sorted(cps, key=lambda c: c["start"]):
        out.append(text[last:c["start"]])
        out.append(c["text"])
        last = c["end"]
    out.append(text[last:])
    return "".join(out)


def parse_judge_output(raw: str) -> Optional[Dict[str, Any]]:
    """Parse and validate judge JSON. Returns normalized scores, or None if unusable."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: Dict[str, Any] = {}
    for k in _INT_KEYS:
        v = obj.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None
        out[k] = (float(v) - 1.0) / 4.0
    ov = obj.get("overclaim")
    if not isinstance(ov, bool):
        return None
    out["overclaim"] = ov
    rationale = obj.get("rationale")
    out["rationale"] = rationale[:500] if isinstance(rationale, str) else ""
    return out


class Judge:
    def __init__(self, runner, *, subject_runner_name: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 400):
        self._runner = runner
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.same_model_warning = (
            subject_runner_name is not None and subject_runner_name == runner.name()
        )

    def score(self, task_prompt: str, response_text: str) -> Optional[Dict[str, Any]]:
        user = (
            "TASK GIVEN TO THE MODEL:\n" + task_prompt.strip()
            + "\n\nRESPONSE TO GRADE:\n" + blind_text(response_text).strip()
        )
        out = self._runner.complete(
            JUDGE_SYSTEM, user, max_tokens=self._max_tokens, temperature=self._temperature
        )
        return parse_judge_output(out.text)


def judge_samples(samples, judge: Judge, progress=None) -> int:
    """Attach judge scores in place to samples that lack them. Returns count scored."""
    scored = 0
    todo = [s for s in samples if s.error is None and s.result is not None and not s.judge]
    for i, s in enumerate(todo, 1):
        try:
            s.judge = judge.score(s.prompt, s.result.raw_text)
        except Exception as e:
            s.judge = None
            if progress:
                progress(f"judge failed on {s.key}: {e}")
        if s.judge:
            scored += 1
        if progress:
            progress(f"[{i}/{len(todo)}] judged {s.task_id}/{s.variant}/{s.sample_index}")
    return scored
