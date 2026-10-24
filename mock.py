"""
Epistemic Stress Harness — Mock Runner (TEST DOUBLE)

Generates synthetic annotated text from a seeded random process. It is NOT a
model and its output means nothing scientifically. It exists to exercise the
pipeline and to test the analysis: with known, configurable effects it lets us
check that the analysis finds real effects and does not invent null ones.

Default effects: 'confidence' and 'commitment_pressure' lower the branch
probability; 'token_constraint_*' shorten output; 'self_at_risk' lowers
correctness. Everything else behaves like baseline, including 'stakes_neutral'
(a true null).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Runner, RunOutput

DEFAULT_EFFECTS: Dict[str, Dict[str, float]] = {
    "baseline":             {"p_branch": 0.80, "p_correct": 0.90, "certain": 0.05, "verbose": 1.0},
    "commitment_pressure":  {"p_branch": 0.25, "p_correct": 0.90, "certain": 0.30, "verbose": 0.8},
    "confidence":           {"p_branch": 0.35, "p_correct": 0.80, "certain": 0.90, "verbose": 0.9},
    "token_constraint_200": {"p_branch": 0.75, "p_correct": 0.88, "certain": 0.05, "verbose": 0.4},
    "token_constraint_100": {"p_branch": 0.70, "p_correct": 0.85, "certain": 0.05, "verbose": 0.2},
    "self_at_risk":         {"p_branch": 0.80, "p_correct": 0.55, "certain": 0.05, "verbose": 1.0},
}


class MockRunner(Runner):
    def __init__(self, seed: int = 0, effects: Optional[Dict[str, Dict[str, float]]] = None,
                 tasks_path: Optional[str] = None, scenario_answers: Optional[Dict[str, Dict[str, str]]] = None):
        self._seed = seed
        self._effects = {**DEFAULT_EFFECTS, **(effects or {})}
        self._answers: Dict[str, Dict[str, str]] = dict(scenario_answers or {})
        if tasks_path is None:
            tasks_path = str(Path(__file__).resolve().parent.parent / "suite" / "tasks_v1.json")
        if Path(tasks_path).exists():
            for t in json.load(open(tasks_path, encoding="utf-8"))["tasks"]:
                if "_mock" in t:
                    self._answers[t["id"]] = t["_mock"]

    def register_answers(self, answers: Dict[str, Dict[str, str]]) -> None:
        self._answers.update(answers)

    def complete(self, system, user, *, max_tokens, temperature) -> RunOutput:
        rng = random.Random(self._hash(system, user))
        return RunOutput(text=self._render(rng, "baseline", {"correct": "ok", "wrong": "no"}, max_tokens),
                         stop_reason="end_turn", model="mock")

    def run_full(self, prompt, variant, variant_config: Dict[str, Any], *,
                 task_id: str = "", sample_index: int = 0,
                 temperature: float = 1.0) -> RunOutput:
        rng = random.Random(self._hash(task_id, variant, sample_index))
        answers = self._answers.get(task_id, {"correct": "ok", "wrong": "no"})
        max_tokens = variant_config.get("max_tokens", 1024)
        text = self._render(rng, variant, answers, max_tokens)
        approx = int(len(text.split()) * 1.3)
        truncated = approx > max_tokens
        return RunOutput(text=text, stop_reason="max_tokens" if truncated else "end_turn",
                         output_tokens=min(approx, max_tokens), model="mock")

    def name(self) -> str:
        return f"mock/seed{self._seed}"

    def _hash(self, *parts) -> int:
        h = hashlib.sha256(("|".join(map(str, parts)) + f"|{self._seed}").encode()).hexdigest()
        return int(h[:16], 16)

    def _render(self, rng: random.Random, variant: str, answers: Dict[str, str], max_tokens: int) -> str:
        eff = self._effects.get(variant, self._effects["baseline"])
        pad = " ".join(["detail"] * max(1, int(rng.randint(8, 24) * eff["verbose"])))
        parts = [f"[ASSUME: standard conditions hold and the inputs are as stated]\n{pad}"]
        parts.append(f"[CLAIM: the relevant rule applies here]\n{pad}")
        if rng.random() < eff["p_branch"]:
            parts.append(f"[BRANCH: approach one vs approach two]\n{pad}")
            parts.append(f"[SELECT: approach one | because: it fits the stated conditions best]")
        else:
            parts.append(f"[SELECT: approach one | because: it is the standard choice here]")
        if rng.random() < eff["p_branch"] * 0.5:
            parts.append(f"[BRANCH: fast path vs slow path]\n{pad}")
            parts.append(f"[SELECT: fast path | because: lower cost with a safe fallback]")
        parts.append(f"[CLAIM: this follows from the earlier steps]\n{pad}")
        if rng.random() < eff["certain"]:
            parts.append("This is definitively the optimal choice and it handles all failure modes.")
        else:
            parts.append("This may depend on the stated assumptions, however.")
        correct = rng.random() < eff["p_correct"]
        parts.append(f"[CONCLUDE: {answers['correct'] if correct else answers['wrong']}]")
        return "\n\n".join(parts)
