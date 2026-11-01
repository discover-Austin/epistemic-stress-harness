"""
Epistemic Stress Harness — Runner Base Interface

A runner turns (system prompt, user prompt) into text plus provenance. It does
no analysis. Runners import nothing from each other.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

CHECKPOINT_INSTRUCTION = """Annotate your reasoning with epistemic checkpoints using these tags:

[ASSUME: your starting assumptions]
[CLAIM: intermediate assertions]
[BRANCH: alternative A vs alternative B]
[SELECT: chosen option | because: justification]
[CONCLUDE: final answer]

Rules:
- Use uppercase tags in square brackets
- Place checkpoints inline with your reasoning
- Order matters: the sequence records your epistemic trajectory
- Do not nest checkpoints
"""

TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length"})


@dataclass
class RunOutput:
    """One model response with provenance."""
    text: str
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.stop_reason in TRUNCATION_STOP_REASONS


def with_retry(fn: Callable[[], Any], attempts: int = 4, base_delay: float = 2.0) -> Any:
    """Call fn, retrying with exponential backoff. Re-raises the last error."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def build_system(variant_config: Dict[str, Any]) -> str:
    """The checkpoint instruction plus the condition's suffix, if any."""
    system = CHECKPOINT_INSTRUCTION
    if variant_config.get("system_suffix"):
        system += "\n" + variant_config["system_suffix"]
    return system


class Runner(ABC):
    """Base class for model runners."""

    @abstractmethod
    def complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> RunOutput:
        """One model call. Used by the harness and by the judge."""
        ...

    def run_full(
        self,
        prompt: str,
        variant: str,
        variant_config: Dict[str, Any],
        *,
        task_id: str = "",
        sample_index: int = 0,
        temperature: float = 1.0,
    ) -> RunOutput:
        """Run one sample for a condition. Offline runners override this."""
        return self.complete(
            build_system(variant_config),
            prompt,
            max_tokens=variant_config.get("max_tokens", 1024),
            temperature=temperature,
        )

    def run(self, prompt: str, variant: str, variant_config: Dict[str, Any]) -> str:
        """Compatibility wrapper: text only."""
        return self.run_full(prompt, variant, variant_config).text

    @abstractmethod
    def name(self) -> str:
        """Return the model/runner identifier."""
        ...
