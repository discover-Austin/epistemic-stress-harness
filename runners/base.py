"""
Epistemic Stress Harness — Runner Base Interface

All model runners must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


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


class Runner(ABC):
    """Base class for model runners."""

    @abstractmethod
    def run(self, prompt: str, variant: str, variant_config: Dict[str, Any]) -> str:
        """
        Send a prompt to the model and return annotated text.

        Args:
            prompt: The base question or task.
            variant: Name of the perturbation variant being applied.
            variant_config: Perturbation-specific parameters.

        Returns:
            Model response containing checkpoint annotations.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the model/runner identifier."""
        ...
