"""
Epistemic Stress Harness — Model Runners

A runner returns text plus provenance (stop reason, token usage). Nothing else.
Runners do not import each other and do not import harness internals.
"""

from .base import Runner, RunOutput, CHECKPOINT_INSTRUCTION, build_system

__all__ = ["Runner", "RunOutput", "CHECKPOINT_INSTRUCTION", "build_system"]
