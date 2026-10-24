"""
Epistemic Stress Harness — Local / Manual Runner

Replays pre-recorded annotated responses. No API calls.

Layout (v0.2), one file per sample:
    {source_dir}/{task_id}/{variant}/{sample_index}.txt
Legacy layout (single sample, no task): {source_dir}/{variant}.txt
A variant_config containing 'text' returns that text directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import Runner, RunOutput


class LocalRunner(Runner):
    def __init__(self, source_dir: str | None = None):
        self._source_dir = Path(source_dir) if source_dir else None

    def complete(self, system, user, *, max_tokens, temperature) -> RunOutput:
        raise NotImplementedError("LocalRunner replays files; it cannot complete prompts")

    def run_full(self, prompt, variant, variant_config: Dict[str, Any], *,
                 task_id: str = "", sample_index: int = 0,
                 temperature: float = 1.0) -> RunOutput:
        if "text" in variant_config:
            return RunOutput(text=variant_config["text"], model="local")
        if self._source_dir is None:
            raise ValueError("LocalRunner requires variant_config['text'] or a source_dir")

        candidates = []
        if task_id:
            candidates.append(self._source_dir / task_id / variant / f"{sample_index}.txt")
        candidates.append(self._source_dir / f"{variant}.txt")
        for path in candidates:
            if path.exists():
                return RunOutput(text=path.read_text(encoding="utf-8"), model="local")
        raise FileNotFoundError(
            "No annotated text found; tried: " + ", ".join(str(p) for p in candidates)
        )

    def name(self) -> str:
        return "local"
