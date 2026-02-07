"""
Epistemic Stress Harness — Local / Manual Runner

For pre-recorded responses or manual annotation.
No API calls. Reads annotated text from files or strings.
"""

from typing import Dict, Any
from pathlib import Path

from .base import Runner


class LocalRunner(Runner):
    """Runner that loads pre-annotated text."""

    def __init__(self, source_dir: str | None = None):
        self._source_dir = Path(source_dir) if source_dir else None

    def run(self, prompt: str, variant: str, variant_config: Dict[str, Any]) -> str:
        """
        Load annotated text from a file or return text directly.

        If variant_config contains 'text', returns it directly.
        Otherwise looks for {source_dir}/{variant}.txt.
        """
        if "text" in variant_config:
            return variant_config["text"]

        if self._source_dir is None:
            raise ValueError(
                "LocalRunner requires either variant_config['text'] or a source_dir"
            )

        path = self._source_dir / f"{variant}.txt"
        if not path.exists():
            raise FileNotFoundError(f"No annotated text found at {path}")

        return path.read_text()

    def name(self) -> str:
        return "local"
