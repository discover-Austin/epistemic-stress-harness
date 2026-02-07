"""
Epistemic Stress Harness — Anthropic Runner

Adapter for Claude models via the Anthropic API.
"""

from typing import Dict, Any

from .base import Runner, CHECKPOINT_INSTRUCTION


class AnthropicRunner(Runner):
    """Runner for Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def run(self, prompt: str, variant: str, variant_config: Dict[str, Any]) -> str:
        client = self._get_client()

        system = CHECKPOINT_INSTRUCTION
        if variant_config.get("system_suffix"):
            system += "\n\n" + variant_config["system_suffix"]

        max_tokens = variant_config.get("max_tokens", 1024)

        message = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def name(self) -> str:
        return f"anthropic/{self._model}"
