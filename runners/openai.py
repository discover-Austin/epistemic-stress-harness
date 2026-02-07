"""
Epistemic Stress Harness — OpenAI Runner

Adapter for OpenAI models via the OpenAI API.
"""

from typing import Dict, Any

from .base import Runner, CHECKPOINT_INSTRUCTION


class OpenAIRunner(Runner):
    """Runner for OpenAI models."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def run(self, prompt: str, variant: str, variant_config: Dict[str, Any]) -> str:
        client = self._get_client()

        system = CHECKPOINT_INSTRUCTION
        if variant_config.get("system_suffix"):
            system += "\n\n" + variant_config["system_suffix"]

        max_tokens = variant_config.get("max_tokens", 1024)

        response = client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content

    def name(self) -> str:
        return f"openai/{self._model}"
