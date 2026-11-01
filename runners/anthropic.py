"""
Epistemic Stress Harness — Anthropic Runner

Adapter for Claude models via the Anthropic Messages API. Requires
`pip install anthropic` and ANTHROPIC_API_KEY (or api_key=). The model id is
required: model names go stale, so there is no default.
"""

from __future__ import annotations

from .base import Runner, RunOutput, with_retry


class AnthropicRunner(Runner):
    def __init__(self, model: str, api_key: str | None = None):
        if not model:
            raise ValueError("AnthropicRunner requires an explicit model id")
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("anthropic package required: pip install anthropic") from e
            kwargs = {"api_key": self._api_key} if self._api_key else {}
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, system, user, *, max_tokens, temperature) -> RunOutput:
        client = self._get_client()
        msg = with_retry(lambda: client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ))
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        usage = getattr(msg, "usage", None)
        return RunOutput(
            text=text,
            stop_reason=getattr(msg, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model=self._model,
        )

    def name(self) -> str:
        return f"anthropic/{self._model}"
