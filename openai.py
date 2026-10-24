"""
Epistemic Stress Harness — OpenAI Runner

Adapter for OpenAI chat models. Requires `pip install openai` and
OPENAI_API_KEY (or api_key=). The model id is required.
"""

from __future__ import annotations

from .base import Runner, RunOutput, with_retry


class OpenAIRunner(Runner):
    def __init__(self, model: str, api_key: str | None = None):
        if not model:
            raise ValueError("OpenAIRunner requires an explicit model id")
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e
            kwargs = {"api_key": self._api_key} if self._api_key else {}
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, system, user, *, max_tokens, temperature) -> RunOutput:
        client = self._get_client()
        resp = with_retry(lambda: client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ))
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return RunOutput(
            text=choice.message.content or "",
            stop_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            model=self._model,
        )

    def name(self) -> str:
        return f"openai/{self._model}"
