"""Anthropic SDK adapter."""

from __future__ import annotations

from ..errors import ErrorCode, McpCstError


def _make_sdk_client():
    import anthropic
    return anthropic.Anthropic()


class AnthropicClient:
    def __init__(self, *, model: str, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = _make_sdk_client()

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not resp.content:
            raise McpCstError(
                ErrorCode.LLM_RESPONSE_INVALID,
                f"anthropic returned empty content (stop_reason={getattr(resp, 'stop_reason', None)!r})",
            )
        block = resp.content[0]
        text = getattr(block, "text", None)
        if not text:
            raise McpCstError(
                ErrorCode.LLM_RESPONSE_INVALID,
                f"anthropic content[0] is not a text block (type={getattr(block, 'type', None)!r})",
            )
        return text
