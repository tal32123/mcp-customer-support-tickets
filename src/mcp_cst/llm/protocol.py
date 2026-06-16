"""Provider-agnostic completion interface."""

from __future__ import annotations
from typing import Protocol


class LlmClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...
