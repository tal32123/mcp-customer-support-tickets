"""Embedder Protocol + the SentenceTransformer-backed concrete impl.

The Embedder abstraction lets tests inject a deterministic stub while
production code loads the real `intfloat/multilingual-e5-small` model. The
Protocol contract guarantees L2-normalised output so downstream code (the
draft_reply grounding selector) can treat dot product as cosine similarity
without re-normalising defensively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Heavy import — only needed for typing. Keeping it behind TYPE_CHECKING
    # means importing this module does not transitively import torch.
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Concrete `Embedder` backed by `sentence-transformers`.

    Loads the model once at construction and reuses it for every call.
    `intfloat/multilingual-e5-small` is trained with task-specific prefixes:
    `"passage: "` for documents at index time, `"query: "` for queries at
    search time. We honour that split here so the protocol contract holds.
    """

    def __init__(self, model_name: str) -> None:
        # Local import: `sentence_transformers` pulls in torch (slow). Keeping
        # the import here means construction is the one place that pays the
        # cost — module import stays cheap.
        from sentence_transformers import SentenceTransformer

        self._model: SentenceTransformer = SentenceTransformer(model_name)
        # `get_sentence_embedding_dimension` is the supported API on
        # SentenceTransformer; we cache it so callers don't pay the lookup
        # on every store insert.
        self.dim: int = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, prefixed: list[str]) -> NDArray[np.float32]:
        vectors = self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed documents for the index. Adds the `"passage: "` E5 prefix."""
        return self._encode([f"passage: {t}" for t in texts])

    def embed_queries(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed search queries. Adds the `"query: "` E5 prefix."""
        return self._encode([f"query: {t}" for t in texts])
