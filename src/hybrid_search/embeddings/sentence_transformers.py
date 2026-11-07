"""Optional neural backend using ``sentence-transformers`` (``uv sync --extra st``).

The import is deferred until the backend is constructed, so the package works
without torch installed. The model is downloaded once by sentence-transformers
and cached in its usual location; nothing is written to the index directory
besides the model name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import EmbeddingBackend, l2_normalize


class SentenceTransformerEmbedder(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "the sentence-transformers backend needs the optional extra: "
                "`uv sync --extra st` (or `pip install 'hybrid-search[st]'`)"
            ) from e
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        self.dim = int(get_dim()) if get_dim else int(self._encode(["probe"]).shape[1])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vecs)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "batch_size": self.batch_size,
        }
