"""The embedding backend interface every dense encoder implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np


class EmbeddingBackend(ABC):
    """Maps text to L2-normalized ``float32`` vectors of a fixed dimension.

    Backends may be *fitted* on the corpus (the hashing backend learns IDF
    weights); pretrained backends simply ignore ``fit``. A backend must be
    reconstructible from ``config()`` + ``arrays()`` so an index can be saved
    and reloaded without the original corpus.
    """

    name: ClassVar[str]
    dim: int

    def fit(self, texts: Sequence[str]) -> EmbeddingBackend:  # noqa: B027 - optional hook
        return self

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents; returns an ``(n, dim)`` array of unit vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (backends may use a query-specific prompt)."""
        return self.embed([text])[0]

    def config(self) -> dict[str, Any]:
        return {}

    def arrays(self) -> dict[str, np.ndarray]:
        return {}

    @classmethod
    def from_state(
        cls, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
    ) -> EmbeddingBackend:
        return cls(**config)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)
