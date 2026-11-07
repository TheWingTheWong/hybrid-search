"""Brute-force cosine index: the ground truth the ANN index is measured against."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class ExactIndex:
    kind = "exact"

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._vecs.shape[0]

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        self._vecs = np.vstack([self._vecs, vectors])

    def similarities(self, query: np.ndarray) -> np.ndarray:
        return self._vecs @ np.asarray(query, dtype=np.float32)

    def search(
        self, query: np.ndarray, k: int = 10, mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        sims = self.similarities(query)
        ids = np.arange(len(self)) if mask is None else np.flatnonzero(mask)
        if ids.size == 0 or k <= 0:
            return []
        sub = sims[ids]
        if ids.size > k:
            part = np.argpartition(-sub, k - 1)[:k]
            ids, sub = ids[part], sub[part]
        order = np.lexsort((ids, -sub))
        return [(int(ids[i]), float(sub[i])) for i in order]

    def config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def arrays(self) -> dict[str, np.ndarray]:
        return {"vectors": self._vecs}

    @classmethod
    def from_state(cls, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> ExactIndex:
        idx = cls(int(config["dim"]))
        idx._vecs = np.asarray(arrays["vectors"], dtype=np.float32)
        return idx
