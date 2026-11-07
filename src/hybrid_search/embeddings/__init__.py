"""Embedding backends and a small registry to construct them by name."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import EmbeddingBackend, l2_normalize
from .hashing import HashingEmbedder
from .sentence_transformers import SentenceTransformerEmbedder

BACKENDS: dict[str, type[EmbeddingBackend]] = {
    HashingEmbedder.name: HashingEmbedder,
    SentenceTransformerEmbedder.name: SentenceTransformerEmbedder,
}

_ALIASES = {"st": SentenceTransformerEmbedder.name, "hash": HashingEmbedder.name}


def create_backend(spec: str = "hashing", **kwargs: Any) -> EmbeddingBackend:
    """Build a backend from a spec such as ``"hashing"`` or ``"st:all-MiniLM-L6-v2"``."""
    name, _, arg = spec.partition(":")
    name = _ALIASES.get(name, name)
    if name not in BACKENDS:
        raise ValueError(f"unknown embedding backend {spec!r}; choose from {sorted(BACKENDS)}")
    if arg:
        if name != SentenceTransformerEmbedder.name:
            raise ValueError(f"backend {name!r} takes no model argument")
        kwargs.setdefault("model_name", arg)
    return BACKENDS[name](**kwargs)


def load_backend(
    name: str, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> EmbeddingBackend:
    if name not in BACKENDS:
        raise ValueError(f"index was built with unknown backend {name!r}")
    return BACKENDS[name].from_state(config, arrays)


__all__ = [
    "BACKENDS",
    "EmbeddingBackend",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "create_backend",
    "l2_normalize",
    "load_backend",
]
