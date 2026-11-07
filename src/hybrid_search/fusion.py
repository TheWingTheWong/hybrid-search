"""Rank fusion: reciprocal rank fusion and normalized weighted score blending.

RRF (Cormack, Clarke & Buettcher, SIGIR 2009) only looks at ranks, so it needs
no calibration between BM25 scores (unbounded, query-dependent) and cosine
similarities (bounded, compressed). Weighted blending uses the raw scores and
therefore has to normalize each list first; it is sharper when one retriever is
clearly better, but more sensitive to the score distribution of each query.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[K]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse ranked id lists: ``score(d) = sum_r w_r / (k + rank_r(d))`` (rank from 1)."""
    if k < 0:
        raise ValueError("RRF constant k must be non-negative")
    scores: dict[K, float] = {}
    for name, ranking in rankings.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return _sorted(scores)


def normalize_scores(scores: Mapping[K, float], method: str = "minmax") -> dict[K, float]:
    """Map one result list's scores onto a comparable scale."""
    if not scores:
        return {}
    keys = list(scores)
    vals = np.fromiter((scores[x] for x in keys), dtype=np.float64, count=len(keys))
    if method == "minmax":
        lo, hi = vals.min(), vals.max()
        norm = np.ones_like(vals) if hi - lo < 1e-12 else (vals - lo) / (hi - lo)
    elif method == "zscore":
        sd = vals.std()
        norm = np.zeros_like(vals) if sd < 1e-12 else (vals - vals.mean()) / sd
    elif method == "none":
        norm = vals
    else:
        raise ValueError(f"unknown normalization {method!r}")
    return dict(zip(keys, norm.tolist(), strict=True))


def weighted_fusion(
    results: Mapping[str, Sequence[tuple[K, float]]],
    weights: Mapping[str, float],
    normalization: str = "minmax",
) -> list[tuple[K, float]]:
    """Blend normalized scores: ``score(d) = sum_r w_r * norm_r(d)``.

    A document missing from a list receives that list's minimum normalized
    score (0 for min-max), i.e. it is treated as the weakest observed match.
    """
    normed = {name: normalize_scores(dict(res), normalization) for name, res in results.items()}
    floor = {name: min(n.values(), default=0.0) for name, n in normed.items()}
    docs: dict[K, None] = {}
    for n in normed.values():
        docs.update(dict.fromkeys(n))
    fused = {
        d: sum(float(weights.get(name, 0.0)) * n.get(d, floor[name]) for name, n in normed.items())
        for d in docs
    }
    return _sorted(fused)


def _sorted(scores: Mapping[K, float]) -> list[tuple[K, float]]:
    # Stable, deterministic order: score desc, then first-seen order.
    return sorted(scores.items(), key=lambda kv: -kv[1])
