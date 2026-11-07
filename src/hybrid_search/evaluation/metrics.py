"""Ranking metrics over document-level rankings with graded relevance.

``relevance`` maps doc ids to grades (0 = not relevant, higher = better);
unlisted documents are non-relevant. Documents with grade >= 1 count as
relevant for MRR and recall.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def reciprocal_rank(ranked: Sequence[str], relevance: Mapping[str, int], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if relevance.get(doc, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, int], k: int = 10) -> float:
    """nDCG with exponential gain ``2^rel - 1`` (Burges et al., 2005)."""
    gains = [2 ** relevance.get(doc, 0) - 1 for doc in ranked[:k]]
    ideal = sorted((2**g - 1 for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevance: Mapping[str, int], k: int = 10) -> float:
    relevant = {d for d, g in relevance.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def overlap_at_k(approx: Sequence[int], exact: Sequence[int], k: int = 10) -> float:
    """ANN recall@k: fraction of the exact top-k that the approximate search found."""
    truth = set(exact[:k])
    return len(truth.intersection(approx[:k])) / len(truth) if truth else 1.0
