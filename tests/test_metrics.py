import math

import pytest

from hybrid_search.evaluation.metrics import (
    ndcg_at_k,
    overlap_at_k,
    recall_at_k,
    reciprocal_rank,
)

REL = {"a": 2, "b": 1, "c": 0}


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "b", "a"], REL) == 0.5
    assert reciprocal_rank(["x", "c"], REL) == 0.0  # grade 0 is not relevant
    assert reciprocal_rank(["x", "y", "a"], REL, k=2) == 0.0


def test_ndcg_perfect_and_hand_computed() -> None:
    assert ndcg_at_k(["a", "b"], REL) == pytest.approx(1.0)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    got = (1 / math.log2(2) + 3 / math.log2(3)) / ideal
    assert ndcg_at_k(["b", "a"], REL) == pytest.approx(got)
    assert ndcg_at_k(["x"], REL) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "x", "b"], REL, k=2) == 0.5
    assert recall_at_k(["a", "x", "b"], REL, k=3) == 1.0


def test_overlap_at_k() -> None:
    assert overlap_at_k([1, 2, 3], [3, 2, 9], k=3) == pytest.approx(2 / 3)
    assert overlap_at_k([], [], k=3) == 1.0
