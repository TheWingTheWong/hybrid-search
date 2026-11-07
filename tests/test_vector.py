import numpy as np
import pytest

from hybrid_search.evaluation.metrics import overlap_at_k
from hybrid_search.vector import ExactIndex, HNSWIndex


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 32))
    pts = unit(centers[rng.integers(0, 20, 1500)] + 0.8 * rng.normal(size=(1500, 32)))
    qs = unit(centers[rng.integers(0, 20, 50)] + 0.8 * rng.normal(size=(50, 32)))
    return pts, qs


@pytest.fixture(scope="module")
def hnsw(data: tuple[np.ndarray, np.ndarray]) -> HNSWIndex:
    idx = HNSWIndex(32, M=12, ef_construction=80)
    idx.add(data[0])
    return idx


def test_exact_matches_brute_force(data: tuple[np.ndarray, np.ndarray]) -> None:
    pts, qs = data
    ex = ExactIndex(32)
    ex.add(pts)
    got = [i for i, _ in ex.search(qs[0], 5)]
    assert got == np.argsort(-(pts @ qs[0]))[:5].tolist()


def test_exact_mask_and_edge_cases(data: tuple[np.ndarray, np.ndarray]) -> None:
    pts, qs = data
    ex = ExactIndex(32)
    assert ex.search(qs[0], 3) == []
    ex.add(pts[:10])
    mask = np.zeros(10, dtype=bool)
    mask[[2, 7]] = True
    assert sorted(i for i, _ in ex.search(qs[0], 5, mask)) == [2, 7]


def test_hnsw_recall_against_exact(data: tuple[np.ndarray, np.ndarray], hnsw: HNSWIndex) -> None:
    pts, qs = data
    ex = ExactIndex(32)
    ex.add(pts)
    recall = np.mean(
        [overlap_at_k([i for i, _ in hnsw.search(q, 10)], [i for i, _ in ex.search(q, 10)]) for q in qs]
    )
    assert recall >= 0.95


def test_hnsw_graph_invariants(hnsw: HNSWIndex) -> None:
    assert len(hnsw) == 1500
    assert hnsw.levels[hnsw.entry_point] == hnsw.max_level
    for node, links in enumerate(hnsw.links):
        assert len(links) == hnsw.levels[node] + 1
        assert len(links[0]) <= hnsw.M0
        assert all(len(level) <= hnsw.M for level in links[1:])
        assert node not in links[0]
    # Level populations shrink geometrically.
    counts = np.bincount(hnsw.levels)
    assert counts[0] > counts[1] > 0


def test_hnsw_results_sorted(data: tuple[np.ndarray, np.ndarray], hnsw: HNSWIndex) -> None:
    res = hnsw.search(data[1][0], 10)
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True) and len(res) == 10


def test_hnsw_filtered_search(data: tuple[np.ndarray, np.ndarray], hnsw: HNSWIndex) -> None:
    mask = np.zeros(1500, dtype=bool)
    mask[::3] = True
    res = hnsw.search(data[1][0], 10, mask=mask)
    assert len(res) == 10 and all(mask[i] for i, _ in res)


def test_hnsw_round_trip(data: tuple[np.ndarray, np.ndarray], hnsw: HNSWIndex) -> None:
    clone = HNSWIndex.from_state(hnsw.config(), hnsw.arrays())
    for q in data[1][:10]:
        assert clone.search(q, 10) == hnsw.search(q, 10)


def test_hnsw_small_and_empty() -> None:
    idx = HNSWIndex(4)
    assert idx.search(np.ones(4, np.float32), 3) == []
    idx.add(unit(np.eye(4)))
    assert idx.search(unit(np.array([1.0, 0.1, 0, 0])), 1)[0][0] == 0
    assert len(idx.search(np.ones(4, np.float32) / 2, 10)) == 4
