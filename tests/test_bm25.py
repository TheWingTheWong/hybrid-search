import math

import numpy as np
import pytest

from hybrid_search.bm25 import BM25, BM25F, InvertedIndex, top_k


def build(docs: list[dict[str, str]]) -> InvertedIndex:
    idx = InvertedIndex()
    for d in docs:
        idx.add(d)
    return idx.freeze()


DOCS = [
    {"title": "Raft", "body": "leader election with randomized timeouts and leader heartbeats"},
    {"title": "Paxos", "body": "proposers acceptors learners agree on a single value"},
    {"title": "Bloom filter", "body": "bit array hash functions"},
    {"title": "Gossip", "body": "nodes exchange state with random peers " * 5},
]


def test_postings_and_df() -> None:
    idx = build(DOCS)
    docs, tfs = idx.postings("leader")
    assert docs.tolist() == [0]
    assert tfs[0].tolist() == [0, 0, 2]  # title, heading, body
    assert idx.df("leader") == 1 and idx.df("missing") == 0
    assert idx.n_docs == 4


def test_idf_matches_formula() -> None:
    idx = build(DOCS)
    assert idx.idf("leader") == pytest.approx(math.log(1 + (4 - 1 + 0.5) / (1 + 0.5)))


def test_bm25_matches_hand_computation() -> None:
    idx = build(DOCS)
    k1, b = 1.2, 0.75
    lengths = idx.field_lengths.sum(axis=1)
    avgdl = lengths.mean()
    tf, dl = 2.0, lengths[0]
    expected = idx.idf("leader") * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    scores = BM25(idx, k1=k1, b=b).score("leader")
    assert scores[0] == pytest.approx(expected, rel=1e-5)
    assert np.count_nonzero(scores) == 1


def test_query_term_frequency_counts() -> None:
    bm = BM25(build(DOCS))
    assert bm.score("leader leader")[0] == pytest.approx(2 * bm.score("leader")[0])


def test_length_normalization_prefers_short_docs() -> None:
    idx = build([{"body": "cache"}, {"body": "cache " + "filler " * 50}])
    scores = BM25(idx).score("cache")
    assert scores[0] > scores[1] > 0


def test_bm25f_title_weight_beats_body_mention() -> None:
    idx = build([
        {"title": "Notes", "body": "a short aside mentions compaction once among many other words"},
        {"title": "Compaction", "body": "merging sorted runs in the background"},
    ])  # fmt: skip
    assert BM25F(idx).search("compaction", 2)[0][0] == 1
    # With uniform weights and no field normalization difference, BM25F degrades gracefully.
    flat = BM25F(idx, weights={"title": 1, "heading": 1, "body": 1})
    assert len(flat.search("compaction", 2)) == 2


def test_search_respects_mask_and_k() -> None:
    bm = BM25(build(DOCS))
    assert bm.search("leader", 5, np.array([False, True, True, True])) == []
    hits = bm.search("leader hash random", 2)
    assert len(hits) == 2 and hits[0][1] >= hits[1][1]
    assert {d for d, _ in bm.search("leader hash random", 10)} == {0, 2, 3}


def test_top_k_orders_and_breaks_ties_by_id() -> None:
    scores = np.array([0.0, 2.0, 5.0, 2.0, 1.0])
    assert top_k(scores, 3) == [(2, 5.0), (1, 2.0), (3, 2.0)]
    assert top_k(np.zeros(3), 2) == []


def test_round_trip_arrays() -> None:
    idx = build(DOCS)
    terms, arrays = idx.to_arrays()
    clone = InvertedIndex.from_arrays(terms, arrays)
    q = "leader election hash"
    assert np.allclose(BM25F(idx).score(q), BM25F(clone).score(q))


def test_frozen_index_rejects_adds() -> None:
    idx = build(DOCS)
    with pytest.raises(RuntimeError):
        idx.add({"body": "late"})
