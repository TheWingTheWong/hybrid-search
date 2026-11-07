from pathlib import Path

import numpy as np
import pytest

from hybrid_search.engine import EngineConfig, SearchEngine


def ids(results) -> list[str]:
    return [r.chunk.doc_id for r in results]


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_modes_find_the_raft_note(tiny_engine: SearchEngine, mode: str) -> None:
    results = tiny_engine.search("how does raft elect a leader", 3, mode=mode)
    assert results and results[0].chunk.doc_id == "notes/raft"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_hybrid_reports_both_retrievers(tiny_engine: SearchEngine) -> None:
    top = tiny_engine.search("leader election", 1)[0]
    assert top.bm25_rank == 1 and top.dense_rank is not None
    assert top.snippet is not None and top.snippet.spans
    assert top.chunk.heading == "Leader election"


def test_weighted_fusion_and_alpha(tiny_engine: SearchEngine) -> None:
    lexical_only = tiny_engine.search("bloom filter", 3, fusion="weighted", alpha=0.0)
    assert ids(lexical_only)[0] == "bloom"
    with pytest.raises(ValueError):
        tiny_engine.search("x", 3, alpha=1.5)
    with pytest.raises(ValueError):
        tiny_engine.search("x", 3, mode="sparse")


def test_typo_query_is_rescued_by_dense(tiny_engine: SearchEngine) -> None:
    assert tiny_engine.search("blom filtr", 3, mode="bm25") == []
    assert ids(tiny_engine.search("blom filtr", 1))[0] == "bloom"


def test_filters_restrict_all_retrievers(tiny_engine: SearchEngine) -> None:
    for mode in ("hybrid", "bm25", "dense"):
        res = tiny_engine.search("leader quorum", 10, mode=mode, filters=["tags=consensus"])
        assert res and all("consensus" in r.chunk.metadata["tags"] for r in res)
    assert tiny_engine.search("raft", 5, filters=["year>=2030"]) == []
    res = tiny_engine.search("anything at all", 10, filters=["format=jsonl"], mode="dense")
    assert {r.chunk.doc_id for r in res} <= {"glossary:quorum", "glossary:lsm"}


def test_unique_docs(tiny_engine: SearchEngine) -> None:
    res = tiny_engine.search("raft leader log replication heartbeats", 10, unique_docs=True)
    assert len(ids(res)) == len(set(ids(res)))
    assert len(tiny_engine.search("raft leader log", 10, mode="bm25")) > len(
        tiny_engine.search("raft leader log", 10, mode="bm25", unique_docs=True)
    )


def test_stopword_only_query(tiny_engine: SearchEngine) -> None:
    assert tiny_engine.search("the of and", 5, mode="bm25") == []


def test_save_load_round_trip(tiny_engine: SearchEngine, tmp_path: Path) -> None:
    tiny_engine.save(tmp_path / "idx")
    loaded = SearchEngine.load(tmp_path / "idx")
    for q in ("leader election", "bloom filter false positives", "pager incident"):
        a = [(r.chunk.chunk_id, round(r.score, 6)) for r in tiny_engine.search(q, 5)]
        b = [(r.chunk.chunk_id, round(r.score, 6)) for r in loaded.search(q, 5)]
        assert a == b
    assert loaded.vectors.kind == "hnsw"
    assert not list((tmp_path / "idx").glob("*.pkl"))


def test_load_missing_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SearchEngine.load(tmp_path)


def test_exact_ann_config_and_bm25_scorer(tiny_corpus: Path) -> None:
    engine = SearchEngine.from_paths([tiny_corpus], EngineConfig(ann="exact", scorer="bm25"))
    assert engine.vectors.kind == "exact"
    assert ids(engine.search("raft", 1))[0] == "notes/raft"


def test_dense_exact_override_matches_hnsw_on_small_corpus(corpus_engine: SearchEngine) -> None:
    q = "split brain during a network partition"
    a = corpus_engine.search(q, 5, mode="dense", exact=True)
    b = corpus_engine.search(q, 5, mode="dense")
    assert ids(a) == ids(b)


def test_empty_corpus_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SearchEngine.from_paths([tmp_path])


def test_corpus_stats(corpus_engine: SearchEngine) -> None:
    s = corpus_engine.stats
    assert s["documents"] >= 40 and s["chunks"] > s["documents"]
    assert np.allclose(np.linalg.norm(corpus_engine.exact.vectors, axis=1), 1.0, atol=1e-4)
