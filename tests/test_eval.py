from hybrid_search.engine import SearchEngine
from hybrid_search.evaluation import (
    DEFAULT_RUNS,
    ann_recall,
    evaluate,
    load_queries,
    synthetic_ann_benchmark,
)

from .conftest import QUERIES


def test_query_labels_reference_real_documents(corpus_engine: SearchEngine) -> None:
    queries = load_queries(QUERIES)
    docs = {c.doc_id for c in corpus_engine.chunks}
    assert len(queries) >= 40
    assert all(set(q.relevant) <= docs for q in queries)
    assert len({q.qid for q in queries}) == len(queries)


def test_quality_regression_guard(corpus_engine: SearchEngine) -> None:
    """Pins the README numbers loosely; a retrieval regression should trip this."""
    report = evaluate(corpus_engine, load_queries(QUERIES))
    by_name = {r.spec.name: r.metrics for r in report.runs}
    assert set(by_name) == {r.name for r in DEFAULT_RUNS}
    assert by_name["Hybrid, RRF"]["nDCG@10"] >= 0.93
    assert by_name["Hybrid, RRF"]["nDCG@10"] >= by_name["BM25F"]["nDCG@10"]
    assert by_name["Hybrid, RRF"]["nDCG@10"] >= by_name["Dense, exact"]["nDCG@10"]
    assert by_name["Dense, HNSW"]["R@10"] >= by_name["Dense, exact"]["R@10"] - 0.02
    md = report.to_markdown()
    assert md.startswith("| Run |") and "Hybrid, RRF" in md
    assert "typo" in report.by_type_markdown()


def test_ann_recall_on_corpus(corpus_engine: SearchEngine) -> None:
    rows = ann_recall(corpus_engine, [q.query for q in load_queries(QUERIES)], efs=(64,))
    assert rows[0]["recall@10"] >= 0.98


def test_synthetic_benchmark_recall_increases_with_ef() -> None:
    bench = synthetic_ann_benchmark(n=800, dim=32, n_queries=40, efs=(10, 100), noise=1.0)
    low, high = (r["recall@10"] for r in bench["rows"])
    assert high >= low and high >= 0.95
