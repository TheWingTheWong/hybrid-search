"""Evaluation harness: run labeled queries through several retrieval configs.

Relevance is judged at the *document* level, so each run asks the engine for
one hit per document (``unique_docs=True``) and scores the resulting doc-id
ranking. This keeps chunking choices from inflating or deflating the numbers.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..engine import SearchEngine
from ..vector import ExactIndex, HNSWIndex
from .metrics import ndcg_at_k, overlap_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class LabeledQuery:
    qid: str
    query: str
    relevant: dict[str, int]
    type: str = "other"
    filters: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSpec:
    name: str
    mode: str
    fusion: str = "rrf"
    alpha: float = 0.5
    scorer: str | None = None
    exact: bool = False


DEFAULT_RUNS: tuple[RunSpec, ...] = (
    RunSpec("BM25", "bm25", scorer="bm25"),
    RunSpec("BM25F", "bm25", scorer="bm25f"),
    RunSpec("Dense, exact", "dense", exact=True),
    RunSpec("Dense, HNSW", "dense"),
    RunSpec("Hybrid, RRF", "hybrid", "rrf", scorer="bm25f"),
    RunSpec("Hybrid, weighted", "hybrid", "weighted", scorer="bm25f"),
)


@dataclass
class RunResult:
    spec: RunSpec
    metrics: dict[str, float]
    by_type: dict[str, dict[str, float]]
    per_query: list[dict[str, Any]]
    latency_ms: dict[str, float]


@dataclass
class EvalReport:
    runs: list[RunResult]
    n_queries: int
    k: int
    recall_ks: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_names(self) -> list[str]:
        return [f"MRR@{self.k}", f"nDCG@{self.k}"] + [f"R@{k}" for k in self.recall_ks]

    def to_markdown(self) -> str:
        cols = self.metric_names + ["p50 ms"]
        lines = [
            "| Run | " + " | ".join(cols) + " |",
            "|---|" + "---:|" * len(cols),
        ]
        best = {m: max(r.metrics[m] for r in self.runs) for m in self.metric_names}
        for r in self.runs:
            cells = []
            for m in self.metric_names:
                v = f"{r.metrics[m]:.3f}"
                cells.append(f"**{v}**" if abs(r.metrics[m] - best[m]) < 1e-9 else v)
            cells.append(f"{r.latency_ms['p50']:.2f}")
            lines.append(f"| {r.spec.name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def by_type_markdown(self, metric: str | None = None) -> str:
        metric = metric or f"nDCG@{self.k}"
        types = sorted({t for r in self.runs for t in r.by_type})
        counts = self.meta.get("type_counts", {})
        head = [f"{t} ({counts[t]})" if t in counts else t for t in types]
        lines = [
            f"| {metric} by query type | " + " | ".join(head) + " |",
            "|---|" + "---:|" * len(types),
        ]
        for r in self.runs:
            vals = [f"{r.by_type.get(t, {}).get(metric, 0.0):.3f}" for t in types]
            lines.append(f"| {r.spec.name} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "meta": self.meta,
            "runs": [
                {
                    "name": r.spec.name,
                    "spec": r.spec.__dict__,
                    "metrics": r.metrics,
                    "by_type": r.by_type,
                    "latency_ms": r.latency_ms,
                    "per_query": r.per_query,
                }
                for r in self.runs
            ],
        }


def load_queries(path: str | Path) -> list[LabeledQuery]:
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("relevant"):
                raise ValueError(f"{path}:{lineno}: query has no relevance labels")
            out.append(
                LabeledQuery(
                    qid=str(rec.get("id", lineno)),
                    query=rec["query"],
                    relevant={str(k): int(v) for k, v in rec["relevant"].items()},
                    type=rec.get("type", "other"),
                    filters=tuple(rec.get("filters", ())),
                )
            )
    return out


def evaluate(
    engine: SearchEngine,
    queries: Sequence[LabeledQuery],
    runs: Iterable[RunSpec] = DEFAULT_RUNS,
    k: int = 10,
    recall_ks: tuple[int, ...] = (5, 10),
) -> EvalReport:
    depth = max(k, *recall_ks)
    known = {c.doc_id for c in engine.chunks}
    unknown = {d for q in queries for d in q.relevant} - known
    if unknown:
        raise ValueError(f"relevance labels reference unknown documents: {sorted(unknown)}")

    results = []
    for spec in runs:
        per_query, latencies = [], []
        for q in queries:
            t0 = time.perf_counter()
            hits = engine.search(
                q.query, depth, mode=spec.mode, fusion=spec.fusion, alpha=spec.alpha,
                filters=q.filters, unique_docs=True, snippets=False, scorer=spec.scorer,
                exact=spec.exact,
            )  # fmt: skip
            latencies.append((time.perf_counter() - t0) * 1000)
            ranked = [h.chunk.doc_id for h in hits]
            row: dict[str, Any] = {
                "id": q.qid,
                "query": q.query,
                "type": q.type,
                f"MRR@{k}": reciprocal_rank(ranked, q.relevant, k),
                f"nDCG@{k}": ndcg_at_k(ranked, q.relevant, k),
            }
            for rk in recall_ks:
                row[f"R@{rk}"] = recall_at_k(ranked, q.relevant, rk)
            row["top3"] = ranked[:3]
            per_query.append(row)

        names = [f"MRR@{k}", f"nDCG@{k}"] + [f"R@{rk}" for rk in recall_ks]
        metrics = {m: float(np.mean([r[m] for r in per_query])) for m in names}
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_query:
            groups[r["type"]].append(r)
        by_type = {
            t: {m: float(np.mean([r[m] for r in rows])) for m in names}
            for t, rows in sorted(groups.items())
        }
        lat = sorted(latencies)
        latency = {
            "p50": statistics.median(lat),
            "p95": lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))],
        }
        results.append(RunResult(spec, metrics, by_type, per_query, latency))

    type_counts: dict[str, int] = defaultdict(int)
    for q in queries:
        type_counts[q.type] += 1
    return EvalReport(results, len(queries), k, recall_ks, {"type_counts": dict(type_counts)})


# ---------------------------------------------------------------------------
# ANN quality: HNSW vs exact
# ---------------------------------------------------------------------------


def _recall_sweep(
    hnsw: HNSWIndex,
    exact: ExactIndex,
    queries: np.ndarray,
    k: int,
    efs: Sequence[int],
) -> list[dict[str, float]]:
    truth, exact_ms = [], []
    for q in queries:
        t0 = time.perf_counter()
        truth.append([i for i, _ in exact.search(q, k)])
        exact_ms.append((time.perf_counter() - t0) * 1000)
    rows = []
    for ef in efs:
        recalls, ms = [], []
        for q, t in zip(queries, truth, strict=True):
            t0 = time.perf_counter()
            approx = [i for i, _ in hnsw.search(q, k, ef=ef)]
            ms.append((time.perf_counter() - t0) * 1000)
            recalls.append(overlap_at_k(approx, t, k))
        rows.append(
            {
                "ef": ef,
                f"recall@{k}": float(np.mean(recalls)),
                "hnsw_ms": statistics.median(ms),
                "exact_ms": statistics.median(exact_ms),
            }
        )
    return rows


def ann_recall(
    engine: SearchEngine,
    queries: Sequence[str],
    k: int = 10,
    efs: Sequence[int] = (10, 16, 32, 64, 128),
) -> list[dict[str, float]]:
    """HNSW recall@k against exact search, using real query embeddings over the corpus."""
    hnsw = engine.vectors
    if not isinstance(hnsw, HNSWIndex):
        hnsw = HNSWIndex(engine.embedder.dim)
        hnsw.add(engine.exact.vectors)
    qv = np.stack([engine.embedder.embed_query(q) for q in queries])
    return _recall_sweep(hnsw, engine.exact, qv, min(k, len(hnsw)), efs)


def synthetic_ann_benchmark(
    n: int = 5000,
    dim: int = 128,
    n_queries: int = 200,
    k: int = 10,
    efs: Sequence[int] = (10, 16, 32, 64, 128, 256),
    clusters: int = 100,
    noise: float = 2.0,
    M: int = 16,
    ef_construction: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """HNSW vs exact on a Gaussian-mixture point cloud projected onto the unit sphere.

    The defaults (128-d, noise twice the center scale) make the clusters
    overlap heavily, which is a hard regime for graph search and shows a real
    recall/latency curve across ``ef``; well-separated clusters saturate at
    recall 1.0 almost immediately and tell you nothing.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))

    def sample(m: int) -> np.ndarray:
        x = centers[rng.integers(0, clusters, m)] + noise * rng.normal(size=(m, dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    data, queries = sample(n), sample(n_queries)
    exact = ExactIndex(dim)
    exact.add(data)
    hnsw = HNSWIndex(dim, M=M, ef_construction=ef_construction)
    t0 = time.perf_counter()
    hnsw.add(data)
    build_s = time.perf_counter() - t0
    return {
        "n": n,
        "dim": dim,
        "M": M,
        "ef_construction": ef_construction,
        "build_s": build_s,
        "rows": _recall_sweep(hnsw, exact, queries, k, efs),
    }
