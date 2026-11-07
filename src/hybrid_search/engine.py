"""The hybrid search engine: ties chunking, BM25F, dense retrieval and fusion together.

Query flow::

    query --+--> analyzer --> BM25/BM25F over inverted index --> top-N lexical --+
            |                                                                    +--> RRF or
            +--> embedder --> HNSW (or exact) over chunk vectors --> top-N dense -+    weighted blend
                                                                                       |
                       metadata filters --> allow-mask applied inside both retrievers  |
                                                                                       v
                                                             (optional) one hit per document
                                                                                       |
                                                                         snippet + highlights

An index directory contains ``manifest.json``, ``chunks.jsonl`` and three
``.npz`` files (lexical postings, embedder state, vector index). Nothing is
pickled.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bm25 import BM25, BM25F, InvertedIndex
from .documents import Chunk, Document
from .embeddings import EmbeddingBackend, create_backend, load_backend
from .filters import Filter, filter_mask, parse_filters
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .highlight import Snippet, make_snippet
from .ingest import ChunkingConfig, chunk_documents, load_documents
from .vector import ExactIndex, HNSWIndex

FORMAT_VERSION = 1
MODES = ("hybrid", "bm25", "dense")
FUSIONS = ("rrf", "weighted")

# Below this many allowed vectors, a filtered dense query scans the subset
# exactly instead of walking the HNSW graph (which degrades under selective filters).
FILTERED_EXACT_THRESHOLD = 2_000


@dataclass
class EngineConfig:
    backend: str = "hashing"
    ann: str = "hnsw"  # "hnsw" | "exact"
    scorer: str = "bm25f"  # "bm25f" | "bm25"
    k1: float = 1.2
    b: float = 0.75
    field_weights: dict[str, float] = field(
        default_factory=lambda: {"title": 3.0, "heading": 2.0, "body": 1.0}
    )
    hnsw_m: int = 16
    ef_construction: int = 100
    ef_search: int = 64
    max_tokens: int = 160
    overlap: int = 32

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(self.max_tokens, self.overlap)


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: Chunk
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    snippet: Snippet | None = None

    def to_dict(self) -> dict[str, Any]:
        c = self.chunk
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "heading": c.heading,
            "source": c.source,
            "metadata": c.metadata,
            "bm25": None
            if self.bm25_rank is None
            else {"rank": self.bm25_rank, "score": round(self.bm25_score or 0.0, 6)},
            "dense": None
            if self.dense_rank is None
            else {"rank": self.dense_rank, "score": round(self.dense_score or 0.0, 6)},
            "snippet": self.snippet.to_dict() if self.snippet else None,
        }


class SearchEngine:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        lexical: InvertedIndex,
        embedder: EmbeddingBackend,
        vectors: ExactIndex | HNSWIndex,
        config: EngineConfig,
        stats: dict[str, Any] | None = None,
    ):
        self.chunks = list(chunks)
        self.lexical = lexical
        self.embedder = embedder
        self.vectors = vectors
        self.config = config
        self.stats = stats or {}
        cls = BM25F if config.scorer == "bm25f" else BM25
        kwargs: dict[str, Any] = {"k1": config.k1, "b": config.b}
        if cls is BM25F:
            kwargs["weights"] = config.field_weights
        self.scorer = cls(lexical, **kwargs)
        # Brute-force view over the same vectors: ground truth + filtered fallback.
        self.exact = vectors if isinstance(vectors, ExactIndex) else _exact_view(vectors)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_paths(
        cls, paths: Iterable[str | Path], config: EngineConfig | None = None, **kw: Any
    ) -> SearchEngine:
        return cls.from_documents(load_documents(paths), config, **kw)

    @classmethod
    def from_documents(
        cls, docs: Sequence[Document], config: EngineConfig | None = None, **kw: Any
    ) -> SearchEngine:
        config = config or EngineConfig()
        engine = cls.from_chunks(chunk_documents(docs, config.chunking), config, **kw)
        engine.stats["documents"] = len(docs)
        return engine

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        config: EngineConfig | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> SearchEngine:
        config = config or EngineConfig()
        if not chunks:
            raise ValueError("nothing to index: no chunks were produced")
        timings: dict[str, float] = {}

        t = time.perf_counter()
        lexical = InvertedIndex()
        for c in chunks:
            lexical.add(c.fields())
        lexical.freeze()
        timings["lexical_s"] = time.perf_counter() - t

        t = time.perf_counter()
        embedder = embedder or create_backend(config.backend)
        texts = [c.embedding_text() for c in chunks]
        embedder.fit(texts)
        matrix = embedder.embed(texts)
        timings["embed_s"] = time.perf_counter() - t

        t = time.perf_counter()
        vectors: ExactIndex | HNSWIndex
        if config.ann == "hnsw":
            vectors = HNSWIndex(
                embedder.dim, M=config.hnsw_m, ef_construction=config.ef_construction,
                ef_search=config.ef_search,
            )  # fmt: skip
        elif config.ann == "exact":
            vectors = ExactIndex(embedder.dim)
        else:
            raise ValueError(f"unknown ann index {config.ann!r}")
        vectors.add(matrix)
        timings["vector_index_s"] = time.perf_counter() - t

        stats = {
            "documents": len({c.doc_id for c in chunks}),
            "chunks": len(chunks),
            "terms": lexical.n_terms,
            "dim": embedder.dim,
            "timings": {k: round(v, 4) for k, v in timings.items()},
        }
        return cls(chunks, lexical, embedder, vectors, config, stats)

    # -- retrieval --------------------------------------------------------
    def lexical_search(
        self, query: str, k: int, mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        return self.scorer.search(query, k, mask)

    def dense_search(
        self, query: str, k: int, mask: np.ndarray | None = None, exact: bool = False
    ) -> list[tuple[int, float]]:
        q = self.embedder.embed_query(query)
        if not np.any(q):
            return []  # nothing hashable in the query
        if exact or mask is not None and int(mask.sum()) <= FILTERED_EXACT_THRESHOLD:
            return self.exact.search(q, k, mask)
        return self.vectors.search(q, k, mask)

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        mode: str = "hybrid",
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        filters: Iterable[str | Filter] | None = None,
        depth: int = 50,
        unique_docs: bool = False,
        snippets: bool = True,
        snippet_width: int = 36,
    ) -> list[SearchResult]:
        """Run a query.

        ``alpha`` is the weight of the *dense* retriever (``1 - alpha`` for BM25)
        in weighted fusion, and scales the dense list's contribution in RRF.
        ``depth`` is how many candidates each retriever contributes to fusion.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        mask = filter_mask(self.chunks, parse_filters(filters))
        n = max(depth, k * (4 if unique_docs else 1))

        lex = self.lexical_search(query, n, mask) if mode in ("hybrid", "bm25") else []
        dense = self.dense_search(query, n, mask) if mode in ("hybrid", "dense") else []

        if mode == "bm25":
            fused = lex
        elif mode == "dense":
            fused = dense
        elif fusion == "rrf":
            fused = reciprocal_rank_fusion(
                {"bm25": [i for i, _ in lex], "dense": [i for i, _ in dense]},
                k=rrf_k,
                weights={"bm25": 2 * (1 - alpha), "dense": 2 * alpha},
            )
        else:
            fused = weighted_fusion(
                {"bm25": lex, "dense": dense}, weights={"bm25": 1 - alpha, "dense": alpha}
            )

        lex_info = {i: (r, s) for r, (i, s) in enumerate(lex, start=1)}
        dense_info = {i: (r, s) for r, (i, s) in enumerate(dense, start=1)}
        results: list[SearchResult] = []
        seen_docs: set[str] = set()
        for idx, score in fused:
            chunk = self.chunks[idx]
            if unique_docs:
                if chunk.doc_id in seen_docs:
                    continue
                seen_docs.add(chunk.doc_id)
            lr, ls = lex_info.get(idx, (None, None))
            dr, ds = dense_info.get(idx, (None, None))
            results.append(SearchResult(len(results) + 1, score, chunk, lr, ls, dr, ds))
            if len(results) == k:
                break
        if snippets:
            for r in results:
                r.snippet = make_snippet(r.chunk.text, query, width=snippet_width)
        return results

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        terms, lex_arrays = self.lexical.to_arrays()
        np.savez_compressed(path / "lexical.npz", terms=np.asarray(terms, dtype=str), **lex_arrays)
        np.savez(path / "embedder.npz", **self.embedder.arrays())
        np.savez(path / "vectors.npz", **self.vectors.arrays())
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        manifest = {
            "format_version": FORMAT_VERSION,
            "hybrid_search_version": __version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(self.config),
            "embedder": {"name": self.embedder.name, "config": self.embedder.config()},
            "vectors": {"kind": self.vectors.kind, "config": self.vectors.config()},
            "stats": self.stats,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SearchEngine:
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no index at {path} (run `hs index <dir>` first)")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"index format {manifest.get('format_version')} is not supported")
        config = EngineConfig(**manifest["config"])
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        with np.load(path / "lexical.npz", allow_pickle=False) as z:
            lexical = InvertedIndex.from_arrays(z["terms"].tolist(), {k: z[k] for k in z.files})
        with np.load(path / "embedder.npz", allow_pickle=False) as z:
            emb = manifest["embedder"]
            embedder = load_backend(emb["name"], emb["config"], {k: z[k] for k in z.files})
        with np.load(path / "vectors.npz", allow_pickle=False) as z:
            vec = manifest["vectors"]
            arrays = {k: z[k] for k in z.files}
            vectors = (HNSWIndex if vec["kind"] == "hnsw" else ExactIndex).from_state(
                vec["config"], arrays
            )
        return cls(chunks, lexical, embedder, vectors, config, manifest.get("stats"))


def _exact_view(index: HNSWIndex) -> ExactIndex:
    exact = ExactIndex(index.dim)
    exact._vecs = index.vectors  # shares memory; no copy
    return exact
