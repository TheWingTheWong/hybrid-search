"""Inverted index with BM25 and BM25F scoring, implemented from scratch.

Postings are accumulated in Python dicts while indexing and then *frozen* into a
CSR layout (``offsets``/``doc_ids``/``tfs``) of numpy arrays. Scoring is
term-at-a-time: for every query term we vectorize the BM25 saturation over its
postings list and scatter-add into a dense accumulator.

Formulas (Robertson & Zaragoza, "The Probabilistic Relevance Framework", 2009):

* IDF (Lucene variant, always non-negative)::

      idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

* BM25 over the concatenation of all fields::

      score = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

* BM25F: per-field length normalization *before* saturation, then one
  saturation over the weighted pseudo-frequency::

      tf~ = sum_f w_f * tf_f / (1 - b_f + b_f * len_f / avglen_f)
      score = sum_t idf(t) * tf~ * (k1 + 1) / (tf~ + k1)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .text import DEFAULT_ANALYZER, Analyzer

DEFAULT_FIELDS: tuple[str, ...] = ("title", "heading", "body")


class InvertedIndex:
    """A multi-field inverted index storing per-field term frequencies."""

    def __init__(self, fields: Sequence[str] = DEFAULT_FIELDS, analyzer: Analyzer | None = None):
        self.fields = tuple(fields)
        self.analyzer = analyzer or DEFAULT_ANALYZER
        self._building: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
        self._lengths: list[tuple[int, ...]] = []
        self._frozen = False
        self.vocab: dict[str, int] = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int32)
        self.tfs = np.zeros((0, len(self.fields)), dtype=np.float32)
        self.field_lengths = np.zeros((0, len(self.fields)), dtype=np.float32)

    # -- building ---------------------------------------------------------
    def add(self, fields: Mapping[str, str]) -> int:
        """Index one document given as ``{field: text}``; returns its doc number."""
        if self._frozen:
            raise RuntimeError("index is frozen; build a new InvertedIndex to add documents")
        doc = len(self._lengths)
        per_field = [Counter(self.analyzer.analyze(fields.get(f, "") or "")) for f in self.fields]
        self._lengths.append(tuple(sum(c.values()) for c in per_field))
        for term in set().union(*per_field):
            tf = tuple(c.get(term, 0) for c in per_field)
            self._building.setdefault(term, []).append((doc, tf))
        return doc

    def freeze(self) -> InvertedIndex:
        if self._frozen:
            return self
        terms = sorted(self._building)
        self.vocab = {t: i for i, t in enumerate(terms)}
        sizes = np.fromiter(
            (len(self._building[t]) for t in terms), dtype=np.int64, count=len(terms)
        )
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        total = int(self.offsets[-1])
        self.doc_ids = np.empty(total, dtype=np.int32)
        self.tfs = np.empty((total, len(self.fields)), dtype=np.float32)
        for i, t in enumerate(terms):
            lo, hi = self.offsets[i], self.offsets[i + 1]
            plist = self._building[t]
            self.doc_ids[lo:hi] = [d for d, _ in plist]
            self.tfs[lo:hi] = [tf for _, tf in plist]
        self.field_lengths = np.asarray(self._lengths, dtype=np.float32).reshape(
            -1, len(self.fields)
        )
        self._building.clear()
        self._lengths.clear()
        self._frozen = True
        return self

    # -- stats ------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self._lengths) if not self._frozen else int(self.field_lengths.shape[0])

    @property
    def n_terms(self) -> int:
        return len(self.vocab) if self._frozen else len(self._building)

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(doc_ids, per-field tfs)`` for an analyzed term."""
        self.freeze()
        i = self.vocab.get(term)
        if i is None:
            return self.doc_ids[:0], self.tfs[:0]
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.doc_ids[lo:hi], self.tfs[lo:hi]

    def df(self, term: str) -> int:
        return len(self.postings(term)[0])

    def idf(self, term: str) -> float:
        n, df = self.n_docs, self.df(term)
        return float(np.log1p((n - df + 0.5) / (df + 0.5)))

    # -- persistence ------------------------------------------------------
    def to_arrays(self) -> tuple[list[str], dict[str, np.ndarray]]:
        self.freeze()
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        return terms, {
            "offsets": self.offsets,
            "doc_ids": self.doc_ids,
            "tfs": self.tfs,
            "field_lengths": self.field_lengths,
        }

    @classmethod
    def from_arrays(
        cls,
        terms: Sequence[str],
        arrays: Mapping[str, np.ndarray],
        fields: Sequence[str] = DEFAULT_FIELDS,
        analyzer: Analyzer | None = None,
    ) -> InvertedIndex:
        idx = cls(fields, analyzer)
        idx.vocab = {t: i for i, t in enumerate(terms)}
        idx.offsets = np.asarray(arrays["offsets"], dtype=np.int64)
        idx.doc_ids = np.asarray(arrays["doc_ids"], dtype=np.int32)
        idx.tfs = np.asarray(arrays["tfs"], dtype=np.float32)
        idx.field_lengths = np.asarray(arrays["field_lengths"], dtype=np.float32)
        idx._frozen = True
        return idx


@dataclass
class BM25:
    """Classic Okapi BM25 over all fields concatenated."""

    index: InvertedIndex
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        self.index.freeze()
        lengths = self.index.field_lengths.sum(axis=1)
        avg = float(lengths.mean()) if lengths.size else 1.0
        self._norm = self.k1 * (1.0 - self.b + self.b * lengths / max(avg, 1e-9))

    def _term_scores(self, docs: np.ndarray, tfs: np.ndarray, idf: float) -> np.ndarray:
        tf = tfs.sum(axis=1)
        return idf * tf * (self.k1 + 1.0) / (tf + self._norm[docs])

    def score(self, query: str | Sequence[str]) -> np.ndarray:
        """Dense score vector over all documents (0 for non-matching docs)."""
        terms = self.index.analyzer.analyze(query) if isinstance(query, str) else list(query)
        acc = np.zeros(self.index.n_docs, dtype=np.float32)
        for term, qtf in Counter(terms).items():
            docs, tfs = self.index.postings(term)
            if len(docs):
                acc[docs] += qtf * self._term_scores(docs, tfs, self.index.idf(term))
        return acc

    def search(
        self, query: str | Sequence[str], k: int = 10, mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc, score)`` pairs; ``mask`` is an optional boolean allow-list."""
        return top_k(self.score(query), k, mask)


@dataclass
class BM25F(BM25):
    """BM25F with per-field weights and per-field length normalization."""

    weights: Mapping[str, float] = field(
        default_factory=lambda: {"title": 3.0, "heading": 2.0, "body": 1.0}
    )
    field_b: Mapping[str, float] = field(
        default_factory=lambda: {"title": 0.5, "heading": 0.5, "body": 0.75}
    )

    def __post_init__(self) -> None:
        self.index.freeze()
        fl = self.index.field_lengths
        avg = fl.mean(axis=0) if fl.size else np.ones(len(self.index.fields))
        avg = np.maximum(avg, 1e-9)
        b = np.array([self.field_b.get(f, self.b) for f in self.index.fields], dtype=np.float32)
        self._w = np.array([self.weights.get(f, 1.0) for f in self.index.fields], dtype=np.float32)
        # Per-document, per-field normalizer: (1 - b_f + b_f * len_f / avglen_f)
        self._field_norm = (1.0 - b + b * fl / avg).astype(np.float32)

    def _term_scores(self, docs: np.ndarray, tfs: np.ndarray, idf: float) -> np.ndarray:
        pseudo_tf = (tfs * self._w / self._field_norm[docs]).sum(axis=1)
        return idf * pseudo_tf * (self.k1 + 1.0) / (pseudo_tf + self.k1)


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Indices of the ``k`` largest strictly-positive scores, best first."""
    if mask is not None:
        scores = np.where(mask, scores, 0.0)
    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0 or k <= 0:
        return []
    if candidates.size > k:
        part = np.argpartition(-scores[candidates], k - 1)[:k]
        candidates = candidates[part]
    order = np.lexsort((candidates, -scores[candidates]))
    return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]
