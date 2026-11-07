"""Zero-dependency dense embeddings: hashed n-gram TF-IDF + sparse random projection.

Pipeline for one text:

1. **Features**, in three groups:
   word unigrams (stemmed), word bigrams, and character n-grams (default 3-5)
   of each word wrapped in boundary markers (``<raft>`` -> ``<ra``, ``raf`` ...).
   Character n-grams make the representation robust to typos and morphology
   ("replication" ~ "replicas" ~ "replicaton").
2. **Hashing trick** - every feature is hashed (CRC32) into one of ``2**n_bits``
   buckets, so there is no vocabulary to store.
3. **TF-IDF** - sublinear term frequency ``1 + ln(tf)`` times a smoothed IDF
   learned per bucket in :meth:`HashingEmbedder.fit`. Each group is
   L2-normalized and scaled by its group weight.
4. **Random projection** - the ``2**n_bits``-dimensional sparse vector is
   projected to ``dim`` dimensions with a *very sparse* random matrix
   (Li, Hastie & Church, 2006): each bucket maps to ``nnz`` output coordinates
   with random signs. By Johnson-Lindenstrauss, cosine similarity is
   approximately preserved. The matrix is generated from a SplitMix64 hash of
   ``(seed, bucket, slot)``, so it is never stored and is identical across
   platforms and numpy versions.

This is not a semantic model - it cannot know that "elect" relates to "vote" -
but it is a strong, fast, fully offline lexical-fuzzy retriever and a useful
complement to BM25 for misspellings and partial words.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..text import DEFAULT_ANALYZER, iter_words, normalize
from .base import EmbeddingBackend, l2_normalize

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer over a ``uint64`` array (wrapping arithmetic)."""
    z = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return z ^ (z >> np.uint64(31))


class HashingEmbedder(EmbeddingBackend):
    name = "hashing"

    def __init__(
        self,
        dim: int = 512,
        n_bits: int = 18,
        char_ngrams: tuple[int, int] = (3, 5),
        word_weight: float = 1.0,
        bigram_weight: float = 0.5,
        char_weight: float = 0.8,
        nnz: int = 4,
        seed: int = 13,
        idf: np.ndarray | None = None,
    ):
        self.dim = int(dim)
        self.n_bits = int(n_bits)
        self.n_buckets = 1 << self.n_bits
        self.char_ngrams = (int(char_ngrams[0]), int(char_ngrams[1]))
        self.weights = {"w": word_weight, "b": bigram_weight, "c": char_weight}
        self.nnz = int(nnz)
        self.seed = int(seed)
        self.idf = (
            np.asarray(idf, dtype=np.float32)
            if idf is not None
            else np.ones(self.n_buckets, np.float32)
        )
        if self.idf.shape != (self.n_buckets,):
            raise ValueError("idf array does not match n_bits")
        self._build_projection()

    # -- projection -------------------------------------------------------
    def _build_projection(self) -> None:
        keys = np.arange(self.n_buckets * self.nnz, dtype=np.uint64)
        keys += np.uint64((self.seed * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF)
        h = splitmix64(keys).reshape(self.n_buckets, self.nnz)
        self._proj_dim = (h % np.uint64(self.dim)).astype(np.int32)
        sign = np.where((h >> np.uint64(63)) == 1, -1.0, 1.0).astype(np.float32)
        self._proj_sign = sign / np.float32(np.sqrt(self.nnz))

    # -- features ---------------------------------------------------------
    def _hash(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) & (self.n_buckets - 1)

    def features(self, text: str) -> dict[str, Counter[int]]:
        """Bucketed feature counts per group (``w`` words, ``b`` bigrams, ``c`` chars)."""
        terms = DEFAULT_ANALYZER.analyze(text)
        groups: dict[str, Counter[int]] = {"w": Counter(), "b": Counter(), "c": Counter()}
        groups["w"].update(self._hash("w:" + t) for t in terms)
        groups["b"].update(self._hash(f"b:{a} {b}") for a, b in zip(terms, terms[1:], strict=False))
        lo, hi = self.char_ngrams
        chars = groups["c"]
        for m in iter_words(text):
            word = normalize(m.group())
            if len(word) < 2 or word in DEFAULT_ANALYZER.stopwords:
                continue
            padded = f"<{word}>"
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    chars[self._hash("c:" + padded[i : i + n])] += 1
        return groups

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        """Learn smoothed IDF per bucket: ``ln((1 + N) / (1 + df)) + 1``."""
        df = np.zeros(self.n_buckets, dtype=np.float64)
        for text in texts:
            seen: set[int] = set()
            for counts in self.features(text).values():
                seen.update(counts)
            if seen:
                df[np.fromiter(seen, dtype=np.int64, count=len(seen))] += 1
        n = len(texts)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        return self

    def sparse_vector(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Weighted TF-IDF vector as ``(bucket_ids, values)`` before projection."""
        ids, vals = [], []
        for group, counts in self.features(text).items():
            if not counts:
                continue
            b = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            tf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
            v = (1.0 + np.log(tf)) * self.idf[b]
            v *= self.weights[group] / max(float(np.linalg.norm(v)), 1e-12)
            ids.append(b)
            vals.append(v)
        if not ids:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(ids), np.concatenate(vals).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            b, v = self.sparse_vector(text)
            if b.size:
                out[row] = np.bincount(
                    self._proj_dim[b].ravel(),
                    weights=(self._proj_sign[b] * v[:, None]).ravel(),
                    minlength=self.dim,
                )
        return l2_normalize(out)

    # -- persistence ------------------------------------------------------
    def config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_bits": self.n_bits,
            "char_ngrams": list(self.char_ngrams),
            "word_weight": self.weights["w"],
            "bigram_weight": self.weights["b"],
            "char_weight": self.weights["c"],
            "nnz": self.nnz,
            "seed": self.seed,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {"idf": self.idf}

    @classmethod
    def from_state(
        cls, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
    ) -> HashingEmbedder:
        cfg = dict(config)
        cfg["char_ngrams"] = tuple(cfg.get("char_ngrams", (3, 5)))
        return cls(**cfg, idf=arrays.get("idf"))
