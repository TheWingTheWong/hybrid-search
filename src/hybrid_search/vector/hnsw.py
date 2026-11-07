"""Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016), from scratch.

Structure
    Every node gets a random top level ``floor(-ln(U) * mL)`` with
    ``mL = 1 / ln(M)``, so level populations shrink geometrically. Upper levels
    form a sparse "express" graph; level 0 contains every node with up to
    ``2 * M`` links.

Insert (Algorithm 1)
    Greedy descent from the entry point through levels above the new node's
    level with ``ef = 1``, then a beam search with ``ef_construction`` on each
    remaining level. Neighbors are chosen with the diversity heuristic
    (Algorithm 4): a candidate is kept only if it is closer to the new node
    than to any neighbor already kept, which preserves long-range links
    across clusters. Reverse links that overflow are pruned the same way.

Search (Algorithm 5)
    Greedy descent to level 0, then a beam search with ``ef = max(ef_search, k)``.

Similarity is the dot product, so vectors must be L2-normalized (cosine).
Query-time filters are supported by traversing the full graph but admitting
only allowed nodes into the result beam.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


class HNSWIndex:
    kind = "hnsw"

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        seed: int = 42,
    ):
        if M < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._mL = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)
        self._vecs = np.zeros((16, dim), dtype=np.float32)
        self._n = 0
        self.levels: list[int] = []
        self.links: list[list[list[int]]] = []  # links[node][level] -> neighbor ids
        self.entry_point = -1
        self.max_level = -1

    def __len__(self) -> int:
        return self._n

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs[: self._n]

    # -- construction -----------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        needed = self._n + len(vectors)
        if needed > len(self._vecs):
            cap = max(needed, 2 * len(self._vecs))
            grown = np.zeros((cap, self.dim), dtype=np.float32)
            grown[: self._n] = self._vecs[: self._n]
            self._vecs = grown
        for v in vectors:
            self._insert(v)

    def _random_level(self) -> int:
        return int(-math.log(1.0 - self._rng.random()) * self._mL)

    def _insert(self, q: np.ndarray) -> None:
        node = self._n
        self._vecs[node] = q
        self._n += 1
        level = self._random_level()
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])

        if self.entry_point < 0:
            self.entry_point, self.max_level = node, level
            return

        ep = self.entry_point
        for lc in range(self.max_level, level, -1):
            ep = self._search_layer(q, [ep], 1, lc)[0][1]

        entry = [ep]
        for lc in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entry, self.ef_construction, lc)
            max_links = self.M0 if lc == 0 else self.M
            neighbors = self._select_neighbors(found, self.M)
            self.links[node][lc] = neighbors
            for nb in neighbors:
                nb_links = self.links[nb][lc]
                nb_links.append(node)
                if len(nb_links) > max_links:
                    sims = self._vecs[nb_links] @ self._vecs[nb]
                    cand = sorted(zip(sims.tolist(), nb_links, strict=True), reverse=True)
                    self.links[nb][lc] = self._select_neighbors(cand, max_links)
            entry = [i for _, i in found]

        if level > self.max_level:
            self.entry_point, self.max_level = node, level

    def _select_neighbors(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Diversity heuristic over ``(sim, id)`` candidates sorted best-first."""
        selected: list[int] = []
        pruned: list[int] = []
        for sim, cid in candidates:
            if len(selected) >= m:
                break
            # Reject c if some already-selected neighbor is closer to c than q is.
            if selected and float(np.max(self._vecs[selected] @ self._vecs[cid])) > sim:
                pruned.append(cid)
                continue
            selected.append(cid)
        # keepPrunedConnections: top up with the best rejected candidates.
        for cid in pruned:
            if len(selected) >= m:
                break
            selected.append(cid)
        return selected

    # -- search -----------------------------------------------------------
    def _search_layer(
        self,
        q: np.ndarray,
        entry: list[int],
        ef: int,
        level: int,
        mask: np.ndarray | None = None,
    ) -> list[tuple[float, int]]:
        """Beam search on one level; returns up to ``ef`` ``(sim, id)`` best-first."""
        visited = set(entry)
        sims = self._vecs[entry] @ q
        candidates = [(-float(s), e) for s, e in zip(sims, entry, strict=True)]
        heapq.heapify(candidates)
        results: list[tuple[float, int]] = []  # min-heap: worst result on top
        for s, e in zip(sims, entry, strict=True):
            if mask is None or mask[e]:
                results.append((float(s), e))
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            neg_sim, c = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break
            fresh = [n for n in self.links[c][level] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for s, n in zip((self._vecs[fresh] @ q).tolist(), fresh, strict=True):
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, n))
                    if mask is None or mask[n]:
                        heapq.heappush(results, (s, n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return sorted(results, reverse=True)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        mask: np.ndarray | None = None,
        ef: int | None = None,
    ) -> list[tuple[int, float]]:
        if self._n == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        ep = self.entry_point
        for lc in range(self.max_level, 0, -1):
            ep = self._search_layer(q, [ep], 1, lc)[0][1]
        found = self._search_layer(q, [ep], max(ef or self.ef_search, k), 0, mask)
        return [(i, s) for s, i in found[:k]]

    # -- persistence ------------------------------------------------------
    def config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self.seed,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """Graph as one CSR adjacency per level (empty rows for absent nodes)."""
        out: dict[str, np.ndarray] = {
            "vectors": self.vectors.copy(),
            "levels": np.asarray(self.levels, dtype=np.int32),
        }
        for lc in range(self.max_level + 1):
            rows = [links[lc] if len(links) > lc else [] for links in self.links]
            out[f"indptr_{lc}"] = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(
                np.int64
            )
            out[f"indices_{lc}"] = np.asarray([i for r in rows for i in r], dtype=np.int32)
        return out

    @classmethod
    def from_state(cls, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> HNSWIndex:
        idx = cls(
            int(config["dim"]),
            M=int(config["M"]),
            ef_construction=int(config["ef_construction"]),
            ef_search=int(config["ef_search"]),
            seed=int(config["seed"]),
        )
        vecs = np.asarray(arrays["vectors"], dtype=np.float32)
        idx._vecs = vecs.copy() if len(vecs) else idx._vecs
        idx._n = len(vecs)
        idx.levels = [int(x) for x in arrays["levels"]]
        idx.entry_point = int(config["entry_point"])
        idx.max_level = int(config["max_level"])
        idx.links = [[[] for _ in range(lvl + 1)] for lvl in idx.levels]
        for lc in range(idx.max_level + 1):
            indptr, indices = arrays[f"indptr_{lc}"], arrays[f"indices_{lc}"]
            for node, lvl in enumerate(idx.levels):
                if lvl >= lc:
                    idx.links[node][lc] = indices[indptr[node] : indptr[node + 1]].tolist()
        return idx
