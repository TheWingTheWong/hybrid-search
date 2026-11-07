from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_search.engine import EngineConfig, SearchEngine

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
QUERIES = ROOT / "data" / "queries.jsonl"

RAFT_MD = """---
title: Raft Notes
tags: [consensus, raft]
year: 2021
---

# Raft Notes

Raft keeps a replicated log consistent across a cluster.

## Leader election

Followers wait for heartbeats. When the election timeout fires, a follower becomes a
candidate, increments its term and requests votes from its peers.

```
# this is a comment inside a code fence, not a heading
request_vote(term, candidate_id)
```

## Log replication

The leader appends entries and replicates them to followers.
"""

BLOOM_MD = """---
title: Bloom Filters
tags: [data-structures, probabilistic]
year: 2019
---

# Bloom Filters

A bloom filter is a bit array with k hash functions. Lookups may return false
positives but never false negatives.
"""


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "raft.md").write_text(RAFT_MD)
    (root / "bloom.md").write_text(BLOOM_MD)
    (root / "handbook.txt").write_text(
        "Pager Handbook\n\nWhen paged, acknowledge within five minutes and open an incident.\n"
    )
    records = [
        {
            "id": "quorum",
            "title": "Quorum",
            "text": "A majority of replicas.",
            "tags": ["consensus"],
        },
        {
            "id": "lsm",
            "title": "LSM tree",
            "text": "Memtables flush to sorted runs.",
            "tags": ["storage"],
        },
    ]
    (root / "glossary.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "skip.md").write_text("# ignored")
    return root


@pytest.fixture()
def tiny_engine(tiny_corpus: Path) -> SearchEngine:
    return SearchEngine.from_paths([tiny_corpus], EngineConfig(ann="hnsw"))


@pytest.fixture(scope="session")
def corpus_engine() -> SearchEngine:
    return SearchEngine.from_paths([CORPUS])
