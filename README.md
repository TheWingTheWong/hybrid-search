# hybrid-search

A local search engine for Markdown, plain-text and JSON Lines files. It ranks with
BM25F over an inverted index, retrieves with dense vectors through an HNSW graph,
and merges the two lists with reciprocal rank fusion. The ranking functions, the
inverted index, the HNSW graph and the default embedding model are written from
scratch on top of numpy. There are no search libraries, no network calls and no
API keys.

```console
$ hs index data/corpus
indexed 45 documents -> 173 chunks, 1971 terms, 512-d hashing vectors (hnsw) in 0.59s
wrote .hs/

$ hs search "how does raft elect a leader" -k 3 --explain
 1  Raft Consensus > Roles and Terms > Leader Election                                        0.0325
    raft-consensus.md  #consensus #distributed-systems #replication
    bm25 #1 (14.44)   dense #2 (0.286)
    Followers expect periodic heartbeats from the leader. If a follower hears nothing for its
    election timeout, it increments the term, votes for itself, and asks peers for votes. A node
    grants at most one vote per ...

 2  Leader Election                                                                           0.0320
    glossary.jsonl:16  #consensus #distributed-systems #coordination
    bm25 #4 (11.34)   dense #1 (0.370)
    Leader election is the process by which nodes in a cluster choose a single coordinator
    responsible for ordering writes or assigning work. Elections typically run after the current
    leader stops sending heartbeats and require votes from ...

 3  Raft Consensus > Roles and Terms                                                          0.0318
    raft-consensus.md  #consensus #distributed-systems #replication
    bm25 #2 (12.42)   dense #4 (0.238)
    Every server is in one of three states: follower, candidate, or leader. Time is divided into
    numbered terms, and each term begins with an election. A term number acts like a logical epoch;
    any message carrying ...

3 results for "how does raft elect a leader" in 1.8 ms (hybrid/rrf, 173 chunks)
```

In a terminal, matched terms are highlighted. `--explain` prints where each hit
ranked in each retriever. The glossary entry at #2 was only 4th for BM25, but the
dense retriever ranked it 1st, and fusion moved it up.

## Quickstart

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repo> && cd hybrid-search
uv sync
uv run hs index data/corpus                # writes ./.hs (or --index-dir / $HS_INDEX)
uv run hs search "token bucket burst" -k 5
uv run hs search "compaction" -f tags=storage -f "year>=2022" --docs
uv run hs search "bloom filtr false positve" --mode dense      # bm25 | dense | hybrid
uv run hs search "leader lease" --fusion weighted --alpha 0.3  # 0 = all BM25, 1 = all dense
uv run hs search "quorum" --json | jq '.results[0].snippet'
uv run hs eval                                                 # reproduce the tables below
uv run hs serve                                                # JSON API on 127.0.0.1:8765
```

`hs index` accepts any mix of files and directories. Files in hidden
directories are skipped.

| Format | Parsed as |
|---|---|
| `.md`, `.markdown` | Optional `---` front matter (`title`, `tags: [a, b]`, any scalar), title from front matter, then the first `# H1`, then the file name |
| `.txt` | The first line is the title if it is short and followed by a blank line |
| `.jsonl`, `.ndjson` | One document per line. `text` is required; `id` and `title` are optional; every other key becomes filterable metadata |

## How it works

```
                 index time                                         query time
 files ─► loaders ─► heading split ─► token windows ─┐       query ─┬─► analyzer ─► BM25F ─► top 50 ┐
                                                     │              │                               ├─► RRF ─► top k ─► snippets
               ┌─────────────────────────────────────┤              └─► embedder ─► HNSW ──► top 50 ┘
               ▼                                     ▼                       ▲           ▲
   inverted index (CSR postings,           hashing embedder ─► HNSW graph    │           │
   per-field tf, field lengths)            (fit IDF, project)                └─ metadata filters (allow-mask)
```

| Module | Responsibility |
|---|---|
| `text.py` | Unicode normalization, tokenizer, stopwords, Porter stemmer (from the 1980 paper) |
| `ingest.py` | Loaders, front matter, Markdown sectioning, overlapping token windows |
| `bm25.py` | Inverted index, BM25 and BM25F |
| `embeddings/` | `EmbeddingBackend` interface, `hashing` (default) and `sentence-transformers` backends |
| `vector/` | `ExactIndex` (brute-force cosine) and `HNSWIndex` |
| `fusion.py` | Reciprocal rank fusion, min-max / z-score weighted blending |
| `filters.py`, `highlight.py` | Query-time metadata filters; query-biased snippets with highlight spans |
| `engine.py` | Builds, queries, saves and loads everything above |
| `evaluation/` | MRR, nDCG, recall@k, HNSW-vs-exact recall sweeps |
| `cli.py`, `server.py` | `hs` command and the stdlib HTTP API |

### Chunking

Markdown is split on ATX headings. `#` lines inside fenced code blocks are not
treated as headings. Each section keeps a heading path such as
`Compaction Strategies > Leveled Compaction`. Any section longer than
`--max-tokens` (default 160 whitespace tokens) is cut into windows that overlap by
`--overlap` (default 32). Windows are sliced from the original text by character
offset, so code blocks keep their line breaks. Each chunk has three fields:
document title, heading path and body.

### Lexical retrieval: BM25 and BM25F

Postings are collected in dicts while indexing. They are then frozen into a CSR
layout: an `offsets` array, an `int32` document-id array and a
`(postings × fields)` term-frequency matrix. Scoring works one term at a time.
For each query term, BM25 saturation is computed for all of its postings in one
numpy expression and added into a dense accumulator.

- IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))`, the Lucene variant, which is never negative.
- BM25 treats all fields as one: `idf · tf·(k1+1) / (tf + k1·(1 - b + b·|d|/avgdl))`, with `k1=1.2`, `b=0.75`.
- BM25F (the default) normalizes length per field before saturating:
  `tf~ = Σ_f w_f · tf_f / (1 - b_f + b_f·len_f/avglen_f)`, then `idf · tf~·(k1+1)/(tf~+k1)`.
  Weights are title 3, heading 2, body 1.

### Dense retrieval: the zero-dependency embedder

The default `hashing` backend is fit on the corpus at index time:

1. Features: stemmed word unigrams, word bigrams, and character 3- to 5-grams of
   each word with boundary markers (`<raft>` → `<ra`, `raf`, `aft`, `ft>`, `<raf`, ...).
2. Each feature is hashed with CRC32 into one of 2^18 buckets, so there is no
   vocabulary to store.
3. Weighting is sublinear TF × smoothed IDF per bucket. Each feature group is
   L2-normalized and then scaled (words 1.0, bigrams 0.5, char n-grams 0.8).
4. A very sparse random projection (Li, Hastie & Church 2006) maps the vectors
   to 512 dimensions. Each bucket maps to 4 output coordinates with random signs.
   The projection matrix is never stored. It is derived from SplitMix64 of
   `(seed, bucket, slot)`, so it is identical on every platform and numpy version.

This is not a semantic model, and it will never learn that "elect" and "vote"
are related. What it does is give a fuzzy lexical signal that tolerates
misspellings and partial words, which is the main weakness of BM25. The
evaluation below shows that trade-off.

For semantic embeddings, install the optional backend:

```sh
uv sync --extra st
uv run hs index data/corpus --backend st                          # all-MiniLM-L6-v2
uv run hs index data/corpus --backend st:BAAI/bge-small-en-v1.5
```

The backend name and config are saved in the index manifest, so `hs search`
reloads the same model.

### Approximate nearest neighbours: HNSW

`vector/hnsw.py` follows Malkov & Yashunin (2016):

- Each node gets a random top level, `floor(-ln U · 1/ln M)`.
- Inserts descend greedily through the upper layers, then run a beam search
  with `ef_construction` on each lower layer.
- Neighbours are chosen with the diversity heuristic (Algorithm 4), with pruned
  connections kept as backfill. When a reverse link overflows `M` (or `2M` on
  layer 0), the list is pruned with the same heuristic.
- The graph is saved as one CSR adjacency per layer.

Filtered queries walk the whole graph but only let allowed nodes into the result
beam. When a filter allows 2,000 vectors or fewer, the engine scans that subset
exactly instead, because graph search degrades under very selective filters.

### Fusion

- **RRF** (default): `score(d) = Σ_r w_r / (60 + rank_r(d))`. It uses only
  ranks, so BM25's unbounded scores never have to be calibrated against cosine
  similarities.
- **Weighted**: each list is min-max (or z-score) normalized, then blended as
  `(1-α)·bm25 + α·dense`. A document missing from one list gets that list's
  minimum.

`--alpha` sets the dense share in both modes. The default is 0.5, i.e. equal
weight.

### Filters and snippets

```
-f tags=raft            list fields: contains any of   (tags=raft,paxos)
-f source=notes/*.md    glob match on = and !=
-f year>=2022           numeric > >= < <=
-f title~hash           case-insensitive substring
```

Filters are ANDed. Available fields are `doc_id`, `title`, `heading`, `source`,
`format`, and any front-matter or JSON Lines key. Filters are applied as a mask
inside both retrievers, not after the top k is cut, so a filtered query still
returns k results when k matching chunks exist.

For snippets, a window of 36 words slides across the chunk. The window that
covers the most distinct query terms is kept; ties go to the most total matches,
then the earliest position. Highlights are returned as character spans
(`snippet.highlights` in JSON), so clients can render them however they like.

## Evaluation

`data/` contains a small test collection written for this repository:

- **Corpus**: 24 Markdown engineering notes on distributed systems and storage
  (Raft, Paxos, LSM trees, MVCC, HNSW, and so on), a plain-text on-call handbook,
  and a 20-entry JSON Lines glossary. That is 45 documents and 173 chunks.
- **Queries**: 48 queries with graded relevance (2 = primary document,
  1 = partially relevant), split into four types:
  - `keyword`: terse queries in the documents' own vocabulary
  - `question`: natural-language questions
  - `paraphrase`: the concept described without its usual terms
  - `typo`: misspelled queries

Relevance is judged per document. Each run collapses its ranking to one hit per
document before scoring. nDCG uses exponential gain (`2^rel − 1`).

The tables below are the output of `uv run hs eval --markdown --ann-recall
--synthetic 5000` with the default hashing backend, measured on an Apple M2.

| Run | MRR@10 | nDCG@10 | R@5 | R@10 | p50 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.979 | 0.939 | 0.903 | 0.924 | 0.10 |
| BM25F | 0.969 | 0.935 | 0.903 | 0.924 | 0.09 |
| Dense, exact | 0.972 | 0.942 | 0.917 | 0.944 | 0.20 |
| Dense, HNSW | 0.972 | 0.942 | 0.917 | 0.944 | 0.36 |
| Hybrid, RRF | 0.986 | **0.965** | 0.924 | **0.965** | 0.55 |
| Hybrid, weighted | **0.990** | 0.961 | **0.931** | 0.955 | 0.54 |

| nDCG@10 by query type | keyword (26) | paraphrase (10) | question (7) | typo (5) |
|---|---:|---:|---:|---:|
| BM25 | 0.998 | 0.827 | 0.980 | 0.800 |
| BM25F | 0.998 | 0.845 | 0.927 | 0.800 |
| Dense, exact | 0.997 | 0.774 | 0.934 | 1.000 |
| Dense, HNSW | 0.997 | 0.774 | 0.934 | 1.000 |
| Hybrid, RRF | 0.998 | 0.847 | 0.982 | 1.000 |
| Hybrid, weighted | 0.997 | 0.831 | 0.987 | 1.000 |

What these tables show:

- **The two retrievers fail on different queries, and that is where fusion
  gains.** BM25 returns nothing at all for "conssistent hashng", because every
  term is misspelled. It survives the other typo queries only because each still
  contains one correctly spelled word ("proof", "retries", "generations"). The
  character n-gram embedder ranks the right document first for all five. The
  embedder is weakest on paraphrases. RRF has the best nDCG@10 and R@10 overall
  and is at or near the top in every query type.
- **BM25F is not automatically better than BM25.** Field weighting helps
  paraphrases (+0.018). For "detect divergent replicas by comparing hash trees",
  the title *Merkle Trees* lifts the right note above a glossary entry. It hurts
  questions (−0.053), and the whole loss comes from one query: in "how do I cap
  how many calls a client can make each second", the verb "cap" matches the
  title *CAP Theorem*, and the 3× title weight puts that note above *Rate
  Limiting*. The weights (3/2/1) were not tuned on this set, and with 48 queries
  they should not be.
- **The keyword queries are saturated.** Every run scores ≥ 0.997 on them, so
  the differences come from the 22 harder queries.
- **The remaining miss is a paraphrase.** For "deleted keys are marked instead
  of removed", the glossary entry *Tombstone* is ranked 3rd, after garbage
  collection and skip lists. No retriever here can connect "marked" with
  "tombstone"; that requires a semantic model. `hs eval --failures` lists
  misses like this one.

Keep these numbers in perspective. The collection is small and the same person
wrote the documents and the labels. Use it to compare configurations of this
engine and to catch regressions; it does not support claims about retrieval
quality in general. `tests/test_eval.py` fails if hybrid RRF falls below either
single retriever.

### HNSW vs exact search

Recall@10 is measured against brute-force cosine search over the same vectors.

On the corpus (173 chunk vectors, 48 real queries):

| ef_search | recall@10 | HNSW p50 ms | exact p50 ms |
|---:|---:|---:|---:|
| 10 | 0.952 | 0.108 | 0.018 |
| 16 | 0.998 | 0.138 | 0.018 |
| 32 | 1.000 | 0.211 | 0.018 |
| 64 | 1.000 | 0.247 | 0.018 |
| 128 | 1.000 | 0.421 | 0.018 |

On 5,000 synthetic 128-d vectors from heavily overlapping Gaussian clusters
(M=16, ef_construction=100, built in 11.4 s):

| ef_search | recall@10 | HNSW p50 ms | exact p50 ms |
|---:|---:|---:|---:|
| 10 | 0.718 | 0.179 | 0.106 |
| 16 | 0.779 | 0.242 | 0.106 |
| 32 | 0.861 | 0.624 | 0.106 |
| 64 | 0.931 | 0.946 | 0.106 |
| 128 | 0.979 | 1.158 | 0.106 |
| 256 | 0.996 | 2.182 | 0.106 |

At these sizes, one BLAS matrix-vector product beats a graph walk in
interpreted Python, so exact search is faster here. The HNSW index is written
for clarity, with each step of the paper visible in the code. What carries over
to larger collections is the shape of the recall/ef curve: exact search cost
grows linearly with N, while HNSW cost grows roughly logarithmically. Use
`--ann exact` for small collections. The synthetic data is deliberately hard;
with well-separated clusters, recall reaches 1.0 by ef=32 and the curve shows
nothing.

## HTTP API

`hs serve` starts a threaded `http.server`. There are no extra dependencies.

```console
$ curl -s 'localhost:8765/search?q=merkle+tree+proof&k=2&docs=1&filter=year>=2020' | jq '.results[].doc_id'
$ curl -s localhost:8765/search -d '{"query": "gc pauses", "k": 3, "filters": ["tags=memory"], "mode": "hybrid", "fusion": "rrf"}'
$ curl -s localhost:8765/stats
```

Each result includes the score, chunk and document ids, the heading path,
metadata, `bm25` and `dense` sub-objects (rank and score, or `null`), and
`snippet.text` with `snippet.highlights` spans. Bad parameters return 400 with
`{"error": ...}`.

## Python API

```python
from hybrid_search.engine import EngineConfig, SearchEngine

engine = SearchEngine.from_paths(["data/corpus"], EngineConfig(ann="hnsw", scorer="bm25f"))
for r in engine.search("why must read and write quorums overlap", k=3, unique_docs=True):
    print(r.rank, r.chunk.doc_id, r.bm25_rank, r.dense_rank, r.snippet.render("[", "]"))

engine.save(".hs")
engine = SearchEngine.load(".hs")
```

To plug in another embedding model, subclass `EmbeddingBackend` (`embed`,
`embed_query`, and optionally `fit`, `config` and `arrays`) and pass an
instance to `SearchEngine.from_chunks(..., embedder=...)`.

## Index format

An index is a directory with `manifest.json` (format version, config, embedder
and vector-index parameters, build stats), `chunks.jsonl`, and three `.npz`
files: CSR postings, embedder state (the IDF table) and vectors plus the HNSW
adjacency. Everything loads with `allow_pickle=False`. On the bundled corpus the
index is 1.5 MB, most of it the 2^18-entry IDF table.

## Development

```sh
uv sync                      # numpy runtime; pytest and ruff in the dev group
uv run pytest                # 154 tests, ~10 s
uv run ruff check . && uv run ruff format --check .
```

CI runs lint, then the tests on Python 3.10 to 3.13 (Ubuntu) and 3.13 (macOS),
then indexes, searches and evaluates the bundled corpus.

## Roadmap

- Run the evaluation with the `sentence-transformers` backend and add those
  rows to the table. The paraphrase column is where a semantic model should
  help.
- Incremental indexing: append-only segments with a merge policy, instead of a
  full rebuild.
- Block-max WAND for top-k BM25 without scoring every posting.
- Product quantization for the vector store, and a vectorized HNSW
  neighbour-expansion path.
- Tune per-field BM25F weights and the RRF constant on a held-out split, once
  the query set is large enough to split.
- Phrase and proximity scoring using stored positions.

## License

MIT. See [LICENSE](LICENSE).
