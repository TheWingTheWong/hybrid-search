---
title: Vector Search
tags: [search, machine-learning, indexing]
year: 2025
---

# Vector Search

Vector search, also known as dense retrieval or approximate nearest neighbor search, finds items whose meaning is close to a query rather than items that share its exact words. It is the retrieval layer behind semantic search, recommendation, deduplication, and retrieval-augmented generation for language models. The pipeline has two halves: turning content into vectors, and finding the nearest vectors quickly.

## Embeddings

An embedding model maps a piece of text, an image, or another object to a fixed-length array of floating-point numbers, commonly 384, 768, or 1536 dimensions. Models are trained so that related inputs land near each other in this space. A query about "how a database survives a power failure" should end up close to a passage describing write-ahead logging, even though they share few words. That ability to bridge vocabulary gaps is the main reason to add dense retrieval next to keyword search.

### Similarity Measures

Closeness is measured with cosine similarity, dot product, or Euclidean distance. Cosine similarity is the dot product of two vectors divided by the product of their lengths, ranging from -1 to 1. If vectors are normalized to unit length up front, cosine and dot product give identical rankings, and Euclidean distance becomes a monotonic function of them, so many systems normalize once and use the cheapest operation.

```
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

## Exact Versus Approximate Search

Brute-force search compares the query against every vector. For a million 768-dimensional vectors, that is about 768 million multiply-adds per query: feasible on a GPU or with SIMD for small collections, but too slow and costly at hundreds of millions of items. Approximate nearest neighbor (ANN) indexes trade a small amount of accuracy for orders-of-magnitude speedups.

## Index Families

### HNSW

Hierarchical Navigable Small World graphs connect each vector to a handful of neighbors, controlled by a parameter M, typically 16 to 64. Upper layers are sparse and act like express highways; the bottom layer contains every point. A search greedily walks toward the query on each layer, then descends. The efSearch parameter sets how many candidates are kept during the walk; raising it increases accuracy and latency. HNSW offers excellent recall at low latency but uses a lot of memory and is slow to build.

### IVF

Inverted file indexes cluster vectors with k-means into, say, 4,096 centroids. Each vector is filed under its nearest centroid. At query time, only the nprobe closest clusters are scanned. IVF is cheaper to build and smaller than HNSW, and it combines well with product quantization (IVF-PQ), which compresses each vector into a few dozen bytes so billions fit in RAM, at some cost in precision.

## Recall Versus Latency

ANN quality is measured as recall at k: the fraction of the true top-k neighbors that the index actually returns. Every index exposes knobs that slide along a curve: more probes or a larger candidate list mean higher recall and slower queries. A common target is 0.95 recall at 10 with p99 latency under 20 milliseconds. Tune against a held-out set with exact ground truth, and remember that filtering by metadata, such as restricting to one tenant, can cripple graph-based search unless the engine supports filtered traversal. Finally, dense retrieval can miss exact identifiers and rare terms, so combining it with BM25 in a hybrid ranker usually beats either alone.
