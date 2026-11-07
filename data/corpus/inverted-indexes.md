---
title: Inverted Indexes and Lexical Ranking
tags: [search, information-retrieval, data-structures]
year: 2024
---

# Inverted Indexes and Lexical Ranking

Full-text search engines such as Lucene, Elasticsearch, OpenSearch, and Tantivy are built around one data structure: the inverted index. A forward index maps each document to the words it contains. An inverted index flips that around and maps each word to the documents that contain it, which makes "find all pages mentioning replication" a direct lookup rather than a scan over every file.

## Building the Index

### Analysis

Raw text first passes through an analyzer. A tokenizer splits text into terms, a lowercase filter normalizes case, stop-word removal may drop very common words like "the", and a stemmer reduces variants such as "replicating", "replicated", and "replicas" toward a common root. The same analysis must run on queries, or the terms will not match.

### Postings Lists

For each distinct term, the index stores a postings list: the sorted ids of documents containing it, usually with the term frequency in each document and optionally the positions where it occurs. Positions enable phrase queries like "write ahead log". The set of all terms, the dictionary or lexicon, is stored in a compact structure such as a finite state transducer so lookups are fast and prefix or fuzzy matching is possible.

Postings are compressed aggressively. Because ids are sorted, the engine stores gaps between consecutive ids, which are small numbers, and packs them with variable-byte or bit-packing schemes. Skip pointers let intersection jump ahead in long lists. Lucene writes immutable segments and merges them over time, much like compaction in a log-structured store.

## Scoring with TF-IDF

Matching documents must be ranked. The classic intuition has two parts. Term frequency (tf): a document that mentions a term many times is more likely about it. Inverse document frequency (idf): a term that appears in few documents is more discriminating than one appearing everywhere. A common idf is log(N / df), where N is the corpus size and df the number of documents containing the term. The tf-idf score of a document for a query sums tf times idf over query terms, often combined with cosine normalization over document vectors.

## BM25

BM25 refines tf-idf and has been the default lexical scorer in most engines for years. It adds two ideas. First, term frequency saturates: the tenth occurrence of a word adds much less than the second. Second, scores are normalized by document length, so long documents do not win merely by containing more words.

```
score(D, Q) = sum over t in Q of
   idf(t) * tf(t,D) * (k1 + 1) /
   (tf(t,D) + k1 * (1 - b + b * len(D) / avg_len))
```

Typical parameter values are k1 between 1.2 and 2.0, controlling how quickly tf saturates, and b = 0.75, controlling how strongly length is penalized. Setting b to zero disables length normalization entirely.

## Strengths and Limits

Lexical retrieval is fast, explainable, and excellent for exact identifiers, error codes, product names, and rare jargon. Query evaluation tricks such as WAND and block-max WAND skip documents that cannot reach the current top-k threshold, keeping latency low over billions of documents. Its weakness is vocabulary mismatch: a query for "car crash" will not match a document that only says "automobile collision". Synonym lists and query expansion help a little. Dense embedding retrieval addresses the problem more directly, which is why many modern stacks run both and fuse the results, for example with reciprocal rank fusion.
