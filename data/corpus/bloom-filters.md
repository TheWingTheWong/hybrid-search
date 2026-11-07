---
title: Bloom Filters
tags: [data-structures, probabilistic, storage]
year: 2019
---

# Bloom Filters

A Bloom filter is a compact, probabilistic set that answers one question: could this item be in the set? It can say "definitely not" or "probably yes". It never produces false negatives, but it occasionally produces false positives. In exchange for that small uncertainty it uses a tiny fraction of the memory a real set would need, often under 10 bits per element regardless of how large the elements are.

## Structure and Operations

The filter is a bit array of m bits, all starting at zero, plus k independent hash functions. To insert an item, compute its k hashes, reduce each modulo m, and set those bit positions to one. To query, compute the same k positions; if any of them is zero, the item was never inserted. If all are one, the item is probably present, though those bits might have been set by other items.

```
insert(x): for i in 1..k: bits[h_i(x) % m] = 1
contains(x): return all(bits[h_i(x) % m] == 1 for i in 1..k)
```

In practice you rarely need k truly independent hash functions. The Kirsch-Mitzenmacher trick derives them from two base hashes as h1 + i * h2, which works just as well and is much cheaper.

## Sizing the Filter

For n inserted items, the false positive rate is approximately (1 - e^(-kn/m))^k. The optimal number of hashes is k = (m/n) ln 2. Some handy rules of thumb:

- About 4.8 bits per element gives a 10 percent false positive rate.
- About 9.6 bits per element gives roughly 1 percent, with k around 7.
- Each additional 4.8 bits per element cuts the error rate by another factor of ten.

So a filter for 10 million keys at 1 percent error needs around 12 megabytes. Overfilling is the classic mistake: once you insert far more than the design capacity, almost every bit becomes one and the filter says yes to everything.

## Where They Show Up

Storage engines are the textbook use case. In an LSM tree, each sorted run on disk carries a Bloom filter over its keys, so a point lookup can skip files that cannot contain the key and avoid expensive disk reads. Web caches use them to avoid caching one-hit wonders, browsers once used them to check URLs against malicious-site lists, and distributed joins use them to prune rows before shipping data across the network. Databases also use them to cut down on pointless lookups during hash joins.

## Variants and Limitations

The standard filter does not support deletion, since clearing a bit could erase evidence of other items. A counting Bloom filter replaces each bit with a small counter, typically 4 bits, allowing removal at four times the space. Cuckoo filters store short fingerprints in a cuckoo hash table, support deletion, and are often more space efficient at low error rates. Scalable Bloom filters chain progressively larger filters so the capacity need not be known in advance. Quotient filters and the newer XOR and ribbon filters trade build time or mutability for better memory efficiency. Whichever flavor you pick, remember that the filter only answers membership; it cannot enumerate its contents or tell you how many items it holds with precision.
