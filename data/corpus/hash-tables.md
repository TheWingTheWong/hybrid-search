---
title: Hash Tables
tags: [data-structures, algorithms, performance]
year: 2021
---

# Hash Tables

A hash table maps keys to values with expected constant-time inserts, lookups, and deletes. It is the workhorse behind dictionaries in Python, maps in Go, HashMap in Java, symbol tables in compilers, and the in-memory index of countless caches. The idea is to turn a key into an array index with a hash function and store the entry in that slot, or near it. The interesting engineering lies in what to do when two keys land in the same place.

## Hash Functions

A good hash function spreads keys uniformly over the output range and is cheap to compute. For in-memory tables, fast non-cryptographic functions such as xxHash, wyhash, or MurmurHash are common. If attackers control the keys, for example HTTP header names, a predictable hash lets them craft many colliding keys and degrade the table to linear time, a hash-flooding denial of service. Keyed functions like SipHash, seeded randomly per process, defend against this, which is why several language runtimes use them by default.

## Collision Resolution

### Separate Chaining

Each bucket holds a linked list or small array of entries. Collisions just extend the chain. Chaining is forgiving of high load factors and easy to implement with deletion, but following pointers scatters memory accesses and hurts cache performance. Java's HashMap converts a long chain into a balanced tree once it exceeds eight entries to cap the worst case.

### Open Addressing

All entries live directly in the array. On collision, the table probes other slots in a fixed sequence. Linear probing checks the next slot, then the next; it is extremely cache-friendly but prone to primary clustering, where runs of occupied slots grow and lengthen probes. Quadratic probing and double hashing spread probes out. Robin Hood hashing moves entries so that no key sits much farther from its home slot than others, which keeps probe lengths tight. Deletion in open addressing needs care: either tombstone markers or backward-shift deletion to keep probe chains intact.

```
lookup(key):
  i = hash(key) % capacity
  while slots[i] is not EMPTY:
     if slots[i].key == key: return slots[i].value
     i = (i + 1) % capacity
  return NOT_FOUND
```

## Load Factor and Resizing

The load factor is entries divided by slots. Chained tables often resize around 0.75 to 1.0; open-addressing tables usually resize around 0.5 to 0.875, since probe lengths explode as the array fills. Resizing allocates a larger array, typically double, and rehashes every entry, an O(n) operation that is amortized to O(1) per insert. Latency-sensitive systems such as Redis resize incrementally, migrating a few buckets on each operation so that no single request pays the full cost.

## Modern Designs

Swiss tables, used in Abseil and Rust's standard library, store a byte of metadata per slot holding seven bits of the hash. A lookup loads 16 metadata bytes at once and compares them with SIMD instructions, filtering candidates before touching any keys. This gives excellent performance even at high load factors. Cuckoo hashing guarantees worst-case constant lookups by giving each key two possible positions and evicting occupants on conflict. For concurrent access, designs range from striped locks to lock-free split-ordered lists. Remember that hash tables do not keep keys in order; if you need range queries or sorted iteration, reach for a tree or a skip list instead.
