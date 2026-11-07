---
title: CRDTs
tags: [distributed-systems, replication, data-structures]
year: 2023
---

# CRDTs

Conflict-free replicated data types are data structures built so that replicas can be modified independently, without coordination, and still converge to the same state once they have exchanged updates. They power collaborative editors, offline-first mobile apps, and multi-region databases where waiting for a global lock is not acceptable. The promise is strong eventual consistency: any two replicas that have received the same set of updates are in identical states, no matter the order of delivery.

## Two Families

### State-Based (Convergent)

Each replica holds a full state and periodically ships it to peers. The receiver combines it with a merge function that must be commutative, associative, and idempotent. Mathematically, the states form a join semilattice and merge computes the least upper bound. Because merge is idempotent, duplicated or re-sent messages are harmless, which makes state-based CRDTs a good fit for gossip-style anti-entropy over unreliable networks. The downside is bandwidth, since full states can be large; delta-state CRDTs send only recent changes to soften this.

### Operation-Based (Commutative)

Replicas broadcast operations instead of states. Concurrent operations must commute, and the delivery layer usually has to guarantee exactly-once, causally ordered delivery. Messages are small, but the transport requirements are stricter.

## A Tour of Common Types

- G-Counter: a grow-only counter stored as one slot per replica. Increment your own slot; merge takes the per-slot maximum; the value is the sum.
- PN-Counter: two G-Counters, one for increments and one for decrements, with the value being their difference.
- LWW-Register: keeps the value with the highest timestamp. Simple, but it silently discards concurrent writes and depends on decent clocks.
- OR-Set (observed-remove set): each add carries a unique tag; a remove deletes only the tags it has seen. As a result, a concurrent add wins over a remove, which matches user intuition.
- Sequence CRDTs such as RGA or Logoot assign position identifiers to characters so that concurrent inserts in a text document interleave deterministically.

```
GCounter.merge(a, b):
  for r in replicas: out[r] = max(a[r], b[r])
GCounter.value(c): return sum(c[r] for r in replicas)
```

## Costs and Caveats

Metadata overhead is the main tax. Removing an element from an OR-Set leaves tombstones or causal context behind, and text CRDTs may store an identifier per character. Garbage collecting this history safely requires knowing that every replica has seen an operation, which reintroduces a form of coordination. Modern libraries such as Automerge and Yjs use compact encodings and run-length tricks to keep documents manageable.

The other caveat is semantic. Convergence does not mean the merged result makes sense for your application. Two users concurrently setting a booking's status to confirmed and cancelled will converge, but perhaps to a state neither intended. Invariants such as "balance never goes negative" cannot be enforced by a CRDT alone, because preserving them requires replicas to coordinate before acting. Teams often combine CRDTs for most data with a consensus-backed path for the few operations that need global constraints.
