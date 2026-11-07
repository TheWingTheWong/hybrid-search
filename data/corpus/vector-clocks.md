---
title: Vector Clocks
tags: [distributed-systems, causality, replication]
year: 2020
---

# Vector Clocks

When replicas accept writes independently, you eventually need to know whether one update happened after another or whether the two were made without knowledge of each other. Wall-clock timestamps cannot answer this reliably because machine clocks drift. A vector clock is a small piece of metadata that captures the happens-before relation exactly, letting a system tell causally ordered updates apart from truly concurrent ones.

## The Mechanism

Each node keeps an array of counters, one slot per node in the system. Node i maintains its vector V like this:

```
on local event:        V[i] += 1
on send message m:     V[i] += 1; attach copy of V to m
on receive message m:  for each k: V[k] = max(V[k], m.V[k]); V[i] += 1
```

Each counter answers the question: how many events from node k does this state know about? Merging with an elementwise maximum means a node absorbs everything its peer had seen.

## Comparing Two Clocks

Given vectors A and B, A happened before B if every entry of A is less than or equal to the matching entry of B and at least one is strictly smaller. If neither dominates the other, the events are concurrent. For example, with three nodes, [2,1,0] precedes [2,2,0], while [2,1,0] and [1,1,1] are concurrent because each is ahead in some slot. This is the crucial advantage over Lamport timestamps: a single scalar can show that A might precede B, but it can never prove two events are independent.

## Use in Replicated Stores

Dynamo-style key-value stores attach a version vector to every stored value. When a client reads, it receives the value plus its vector, and when it writes back, it sends that context along. A replica receiving a write can then decide:

- If the new version descends from the stored one, overwrite it.
- If the stored version descends from the new one, the write is stale and can be dropped.
- If they are concurrent, keep both as siblings and let the application or the next reader reconcile them.

The shopping-cart example is the classic illustration: two partitioned replicas each add an item, and on merge the cart becomes the union of both siblings. Version vectors are a close cousin of vector clocks that track replicas rather than every client or event, which keeps them smaller.

## Practical Limits

The obvious problem is size. A vector grows with the number of participants, and if clients rather than servers own slots, it can balloon. Systems prune old entries once they exceed a threshold, accepting that pruning can occasionally make causally related versions look concurrent and produce spurious siblings. Dotted version vectors refine the scheme so that concurrent writes through the same server are not mistakenly collapsed.

Another cost is that conflict resolution moves to the application. Developers must write merge logic or adopt structures designed to merge automatically, which is exactly the motivation for CRDTs. Finally, vector clocks capture causality only through messages the system observes; if two users coordinate through a side channel such as a phone call, the system will still consider their updates concurrent.
