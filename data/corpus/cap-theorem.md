---
title: The CAP Theorem
tags: [distributed-systems, consistency, theory]
year: 2019
---

# The CAP Theorem

The CAP theorem is probably the most quoted and most misquoted result in distributed systems. Informally it says that a replicated data store cannot simultaneously guarantee consistency, availability, and partition tolerance. The popular slogan "pick two of three" is catchy but misleading, and it is worth restating the claim carefully before using it to justify a design.

## What the Letters Actually Mean

- Consistency here means linearizability: every operation appears to take effect atomically at a single instant between its start and end, so all clients see one up-to-date copy. It is not the C in ACID, which is about application invariants.
- Availability means every request received by a non-failed node eventually gets a non-error response. It says nothing about latency, and a node that only answers "try later" does not count.
- Partition tolerance means the system keeps operating even when the network drops or delays arbitrary messages between nodes.

## The Real Choice

Networks partition. Switches fail, cables get unplugged, cloud availability zones lose connectivity, and long garbage-collection pauses look like partitions from the outside. So partition tolerance is not optional in any system that spans more than one machine. The actual statement is: during a partition, you must choose between consistency and availability.

A CP system refuses or delays some requests on the minority side of the split so that it never returns stale or conflicting data. Consensus-based stores like etcd, ZooKeeper, and Spanner fall here; a node cut off from the majority stops serving writes and linearizable reads.

An AP system keeps answering on both sides of the split and reconciles later. Dynamo-style databases such as Cassandra or Riak with low quorum settings behave this way, accepting divergent writes and repairing them through read repair, hinted handoff, or anti-entropy.

### A Short Proof Sketch

Put two replicas on either side of a partition. A client writes x=1 to the first. Another client reads x from the second. If the second must respond (availability), it cannot know about the new write, so it returns the old value and violates linearizability. If it must be linearizable, it has to wait or fail, violating availability.

## PACELC and Beyond

CAP only talks about behavior during partitions, which are relatively rare. PACELC extends it: if there is a Partition, choose Availability or Consistency; Else, during normal operation, choose Latency or Consistency. This captures the everyday trade-off better. A geo-replicated database that waits for a cross-region quorum on every write pays 50 to 150 milliseconds of latency even when the network is perfectly healthy.

## Practical Takeaways

Real systems rarely sit at a pure extreme. Many are tunable per request: Cassandra lets you choose consistency levels such as ONE, QUORUM, or ALL, and choosing R + W > N overlapping quorums gives you read-your-latest-write behavior at the cost of availability. Others offer weaker but still useful guarantees between the poles, like causal consistency, read-your-writes, or bounded staleness, which can stay available under partitions while ruling out the most confusing anomalies. When someone says a system is "CA", what they usually mean is that it runs on a single node or assumes partitions never happen, which is a reasonable choice only if that assumption actually holds.
