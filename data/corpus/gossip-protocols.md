---
title: Gossip Protocols
tags: [distributed-systems, networking, membership]
year: 2022
---

# Gossip Protocols

Gossip, also called epidemic dissemination, spreads information through a cluster the way rumors move through an office. Every so often, each node picks a few random peers and tells them what it knows. No node needs a full view of the network, there is no central broadcaster to overload, and the scheme keeps working when a sizable fraction of machines are down. Cassandra, Consul, Serf, and many service meshes rely on gossip for membership, failure detection, and metadata propagation.

## How Information Spreads

In each round, a node selects k random peers, where k is the fanout, typically 2 or 3, and exchanges state with them. Because the number of informed nodes roughly multiplies each round, an update reaches the whole cluster in about log(N) rounds. With 1,000 nodes, a fanout of 3, and a one-second interval, nearly everyone hears an update within 7 or 8 seconds.

### Push, Pull, and Push-Pull

- Push: the informed node sends news to its targets. Fast at first, but slow to reach the last few stragglers.
- Pull: nodes ask peers for anything new. Slow at first, but very efficient at mopping up the tail.
- Push-pull: both directions in one exchange. Most production systems use this hybrid because it converges fastest overall.

## Anti-Entropy Versus Rumor Mongering

Two styles serve different goals. Anti-entropy has nodes compare their entire state with a partner and repair differences, often by exchanging digests or Merkle tree roots first so that only mismatched ranges travel over the wire. It guarantees eventual convergence but costs more bandwidth. Rumor mongering spreads only fresh updates and stops forwarding a rumor once it seems stale, for instance after a node has pushed it to several peers that already knew it. That is cheap, but some nodes may miss an update, so systems often run slow anti-entropy in the background as a safety net.

## Failure Detection with SWIM

Gossip is a natural vehicle for membership. The SWIM protocol has each node periodically probe a random member directly. If the probe is not acknowledged, the node asks a few other members to probe the target on its behalf, which separates a truly dead process from a single flaky link. Unreachable members are first marked suspect rather than dead, and the suspicion is gossiped. The suspect can refute it by gossiping a higher incarnation number. Only after a timeout does the cluster declare it failed.

```
every T ms:
  target = random_member()
  if not ping(target):
    if not any(indirect_ping(helper, target) for helper in sample(3)):
      mark_suspect(target); gossip(suspect, target, incarnation)
```

Accrual detectors such as the phi detector take a different angle, outputting a continuous suspicion level based on the history of heartbeat arrival times instead of a binary verdict.

## Trade-offs

Gossip gives probabilistic, not deterministic, guarantees. Convergence time grows slowly with cluster size, but message volume is constant per node, so total network chatter grows linearly. Large state payloads must be compressed or summarized. Gossip also cannot order events or provide agreement, so it pairs well with consensus systems rather than replacing them: a Raft cluster might hold the authoritative config while gossip handles the fast, fuzzy question of who is currently alive.
