---
title: Two-Phase Commit
tags: [transactions, distributed-systems, databases]
year: 2020
---

# Two-Phase Commit

Two-phase commit, usually shortened to 2PC, is the standard technique for making a transaction that touches several independent resources either take effect everywhere or nowhere. Think of moving money between accounts that live on different database shards, or writing to a database and a message broker as one unit. The protocol relies on a coordinator that drives the decision and a set of participants that each guard their own data.

## How the Protocol Runs

### Phase One: Voting

The coordinator sends a prepare request to every participant. Each participant does all the work required to guarantee it could commit later: it acquires locks, validates constraints, and forces its intended changes to durable storage, usually its write-ahead log. Then it votes yes. If anything fails, it votes no. A yes vote is a promise; after sending it, the participant gives up the right to abort on its own.

### Phase Two: Decision

If every vote is yes, the coordinator writes a commit record to its own log and tells everyone to commit. If any vote is no, or a participant times out during voting, it records abort and broadcasts that. Participants apply the outcome, release locks, and acknowledge. Once all acknowledgments arrive, the coordinator can forget the transaction.

```
coordinator:
  votes = [p.prepare(tx) for p in participants]
  decision = COMMIT if all(v == YES for v in votes) else ABORT
  log.force(decision)
  for p in participants: p.finish(tx, decision)
```

## The Blocking Problem

The weakness of 2PC appears when the coordinator crashes after participants have voted yes but before they learn the outcome. Those participants are now in doubt. They cannot commit, since someone else might have voted no, and they cannot abort, since the coordinator might have already told another participant to commit. So they hold their locks and wait for the coordinator to recover. In a busy system this can freeze hot rows for minutes. Operators sometimes resolve in-doubt transactions by hand, which is risky.

Three-phase commit adds a pre-commit round to avoid blocking, but it assumes bounded network delays and breaks under partitions, so it is rarely used. A more practical fix is to make the coordinator itself fault tolerant by storing its decision in a consensus-replicated log, as systems like Spanner do by layering 2PC over Paxos groups.

## Costs and Alternatives

Each transaction needs at least two round trips and several forced disk writes, and locks are held for the full duration, which hurts throughput and tail latency. Presumed-abort and presumed-commit optimizations reduce logging for the common case, and read-only participants can drop out after phase one.

Many service-oriented architectures avoid distributed transactions entirely. The saga pattern breaks a business operation into local steps, each with a compensating action that undoes it if a later step fails. The transactional outbox pattern writes an event into the same local database transaction as the state change, then relays it to a broker asynchronously. These approaches give up atomic isolation across services in exchange for availability and loose coupling, and they lean heavily on idempotent handlers.
