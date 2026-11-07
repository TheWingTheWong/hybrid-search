---
title: Paxos
tags: [consensus, distributed-systems, fault-tolerance]
year: 2019
---

# Paxos

Paxos is the classic answer to the question of how a group of unreliable processes can settle on a single value without a central coordinator. It tolerates crashed machines, lost or reordered messages, and arbitrary delays, as long as a majority of participants eventually respond. It does not handle malicious participants; that is the territory of Byzantine protocols. Paxos has a reputation for being hard to follow, mostly because the original description is terse and real deployments require many extensions on top of the core idea.

## The Single-Decree Protocol

There are three logical roles: proposers suggest values, acceptors vote on them, and learners find out what was chosen. One machine often plays all three. A value is chosen once a majority of acceptors have accepted it under the same proposal number.

### Phase 1: Prepare and Promise

A proposer picks a proposal number n that is unique and larger than any it has used, typically by combining a counter with its node id. It sends prepare(n) to the acceptors. An acceptor that has not already promised a higher number replies with a promise not to accept anything below n, and it includes the highest-numbered proposal it has already accepted, if any.

### Phase 2: Accept and Accepted

If the proposer collects promises from a majority, it must propose the value attached to the highest-numbered prior acceptance it heard about. Only if no acceptor reported an earlier acceptance is it free to use its own value. It then sends accept(n, v). Acceptors accept unless they have since promised a higher number.

```
acceptor state: promised_n, accepted_n, accepted_v
on prepare(n):  if n > promised_n: promised_n = n; reply(accepted_n, accepted_v)
on accept(n,v): if n >= promised_n: accepted_n = n; accepted_v = v; reply ok
```

## Why It Is Safe

The key insight is that any two majorities intersect. If some value v was chosen with number m, then any later proposer with a number above m must hear from at least one acceptor that accepted v, and is forced to carry v forward. So once a decision exists, it can never be replaced, even if the proposer that achieved it crashed before telling anyone.

## Multi-Paxos and Practical Concerns

Agreeing on one value is not very useful alone. Multi-Paxos runs many instances, one per slot of a replicated log. A stable leader runs phase 1 once for all future slots and then issues only phase 2 messages, reducing each decision to a single round trip to a majority. This makes it look a lot like Raft in steady state.

Liveness is the weak spot. Two proposers can keep preempting each other with ever higher numbers, a pattern often called dueling proposers, and nothing gets decided. Practical systems elect a distinguished proposer and use randomized backoff. The FLP impossibility result explains why no deterministic asynchronous protocol can guarantee progress in every execution, so timeouts are unavoidable.

Variants abound: Fast Paxos trims a message delay at the cost of larger quorums, Flexible Paxos observes that only phase 1 and phase 2 quorums need to intersect, and EPaxos removes the single leader to spread load across replicas. Systems such as Chubby and Spanner built their coordination layers on Paxos derivatives.
