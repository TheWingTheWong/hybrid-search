---
title: Raft Consensus
tags: [consensus, distributed-systems, replication]
year: 2021
---

# Raft Consensus

Raft is a protocol that lets a small cluster of servers agree on an ordered sequence of commands even when some of them crash or become unreachable. It was designed with understandability as an explicit goal, so it splits the problem into a few pieces that can be reasoned about separately: choosing a leader, copying the log, and keeping everything safe across failures. Most teams meet Raft through systems like etcd, Consul, or the metadata layer of a database, where a cluster of three or five nodes holds critical configuration.

## Roles and Terms

Every server is in one of three states: follower, candidate, or leader. Time is divided into numbered terms, and each term begins with an election. A term number acts like a logical epoch; any message carrying an older term is rejected, and a server that sees a newer term immediately steps down to follower. At most one leader can exist per term, which is the property that keeps the replicated state machine from diverging.

### Leader Election

Followers expect periodic heartbeats from the leader. If a follower hears nothing for its election timeout, it increments the term, votes for itself, and asks peers for votes. A node grants at most one vote per term and only to a candidate whose log is at least as up to date as its own. A candidate that gathers a majority becomes leader. Timeouts are randomized, often somewhere between 150 and 300 milliseconds, so that split votes are rare and resolve quickly.

## Log Replication

Clients send commands to the leader, which appends them to its log and ships them to followers with AppendEntries calls. Each entry records the term in which it was created and its index. A follower accepts a batch only if the entry just before it matches by index and term; otherwise it rejects, and the leader backs up and retries with earlier entries until the logs line up. Once an entry is stored on a majority, the leader marks it committed and applies it to its state machine.

```
on AppendEntries(term, prevIndex, prevTerm, entries, commit):
  if term < currentTerm: reject
  if log[prevIndex].term != prevTerm: reject
  truncate conflicting suffix, append entries
  commitIndex = min(commit, lastIndex)
```

## Safety Arguments

The election restriction is what makes Raft correct: since a winner needs votes from a majority, and any committed entry lives on a majority, the two groups overlap, so every new leader already holds all committed entries. A subtle rule says a leader only counts replicas for entries from its own term when advancing the commit index; older entries become committed indirectly. Skipping this rule leads to the classic scenario where an entry looks committed and is later overwritten.

## Operational Trade-offs

A five-node cluster tolerates two failures, a three-node cluster tolerates one. Adding more nodes increases fault tolerance but slows writes, because each commit waits for more acknowledgments. Logs grow without bound, so implementations take snapshots and truncate old entries. Membership changes are handled with joint consensus or single-server changes to avoid two disjoint majorities. Reads can be served by the leader after confirming it is still leader, either via a heartbeat round or by relying on a lease with bounded clock drift. Compared with Paxos, Raft is less flexible but much easier to implement and debug, which explains its popularity.
