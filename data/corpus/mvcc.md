---
title: Multiversion Concurrency Control
tags: [databases, transactions, concurrency]
year: 2022
---

# Multiversion Concurrency Control

Multiversion concurrency control, or MVCC, lets readers and writers stop blocking each other. Instead of overwriting a row in place and forcing readers to wait for a lock, the database keeps several versions of each row and shows each transaction the version that matches its point in time. PostgreSQL, MySQL InnoDB, Oracle, SQL Server in snapshot mode, CockroachDB, and many newer engines use some form of it.

## Versions and Visibility

Every transaction receives an identifier or timestamp when it starts. Each row version records which transaction created it and, once replaced or deleted, which transaction ended it. When a transaction reads, it consults its snapshot: a description of which transactions had committed at the moment it began. A version is visible if its creator committed before the snapshot and its deleter either does not exist or had not committed before the snapshot.

```
visible(version, snap):
  return committed_before(version.xmin, snap) and
         not committed_before(version.xmax, snap)
```

Because readers only look at already-committed history, they never take locks and never wait for writers. A long analytical report can run alongside heavy updates and still see a perfectly consistent picture.

## Storage Layouts

### Append-Only Versions in the Table

PostgreSQL writes each new version as a fresh tuple in the heap and marks the old one dead. Indexes may point to several versions, and readers follow them to find the visible one. This is simple but bloats tables when rows are updated frequently.

### Undo Logs

InnoDB and Oracle update the row in place and keep the previous image in an undo segment. Readers that need older data reconstruct it by walking backward through undo records. The main table stays compact, but very old snapshots can make readers chase long version chains.

## Isolation Levels

MVCC naturally provides snapshot isolation: all reads in a transaction come from the same snapshot, and two transactions that write the same row conflict, with the later committer aborting under first-committer-wins. Snapshot isolation prevents dirty reads, non-repeatable reads, and most phantoms, but it permits write skew. The classic example is two doctors each checking that someone else is on call and then both going off call; each transaction read a consistent snapshot, neither wrote the other's row, and the invariant breaks. Serializable snapshot isolation tracks read-write dependencies between concurrent transactions and aborts one when a dangerous cycle appears, giving full serializability with modest overhead.

## Cleaning Up Old Versions

Old versions must eventually be reclaimed once no active snapshot can see them. PostgreSQL does this with VACUUM, which scans for dead tuples and frees space; autovacuum runs it automatically based on thresholds such as 20 percent of a table changing. InnoDB has a purge thread that trims undo logs. The danger in both is a forgotten long-running transaction or an idle session holding an old snapshot: it pins every version created since, bloating storage and slowing queries. PostgreSQL additionally uses 32-bit transaction ids, so vacuum must periodically freeze old rows to avoid wraparound. Monitoring the age of the oldest open snapshot is one of the most useful health checks for any MVCC database.
