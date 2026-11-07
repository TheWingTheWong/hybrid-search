---
title: Write-Ahead Logging
tags: [databases, storage, durability]
year: 2021
---

# Write-Ahead Logging

Write-ahead logging is the discipline that lets a database promise durability without flushing every modified page to disk on every commit. The rule is short: before a change to a data page reaches stable storage, a record describing that change must already be on stable storage in the log. With that invariant in place, the system can always reconstruct a correct state after a crash by replaying or undoing log records.

## Why Not Just Write the Pages?

Data pages are scattered across the disk, so persisting each modified page at commit time means lots of random I/O. A single transaction might touch an index page, a heap page, and a free-space map. The log, by contrast, is append-only and sequential. Appending a few hundred bytes to the tail of one file and calling fsync is far cheaper than scattering writes everywhere. Modified pages stay dirty in the buffer pool and get written lazily by a background writer or at checkpoint time.

## Log Records and LSNs

Every record gets a log sequence number, a monotonically increasing position in the log. Each data page stores the LSN of the last record that modified it, the pageLSN. Before the buffer manager evicts a dirty page, it ensures the log has been flushed at least up to that pageLSN. That is the write-ahead rule enforced mechanically.

Records can be physical (byte images of a page region), logical (insert key K into table T), or a mix called physiological: identify a physical page, describe the change logically within it. Physiological logging is the most common because it keeps records small while still being easy to replay page by page.

## Recovery in the ARIES Style

The ARIES algorithm, the template for most commercial engines, recovers in three passes:

1. Analysis: scan forward from the last checkpoint to rebuild the table of active transactions and the set of dirty pages.
2. Redo: repeat history by reapplying every logged change whose LSN is newer than the page's stored LSN, including changes from transactions that will later be rolled back.
3. Undo: roll back transactions that never committed, writing compensation log records so that a crash during recovery does not repeat work.

```
for rec in log from redo_start:
    page = fetch(rec.page_id)
    if page.lsn < rec.lsn:
        apply(rec, page); page.lsn = rec.lsn
```

Checkpoints bound the work: by recording which pages are dirty and which transactions are open, they let recovery start near the end of the log instead of at the beginning.

## Tuning and Trade-offs

### Group Commit

The cost of fsync dominates commit latency, often 0.1 to 10 milliseconds depending on the device. Group commit batches the log flushes of many concurrent transactions into one, raising throughput dramatically under load at the cost of a small delay for each commit.

### Durability Knobs

Many systems let you relax durability, for instance PostgreSQL's synchronous_commit setting or flushing the log once per second in MySQL. A crash may then lose the last fraction of a second of acknowledged writes, but the database stays internally consistent because the log ordering is preserved.

The same log also feeds replication. Streaming the WAL to followers keeps replicas byte-for-byte in sync, and change data capture tools read it to publish row changes to other systems.
