---
title: LSM Trees
tags: [storage, databases, data-structures]
year: 2022
---

# LSM Trees

The log-structured merge tree is the storage design behind LevelDB, RocksDB, Cassandra, ScyllaDB, HBase, and many time-series engines. Its central idea is to never modify data in place on disk. Instead, writes are buffered in memory, flushed as immutable sorted files, and later merged in the background. This converts random writes into large sequential ones, which is exactly what both spinning disks and flash devices handle best.

## Write Path

A write first goes to a write-ahead log for durability and then into the memtable, an in-memory sorted structure usually implemented as a skip list. When the memtable reaches a threshold, commonly 64 megabytes, it is frozen and a fresh one takes over. A background thread writes the frozen memtable to disk as an SSTable (sorted string table), a file of key-value pairs in key order, with a sparse index and a Bloom filter at the end. Deletes are just writes of a special marker called a tombstone, which shadows older values until compaction removes both.

## Read Path

A lookup checks the active memtable, then any frozen memtables, then the on-disk files from newest to oldest, stopping at the first match. Without help this would touch many files, so each SSTable's Bloom filter is consulted first to skip files that certainly lack the key, and a block cache keeps hot index and data blocks in memory. Range scans have to merge iterators across all levels, which is why they tend to be slower than in a B-tree.

## Compaction Strategies

Compaction merges SSTables, discards overwritten values and expired tombstones, and keeps the number of files a read must consult under control. The policy determines most of the engine's performance profile.

### Leveled Compaction

Files are organized into levels L0, L1, L2, and so on, each roughly ten times larger than the previous. Within a level beyond L0, key ranges do not overlap, so a point read touches at most one file per level. The cost is high write amplification: a single byte may be rewritten 10 to 30 times as it trickles down.

### Size-Tiered Compaction

Files of similar size are grouped and merged together once enough accumulate, often four. Write amplification is much lower, but several overlapping files may hold the same key, so reads and space usage suffer; temporary space during a big merge can double the dataset.

## Amplification Trade-offs

Engineers describe LSM tuning in terms of three amplification factors: write amplification (bytes written to disk per byte written by the user), read amplification (I/Os per lookup), and space amplification (disk bytes per live byte). You can improve any two at the expense of the third. Leveled favors reads and space, tiered favors writes. Hybrid approaches such as tiered L0 with leveled lower levels, or universal compaction in RocksDB, try to land in between.

```
put(k, v):
  wal.append(k, v)
  memtable.insert(k, v)
  if memtable.size > LIMIT: schedule_flush(memtable)
```

Operationally, watch for write stalls: if compaction cannot keep up, L0 file counts climb and the engine deliberately throttles incoming writes. Compared with B-trees, LSM trees win on ingest throughput and compression, while B-trees usually deliver more predictable read latency.
