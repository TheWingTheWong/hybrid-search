---
title: B-Trees
tags: [data-structures, storage, databases]
year: 2019
---

# B-Trees

The B-tree has been the default on-disk index for relational databases for decades. PostgreSQL, MySQL InnoDB, SQLite, and most file systems lean on some flavor of it. The structure is a balanced search tree whose nodes are sized to match a disk or memory page, so that every step down the tree costs at most one I/O. Because each node holds hundreds of keys, the tree is extremely shallow: four levels are enough to index billions of rows.

## Shape of the Tree

Each node is a page, often 4, 8, or 16 kilobytes. An internal node stores sorted separator keys and child pointers; a node with k keys has k+1 children. All leaves sit at the same depth, which is the balance guarantee. The branching factor, sometimes called fanout, is typically between 100 and 500 depending on key size. With a fanout of 200, three levels address 8 million leaf pages, and the top levels almost always stay cached in memory, so a lookup often needs just one or two physical reads.

### B+ Tree Variant

Nearly every database actually uses a B+ tree. Internal nodes hold only keys for routing, and all values or row pointers live in the leaves. Leaves are chained together with sibling pointers, so a range scan finds the starting leaf and then walks sideways without climbing back up the tree. Keeping values out of internal nodes also raises fanout.

## Insertions and Splits

To insert, descend to the correct leaf and place the key in sorted position. If the page overflows, split it in half, move the middle key up into the parent, and repeat if the parent overflows too. When the root splits, the tree gains a level. Deletions do the reverse: underfull nodes borrow from siblings or merge with them. Many engines relax the merge rule and tolerate partly empty pages, since workloads that delete often re-insert soon after.

```
insert(key, val):
  leaf = descend(root, key)
  leaf.insert_sorted(key, val)
  while node.overfull():
     left, sep, right = node.split()
     node = node.parent; node.insert_child(sep, right)
```

## Concurrency and Crash Safety

Many threads read and modify the tree at once, so engines use latch crabbing: grab a latch on a child before releasing the parent, and release ancestors early once a node is known to be safe from splitting. B-link trees add a right-link pointer and a high key per node so readers can recover from a concurrent split without heavy locking.

Because page splits touch several pages, a crash in the middle could leave the tree inconsistent. Databases protect against this by logging changes in a write-ahead log before modifying pages. Alternatively, copy-on-write B-trees, used in LMDB and Btrfs, never overwrite a page; they write new versions up to a new root and switch atomically.

## Compared With Log-Structured Designs

B-trees update in place, which gives stable read latency and efficient range scans but costs random writes and some write amplification when a small change dirties a whole page. Fragmentation leaves pages around 70 percent full on average after random inserts. LSM trees invert those trade-offs, absorbing writes faster at the expense of read work and background compaction. For read-heavy transactional workloads with secondary indexes, the B-tree remains the safe choice.
