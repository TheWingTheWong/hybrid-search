---
title: Merkle Trees
tags: [data-structures, cryptography, replication]
year: 2020
---

# Merkle Trees

A Merkle tree, or hash tree, summarizes a large collection of data with a single short fingerprint while still allowing efficient verification of any individual piece. Leaves hold hashes of data blocks, and each internal node holds the hash of its children concatenated together. The value at the top, the root hash, changes if any byte anywhere in the data changes. This simple construction powers integrity checks in Git, BitTorrent, IPFS, certificate transparency logs, blockchains, and replica repair in distributed databases.

## Construction

Split the data into blocks, for example 4 kilobyte chunks or key ranges. Hash each block with a collision-resistant function such as SHA-256 to form the leaves. Pair up adjacent leaves and hash each pair to form the next level. Repeat until a single root remains. With n leaves, the tree has about log2(n) levels.

```
build(blocks):
  level = [H(b) for b in blocks]
  while len(level) > 1:
     if len(level) % 2: level.append(level[-1])
     level = [H(level[i] + level[i+1]) for i in range(0, len(level), 2)]
  return level[0]
```

Real implementations add domain separation, for example prefixing leaf hashes with a 0x00 byte and internal hashes with 0x01, to prevent an attacker from passing off an internal node as a leaf.

## Inclusion Proofs

Suppose a client trusts only the root hash and wants to confirm that one block belongs to the dataset. The server sends the block plus the sibling hash at each level on the path to the root, called an audit path. The client recomputes upward and checks that it arrives at the trusted root. For a million blocks, that is only 20 hashes, a few hundred bytes, rather than downloading everything. Light clients in blockchains verify transactions this way, and certificate transparency logs offer both inclusion proofs and consistency proofs showing that a newer log is an append-only extension of an older one.

## Replica Synchronization

Distributed stores such as Cassandra and Riak, inspired by Dynamo, use Merkle trees for anti-entropy repair. Each replica builds a tree over its key ranges. Two replicas exchange root hashes; if they match, the ranges are identical and nothing more is sent. If not, they compare children and descend only into subtrees that differ, quickly isolating the handful of ranges that actually diverged. Network transfer scales with the amount of difference rather than the size of the dataset.

### Practical Wrinkles

Trees must be rebuilt or incrementally maintained as data changes, which costs CPU and disk reads; Cassandra historically built them during scheduled repair runs, which could be expensive on large nodes. Granularity matters too: a leaf covering too large a key range forces big transfers when one key differs, while very fine leaves make the tree itself large.

## Content Addressing and Beyond

Git stores every file and directory as an object named by its hash, and a commit hash covers the whole tree beneath it, so two repositories can compare histories by exchanging a few ids. Backup tools and file sync services split files into content-defined chunks and use Merkle structures to upload only changed chunks. Variants include Merkle Patricia tries, which combine hashing with a prefix tree to support efficient key lookups and proofs of absence, and sparse Merkle trees, which provide proofs of non-membership over enormous key spaces.
