---
title: Skip Lists
tags: [data-structures, algorithms, concurrency]
year: 2020
---

# Skip Lists

A skip list is an ordered collection that supports search, insert, and delete in expected logarithmic time, much like a balanced binary tree, but it achieves balance through coin flips instead of rotations. The result is a structure that is short to implement, easy to reason about, and friendly to concurrent modification. Redis uses skip lists for sorted sets, and LevelDB and RocksDB use them for the in-memory memtable.

## Structure

Start with an ordinary sorted linked list. Searching it takes linear time because you must walk node by node. Now add an express lane: a second list containing roughly every other node, with pointers that skip over the ones in between. Add a third lane with every fourth node, and so on. A search starts at the top lane, moves right while the next key is smaller than the target, then drops down a level and repeats. Each level roughly halves the remaining distance, giving about log2(n) levels and O(log n) steps.

Rather than maintaining this perfect pattern, which would be expensive under updates, a skip list assigns each new node a random height. With probability p (commonly 1/2 or 1/4) a node is promoted one level higher, repeating until the coin comes up tails. The expected number of nodes at level i is n * p^i, which reproduces the shape probabilistically.

```
random_level():
  lvl = 1
  while random() < P and lvl < MAX_LEVEL: lvl += 1
  return lvl

search(key):
  x = head
  for i from top_level down to 1:
     while x.next[i] and x.next[i].key < key: x = x.next[i]
  x = x.next[1]
  return x if x and x.key == key else None
```

## Insert and Delete

To insert, perform a search while remembering the last node visited on each level; these are the predecessors. Pick a random height for the new node, then splice it in after each predecessor up to that height. Deletion is symmetric: find predecessors and unlink the node at each level. No rebalancing step exists, which is the main appeal compared with red-black or AVL trees.

## Performance Characteristics

### Space

With p = 1/2, each node has on average two forward pointers; with p = 1/4, about 1.33, trading slightly longer searches for less memory. A cap such as 32 levels comfortably handles billions of entries.

### Time

Operations are O(log n) in expectation and with high probability, although a deliberately unlucky sequence of random heights could make one search slow. Because the randomness comes from the structure rather than the input, adversarial key orders cannot force bad behavior, unlike a naive binary search tree fed sorted keys.

### Cache Behavior

Skip lists chase pointers across scattered nodes, so they tend to be slower than B-trees or sorted arrays for large in-memory datasets. Some implementations allocate nodes from an arena to improve locality.

## Concurrency

Skip lists shine when many threads share an ordered map. Since an insert only changes a few local pointers, lock-free versions can link a node bottom-up with compare-and-swap operations; the bottom level is the source of truth, and upper levels are hints that may lag briefly. Java's ConcurrentSkipListMap is a widely used example. A memtable benefits for the same reason: writers append concurrently while readers iterate in key order without blocking, and the sorted iteration makes flushing to an on-disk file a simple sequential scan.
