---
title: Consistent Hashing
tags: [distributed-systems, partitioning, caching]
year: 2019
---

# Consistent Hashing

Any system that spreads keys across a pool of servers has to answer a simple question: which machine owns this key? The naive answer, hash the key and take the result modulo the number of servers, works until the pool size changes. Going from 10 to 11 nodes with modulo placement moves roughly 90 percent of all keys, which for a cache means a sudden flood of misses and for a database means a massive data shuffle. Consistent hashing reduces that churn so that adding or removing one node relocates only about 1/N of the keys.

## The Ring

Picture the output space of a hash function, say 0 to 2^32 - 1, bent into a circle. Each server is hashed onto one or more points on that circle. To place a key, hash it and walk clockwise until you hit the first server point; that server owns the key. When a new server joins, it takes over only the arc between itself and its predecessor. When a server leaves, its arc falls to the next server clockwise. Everybody else is untouched.

```
def owner(key):
    h = hash(key)
    idx = bisect_right(sorted_points, h) % len(sorted_points)
    return point_to_node[sorted_points[idx]]
```

## Virtual Nodes

With a single point per server, the arcs are wildly uneven, and one machine may own several times the average share. The standard fix is to give each physical server many virtual nodes, often 100 to 256 points. Load then averages out, and when a machine dies, its many small arcs are absorbed by many different neighbors instead of dumping everything on one successor. Virtual nodes also make heterogeneous hardware easy: a beefier box simply gets more points.

## Replication and Real Systems

Dynamo, Cassandra, and Riak use the ring to define a preference list. A key is stored on the first owner and the next N-1 distinct physical nodes clockwise, which gives replica placement for free. Memcached client libraries such as ketama popularized the ring for cache sharding, and many CDNs and load balancers use it to keep a given client or URL pinned to the same backend so local caches stay warm.

## Alternatives and Trade-offs

### Rendezvous Hashing

Highest random weight hashing computes a score hash(key, server) for every server and picks the maximum. It needs no ring structure and achieves excellent balance, but lookup cost is linear in the number of servers, which is fine for dozens of nodes and less fine for thousands.

### Jump Consistent Hash

Jump hash maps a key to a bucket number in a handful of arithmetic steps with no memory overhead and near-perfect balance. Its limitation is that buckets are numbered sequentially, so you can only add or remove at the end, which suits sharded storage more than dynamic cache fleets.

### Bounded Loads

Even with good hashing, popular keys create hot spots. Consistent hashing with bounded loads caps each server at a factor such as 1.25 times the average and forwards overflow to the next server on the ring, trading a little locality for much flatter utilization.
