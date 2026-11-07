---
title: Rate Limiting
tags: [reliability, api, traffic-management]
year: 2022
---

# Rate Limiting

Rate limiting caps how many requests a client, user, API key, or IP address may make in a given period. It protects shared infrastructure from overload, keeps one noisy tenant from starving everyone else, blunts credential stuffing and scraping, and enforces commercial quotas. The interesting design questions are which algorithm to use, where to enforce it, and how to tell clients they have been throttled.

## Token Bucket

A token bucket holds up to B tokens and refills at a steady rate r tokens per second. Each request removes one token; if the bucket is empty, the request is rejected or delayed. The bucket size defines how big a burst is tolerated, and the refill rate defines the sustained throughput. With B = 20 and r = 5, a client can fire 20 requests instantly after being idle, then settle at 5 per second.

Implementations do not need a timer. Store the token count and the last refill timestamp, and compute the refill lazily on each request:

```
allow(now):
  tokens = min(B, tokens + (now - last) * r)
  last = now
  if tokens >= 1: tokens -= 1; return true
  return false
```

## Leaky Bucket

The leaky bucket models a queue that drains at a constant rate. Incoming requests join the queue; if it is full, they are dropped. Output is perfectly smooth, which suits shaping traffic toward a fragile downstream system, but bursts are delayed rather than served quickly. When implemented as a meter rather than a queue, it behaves almost identically to a token bucket, and the two names are often used loosely.

## Window-Based Counters

### Fixed Window

Count requests per calendar interval, for example per minute keyed by user and minute number. It is trivially cheap with an atomic increment and expiry in Redis. The flaw is the boundary effect: a client can send the full quota at 12:00:59 and again at 12:01:00, doubling the intended rate across a two-second span.

### Sliding Window Log

Keep a timestamp for every request in the trailing window and count them. This is exact, but memory grows with the quota, which is costly for limits like 10,000 per hour.

### Sliding Window Counter

A compromise: keep counts for the current and previous fixed windows and weight the previous one by how much of it still overlaps the trailing window. If 40 percent of the previous minute overlaps, the estimate is current + 0.4 * previous. It uses constant memory and is accurate enough for most APIs.

## Distributed Enforcement and Client Experience

With many gateway instances, limits must be shared. A central store such as Redis with atomic scripts gives accurate global counts but adds a network hop per request and a dependency that can fail; decide in advance whether to fail open or closed. Local per-instance limits avoid the hop but are only approximate, since a client hitting ten instances gets ten times the allowance. Hybrid designs sync local counters periodically.

Throttled responses should use HTTP 429 Too Many Requests, include a Retry-After header, and ideally expose remaining quota via headers so well-behaved clients can pace themselves. Layer limits: per-IP at the edge, per-key at the API gateway, and concurrency limits in front of expensive endpoints, since a request-per-second cap alone does not stop a few very slow queries from exhausting a database connection pool.
