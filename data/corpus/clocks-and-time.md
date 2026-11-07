---
title: Clocks and Time in Distributed Systems
tags: [distributed-systems, time, synchronization]
year: 2024
---

# Clocks and Time in Distributed Systems

Every server has a notion of the current time, and nearly every one of them is a little wrong. Code that compares timestamps from different machines, expires a lock after a few seconds, or orders events by when they happened is quietly depending on how wrong. These notes cover the physical clocks we have, the logical clocks we build instead, and the hybrids that try to get the best of both.

## Physical Clocks and Their Flaws

A machine's time-of-day clock is driven by a quartz oscillator whose frequency varies with temperature and manufacturing tolerance. A typical drift rate is around 20 to 50 parts per million, which adds up to a couple of seconds per day if left uncorrected.

### NTP and Clock Skew

The Network Time Protocol periodically queries reference servers, estimates the round-trip delay, and nudges the local clock. Over a good LAN it keeps hosts within a millisecond or so; across the internet, tens of milliseconds is more realistic, and an asymmetric route can bias the estimate. The difference between two machines' clocks at the same instant is called skew. NTP may also step the clock backward, so wall-clock time is not monotonic. Operating systems therefore expose a separate monotonic clock for measuring durations, and timeouts should always use it. Leap seconds add yet another discontinuity; many large operators smear them over a day instead.

## Logical Clocks

### Lamport Timestamps

A Lamport clock is just a counter. Increment it on each local event, attach it to outgoing messages, and on receipt set it to max(local, received) + 1. If event a causes event b, then a's timestamp is smaller. The converse is not true, so Lamport clocks give a total order consistent with causality (tie-break by node id) but cannot detect concurrency; vector clocks are needed for that.

## Hybrid Logical Clocks

A hybrid logical clock pairs a physical component with a logical counter. Each node tracks the largest physical time it has seen, from its own clock or from any incoming message, and bumps the counter when the physical part does not advance. The result stays close to real time, remains monotonic, captures causality like a Lamport clock, and fits in 64 bits. CockroachDB and YugabyteDB use HLCs to order transactions without special hardware.

```
on send or local event:
  pt = now()
  if pt > l: l = pt; c = 0 else: c += 1
on receive (lm, cm):
  l_new = max(l, lm, now())
  if l_new == l and l_new == lm: c = max(c, cm) + 1
  elif l_new == l: c += 1
  elif l_new == lm: c = cm + 1
  else: c = 0
  l = l_new
```

## Bounded Uncertainty and Leases

Google's TrueTime takes a different path: GPS receivers and atomic clocks in each datacenter let the API return an interval [earliest, latest] guaranteed to contain true time, typically a few milliseconds wide. Spanner implements commit wait, pausing until the uncertainty has passed so that timestamp order matches real-time order, which yields externally consistent transactions.

Leases show why bounded skew matters even without fancy hardware. A leader holding a 10-second lease may serve reads locally without consulting a quorum, but only if it stops before any other node could believe the lease expired. Implementations subtract a safety margin for maximum drift and rely on monotonic timers. A long process pause, such as a stop-the-world GC, can still cause a node to act on an expired lease, which is why storage systems pair leases with fencing tokens that reject writes from stale holders.
