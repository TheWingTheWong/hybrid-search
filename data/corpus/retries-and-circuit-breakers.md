---
title: Retries and Circuit Breakers
tags: [reliability, resilience, distributed-systems]
year: 2023
---

# Retries and Circuit Breakers

Remote calls fail. Packets drop, a pod restarts mid-request, a dependency briefly runs out of connections. Retrying is the obvious remedy for transient faults, but careless retries are also one of the most common causes of cascading outages. These notes cover how to retry safely and how to stop calling a dependency that is clearly unwell.

## When Is It Safe to Retry?

Only retry errors that are plausibly transient: timeouts, connection resets, HTTP 503 or 429, and similar. Do not retry validation errors or 4xx responses that will fail identically every time.

### Idempotency

A request is idempotent if performing it twice has the same effect as performing it once. GET, PUT, and DELETE are idempotent by design; POST usually is not. The danger is the ambiguous failure: a timeout does not tell you whether the server processed the request. Retrying a non-idempotent payment could charge the customer twice. The standard fix is an idempotency key: the client generates a unique id per logical operation and sends it with every attempt, and the server stores the result keyed by that id, returning the saved response for duplicates instead of executing again.

## Exponential Backoff with Jitter

Retrying immediately just hammers a struggling service. Exponential backoff waits progressively longer: for example 100 ms, 200 ms, 400 ms, 800 ms, capped at a few seconds, with a limit of three to five attempts. But if a thousand clients fail at the same moment, pure exponential backoff keeps them synchronized, and they return in waves. Adding randomness, called jitter, spreads them out.

```
delay(attempt):
  ceiling = min(MAX_DELAY, BASE * 2 ** attempt)
  return random_uniform(0, ceiling)      # "full jitter"
```

Full jitter, picking uniformly between zero and the ceiling, tends to give the lowest total load. Also respect any Retry-After header the server provides.

### Retry Amplification

Retries multiply across layers. If each of four tiers retries three times, a single failing database call can turn into 3^4 = 81 attempts. Retry at one layer only, usually closest to the user or the edge, and use a retry budget, for instance allowing retries to be at most 10 percent of total requests, so that during a real outage retries cannot double the load.

## Circuit Breakers

A circuit breaker sits in front of a dependency and tracks recent failures. It borrows its name from electrical breakers that trip to prevent a fire.

### The Three States

- Closed: calls flow normally while the breaker counts failures in a rolling window.
- Open: once the failure rate crosses a threshold, say 50 percent over at least 20 calls, the breaker trips. Calls fail immediately without touching the dependency, giving it room to recover and freeing the caller's threads.
- Half-open: after a cooldown, perhaps 30 seconds, a limited number of trial requests are let through. If they succeed, the breaker closes; if they fail, it opens again and the cooldown restarts.

Fast failure is the point. Without a breaker, callers pile up waiting on timeouts, exhaust thread pools and connection pools, and the outage spreads upstream.

## Putting It Together

Set timeouts on every call, shorter than the caller's own deadline, and propagate deadlines downstream so work is abandoned once nobody is waiting. Combine bounded, jittered retries for transient errors with a breaker for sustained ones, and provide a fallback such as cached data or a degraded response when the breaker is open. Bulkheads, separate pools per dependency, keep one slow service from consuming every worker. Test all of it with fault injection, because these paths rarely run until the day they matter.
