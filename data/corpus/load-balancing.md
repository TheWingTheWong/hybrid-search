---
title: Load Balancing
tags: [networking, reliability, scalability]
year: 2021
---

# Load Balancing

A load balancer spreads incoming work across a pool of backend servers so that no single instance is overwhelmed and so that failed instances can be removed without users noticing. It is one of the first components introduced when a service grows beyond one machine, and it shows up at several layers: DNS, network edge, service mesh sidecars, and client libraries.

## Layer 4 Versus Layer 7

A layer 4 balancer works with TCP or UDP connections. It picks a backend when a connection opens and forwards packets without inspecting their contents. This is fast, protocol-agnostic, and can handle millions of connections per second, especially with kernel-bypass or direct server return, where responses skip the balancer entirely.

A layer 7 balancer terminates the connection and understands the application protocol, typically HTTP. It can route by path or header, terminate TLS, retry failed requests, add tracing headers, and balance individual requests rather than whole connections. That last point matters for HTTP/2 and gRPC, where one long-lived connection carries many requests; an L4 balancer would pin all of them to a single backend.

## Balancing Algorithms

### Static Policies

- Round robin cycles through backends in order. Simple and fair when requests cost about the same.
- Weighted round robin sends proportionally more traffic to larger machines.
- Hashing on client IP or a session key keeps a user on the same server, useful for local caches; consistent hashing limits reshuffling when the pool changes.

### Dynamic Policies

- Least connections routes to the backend with the fewest active connections, adapting to uneven request costs.
- Least response time or peak EWMA favors servers that have recently answered quickly.
- Power of two choices samples two random backends and picks the less loaded one. It achieves nearly the balance of checking every server with a fraction of the coordination, and it avoids herding where many balancers simultaneously pile onto the single least-loaded host.

```
pick():
  a, b = random.sample(backends, 2)
  return a if a.inflight <= b.inflight else b
```

## Health Checking

Balancers must know which backends are alive. Active health checks probe an endpoint, for example GET /healthz every 5 seconds, ejecting a server after 3 consecutive failures and readmitting it after 2 successes. Passive checks, sometimes called outlier detection, watch real traffic and eject hosts with elevated error rates or latency. A good health endpoint verifies that the process can actually serve, but should not fail just because a shared dependency is down, or the balancer may eject the entire fleet at once. Connection draining lets in-flight requests finish when a host is removed during a deploy.

## Topologies and Pitfalls

Centralized balancers are easy to operate but are themselves a bottleneck and a single point of failure, so they run in redundant pairs with a floating IP or behind anycast. Client-side load balancing, common in gRPC and service meshes, removes the extra hop by letting each caller pick a backend from a service registry. Global server load balancing steers users to the nearest healthy region using DNS or anycast.

Common pitfalls include sticky sessions that concentrate load, slow-start problems where a freshly added instance receives a full share before its caches and JIT are warm, and retry storms where the balancer amplifies traffic toward a struggling pool. Pair balancing with sensible retry budgets and circuit breaking.
