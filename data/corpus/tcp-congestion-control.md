---
title: TCP Congestion Control
tags: [networking, protocols, performance]
year: 2020
---

# TCP Congestion Control

TCP has two separate throttles on how fast a sender may transmit. Flow control protects the receiver: the advertised receive window says how much buffer space is left. Congestion control protects the network itself: the sender maintains a congestion window, cwnd, that limits how many unacknowledged bytes may be in flight. The effective sending limit is the smaller of the two. Without congestion control, the internet of the 1980s experienced congestion collapse, where links were busy but almost all traffic was retransmissions.

## Slow Start and Congestion Avoidance

A new connection does not know the path capacity, so it probes. Modern stacks begin with an initial window of 10 segments, roughly 14 kilobytes. During slow start, cwnd grows by one segment per acknowledgment, which doubles it every round trip. That exponential ramp continues until cwnd reaches the slow start threshold (ssthresh) or a loss occurs.

Beyond ssthresh, the sender enters congestion avoidance and increases cwnd by about one segment per round trip. This additive increase, combined with multiplicative decrease on loss, is the famous AIMD pattern that produces the sawtooth throughput graph and lets competing flows converge toward a fair share.

```
on ack:
  if cwnd < ssthresh: cwnd += MSS            # slow start
  else:               cwnd += MSS * MSS / cwnd  # avoidance
on triple duplicate ack:
  ssthresh = cwnd / 2; cwnd = ssthresh       # fast recovery
on timeout:
  ssthresh = cwnd / 2; cwnd = 1 MSS          # restart slow start
```

## Detecting and Reacting to Loss

Classic TCP treats packet loss as the signal of congestion. Three duplicate acknowledgments mean a segment was probably dropped while later ones arrived, so the sender performs fast retransmit and fast recovery, halving the window rather than starting over. A retransmission timeout is a stronger signal that the pipe has drained, and the window collapses to one segment. Selective acknowledgments (SACK) tell the sender exactly which ranges arrived, making recovery from multiple losses in one window much faster.

## Modern Algorithms

### CUBIC

CUBIC, the default in Linux for many years, grows the window as a cubic function of time since the last loss instead of linearly per round trip. It ramps quickly toward the previous maximum, plateaus near it, then probes beyond. Because growth depends on elapsed time rather than RTT, long-distance flows are not as disadvantaged as with Reno.

### BBR

BBR abandons loss as the primary signal. It continuously estimates the bottleneck bandwidth and the minimum round-trip time, then paces packets at the estimated rate with a window around the bandwidth-delay product. On paths with shallow buffers or random wireless loss, BBR can achieve many times the throughput of loss-based schemes. Early versions were criticized for unfairness toward CUBIC flows and for causing high retransmission rates; later revisions address some of this.

## Practical Implications

Bufferbloat, the presence of oversized buffers in routers and modems, lets loss-based algorithms fill queues before any drop occurs, adding hundreds of milliseconds of latency. Active queue management such as CoDel and fq_codel, along with ECN marking, signals congestion before buffers overflow. For application engineers, the main lessons are that new connections start slow, so reusing connections and keeping them warm matters, and that a high bandwidth-delay product link needs large socket buffers to reach full speed. QUIC reimplements these same algorithms in user space, which makes experimenting with new controllers much easier.
