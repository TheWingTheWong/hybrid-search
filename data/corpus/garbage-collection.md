---
title: Garbage Collection
tags: [runtime, memory, performance]
year: 2023
---

# Garbage Collection

Garbage collection is automatic memory reclamation: the runtime figures out which heap objects the program can no longer reach and frees them, so developers do not call free by hand. It removes whole classes of bugs, such as use-after-free and double free, but it costs CPU, memory headroom, and sometimes latency. Understanding how collectors work is essential for tuning services in Java, Go, C#, JavaScript, and similar managed languages.

## Finding Live Objects

### Reference Counting

Each object keeps a count of incoming references. When the count drops to zero, the object is freed immediately. This gives predictable, incremental reclamation and is used by Python, Swift, and smart pointers in C++. Its weaknesses are the overhead of updating counts on every pointer write and the inability to reclaim cycles without an auxiliary cycle detector.

### Tracing

Tracing collectors start from roots (stack slots, registers, globals) and follow pointers to mark everything reachable. Anything left unmarked is garbage. Mark-sweep then walks the heap and adds dead objects to free lists, which is simple but leaves fragmentation. Mark-compact slides live objects together, eliminating fragmentation at the cost of updating every pointer. Copying collectors evacuate live objects from one space into another, which makes allocation a trivial pointer bump but needs twice the space for the region being collected.

```
mark(root):
  if root.marked: return
  root.marked = true
  for child in root.references: mark(child)
```

## Generational Collection

The weak generational hypothesis observes that most objects die young: temporary strings, iterators, request-scoped buffers. Generational collectors split the heap into a young generation and an old one. The young space is collected often with a fast copying pass that only touches survivors, usually a small fraction. Objects that survive several cycles are promoted to the old generation, which is collected rarely. To avoid scanning the entire old space during a minor collection, a write barrier records old-to-young pointers in a remembered set or card table.

## Pauses and Concurrent Collectors

The simplest collectors stop all application threads for the whole cycle, a stop-the-world pause that grows with heap size. On a 32 gigabyte heap, a full compacting pause can take seconds, which looks like an outage to clients and can even cause a node to lose a lease or be declared dead by its peers.

Modern collectors do most of the work concurrently with the application. Java's G1 divides the heap into regions and targets a pause-time goal, for example 200 milliseconds. ZGC and Shenandoah use load barriers and colored pointers or forwarding pointers to relocate objects while threads run, keeping pauses under a millisecond even on terabyte heaps. Go's collector is a concurrent, non-moving tri-color mark-sweep tuned for low latency, adjusted with the GOGC setting that controls how much the heap may grow before the next cycle.

## Tuning Trade-offs

Every collector balances three resources: throughput (fraction of CPU spent on the application), latency (pause length), and footprint (extra memory). Giving the collector more headroom generally means fewer cycles and better throughput. Reducing allocation rate is usually the most effective optimization of all: object pools, reusing buffers, and avoiding boxing cut GC work at the source. When profiling tail latency, always check GC logs before blaming the network.
