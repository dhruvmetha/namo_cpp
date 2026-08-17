---
type: experiment
status: running
created: 2026-08-17
thread: full-namo-multihop
robot: car
metric: exact preservation of keyhole-2 reachable contact edges across the committed keyhole-1 HY5U solution
tags: [experiment, full-namo, multihop, hy5u, lp1, reachability, amarel]
---

# Two-hop keyhole-independence audit

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U remains the ranker inside each simulator-verified local region-opening search; this audit tests whether composing two such searches preserves the second local problem.

## Question

Among the 197 HY5U scenes solved through exactly two keyhole attempts, how often does the first committed opening leave the second keyhole's object identity, object pose, and exact locally reachable contact-edge set unchanged?

## Definition

Before keyhole 1, object 2 is globally unreachable by construction. The audit therefore creates a separate shadow environment at the same physical state, seeds its robot wavefront from inside the middle region, and records object 2's reachable edge indices there. After HY5U commits keyhole 1 in the live environment, it compares the exact edge-index set, not only the count.

A scene passes only if the remaining path loses exactly one hop, the next boundary has the same object set, every next-keyhole object moved by at most 0.1 mm and 0.1 degrees, and every reachable-edge set is identical.

## Protocol

Rerun the registered HY5U protocol on the existing 197-scene exact-two-keyhole solved manifest: `hmax=2`, 300 simulator calls per keyhole reset independently, `1x_car_d5_`, raw `q`, discount off, and `/cache/home/dm1487/aquaman0/ckpts_bfix/HY5U_s2.ckpt`. The shadow audit is read-only with respect to the live search.

## Result

Pending Amarel smoke and scale run.
