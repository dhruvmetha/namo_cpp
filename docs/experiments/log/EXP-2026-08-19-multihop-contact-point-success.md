---
type: experiment
status: live
created: 2026-08-19
commit: 4f0c1e6
metric:
tags: [experiment, multihop, full_namo]
---
# Multi-hop: per-hop success = next blocker's contact points reachable

## Hypothesis
_(you, from chat 2026-08-17)_ **[USER]** In multi-hop (Full-NAMO), change the per-hop success condition: instead of "region opened" (20% of sampled next-region points reachable), a hop succeeds when the pushes make the robot able to reach the contact points of the NEXT object in the chain. Expectation: hop-i success becomes hop-i+1's precondition, so commit-dead-ends (accepting an opening that strands the next blocker) drop, and end-to-end solve rate under the same keyhole budget rises.

## Plan
_(Claude — revised 2026-08-21 after the component-interface clarification)_

- **Invariant:** Before opening K1, profile the original K2 blocker(s), poses, and every contact edge reachable with the robot placed inside C2. A K1 candidate is accepted only after it merges C1+C2 and every one of those original K2 contact edges remains reachable from C12; gained edges are allowed, and motion of a K2 blocker beyond 0.1 mm or 0.1° is rejected. If the candidate directly reaches the final robot goal, accept it as strictly greater progress.
- **Search behavior:** Apply the invariant inside best-first's `is_open` callback. A candidate that passes the old local C1+C2 opening bar but fails the future-interface check stays a failed pop, so the existing heap continues trying alternate one- or two-push K1 solutions under the same per-keyhole budget.
- **Pilot:** On the CS `unlimited` partition, run a one-scene end-to-end smoke, calibrate from measured simulator calls and wall time, then run paired control/gated exact-two-hop jobs on the same current-physics scene list and HY5U checkpoint. Primary readouts are end-to-end solve count, K1 future-interface rejection/acceptance counts, per-keyhole simulator calls, and terminal failure kinds.
- **Scope:** This first pilot protects the immediate future K2 interface only. It does not yet preserve every non-C1 adjacency formerly incident to C2; generalize only if the narrow mechanism improves end-to-end outcomes.

## Run
_(Claude, auto)_ —

## Result + Verdict
_(Claude, auto from run output)_ —

## Next
—

## Discussion
**[Claude 2026-08-17]** Assessment from the chat session, recorded here. Idea sound with one fix: the bar should be ≥1 (or ≥K) contact points, not all. What it fixes: the current 20% sampled-points bar is isotropic ("how much area opened"), while the chain needs anisotropic, task-directed opening ("did we open the path to the thing we push next") — exactly the commit-dead-end failure bucket. Risks flagged: (a) necessary-not-sufficient — a reachable contact point doesn't guarantee a successful push exists from it, so dead-ends shrink but don't vanish; (b) brittle corridors — the minimal condition can accept a squeeze-through that the next hop's push re-blocks; consider combining with a weak area floor; (c) a stricter accept test can burn more per-hop budget rejecting old-condition successes — net sims is empirical; (d) new success condition = new protocol — numbers not comparable to registered runs, and keyhole-1 difficulty tiers shift; register separately.
