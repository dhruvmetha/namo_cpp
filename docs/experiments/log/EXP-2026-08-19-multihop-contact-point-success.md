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
- **Final-keyhole correction:** When the remaining path is the final K2 boundary, keep the sampled goal-region points as ranker input but replace the 20%-of-region terminal bar with exact `is_robot_goal_reachable()`. This keeps alternative K2 candidates in the same heap until the actual task goal is reachable and prevents a false local success followed by an identical outer retry.
- **Pilot:** On the CS `unlimited` partition, run a one-scene end-to-end smoke, calibrate from measured simulator calls and wall time, then run paired control/gated exact-two-hop jobs on the same current-physics scene list and HY5U checkpoint. Primary readouts are end-to-end solve count, K1 future-interface rejection/acceptance counts, per-keyhole simulator calls, and terminal failure kinds.
- **Scope:** This first pilot protects the immediate future K2 interface only. It does not yet preserve every non-C1 adjacency formerly incident to C2; generalize only if the narrow mechanism improves end-to-end outcomes.

## Run
_(Claude, auto)_ **CS smoke 2026-08-21:** Commit `4f0c1e6`; `unlimited` partition; one current-physics exact-two-hop scene selected from the first four entries of `$NAMO_SCRATCH/multihop_aug9_hy5u/scale_20260817_0000/static_probe/surviving_xmls_cspaths.txt`; HY5U seed-2 epoch-11 checkpoint; best-first hmax 2; 900 simulations per keyhole; one worker; model CPU. Gated job `223027` ran on `rlab7` for 5m26s; paired control `223097` ran on `ilab3` for 14s. Raw outputs: `$NAMO_SCRATCH/eval/multihop_future_interface_20260821/{smoke_gate,smoke_control}/shard_0000/`.

_(Claude, auto)_ **Smoke diagnosis:** Control K1 accepted its first candidate after one simulation, moved the K2 blocker 7.403 mm / 5.5862°, and terminated `goal_region_invalid`. Gated K1 rejected eight locally open candidates for `next_blocker_moved`, accepted candidate nine with zero lost original K2 edges and only 0.0401 mm / 0.0206° blocker drift, and reduced the remaining path from two hops to one. K2 then produced a false one-simulation local opening while the exact robot goal remained unreachable; the outer planner retried the same K2 and exhausted 900 simulations. Total gated usage was `[9, 1, 900]`, proving the long smoke was the final-keyhole false-positive/retry failure rather than K1 search cost.

## Result + Verdict
_(Claude, auto from run output)_ —

## Next
—

## Discussion
**[Claude 2026-08-17]** Assessment from the chat session, recorded here. Idea sound with one fix: the bar should be ≥1 (or ≥K) contact points, not all. What it fixes: the current 20% sampled-points bar is isotropic ("how much area opened"), while the chain needs anisotropic, task-directed opening ("did we open the path to the thing we push next") — exactly the commit-dead-end failure bucket. Risks flagged: (a) necessary-not-sufficient — a reachable contact point doesn't guarantee a successful push exists from it, so dead-ends shrink but don't vanish; (b) brittle corridors — the minimal condition can accept a squeeze-through that the next hop's push re-blocks; consider combining with a weak area floor; (c) a stricter accept test can burn more per-hop budget rejecting old-condition successes — net sims is empirical; (d) new success condition = new protocol — numbers not comparable to registered runs, and keyhole-1 difficulty tiers shift; register separately.
