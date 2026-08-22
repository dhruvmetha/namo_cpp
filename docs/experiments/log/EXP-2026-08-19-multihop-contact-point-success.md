---
type: experiment
status: done
created: 2026-08-19
commit: 7193a93
metric: "REJECT: strict future-interface gate solved 157/194 versus control 159/194 and more than doubled mean simulator cost (101.5 versus 47.0 calls)"
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

_(Claude, auto)_ **Corrected smoke and pilot:** Commit `8c96a8c` changed final-keyhole acceptance to exact robot-goal reachability. Corrected smoke job `223098` used `[9, 900]`: the same preserving K1 was found and the duplicate K2 outer retry disappeared, but no exact-goal K2 action was found. Paired five-scene jobs were control `223110` and gate `223109`; raw outputs are `$NAMO_SCRATCH/eval/multihop_future_interface_20260821/pilot5_{control,gate}/shard_*/`.

_(Claude, auto)_ **Five-scene outcome:** Both arms solved 0/5. Control spent 1,530 total simulations and every one of its four committed K1 openings failed the old post-commit audit: K2 was moved 7–90 mm, lost contacts or changed blocker identity; the fifth scene had no K1 opening. The gate spent 3,828 simulations: one scene accepted a preserving K1 after eight rejections and then exhausted K2; two scenes rejected all 797 and 704 locally open K1 candidates; one rejected all 770 locally open K1 candidates; the fifth matched the control's no-K1 failure. This proves the strict invariant prevents destructive commits but is often infeasible and did not recover a solve in this sample.

_(Claude, auto)_ **Full corrected queue 2026-08-21:** Code commit `8c96a8c`, experiment record commit `3ff2ac4`; paired current-physics revalidation of all 197 formerly solved exact-two-keyhole scenes; manifest SHA-256 `7ef9b22df49d65059e50574297114f9856a543c49ed8502c39d890acaaf3d716`; 900 simulations per keyhole; control array `223319`, gated array `223320`; each arm capped at ten concurrent CS `unlimited` tasks, 45 minutes and 8 GB per shard. Outputs: `$NAMO_SCRATCH/eval/multihop_future_interface_20260821/full197/{control,gate}/shard_*/`.

_(Claude, auto)_ **Cancellation correction:** The 197-scene cohort was selected for successful two-keyhole execution before the physics fix. Recomputing a current two-hop graph does not re-establish current-physics solvability, so this population would confound future-interface preservation with physics-induced loss of solutions. Arrays `223319` and `223320` were cancelled after only 10 control and 7 gated shards had completed; partial outputs are retained as non-comparative diagnostics and must not be aggregated as an experiment result.

_(Claude, auto)_ **User-directed paired legacy-cohort completion:** The user correctly noted that the pre-fix provenance does not invalidate a same-population A/B under current physics, provided the result is framed as a legacy selected cohort rather than a current-solver success-rate estimate. After capping OpenCV and numerical-library thread pools in commit `7193a93`, a clean thread-cap smoke completed as job `223537`, and the missing shards completed on `rlab7` as control array `223572` and gated array `223574`. Both arms have 197/197 summaries, all final jobs exited zero, and the log scan found no OpenCV thread errors, tracebacks, OOMs or timeouts. Three scenes no longer formed an exact two-hop path under current physics, leaving 194 paired evaluated scenes. Raw outputs are `$NAMO_SCRATCH/eval/multihop_future_interface_20260821/full197/{control,gate}/shard_*/`; aggregates are the sibling `{control_aggregate,gate_aggregate}/` directories.

## Result + Verdict
_(Claude, auto from run output)_ **REJECT the strict all-contact invariant.** On the 194 current exact-two-hop scenes, control solved 159/194 (81.96%) and the gate solved 157/194 (80.93%), a -1.03 point delta. Paired outcomes were both 149, gate-only 8, control-only 10 and neither 27 (McNemar exact p=0.815). The gate improved strict audited K2-interface preservation among solved scenes from 121/159 (76.1%) to 133/157 (84.7%), but preservation did not translate into more complete solutions.

| template | n | control solved | gated solved | delta |
|---|---:|---:|---:|---:|
| `set1/benchmark_1` | 1 | 0 | 0 | 0 |
| `set1/benchmark_4` | 23 | 15 | 16 | +1 |
| `set2/benchmark_3` | 28 | 17 | 17 | 0 |
| `set2/benchmark_5` | 142 | 127 | 124 | -3 |
| **all** | **194** | **159** | **157** | **-2** |

_(Claude, auto)_ **The gate is overconstrained, not merely under-budgeted.** It checked 9,238 locally open K1 candidates and rejected 9,065: 9,048 lost at least one original K2 contact edge and 6,072 moved a K2 blocker beyond tolerance. It encountered at least one rejection on 38 scenes, eventually found an acceptable alternative on 24, and found none on 14. Nine of the ten control-only scenes were cheap control solves using 2–10 calls, while the gate rejected every locally open K1 candidate and exhausted all 900 K1 calls; the tenth accepted a gated K1 but later exhausted the remaining route. Thus the implementation does let K1 search continue and can recover alternatives—the eight gate-only solves are direct evidence—but requiring every original K2 edge plus an unchanged K2 pose rules out useful compositions.

| cost/result | control | gated |
|---|---:|---:|
| solved within 10 total simulator calls | 145 | 131 |
| median calls over all 194 scenes | 2 | 3 |
| mean calls over all 194 scenes | 47.0 | 101.5 |
| total calls | 9,114 | 19,697 |
| failure: goal region invalid | 4 | 1 |
| failure: region path exhausted | 26 | 20 |
| failure: simulation budget exhausted | 5 | 16 |

_(Claude, auto)_ **Interpretation:** The user's component-level invariant is still the right abstraction—after C1+C2 merge, retain a usable continuation from old C2 to K2—but exact preservation of the complete old action set is stronger than that invariant. The next test should preserve existence of a viable K2 continuation, such as at least one reachable K2 contact/action family or a cheap K2 feasibility probe, instead of all old contact indices. This cohort has only the composed exact-two-hop horizon and no registered easy/medium/hard labels, so no canonical difficulty×horizon table exists; the template split above is reported rather than inventing tiers. Because the 197 were selected before the physics fix, the paired A/B is valid while the absolute 81–82% solve rate is not a current-population estimate.

## Next
Replace exact all-edge preservation with a weaker existential K2-continuation gate, then rerun the same paired cohort before building a fresh current-physics population.

## Discussion
**[Claude 2026-08-17]** Assessment from the chat session, recorded here. Idea sound with one fix: the bar should be ≥1 (or ≥K) contact points, not all. What it fixes: the current 20% sampled-points bar is isotropic ("how much area opened"), while the chain needs anisotropic, task-directed opening ("did we open the path to the thing we push next") — exactly the commit-dead-end failure bucket. Risks flagged: (a) necessary-not-sufficient — a reachable contact point doesn't guarantee a successful push exists from it, so dead-ends shrink but don't vanish; (b) brittle corridors — the minimal condition can accept a squeeze-through that the next hop's push re-blocks; consider combining with a weak area floor; (c) a stricter accept test can burn more per-hop budget rejecting old-condition successes — net sims is empirical; (d) new success condition = new protocol — numbers not comparable to registered runs, and keyhole-1 difficulty tiers shift; register separately.
