---
type: experiment
status: done
created: 2026-08-22
commit: 8bae26f
metric: 5 accepted from 26 easy-easy pairs; 5/5 replayed
tags: [experiment, multihop, data, easy]
---
# Easy room-directed two-hop stitch

## Hypothesis

_(you, from chat)_ Try the simplest construction first: use simple easy one-push environments to make a stitched two-push scene, and place the robot in a room-like environment where two hops are possible.

## Plan

_(Codex)_ Start with two canonical easy one-push donors on the known productive three-room `set2/benchmark_5` wall template. First run the existing composition gate as a control. Then make endpoint placement explicit: robot in the first room, goal in the third room, K1 at the first boundary and K2 at the second, with the canonical pinned trace requiring `[closed, closed] → [open, closed] → [open, open]`. Render accepted scenes with K1/K2 labels before changing difficulty or scaling.

## Run

_(Codex)_ Ran locally on `ilab3` at commit `684eca3` using two canonical easy one-push donors on `set2/benchmark_5`, minimum blocker-center separation 0.30 m, at most 100 candidate pairs, and a target of five accepted scenes. Every output was required to have the exact initial region path `robot → middle room → goal`, boundary objects `[K1]` then `[K2]`, and a forward donor-action replay against the two pinned 100-point target components. Artifacts: `$NAMO_SCRATCH/eval/easy_room_stitch_20260822/control_easy_easy/`.

## Result

The run accepted five scenes after 26 candidates. The other 21 failed only the static room-chain check; all five statically valid scenes replayed, so dynamic continuation success was 5/5 with zero donor-action failures.

| scene | initial path | pinned-component trace | K1 action | K2 action |
|---|---|---|---|---|
| 0 | `robot → region_3 → goal` | `[0,0] → [100,0] → [100,80]` | `(9,2)` | `(1,1)` |
| 1 | `robot → region_3 → goal` | `[0,0] → [97,0] → [97,74]` | `(3,3)` | `(37,2)` |
| 2 | `robot → region_3 → goal` | `[0,0] → [100,1] → [100,94]` | `(44,1)` | `(30,0)` |
| 3 | `robot → region_3 → goal` | `[0,0] → [100,0] → [100,80]` | `(44,1)` | `(1,1)` |
| 4 | `robot → region_3 → goal` | `[0,0] → [100,0] → [100,87]` | `(32,1)` | `(0,1)` |

_(Codex)_ The representative rendered scene places the robot in the bottom endpoint room and the goal in the top endpoint room. The C++ region graph certifies that the shortest path crosses exactly `obstacle_0_movable` and then `obstacle_1_movable`; the middle component is explicit rather than inferred from Euclidean blocker distance. Three rendered environment/region/graph panels live under the thread visualization directory `easy-room-renders/`.

## Verdict

**ACCEPT the simple easy-room construction.** Two-hop composition is not intrinsically rare here: once an easy/easy pair forms the intended room chain, all five tested pairs preserve the second hop. The earlier difficulty came from sampling medium/hard blockers before choosing compatible doorway slots and endpoint rooms. The next generator should encode a directed room module—entry room, blocker doorway, exit room—and join K1's exit to K2's entry, rather than relying on rejection sampling over blocker centers.
