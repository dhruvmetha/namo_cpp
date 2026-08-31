# v3_b2, whole pool, for the table

390 scenes, two movables each. Copy `v3_b2/` to
`/home/dhruv/projects_dhruv/namo/robot_control/real_exp/environments/` and the layout lines up.

## Per scene

```
v3_b2/<group>/<scene>/env.xml                            the scene, as the pool ships it
                     /build_sheet_derived.json           poses, for check_build.py
                     /sweep_record.json                  the raw label, every enumerated cell
                     /gallery_card__<object>.json        one per episode, when it has one
                     /gallery_replay__<object>.json      the recorded solution, when it replays
```

Coverage: 390 scenes, 388 with a sweep record, 224 with at least one gallery card, 309 cards and 309 replays in total. A scene has more than one card when more than one (pushed object, goal region) episode qualified, which is why the object name is in the filename. The 166 scenes with no card did not qualify for the gallery: either no push both opens the region and shoves the second block, or the goal region was already reachable at the root.

## build_sheet_derived.json is not a generator sheet

Its own `schema` field says `derived_from_env_xml`. This pool ships `env.xml` and nothing else, was never part of a handoff campaign, and has no tier CSV. Poses are read back out of the XML. `robot_start_m` is NOT in the file, the harness places the robot, so it comes from a simulator reset. All 390 have one.

## What the labels mean

Produced by `scripts/pipeline/exhaustive_hmax2.py`, which enumerates rather than searches. Every reachable (object, edge, depth) at the start state is executed, and any push that does not open the goal gets EVERY reachable follow-up tried from the resulting state until one opens or all fail. No frontier, no beam, no budget, so a `dead` cell means every option was actually simulated.

Success is at least 20 of the 100 poses sampled inside the goal region becoming reachable, and not having been before, sampled once at the root with seed 42. That is the region test, NOT the XML goal marker. `env.is_robot_goal_reachable()` asks a different question, disagrees on roughly a third of real-table scenes, and is not what any label here means.

`1push` means a single push opens it. `2push` means none does but some first push has a follow-up that does. Tiers come from the working fraction over enumerated pushes: hard below 5%, medium below 30%, easy at or above. A scene's tier is per episode, so the same room can be medium for one block and easy for the other.

## Two things that will bite

**Edge indices do not transfer without canonicalising yaw.** You plan on a camera capture, not on these files. A block placed 180 degrees from the sheet reads 180 off and occupies the same space, and both sides are right, but `contact_points` emits `(u, +hy)` then `(u, -hy)` alternating, so a 180 flip swaps face parity. Map your index onto ours with yaw mod 180 and `i XOR 1` within each 30-block when the yaws differ by ~180. Execution on the table is self-consistent either way; this only matters when comparing indices.

**The finish push often lands on the other block.** That is the point of this pool. A card names the object its episode is about, and the recorded solution may still finish on the neighbour.

## Which scenes are the hard ones

Cards carry `door_needs_both_blocks` and `has_route_around`. The interesting set is both-blocks true and route-around false: a doorway no single block opens, with no path to the goal that avoids it. Measured over the whole two-movable gallery with best-first at budget 900, those rooms exhaust the budget 13.8% of the time against 2.9% for rooms with the same kind of doorway and an alternative route. Same shape of scene, 5x the failure rate.
