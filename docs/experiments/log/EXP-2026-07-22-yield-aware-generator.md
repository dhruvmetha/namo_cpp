---
status: parked
thread: rl_loop
robot: car
updated: 2026-07-22
parent: EXP-2026-07-21-colossus-data-scaleup
---

# EXP-2026-07-22 — Yield-aware scene generator (single-hop, mix-controllable)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** An episode = robot region + goal region + the ONE movable between them; the ranker orders the pushes on that object. A scene is only usable if the wavefront makes a distinct `goal` region **immediately adjacent** to the robot region with a movable on the boundary. This card is about generating scenes that actually satisfy that.

## The one sentence

Build a pure-XML generator that produces **valid single-hop region-opening scenes at high yield, biased openable** (goal immediately adjacent to the robot region, one movable in the doorway that CAN merge them in 1-2 pushes) — so we can grow the **positive base** (openers/setups/true-2push), which no pre-screened source can give us.

## Scope (dead is NOT a target) [USER]

We already have plenty of dead (bank 234k + 59k leftover + colossus ~40k). **Do not generate dead.** The generator exists to make **positives** — the gap the dead-biased bank cannot fill (colossus grew positives only +4%). Dead scenes that fall out of the physics label pass are a free bonus, not the goal. So: one goal — **high-yield openable scenes**, with a difficulty knob for the scarce hard tier.

## Why (measured, not theorized)

Colossus round-0 (EXP-2026-07-21) labeled 175k `collect2/gen` pair scenes through region_opening and only **~29% were usable** (dead-root or solved). The other ~71% produce **zero training boards** — the planner correctly rejects them as out-of-scope:

| outcome | share (404 live boards) | usable? |
|---|---|---|
| `goal_region_not_in_snapshot` (goal walled-off / enclosed, no `goal` region in wavefront) | ~60% | ✗ 0 boards |
| `no_reachable_objects` (robot region has no adjacent movable to push) | ~11% | ✗ 0 boards |
| `all_pushes_failed` (dead root — swept exhaustively) | ~24% | ✓ dead root |
| `success` (opener / setup+finish found) | ~5% | ✓ |

Verified live (`colossus/repro.py`): failing scenes give wavefront labels like `['robot']` or `['robot','region_6']` — **no `goal` region at all** (the XML goal point sits in a buried/enclosed cell); good scenes give `['robot','goal','region_5','region_6']` with `goal` adjacent to `robot`. So the current pair-generator (`python/template_generation.py` / `scripts/rl_loop/build_gen0_pool.py`) is **mis-targeted for our scope** — it places goals without guaranteeing single-hop adjacency + a blocking movable.

**Cost of doing nothing:** generating naively reproduces 29% yield → ~3.4× the compute for the same number of boards, AND still overwhelmingly dead (no positive growth). The reject scenes are ~free per-scene at label time (instant), but they are pure waste when we PAY to generate them.

## How (pure-XML — the mechanism)

Generation is **pure geometry → MuJoCo XML, no physics at gen time** (grid/maze layout, place walls + robot + goal + movables, write XML). That splits the problem cleanly:

- **Yield = guaranteed geometrically, no sim.** Construct the layout directly — robot cell, an *adjacent* goal cell, a wall gap (doorway) between them plugged by **one movable** — instead of dropping the goal in an open area and hoping. Then verify each scene with `get_region_snapshot(..., use_xml_goal=True)` (pure wavefront reachability, no physics): keep only scenes where `goal` is an immediate neighbour of `robot` with a movable on the boundary. Turns 29%→~100% valid.
- **Openable = biased by geometry, confirmed by the labeler.** Whether a push actually merges is physics — pure XML can't *prove* it. But it makes it near-certain: **doorway wider than the object + clear displacement room** (side/far) → the movable is almost always pushable-to-merge. Set that geometry → high openable rate; the normal one-sim label pass confirms live-vs-dead and we keep openers/setups as positives. Geometry biases, physics confirms.
- **Difficulty = doorway/clutter tuning, still pure geometry.** Tighten the gap / awkward object angle / add clutter → fewer solving first-pushes → pushes the live scene into med/hard. This is how we make the scarce **hard-1push** positives.

## Plan (round-0, when un-parked)

1. **Diagnose the current gen** — read the actual pair-gen (`scripts/rl_loop/build_gen0_pool.py` / `build_growth_batch.py` — the `env_XXX_pair` producer; NOT the maze `template_generation.py`): how it places goal cell + robot + movables, why ~60% enclose the goal (no `goal` region in the wavefront). Fix vs rewrite call.
2. **Adjacency-by-construction** — build the doorway topology first (robot cell | doorway+movable | goal cell), realize XML, wavefront-verify (above). Fixes yield.
3. **Openability geometry** — doorway width > object + clear displacement room, reachable push pose for the car → bias openable.
4. **Difficulty control** — doorway tightness / clutter → target easy/med/hard.
5. **Yield gate (smoke)** — label a 200-scene smoke through region_opening; require usable-yield ≥80% AND a healthy openable (positive) fraction before any scale run. (scaled-run skill.)

## Success metric

- **Usable yield ≥ 80%** (vs current 29%).
- **Openable/positive fraction high** — the batch actually grows the positive base (openers/setups/true-2push), not another dead harvest.
- Positives span easy/med/hard tiers (not all trivial openers) — hard-1push is the scarce target.
- Scenes geometry-disjoint from `namo_testset_v1` (hold-out by room; [[reference_room_pool_lineage]]).

## Gating (why parked)

**Un-park trigger = the plan calls for growing the positive base** (a true full-data scale-up: more openers/setups/true-2push, harder tiers). This is the ONLY thing that needs this generator — dead we already have in surplus (bank 234k + 59k leftover + colossus), and we are **not** generating dead. The colossus dose-sweep answers a *different* question (does more dead still help); it does not gate this card. Whenever we decide the positive side must grow, this generator un-parks — there is **no** free source of positives to bridge from (every pre-screened pool is dead-biased; keep.txt is only 7k).

## Run

_(parked — un-park on a rising colossus dose curve)_

## Result

_(pending)_
