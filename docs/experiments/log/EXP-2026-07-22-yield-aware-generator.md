---
status: parked
thread: rl_loop
robot: car
updated: 2026-07-22
parent: EXP-2026-07-21-colossus-data-scaleup
---

# EXP-2026-07-22 — Yield-aware scene generator (single-hop, dead-biased)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** An episode = robot region + goal region + the ONE movable between them; the ranker orders the pushes on that object. A scene is only usable if the wavefront makes a distinct `goal` region **immediately adjacent** to the robot region with a movable on the boundary. This card is about generating scenes that actually satisfy that.

## The one sentence

Build a generator that produces **valid single-hop region-opening scenes at high yield** (goal is an immediate neighbour of the robot region, one movable between, biased toward dead), so future dead-data collection stops wasting ~70% of its scenes.

## Why (measured, not theorized)

Colossus round-0 (EXP-2026-07-21) labeled 175k `collect2/gen` pair scenes through region_opening and only **~29% were usable** (dead-root or solved). The other ~71% produce **zero training boards** — the planner correctly rejects them as out-of-scope:

| outcome | share (404 live boards) | usable? |
|---|---|---|
| `goal_region_not_in_snapshot` (goal walled-off / enclosed, no `goal` region in wavefront) | ~60% | ✗ 0 boards |
| `no_reachable_objects` (robot region has no adjacent movable to push) | ~11% | ✗ 0 boards |
| `all_pushes_failed` (dead root — swept exhaustively) | ~24% | ✓ dead root |
| `success` (opener / setup+finish found) | ~5% | ✓ |

Verified live (`colossus/repro.py`): failing scenes give wavefront labels like `['robot']` or `['robot','region_6']` — **no `goal` region at all** (the XML goal point sits in a buried/enclosed cell); good scenes give `['robot','goal','region_5','region_6']` with `goal` adjacent to `robot`. So the current pair-generator (`python/template_generation.py` / `scripts/rl_loop/build_gen0_pool.py`) is **mis-targeted for our scope** — it places goals without guaranteeing single-hop adjacency + a blocking movable.

**Cost of doing nothing:** generating naively reproduces 29% yield → ~3.4× the compute for the same number of boards. The reject scenes are ~free per-scene at label time (instant), but they are pure waste when we PAY to generate them.

## Hypothesis

A generator that **constructs** the robot-region / goal-region / blocking-movable topology directly (rather than placing a goal and hoping the wavefront cooperates) can reach **≥80% usable yield** while keeping a **high dead fraction** (the whole point — dead roots are the scarce, valuable class).

## Plan (round-0, when un-parked)

1. **Diagnose the current 29%** — read `template_generation.py` end-to-end: how goal cell + robot + movables are placed, why ~60% enclose the goal. Is it a placement bug (goal dropped into occupied/walled cell) or a topology choice (multi-region layouts where goal lands 2+ hops away)? This decides whether it's a fix or a rewrite.
2. **Adjacency-guaranteed construction** — generate the wavefront topology first: robot region, one adjacent goal region, a movable straddling the shared boundary; then realize XML from that. Verify each scene with the SAME `get_region_snapshot(..., use_xml_goal=True)` the planner uses — keep only scenes where `goal` is an immediate neighbour of `robot`.
3. **Dead-bias knob** — tune obstacle density / boundary geometry so a target fraction of the movables are un-pushable-to-merge (dead roots) vs openable — we want dead-heavy but with a live tail for openers/setups.
4. **Yield gate (smoke)** — label a 200-scene smoke through region_opening; require usable-yield ≥80% and dead-fraction in the target band before any scale run. (scaled-run skill.)

## Success metric

- **Usable yield ≥ 80%** (dead-root + success) vs current 29%.
- Dead-root fraction high enough to feed the dose-sweep (colossus needs ~tens of k dead roots per dose step).
- Scenes are geometry-disjoint from `namo_testset_v1` (hold-out by room; [[reference_room_pool_lineage]]).

## Gating (why parked)

**Blocked on the colossus dose-response.** Colossus (EXP-2026-07-21) stacks ~51k dead on the d20 base → dose 20%→~45%. Only build+run this generator if hard@1 is **still climbing at ~45%** (i.e. more dead volume than bank+leftover can supply is actually needed). If it plateaus, generation is moot. Free bridge before generation: 59k bank leftover + `collect3/keep.txt` (already generated, disjoint).

## Run

_(parked — un-park on a rising colossus dose curve)_

## Result

_(pending)_
