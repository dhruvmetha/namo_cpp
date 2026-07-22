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

Build a generator that produces **valid single-hop region-opening scenes at high yield** (goal immediately adjacent to the robot region, one movable between) **with a controllable live-vs-dead mix** — so we can grow BOTH the positive base (openers/setups/true-2push) AND the dead pool on demand, instead of wasting ~70% of scenes and harvesting almost only dead.

## Two goals (why "yield-aware" is not enough)

1. **High yield** — stop the ~70% out-of-scope waste (goal enclosed / no adjacent movable).
2. **Controllable positive/live fraction** — the colossus bank is screen-**dead** by construction, so it grows the dead pool but barely the positive base (+4%). A true "full data scale-up" needs **openable** scenes (a movable that CAN merge robot+goal in 1-2 pushes), which no pre-screened source gives us. The generator must be able to dial the live-vs-dead ratio, not just emit dead.

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

A generator that **constructs** the robot-region / goal-region / blocking-movable topology directly (rather than placing a goal and hoping the wavefront cooperates) can reach **≥80% usable yield** AND hit a **target live-vs-dead ratio** via a difficulty knob — giving us positives (openable) and dead on demand from one generator.

## Plan (round-0, when un-parked)

1. **Diagnose the current 29%** — read `template_generation.py` end-to-end: how goal cell + robot + movables are placed, why ~60% enclose the goal. Is it a placement bug (goal dropped into occupied/walled cell) or a topology choice (multi-region layouts where goal lands 2+ hops away)? This decides whether it's a fix or a rewrite.
2. **Adjacency-guaranteed construction** — generate the wavefront topology first: robot region, one adjacent goal region, a movable straddling the shared boundary; then realize XML from that. Verify each scene with the SAME `get_region_snapshot(..., use_xml_goal=True)` the planner uses — keep only scenes where `goal` is an immediate neighbour of `robot`. This alone fixes yield.
3. **Live-vs-dead mix knob** — parametrize the boundary/obstacle geometry so the blocking movable is **openable** (clear merge path in 1-2 pushes → openers/setups/true-2push = positives) vs **dead** (boxed-in, no merge → dead root). Sweep the knob to hit a target ratio (e.g. 60/40 live/dead for base growth, or dead-heavy for a dose top-up). The point: **dial positives, not just harvest dead**.
4. **Difficulty control** — within live scenes, tune how many first-pushes merge (n_setups) to place the scene in easy/med/hard tiers on demand — the hard 1-push tier is the scarce one.
5. **Yield + mix gate (smoke)** — label a 200-scene smoke through region_opening; require usable-yield ≥80% AND the realized live/dead ratio within ±10% of target before any scale run. (scaled-run skill.)

## Success metric

- **Usable yield ≥ 80%** (dead-root + success) vs current 29%.
- **Realized live/dead ratio tracks the knob** (±10%) — can produce a positive-rich batch (grow base) OR a dead batch (dose) from the same generator.
- Live scenes span easy/med/hard tiers (not all trivial openers).
- Scenes geometry-disjoint from `namo_testset_v1` (hold-out by room; [[reference_room_pool_lineage]]).

## Gating (why parked)

Parked, but **two independent un-park triggers** (either fires it):

1. **Dead-dose still rising.** Colossus (EXP-2026-07-21) stacks ~51k dead on the d20 base → dose 20%→~45%. If hard@1 is still climbing at ~45%, we need more dead than bank+leftover (59k) can supply → generate (dead-heavy knob). If it plateaus, this trigger is off.
2. **Positive base needs to grow (full scale-up).** This is NOT gated on the dose curve — the bank is dead-biased and *cannot* grow positives (colossus added only +4%), so any real full-data scale-up needs the generator's **live/openable** output regardless of what the dead-dose does. If the plan is to grow the whole set (more openers/setups/true-2push, harder tiers), un-park for positives even if dead plateaus.

Free bridge before generating dead: 59k bank leftover + `collect3/keep.txt` (already generated, disjoint). There is **no** free bridge for positives — that's the gap only this generator fills.

## Run

_(parked — un-park on a rising colossus dose curve)_

## Result

_(pending)_
