---
status: parked
thread: rl_loop
robot: car
updated: 2026-07-22
parent: EXP-2026-07-21-colossus-data-scaleup
---

# EXP-2026-07-22 — Hard-1push scene scarcity (biasing the fixed generator)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** An episode = robot region + goal region + the ONE movable between them; the ranker orders the pushes on that object. The scarce, valuable class is a **hard 1-push** scene — one where few first-pushes merge robot+goal, so ordering matters.

## Correction — the yield problem is already SOLVED (do NOT reinvent it)

An earlier draft of this card proposed building an "adjacency-by-construction" generator to fix ~70% out-of-scope scenes. **That is already fixed.** Grounded in [EXP-2026-07-14 Run log](EXP-2026-07-14-region-opening-curriculum-marvel.md):

- Real XML producer = **`mujoco_env_creator/generate_envs.py`** (`_runtime_validate_adjacency`, ~L345) — NOT `template_generation.py` / `build_gen0_pool.py` / `build_growth_batch.py`.
- **Fix = commit `mujoco_env_creator@55badcb` (2026-07-14):** re-enabled runtime adjacency validation — each candidate (robot, goal) placement is re-checked against the labeler's own live `get_region_snapshot`, so gen-time "1-hop" agrees with the labeler. Effect: label-time `goal_region_not_in_snapshot` **77.8% (fix OFF) → 0.0% (fix ON)** on a matched 80-scene sample, ~4.5 ms/scene.
- Fixed pilot (2026-07-14, ~560 scenes): gen accept-rate (runtime-validate % kept) **feb 90% / aug9 27%** — feb already ~97% usable. **≥80% yield is a solved bar, not an open one.**

**Why Colossus still saw 29%:** its scenes (`collect3/bank.txt`) were generated **2026-07-13, before the fix** → ~71% pre-fix goal-not-reachable rejects. The scenes that passed are valid and correctly labeled — Colossus data is **not corrupted, just low-yield** (wasted ~3.4× compute, favored dead). Future generation uses the fixed `generate_envs.py` and does not have this problem.

Also correcting a definition from the draft: **`no_reachable_objects`** = a blocking movable exists on the boundary but **the robot cannot reach a valid push pose** for it — NOT "no movable present."

## The real open problem

Among **solvable** episodes from the fixed pilot, the tier split is easy/med/hard = **73/23/3.5%** (aug9) and **77/20/3.5%** (feb) — **genuine hard-1push is the rare bin (~3.5%)**. Brute-force harvest cost: 20k solvable-hard ≈ **0.7–1M scenes (~4–8 h @2k cores)**, vs ~31–51k scenes for 20k easy. Hard is the flipped bottleneck (the "easy is rare" premise was falsified by this pilot).

**Question:** can we bias the fixed `generate_envs.py` toward **hard-1push solvable** scenes — raise the ~3.5% hit-rate — so hard positives stop costing ~1M scenes each round?

## Hypothesis

Geometry that makes *most* first-pushes fail-to-merge but leaves *one* narrow solution (tight doorway, awkward object aspect ratio, one viable push corridor among many blocked) yields hard-1push at a higher rate than uniform generation. Openability itself is unknowable from pure XML — the labeler still decides — but the *distribution* of solving-push counts is steerable by geometry, and that distribution is exactly what tier is.

## Plan (round-0, when un-parked)

1. **Read `generate_envs.py`** end-to-end (placement, obstacle density, doorway geometry, the runtime-validate accept path) — find the knobs that plausibly shift the solving-push-count distribution.
2. **Sweep geometry knobs** (doorway width, clutter near the doorway, object aspect ratio) on small batches; label through region_opening; measure the **hard-1push fraction among solvable**.
3. **Keep what raises hard-rate** without collapsing yield or solvable-rate; compare cost-per-hard-scene vs the ~1M brute-force baseline.
4. **Gate (smoke):** a knob setting is worth scaling only if it beats brute-force cost-per-hard-solvable at ≥ pilot yield. (scaled-run skill.)

## Success metric

- **Cost-per-hard-1push-solvable scene ↓** vs the ~0.7–1M brute-force baseline (the whole point).
- Yield stays ≥ pilot (feb ~90%); solvable-rate not collapsed.
- Scenes geometry-disjoint from `namo_testset_v1` (hold-out by room; [[reference_room_pool_lineage]]).

## Gating (why parked)

Parked. Un-park when the ladder actually needs **more hard-1push positives** and brute-force harvesting (~1M scenes/round) becomes the bottleneck — i.e. when hard@1 stalls and the cause is hard-example supply, not the ranker. NOT gated on the Colossus dead-dose sweep (different question — that is about dead volume; this is about hard-positive supply). The dead side is already oversupplied (bank + leftover + Colossus).

## Run

_(parked — un-park on a hard-positive supply bottleneck)_

## Result

_(pending)_
