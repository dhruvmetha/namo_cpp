---
status: planned
thread: rl_loop
robot: car
updated: 2026-07-14
supersedes: EXP-2026-07-12-opener-curriculum-loop (buggy lineage — retracted, see below)
---

# EXP-2026-07-14 — Region-Opening curriculum, clean restart (Marvel lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply (beat random, fewer sim calls, every tier). This card is the *reproducible framework* for building that ranker; it does not restate the problem.

## The one sentence

Learn the ranker via a **curriculum ladder** (1-push → 2-push → …), running **pure DAgger within each stage**, on a **clean, correctly-labeled, in-sync-generated** data pipeline — starting from a **balanced bootstrap seed**.

## Why we restarted (the old lineage is dead)

model_0/1/2/3 are **discarded**. Three bugs, all found 2026-07-14, invalidated the whole EXP-2026-07-12 lineage:
1. **Labels measured the wrong thing.** `region_opening.target_goal_region` defaults False and was never set, so every label counted opening *any neighbour region*, not *the task goal* — verified by a faithful re-run (99.9% of "budget-hit" scenes are genuine 2-push) + a 15/15 replay. All prior training data is neighbour-opening.
2. **Generator out of sync with labeler.** `--require-adjacent` filters on an idealized static-adjacency test ("would two regions connect if this one object *vanished*") computed on obstacles-only geometry, while the labeler uses a real wavefront BFS with the object *pushed* — so ~22% of generated scenes are multi-hop and get dropped. The safety net `_runtime_validate_adjacency` was dead code (`return True`).
3. **Composition starvation.** The `75% hard` subsample + neighbour-label skew left model_3 with ~4% easy → regressed.

**Prerequisite gate for THIS card:** the generator fix (re-enable `_runtime_validate_adjacency` to use the labeler's own `get_region_snapshot` with robot/goal placed) must be **verified in-sync** (generate-with-fix → labeler drops ~0) before any data is generated.

## The fixed, reproducible pipeline

- **Generator:** `mujoco_env_creator/generate_envs.py`, `--require-adjacent` + the re-enabled per-placement re-check → gen "1-hop" == the labeler's real wavefront. One robot region, one goal region, one blocking object.
- **Labels:** goal-opening only — `region_opening_rung1_exhaustive_car.yaml` **+ `--target-goal-region`** (never the default). Exhaustive 1-push over reachable pushes → the 60×5 opener grid.
- **Difficulty — FIXED cuts, never tertiles** (`sr = openers/reachable`): **hard `sr<0.05`, medium `0.05–0.30`, easy `sr≥0.30`**.

## Lineage & naming (Marvel, alphabetical; stage = character, `-N` = iteration)

- **1-push stage → ANT-MAN:** `antman-0` (balanced seed) → `antman-1`, `antman-2`, … (DAgger rounds)
- **2-push stage → BEAST:** `beast-0` → `beast-1`, …
- (Cyclops, Daredevil, … for later stages.)

## The algorithm

```
BOOTSTRAP  antman-0:
  generate a large fresh batch (fixed gen) → label (goal-opening)
  → build a BALANCED seed: 20k easy / 20k med / 20k hard by fixed cuts
     (easy is the RARE bin under task-goal → gen size is set by the easy target;
      keep all easy, undersample med/hard to 20k) → train antman-0.

PURE DAgger  antman-r (r = 1,2,…):
  1. GENERATE  fresh scenes (fixed gen)
  2. SCREEN    best-first with antman_{r-1} → keep the scenes it gets WRONG (its mistakes)
  3. LABEL     exhaustive goal-opening on the kept scenes
  4. TRAIN     accumulate ALL clean data + retrain antman_r  (pure mistakes → naturally med/hard-heavy; that's fine)
  5. EVAL      solve@k by fixed-cut difficulty vs random
  6. LOOP/STOP  improved → r+1 ; yield→0 / plateau → stage done

LADDER: 1-push loop to plateau → start the 2-push stage (beast-*) on post-setup scenes, same loop.
```

## Decisions locked [USER 2026-07-14]

1. **Fresh restart** — discard the buggy lineage; new Marvel-named lineage.
2. **Pure DAgger** (mistake-targeting) after the seed — NO forced balance beyond the bootstrap.
3. **Balanced bootstrap only** — 20k/20k/20k easy/med/hard (fixed cuts) for `<char>-0`.
4. **Goal-opening labels** (`target_goal_region` ON) — always.
5. **In-sync generator** (the `_runtime_validate_adjacency` fix, verified) — always.
6. **Fixed-cut difficulty**, never tertiles.
7. **Curriculum ladder** 1-push → 2-push, DAgger within each stage.

## Eval

Solve@k by difficulty (easy/med/hard, fixed cuts) on the held-out `namo_testset_v1`, vs the random ranker — both **reactive @1** and **with-search @k**. Bar: beat random on every tier; the with-search curve dominates random's (any solve-rate at a fraction of the sim calls).

## Run log

_(appended as we go — starts with the in-sync fix verification, then `antman-0`)_
