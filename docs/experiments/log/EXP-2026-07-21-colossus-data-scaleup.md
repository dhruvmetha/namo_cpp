---
status: live
thread: rl_loop
robot: car
updated: 2026-07-22
parent: EXP-2026-07-14-region-opening-curriculum-marvel
commit: 18d0ce3
---

# EXP-2026-07-21 — Colossus: data scale-up (overall) + dead dose (Marvel/Beast lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply. This card continues the curriculum framework in [EXP-2026-07-14](EXP-2026-07-14-region-opening-curriculum-marvel.md) (parent) — same ranker, same γ^k grammar, same labeler; the new things here are **overall data scale + dead dosage**.

## The one sentence

**Rejection-sample at least 200,000 genuine 2-push-only object episodes** from the geometry-clean Colossus-0 pool: beast-2c-d20 orders the complete root screen, any verified direct opener rejects the episode immediately, and only root-negative episodes receive the depth-2 setup/finish search; retain their root and post-push boards, then train the ranker on this grown hard-positive/dead base with an **X% dead dose** on top, sweeping X.

## Why (the evidence that triggered this)

The **d20 dose test** (20% dead roots+finishes added back into beast-2c-A-ceil, 50/50 non-dup) settled the question the 2c ablation opened — **dead helps**, measured two independent ways:

| metric | 2c-A-ceil (0% dead) | d20 (20% dead) | Δ |
|---|---|---|---|
| 1p hard@1 | 35.3 | 39.7 | **+4.4** |
| opener-vs-dead AUC (dead-bank GT) | 0.859 | 0.940 | **+0.081** |
| setup-vs-dead AUC | 0.851 | 0.906 | +0.055 |
| dead-cell median | 0.196 | 0.022 | pushed down |

A ~19-episode solve move AND an independent 0.08 AUC jump point the same way — the dead dose sharpens opener-vs-dead **separation**, which is what the hard tier needs.

**Mechanism (as far as measured, not theorized):** dead boards give the fence more signal to keep dead-looking pushes capped, so the rare true opener clears them and ranks higher. Whole-dead-root geometries are a scene class 2c never saw; training on them stops the model over-scoring dead pushes on hard roots. The AUC→top-1 gap stays large (0.94 AUC vs 39.7 hard@1) — pooled separation is decent, per-board top-1 on hard scenes is not yet — which is exactly what more hard/dead volume is meant to close.

**Supply wall → why scale.** The 20% dose used **19,282 of 19,448** available dead roots — 166 left. A bigger non-dup dose on the root side is impossible without new collection. Colossus is that collection.

## Labels — unchanged from parent (NO new label machinery)

Locked after a long clarification pass (see chat 2026-07-21):

- **1.0** exact — opener (root direct finish, or post-push finish)
- **0.9** exact — verified setup · **0.9 ceiling** — dead finish
- **0.81 ceiling** — dead root / dead-branch setup (γ²; ceiling-only, never exact — we never search depth 3)
- **0** floor — unreachable ONLY · **masked** — untried

`ceiling_mask=1` on every dead cell. The fence is one-sided (cap, free below), so **0.81 + ceiling already encodes "≤0.81"** — no sub-0.81 label, no code delta. Only **1.0 and 0.9-exact** are two-sided targets. There is no 0 in the trained DB except unreachable.

## Collection plan (Colossus-0)

- **Source:** the 2,031,481 generated, parseable, geometry-clean pair XMLs; the locked seed-42 1,000,000-XML manifest is wave 1, not a hard campaign ceiling. Full-room geometry is disjoint from the canonical test set.
- **Yield target:** stop staged collection only after the canonical per-episode census contains at least 200,000 genuine 2-push-only roots (`not is_1push_solvable and is_2push_solvable`) plus their collected post-push boards. XML count is an input budget, never the yield definition.
- **Ranker in loop:** beast-2c-d20 (`epoch010-val_loss1.7072`) → Amarel `colossus/d20_finish_ranker.ckpt`; d20 orders both root/setup candidates and finish candidates.
- **Root rejection:** d20 score-orders every reachable root push. The first simulator-verified direct opener stops that object episode immediately and skips all depth-2 work; the small root trial record remains for audit but is excluded from Colossus training. An episode may be retained as root-negative only after every reachable root candidate has failed, because the ranker controls order but the simulator establishes the label.
- **Depth-2 labeler:** for each post-setup board of a retained root-negative episode, d20 ranks every reachable finish: stop at the first verified opener when it appears in positions 1–5; if none of the first five opens, commit to the full remaining finish sweep and do not stop on a later hit. Thus top five is the trigger for exhaustive miss collection, not a five-sample cap; try order never randomizes, and every tried finish keeps its d20 score and rank.
- **Compute:** Amarel `main-redhat`; collect in staged waves of at most 470 tasks × 350 XMLs/task, with 14 CPUs and 12 workers per task.
- **Census (deliverable):** count rejected direct-1push episodes separately from retained true-2push and dead-within-depth2 episodes, always per `(pushed object, goal region)`. Retained true-2push setup/finish positives grow the base and retained dead roots/finishes feed the dose.
- **Data unit:** one XML may yield multiple independent `(pushed object, goal region)` episodes; rejection and the 200,000 target apply per episode, not per XML.

## Training recipe (the reframe)

Colossus is an **overall scale-up**, not a dead-only harvest. The build:

- **Base** = prior positives (d20/2c ≈ 192k) **+ retained Colossus true-2push setup and finish boards** after the full census. Rejected direct-1push audit rows do not enter this training build.
- **Dead dose** = add **X% dead** (of base size) from the enlarged pool of d20 dead examples plus measured colossus dead roots and finishes. Use 50/50 root/finish, non-dup, with ceilings applied at build (root 0.81, finish 0.9).
- **Sweep X** (e.g. 20 / 40 / max) on the grown base; compare hard@1 vs the d20 baseline (39.7) to read the dose-response.
- Base for the stack = **d20** [USER] (d20 positives already = 2c positives; colossus positives append; dead pool grows).

## Yield finding — pre-fix scenes (not a new problem)

Live labeling shows only **~29% of bank scenes are usable** (24% dead-root + 5% success); ~71% are out-of-scope (60% `goal_region_not_in_snapshot` = goal not 1-hop from robot, 11% `no_reachable_objects` = movable exists but robot can't reach a push pose). **This is expected: the bank (`collect3/bank.txt`) was generated 2026-07-13, before the gen↔label adjacency fix (`mujoco_env_creator@55badcb`, 2026-07-14) that took `goal_region_not_in_snapshot` 77.8%→0%.** The scenes that pass are valid and correctly labeled — the data is not corrupted, just low-yield.

**No follow-up needed on the generator** — it is already fixed (`mujoco_env_creator/generate_envs.py`, feb pilot ~90% accept). To get more XMLs (positives OR dead), just **rerun `generate_envs.py`**; there is no research problem here, only a rerun. Hard-1push is the rare tier (~3.5% of solvable, per EXP-2026-07-14 pilot) — a volume/cost fact, not a blocker.

## Run

**Colossus-0 staged rejection collection (updated 2026-07-22).** Screen the clean pool in staged waves with `beast-2c-d20`: reject an object episode at the first verified root opener, expand depth 2 only after a complete negative root screen, retain exhaust-on-top-5-missed finish behavior, and stop after at least 200,000 canonically confirmed genuine 2-push-only roots plus their post-push boards. DAgger follows after this scale-up.

**Target-box smoke.** On Amarel `main-redhat`, the fixed generator emitted 2/2 valid feb pair XMLs and the exact d20 collector completed them with real primitive progress and 3 stored episodes. Collection took 20.6 minutes on 2 workers = 0.344 worker-hours/XML. Together with the earlier 498-scene Beast probe (0.167 worker-hours/scene), the honest 1M collection range is about 25–51 hours at the 6,720-CPU hard ceiling, before queue/straggler loss. Therefore XML generation runs overnight, while collection is staged as safe ≤470-task waves and continues beyond the night; no prior data is overwritten or deleted.

**Launch.** Code commit `18d0ce3` (card stamp `9987060`), isolated checkout `/cache/home/dm1487/projects/namo/namo_cpp_colossus0_1m`, scratch root `/scratch/dm1487/curriculum2_amarel/colossus0_1m`, detached driver PID 1861775, generation array `58742902` (240 exclusive-node shards, three-hour cap). The driver selects exactly 600,000 aug9 + 400,000 feb XMLs only after the canonical full-room geometry gate, then submits ≤470-task collection waves of 350 XMLs/task using the exact d20 checkpoint SHA256 `6c1dfbb7108fb1a84b1a821b7b5d79d54198f3ef1e44af8acd0472dea6746046`. First live check: 50 generation tasks running, pair XML artifacts present, zero matched error logs.

**Generation and manifest gate complete (verified 2026-07-22).** Generation produced 2,031,481 unique parseable XMLs. The geometry gate found all 2,031,481 disjoint from the 8,773 canonical test-room geometries (`n_dropped=0`, `n_unparseable=0`). The clean pools contain 668,547 aug9 and 1,362,934 feb XMLs; `manifests/colossus0_1m.txt` contains exactly 1,000,000 paths selected with seed 42 at 600,000 aug9 + 400,000 feb. Manifest SHA256 is `22430e0b76f17cf248f9fe3e49a46078fed642afbf6b36bf11ec63fa7168c1ff`.

**Finalizer recovery.** The first finalizer, job `58744260`, used `ThreadPoolExecutor` for the CPU-bound XML geometry parser and timed out at 3:00:00 before writing the manifest. The isolated Amarel checkout was changed to `ProcessPoolExecutor`; retry job `58784784` completed the full 2,031,481-XML gate and exact manifest in 37:55. This ProcessPool change is currently uncommitted in that isolated Amarel checkout and must be ported to the main repository before the finalizer is reused.

**100-XML baseline smoke (old full-depth behavior; partial census at 98/100 results).** The canonical builder found 106 object episodes: 65 direct-1push (61.3%), 15 genuine 2-push-only (14.2% of all episodes; 18.8% of solvable episodes), and 26 unsolved within depth 2. The genuine-2push rate is about 0.153 per completed XML in this small sample, implying about 1.31M XMLs in expectation for 200,000 roots; use 1.5–2.0M as the cautious campaign range and stop by measured episode yield rather than this estimate.

**Measured speed limit and rejection savings.** Across those 98 completed XMLs, the old full-depth path executed 138,590 simulator pushes; direct-1push episodes consumed 38,096 of them, while d20's first verified root opener appeared after only 70 total root trials across all 65 such episodes (median rank 1). The rejection gate therefore removes about 38,026 pushes, or 27% of all primitive work in this sample. Runtime profiling attributes 30,469 of 36,539 worker-seconds to push execution, while primitive ranking consumed 0.2 seconds, so the remaining safe speed lever is cluster parallelism; materially faster labeling would require changing coverage by sampling setups or abandoning exhaustive top-5 misses.

**Current state (verified 2026-07-22).** Production collection has not started. The validated clean pool and exact 1M wave-1 manifest remain intact. Root-opener rejection is being target-box smoke-tested before any production wave; after it passes, submit staged waves and run the canonical census between waves until the 200,000 genuine-2push target is reached.

## Result

_(pending: the XML-generation and manifest prerequisite is complete; rejection-gated depth-2 collection, the per-episode census, training, and evaluation have not started.)_
