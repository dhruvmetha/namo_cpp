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

**Scale the whole 2-push dataset** by collecting the geometry-clean Colossus-0 manifest of 1,000,000 fresh XMLs with the fixed generator, exhaust-on-top-5-miss, and the beast-2c-d20 finish ranker: the openers/setups/true-2push it yields are **new positive boards that grow the base**, and the dead roots grow the **dead pool** — then train the ranker on the grown base with an **X% dead dose** on top, sweeping X. NOT a dead-only harvest; dead is one component + a dose knob.

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

- **Source:** exactly 1,000,000 generated pair XMLs selected with seed 42 from the geometry-clean pool at the locked 600,000 aug9 / 400,000 feb mix; full-room geometry is disjoint from the canonical test set.
- **Ranker in loop:** beast-2c-d20 (`epoch010-val_loss1.7072`) → Amarel `colossus/d20_finish_ranker.ckpt`.
- **Labeler:** chain-depth 2 with exhaustive root setups. For each post-setup board, d20 ranks every reachable finish: stop at the first verified opener when it appears in positions 1–5; if none of the first five opens, commit to the full remaining finish sweep and do not stop on a later hit. Thus top five is the trigger for exhaustive miss collection, not a five-sample cap; try order never randomizes, and every tried finish keeps its d20 score and rank.
- **Compute:** Amarel `main-redhat`; collect in staged waves of at most 470 tasks × 350 XMLs/task, with 14 CPUs and 12 workers per task.
- **Census (deliverable):** count 1-push (root direct), 1-push (post-push finish), true 2-push (setup, no direct), dead 1-push (post-push exhausted empty), dead 2-push (root all-dead). Positives AND dead both matter — the positives grow the base, the dead feed the dose.
- **Data unit:** 1,000,000 is the number of input XMLs, not the final number of training episodes; each room may yield multiple independent `(pushed object, goal region)` episodes.

## Training recipe (the reframe)

Colossus is an **overall scale-up**, not a dead-only harvest. The build:

- **Base** = all positive boards = prior positives (d20/2c ≈ 192k) **+ measured colossus positives** after the full census (openers/setups/true-2push). The grown base is the point of "scale-up."
- **Dead dose** = add **X% dead** (of base size) from the enlarged pool of d20 dead examples plus measured colossus dead roots and finishes. Use 50/50 root/finish, non-dup, with ceilings applied at build (root 0.81, finish 0.9).
- **Sweep X** (e.g. 20 / 40 / max) on the grown base; compare hard@1 vs the d20 baseline (39.7) to read the dose-response.
- Base for the stack = **d20** [USER] (d20 positives already = 2c positives; colossus positives append; dead pool grows).

## Yield finding — pre-fix scenes (not a new problem)

Live labeling shows only **~29% of bank scenes are usable** (24% dead-root + 5% success); ~71% are out-of-scope (60% `goal_region_not_in_snapshot` = goal not 1-hop from robot, 11% `no_reachable_objects` = movable exists but robot can't reach a push pose). **This is expected: the bank (`collect3/bank.txt`) was generated 2026-07-13, before the gen↔label adjacency fix (`mujoco_env_creator@55badcb`, 2026-07-14) that took `goal_region_not_in_snapshot` 77.8%→0%.** The scenes that pass are valid and correctly labeled — the data is not corrupted, just low-yield.

**No follow-up needed on the generator** — it is already fixed (`mujoco_env_creator/generate_envs.py`, feb pilot ~90% accept). To get more XMLs (positives OR dead), just **rerun `generate_envs.py`**; there is no research problem here, only a rerun. Hard-1push is the rare tier (~3.5% of solvable, per EXP-2026-07-14 pilot) — a volume/cost fact, not a blocker.

## Run

**Colossus-0 1M scale-up launch (2026-07-22).** Generate a fresh, fixed-generator source and select exactly 1,000,000 geometry-clean pair XMLs at the locked 60/40 aug9/feb mix, then collect every XML at depth 2 with `beast-2c-d20`, exhaustive root setups, and exhaust-on-top-5-missed finishes. Keep the complete natural census: direct 1-push, true 2-push, and dead. This is one Colossus-0 dataset; DAgger follows after the scale-up.

**Target-box smoke.** On Amarel `main-redhat`, the fixed generator emitted 2/2 valid feb pair XMLs and the exact d20 collector completed them with real primitive progress and 3 stored episodes. Collection took 20.6 minutes on 2 workers = 0.344 worker-hours/XML. Together with the earlier 498-scene Beast probe (0.167 worker-hours/scene), the honest 1M collection range is about 25–51 hours at the 6,720-CPU hard ceiling, before queue/straggler loss. Therefore XML generation runs overnight, while collection is staged as safe ≤470-task waves and continues beyond the night; no prior data is overwritten or deleted.

**Launch.** Code commit `18d0ce3` (card stamp `9987060`), isolated checkout `/cache/home/dm1487/projects/namo/namo_cpp_colossus0_1m`, scratch root `/scratch/dm1487/curriculum2_amarel/colossus0_1m`, detached driver PID 1861775, generation array `58742902` (240 exclusive-node shards, three-hour cap). The driver selects exactly 600,000 aug9 + 400,000 feb XMLs only after the canonical full-room geometry gate, then submits ≤470-task collection waves of 350 XMLs/task using the exact d20 checkpoint SHA256 `6c1dfbb7108fb1a84b1a821b7b5d79d54198f3ef1e44af8acd0472dea6746046`. First live check: 50 generation tasks running, pair XML artifacts present, zero matched error logs.

**Generation and manifest gate complete (verified 2026-07-22).** Generation produced 2,031,481 unique parseable XMLs. The geometry gate found all 2,031,481 disjoint from the 8,773 canonical test-room geometries (`n_dropped=0`, `n_unparseable=0`). The clean pools contain 668,547 aug9 and 1,362,934 feb XMLs; `manifests/colossus0_1m.txt` contains exactly 1,000,000 paths selected with seed 42 at 600,000 aug9 + 400,000 feb. Manifest SHA256 is `22430e0b76f17cf248f9fe3e49a46078fed642afbf6b36bf11ec63fa7168c1ff`.

**Finalizer recovery.** The first finalizer, job `58744260`, used `ThreadPoolExecutor` for the CPU-bound XML geometry parser and timed out at 3:00:00 before writing the manifest. The isolated Amarel checkout was changed to `ProcessPoolExecutor`; retry job `58784784` completed the full 2,031,481-XML gate and exact manifest in 37:55. This ProcessPool change is currently uncommitted in that isolated Amarel checkout and must be ported to the main repository before the finalizer is reused.

**Current state (verified 2026-07-22).** The original driver exited after the first finalizer timed out, so it did not submit collection after the successful retry. Amarel currently has no pending or running jobs for this account, and `/scratch/dm1487/curriculum2_amarel/colossus0_1m/collect` contains zero result files. The next action is to submit the staged depth-2 d20 collection waves from the validated 1,000,000-XML manifest; do not regenerate or refilter.

## Result

_(pending: the XML-generation and manifest prerequisite is complete; depth-2 collection, the per-episode census, training, and evaluation have not started.)_
