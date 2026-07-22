---
status: active
thread: rl_loop
robot: car
updated: 2026-07-21
parent: EXP-2026-07-14-region-opening-curriculum-marvel
---

# EXP-2026-07-21 — Colossus: data scale-up (overall) + dead dose (Marvel/Beast lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply. This card continues the curriculum framework in [EXP-2026-07-14](EXP-2026-07-14-region-opening-curriculum-marvel.md) (parent) — same ranker, same γ^k grammar, same labeler; the new things here are **overall data scale + dead dosage**.

## The one sentence

**Scale the whole 2-push dataset** with +175k fresh labeled scenes (collect3 screen-bank, exhaust-on-miss, beast-2c-d20 finish ranker): the openers/setups/true-2push it yields are **new positive boards that grow the base**, and the dead roots grow the **dead pool** — then train the ranker on the grown base with an **X% dead dose** on top, sweeping X. NOT a dead-only harvest; dead is one component + a dose knob.

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

## Collection plan (round-0)

- **Source:** +175k fresh roots from Amarel `collect3/bank.txt` (234,342 screen-dead leads; dead-heavy, disjoint from round-2's ~26k).
- **Ranker in loop:** beast-2c-d20 (`epoch010-val_loss1.7072`) → Amarel `colossus/d20_finish_ranker.ckpt`.
- **Labeler:** exhaust-on-miss, `region_exhaust_on_miss_topk=5`, setups exhaustive, chain-depth 2. Finish sweep follows d20's rank order end-to-end (stop rule flips at the top-5 boundary; try-order never randomizes — records the rank the opener sat at).
- **Compute:** Amarel main-redhat, wide burst up to the 6,720-CPU / 500-task cap, target ≤3-4h. Smoke + calibrate first (scaled-run).
- **Census (deliverable):** count 1-push (root direct), 1-push (post-push finish), true 2-push (setup, no direct), dead 1-push (post-push exhausted empty), dead 2-push (root all-dead). Positives AND dead both matter — the positives grow the base, the dead feed the dose.

## Training recipe (the reframe)

Colossus is an **overall scale-up**, not a dead-only harvest. The build:

- **Base** = all positive boards = prior positives (d20/2c ≈ 192k) **+ colossus positives** (its openers/setups/true-2push, ~5% of scenes ≈ ~8k). The grown base is the point of "scale-up."
- **Dead dose** = add **X% dead** (of base size) from the enlarged dead pool = d20's 38k + colossus dead (~40k roots + finishes). 50/50 root/finish, non-dup, ceilings at build (root 0.81, finish 0.9).
- **Sweep X** (e.g. 20 / 40 / max) on the grown base; compare hard@1 vs the d20 baseline (39.7) to read the dose-response.
- Base for the stack = **d20** [USER] (d20 positives already = 2c positives; colossus positives append; dead pool grows).

## Yield finding — pre-fix scenes (not a new problem)

Live labeling shows only **~29% of bank scenes are usable** (24% dead-root + 5% success); ~71% are out-of-scope (60% `goal_region_not_in_snapshot` = goal not 1-hop from robot, 11% `no_reachable_objects` = movable exists but robot can't reach a push pose). **This is expected: the bank (`collect3/bank.txt`) was generated 2026-07-13, before the gen↔label adjacency fix (`mujoco_env_creator@55badcb`, 2026-07-14) that took `goal_region_not_in_snapshot` 77.8%→0%.** The scenes that pass are valid and correctly labeled — the data is not corrupted, just low-yield. Future generation uses the fixed `generate_envs.py` (feb pilot ~90% accept) and does not repeat this. So the yield hit here costs compute, not correctness; the dead we harvest is fine.

The genuinely open scene problem is **hard-1push scarcity** (only ~3.5% of solvable scenes are hard) → child card [EXP-2026-07-22-hard1push-scarcity](EXP-2026-07-22-hard1push-scarcity.md) (parked; separate from the dead-dose question).

## Run

_(pending — smoke → analyze → full burst)_

## Result

_(pending)_
