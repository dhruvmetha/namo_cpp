---
status: live
thread: rl_loop
robot: car
updated: 2026-07-22
parent: EXP-2026-07-14-region-opening-curriculum-marvel
commit: 897c398
---

# EXP-2026-07-21 — Colossus: data scale-up (overall) + dead dose (Marvel/Beast lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply. This card continues the curriculum framework in [EXP-2026-07-14](EXP-2026-07-14-region-opening-curriculum-marvel.md) (parent) — same ranker, same γ^k grammar, same labeler; the new things here are **overall data scale + dead dosage**.

## The one sentence

**Colossus-0 is a 200,000-new-training-row pilot of censored, d20-guided experience:** reject direct-root openers, search root-negative episodes to depth 2, retain verified setup/finish comparisons and a controlled negative/dead dose, train one successor to d20, and expand toward the million-XML campaign only if it improves ranking/search.

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
- **Pilot yield target:** stop staged collection only after the Colossus builder can materialize the locked 200,000-row learning mix below. XML count and raw node count are inputs, never the yield definition; the earlier 200,000-genuine-root target is deferred until this experience policy passes its training gate.
- **Ranker in loop:** beast-2c-d20 (`epoch010-val_loss1.7072`) → Amarel `colossus/d20_finish_ranker.ckpt`; d20 orders both root/setup candidates and finish candidates.
- **Root rejection:** d20 score-orders every reachable root push. The first simulator-verified direct opener stops that object episode immediately and skips all depth-2 work; the small root trial record remains for audit but is excluded from Colossus training. An episode may be retained as root-negative only after every reachable root candidate has failed, because the ranker controls order but the simulator establishes the label.
- **Depth-2 labeler under smoke test:** d20 ranks every reachable finish and every episode tries at most the top 20 first. A stable seed-42 20% sample of complete `(xml, object, goal-region)` episodes exhausts every top-20-miss board; the other 80% stop at 20 and mark the unresolved parent setup censored/unknown, never dead. A verified opener at any tried rank confirms the setup. Every tried finish keeps its d20 score and rank, and untried cells stay masked.
- **Compute:** Amarel `main-redhat`; collect in staged waves of at most 470 tasks × 350 XMLs/task, with 14 CPUs and 12 workers per task.
- **Cross-cluster gate:** Amarel collection/rendering and CS training/evaluation use isolated checkouts pinned to the recorded commit; every transferred H5 shard is SHA256-verified after rsync, and the combined-H5/training launch must fail on a commit or checksum mismatch.
- **Census (deliverable):** count rejected direct-1push, confirmed true-2push, censored depth-2, audit-proven dead-within-depth2, and eligible training rows separately, always preserving `(xml, pushed object, goal region)` identity.
- **Data unit:** one XML may yield multiple independent `(pushed object, goal region)` episodes and every episode may yield one root board plus many post-push boards; the 200,000 target counts selected H5 board rows, never XMLs or object episodes.

## Data artifact contract — push-depth-ready

Colossus artifacts must make the complete candidate action explicit at every board, including post-push boards: the 60 contact locations are not enough because each contact has five push depths with different nominal object motion.

- **Required per-board field:** carry `action_motion` with shape `(60, 5, 3)`, aligned exactly with the scorer and label tensors as `[edge, push_depth, (dx, dy, dtheta)]`. This is the primitive database's nominal pre-simulation object motion, so collecting it requires no additional simulator calls.
- **Frame and units:** store translation in the world/crop axes and rotation about the object, normalized as `(dx / 0.5 m, dy / 0.5 m, dtheta / pi)` for direct model use. Record the frame, units, and normalization in artifact metadata; never leave them implicit.
- **Provenance:** record the active primitive database identifier and SHA256, the shape-family choice (`square`, `wide`, or `tall`), and the target object's current `(x, y, yaw, size_x, size_y)` at that board. The current state is required because an XML's initial object pose is wrong for post-push finish boards.
- **PKL → NPZ → H5:** preserve the same field and ordering through every conversion rather than reconstructing it later. H5 stores dense `action_motion` as `(N, 60, 5, 3)` in `float16` or `float32`, alongside `contact_px`, `value_target`, and `r_mask`; PKL/NPZ retain the per-board tensor plus provenance metadata.
- **Build-time gate:** fail conversion if `action_motion`, labels, masks, and the live action generator do not agree on all 300 `(edge, push_depth)` candidate indices. Also reject mixed primitive-database hashes within one H5 unless the row-level provenance is retained explicitly.

This contract lets a future ranker consume push depth directly without reopening historical XMLs, inferring object geometry from contact points, or running forward simulations. Existing data can still reconstruct these features from `contact_px` plus the pinned primitive files, but Colossus must store them directly.

## Training recipe (the reframe)

Colossus-0 tests whether **d20-guided, partially censored experience is useful training data**, not whether raw row volume alone helps. Its new-data block is exactly 200,000 rows selected with a fixed seed:

- **Positive/mistake base: 166,666 rows.** Keep root-negative root boards with at least one verified setup and post-push boards with a verified finisher; rank-1-only finish boards have no within-board ranking mistake and are excluded before harder positive rows. Unknown root parents from capped children stay masked.
- **Negative dose: 33,334 rows.** Add verified ceiling supervision from capped top-20 misses and audit-proven dead boards, split 50/50 root/post-push where supply permits; a capped board is useful hard-negative experience but is never called dead, and its unresolved parent stays masked.
- **Rejected direct-1push rows:** preserve them in the raw census but exclude them from the primary Colossus-0 build. A separate later ablation may use rank>1 rejected roots because they are cheap d20 mistakes, but they must use a 0.9 ceiling on prior non-openers rather than false dead labels.
- **Stack and loss:** append the 200,000-row block to the exact d20 training base and use the same HL-Gauss + listwise rank-aux recipe. Recompute root/post-push sampling weights; group the train/validation split by room.
- **Training gate:** compare the successor against d20 on every easy/medium/hard × 1push/2push slice. Continue scale-up only if ranking/search improves without a material regression in any slice; the headline metrics are 1push solve@1/@5 and 2push sims-to-solve/solve@budget.

## Yield finding — pre-fix scenes (not a new problem)

Live labeling shows only **~29% of bank scenes are usable** (24% dead-root + 5% success); ~71% are out-of-scope (60% `goal_region_not_in_snapshot` = goal not 1-hop from robot, 11% `no_reachable_objects` = movable exists but robot can't reach a push pose). **This is expected: the bank (`collect3/bank.txt`) was generated 2026-07-13, before the gen↔label adjacency fix (`mujoco_env_creator@55badcb`, 2026-07-14) that took `goal_region_not_in_snapshot` 77.8%→0%.** The scenes that pass are valid and correctly labeled — the data is not corrupted, just low-yield.

**No follow-up needed on the generator** — it is already fixed (`mujoco_env_creator/generate_envs.py`, feb pilot ~90% accept). To get more XMLs (positives OR dead), just **rerun `generate_envs.py`**; there is no research problem here, only a rerun. Hard-1push is the rare tier (~3.5% of solvable, per EXP-2026-07-14 pilot) — a volume/cost fact, not a blocker.

## Run

**Colossus-0 staged pilot (updated 2026-07-22).** Screen the clean pool in staged waves with `beast-2c-d20`: reject an object episode at the first verified root opener, expand depth 2 only after a complete negative root screen, cap ordinary finish boards at top 20, exhaust top-20 misses on a deterministic 20% episode audit, and stop when the fixed-seed builder has the 166,666 positive/mistake + 33,334 negative rows required above. Train and evaluate this 200,000-row experience pilot before authorizing a larger collection.

**Target-box smoke.** On Amarel `main-redhat`, the fixed generator emitted 2/2 valid feb pair XMLs and the exact d20 collector completed them with real primitive progress and 3 stored episodes. Collection took 20.6 minutes on 2 workers = 0.344 worker-hours/XML. Together with the earlier 498-scene Beast probe (0.167 worker-hours/scene), the honest 1M collection range is about 25–51 hours at the 6,720-CPU hard ceiling, before queue/straggler loss. Therefore XML generation runs overnight, while collection is staged as safe ≤470-task waves and continues beyond the night; no prior data is overwritten or deleted.

**Launch.** Code commit `18d0ce3` (card stamp `9987060`), isolated checkout `/cache/home/dm1487/projects/namo/namo_cpp_colossus0_1m`, scratch root `/scratch/dm1487/curriculum2_amarel/colossus0_1m`, detached driver PID 1861775, generation array `58742902` (240 exclusive-node shards, three-hour cap). The driver selects exactly 600,000 aug9 + 400,000 feb XMLs only after the canonical full-room geometry gate, then submits ≤470-task collection waves of 350 XMLs/task using the exact d20 checkpoint SHA256 `6c1dfbb7108fb1a84b1a821b7b5d79d54198f3ef1e44af8acd0472dea6746046`. First live check: 50 generation tasks running, pair XML artifacts present, zero matched error logs.

**Generation and manifest gate complete (verified 2026-07-22).** Generation produced 2,031,481 unique parseable XMLs. The geometry gate found all 2,031,481 disjoint from the 8,773 canonical test-room geometries (`n_dropped=0`, `n_unparseable=0`). The clean pools contain 668,547 aug9 and 1,362,934 feb XMLs; `manifests/colossus0_1m.txt` contains exactly 1,000,000 paths selected with seed 42 at 600,000 aug9 + 400,000 feb. Manifest SHA256 is `22430e0b76f17cf248f9fe3e49a46078fed642afbf6b36bf11ec63fa7168c1ff`.

**Finalizer recovery.** The first finalizer, job `58744260`, used `ThreadPoolExecutor` for the CPU-bound XML geometry parser and timed out at 3:00:00 before writing the manifest. The isolated Amarel checkout was changed to `ProcessPoolExecutor`; retry job `58784784` completed the full 2,031,481-XML gate and exact manifest in 37:55. This ProcessPool change is now committed to the main repository (byte-identical to the proven Amarel version); the isolated Amarel checkout still carries it as a local modification.

**100-XML baseline smoke (old full-depth behavior; partial census at 98/100 results).** The canonical builder found 106 object episodes: 65 direct-1push (61.3%), 15 genuine 2-push-only (14.2% of all episodes; 18.8% of solvable episodes), and 26 unsolved within depth 2. The genuine-2push rate is about 0.153 per completed XML in this small sample, implying about 1.31M XMLs in expectation for 200,000 roots; use 1.5–2.0M as the cautious campaign range and stop by measured episode yield rather than this estimate.

**Measured speed limit and rejection savings.** Across those 98 completed XMLs, the old full-depth path executed 138,590 simulator pushes; direct-1push episodes consumed 38,096 of them, while d20's first verified root opener appeared after only 70 total root trials across all 65 such episodes (median rank 1). The rejection gate therefore removes about 38,026 pushes, or 27% of all primitive work in this sample. Runtime profiling attributes 30,469 of 36,539 worker-seconds to push execution, while primitive ranking consumed 0.2 seconds, so the remaining safe speed lever is cluster parallelism; materially faster labeling would require changing coverage by sampling setups or abandoning exhaustive top-5 misses.

**Root-rejection target smoke passed.** Amarel array `58886389` completed the known direct-1push and genuine-2push cases in 1:21 and 13:35; the canonical builder recovered exactly one rejected 1push episode and one genuine 2push-only episode. Watcher `58886562` completed successfully. No production collection started.

**Top-20 + 20%-audit smoke, pre-registered on the same 100 XMLs.** Compare against the old full-depth baseline using the exact same manifest positions 0–99 and d20 checkpoint. Required correctness: direct-root openers produce no depth-2 trials; non-audit top-20 misses carry `finish_sweep_censored=true` and never enter the dead/setup denominator; audit top-20 misses exhaust; confirmed 2push roots remain uncontaminated. Report confirmed-root recall against the full baseline, buried finishers recovered only by audit, simulator trials, worker-hours/XML, censored count, and audit rate by true object episode. Gate the 100-XML run behind a target-box one-unit smoke of the final config. **COMPLETE:** collection `58898750`, H5/census `58898766`, watcher `58899235`, all successful.

**100-XML labeling result.** The census contains 111 `(object,goal-region)` episodes across 84 rooms: 67 rejected direct-1push, 16 confirmed genuine-2push, 14 proven dead within depth 2, and 14 censored. On the 106 episodes shared with the old exhaustive baseline, classification was preserved exactly for every solved case: 65/65 direct roots and 15/15 genuine-2push roots recovered; the 26 old dead roots became 14 proven dead + 12 censored, never false-dead. Five additional episodes appeared (2 direct1 / 1 true2 / 2 censored) because the new completed collection emitted object episodes absent from the old census.

**H5 label grammar passed.** The raw H5 has 2,207 rows: 44 roots, 1,601 moved post-push boards, and 562 no-op post-push boards excluded by the final selector. Exact supervision is 269 setup cells at 0.9 + 269 opener cells at 1.0; ceilings are 605 root cells at 0.81 + 35,592 moved-finish cells at 0.9; reachable unknowns are masked (1,436 root + 90,279 moved-finish cells). False exact-zero reachable labels = **0**. Among moved finish winners: 222 rank-1, 40 rank-2–5, 7 rank-6–20, 0 audit-recovered >20. Negative moved boards: 1,047 capped, 272 exhaustively audited no-win, 13 naturally complete no-win.

**Training-row yield.** Under the locked selector, 100 XMLs yield 63 positive/mistake rows (16 positive root rows + 47 finish winners ranked 2–20) and 1,354 eligible negative rows; 222 rank-1-only finish rows and 562 no-op rows are excluded. Therefore the 166,666-positive side is the bottleneck: point estimate **~264,550 XMLs** to fill the exact 200k new block, while the 33,334-negative quota is already cheap. Stage rather than trust this small-n projection.

**Cost result.** The new policy executed 53,226 simulator trials versus 173,902 in the old full-depth 100-XML run, a **69.4% reduction (3.27× fewer trials)** while retaining all 15/15 shared genuine-2push roots. Operational processing totals were 3.64 versus 13.28 worker-hours, but the simulator-trial ratio is the cross-hardware-safe comparison. The two exhaustive-audit stragglers set the array tail at 20–21 minutes; audits are the residual scalability cost. No >20 finisher occurred in the randomly audited subset, so this n=100 smoke validates correctness and root recall but does not estimate rare tail-positive recall.

**Artifacts.** Amarel root `/scratch/dm1487/colossus0_200k_1bf5f7a/smoke100`; `candidates.h5` SHA256 `3fcdd50d07042fd0e26ff40b008d99560a5d318157c7855340be835fff567150`; `census.json` SHA256 `7c82b06244a0c36632f5b8486846842928eb437223a9cdd95c8178a79b176c34`.

**Current state (verified 2026-07-22).** Production collection has not started. The validated 1M manifest remains intact; the top-20 + 20%-audit policy, unknown masking, no-op exclusion, and d20+200k stack builder have passed their gates. Next action is a staged collection with a first yield checkpoint before scaling toward the ~265k-XML point estimate.

## Result

The 100-XML collection/labeling smoke passed. Full Colossus-0 collection, combined d20+200k training, and difficulty×horizon evaluation remain pending.
