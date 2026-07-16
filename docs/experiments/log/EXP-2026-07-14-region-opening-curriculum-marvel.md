---
status: active
thread: rl_loop
robot: car
updated: 2026-07-16
supersedes: EXP-2026-07-12-opener-curriculum-loop (buggy lineage — retracted, see below)
---

# EXP-2026-07-14 — Region-Opening curriculum, clean restart (Marvel lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply (beat random, fewer sim calls, every tier). This card is the *reproducible framework* for building that ranker; it does not restate the problem.

## The one sentence

Learn the ranker via a **curriculum ladder** (1-push → 2-push → …), running **pure DAgger within each stage**, on a **clean, correctly-labeled, in-sync-generated** data pipeline — starting from a **balanced bootstrap seed**.

## Why this design (the motivation, rung d / d-1 / …)

**Why a curriculum at all: SEARCH COST.** Region-opening is tree search over pushes, and depth costs `branching^depth` (~300 pushes/node, so 2-push ≈ 300², 3-push ≈ 300³). Exhaustive deep search does not scale (the "no exhaustive ground truth at scale" constraint). The curriculum keeps us at **effective depth 1 at every rung**: the rung-(d-1) model collapses rung-d's search from `b^d` to ~`b·k` (rank the b first-pushes, verify each continuation with the previous rung's top-k instead of a full sub-search). Exponential-in-depth becomes ~linear-per-rung, paid one rung at a time. It breaks a real chicken-and-egg: you can't get deep value labels without deep search, and you can't afford deep search without the value function. This is Expert Iteration with the depth made explicit and controllable rather than left to emerge from self-play.

**It is ONE ranker, not one model per stage.** The model is a single value/ranker that orders the pushes available at whatever state it's in; "1-push vs 2-push" is only how deep the search expands using that same ranker. A "setup" and a "finish" are the same ranker queried at a parent vs a child node, not two networks. (Ant-Man / Beast are curriculum *stages*, not separate models.)

**Discounting is what lets one ranker order setups correctly.** Value of a push ≈ `γ^(pushes-to-open)`: a direct opener ≈ γ, a setup (opens in 2) ≈ γ², a push that never opens = 0. For any γ<1 this yields `direct > setup > dead` automatically, which does two needed things at once: (a) lifts setups above dead pushes, which *is* the fix for the old "setup bottleneck" (a myopic opens-now model scores a setup 0 and buries it), and (b) keeps direct openers preferred so search tries to open in one push first. It generalizes to any depth (`γ^d`), so **no horizon conditioning / no Q(s,a,H) is needed** — γ encodes depth-to-solution in one scalar. γ is the only real knob (it sets the setup-vs-open margin; the old hand-set target 0/0.9/1 is just γ≈0.9 for a one-step setup). Collecting rung-d labels = collecting the `γ^d` value targets for that depth.

**Labeling = cheap-verify + exhaustive-fallback, and recycle the fallbacks.** For a rung-d problem: rank candidate first-pushes (setups), simulate, and check the continuation with the rung-(d-1) model's top-k. This is sim-confirmed, so PRECISION is perfect and the only error mode is a *miss*. A first-push whose continuation shows up in top-k is a confirmed-good setup, cheaply. Only when top-k finds nothing do you fall back to exhaustive search, which resolves the ambiguity ("no continuation exists" vs "the model just missed it") into a TRUE label. That avoids the setup under-counting bug (41.8% of the old "never opens" labels in solvable scenes were actually real setups). The fallbacks are dual-purpose: "model missed but a solution exists" cases are hard examples that improve rung-(d-1)'s recall (DAgger at depth d-1), and the resolved labels train rung d with true, not model-guessed, targets.

**Why per-rung quality matters (the real reason to "cap" a rung before moving up).** The rung-(d-1) model's recall@k directly sets rung-d's cost: every miss drops to the exhaustive fallback, which is exactly the `b^d` cost the curriculum exists to avoid. So making a rung good before climbing is not perfectionism, it is what keeps the next rung tractable and its labels clean. The right cap criterion is therefore **recall@k on real post-first-push states** (is the continuation reliably inside the top-k), not the deploy-time @1 headline.

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

All generation is **60% aug9_car_v3 / 40% feb_car** [USER — aug9 is denser/more interesting; worth its higher retry cost].

```
BOOTSTRAP  antman-0:
  generate a fresh batch (fixed gen, 60/40 aug9/feb) → label (goal-opening)
  → build a 50k-episode seed (train+val, split BY ROOM ~90/10), difficulty-LEANED, hard RELAXED:
       easy ~25k / med ~20k / hard ~all-available (~2-5k).
     [see PILOT FINDING: hard is the RARE bin, not easy — 20k hard would cost ~1M scenes/~4h]
  → train antman-0.

PURE DAgger  antman-r (r = 1,2,…):
  1. GENERATE  fresh scenes (fixed gen, 60/40 aug9/feb)
  2. SCREEN    best-first with antman_{r-1}:
       solved & opener in top-5  → DROP (model nails it)
       solved & opener NOT top-5 → KEEP (the model's 1-push mistake — THE signal)
       UNSOLVED (no 1-push opener) → 2-push → phase2_bank/ (Beast stage), NOT this stage
  3. LABEL     exhaustive goal-opening on the KEPT scenes; add up to 50k episodes this loop
  4. TRAIN     accumulate ALL clean SOLVABLE data + retrain antman_r
       (NO dead in training [USER — no negatives]; ALL dead → phase2_bank for Beast — it IS the 2-push material)
  5. EVAL      solve@k by fixed-cut difficulty vs random
  6. LOOP/STOP  improved → r+1 ; keep-yield→0 / plateau → stage done

LADDER: 1-push loop to plateau → start the 2-push stage (beast-*) on the banked post-setup scenes, same loop.
```

## Decisions locked [USER 2026-07-14]

1. **Fresh restart** — discard the buggy lineage; new Marvel-named lineage.
2. **Pure DAgger** (mistake-targeting) after the seed — NO forced balance beyond the bootstrap.
3. **Bootstrap = 50k SOLVABLE episodes, difficulty-LEANED, hard RELAXED, NO dead** [USER 2026-07-14, post-pilot — SUPERSEDES the earlier 20k/20k/20k]. Cap easy, take all med + all-available hard; train+val split by room. The pilot FALSIFIED "easy is rare": easy DOMINATES (73–77% of solvable), genuine-1-push-hard is the RARE bin (3.5% in both templates), so 20k hard would cost ~1M scenes. No dead-as-negatives — dead banks for Beast. All generation **60/40 aug9/feb** [USER — aug9 more interesting]. See Run log pilot entry.
4. **Goal-opening labels** (`target_goal_region` ON) — always.
5. **In-sync generator** (the `_runtime_validate_adjacency` fix, verified) — always.
6. **Fixed-cut difficulty**, never tertiles.
7. **Curriculum ladder** 1-push → 2-push, DAgger within each stage.

## Eval

Solve@k by difficulty (easy/med/hard, fixed cuts) on the held-out `namo_testset_v1`, vs the random ranker — both **reactive @1** and **with-search @k**. Bar: beat random on every tier; the with-search curve dominates random's (any solve-rate at a fraction of the sim calls).

## Run log

**Gen↔label sync fix — VERIFIED (2026-07-14, `mujoco_env_creator@55badcb`).** Re-enabled `_runtime_validate_adjacency` to re-check each placement against the labeler's own live `get_region_snapshot` (robot/goal placed). Proof: label-time `goal_region_not_in_snapshot` drops **77.8% (fix OFF) → 0.0% (fix ON)** on a matched 80-scene sample; ~4.5 ms/sample. Generator now emits only true 1-hop scenes.

**Pilot (2026-07-14) — the premise FLIP.** ~560 scenes gen (fixed pipeline) + goal-opening labels, aug9_car_v3 & feb_car. "Easy is rare" is FALSE. Among SOLVABLE episodes: easy/med/hard = **73/23/3.5%** (aug9) and **77/20/3.5%** (feb) — easy dominates, **genuine-1-push-hard is the rare bin (3.5% both)**. Solvable/dead: aug9 42%/48%, feb 72%/25% (dead = 1-hop-adjacent goal that needs 2 pushes = free Beast material). Per-scene easy/med/hard/dead yield: aug9 0.39/0.13/0.019/0.62, feb 0.65/0.17/0.030/0.30. Gen accept-rate (runtime-validate % kept): aug9 27%, feb 90% (feb sparser → fewer multi-hop rejects). Cost to harvest 20k easy ≈ 31–51k scenes (<0.5 h @2k cores); 20k *solvable*-hard ≈ 0.7–1M scenes (~4–8 h) — the flipped bottleneck. sr from `algorithm_stats["primitive_trial_log"]` (openers/tried). Artifacts: `scratch_namo/tmp/antman0_pilot/SUMMARY.json`.
**Decisions from the pilot [USER]:** relax hard; seed = 50k SOLVABLE (no dead); lean composition; 60/40 aug9/feb; dead → Beast bank.

### RESULTS — 1-push ladder (Ant-Man), rounds 0–5 [2026-07-15/16]

Solve@1 by fixed-cut tier on held-out `namo_testset_v1` (1,323 eps; hard 204 / med 421 / easy 698), best-first hmax1 budget300 NOHZ base. Screener for round r = antman_{r-1}.

| model | screened | kept | keep% | train rows | easy@1 | med@1 | **hard@1** | all@1 | hard@20 |
|---|---|---|---|---|---|---|---|---|---|
| antman-0 (seed) | — | — | — | 50,000 | 92.7 | 63.7 | **23.0** | 72.7 | 85.8 |
| antman-1 | 9,033 | 493 | 5.46 | 50,528 | 93.7 | 67.7 | **24.0** | 74.7 | 86.3 |
| antman-2 | 715,432 | 38,266 | 5.35 | 90,700 | 95.6 | 73.2 | **28.4** | 78.1 | 91.2 |
| antman-3 | 686,951 | 28,950 | 4.21 | 120,657 | 96.3 | 77.2 | **32.8** | 80.4 | 91.7 |
| antman-4 | 682,577 | 30,660 | 4.49 | 151,218 | 96.7 | 81.2 | **39.2** | 82.9 | 94.6 |
| antman-5 | 449,284 | 16,959 | 3.78 | 167,655 | 97.1 | 78.9 | **42.6** | 82.9 | 92.6 |
| _random ranker_ | — | — | — | — | 62.6 | 19.2 | **1.5** | ~39.4 | 37.7 |

**Headline: hard@1 23.0 → 42.6 over five rounds, ~28× random, no plateau.** Gains concentrated at low k (the ranker finds openers *sooner*; sim verifies for free). Beats random on every tier at every k.

**Findings:**
- Round 1 (+528 rows) = noise (+1.0, within ~0.3 mm eval jitter). Rounds 2–5 are real (+4.4/+4.4/+6.4/+3.4).
- **Volume tracks the climb**, but keep-rate falls (5.46% → 3.78%) as the model eats its own error distribution, and per-row efficiency *rises* (round 5 best: ~2.0e-4 hard@1/row on the fewest rows). First hint DAgger targeting matters, n=2, not proven.
- Tiers: easy SATURATED (97@1, high-k maxed). med@1 still live (peaked 81.2 @a4, dipped 78.9 @a5). hard = main headroom.
- **Round 5 was a REDISTRIBUTION, not a lift:** gained hard@1 (+3.4), hard@2 (+6.4) but dropped med@1 (−2.3) and hard@20 (−2.0). Sharpened the top of the ranking at a cost to the tail. Also confounded: 449k vs ~700k for rounds 2–4.

**OPEN QUESTIONS (neither answered):**
1. **DAgger targeting vs plain volume** — no control run. Every round confounds mistakes + more rows.
2. **Plateau** — unclear; round 5 was undersized, so its smaller +3.4 can't be read as slowing.

**Beast (2-push) dataset accumulated for FREE:** 72,521 labeled-dead episodes with full `(xml, object_id, robot_goal)` identity (seed 54,268 + r1-r5 ~18,253), plus ~865k unlabeled screen leads (`phase2_bank/screen_dead_scenes.txt`, xml-only). ~42% of every screen is dead, model-stable across rounds.

**Execution notes (bugs found + fixed, so we don't repeat them):**
- **orch wait bug:** `orchestrator.sh` called undefined `wait_amarel_jobid` (typo for `wait_amarel_jobs`) → command-not-found → no wait → halted r1 at "rows=0". FIXED: defined `wait_amarel_jobid <jobid>` in `lib.sh` (blocks via `squeue -j`, +10s submit-race guard).
- **node damage:** `MUJOCO_GL=egl` in `build_array.sbatch` wedged **25 halk nodes** (unkillable GL init on GPU-less nodes; slurmstepd "not ending with signals"; 2 NODE_FAILs). GUARDED with `--exclude=halk[0001-0159]`; **root cause still OPEN** (build renders via matplotlib/Agg + cv2 + reads many pkls; halk-only, not CPU/heat; screens never wedged anything). Paul (admin) flagged it; email drafted.
- **count truncation:** `timeout N find|wc -l` truncates mid-count → 5× scene undercounts + a duplicate ledger row. Never trust a timed-out count. [[feedback_check_process_owner]] neighbourliness: pool gen capped 128 cores; Amarel fair-use ≤200 background / bursts ≤5h.

**NEXT (planned): redo round 5 at 700k + fold in the volume control.** Reuse the 449k screen + 16,959 labels (screened vs antman-4, budget 300, on disk); generate +250k same config; screen vs antman-4; label the delta; rebuild accumulated from the through-round-4 base + all-700k mistakes → new antman-5. From the same 700k pool also train a control (+iid labeled rows, same count). Answers plateau AND targeting-vs-volume in one clean round.
