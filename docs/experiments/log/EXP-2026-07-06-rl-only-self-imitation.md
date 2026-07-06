---
type: experiment
status: live
created: 2026-07-06
commit: 092faec
metric: "Phase 0 gate GREY ZONE (arm(i)-any 76.9±0.8) resolved by the mandated k-sweep to PROCEED: the reactive-vs-search gap is DOMINANTLY learnable setup-ranking. wrong-setup=74.6% of greedy failures (44.3% of all episodes); finish-given-correct-setup mostly works (aliasing 24.3% of failures, control ~1%); and a finishable setup sits in the model's top-8 for 82.5% of episodes (hard 70.4) — surfaced but mis-ranked, exactly what RL targets. Hard-tier caveat: top-8 finishable ceiling 70.4, finish-given-oracle-setup caps 56.3. Anchor open@2=40.7±0.2 reproduces reactive-MPC exactly."
tags: [experiment, rl, self-imitation]
---
# RL-only loop: off-policy self-imitation at depth 10 (no search at train or deploy)

> Born from the 2026-07-06 chat (RL-only commitment [USER]) + the GPT-5.5-xhigh consultation (3 rounds, adversarially interrogated by an Opus agent; transcripts in that session's scratchpad). Sits beside — not replacing — [../policy_value_search_hypothesis.md](../policy_value_search_hypothesis.md): this trains the same two heads but with rollout-RL as the data engine instead of ExIt search.

## Hypothesis
_(you, via chat 2026-07-06)_ **Pure RL — forward rollouts only, no branching/resets at collection, greedy at deploy — can learn both the setup and the finish, because trajectory credit assignment puts setup pushes into the training data as positives.** The measured reactive-vs-search gap (40.7 reactive vs ~96 search on 2push) is substantially the *myopic scorer's* fault, not an irreversibility ceiling: irreversibility amplifies classification error, it does not impose a floor (deterministic, fully-observed-up-to-crop MDP). Falsifiable form: hard-2push greedy open-rate climbs across generations and clears the kill thresholds below.
**Registered external prediction** (GPT-5.5 xhigh, 2026-07-06, before any run): after 5 generations × ~1e6 sims — **2push-all ~70% (range 60–79), 2push-hard ~53% (range 39–67)** greedy on held-out rooms. Beating the upper range = setup geometry transfers unusually well; landing below ~50/~35 = coverage failure.

## Phase 0 — GATE (run before the loop; uses the pure2push eval luxury)
Oracle reactive decomposition on the pure-2push set, greedy protocol (NOT search-conditioned — the existing 97.6% finish-given-setup number is search-conditioned and does not answer this):
1. oracle setup + learned greedy finish;
2. learned setup + oracle continuation;
3. classify every miss: wrong-setup / failed-finish / aliasing-or-control.
**Gate:** (1) ≥85% ⇒ the gap is learnable setup ranking — proceed to Phase 1. (1) <65% ⇒ representation/control dominates — STOP, rethink before burning generations. Grey zone ⇒ teacher-forced finish diagnostic at setup ranks k=1,2,4,8.

### Phase 0 — Plan _(Claude, 2026-07-06, on launch)_
**CAR only**, testset `namo_testset_v1` pure2push key (`labels/pure2push.json`, n=1018 episodes: hard 371 / med 409 / easy 238), region-open criterion, object-constrained to the labeled object, **greedy forced-dive** protocol (same setup+finish machinery as `_reactive_search.md` / `eval_reactive_argmax.py`: setup queries H=2, finish queries H=1). 2push-only ⇒ the horizon split is trivially 2push (stated explicitly).
- **Oracle setup key = `valid_first_push`** per episode: the exhaustive set of `[edge,depth]` first pushes on the labeled object that admit a 2-push solution. Verified the model's greedy candidate `(edge,depth)` always lands inside the GT `tried_first_push` grid (smoke 12/12), so key-based classification is clean. NOTE the key is a conservative LOWER bound (GT's 2nd-push verify budget was limited — a smoke episode opened greedily despite `g1∉valid_first_push`), so the arm-(iii) taxonomy is **sim-grounded**, not key-only.
- **Arm (i) — oracle setup + learned greedy finish:** for each GT-valid setup v (ordered by model score, capped at 12/episode), execute v, then the model greedily picks the 2nd push (H=1), check region open. Report **arm(i)-any** (opens for ≥1 valid setup = the "is there a learnable finishable chain" ceiling, the GATE number), **arm(i)-modelpref** (top-model-scored valid setup = realistic point estimate), **arm(i)-mean** (setup-averaged).
- **Arm (ii) — learned setup + oracle continuation:** the model's greedy #1 first push ∈ `valid_first_push` ⇒ resulting state admits a GT solution = recoverable (key, zero-sim). Plus **arm(ii)′-sim** = model's greedy setup lands in a state where SOME 2nd push opens (sim-verified, robust to the key's conservativeness).
- **Arm (iii) — miss taxonomy** over episodes the fully-learned greedy@2 baseline FAILS (= the ~59% below 40.7): simulate at the model-executed s1 — if an opening 2nd push exists → **aliasing_or_control** (setup landed openable, model's greedy finish missed it); else if g1 was a GT-valid setup → **failed_finish** (good setup executed into a dead state: drift/near-threshold); else → **wrong_setup** (GT+sim agree the setup is dead).
- **Reuse:** `eval_reactive_argmax`/`eval_m3` helpers (`rank_first_pushes_h2`, `make_action`, `goal_open_pts`, `sample_goal_points`); NoHz-v3 registry ckpts (3 seeds s1/s2/s3, best-val ep012 — reuse, no retrain); binning via `pure2push_divisions.json` `division` field. Baseline open@2 must reproduce the reactive-MPC anchor **40.7 all / 59.8 easy / 42.5 med / 26.3 hard**.
- **Compute:** iLab `unlimited` SLURM (CPU, `OMP_NUM_THREADS=2`), sharded 20-way × 3 ckpt-seeds. Model-scoring path confirmed clean off-Amarel (smoke, no `fast_scorer` TypeError). New files under `scripts/rl_loop/`; out dir `$NAMO_SCRATCH/eval/phase0_gate/`.

### Phase 0 — Run _(Claude, 2026-07-06)_
Base commit `092faec` on `feat/horizon-q-redesign` (worktree `agent-aa6c1e592dbdacbf9`; orchestrator owns the commit). Harness `scripts/rl_loop/phase0_oracle_decomp.py` (all 3 arms + baseline in one pass), launcher `scripts/rl_loop/phase0_gate.slurm`, aggregator `scripts/rl_loop/agg_phase0.py`.
- **Jobs (iLab `unlimited`, array 0-19 each, submitted 2026-07-06):** s1 = 172188 · s2 = 172208 · s3 = 172209. Out dirs `/common/users/dm1487/scratch_namo/eval/phase0_gate/s{1,2,3}/`.
- NoHz-v3 ckpts: s1 `qfull_nohz_v3_v4hq_s1/.../wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt` · s2 `.../s2/.../kzph0acr/.../epoch012-val_loss0.6922.ckpt` · s3 `.../s3/.../dlopoael/.../epoch012-val_loss0.6874.ckpt` (root `/common/users/dm1487/scratch_namo/sage_outputs/scorer/`).

### Phase 0 — Result + Verdict
_(Claude, 2026-07-06 — CAR, pure2push key n=1018 [hard 371 / med 409 / easy 238], region criterion, greedy forced-dive, 2push-only. Mean ± std across 3 NoHz-v3 ckpt-seeds. Aggregator `scripts/rl_loop/agg_phase0.py`, plot `AGG/phase0_gate.png`. Eval dir `/common/users/dm1487/scratch_namo/eval/phase0_gate/`.)_

**Anchor — PASS (exact).** Baseline fully-learned greedy@2 open-rate reproduces the reactive-MPC card byte-for-byte: **40.7 ± 0.2 all** (easy 59.8 ± 3.6 / med 42.5 ± 1.6 / hard 26.3 ± 0.7) vs the card's 40.7 / 59.8 / 42.5 / 26.3. The harness and protocol are validated; all numbers below are trustworthy.

**GATE: arm(i)-any (all) = 76.9 ± 0.8 → GREY ZONE (65–85).** Pre-registered response: teacher-forced finish k-sweep (below).

| metric | easy | medium | hard | all |
|---|---|---|---|---|
| BASELINE open@2 (anchor) | 59.8 ± 3.6 | 42.5 ± 1.6 | 26.3 ± 0.7 | **40.7 ± 0.2** |
| **ARM(i)-any** (oracle setup + learned finish, ≥1 valid setup) [GATE] | 95.5 ± 0.7 | 84.7 ± 1.6 | 56.3 ± 1.0 | **76.9 ± 0.8** |
| ARM(i)-modelpref (top-scored valid setup) | 74.1 ± 0.9 | 61.6 ± 2.8 | 51.1 ± 1.1 | 60.7 ± 1.3 |
| ARM(i)-mean (setup-averaged) | 69.9 ± 1.4 | 60.5 ± 1.6 | 50.7 ± 1.6 | 59.1 ± 1.2 |
| ARM(ii) recoverable (learned setup ∈ GT valid, key) | 48.3 ± 3.6 | 35.7 ± 3.2 | 18.7 ± 1.0 | 32.4 ± 0.7 |
| ARM(ii)′ recoverable (sim-grounded: some 2nd push opens) | 76.2 ± 2.1 | 58.4 ± 0.9 | 37.9 ± 1.4 | 55.1 ± 0.5 |

**ARM (iii) miss taxonomy** (sim-grounded; over episodes the fully-learned greedy@2 FAILS, n_fail = 604 ± 2 = 59.3% of 1018):

| miss type | easy | medium | hard | all (share of FAILURES) | all (share of ALL episodes) |
|---|---|---|---|---|---|
| **wrong_setup** (model's setup dead: no finish exists there) | 56.0 ± 0.9 | 71.5 ± 1.4 | 83.8 ± 1.9 | **74.6 ± 1.4** | **44.3 ± 0.7** |
| **aliasing_or_control** (setup landed openable, greedy finish missed) | 40.8 ± 0.5 | 27.5 ± 0.7 | 15.7 ± 2.0 | **24.3 ± 1.1** | **14.4 ± 0.7** |
| **failed_finish** (GT-valid setup drifted to a dead state) | 3.2 ± 1.1 | 1.0 ± 0.7 | 0.5 ± 0.3 | 1.1 ± 0.2 | 0.7 ± 0.1 |

![[phase0_gate.png]] _(L: arms by difficulty with the 85/65 gate lines; R: miss taxonomy stacked as share of all episodes + solved. PNG at `/common/users/dm1487/scratch_namo/eval/phase0_gate/AGG/phase0_gate.png`.)_

**Verdict [on numbers]: the reactive-vs-search gap is DOMINANTLY a learnable setup-ranking problem, not a finish/control problem — the RL loop's core premise holds, but the hard tier carries a real finish component.** Three facts:
1. **Setup ranking is the bottleneck.** The model's greedy first push is a GT-valid setup only **32.4%** of the time (key) / **55.1%** sim-grounded — i.e. ~45–68% of the time it commits to a setup with no continuation. And **wrong-setup is 74.6% of all greedy failures** (44.3% of every episode), rising to **83.8% on hard**. This is exactly what trajectory-credit RL self-imitation targets (good setups enter the training data as positives).
2. **The finish mostly already works, given a correct setup.** Hand the model a correct setup and it finishes **60.7%** greedily (modelpref) / **76.9%** at best (any valid setup). Finish-ranking failure (aliasing) is only **24.3% of failures / 14.4% of episodes**, and pure control failure is **~1%** — control/physics is NOT the wall.
3. **The grey zone is a hard-tier finish ceiling.** Easy/med clear the 85 bar comfortably given oracle setup (95.5 / 84.7); the gate sits at 76.9 only because **hard caps at 56.3% even with an oracle setup** — on hard scenes, ~44% can't be finished greedily even from a correct setup. Whether that is recoverable mis-ranking or a coverage wall is exactly what the k-sweep resolves.

### Phase 0 — grey-zone k-sweep (teacher-forced finish over model setup ranks k=1,2,4,8)
_(Claude, 2026-07-06 — `scripts/rl_loop/phase0_ksweep.py` + `agg_ksweep.py`, 3 ckpt-seeds; eval dir `/common/users/dm1487/scratch_namo/eval/phase0_ksweep/`. setup-hit@k = a FINISHABLE setup (oracle finish: some 2nd push opens) appears in the model's top-k setup ranking. Cross-check PASS: hit@1(key)=32.4 = arm(ii)-key exactly; hit@1(sim)=54.0 ≈ arm(ii)′-sim 55.1 within the FIN_CAP=25 cap.)_

**setup-hit@k — SIM-GROUNDED (oracle finish):**

| k (model's top setups) | easy | medium | hard | all |
|---|---|---|---|---|
| hit@1 (greedy setup) | 75.2 ± 2.1 | 57.2 ± 1.0 | 36.8 ± 0.9 | 54.0 ± 0.3 |
| hit@2 | 83.1 ± 2.3 | 68.5 ± 0.4 | 47.3 ± 2.2 | 64.2 ± 0.5 |
| hit@4 | 89.5 ± 1.6 | 78.1 ± 2.3 | 59.4 ± 1.3 | 73.9 ± 0.3 |
| hit@8 | 93.8 ± 1.7 | 87.0 ± 0.8 | 70.4 ± 0.1 | **82.5 ± 0.5** |

**setup-hit@k — KEY-BASED (GT valid_first_push, conservative LB):** hit@1 32.4 → hit@2 43.5 → hit@4 56.5 → **hit@8 70.6** (all); hard 18.7 → 27.9 → 41.2 → **55.3**.

**Grey-zone resolution [on numbers]: the residual is DOMINANTLY mis-ranking, NOT a coverage/representation wall — the model SURFACES a finishable setup in its top-8, it just fails to rank it #1.** A finishable setup climbs from **54.0% (rank 1) → 82.5% (top-8)**, i.e. **+28.5pp** available from re-ordering alone (hard: **36.8 → 70.4, +33.6pp**). The representation is therefore not the bottleneck for the bulk of the gap; the ordering is. This is exactly what RL self-imitation learns (reinforce the first pushes that led to a solved rollout → promote them to rank #1), and a zero-undo width-8 setup beam would already recover most of it. **This tips the grey zone to the PROCEED side.**

**Final Phase-0 verdict [on numbers]: PROCEED to Phase 1 — the RL-only loop is well-founded — with a hard-tier caveat.** The reactive-vs-search gap is dominantly a **learnable setup-ranking** problem: (a) the model's greedy first push is a valid setup only 32–54% of the time, and wrong-setup is 74.6% of all greedy failures; (b) given a correct setup the finish mostly works (60.7% greedy / 76.9% best), with only ~24% finish-aliasing and ~1% control failure; (c) the k-sweep shows a finishable setup sits in the model's top-8 for 82.5% of episodes — the model has the right candidate, just mis-ordered. All three point at ranking, which is precisely what trajectory-credit RL targets. **Caveat (hard tier):** even the top-8 finishable-setup ceiling is 70.4% on hard and the finish-given-oracle-setup caps at 56.3% — so ~30% of hard episodes have no finishable setup even in the model's top-8, and the hard finish carries a genuine (smaller) representation/control residual. Expect RL to lift easy/med strongly and hard partially; this is consistent with the registered GPT-5.5 forecast (2push-all ~70, 2push-hard ~53). Recommend launching Phase 1 with the hard-tier exploration floor (forced first-action sweeps) armed from gen-0, and tracking hard-tier unique-solve coverage against kill-signal 3.

### Phase 0 — files (for merge-back)
Scripts (worktree `agent-aa6c1e592dbdacbf9`, under `scripts/rl_loop/`): `phase0_oracle_decomp.py`, `phase0_gate.slurm`, `agg_phase0.py`, `plot_phase0.py`, `phase0_ksweep.py`, `phase0_ksweep.slurm`, `agg_ksweep.py`. Eval outputs (`$NAMO_SCRATCH/eval/`): `phase0_gate/{s1,s2,s3,AGG}` + `phase0_gate/AGG/phase0_gate.png`, `phase0_ksweep/{s1,s2,s3,AGG}`. No commits made (orchestrator owns).

## Plan
_(fill on launch — spec agreed in chat)_
- **MDP:** state = scene crop (existing encoder); actions = masked reachable candidates (existing 60×5 head); reward = 1 on goal-region open else 0, γ discount; horizon 10 pushes, early-stop on open. Car robot, `namo_config_complete_skill15_car_1x.yaml`.
- **Algorithm (off-policy self-imitation):** actor = filtered BC — train π ONLY on solved trajectories (failures censored, zero gradient); critic = V(s) on Monte-Carlo returns INCLUDING zeros from failed rollouts (unconditional V^π = P(solve)×speed), recency-weighted toward recent generations. NO bootstrapping. AWR stays OFF unless the identifiability trigger fires (≥50% of hard episodes have a first action tried ≥4× with ≥2 solves).
- **BC weighting [GPT-5.5 fix]:** uniform across steps WITHIN a trajectory (setup weighs as much as finish); across trajectories of one episode, mass ∝ 2^−(T−T_min); every episode totals weight 1. NOT γ-return-weighted (that mutes the setup — backwards).
- **Buffer [GPT-5.5 fix]:** all generations kept (off-policy); per episode keep ≤8 solves, only T ≤ T_min+2, keyed by first-two action IDs, ≤2 per first-action bucket (setup diversity, no best-only dedup). Periodic revalidation of near-threshold solves (~0.3mm sim nondeterminism).
- **Exploration:** rollout temperature + ε=0.10 uniform (residual protection) + **forced first-action sweeps** on hard episodes lacking setup diversity: force only the first push to the least-tried of the policy's top-8 absent from the success buffer (≤4 attempts/action/gen), then ε-greedy; fresh rollout from s0 each time — no branching, no mid-rollout resets (= depth-1 MCTS used for data collection only).
- **Arms:** A = π₀ from scratch (uniform over candidates); B = π₀ BC-pretrained on the existing 1–2push solve archive. Same sim budget, same generations, side-by-side stratified curves.
- **Pool & splits:** existing rooms/episodes, mixed difficulty; room-held-out 80/10/10 frozen before gen-0; memorization tripwire (train +5pt while dev +<1pt over 2 gens, or train−dev >10pt).
- **Per-generation report:** open@1/2/5/10 by difficulty × horizon (never horizon-10-only), buffer composition (unique solves per tier), hard-episode positive coverage, and the **setup-ranking dashboard [USER 2026-07-06]**: setup-hit@1/@8 + median rank of solving setups on held-out dev (reuse the Phase-0 k-sweep harness; baseline to beat: hit@1 54.0 / hit@8 82.5, hard 36.8/70.4) — the headline curve is setup ranking improving per generation, setups never buried.
- **Compute:** collection = forward rollouts via a new thin collector (reuse env bindings + `modular_parallel_collection.py` patterns — `namo-data-pipeline` skill before writing it), SLURM CPU `main-redhat`; training = sage `train_scorer_edge` pattern on GPU; eval = `eval_scorer.py` on `namo_testset_v1` (canonical) + reactive protocol from [archive/EXP-2026-07-06-reactive-mpc-depth5.md](../archive/EXP-2026-07-06-reactive-mpc-depth5.md).
- **Deferred patches (add ONLY on a measured stall, one at a time):** per-tier rollout budgets → HER goal relabeling → wavefront potential shaping → reverse-generated deep instances. Branching/resets stay banned (the RL-only rule).

## Kill signals (pre-registered)
1. Gen-1 collection, before training: hard-episode positive coverage <50% (fraction of hard training episodes with ≥1 verified 2push solve; repeats on few episodes don't count) ⇒ collection redesign before proceeding.
2. End of gen-1: held-out hard-2push greedy <35% ⇒ the ~70% forecast is falsified; stop and diagnose.
3. Whole approach: hard-tier unique-solve coverage flat across two consecutive generations despite the exploration floor ⇒ coverage failure — the RL-only constraint itself is the binding limit.

## Existing data — what gets reused vs generated fresh
- **Reused:** 1–2push solve archive → arm-B pretraining; existing rooms → rollout pool; `namo_testset_v1` → canonical eval; pure2push GT → Phase 0 only (eval luxury, never training).
- **Fresh:** all loop training data (solved rollouts) is generated by the policy itself — that is the experiment.

## Run
_(auto on launch)_

## Result + Verdict
_(auto)_ — accept iff hard-2push greedy climbs across generations AND clears kill signal 2; headline = greedy open-rate by tier × horizon per generation vs the registered prediction; secondary = π+V dropped into best-first search vs `combine=q` (NoHorizon-v3: reactive 40.7 / best-first 37.8 @2) — the free test of the two-head hypothesis.

## Next
_(tbd)_

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_
