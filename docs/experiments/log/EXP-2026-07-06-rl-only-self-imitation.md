---
type: experiment
status: idea
created: 2026-07-06
commit:
metric:
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

## Plan
_(fill on launch — spec agreed in chat)_
- **MDP:** state = scene crop (existing encoder); actions = masked reachable candidates (existing 60×5 head); reward = 1 on goal-region open else 0, γ discount; horizon 10 pushes, early-stop on open. Car robot, `namo_config_complete_skill15_car_1x.yaml`.
- **Algorithm (off-policy self-imitation):** actor = filtered BC — train π ONLY on solved trajectories (failures censored, zero gradient); critic = V(s) on Monte-Carlo returns INCLUDING zeros from failed rollouts (unconditional V^π = P(solve)×speed), recency-weighted toward recent generations. NO bootstrapping. AWR stays OFF unless the identifiability trigger fires (≥50% of hard episodes have a first action tried ≥4× with ≥2 solves).
- **BC weighting [GPT-5.5 fix]:** uniform across steps WITHIN a trajectory (setup weighs as much as finish); across trajectories of one episode, mass ∝ 2^−(T−T_min); every episode totals weight 1. NOT γ-return-weighted (that mutes the setup — backwards).
- **Buffer [GPT-5.5 fix]:** all generations kept (off-policy); per episode keep ≤8 solves, only T ≤ T_min+2, keyed by first-two action IDs, ≤2 per first-action bucket (setup diversity, no best-only dedup). Periodic revalidation of near-threshold solves (~0.3mm sim nondeterminism).
- **Exploration:** rollout temperature + ε=0.10 uniform (residual protection) + **forced first-action sweeps** on hard episodes lacking setup diversity: force only the first push to the least-tried of the policy's top-8 absent from the success buffer (≤4 attempts/action/gen), then ε-greedy; fresh rollout from s0 each time — no branching, no mid-rollout resets (= depth-1 MCTS used for data collection only).
- **Arms:** A = π₀ from scratch (uniform over candidates); B = π₀ BC-pretrained on the existing 1–2push solve archive. Same sim budget, same generations, side-by-side stratified curves.
- **Pool & splits:** existing rooms/episodes, mixed difficulty; room-held-out 80/10/10 frozen before gen-0; memorization tripwire (train +5pt while dev +<1pt over 2 gens, or train−dev >10pt).
- **Per-generation report:** open@1/2/5/10 by difficulty × horizon (never horizon-10-only), buffer composition (unique solves per tier), hard-episode positive coverage.
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
