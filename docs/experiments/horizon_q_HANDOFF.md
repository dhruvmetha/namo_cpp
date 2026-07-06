---
status: ref
tags: [experiment]
updated: 2026-06-25
---

# Horizon-Q — HANDOFF (what we've built & seen so far)

> **⚠ HISTORICAL / SUPERSEDED (2026-07-06): the budget-conditioned "Horizon-Q" framing was DROPPED.** Horizon/budget-conditioning measured ≈ no-horizon, with **NoHz** (a single value/ranker, "no-horizon") ahead — reactive **40.7 vs 34.1** for the budget/depth-value variant, and NoHz ≥ the "Hz" head on both regimes; at ≤2 pushes the budget input has nothing to do. The live model's job is **first-push (setup) ranking** — current framing in [../problem_and_approach.md](../problem_and_approach.md). Everything below is the **verbatim past-tense record** of what we built and measured (including *why* budget-conditioning was cut) — history, not the current design; all checkpoint paths / numbers stay valid.

> Self-contained brief for a fresh chat. Covers: the **problem**, the **model architecture**, the **full algorithm**
> (labels → training → deploy), the **data versions (v2/v3/v4)**, the **results** (emphasis v2/v3), and the **current
> reframe**. Deeper records: `horizon_q_build_journal.md` (empirical log), `horizon_q_search_redesign_journal.md` (the
> active pivot), `horizon_q_model_registry.md` (exact ckpt paths/numbers). All numbers below are grounded experiments.

---

## 1. The problem (Region Opening, RO)
A 7 cm diff-drive **car** must open a path to a **goal region** that is blocked by **ONE movable object**, by pushing that object with **1–3 pushes** (we work at **H_max = 2**). The object has up to **300 discrete pushes** = **60 contact edges × 5 depths**. The simulator (MuJoCo) is a **perfect, deterministic oracle**, ~1 s per push.

- **Success criterion (FROZEN):** the goal region is "open" iff **≥20%** of 100 goal-region points sampled at the initial state s0 become reachable (`region_opening._validate_opening`; matches the collection labels exactly).
- **Unit of evaluation = one EPISODE = (scene xml, object_id, goal)**, never the xml (one room hosts many episodes).
- **The deploy prize is REACTIVE** (act with ~0 simulations). The sim is a perfect-but-expensive verifier.

## 2. Model architecture — `EdgeCrossAttn` (`sage_learning/src/model/dit/edge_crossattn.py`)
A scene encoder + **60 edge tokens** (one per contact edge) that cross-attend to the scene and self-attend among themselves, then a shared per-edge head emits a value for each of 5 depths. **Verified params:**

- **Input:** 5-channel **64×64** scene crop = {robot, robot-goal, movable objects, static walls, reachable mask}.
- **Scene encoder:** patch-embed (patch=4 → 16×16=256 patches), `dim=192`, **4** scene transformer blocks, 6 heads.
- **Edge tokens (60):** each edge's **contact pixel** is **Fourier-encoded** (`pos_fourier`, `fourier_L=8`) → MLP (`edge_pos`); plus a per-edge learned id (`use_edge_embed`); plus **local features gathered via `grid_sample` (bilinear)** at the contact pixel from the scene feature map.
- **Budget conditioning (the "Horizon"):** `budget_embed = nn.Embedding(max_budget+1, dim)`; `e = e + budget_embed(H)` added **ONCE**, broadcast over all 60 edge tokens, **before** the **4** edge attention blocks (UVFA-style). Scene tokens are H-agnostic.
- **Head:** shared `Linear→GELU→Linear` per edge → **`num_depths(5) × value_bins(51)`** logits → reshaped to `(B, 60, 5, 51)`. This is an **HL-Gauss classification value head** (Stop-Regressing, arXiv:2403.03950): each (edge,depth) cell predicts a distribution over 51 bins of [0,1]; the cell's scalar value = **E[bin]**.
- **Output:** a **60×5 value map** Q(s, edge, depth, H) ∈ [0,1].
- **Policy = top-k of the map; state-value V(s) = mean of the top-5 cell values** (NOT raw max — max is fluke-dominated on OOD states).

**Two architecture variants (the core ablation):**
- **Horizon (Hz):** `budget_cond=True` — the value is conditioned on remaining budget H (query at H=1 vs H=2 gives different maps).
- **NoHorizon (NoHz):** `budget_cond=False` — a single H-invariant "how good is this push" value.

## 3. The full algorithm (end-to-end)

### 3a. What the value means + labels (search-distilled, γ-discounted)
`Q(s, a, H)` = "does push `a` open the region within `H` pushes under best play," trained on **γ-discounted labels** computed by **search in the perfect sim** (exhaustive or sampled tree):
- **opener** (opens in 1) → **1.0**
- **setup** (opens only in 2) → **γ = 0.9**
- **dead** → **0** Labels are **per-cell** over the 60×5 grid; **only reachable cells are supervised** (a loss mask; `sample_k=30` subsamples reachable cells per row). **Crucially: there is NO bootstrap / NO recurrence** — these are flat, precomputed supervised labels (we did NOT train `Q(s,a,2)=γ·max Q(s',a',1)`). [This is the central limitation the redesign targets.]

### 3b. Training
HL-Gauss classification loss to the γ-labels, masked to reachable cells, **budget-conditioned** (the same head is queried at H=1 and H=2; budget supplied via `budget_embed`). 3 seeds per condition. ~14 h/run on one GPU (data-loader bound). This is **search-distilled value learning + ExIt** — the AlphaZero/Expert-Iteration family with **Monte-Carlo (search) targets, no TD bootstrap** — NOT model-free RL, NOT behavioral cloning.

### 3c. Deploy — the model is a RANKER, the sim is the verifier
The model **never predicts effects or checks connectivity** — it only **ranks** the reachable pushes; the **sim executes** the chosen push and the **wavefront checks** if the region opened.
- **Reactive@2** (`eval_reactive_argmax.py`): argmax `Q(s0,·,H=2)` → **sim setup** → argmax `Q(s1,·,H=1)` → **sim finish** → open? = **exactly 2 sims** (forces the "dive" into the finish).
- **Search (best-first)** (`eval_bestfirst.py`): one priority queue of unsimulated pushes; priority = **`combine`** of `Q` and `V` — `combine=q` (raw Q, the canonical setting) or `blend` (0.5Q+0.5V). Pop the top, **sim it**, check open, expand its children at the next budget; stop on first open. **`sim_budget` is the single reactive↔search dial.**
- **`sims-to-solve` decomposes:** `≈ rank(true setup) + rank(true finish | that setup)`. Reactive@2 = the **rank-1 corner**.

### 3d. ExIt loop (on-policy data)
Deploy the model → collect the post-setup states **s1 it actually visits** → exhaustively label the finishes there → retrain the finish. (This is how v3/v4 finish data was made; it fixes the off-policy/deploy-distribution-shift gap.)

### 3e. How the data is collected (the pipeline)
- **Scenes:** small car environments (feb + aug9 sets, generated by `mujoco_env_creator`) — each a room with movable objects and a goal region blocked by one object. Push duration `control_steps_per_push = 550`; 300 motion primitives per object shape (`motion_primitives_1x_car_d5_{square,wide,tall}.dat`); pushes track a pure-pursuit path follower.
- **Collector:** the `region_opening` planner (`RegionOpeningPlanner`) driven by `modular_parallel_collection.py` (multi-worker CPU). Per episode = (object, goal):
  - **1-push (H=1), EXHAUSTIVE:** try **every reachable (edge, depth)** push on the object → execute in sim → check the ≥20% region criterion. Opener → **1.0**, else → **0**. Yields the full 60×5 `f_grid` over the reachable cells.
  - **2-push (H=2), SAMPLED:** for first pushes that don't open in 1 (candidate setups), execute → land in **s1** → **sample ~k=30 second pushes** → if any opens, the first push is a **SETUP** (label **γ=0.9**); if all sampled follow-ups fail → **DEAD** (0).
- **Sampling philosophy [USER decision]:** exhaustive collection beyond 1-push doesn't scale ⇒ we **"sample all levels"** (sampled setups AND sampled finishes). A setup whose ~30 sampled finishes all fail is labeled dead — occasionally a per-scene false zero, but across environments **E[label | cell] = the fraction of working follow-ups**, which BCE/regression converges to. One **exhaustive** depth-2 sweep was kept for the **TEST set only** (the `(a1,a2)→opens` answer-key / oracle pairmap, `exhaustive_pairmap_pure2.pkl`).
- **ExIt (on-policy finish):** the collector **saves each post-push state s1**; `exit_collect.py` then steps either the **model's top-K setups** (on-policy — what the deployed model commits to) or the GT `valid_first_push` setups → lands in the s1 the model actually visits → **exhaustively labels the finishes there**. This is the v3/v4 finish data.
- **Rendering → H5:** each state → **5-channel local crops** (`NAMODataVisualizer.generate_all_masks_highres`, resized to 64×64) → `build_scorer_dataset.py` (+ `add_contact_px.py`) → scorer H5 with keys `{ctx (5×64×64), f_grid, r_mask (reachable), contact_px, H, dead, object_center, xml, ratio}`.
- **Per-episode invariants (HARD gate):** unit = **(object, goal)**, never the xml (one room hosts many episodes); samples matched to episodes by **`object_center` (~0 mm)**; difficulty binned **per episode**; train/val/test split **grouped by ROOM (xml)** so a scene's states never straddle the split.

## 4. Data versions (the v-number tracks how we fixed the FINISH)
All share `m2b` + `h2` + `aug`; they differ only in the **finish** ingredient:
- **m2b** (`v4_hq_m2b_scorer`, 252,805 rows): 1-push openers from **initial** states, **+ dead-end rows**. (Best pure 1-push model = M2b, hard@1 **32.86**.)
- **h2** (`v4_hq_h2_scorer`, 311,324): the **2-push SETUP** data — first-push-only at s0, mixed H=1/H=2 rows (5% of H=2 setups succeed; γ=0.9 labels).
- **aug** (`v4_hq_onepush_h2_aug`, 80,000): 1-push openers relabeled as H=2 rows (opener=1.0) — fixes the H=2 dilution.
- **v2** = m2b+h2+aug + **narrow "postpush" finish** (~300k rows but only ~58k distinct scenes, 4:1 fail-skewed, the WRONG/too-easy task → it overfits: finish train-separation 0.75 → test 0.30).
- **v3** = v2 with postpush **REPLACED by ExIt finish** (`v4_hq_exit_finish*`, ~24k on-policy/valid-setup s1, exhaustive finishes at the true ~7% opener density). The generalization fix.
- **v4** = v3 with ExIt **scaled 24k→47k + dead-s1 coverage** (the "finish rebalance").

## 5. RESULTS (the headline table — 3 seeds, region criterion, n=1018 pure-2-push episodes)
**Reactive@2** = forced-dive (argmax setup → argmax finish, 2 sims). **Best-first@2 (combine=q)** = solved within ≤2 sims by value-guided search. **dive tax** = reactive − best-first (how much forcing the dive buys). **s@900** = search ceiling (≤900 sims). Eval dirs: `/scratch/dm1487/eval/{reactarg_*,bfq_*}`.

| cell | reactive@2 | best-first@2 (q) | dive tax | s@900 (ceiling) |
|---|---|---|---|---|
| **Horizon-v2**   | 38.5 ± 2.1 | 27.3 ± 2.2 | **+11.2** | 97.6 ± 0.5 |
| **NoHorizon-v2** | 38.2 ± 3.0 | 34.9 ± 2.6 | +3.3 | 95.0 ± 0.4 |
| **Horizon-v3**   | **43.0 ± 2.8** | 32.2 ± 3.5 | +10.8 | 97.9 ± 0.2 |
| **NoHorizon-v3** | **40.7 ± 0.3** | 37.8 ± 0.3 | +2.9 | 95.4 ± 0.5 |
| **Horizon-v4**   | 40.4 ± 2.2 | 30.2 ± 0.8 | +10.2 | 97.5 ± 0.3 |
| **NoHorizon-v4** | 41.2 ± 0.4 | 38.3 ± 0.3 | +2.9 | 95.8 ± 0.3 |
| RANDOM (5-seed)  | ~4–5 | 3.3 | — | ~91 |

**1-push ranking (separate eval, `eval_scorer` hard@1, label-graded):** champion B30 23.27 → M1 29.40 → M2a 29.62 → **M2b 32.86** (current best 1-push). Horizon-v2 1-push: **H=1 36.0 / H=2 30.7** (the H=2-dilution fix; v1 was 12.2).

### What the results say (verdicts, on the error bars)
1. **ExIt (v2→v3) lifted reactive — REAL, both architectures, MODEST.** Hz **38.5→43.0** (+4.5), NoHz **38.2→40.7** (+2.5). (Single-seed Hz-v3 had read 45.6 — a high seed; the 3-seed mean is 43.0. Single seeds over-call ~5pp gaps.)
2. **v3→v4 rebalance SATURATED.** Hz 43.0→40.4 (−2.6, noise), NoHz 40.7→41.2 (flat). **Scaling finish data + dead coverage bought nothing over v3 — the finish-DATA lever is tapped out.**
3. **Horizon ≈ NoHorizon at every version (reactive is a TIE).** 38.5/38.2 → 43.0/40.7 → 40.4/41.2. The horizon never clearly wins reactive. (Earlier single-seed "Horizon wins reactive" was noise; ±2–3pp seed noise swamps ~5pp gaps.)
4. **The "dive tax" is structural & data-invariant.** Hz ~**+11** every version, NoHz ~**+3**. Horizon's H1/H2 heads don't agree on scale, so its search won't "dive" into the finish (it shops for fresh setups); forcing the dive recovers it to **parity** with NoHorizon (not a win). Better finish data did NOT shrink this ⇒ it needs the recurrence, not data.
5. **The model crushes random** (~38–43 vs ~4–5 reactive, ~8–9×) but at the **search ceiling barely beats brute force** (s@900 ~95–98 vs random ~91) ⇒ the model is a **search-amortizer**; the real win is sim-efficiency + the reactive regime, not the asymptote.

## 6. Eval methodology (so the numbers are interpretable)
- **Test set:** `namo_testset_v1`, key `pure2push.json` (983 scenes / **1018 episodes**, EVERY one is `is_1push_solvable=False & is_2push_solvable=True` — a true setup-then-open problem). Difficulty tiers (by setup-cell density): `pure2push_divisions.json` (hard ≤2 setups / med 3–8 / easy >8).
- **Object-constrained:** the search may push ONLY the labeled `object_id` (else it "solves" via a different easier object). This makes @1-sim solves = 0 = the honest 2-push problem.
- **Region criterion** (not the single xml goal-point) — matches the labels (≥20% of s0-sampled goal pts reachable).
- **Two eval scripts, one shared scoring core:** `eval_scorer.py` (offline RANKING hit@k, no sim — the M-series) and `eval_bestfirst.py`/`eval_reactive_argmax.py` (live SOLVE, runs the search/sim). Consistent + comparable.

## 7. Current reframe (the active direction — see `horizon_q_search_redesign_journal.md`)
The v2→v3→v4 saturation + the structural dive tax drove a pivot: **the model is a sims-minimizing SEARCH HEURISTIC (a ranker), not a value-classifier; the objective is E[sims-to-solve] = a COST-TO-GO measured in SIMS (our γ-labels discount DEPTH, which is ~constant at H=2 — the wrong cost).** A pairmap sims-decomposition (oracle ladder, naive 31.5 sims → 2.0) showed the **finish ranker (D2)** is the dominant lever (saves 15–19 sims every tier incl hard) and **subsumes** the setup-findability idea; the **recurrence (D3)** fixes the structural dive tax. Planned next:
- **D2 — finish ranker:** add a **pairwise-margin ranking loss** (opener > hard-negative impostors; NOT softmax, which collapses the **multi-modal** opener set) on the existing ExIt finish rows. Keep the per-cell value for calibration.
- **D3 — recurrence:** train the setup value as `γ·V_finish(s1)` (bootstrap off the calibrated finish) ⇒ finishability-aware setups + ties the scales so search dives (the +11pp). Gated AFTER D2.
- **Open caveat:** the finish is **cheap to verify (1 sim)** — under "sim the top-k," the current head may already suffice except on the **hard tail**; a Phase-0 measurement (current finish head's realized rank-of-opener on hard, and whether the opener is even distinguishable vs aliased) should gate building D2.

## 8. Key paths
- Code: `sage_learning/src/model/dit/edge_crossattn.py` (arch), `src/model/classifier_module.py` (loss), `src/data/scorer_data.py` (data). Evals: `namo_cpp/scripts/sandbox/{eval_reactive_argmax,eval_bestfirst,eval_m3, eval_scorer}.py`. Aggregator: `scripts/sandbox/agg_seed_table.py`.
- Ckpts/numbers: `docs/experiments/horizon_q_model_registry.md` (NEVER glob — wandb-hash dirs).
- Python: `/scratch/dm1487/envs/namo/bin/python`; build C++ bindings via `./build_python_bindings.sh`.
