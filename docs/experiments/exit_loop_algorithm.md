---
status: reference
thread: rl_loop
updated: 2026-07-11
---

# ExIt depth-ladder — the training algorithm, AS IMPLEMENTED

This is the **exact spec of what the code does** for EXP-2026-07-10 (the ranker/value loop), file by file. It is deliberately literal — where the implementation **deviates** from the design in [log/EXP-2026-07-10-exit-search-loop.md](log/EXP-2026-07-10-exit-search-loop.md), the deviation is flagged **[DEV]**. Read the card for the *why*; read this for the *what*.

## 0. The reward / label scheme (the one grid everything writes to)

γ = 0.9. Every label lives on the **60×5 grid** = (60 contact points = 4 edges × 15 points) × (5 push depths, `1x_car_d5_` primitives).

Per (state, cell):
| value | meaning | who assigns it |
|---|---|---|
| **1** (= γ⁰) | this push opens the goal *now* (wavefront check passes) | sim |
| **γ^k** (setup = γ¹ = 0.9) | on a discovered solution path, k pushes before the opener | search |
| **0** | searched (to the depth/budget) and no solution found through it — the soft, budget-relative negative | search |
| **−1** | unreachable (geometry — robot can't get there) | wavefront |
| **MASK** | reachable but never tried — no gradient | — |

**[DEV] Reachable-only arm.** Both Q1 and Q2 train with `loss_mask = value_mask × r_mask` — i.e. **the −1 band is stored in the H5 but masked OUT of the loss**. The −1 feasibility fold-in is a deferred A/B. On the trained cells the target is only ∈ {0, 0.9, 1}.

## 1. Rung-1 → Q1 (opener classifier)

- **Config:** `python/namo/data_collection/region_opening_rung1_car.yaml` — `region_exhaustive_mode: true`, `region_sample_k: 25`, `region_sample_restarts: 3`, `region_max_chain_depth: 1`, `region_selection_strategy: cost_first`, `goal_strategy: primitive`, `primitive_prefix: "1x_car_d5_"`.
- **Collect:** `python/namo/data_collection/modular_parallel_collection.py` runs `region_opening` — samples ~25 reachable shoves/episode, executes each as a single push from the start state, records `opened?` per shove in `primitive_trial_log` (+ `reachability_log` for the reachable set).
- **Label:** `scripts/pipeline/build_rung1_h5.py` — **one row per episode** (all 25 shoves share the start-state `ctx`): opener→**1**, tried-didn't-open→**0**, unreachable→**−1**, reachable-untried→**MASK**.
- **Train:** `scripts/rl_loop/train_q1.py` — `EdgeCrossAttn` (60,5) **sigmoid head**, `WeightedClassifierModule(head_mode="sigmoid_bce", bce_reachable_only=True)`. **Per-cell BCE(+Dice)** over `loss_mask`, target = `f_grid` ∈ {0,1}. Room-grouped split. → **Q1** (held-out opener **AUC 0.82**).

## 2. Rung-2 → the depth-2 search tree

- **Config:** `python/namo/data_collection/region_opening_rung2_car.yaml` — `region_max_chain_depth: 2`, `region_exhaustive_mode: true`, `region_sample_k: 20`, `region_frontier_beam_width: 15`, `goal_strategy: scorer`, `scorer_ckpt: <Q1>`, `region_selection_strategy: ml_first`, `ml_device: cpu`, `primitive_prefix: "1x_car_d5_"`.
- **region_opening.py [+12 lines]:** on setup pushes (`chain_depth < max_chain_depth`, exhaustive mode) it now persists the **post-push `resulting_state` (qpos/qvel)** into the trial-log entry — so a depth-2 node's `ctx` can be rendered from the exact post-shove state it was searched from.
- **modular_parallel_collection.py [+3 lines]:** passes `--ml-device` through to the scorer branch (was hardcoded `cuda` → CPU nodes crashed).
- **Collect:** Q1-guided best-first search to depth 2, recording the whole tree (`primitive_trial_log` + `reachability_log`, joinable by `chain_depth`/`parent_edge`/`parent_depth`).
- **Label:** `scripts/pipeline/build_rung2_h5.py` — **one row per tree-node**: pushes on a discovered solution path → **γ^k** (opener leaf = 1, setup = 0.9), searched-dead → **0**, unreachable → **−1**, untried → **MASK**. `ctx` per node rendered from *that node's* state (start-state for depth-1 nodes, the stored post-shove state for depth-2 nodes). Winning second-shoves at post-shove states are opener-leaves (target 1) — the "free 1-push finishes."

## 3. Pool → Q2 (value field)

- **Pool:** `scripts/pipeline/pool_q2_h5.py` — rung-1 rows wholesale + rung-2 rows **EXCEPT `setup_moved==0` no-op nodes** (search expanded pushes that didn't move the object; redundant ctx==start, dropped). Common cols only (`ctx, contact_px, r_mask, value_target, value_mask, xml`). → **~85K rows** (value dist on trained cells: 85.5% dead / 13.9% opener-finisher / 0.59% setup).
- **Train:** `scripts/rl_loop/train_q2.py` + `python/namo/rl_loop/sage_ext/q2_dataset.py` — `EdgeCrossAttn` (60,5) **`hl_gauss` value head (51 bins, range [0,1])** **[DEV: distributional value regression, not plain MSE — the repo's registered value head; eval_scorer-compatible]**. Target = `value_target` ∈ {0, 0.9, 1} on `loss_mask` (reachable-tried), masked CE to the Gaussian-smoothed target. Room-grouped split. → **Q2**.
- **[Calibration knob, 2026-07-11]** `Q2_POS_WEIGHT` (env var) → per-cell loss weight up-weighting opener/setup cells vs the dominant dead cells (default 1.0 = unchanged).

## 4. Eval (Q2 vs random)

- `scripts/sandbox/time_bestfirst.py` — Q2 rides the **`--nohz-ckpt` scorer slot**; best-first depth-2 search, `--hmax 2 --budget 900`, interleaved with `--models random` on identical nodes. Metric = **sims-to-solve** by difficulty × horizon. Geometry-disjointness vs `namo_testset_v1` verified (0 leaks).
- **[KEY / DEV]** the search ranks **first pushes by Q2's DIRECT score** `Q2(s, a)` (frontier priority, before expansion) — **not** the lookahead `max_b Q2(sim(s,a), b)`. So setup ranking rests on Q2 *directly predicting* setup-value — the part it learns worst (see the ranker-bottleneck result).

## 5. Deviations from the design (EXP-2026-07-10 card) — the honest list

1. **Reachable-only, not −1 fold-in** — the feasibility band is masked out of the loss (§0).
2. **hl_gauss value regression, not MSE** (§3).
3. **Deploy uses Q2's direct scores, not explicit lookahead** (§4) — arguably the biggest gap: the setup value is intrinsically `max_b Q(s',b)`, but the frontier ranks by the direct guess.
4. **Single pooled value head** does both opener-detection (depth-1) and setup-valuation (depth-2) — no staging/separate heads, no horizon conditioning.
5. **One flywheel turn only** — Q2 trained on Q1-collected data; never re-collected on Q2's own failure distribution (no DAgger).

## 6. Result (2026-07-10/11)

Mixed: Q2 **wins the low-sim regime** (median 2push 35 vs random 43; solve@2 10.9% vs 3.0%) but **loses the solve-rate ceiling** (2push 84.8% vs 91.0%). Ranker-bottleneck decomposition: **Q2 still buries the true setup** (setup ranked #1 only 21.6% / hard 14.8%, vs NoHz 50.9% / 39%) — the intervention did not fix the bottleneck it targeted. Root cause: setup-value is a lookahead property learned poorly by direct regression on a rare (0.59%) subtle signal. See the card's Result/Discussion.
