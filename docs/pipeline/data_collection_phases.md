# Data Collection Phase Cookbook

Reference for the parameter sets used in each phase of the region-opening data collection cascade. Each phase has a distinct purpose; combine them to build a corpus.

> See also: `DATA_COLLECTION_GUIDE.md` (general workflow), `python/namo/data_collection/modular_parallel_collection.py` (the entry point).

---

## Common cross-phase parameters

These apply to every phase unless overridden.

| Param | Value | What it does |
|---|---|---|
| `--algorithm` | `region_opening` | Use the region-opening planner |
| `--config-file` | `config/namo_config_complete_skill15_car_1x.yaml` | C++ runtime config; sets robot footprint, wavefront resolution, push speed |
| `--primitive-data-dir` | `data/` | Where to find motion primitive `.dat` files |
| `--primitive-prefix` | `1x_car_` *(or `1x_car_d5_` for v3)* | Selects which primitive calibration to load; must match the robot in `--config-file` |
| `--region-allow-collisions` | true | Allow primitives that brush other movables |
| `--episodes-per-env` | 1 | One attempt set per XML |
| `--region-min-reachable-fraction` | 0.2 | A push counts as "opening" if ≥ 20 % of region samples become reachable |

### Primitive calibration

- `1x_car_` → `motion_primitives_1x_car_{square,wide,tall}.dat` — 60 edges × 10 push_steps = **600 primitives/shape**, max push ≈ **45 cm**
- `1x_car_d5_` → truncated to push_steps ≤ 5 = **300 primitives/shape**, max push ≈ **22 cm**

For v3 feb_car (0.49 × 0.775 m envs), **use `1x_car_d5_`** to avoid wall-clipping.

### Wavefront resolution at collection time

The C++ planning wavefront runs at the `planning.high_level_resolution` from the config (currently **1 cm**, NOT the 2 mm in `wavefront_planner.resolution`). The mask-gen side uses 2 mm.

---

## Goal-strategy choices

| Strategy | Behavior | When to use |
|---|---|---|
| `primitive` | Try primitives in cost-sorted order. Stops at first success per attempt. | Biased data — first-found (cheap) primitives dominate |
| `random_rollout` | Try primitives in random order. Stops when `max_solutions_per_neighbor` reached. | Unbiased per-sample first-find |
| `ml` / `ml_*` | Use a learned model to score / propose primitives | Bootstrapping with a pre-trained policy |

---

## Phase semantics

### Phase 1 — single-push opening, fast first pass

Purpose: cover the easy cases (one push solves the problem).

```
--region-max-chain-depth 1                 # single push only
--goal-strategy random_rollout             # unbiased ordering
--rollout-samples-per-state 600            # K — try all primitives in random order (300 if d5 cap)
--region-max-solutions-per-neighbor 1      # stop at first success (A) — OR see B below
--region-max-recorded-solutions-per-neighbor 1
--target-goal-region                       # only attempt the labeled 'goal' region (not all neighbours)
--search-timeout 60                         # per env
```

**Pattern A** (stop-at-first) vs **Pattern B** (find-all, pick-random) vs **Pattern C** (find-all, emit-all):

| | `max_solutions_per_neighbor` | `max_recorded_solutions_per_neighbor` | NPZs/attempt | Quality |
|---|---|---|---|---|
| **A** | 1 | 1 | 1 (first-found) | order-biased |
| **B** | 300 (or 600) | 1 | 1 (uniform random over found solutions) | unbiased ✅ |
| **C** | 300 (or 600) | 300 | N (one per recorded solution) | unbiased ✅ × N data |

Pattern A is fastest, B is preferable for clean signal, C maximizes data per env.

For B: with `random_rollout` shuffling + `max_solutions=300`, the planner finds *all* valid primitives; with `max_recorded=1`, the first-found-in-random-order is the recorded one — effectively a random pick across the valid set.

### Phase 2 — depth-2 chains, narrow per-state branching

Purpose: solve envs that need a 2-step push chain. Tighter `K` than phase 1 since chain search is exponential in depth.

```
--region-max-chain-depth 2                 # allow 2-push chains
--goal-strategy random_rollout
--rollout-samples-per-state 20             # K=20: 20 random primitives per state per chain link
--region-max-solutions-per-neighbor 10
--region-max-recorded-solutions-per-neighbor 1  # record 1 (pattern B w/ random_rollout shuffle)
--target-goal-region
--search-timeout 300
# region_exhaustive_mode: false             ← OFF for phase 2-5; early-stop is fine
```

K=20 means at each chain step, 20 primitive candidates are tried. With chain_depth=2 that's at most 20 × 20 = 400 chain attempts per attempt — tractable.

**Why `exhaustive_mode=false` for phase 2 (and beyond)**: chain search is large; full exhaustive coverage would blow the time budget. The cascade design instead uses **multiple phases + seed sweeps** to cover the space: phase 2 + phase 3 (K=20 + K=50) plus phase 5A-E (5 independent seeds × random rollout) give effective coverage via parallel angles rather than one expensive exhaustive run.

### Phase 3 — depth-2 chains, wider per-state branching

Same as Phase 2 but K=50 — explores more primitives per state, catching cases Phase 2's narrow K missed.

```
--rollout-samples-per-state 50             # K=50
# region_exhaustive_mode: false
```

Otherwise identical to Phase 2.

### Phase 4 — partial-fail union

Purpose: re-attempt envs where phases 1-3 found *no* valid solution. Increases search budget and goal samples to scrape out marginal cases.

```
--region-max-chain-depth 2
--rollout-samples-per-state 300
--goals-per-region 200                      # more goal samples per region
--search-timeout 600
# region_exhaustive_mode: false              ← OFF
```

### Phase 5A-E — seed sweep (optionally parallel)

Purpose: maximize coverage by re-running difficult envs with different random seeds. Each variant (A-E) uses a different RNG seed.

```
--shuffle-seed <A=1, B=2, C=3, D=4, E=5>
--region-max-chain-depth 2
--rollout-samples-per-state 300
# region_exhaustive_mode: false              ← OFF
```

Variants A-E can run in parallel for ~5× the coverage of any single phase-4 run.

The 5-seed sweep is **how phases 2-5 achieve good coverage WITHOUT exhaustive_mode**: instead of one expensive exhaustive run, you get 5 independent random orderings each finding different solutions. The union of 5 non-exhaustive runs ≈ 1 exhaustive run, at the same wall time (parallel).

---

## v3 feb_car cascade — current recommended settings

Based on the d5 primitive cap + 0.49 × 0.775 m envs:

```
# Common
--primitive-prefix "1x_car_d5_"
--config-file config/namo_config_complete_skill15_car_1x.yaml
--target-goal-region

# Phase 1 (Pattern B — find-all goal-region only)
--algorithm region_opening
--region-max-chain-depth 1
--goal-strategy random_rollout
--rollout-samples-per-state 600
--region-max-solutions-per-neighbor 300
--region-max-recorded-solutions-per-neighbor 1
--search-timeout 60
```

Then NPZ generation:
```
--wide-crop-size 0.6      # tight match to max push (22 cm) + obj half-extent
--tight-crop-size 0.5     # unchanged
--namo-config config/namo_config_complete_skill15_car_1x.yaml
--local-only
--filter-overlaps
```

---

## Slurm orchestration (v3 cascade)

`scripts/amarel/run_batch_collection_smoke.slurm` is the canonical collection-array template — reads `PKL_MANIFEST`, `SHARD_SIZE`, `OUTPUT_DIR` and slices the manifest by `SLURM_ARRAY_TASK_ID`.

The v3 cascade:
- `scripts/amarel/v3_phase1_collect.slurm` — phase-1 sharded array (single-push, exhaustive, pattern B).
- `scripts/amarel/v3_cascade_driver.slurm` — sequential driver: waits for phase 1, then mines + runs phases 2→3→4→5A-E in one allocation (survives disconnects).
- `scripts/amarel/v3_cascade_collect.slurm` — reusable per-phase collection (env vars `MANIFEST`, `OUTPUT_DIR`, `DEPTH`, `KROLL`, `NUM_SHARDS`, `GOAL_SEED`).
- Inter-phase mining: `scripts/build_phase2_manifest.py` (sharded-aware) + `scripts/build_phase4_manifest.py` (partial-fail union).

All paths and per-phase params (depth / K / seeds / mining) come from **`config/corpora/<id>.yaml`**, resolved by **`scripts/corpus.py`** — the driver is corpus-agnostic, so a new corpus is a new config, not a new script:

```bash
CORPUS=v3_aug9 PHASE1_JID=<phase-1 array id> sbatch scripts/amarel/v3_cascade_driver.slurm
# preview the plan, no SLURM:  CORPUS=v3_aug9 PHASE1_JID=0 DRYRUN=1 NAMO_PY=<py> bash <driver>
```

`layout: flat` in the config matches the existing on-disk v3 dirs (`outputs/<id>_phaseN`); new corpora can set `nested` (`outputs/<id>/phaseN`) for one-prefix cleanup.

`v3_phase1_collect.slurm` and `submit_npz_gen.sh` are already env-parameterized (`MANIFEST`, `OUTPUT_DIR`, `PHASE_DIR`) — fill them from `corpus.py <id> paths` (e.g. `PHASE1_OUT`) to keep phase-1 and NPZ-gen on the same layout as the cascade. The driver also writes a `meta.json` lineage stamp (`corpus.py <id> stamp`) on completion. The combined training H5 is its own declaration in `config/datasets/` (e.g. `v3_balanced_1to1.yaml`), and the NPZ/H5 key contract lives in `config/dataset_schema.yaml`.

---

## NPZ → H5 (training corpus)

Phases produce PKLs → NPZ masks → a single HDF5 the trainer reads. End to end:

1. **NPZ gen (dual-crop, overlap-filtered)**: `scripts/amarel/submit_npz_gen.sh <phase_dir> <out_masks_dir>` builds a PKL manifest and submits the mask array → `outputs/<corpus>_<phase>_masks`. Defaults (car): wide 0.6 m + tight 0.5 m crop, 2 mm wavefront, `--local-only`, `--filter-overlaps`. (Rolling per-phase variant for the aug9 cascade: `aug9_post_phase4_5_npz.slurm` / `aug9_rolling_npz_driver.slurm`.)
2. **Cross-phase overlap filter**: `scripts/amarel/filter_npz_overlaps.slurm` dedups masks shared across phases → `*_masks_overlap_filtered`.
3. **Build the balanced H5** (in `sage_learning`): `scripts/build_v3_balanced_h5.slurm` → `/scratch/dm1487/h5/v3_balanced_1to1*` — 1:1 balanced, lzf-compressed, tight crop. This is the trainer's input.
   - Generic raw-concat alternative (no balancing): `scripts/amarel/build_h5_all.slurm` (env vars `STAGE_DIR`, `OUTPUT_H5`).
   - Optional chain-depth filter on a built H5: `scripts/amarel/filter_h5_chain_depth.slurm`.
4. **Train** in `sage_learning` with `data_dir=/scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data`, `use_h5=true`, `crop_prefix=local_tight`.

---

## Multi-push → training examples (suffix decomposition)

A recorded n-push episode is split into n training examples by **trajectory suffix decomposition**: from each intermediate state `S_i`, the target is the *remaining* goals `[A_i … A_{n-1}]`. A 3-push solution yields 3 examples (all-from-start, remaining-from-S1, final-from-S2). This is why the NPZ carries per-push goal masks `goal_mask_a1..aN` (`a1` = the next push from the current state; count = remaining horizon) — it teaches the model to predict goals from *any* intermediate state, not just the initial one.

---

## Pitfalls

1. **`--target-goal-region` vs all-neighbours**: without the flag, the planner attempts to open every unreachable region per env, producing 5-10× more episodes per env but with non-goal-aligned semantics. The flag restricts to the labeled 'goal' region only.

2. **`max_push_steps` vs `region_max_chain_depth`**: different things. `max_push_steps` (config) is the number of physics ticks per primitive (depth axis); `region_max_chain_depth` is the number of pushes in the chained solution (chain axis).

3. **`primitive_prefix` must match config robot**: passing `--primitive-prefix car_` while using a point-robot `--config-file` causes mismatched physics and 0 % success.

4. **`cost_first` vs `random_rollout`**: `cost_first` biases data toward short / cheap primitives. `random_rollout` is unbiased per attempt but adds collection cost (more primitives tried). Phase-1 typically uses `random_rollout` for clean signal.

5. **Pattern A's "stop-at-first" with `random_rollout`**: still biased — the FIRST valid in random order isn't uniform over the valid set. Pattern B (find-all + record-1) is the actual uniform sampler.

6. **Wavefront resolution mismatch**: planning uses 1 cm, mask-gen uses 2 mm. For envs with passages < 10 cm, this can cause topology disagreements (rare in feb_car since passages are mostly ≥ 8 cm).
