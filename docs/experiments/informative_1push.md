# Experiment: informative-hard 1-push training

**Hypothesis:** training the goal model on informative-hard (solve_rate ≤10%) 1-push
scenes beats training on the same *amount* of random 1-push data — most on HARD test scenes.

**Decisions (locked):** train fresh (no warm-start) · 2× A100/L40S preferred, fallback 1×,
never Camden · eval = feasibility · uniform 416/416/416 test divisions · test used as
validation (plateau ⇒ stop) + hard wall-time cap.

## Checklist

### Structure / docs
- [x] `docs/cluster_resources.md` (GPU policy: A100=ampere, L40S=adalovelace, never Camden)
- [x] `config/datasets/v3_1push_le10.yaml` (informative dataset registry entry)
- [x] GPU policy in `sage_learning/docs/training.md` + memory

### Train — informative model
- [x] Stage 1 — informative ≤10% manifest → **21,831 scenes**
- [x] Stage 2 — regen NPZs → **27,931 NPZs**
- [x] Stage 3 — build H5 → 1.1 GB / 27,931 (`h5/v3_1push_le10_lzf_tight_data/data.h5`)
- [x] Stage 4 — train fresh on L40S  ← job 55533764 **DONE** (max_epochs=100, 40 min).
      Best ckpt **epoch066 val_loss 0.0037**, run `23-30-23`.
      (⚠️ `ampere` covers RTX 3090 too — use `--constraint=adalovelace` for L40S.
       Lightning+SLURM `ntasks=1` ⇒ effective 1 GPU; model tiny so fine.)

### Train — baseline (control)
- [x] manifest — random 21,831 from all-solvable 1-push (pool 211,021)
- [x] regen NPZs → 26,698 → H5 built (1.05 GB, `h5/v3_1push_baseline_lzf_tight_data`)
- [x] train fresh on L40S  ← job 55533894 **DONE** (cancelled @ep62, plateaued; identical
      hparams batch 128 / lr 8e-4). Best ckpt **epoch019 val_loss 0.0038**, run `23-34-49`.

### Test eval set
- [x] test difficulty distribution (hard 416 / med 2,327 / easy 7,968 ; not-solvable 4,966)
- [x] 3 division manifests — hard 412 / med 414 / easy 413 (seeded)
- [x] regen NPZs → 3 H5s (~20 MB each, `h5/v3_test_{hard,med,easy}_lzf_tight_data`)
- [x] export valid `(edge,depth)` sets + solve_rate → `manifests/v3_test_validsets.json`

### Eval / comparison
- [x] eval harness `scripts/sandbox/eval_feasibility.py` (+ `.slurm`) — decode→snap→score
- [x] validsets augmented with `tried` set (`augment_validsets_tried.py`, job 55534237)
- [x] geometry pinned: **car 1x d5 dat** (`motion_primitives_1x_car_d5_*`); se2_target is
      object-local; self-check A(geom)=0.00 mm/0.00°, B(decode)≈6 mm — exact
- [x] metrics: success@1, success@k (1/5/10/20), expected tries-to-valid — per division
- [x] 3 samplers: random (analytical floor), baseline-data model, informative model
- [x] best-ckpt eval = validation; both plateaued (info ep66 / base ep19), base cancelled @ep62
- [x] hard wall-time cap — train jobs --time=04:00:00 (info ran 40 min; base cancelled)

**RESULT** (`docs/experiments/informative_1push_results.md`, full json `/scratch/dm1487/eval/`):
informative ≫ baseline on **every** division at s@1 — hard 0.063 vs 0.010 (6.3×),
med 0.099 vs 0.020 (5×), easy 0.115 vs 0.034 (3.4×). Informative beats the random floor
on **hard** (0.063 vs 0.027); baseline model is **below the floor everywhere**. val_loss was
near-identical (0.0037 vs 0.0038) — feasibility test, not val_loss, separates them.

### Analysis
- [x] seed overlap — ⚠️ test shares 100% of templates with train (same arenas, new objects)
- [ ] env-file duplication check (SLURM)

_Updated as stages land. Sandbox scripts in `scripts/sandbox/`; promote+doc if reused, else delete._
