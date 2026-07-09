---
status: hub
tags: [experiment]
thread: scorer-search
updated: 2026-06-26
---

# Horizon-Q Model Registry

> **⚠ Framing note (2026-07-06): budget/horizon-conditioning was DROPPED** (measured ≈ no-horizon, **NoHz** ahead — 40.7 vs 34.1). This registry is STILL the authoritative catalog — **all checkpoint paths / numbers / eval dirs below stay valid** — but the horizon-conditioned models (Horizon-v2/v3, the `budget_cond` variants) are a **historical** line; the live model is **NoHz** ("no-horizon", a single value/ranker). Current framing: [../problem_and_approach.md](../problem_and_approach.md).

> THE authoritative location list — every trained model, its exact best-val checkpoint, headline number,
> training data, and eval-output dir. **Do not reconstruct paths by glob; read here.** Never retrain
> registered models ([[feedback_reuse_baselines]]). Roots: ckpts `/scratch/dm1487/sage_outputs/scorer/`,
> evals `/scratch/dm1487/eval/`, H5s `/scratch/dm1487/h5/`. Updated 2026-06-15 (added Horizon-v2 / NoHorizon-v2).

## Models (all `edge_crossattn`, pos_fourier + use_edge_embed, 3 seeds; "BEST" = lowest val_loss ckpt)

### champion B30 (the pre-v4 baseline — old data v3_scorer_e4)
- ckpts: `/scratch/dm1487/eval/final_verdict_snapshot/h5samp_B30_s{1,2,3}.ckpt`
- evals: `/scratch/dm1487/eval/newbar_verdict/eval_h5samp_B30_s*.json`
- **hard@1 = 23.27 ± 1.38** (@5 52.4). Recipe: sigmoid_bce, sample_k=30.

### M1 — v4 data factory (plain scorer, champion recipe on v4 data)
- run dirs `m1_v4hq_s{1,2,3}`; BEST ckpts:
  - s1 `m1_v4hq_s1/namo-classifier/kibc9ig2/checkpoints/epoch018-val_loss0.4886.ckpt`
  - s2 `m1_v4hq_s2/namo-classifier/89wrbgn1/checkpoints/epoch016-val_loss0.4913.ckpt`
  - s3 `m1_v4hq_s3/namo-classifier/06wmgcvz/checkpoints/epoch019-val_loss0.4815.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m1_scorer/data.h5` (123,269 solvable, 65:35 feb:aug9)
- evals: `/scratch/dm1487/eval/m1_verdict/`  · **hard@1 = 29.40 ± 1.50** (@5 59.5)

### M2a — budget-Q arch (H-embed + HL-Gauss), same M1 data
- BEST: s1 `m2a_v4hq_s1/namo-classifier/fixo4s7i/checkpoints/epoch015-val_loss0.7605.ckpt` · s2 `m2a_v4hq_s2/namo-classifier/670z73ct/checkpoints/epoch015-val_loss0.7674.ckpt` · s3 `m2a_v4hq_s3/namo-classifier/hi0u7t1o/checkpoints/epoch015-val_loss0.7558.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m1_scorer/data.h5` · evals: `/scratch/dm1487/eval/m2a_verdict/`
- **hard@1 = 29.62 ± 0.93** · flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h

### M2b — + dead-ends (THE current best 1-push model / Q-full warm-start + baseline)
- BEST: s1 `m2b_v4hq_s1/namo-classifier/tryggakf/checkpoints/epoch012-val_loss0.6814.ckpt` · s2 `m2b_v4hq_s2/namo-classifier/rxn54385/checkpoints/epoch016-val_loss0.6802.ckpt` · **s3 (best-val, used for fpv) `m2b_v4hq_s3/namo-classifier/ql60myva/checkpoints/epoch013-val_loss0.6780.ckpt`**
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (252,805 = 123,269 solvable + 129,536 dead)
- evals: `/scratch/dm1487/eval/m2b_verdict/` · 2-push search: `/scratch/dm1487/eval/fpv_m2b/`
- **hard@1 = 32.86 ± 2.38** (@5 65.4) · dead-slice cand-pool V_dead 0.065, AUC 0.987 · 2-push search 75.2@1 / e2e 61.9%

### M2c — + 20 unreachable-cell zeros (ablation; data-side flag unreachable_k=20)
- BEST: **s1 (fpv) `m2c_v4hq_s1/namo-classifier/y1rtra4f/checkpoints/epoch013-val_loss0.6793.ckpt`** · s2 `m2c_v4hq_s2/namo-classifier/i5fu1h23/checkpoints/epoch016-val_loss0.6853.ckpt` · s3 `m2c_v4hq_s3/namo-classifier/d9l0zs60/checkpoints/epoch013-val_loss0.6845.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (same as M2b) · evals: `/scratch/dm1487/eval/m2c_verdict/` · 2-push search: `/scratch/dm1487/eval/fpv_m2c/` (job 56008453, running)
- **hard@1 = 32.21 ± 1.48** (ranking ≈ M2b) · all-cells V_dead **0.072** (vs M2b 0.327 — hallucination killed)

### M2d — + reachability input bit (ablation; reach_flag_input, network flag)
- BEST: s1 `m2d_v4hq_s1/namo-classifier/eoz348sj/checkpoints/epoch015-val_loss0.6828.ckpt` · s2 `m2d_v4hq_s2/namo-classifier/yvq3jtsq/checkpoints/epoch012-val_loss0.6793.ckpt` · s3 `m2d_v4hq_s3/namo-classifier/be1ehs8b/checkpoints/epoch016-val_loss0.6771.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` · evals: `/scratch/dm1487/eval/m2d_verdict/`
- **hard@1 = 34.20 ± 2.09** (within noise of M2b) · all-cells V_dead 0.621 (degraded — tell<teach)

### Q-full (M3/M4) — mixed-H, TRAINING (job **56015587**, 3 seeds, L40S; 56013237/56013312 = dead false-starts)
- run dirs `/scratch/dm1487/sage_outputs/scorer/qfull_v4hq_s{1,2,3}/`; ep11 BEST ckpts (still training, val↓):
  - s1 `qfull_v4hq_s1/namo-classifier/zxt3n1tm/checkpoints/epoch011-val_loss0.6572.ckpt`
  - s2 `qfull_v4hq_s2/namo-classifier/a91eflex/checkpoints/epoch011-val_loss0.6582.ckpt`
  - s3 `qfull_v4hq_s3/namo-classifier/h2lcraeg/checkpoints/epoch011-val_loss0.6582.ckpt`
- **ep11 H=1 feeler (job 56022469): hard@1/@5 — s1 34.4/65.1 · s2 41.3/67.2 · s3 30.7/68.3 · MEAN 35.5/66.9 vs M2b 32.86/65.4 (+2.6/+1.5pp). H=1 sanity only; M3 foresight = headline. wrong-edge/miss≈89%.**
- data (multi-H5, ';'-joined): `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (252,805 H=1 rows) + `/scratch/dm1487/h5/v4_hq_h2_scorer/data.h5` (311,324 = 155,662 ep × {H=1,H=2}; 172,104 dead; gamma=0.9, format=twopush) → ~564k total rows, room-grouped across files (realpath-normalized)
- flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h (NO unreachable_k/reach_flag — M2c/M2d parked)
- evals (planned): `/scratch/dm1487/eval/qfull_verdict/` + `fpv_qfull/` + bifurcation/M3 slices
- GATES when done: (a) H=1 ranking ≈ M2b 32.86; (b) M3 = zero-sim setup-pick on pure2push vs 34.5 (registered) & 75.2-with-sims (fpv_m2b); (c) H-bifurcation probe (525ea31); (d) per-division (pure2push_divisions)
- NOTE: this Q-full = **Horizon-v1** (`qfull_v4hq`, ep16 converged). Its NoHorizon twin = `qfull_nohz_v4hq` (unregistered; glob if needed). v2 below adds the 1push@H2 augmentation on top of this mix.

### Horizon-v2 / NoHorizon-v2 — Q-full mix + 1push@H2 augmentation (THE 2×2 v2 cells; aug fixes H=2 dilution)
- **data (v2 mix, ~944k rows):** M2B `v4_hq_m2b_scorer/data.h5` (252,805) + H2 `v4_hq_h2_scorer/data.h5` (311,324) + AUG `v4_hq_onepush_h2_aug/data.h5` (80,000 — sparse-positive 1push@H2 rows, the dilution fix) + postpush `v4_hq_postpush` shards (~300k narrow finish; v3/ExIt REPLACES this). Room-grouped, realpath-normalized.
- **Horizon-v2** `qfull_v2_v4hq` — flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h. BEST-val ckpts:
  - s1 `qfull_v2_v4hq_s1/namo-classifier/10whb62b/checkpoints/epoch008-val_loss0.6728.ckpt` (HEADLINE seed)
  - s2 `qfull_v2_v4hq_s2/namo-classifier/whv2sdf3/checkpoints/epoch008-val_loss0.6771.ckpt`
  - s3 `qfull_v2_v4hq_s3/namo-classifier/a81jq5ob/checkpoints/epoch008-val_loss0.6689.ckpt`
  - **1-push hard@1: H=1 36.0 / H=2 30.7** (aug fixed v1's H=2 dilution 12.2→30.7) · **2-push s1: s@2 24.2, s@900 94.9** (avg-sims 54.6)
- **NoHorizon-v2** `qfull_nohz_v2_v4hq` — flags: value_bins=51, head_mode=hl_gauss, budget_h=false (NO budget_cond ⇒ H-invariant). BEST-val ckpts:
  - s1 `qfull_nohz_v2_v4hq_s1/namo-classifier/4w1hovo4/checkpoints/epoch007-val_loss0.7041.ckpt` (HEADLINE seed)
  - s2 `qfull_nohz_v2_v4hq_s2/namo-classifier/rbbqq0ya/checkpoints/epoch009-val_loss0.7004.ckpt`
  - s3 `qfull_nohz_v2_v4hq_s3/namo-classifier/c82jwuw5/checkpoints/epoch010-val_loss0.6968.ckpt`
  - **1-push hard@1: 31.7** (H-invariant) · **2-push s1: s@2 32.6, s@900 91.6** (avg-sims 76.7)
- ⚠ s1 headline evals (2-push solve + 1-push rank) used the ADJACENT final epoch (Hz epoch010 val0.6734 / NoHz epoch009 val0.7050) — val Δ<0.001 vs best-val above, ranking-identical. ckpt root `/scratch/dm1487/sage_outputs/scorer/`.
- **evals:** 2-push solve `/scratch/dm1487/eval/bf900_qfull_v2_v4hq_s1/` + `bf900_qfull_nohz_v2_v4hq_s1/` · 1-push rank `/scratch/dm1487/eval/onepush_rank_v2/` (running 56308524 H1 / 56308525 H2 as of 2026-06-15)
- **TAKEAWAY:** reactive@2 NoHz>Hz EVERY difficulty tier; search@900 Hz>NoHz (decisive on hard 90 vs 82). Horizon = search accelerator, not a reactive win. See journal DIFFICULTY DEEP-DIVE + `results_design_report_2026-06-15.md`. ⚠ **SUPERSEDED for reactive [2026-06-22]:** that "NoHz>Hz reactive" was best-first @2 (search free to NOT dive). Under the FORCED-DIVE reactive (`eval_reactive_argmax`, argmax setup→argmax finish, region, 3 seeds) **Hz≈NoHz (38.5±2.1 vs 38.2±3.0 — TIED)**; the single-seed gap was noise. Horizon's best-first deficit is the un-forced dive; force it and Horizon reaches parity (not a win). See Horizon-v3/NoHorizon-v3 below + journal §9 [2026-06-22].

### Horizon-v3 / NoHorizon-v3 — the ExIt FINISH retrain (v3 mix = v2 with narrow postpush REPLACED by ExIt finish)
- **data (v3 mix):** M2B + H2 + AUG + **ExIt** (`v4_hq_exit_finish_valid` + `v4_hq_exit_finish`) — the narrow 300k postpush (the data that failed to generalize) REPLACED by ~24k diverse on-policy/valid-setup exhaustive finish rows at the true ~7% difficulty. Same recipe as v2. Retrained FROM SCRATCH (clean v2-vs-v3 = data effect). ~11 ep to converge.
- **Horizon-v3** `qfull_v3_v4hq` — flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h. BEST-val ckpts:
  - s1 `qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt` (HEADLINE seed)
  - s2/s3 TRAINING (jobs 57014837 array 10/11) — register on convergence (~3-5am ET 2026-06-23)
- **NoHorizon-v3** `qfull_nohz_v3_v4hq` — flags: value_bins=51, head_mode=hl_gauss, budget_h=false. BEST-val ckpts:
  - s1 `qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt`
  - s2/s3 TRAINING (job 57014838 array 10/11)
- **HEADLINE (s1, region criterion, n=1018):** reactive@2 (forced dive) **Hz 45.6 / NoHz 40.7**; best-first@2 combine=q **Hz 36.1 / NoHz 38.0**; search s@900 **Hz 97.7 / NoHz 95.9**. ExIt gate (fixed-s1, n≈990): finish-sep 0.385, top1_finish_opens 0.602 (+0.05 ~3σ vs v2 0.55) = MODEST-but-REAL, gate (0.6) NOT cleared. v3 single-seed lifts BOTH reactive regimes vs v2 (Hz 38.5→45.6, NoHz 38.2→40.7) + shrinks the dive tax — error bars pending s2/s3.
- **evals:** reactive `reactarg_{Hz,NoHz}_v3/` · best-first@2 combine=q `bfq_{hz,nohz}_v3_s1/` · ExIt gate `check_h2_finish`.
- **HORIZON-ROLE PROBE [2026-07-09, arrakis, s1, `…/eval/horizon_probe/`]:** reproduces the above (NoHz pure2 reactive 40.8 / bf@2 38.1; Hz reactive 45.3 / bf@2 35.9). Findings: (1) horizon is a *working route knob* on 1-push (NoHz H=2≡H=1 byte-identical = control; Hz H=2 demotes opener→setup −7.9pp react@1, net-neutral react@2, +3.6 hard). (2) **BUDGET SWEEP corrects the "NoHz wins best-first" line — that's a budget-2 artifact:** Hz/NoHz cross at ~budget 3, then Hz wins best-first +3→+9pp (hard +13.5 @20), matching s@900 Hz>NoHz — **horizon is a search accelerator.** See [[EXP-2026-07-09-horizon-role-probe]] + RESULTS §7 (Table 7d + curve).

### qboot_density / qboot_depth — STAGE-1 bootstrapped single-Q (Horizon DROPPED), trained on ILAB [2026-06-26]
- **THE Stage-1 redesign model** (search-first redesign, drop-Horizon). Single Q, NoHorizon flags (`budget_h=false head_mode=hl_gauss value_bins=51`), `sample_k=30`, FROM SCRATCH. Mix = M2B + ExIt-finish-v4 + **boot_setup_{density|depth}** (s0 relabeled with grounded target **γ·V_GT(s1)** from `frac_first_push`, no re-sim). **Trained on ilab** (not Amarel): launcher `sage_learning/scripts/train_bootstrap_ilab.slurm` (32 workers — `ctx` is LZF-compressed ⇒ dataloading is CPU-bound), data `/common/users/dm1487/fresh_start/projects/namo/h5`, 305,116 rows (train 274,604 / val 30,512). **ckpt root (ILAB, NOT /scratch):** `/common/users/dm1487/scratch_namo/outputs/scorer/`
- **qboot_density_s1** (γ·findability target = Stage-1): BEST-val `qboot_density_s1/namo-classifier/v5x21lsi/checkpoints/epoch012-val_loss0.7152.ckpt` · val_top1 **0.674** / top5 0.745 (peak top1 0.697) · job 166181, COMPLETED 5:15h, early-stop ep37.
- **qboot_depth_s1** (γ·existence target = Stage-3 density-vs-depth control): BEST-val `qboot_depth_s1/namo-classifier/xdbdc8vv/checkpoints/epoch014-val_loss0.7192.ckpt` · val_top1 **0.704** / top5 0.774 (peak 0.704) · job 166182, COMPLETED 5:14h, early-stop ep39.
- **EARLY VAL SIGNAL (NOT the gate): depth val-top1 0.704 > density 0.674** — directionally matches the pre-registered "depth ≥ density". Caveat: val = all-difficulty room-grouped ranking, NOT test-set hard@1; both overfit after ep12-14.
- **⏳ GATE PENDING (runs on Amarel — where the SLURM eval harness + v3 baseline ckpts live; NOT Amarel-only: the eval script is portable repo code and `namo_testset_v1` is also mirrored on arrakis `/common/users/dm1487/scratch_namo/datasets/`):** reactive@2 + best-first@2(combine=q) vs **NoHz-v3 reactive 40.7 / best-first 37.8 @2** (region, n=1018). Handoff: rsync best-val ckpt → Amarel `/scratch/dm1487/sage_outputs/scorer/qboot_{density,depth}_s1/.../` then `eval_afterok.slurm RUN_DIR=qboot_density_s1 LABEL=boot_density MINEP=8` (+ depth). Eval dirs will be `reactarg_boot_*` + `bfq_boot_*`.

### Eval tools for the reactive/search comparison (2026-06-22, combine=q standard)
- **reactive@2 (forced dive):** `scripts/sandbox/eval_reactive_argmax.py` (argmax setup@H2 → argmax finish@H1 → region open?, exactly 2 sims, object-constrained). Sharded by `scripts/amarel/reactive_argmax.slurm`. Out `reactarg_*`.
- **best-first@2 (combine=q):** `scripts/amarel/bestfirst_multiseed_q.sh` → `bestfirst_eval.slurm COMBINE=q SIM_BUDGET=900`. s@2 (= dive-tax point) + s@900 (search ceiling) read off ONE budget-900 run. Out `bfq_*`. **combine=q = raw Q priority, NO 0.5Q+0.5V blend** [USER: "don't multiply the value of the state for the dive"]. The blend `bf900_*` dirs are SUPERSEDED.

## Key H5 datasets
- `v4_hq_m1_scorer/data.h5` — M1/M2a solvable-only (123,269)
- `v4_hq_m2b_scorer/data.h5` — M2b/M2c/M2d, +dead (252,805)
- `v4_hq_h2_scorer/data.h5` — Q-full H=2 mixed rows (≈311k = 155,662 episodes × 2), BUILDING (chain2 55972757)
- mask H5s: `v4_hq_m1_65_35`, `v4_hq_de_masks(_rest)`, `v4_hq_h2_root_p{0..3}`

## Eval / test keys
- 1-push answer key: `/scratch/dm1487/datasets/namo_testset_v1/labels/onepush_episodes.json`
- 2-push key + divisions: `.../labels/pure2push.json`, `.../labels/pure2push_divisions.json` (hard≤2/med 3-8/easy>8 setups)
- fpv aggregate (M2b): `/scratch/dm1487/eval/diag_fpv_aggregate.json`
