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

## Canonical offline ranking panel (2026-07-26) — the ONE place AUC/rank numbers come from

Produced by `scripts/eval_auc.py` (2-push, `twopush_gt_h5`) + `eval_scorer.py --live-canonical` (1-push); grid JSONs under `$NAMO_SCRATCH/eval/auc_grid/`. **Any AUC quoted anywhere must name its variant** — grammar, seed bands, and the retired numbers are in [auc_metrics_reconciliation.md](auc_metrics_reconciliation.md). Seed-mean; noise bands ±0.01 (V1/F1), ±0.025 (V5), ±1.1 pt (setup hit@1), ±2.2 pt (1-push hard hit@1).

| model | V1 | V5 | F1 | setup hit@1 (floor 27.6) | finish hit@1 (floor 15.4) | 1p hard hit@1 (floor 2.5) |
|---|--:|--:|--:|--:|--:|--:|
| beast2c_d20_ceil (`d20_base`) | 0.779 | 0.490 | 0.862 | 51.9 | 70.0 | 40.7 |
| exact-value-rank v2 (3 seeds) | 0.770 | 0.502 | 0.858 | 50.7 | 70.3 | 40.7 |
| … its paired control (3 seeds) | 0.764 | 0.475 | 0.874 | 46.4 | 68.9 | 41.0 |
| **d20+setup-only split (deploy, 3 seeds)** | **0.827** | 0.543 | 0.843 | **59.8** | 69.8 | 39.2 |
| colossus0 opener-only | 0.825 | 0.593 | 0.820 | 59.4 | 68.4 | **44.1** |
| colossus0 split-full | 0.831 | **0.624** | **0.796** | **63.8** | 68.6 | **35.3** |

⚠ The `round2_eval.h5` AUCs quoted in the colossus / exact-value cards (0.876–0.925) are on the **dead-bank** distribution and run **~+0.12 pooling-inflated** vs these; within-board they are identical. Never put the two on one axis.

## Marvel 1-push DAgger ladder (antman-0..5) — 🔵 ACTIVE [2026-07-16]

`train_q2_rankaux` (λ=0.1), NOT the horizon `edge_crossattn` line. Screener for round r = antman_{r-1}; best ckpt = min val_loss in each `checkpoints/`. Eval: best-first hmax1 budget300 on `namo_testset_v1`, `agg_table.txt` beside each. Full detail → [log/EXP-2026-07-14-region-opening-curriculum-marvel.md](log/EXP-2026-07-14-region-opening-curriculum-marvel.md). Ckpts are CS-side (arrakis) under `/common/users/dm1487/scratch_namo/`.

| model | train rows | hard@1 | all@1 | ckpt dir (CS) |
|---|---|---|---|---|
| antman-0 (seed) | 50,000 | 23.0 | 72.7 | `antman0/train_run/checkpoints/` (epoch018-val_loss0.8429) |
| antman-1 | 50,528 | 24.0 | 74.7 | `curriculum2/dagger_orchestrator/antman_1/checkpoints/` |
| antman-2 | 90,700 | 28.4 | 78.1 | `.../antman_2/checkpoints/` |
| antman-3 | 120,657 | 32.8 | 80.4 | `.../antman_3/checkpoints/` |
| antman-4 | 151,218 | 39.2 | 82.9 | `.../antman_4/checkpoints/` (epoch014-val_loss0.7961) |
| antman-5 | 167,655 | 42.6 | 82.9 | `.../antman_5/checkpoints/` |

Accumulated train h5: `.../dagger_orchestrator/accumulated/accumulated_train.h5` (167,655 rows @ antman-5). Testset key: `datasets/namo_testset_v1/labels/onepush_episodes.json`. 2-push (Beast) bank: `.../phase2_bank/` = 72,521 labeled-dead episodes + ~865k xml-only leads. [USER: `random` ranker baseline hard@1 1.5 / all@1 ~39.4.]

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

## Beast line (2-push rankers, curriculum2) — added 2026-07-21
All CS paths under `/common/users/dm1487/scratch_namo/curriculum2/beast/`. Full numbers: RESULTS.md + card EXP-2026-07-14. Single-seed each.
- **beast-1-c081 (champion, round-1):** `round1/models/beast1_c081/…` — 1p 97.9/86.9/48.5 all 86.8 · 2p 95.1 solve/93.0 sims/32.4@2/69.4@30 · hardh2 98.0/8.5. Trained on 191k (incl. 23.6k extras).
- **beast-1-clean / clean-hard (label-rule ablation pair):** `round1/models/beast1_clean{,_hard}/checkpoints/epoch016-*.ckpt` — purged 166,325 rows; the soft-vs-hard WASH result.
- **beast-2 armA (uniform):** `round2/models/beast2_armA/checkpoints/epoch011-val_loss0.5267.ckpt` — 2p solve 96.7 / 81.6 avg sims / 71.4@30 (best-ever 2p search axis); 1p hard@1 43.1.
- **beast-2 armB (balanced 50/50):** `round2/models/beast2_armB/checkpoints/epoch004-val_loss0.5799.ckpt` — 1p hard@1 **49.5** (best of line); 2p 96.3/94.5/67.4.
- Train data: `round2/h5/beast2_all.h5` (1,039,341 rows); dead-bank eval GT: `round2/h5/round2_eval.h5` (73,368 rows, 940 rooms).
- **beast-2 2×2 twins (corrected data, 2026-07-21):** `round2/models/beast2_arm{A,B}_{ceil,hard}/checkpoints/` — identical 859,766 rows (`beast2_exh_ceil/hard.h5`), one-variable label ablation × exposure. **armB_ceil = 2p front-runner (97.2 solve/78.6 sims/69.7 @30/86.4 1p-all@1)**; hard twins deploy-degraded (magnitude collapse). v0 arms (beast2_all.h5, pre-correction) = lineage only.
- **Canonical finish-layer GT:** `round2/h5/testset_gt.h5` (66,456 nodes, 982/983 pure2push scenes, REF full-exhaustive) — first exhaustive root+finish GT on the canonical set. EVAL-ONLY, never train.
- **beast-2c-d20 (20% dead dose test, 2026-07-21):** `round2/models/beast2c_d20_ceil/checkpoints/epoch010-val_loss1.7072.ckpt` — base beast-2c-A-ceil (192,822 pos) + 20% dead 50/50 (19,282 root + 19,282 finish) = `round2/h5/beast2c_d20_ceil.h5` (231,386 rows). Deploy 1p 97.4/80.8/**39.7**/83.2 · 2p 96.5 solve/54.6 sims/**26.6**@2/69.1@30 (vs 2c-A-ceil 35.3 hard@1 / 24.3 @2). Dead-bank AUC opener-vs-dead 0.940 (2c 0.859). **Dead helps; hard@1 +4.4.** Full → RESULTS.md + card EXP-2026-07-21-colossus. Single seed. Serves as colossus finish-ranker (`colossus/d20_finish_ranker.ckpt`). Base for the colossus dose-sweep (stack colossus dead on THIS).
- **Antman corrected final-pose action head (3-seed negative control, 2026-07-22):** `curriculum2/push_depth/final_pose_3seed/full/seed{1,2,3}/{baseline,corrected}` — same 178,364 Antman-5c ceiling-labeled boards, fresh baseline/corrected pairs, 20 epochs. Canonical prediction-only 1push mean easy/med/hard @1 baseline `97.1/82.3/40.4` vs corrected `97.6/80.4/37.2`; hard delta `-3.1` and corrected lost all three hard pairs. Hard wrong-contact `53.6→54.6`; @5 `99.6/95.0/72.9→99.7/95.4/73.0` (flat). **Reject corrected additive late fusion; not a deployment candidate.** Evaluations: `/common/users/dm1487/scratch_namo/eval/push_depth/final_pose_3seed/`; no D20 or 2push simulation. Full → RESULTS.md + card EXP-2026-07-22-push-depth-aware-ranker.
- **Antman crop-relative action heads (two 3-seed negative controls, 2026-07-23):** `curriculum2/push_depth/crop_relative_3seed/full/seed{1,2,3}/{plain,sharp}` — correct raw `(2dx/0.5m,2dy/0.5m,dtheta/pi)` versus the same feature with eight-band Fourier encoding + learned depth identity; matched to the saved fresh baselines above. Canonical mean easy/med/hard @1 baseline `97.1/82.3/40.4`, plain `97.7/82.0/39.9`, sharp `97.1/81.4/40.5`; hard @5 `72.9→75.3/76.8`. Plain improved hard @1 in 0/3 seeds, sharp in 1/3; hard wrong-contact `53.6→54.2/54.1`. **Reject both for top-1 ordering; not deployment candidates.** Evaluations: `/common/users/dm1487/scratch_namo/eval/push_depth/crop_relative_3seed/`; no D20 or 2push simulation. Full → RESULTS.md + card EXP-2026-07-22-push-depth-aware-ranker.
- **d20 exact-value ranking pilot (2026-07-23, NOT promoted):** `exact_value_rank/d20_exact_value_rank_seed1/checkpoints/epoch011-val_loss1.6855.ckpt` — same d20 data/architecture/seed recipe, loss-only change generalizing listwise supervision from exact openers to every exact value tier. Setup mechanism PASS: AUC 0.9063→0.9252, hit@1 55.0→64.5, hit@5 83.9→88.1. Deploy 1p 96.8/82.9/40.2/83.7 @1, hard@5 66.2; 2p 96.9 solve/55.1 sims-to-solve/29.7@2/46.8@5/72.3@30. Mixed: tight-budget 2p wins, but hard-1push@5 −5.4pp fails the guard. Card EXP-2026-07-22-exact-value-ranking-loss; single treatment seed. **SUPERSEDED by the v2 split-budget loss + 3-seed confirmation below.**
- **⭐ exact-value ranking v2 = split budget (`opener 0.10 + lower-exact 0.05`), 3-SEED CONFIRMED WIN (2026-07-23):** treatment ckpts `exact_value_rank/d20_exact_value_rank_v2_seed{1,2,3}/checkpoints/` (seed1 `epoch010-val_loss1.6964.ckpt`), paired control (opener-only, `LOWER_RANK_LAMBDA=0`) `exact_value_rank/ctrl_seed{1,2,3}/`. Loss-only change on unchanged d20 (`beast2c_d20_ceil.h5`); v2 keeps the opener auxiliary independent at 0.10 and adds a bounded lower-exact pool at 0.05 (fixes v1's opener-halving). Seed-averaged 2p all +3.2@2/+4.4@5/+5.9@10, hard +6.5@5/+8.2@10, avg-sims −9.9 (hard −18.6); hard-2p seed ranges NON-overlapping. 1p FLAT (hard@5 −0.3, all@1 −0.5 — inside ~5pt seed noise; the v1/v2/v3 single-seed hard-1p dips were eval-sim noise). **Recommend making this the default ranking-aux loss** (`RANK_LAMBDA=0.10 LOWER_RANK_LAMBDA=0.05`; already the `scripts/slurm/train.slurm` default). Code: `scripts/rl_loop/train_q2_rankaux.py` (worktree `exp/exact-value-ranking`). Colossus card default-flip staged for user (live-run timing). Card EXP-2026-07-22-exact-value-ranking-loss.
- **Eval/test sets** — see the [eval-set registry](eval_set_registry.md) for every test manifest, exhaustive GT, and their coverage/distribution (canonical vs dead-bank, EVAL-ONLY, the testset_gt↔pure2push 981-alignment). Training H5 disk-cleanup candidates: `beast2_all.h5` (3.3 GB, v0 lineage-only), `round2_raw.h5` (4.08 GB, raw intermediate) — retained pending user call.
- **Colossus d20+200k successors (DATA×LOSS grid, 2026-07-23):** all stack on d20 base (`beast2c_d20_ceil.h5`, 231,386 rows). Data axis = what Colossus experience is added; loss axis = opener-only (`LOWER_RANK_LAMBDA=0`) vs split (`0.1+0.05`). Single-seed unless noted; seed 2/3 confirmation of setup-only + split-full IN FLIGHT. Full → card EXP-2026-07-21-colossus, eval `round3/eval/*/`, plots `round3/eval/plots/`.
  - **opener-only colossus** (data +full 200k, opener-only loss, job 187062): `round3/models/colossus0_d20plus200k/checkpoints/epoch011-val_loss1.6542.ckpt`. 1p @1 easy/med/hard 97.6/83.6/**42.2**; hard tail @5/@10/@20 70.6/78.9/87.3; 2p all solve@900 96.7 / sims 51.1; reactive open@2 45.1. Net win over d20 (+6.0 reactive, faster 2p) with hard-1p deep-tail regression.
  - **split-full colossus** (data +full 200k, split loss, job 187440): `round3/models/colossus0_d20plus200k_splitloss/checkpoints/epoch010-val_loss1.6506.ckpt`. 1p @1 97.9/84.6/**33.3**; hard tail 64.2/77.0/85.8; 2p 96.3 / sims 44.0; reactive 47.3. Fastest 2p search + best reactive, but WORST hard-1p tail (split loss did NOT fix it — hypothesis falsified).
  - **⭐ setup-only colossus (the winner)** (data +setup-roots only 26,023 = 257,409 rows `d20_plus_setup_only.h5`, split loss, job 187593): `round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt`. 1p @1 97.6/84.6/**39.2**; hard tail @5/@10 **72.1/85.8 (best-of-four)**, @20 89.7, avg-sims 7.92 (lowest); 2p solve@900 all **97.5**/hard **94.1** (best-of-four) / sims 46.0; reactive 44.6. **RECOVERED the hard-1push tail** (dropping the 157k finish boards) while holding 2p — best overall tradeoff, RECOMMENDED successor. val_loss WORST of the variants yet deploy BEST (val_loss doesn't track deploy). Seed 2/3 + the setup+opener-loss cell (3 seeds) launching to confirm.
- **Colossus seeded confirmation (2026-07-24, 4 conditions × 3 seeds).** All ckpts under `round3/models/` unless noted; jobs 187732-187738. 1push seed-averaged CONFIRMS the setup-only tail recovery; 2push/reactive pending. Full → card EXP-2026-07-21-colossus.
  - **d20-ctrl** (d20 base + opener-only, the seeded d20): `exact_value_rank/ctrl_seed{1,2,3}/checkpoints/` = `epoch010-1.6909`/`epoch011-1.7179`/`epoch010-1.6749`. Reused exact-value control evals (`exact_value_rank/eval_ctrl_seed{1,2,3}/`). 3-seed mean reproduces registered d20 (hard 39.7/70.9/83.0/90.2, sims 8.0).
  - **setup+split** (job 187593/187732/187733): `d20_plus_setup_only_splitloss{,_seed2,_seed3}/checkpoints/` val_loss 1.6952/1.6927/1.7243. Hard-1p seed-avg [min,max]: @5 **72.4 [72.1,73.0]**, @10 82.5 [79.9,85.8], @20 90.9 [89.7,92.2], sims 8.2 [7.9,8.4].
  - **full+split** (187440/187734/187735): `colossus0_d20plus200k_splitloss{,_seed2,_seed3}/` val 1.6506/1.6561/1.6314. Hard-1p: @5 66.2 [64.2,68.1], @10 77.6 [77.0,78.4], @20 86.6 [85.8,87.3], sims 9.9 [9.6,10.2]. **Tail collapse REAL — non-overlapping ranges vs setup+split AND below seeded d20.**
  - **setup+opener** (187736/187737/187738): `d20_plus_setup_only_openeronly_seed{1,2,3}/` val 1.6930/1.6880/1.7170. Hard-1p: @5 70.6 [66.7,73.5], @10 82.8 [80.9,83.8], @20 90.7 [89.2,92.2], sims 8.3. **Recovers to d20-class = ranges overlap setup+split.**
  - **2push seed-averaged, mean [min,max]** — sims-to-solve all: d20 55.2 [48,61] / setup+split 45.2 [43.8,46] / full+split 42.0 [40.5,44] / setup+opener 46.2 [44.6,49]; hard 86.1/70.5/65.1/71.6. solve@900 4-way TIE (all ~96 / hard ~92). Reactive 3/12 seeds partial (not final).
  - **setup-only HARD+dense labels (2026-07-25, REJECTED, 1 seed):** `round3/models/setup_split_HARD_seed1/checkpoints/epoch011-val_loss0.8787.ckpt` (job 189366), data `round3/h5/d20_plus_setup_only_HARD.h5` — every reachable cell exact two-sided (opener 1.0/setup 0.9/dead+unknown 0), no fence, nothing masked. **No magnitude collapse** (opener raw median 0.949 vs ceiling 0.965 — marvel's 0.11–0.19 crush was FLOOD-driven, not intrinsic), but **ordering regresses**: 2push sims-to-solve 46.0→53.8 (+17%, above the ceiling 3-seed band max), solve@2/@5/@10 below band min, 1push top-1 down every tier; solve@900 unchanged. WINS the deep 1push tail (hard@5 **77.0** best-of-all, avg-sims 7.32 lowest). **Ceiling/fence stays default;** keep this rule in mind only for a deep-1push-search arm. val_loss NOT comparable across label regimes.
  - **Verdicts (seed-robust):** (a) full+split hard-1push tail collapse replicates across 3 seeds (not noise); (b) **recovery is a PURE DATA effect — dropping the 157k finisher boards, loss-independent** (setup recovers under BOTH losses; full collapses under both). (c) split vs opener loss INDISTINGUISHABLE on 1push AND 2push → loss doesn't matter on setup data, keep simpler. (d) 2push: all colossus beat d20 on sims (data helps); solve@900 4-way tie; full+split fastest sims-mean but NOT seed-robust vs setup+split (ranges overlap). **NET: setup-only = best all-around — only variant with BOTH d20-class hard-1push AND colossus-class 2push efficiency; full+split trades the hard tail for a non-robust sims edge. Seed-confirmed winner = setup DATA.**
