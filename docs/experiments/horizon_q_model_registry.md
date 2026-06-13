# Horizon-Q Model Registry

> THE authoritative location list — every trained model, its exact best-val checkpoint, headline number,
> training data, and eval-output dir. **Do not reconstruct paths by glob; read here.** Never retrain
> registered models ([[feedback_reuse_baselines]]). Roots: ckpts `/scratch/dm1487/sage_outputs/scorer/`,
> evals `/scratch/dm1487/eval/`, H5s `/scratch/dm1487/h5/`. Updated 2026-06-13.

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
- BEST: s1 `m2a_v4hq_s1/namo-classifier/fixo4s7i/checkpoints/epoch015-val_loss0.7605.ckpt` ·
  s2 `m2a_v4hq_s2/namo-classifier/670z73ct/checkpoints/epoch015-val_loss0.7674.ckpt` ·
  s3 `m2a_v4hq_s3/namo-classifier/hi0u7t1o/checkpoints/epoch015-val_loss0.7558.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m1_scorer/data.h5` · evals: `/scratch/dm1487/eval/m2a_verdict/`
- **hard@1 = 29.62 ± 0.93** · flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h

### M2b — + dead-ends (THE current best 1-push model / Q-full warm-start + baseline)
- BEST: s1 `m2b_v4hq_s1/namo-classifier/tryggakf/checkpoints/epoch012-val_loss0.6814.ckpt` ·
  s2 `m2b_v4hq_s2/namo-classifier/rxn54385/checkpoints/epoch016-val_loss0.6802.ckpt` ·
  **s3 (best-val, used for fpv) `m2b_v4hq_s3/namo-classifier/ql60myva/checkpoints/epoch013-val_loss0.6780.ckpt`**
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (252,805 = 123,269 solvable + 129,536 dead)
- evals: `/scratch/dm1487/eval/m2b_verdict/` · 2-push search: `/scratch/dm1487/eval/fpv_m2b/`
- **hard@1 = 32.86 ± 2.38** (@5 65.4) · dead-slice cand-pool V_dead 0.065, AUC 0.987 ·
  2-push search 75.2@1 / e2e 61.9%

### M2c — + 20 unreachable-cell zeros (ablation; data-side flag unreachable_k=20)
- BEST: **s1 (fpv) `m2c_v4hq_s1/namo-classifier/y1rtra4f/checkpoints/epoch013-val_loss0.6793.ckpt`** ·
  s2 `m2c_v4hq_s2/namo-classifier/i5fu1h23/checkpoints/epoch016-val_loss0.6853.ckpt` ·
  s3 `m2c_v4hq_s3/namo-classifier/d9l0zs60/checkpoints/epoch013-val_loss0.6845.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (same as M2b) · evals: `/scratch/dm1487/eval/m2c_verdict/`
  · 2-push search: `/scratch/dm1487/eval/fpv_m2c/` (job 56008453, running)
- **hard@1 = 32.21 ± 1.48** (ranking ≈ M2b) · all-cells V_dead **0.072** (vs M2b 0.327 — hallucination killed)

### M2d — + reachability input bit (ablation; reach_flag_input, network flag)
- BEST: s1 `m2d_v4hq_s1/namo-classifier/eoz348sj/checkpoints/epoch015-val_loss0.6828.ckpt` ·
  s2 `m2d_v4hq_s2/namo-classifier/yvq3jtsq/checkpoints/epoch012-val_loss0.6793.ckpt` ·
  s3 `m2d_v4hq_s3/namo-classifier/be1ehs8b/checkpoints/epoch016-val_loss0.6771.ckpt`
- data: `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` · evals: `/scratch/dm1487/eval/m2d_verdict/`
- **hard@1 = 34.20 ± 2.09** (within noise of M2b) · all-cells V_dead 0.621 (degraded — tell<teach)

### Q-full (M3/M4) — mixed-H, TRAINING (job **56015587**, 3 seeds, L40S; 56013237/56013312 = dead false-starts)
- run dirs `/scratch/dm1487/sage_outputs/scorer/qfull_v4hq_s{1,2,3}/`; ep11 BEST ckpts (still training, val↓):
  - s1 `qfull_v4hq_s1/namo-classifier/zxt3n1tm/checkpoints/epoch011-val_loss0.6572.ckpt`
  - s2 `qfull_v4hq_s2/namo-classifier/a91eflex/checkpoints/epoch011-val_loss0.6582.ckpt`
  - s3 `qfull_v4hq_s3/namo-classifier/h2lcraeg/checkpoints/epoch011-val_loss0.6582.ckpt`
- **ep11 H=1 feeler (job 56022469): hard@1/@5 — s1 34.4/65.1 · s2 41.3/67.2 · s3 30.7/68.3 · MEAN 35.5/66.9
  vs M2b 32.86/65.4 (+2.6/+1.5pp). H=1 sanity only; M3 foresight = headline. wrong-edge/miss≈89%.**
- data (multi-H5, ';'-joined): `/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5` (252,805 H=1 rows) +
  `/scratch/dm1487/h5/v4_hq_h2_scorer/data.h5` (311,324 = 155,662 ep × {H=1,H=2}; 172,104 dead; gamma=0.9,
  format=twopush) → ~564k total rows, room-grouped across files (realpath-normalized)
- flags: budget_cond, value_bins=51, head_mode=hl_gauss, budget_h (NO unreachable_k/reach_flag — M2c/M2d parked)
- evals (planned): `/scratch/dm1487/eval/qfull_verdict/` + `fpv_qfull/` + bifurcation/M3 slices
- GATES when done: (a) H=1 ranking ≈ M2b 32.86; (b) M3 = zero-sim setup-pick on pure2push vs 34.5 (registered)
  & 75.2-with-sims (fpv_m2b); (c) H-bifurcation probe (525ea31); (d) per-division (pure2push_divisions)

## Key H5 datasets
- `v4_hq_m1_scorer/data.h5` — M1/M2a solvable-only (123,269)
- `v4_hq_m2b_scorer/data.h5` — M2b/M2c/M2d, +dead (252,805)
- `v4_hq_h2_scorer/data.h5` — Q-full H=2 mixed rows (≈311k = 155,662 episodes × 2), BUILDING (chain2 55972757)
- mask H5s: `v4_hq_m1_65_35`, `v4_hq_de_masks(_rest)`, `v4_hq_h2_root_p{0..3}`

## Eval / test keys
- 1-push answer key: `/scratch/dm1487/datasets/namo_testset_v1/labels/onepush_episodes.json`
- 2-push key + divisions: `.../labels/pure2push.json`, `.../labels/pure2push_divisions.json` (hard≤2/med 3-8/easy>8 setups)
- fpv aggregate (M2b): `/scratch/dm1487/eval/diag_fpv_aggregate.json`
