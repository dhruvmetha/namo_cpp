---
type: experiment
status: idea
created: 2026-08-02
commit: pending
metric: Preflight complete; no training or evaluation launched
thread: region_opening
parent: EXP-2026-07-22-push-depth-aware-ranker
tags: [experiment, ranker, architecture, action-motion, depth-attention, colossus]
---
# Motion-grounded depth tokens with local attention

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** This remains one ranker that orders simulator-verified pushes; it is not a push-success classifier or a replacement for search.

## Hypotheses

_(you, via chat 2026-08-02)_ Use the best `d20+setup-only` model dataset, convert the 60 contact tokens into 60×5 tokens with depth information injected, and let one attention head exchange information among the five push depths belonging to an edge.

_(Claude, falsifiable refinement)_ The earlier additive motion heads improved hard-1push shortlist recall but failed to improve top-1 reliably because each depth was scored independently after expansion. Letting the five motion-grounded depths of one contact compare against each other should improve their relative order, especially the setup-versus-overpush choice in medium/hard 2push search, without paying for or permitting attention across all 300 actions.

## Plan

Use the exact deployed lineage as the control: `/common/users/dm1487/scratch_namo/curriculum2/beast/round3/h5/d20_plus_setup_only.h5`, seed 1, room-grouped split seed 0, 12 epochs, batch 256, learning rate `3e-4`, patience 2, 51-bin HL-Gauss value head, one-sided ceiling loss, and listwise coefficients `RANK_LAMBDA=0.10`, `LOWER_RANK_LAMBDA=0.05`, `RANK_TEMP=0.15`. The deployed seed-1 checkpoint is `/common/users/dm1487/scratch_namo/curriculum2/beast/round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt`.

Keep the existing scene encoder and four contact blocks unchanged: each of the 60 contact tokens still gathers its local visual feature, cross-attends to the full scene, and self-attends across contacts. Only after that exchange finishes, add the existing corrected sharp crop-relative action motion and learned depth identity to form `(B,60,5,D)` complete-push tokens.

Reshape only to `(B×60,5,D)` and apply one local attention head, so a contact's five depths can compare with one another while different contacts cannot mix at this stage. Reshape back to `(B,60,5,D)` and preserve the deployed `(B,60,5,51)` HL-Gauss output contract. Do not add an all-300-token attention block.

The treatment is enabled only by `NAMO_ACTION_MOTION=1 NAMO_ACTION_MOTION_SHARP=1 NAMO_ACTION_DEPTH_SELF_ATTN=1`; all three defaults remain off, and checkpoint loading detects the local-depth block from its state dictionary. The prior baseline and motion-only checkpoints must continue to load strictly.

Run the seed-1 treatment first. Evaluate both canonical horizons and every easy/medium/hard tier. For 1push, report solve@1/@5 and the right-contact/wrong-depth decomposition. For 2push, make the primary architecture comparison under `hmax=2`, `combine=q`, **discount off**, no-op deduplication, and jam-depth pruning, reporting solve@2/@5/@10/@30/@900 and simulator calls to solution. This keeps post-failure board switching from confounding whether the new ranker itself improved. A confidence-discount `tau=0.15` run may be reported separately as a deployed-system compatibility check, never as the causal architecture readout. Use `scripts/eval_auc.py` on the exhaustive 2push GT to report setup hit@1/@5, setup rank, and finish rank rather than inventing another ranking metric.

Advance to three treatment seeds only if seed 1 reduces medium+hard 2push simulator calls by at least 10%, does not reduce solve@900 by more than one point on either tier, and keeps every 1push tier within two points at solve@1. The architecture claim is rejected if validation loss improves without the search-order improvement.

## Run

**Preflight only; no training or evaluation job has been launched.** The implementation adds a single-head local-depth block behind the opt-in flag, wires it through training and all canonical checkpoint loaders, and leaves the prior architectures free of new state-dict keys.

The deployed network has 4,397,055 parameters, the existing sharp-motion late-fusion network has 4,405,491, and this treatment has 4,554,099: +157,044 parameters (+3.6%) versus deployed and +148,608 versus motion-only. The extra attention is linear over 60 groups of five tokens, not quadratic over 300 tokens.

Focused CPU gates pass: 13/13 action-motion/architecture/loader tests and 10/10 existing rank-loss/scorer-diagnostic tests. A real row from `d20_plus_setup_only.h5` completed forward, the existing split ceiling loss, and backward with input motion `(1,60,5,3)`, logits `(1,60,5,51)`, finite loss `4.19665`, and finite nonzero local-attention gradient. The actual deployed checkpoint also reloads strictly as 4,397,055 parameters with motion and local-depth attention both off.

Recommended run name: `d20_setup_depthlocal_s1`.

One-epoch CS target smoke command, intentionally not run:

```bash
NAMO_ACTION_MOTION=1 NAMO_ACTION_MOTION_SHARP=1 NAMO_ACTION_DEPTH_SELF_ATTN=1 H5=/common/users/dm1487/scratch_namo/curriculum2/beast/round3/h5/d20_plus_setup_only.h5 OUT=/common/users/dm1487/scratch_namo/curriculum2/beast/round4/models/d20_setup_depthlocal_s1_smoke EPOCHS=1 BATCH=256 WORKERS=0 LR=3e-4 SEED=1 PATIENCE=0 POSTCHECK_LIMIT=3000 sbatch --job-name=depthlocal_smoke --time=00:30:00 scripts/slurm/train.slurm
```

Full seed-1 command after the required pre-run commit, intentionally not run:

```bash
NAMO_ACTION_MOTION=1 NAMO_ACTION_MOTION_SHARP=1 NAMO_ACTION_DEPTH_SELF_ATTN=1 H5=/common/users/dm1487/scratch_namo/curriculum2/beast/round3/h5/d20_plus_setup_only.h5 OUT=/common/users/dm1487/scratch_namo/curriculum2/beast/round4/models/d20_setup_depthlocal_s1 EPOCHS=12 BATCH=256 WORKERS=8 LR=3e-4 SEED=1 PATIENCE=2 POSTCHECK_LIMIT=3000 sbatch --job-name=depthlocal_s1 --time=08:00:00 scripts/slurm/train.slurm
```

## Result + Verdict

Pending training and the required difficulty×horizon evaluations.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
