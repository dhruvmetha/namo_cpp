---
type: experiment
status: done
created: 2026-08-02
commit: namo 1aea06a; sage 6f90dc6
metric: REJECT — hard 1p @1 −5.9 pp; medium+hard 2p calls +11.0%; hard 2p @5 +6.6 pp
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

The implementation adds a single-head local-depth block behind the opt-in flag, wires it through training and all canonical checkpoint loaders, and leaves the prior architectures free of new state-dict keys.

The deployed network has 4,397,055 parameters, the existing sharp-motion late-fusion network has 4,405,491, and this treatment has 4,554,099: +157,044 parameters (+3.6%) versus deployed and +148,608 versus motion-only. The extra attention is linear over 60 groups of five tokens, not quadratic over 300 tokens.

Focused CPU gates pass: 13/13 action-motion/architecture/loader tests and 10/10 existing rank-loss/scorer-diagnostic tests. A real row from `d20_plus_setup_only.h5` completed forward, the existing split ceiling loss, and backward with input motion `(1,60,5,3)`, logits `(1,60,5,51)`, finite loss `4.19665`, and finite nonzero local-attention gradient. The actual deployed checkpoint also reloads strictly as 4,397,055 parameters with motion and local-depth attention both off.

Recommended run name: `d20_setup_depthlocal_s1`.

**Training completed 2026-08-02.** The first target-box smoke on ilab3 completed one epoch and wrote a reloadable checkpoint; Blackwell job `197731` then passed a real forward/backward smoke with the CUDA-12.8 environment. Full seed-1 job `198016` completed all 12 epochs on one RTX PRO 5000 Blackwell in 22m14s with exit code 0.

The full run used exactly 257,409 rows and the registered room-grouped split: 231,668 train rows and 25,741 validation rows across 20,162 validation rooms. Validation loss fell from 2.3345 at epoch 0 to the best 1.7126 at epoch 11; the deployed same-data control is 1.6952, so validation loss alone does not favor the treatment.

Checkpoint: `/common/users/dm1487/scratch_namo/curriculum2/beast/round4/models/d20_setup_depthlocal_s1/checkpoints/epoch011-val_loss1.7126.ckpt`. Reload is bit-identical (`max|Δlogit|=0`), reloaded validation loss is 1.7127, and the canonical eval loader detects the `(60,5,51)` value head plus `action_motion_encoding=crop_relative` and the local-depth attention block.

Canonical seed-1 evaluation completed 2026-08-02 on all 1,322 registered 1push and 1,012 registered 2push episodes. Both treatment and control used `hmax=2`, budget 900, `combine=q`, discount off, no-op dedupe on, and jam-depth pruning on; raw JSONL rows record this full search dictionary and the aggregate gate rejected duplicates, missing episodes, mixed configs, or wrong populations.

Reusable artifacts are rooted at `/common/users/dm1487/scratch_namo/curriculum2/beast/round4/eval/depthlocal_s1_nodiscount/`: `treatment/aggregate_hmax2.json` and `control/aggregate_hmax2.json`; raw rows live under each arm's `1push_hmax2/` and `2push/`. Offline exhaustive-GT ranking is `auc/result.json`; direct canonical 1push diagnostics are `{treatment,control}/1push/result.json` with raw `leaf.jsonl`. These are registered as `depth-token-nodiscount-hmax2-s1`, `deploy-nodiscount-hmax2-v1`, and `depth-token-offline-s1` in the model/evaluation artifact registry.

One-epoch CS target smoke command, intentionally not run:

```bash
NAMO_ACTION_MOTION=1 NAMO_ACTION_MOTION_SHARP=1 NAMO_ACTION_DEPTH_SELF_ATTN=1 H5=/common/users/dm1487/scratch_namo/curriculum2/beast/round3/h5/d20_plus_setup_only.h5 OUT=/common/users/dm1487/scratch_namo/curriculum2/beast/round4/models/d20_setup_depthlocal_s1_smoke EPOCHS=1 BATCH=256 WORKERS=0 LR=3e-4 SEED=1 PATIENCE=0 POSTCHECK_LIMIT=3000 sbatch --job-name=depthlocal_smoke --time=00:30:00 scripts/slurm/train.slurm
```

Full seed-1 command after the required pre-run commit, intentionally not run:

```bash
NAMO_ACTION_MOTION=1 NAMO_ACTION_MOTION_SHARP=1 NAMO_ACTION_DEPTH_SELF_ATTN=1 H5=/common/users/dm1487/scratch_namo/curriculum2/beast/round3/h5/d20_plus_setup_only.h5 OUT=/common/users/dm1487/scratch_namo/curriculum2/beast/round4/models/d20_setup_depthlocal_s1 EPOCHS=12 BATCH=256 WORKERS=8 LR=3e-4 SEED=1 PATIENCE=2 POSTCHECK_LIMIT=3000 sbatch --job-name=depthlocal_s1 --time=08:00:00 scripts/slurm/train.slurm
```

## Result + Verdict

### Live hmax=2 search: exact verified success versus simulator calls

“Tight” is solve@1 for 1push and solve@2 for genuine 2push. Average calls include budget-900 failures, so it is the honest fixed-budget cost.

| horizon | tier | tight control → depth-token | solve@5 control → depth-token | solve@30 control → depth-token | solve@900 control → depth-token | avg calls control → depth-token |
|---|---|---:|---:|---:|---:|---:|
| 1push | easy | 97.7 → 98.3 | 99.9 → 99.9 | 100.0 → 100.0 | 100.0 → 100.0 | 1.1 → 1.0 |
| 1push | medium | 84.6 → 84.3 | 97.9 → 97.6 | 99.8 → 100.0 | 100.0 → 100.0 | 1.8 → 1.4 |
| 1push | hard | **39.7 → 33.8** | **82.4 → 77.5** | 96.6 → 95.1 | 100.0 → 100.0 | **7.6 → 10.1** |
| 2push | easy | 44.4 → 44.7 | 67.3 → 69.9 | 94.3 → 93.0 | 100.0 → 100.0 | 9.3 → 10.9 |
| 2push | medium | **32.8 → 30.3** | **57.4 → 51.0** | 80.1 → 81.8 | 99.6 → 99.2 | **38.1 → 46.6** |
| 2push | hard | **9.5 → 13.9** | **22.6 → 29.2** | **50.4 → 54.0** | 92.0 → 92.7 | 149.5 → 150.7 |

The intended hard-2push effect is real: depth-token improves hard solve@2/@5/@10/@30 by +4.4/+6.6/+3.6/+3.6 points and lowers solved-case median calls from 22 to 15. It does not improve the tail cost: hard average calls are flat-to-worse, and solved-only average rises 84.0→91.7 because the extra solves are expensive.

The trade is unacceptable for the single deployed ranker. Hard-1push solve@1 falls 5.9 points and average calls rise 33%; medium-2push solve@5 falls 6.4 points and average calls rise 22%. Weighted over medium+hard 2push, average calls rise from 62.52 to 69.42, **+11.0%**, instead of the preregistered ≥10% reduction.

### Exhaustive-GT ranking diagnosis

| tier | setup @1 control → depth-token | setup @5 control → depth-token | finisher @1 control → depth-token | finisher @5 control → depth-token | live-vs-dead board max AUC control → depth-token |
|---|---:|---:|---:|---:|---:|
| easy | 76.9 → 73.2 | 94.7 → 95.2 | 72.0 → 70.4 | 92.2 → 91.5 | 0.653 → 0.705 |
| medium | 55.1 → 54.1 | 80.7 → 78.1 | 64.3 → 61.9 | 88.5 → 87.3 | 0.756 → 0.767 |
| hard | 21.2 → 22.0 | **44.9 → 55.1** | 51.2 → 53.7 | 83.9 → 81.8 | 0.725 → 0.752 |

This explains the mixed search curve: local depth attention moves hard setups into the shortlist and improves live-vs-dead board-max separation, but it sacrifices precise hard-1push contact/depth ordering and medium setup/finisher ranking. The architecture changes which regime wins; it does not produce a uniformly better ordering.

**VERDICT: REJECT as the next deployed single ranker; do not advance to three seeds.** It fails two preregistered gates: every 1push tier had to remain within two points at solve@1, but hard is −5.9; medium+hard 2push calls had to fall ≥10%, but they rise 11.0%. The useful retained finding is narrower: five-depth local attention is a credible hard-setup-shortlist mechanism, worth revisiting only with a loss/data design that preserves 1push and medium ordering or as a specialized ablation—not as a larger generic model.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
