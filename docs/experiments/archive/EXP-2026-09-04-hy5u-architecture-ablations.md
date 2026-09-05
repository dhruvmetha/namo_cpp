---
type: experiment
status: done
created: 2026-09-04
updated: 2026-09-05
commit: ffe34dda
metric: "Canonical fixed-physics-v3 success versus simulator calls, split by easy/medium/hard and 1push/2push; three seeds per architecture control."
tags:
  - experiment
  - ablation
  - architecture
  - icra27
---
# HY5U architecture ablations

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a search heuristic that ranks pushes; the simulator remains the verifier. This experiment asks which candidate-specific parts of the ranker architecture produce its low-simulation ordering.

## Hypotheses

The user requested an architecture ablation and explicitly excluded a smaller-model control. The completed independent-contacts arm already removes inter-contact self-attention and will be reused rather than retrained.

- Replacing the 60 contact-token critic with one global scene token and a global 300-action readout will damage low-budget ordering because every action loses candidate-specific local and cross-attention reasoning.
- Removing only the local feature sampled at each contact will produce a smaller loss because Fourier contact position, learned edge identity, scene cross-attention, and inter-contact attention remain intact.

Historical pilots motivate both tests but do not replace them: on the older one-push stack, per-contact cross-attention improved hard solve@1 from 14.3 to 25.6 over global readout, while removing the local gather cost 3.2 points. The current run repeats those comparisons under HY5U's data, loss, seeds, and canonical best-first deployment.

## Plan

Train two new arms on the exact HY5U `hybrid_train_v1.h5` with the room-grouped split, setup target 0.5, unreachable-cell regression weight 0.1, complete ranking loss, 51-bin HL-Gauss head, 12 epochs, batch 256, learning rate 3e-4, and seeds 1/2/3.

| arm | one architecture change from HY5U |
|---|---|
| HY5U-global | replace all contact tokens and contact/scene attention with a CLS-style global scene readout over all 60×5 actions |
| HY5U-no-local | remove the scene feature sampled at each contact; retain Fourier contact coordinates, learned edge identity, scene cross-attention, and inter-contact self-attention |

Use the already completed HY5U-independent arm as the third architecture control. Do not train a width/depth control, no-edge-ID control, patch-resolution control, dual crop, or depth-token arm: the first two are less direct for the paper question and the latter three were already historically flat or negative.

Run a complete one-epoch smoke for both new architectures on the exact target box before releasing the six full trainings. Require finite train and validation losses, a checkpoint, strict reload, and the `eval_scorer-load check` marker. The older architecture experiments supply the exploratory pilot signal; this smoke gates current-code mechanics and calibrates target-box runtime.

After training, evaluate all three seeds of both arms on the canonical fixed-physics-v3 1-push and 2-push populations using `hmax=2`, simulator budget 900, `prior=model`, `agg=mean5`, raw `q`, discount off, no-op deduplication on, and jam-depth pruning on. Report solve@1 for 1-push and solve@5 for 2-push by easy/medium/hard/overall, plus the complete success-versus-simulator-call curves. Do not compare wall time because this campaign is not a pinned-hardware timing run.

## Run

Live placement on 2026-09-04 found iLab's two-hour GPU request projected for 2026-09-09 and Amarel's for 2026-09-05 01:51, beyond the one-hour fallback limit. Arrakis GPUs 1 and 2 were idle RTX 6000 Ada cards, so the target-box smoke and full background fleet use those two GPUs with six data workers per process, staying within the direct-box CPU courtesy limit.

The run uses NAMO commit `ffe34dda` and Sage commit `ceff4bf`.

Background supervisor PID `1655534` started on Arrakis at 14:50 EDT. It writes `architecture_ablations_20260904/supervisor.log` under the round-0 scratch directory, runs the global and no-local smokes concurrently on GPUs 1 and 2, and releases three paired seed waves only after both completion markers pass.

The post-training handoff is precommitted and runs as a separate background supervisor. It waits for all six 12-epoch completion markers, selects the minimum-validation-loss checkpoint from each seed, verifies hashes after transfer to a dedicated Amarel checkout, and submits one target-box smoke per architecture. Each successful smoke releases three 36-task canonical evaluation arrays, followed by strict per-seed aggregation jobs. The evaluation reuses the registered HY5U, Random, and independent-contacts results rather than recomputing them.

Both training smokes passed at 15:08 EDT after 1,116 seconds, and the paired seed-1 full runs started immediately. Post-training handoff supervisor PID `1722492` started on Arrakis at 15:16 EDT using NAMO evaluation commit `61b95a17` and Sage commit `ceff4bf`; it polls at five-minute intervals and will stage the Amarel target-box smokes and dependent full arrays without an interactive session.

All six trainings completed at 05:23 EDT on 2026-09-05, reached epoch 11, passed strict reload and scorer-load checks, and selected the epoch-11 checkpoint in every seed. The first Amarel smoke attempt, jobs `61242719` and `61242726`, failed before evaluating an episode because the remote launcher sourced `env.amarel.sh` after receiving the dedicated Sage path and silently replaced it with the older shared checkout; the resulting `EdgeCrossAttn` lacked `global_readout`. All six dependent full arrays and aggregate jobs were cancelled by `afterok`, so this attempt produced no test result and no partial population. The retry captures and restores the committed Sage and bindings paths across environment activation and writes to a fresh evaluation root.

The fixed launcher is commit `26f3ced0`. Retry smokes `61247464` and `61247471` started on Amarel at 15:45 EDT and produced real 1-push and 2-push rows with no loader errors. Their dependent full arrays are `61247465`, `61247467`, `61247469`, `61247472`, `61247474`, and `61247476`; per-seed aggregate jobs are `61247466`, `61247468`, `61247470`, `61247473`, `61247475`, and `61247477`. All retry artifacts use the fresh `eval/hy5u_arch_ablations_20260905_retry1` root under Amarel scratch.

All 216 full-evaluation array tasks completed with exit code `0:0` and wrote the exact 1,328 one-push plus 992 two-push rows per seed. Three aggregate jobs completed, while `61247470`, `61247473`, and `61247475` started immediately after their parent arrays and observed only 989–990 of the 992 two-push rows through the shared filesystem. A later direct recount and canonical-key match found all 992 unique rows with zero worker errors in every affected seed, so no simulator work is missing; only those three strict aggregation jobs will be retried against the now-stable files.

## Result

All 216 full-evaluation array tasks completed with exit code `0:0`. After the three aggregation retries described above, `agg_search_eval.py` accepted exactly 1,328 one-push and 992 two-push episode rows for every seed, rejected no duplicate population or mixed search configuration, and verified `hmax=2`, no-op deduplication, and jam-depth pruning. Values below are solve rate in percent, mean ± sample SD across three seeds. Wall time is excluded because this was not a pinned-hardware timing campaign.

| architecture | change from HY5U | 1push easy@1 | medium@1 | hard@1 | all@1 |
|---|---|---:|---:|---:|---:|
| HY5U | full contact-token ranker | 97.1±0.5 | 79.8±0.3 | 40.2±1.2 | 82.5±0.4 |
| no local feature | remove the scene feature sampled at each contact | 97.2±0.8 | 80.3±0.6 | 38.7±1.8 | 82.5±0.8 |
| independent contacts | remove inter-contact self-attention | 96.2±1.0 | 76.3±1.2 | 35.4±0.6 | 80.2±0.4 |
| global readout | replace contact tokens with one global scene readout | 76.5±1.9 | 50.2±0.8 | 14.8±1.6 | 58.3±1.0 |
| Random | uniform ordering | 61.1±4.6 | 14.1±1.7 | 2.9±0.8 | 36.5±2.8 |

| architecture | 2push easy@5 | medium@5 | hard@5 | all@5 | all@900 |
|---|---:|---:|---:|---:|---:|
| HY5U | 80.6±1.6 | 59.3±0.6 | 35.9±2.1 | 64.8±0.8 | 93.0±0.2 |
| no local feature | 79.5±1.3 | 58.3±0.9 | 33.6±2.1 | 63.6±0.6 | 93.4±0.2 |
| independent contacts | 76.7±1.2 | 52.8±1.7 | 30.5±3.0 | 59.5±1.1 | 92.8±0.1 |
| global readout | 57.8±2.3 | 36.6±1.7 | 17.0±1.4 | 42.5±1.8 | 92.4±0.4 |
| Random | 22.8±3.6 | 7.2±1.7 | 2.0±1.3 | 12.7±2.0 | 88.4±0.6 |

![Exact three-seed verified-success curves for HY5U, three architecture controls, and Random, split by difficulty and horizon.](../plots/hy5u_architecture_ablations/success_vs_sims_both_horizons.png)

![Paired-seed change from HY5U at one-push solve@1 and two-push solve@5.](../plots/hy5u_architecture_ablations/ablation_effects.png)

The global readout loses 24.3 points at one-push solve@1 and 22.3 points at two-push solve@5 overall. Its two-push ceiling nearly recovers by 900 calls, 92.4% versus HY5U's 93.0%, so its main failure is the ordering that the ranker is meant to provide rather than broad loss of solvability. The remaining hard-tier ceiling gap is larger, 82.5% versus 87.6%.

Removing the local sampled scene feature is effectively neutral: one-push solve@1 is unchanged overall and two-push solve@5 falls only 1.2 points, with paired-seed mean ± SD spanning zero. Removing inter-contact self-attention has the clearer intermediate effect, losing 2.4 points at one-push solve@1 and 5.4 points at two-push solve@5 overall.

The six new aggregate JSONs and mirrored raw JSONLs are under `$NAMO_SCRATCH/eval/hy5u_arch_ablations_20260905_retry1/full/`; Amarel retains the complete raw campaign under the matching scratch-relative path. Reproducible PDFs, PNGs, and the five-series aggregate are under `docs/experiments/plots/hy5u_architecture_ablations/`; `scripts/experiments/hy5u_architecture_ablation_series.json` is the exact plot specification.

## Verdict

The global-readout hypothesis is confirmed strongly: candidate-specific contact tokens and their scene interaction are necessary for HY5U's low-simulation ordering. The no-local hypothesis is confirmed only in its predicted ordering of effect sizes; the measured change is too small and inconsistent to claim that the sampled local feature contributes under this architecture. Inter-contact attention earns its place through a smaller but consistent gain. The architectural attribution is therefore: candidate-specific representation is essential, cross-contact reasoning helps, and the explicit sampled local feature is redundant given contact coordinates, edge identity, and scene cross-attention.
