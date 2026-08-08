---
type: experiment
status: live
created: 2026-08-02
commit: namo 8891de3
metric: pending
thread: region_opening
parent: EXP-2026-07-29-post-pruning-canonical-search
tags: [experiment, search, random-baseline, canonical-eval, no-discount, hmax2]
---
# Canonical no-discount random baseline

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** Random is the baseline ordering for the same simulator-verified search; the measurement is success versus simulator calls, split by horizon and difficulty.

## Hypothesis

_(you, via chat 2026-08-02)_ Verify whether the full canonical three-seed random baseline already exists under the clean no-discount search; if it does not, prepare and run it rather than rerunning an incompatible control.

_(Claude, falsifiable refinement)_ Under identical `hmax=2`, budget 900, `combine=q`, discount-off, no-op-dedupe, and jam-depth-pruning search, the deployed ranker should retain a simulator-efficiency advantage over uniform random ordering on every easy/medium/hard tier of both registered horizons.

## Plan

Reuse the registered finalized populations exactly: 1,322 episodes from `namo.eval_sets.ONEPUSH` and 1,012 episodes from `namo.eval_sets.PURE2PUSH`, with 2push tiers from `namo.eval_sets.DIVISIONS`. The episode unit remains `(xml, object_id, goal region)`; sharding may slice XML files, but aggregation must retain and count every episode separately.

Run uniform-random seeds 7000/8000/9000 through the same evaluator and final search mechanics as `deploy-nodiscount-hmax2-v1`: `hmax=2`, budget 900, `agg=mean5`, `combine=q`, `discount=off`, `tau=1.0` (recorded but inactive), no-op dedupe on, jam-depth pruning on, raw scores off. Do not rerun the deployed model control.

Use the existing `scripts/amarel/launch_postprune_eval.sh` and `scripts/amarel/bestfirst_eval.slurm`; the launcher is extended in place with configurable discount and a `RUN_MODEL=0` switch rather than copied. Output root: `/scratch/dm1487/eval/postprune_hmax2/nodiscount_random_v1/` on Amarel, then sync to `/common/users/dm1487/scratch_namo/eval/postprune_hmax2/nodiscount_random_v1/`.

Before the full launch, run one real canonical hard-2push unit end-to-end on Amarel `main-redhat`, confirm a JSONL row whose embedded search dictionary matches every required field, and calibrate the shard walltime from that smoke. The planned full shape is six arrays, `{1push,2push} × seeds {7000,8000,9000}`, 30 one-push tasks plus 38 two-push tasks per seed, 204 tasks total, two CPUs per task, with at most 40 tasks from each array active and a three-hour task limit.

Aggregate each seed with `scripts/rl_loop/agg_search_eval.py`, requiring exactly 1,322 matched 1push rows and 1,012 matched 2push rows, then report seed mean ± sample standard deviation at solve@{1,2,5,10,30,100,300,900}, average simulator calls, and solved-only calls for every tier and horizon. Compare only simulator calls across boxes; no wall-time claim is allowed from this parallel array.

On completion, add `random-nodiscount-hmax2-v1` to the evaluated-artifact registry, add the machine-readable aggregate/raw paths to the eval-set registry, and append the main difficulty×horizon table to `RESULTS.md`.

## Run

**Artifact audit complete; no jobs submitted.** The exact three-seed finalized no-discount baseline is absent from the evaluated-artifact registry, CS scratch, and Amarel scratch.

The nearest old artifacts are not reusable as the final baseline: one pre-finalization no-discount 2push run covers 1,018 episodes with one random seed; a three-seed no-discount depth study covers only 180 2push episodes; and the finalized 1,322+1,012 three-seed random run uses confidence discount `tau=0.15`. None combines the finalized populations, three seeds, discount off, and both adopted pruning rules.

Preparation stops before the required target-box smoke at the user's request. No SLURM job ID exists yet.

The prepared smoke target is sorted 2push XML index 2, a single-episode hard case: `aug9_car/set1/benchmark_1/run_0367/env_0367_pair_001.xml`, object `obstacle_0_movable`, region `goal`. Submit only array task 2 with `SHARD=1`, seed 7000, and the full 2push key; this runs exactly that one episode and exercises the expensive tier.

Prepared smoke command, intentionally not run:

```bash
cd /cache/home/dm1487/projects/namo/namo_postprune_eval && source env.amarel.sh >/dev/null && sbatch --array=2 --export="ALL,MANIFEST=,NAMO_REPO=$PWD,CKPT=/cache/home/dm1487/eval_inputs/postprune_hmax2/setup_only.ckpt,KEY=/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push_search_eval.json,OUT_DIR=/scratch/dm1487/eval/postprune_hmax2/nodiscount_random_v1/smoke_hard2_s7000,HMAX=2,SIM_BUDGET=900,PRIOR=uniform,AGG=mean5,COMBINE=q,DISCOUNT=off,TAU=1.0,SEED_BASE=7000,SHARD=1" scripts/amarel/bestfirst_eval.slurm
```

Prepared full command after the smoke passes and sets the calibrated walltime, intentionally not run:

```bash
cd /cache/home/dm1487/projects/namo/namo_postprune_eval && source env.amarel.sh >/dev/null && CKPT=/cache/home/dm1487/eval_inputs/postprune_hmax2/setup_only.ckpt OUT_ROOT=/scratch/dm1487/eval/postprune_hmax2/nodiscount_random_v1 DISCOUNT=off TAU=1.0 RUN_MODEL=0 MAX_PARALLEL=40 bash scripts/amarel/launch_postprune_eval.sh
```

## Result + Verdict

Pending launch and aggregation.
