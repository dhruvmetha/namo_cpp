#!/bin/bash
# Full-set best-first WALL-TIME campaign: 3 seeds x {1-push, 2-push}, budget 900, sharded.
# - random gets a DISTINCT --rng-seed per seed (7/8/9) -> three genuinely different random baselines.
# - 2-push pinned icelake, 1-push pinned sapphirerapids: same CPU per push count => cross-seed times are poolable.
# Output dirs: full{1,2}_{s1,s2,s3}_b900/shard_*.jsonl
set -euo pipefail
REPO="${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"; cd "$REPO"
S=/scratch/dm1487/sage_outputs/scorer
DS=/scratch/dm1487/datasets/namo_testset_v1/labels
OUT=/scratch/dm1487/eval/timebench
declare -A HZ=(
  [s1]=$S/qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt
  [s2]=$S/qfull_v3_v4hq_s2/namo-classifier/xt0cuus6/checkpoints/epoch014-val_loss0.6640.ckpt
  [s3]=$S/qfull_v3_v4hq_s3/namo-classifier/fw9nr7kd/checkpoints/epoch010-val_loss0.6630.ckpt )
declare -A NOHZ=(
  [s1]=$S/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt
  [s2]=$S/qfull_nohz_v3_v4hq_s2/namo-classifier/kzph0acr/checkpoints/epoch012-val_loss0.6922.ckpt
  [s3]=$S/qfull_nohz_v3_v4hq_s3/namo-classifier/dlopoael/checkpoints/epoch011-val_loss0.6897.ckpt )
declare -A RNG=( [s1]=7 [s2]=8 [s3]=9 )
for sd in s1 s2 s3; do
  sbatch --constraint=icelake --array=0-19 \
    --export="ALL,OUT_DIR=$OUT/full2_${sd}_b900,SHARD=51,BUDGET=900,HMAX=2,KEY=$DS/pure2push.json,HZ_CKPT=${HZ[$sd]},NOHZ_CKPT=${NOHZ[$sd]},RNG_SEED=${RNG[$sd]},NAMO_REPO=$REPO" \
    scripts/amarel/time_bestfirst_shard.slurm | sed "s/^/  2push $sd -> /"
  sbatch --constraint=sapphirerapids --array=0-19 \
    --export="ALL,OUT_DIR=$OUT/full1_${sd}_b900,SHARD=67,BUDGET=900,HMAX=1,KEY=$DS/onepush_episodes.json,HZ_CKPT=${HZ[$sd]},NOHZ_CKPT=${NOHZ[$sd]},RNG_SEED=${RNG[$sd]},NAMO_REPO=$REPO" \
    scripts/amarel/time_bestfirst_shard.slurm | sed "s/^/  1push $sd -> /"
done
echo "launched 6 arrays (3 seeds x 2 push, full set, budget 900, random rng 7/8/9)"
