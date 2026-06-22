#!/bin/bash
# ExIt ON-POLICY FINISH collection, sharded by TRAIN-scene index. Each task collects the model's on-policy
# s1 + exhaustive-a2 finish labels for its slice and writes one scorer H5 shard. Join the shards into the
# retrain mix in place of / alongside postpush. CPU (sims dominate; one model fwd/setup).
# Required env: CKPT (the deployed model = Hz-v2). Optional: KEY, TOPK_SETUPS(2), OUT_DIR, SHARD(102), NSCENES(5076).
#SBATCH --job-name=exit_collect
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-49
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/dm1487/logs/exit-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/exit-%A_%a.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""

: "${CKPT:?}"
KEY=${KEY:-/scratch/dm1487/datasets/v4_hq_h2/labels_exhaustive_pure2push.json}
TOPK_SETUPS=${TOPK_SETUPS:-2}; SHARD=${SHARD:-102}; SETUPS=${SETUPS:-model}
OUT_DIR=${OUT_DIR:-/scratch/dm1487/h5/v4_hq_exit_finish}
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
echo "task ${SLURM_ARRAY_TASK_ID}: scenes [$START,$END) topk=$TOPK_SETUPS setups=$SETUPS -> $OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.h5 on $(hostname)"
/scratch/dm1487/envs/namo/bin/python -u scripts/pipeline/exit_collect.py \
    --ckpt "$CKPT" --key "$KEY" --start "$START" --end "$END" --topk-setups "$TOPK_SETUPS" --setups "$SETUPS" \
    --out-h5 "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.h5"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
