#!/bin/bash
# FULL-test-set finish-separation gate, sharded by xml-index over CPUs (the n=120 feeler was too small for a verdict).
# Each task scores one xml-slice's finish (fixed-s1 = same s1 both models) and writes a shard json with raw sums+counts
# for EXACT aggregation. Required env: CKPT, OUT_DIR. Optional: KEY, SHARD(25), FIXED(--fixed-setup).
#   CKPT=... OUT_DIR=/scratch/dm1487/eval/fsep_full_v3 sbatch scripts/amarel/finish_sep.slurm
#SBATCH --job-name=fsep
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/dm1487/logs/fsep-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/fsep-%A_%a.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${CKPT:?}"; : "${OUT_DIR:?}"
KEY=${KEY:-/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json}
SHARD=${SHARD:-25}; FIXED=${FIXED:---fixed-setup}
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
echo "task ${SLURM_ARRAY_TASK_ID}: xmls [$START,$END) ckpt=$(basename "$CKPT") fixed='$FIXED' -> shard_${SLURM_ARRAY_TASK_ID}.json"
/scratch/dm1487/envs/namo/bin/python -u scripts/sandbox/check_h2_finish.py \
    --ckpt "$CKPT" --key "$KEY" --start "$START" --end "$END" $FIXED \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
