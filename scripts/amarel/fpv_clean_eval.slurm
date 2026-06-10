#!/bin/bash
# Step-0 "mean_top5" first-push-value eval on the CLEAN pure-2-push set (namo_testset_v1).
# Per first-push a1: sim s0->s1, score s1, log first-push-value SCALARS (mean_top5, maxP, ...) + edge1/depth1.
# Graded POST-HOC vs the exhaustive F1' (labels/pure2push.json) by aggregate_fpv_vs_f1prime.py — NOT the
# beam's budget-limited good/dead label. Reuses scripts/sandbox/diag_leaf_s1.py (no fork).
# Pinned settings: collisions-allowed + target-region-goal.
#SBATCH --job-name=fpv-clean
#SBATCH --partition=main-redhat
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dm1487/datasets/namo_testset_v1/logs/fpv-clean-%A_%a.out
#SBATCH --error=/scratch/dm1487/datasets/namo_testset_v1/logs/fpv-clean-%A_%a.err
set -euo pipefail

cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

MANIFEST=/scratch/dm1487/datasets/namo_testset_v1/manifests/pure2push_scenes.txt
OUT=/scratch/dm1487/datasets/namo_testset_v1/eval_fpv
mkdir -p "$OUT"
SHARD=20
START=$(( SLURM_ARRAY_TASK_ID * SHARD ))
END=$(( START + SHARD ))

echo "task ${SLURM_ARRAY_TASK_ID}: scenes [${START}, ${END}) on $(hostname)"
/scratch/dm1487/envs/namo/bin/python -u scripts/sandbox/diag_leaf_s1.py \
  --manifest "$MANIFEST" --start "${START}" --end "${END}" \
  --first-cap 60 --topk 20 \
  --out      "$OUT/fpv_shard${SLURM_ARRAY_TASK_ID}.json" \
  --leaf-out "$OUT/fpv_shard${SLURM_ARRAY_TASK_ID}.jsonl"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
