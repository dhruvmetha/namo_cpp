#!/bin/bash
# Phase-0 grey-zone k-sweep (EXP-2026-07-06-rl-only-self-imitation). iLab `unlimited`, sharded by xml-index. CPU-only.
# Env in: CKPT, OUT_DIR (required); H(2), SHARD(34). Submit once per ckpt-seed.
#SBATCH --job-name=p0ksw
#SBATCH --partition=unlimited
#SBATCH --array=0-29
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/eval/phase0_ksweep/logs/ksw-%A_%a.out
#SBATCH --error=/common/users/dm1487/scratch_namo/eval/phase0_ksweep/logs/ksw-%A_%a.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
WT=${REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$MAIN"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export SAGE_REPO=/common/home/dm1487/robotics_research/ktamp/sage_learning
export PYTHONPATH="$MAIN/build_python:$MAIN/python:$MAIN/scripts:$MAIN/scripts/sandbox:$MAIN/scripts/pipeline:$SAGE_REPO"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENCV_FOR_THREADS_NUM=1
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-34}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
mkdir -p "$OUT_DIR" /common/users/dm1487/scratch_namo/eval/phase0_ksweep/logs
"$PY" -u "$WT/scripts/rl_loop/phase0_ksweep.py" \
    --ckpt "$CKPT" --h "${H:-2}" --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" \
    --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
