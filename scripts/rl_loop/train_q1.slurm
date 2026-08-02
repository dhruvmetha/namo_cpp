#!/bin/bash
# Train Q1 (rung-1 dense opener classifier, REACHABLE-ONLY arm) on the full rung-1 H5.
# iLab `unlimited`, pinned to one A100 node (rlab1). unlimited does NOT cgroup-slice CPUs, so one
# job/node; submit --gres=gpu:1 only (NO --cpus-per-task — unlimited rejects it). ctx is stored
# UNCOMPRESSED in the rung-1 H5, so the dataloader is not decompression-bound (8 workers is plenty).
#   Env in (required): H5. Optional: OUT, EPOCHS(40), BATCH(256), LR(3e-4), WORKERS(8), PATIENCE(8).
#   Submit: H5=/path/to/rung1_full.h5 sbatch -w rlab1 --gres=gpu:1 scripts/rl_loop/train_q1.slurm
#SBATCH --job-name=q1_rung1
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/exit/logs/q1-%j.out
#SBATCH --error=/common/users/dm1487/scratch_namo/exit/logs/q1-%j.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
cd "$MAIN"
source env.ilab.sh >/dev/null 2>&1
export PYTHONPATH="$MAIN/build_python:$MAIN/python:$MAIN/scripts:$MAIN/scripts/sandbox:$MAIN/scripts/pipeline:${SAGE_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/tmp                 # spawn-worker tempdirs off NFS (avoids .nfs busy cleanup noise)
PY="${NAMO_PYTHON:?source env.ilab.sh or set NAMO_PYTHON to an interpreter}"
: "${H5:?set H5=/path/to/rung1_full.h5}"
OUT=${OUT:-/common/users/dm1487/scratch_namo/exit/q1_full}
EPOCHS=${EPOCHS:-40}; BATCH=${BATCH:-256}; LR=${LR:-3e-4}; WORKERS=${WORKERS:-8}; PATIENCE=${PATIENCE:-8}
mkdir -p /common/users/dm1487/scratch_namo/exit/logs "$OUT"
"$PY" -u scripts/rl_loop/train_q1.py \
  --h5 "$H5" --out-dir "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" --num-workers "$WORKERS" \
  --dice-weight 1.0 --pos-weight 1.0 --warmup-steps 200 --decay-steps 100000 \
  --train-split 0.9 --split-seed 0 --patience "$PATIENCE" --seed 0
# best ckpt -> $OUT/checkpoints/epochNNN-val_lossX.ckpt  (also last.ckpt); diagnostics printed at the tail.
