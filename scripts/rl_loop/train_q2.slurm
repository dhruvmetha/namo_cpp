#!/bin/bash
# Train Q2 (pooled depth-<=2 dense VALUE field, hl_gauss regression, REACHABLE-ONLY) on the pooled H5.
# iLab `unlimited`, pinned to one A100 node (rlab1). unlimited does NOT cgroup-slice CPUs, so one
# job/node; submit --gres=gpu:1 only (NO --cpus-per-task — unlimited rejects it). ctx is stored
# UNCOMPRESSED float16 in the pooled H5, so the dataloader is not decompression-bound (8 workers OK).
#   Env in (required): H5 (the POOLED rung1+rung2 H5 from pool_q2_h5.py).
#   Optional: OUT, EPOCHS(40), BATCH(256), LR(3e-4), WORKERS(8), PATIENCE(8).
#   Submit: H5=/path/to/q2_pool_full.h5 sbatch -w rlab1 --gres=gpu:1 scripts/rl_loop/train_q2.slurm
#SBATCH --job-name=q2_pool
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/exit/logs/q2-%j.out
#SBATCH --error=/common/users/dm1487/scratch_namo/exit/logs/q2-%j.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
cd "$MAIN"
source env.ilab.sh >/dev/null 2>&1
export PYTHONPATH="$MAIN/build_python:$MAIN/python:$MAIN/scripts:$MAIN/scripts/sandbox:$MAIN/scripts/pipeline:${SAGE_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/tmp                 # spawn-worker tempdirs off NFS (avoids .nfs busy cleanup noise)
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${H5:?set H5=/path/to/q2_pool_full.h5}"
OUT=${OUT:-/common/users/dm1487/scratch_namo/exit/q2_full}
EPOCHS=${EPOCHS:-40}; BATCH=${BATCH:-256}; LR=${LR:-3e-4}; WORKERS=${WORKERS:-8}; PATIENCE=${PATIENCE:-8}
mkdir -p /common/users/dm1487/scratch_namo/exit/logs "$OUT"
"$PY" -u scripts/rl_loop/train_q2.py \
  --h5 "$H5" --out-dir "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" --num-workers "$WORKERS" \
  --warmup-steps 200 --decay-steps 100000 \
  --train-split 0.9 --split-seed 0 --patience "$PATIENCE" --seed 0
# best ckpt -> $OUT/checkpoints/epochNNN-val_lossX.ckpt  (also last.ckpt); diagnostics printed at the tail.
