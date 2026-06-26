#!/bin/bash
# Build the full bootstrap-SETUP H5 from the exhaustive training labels (cheap: s0 renders only, no sims).
# Env: VSUMMARY(density|depth), OUT_H5, CKPT(for renderer). Single task over all episodes.
#SBATCH --job-name=bootbuild
#SBATCH --partition=main,main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/dm1487/logs/bootbuild-%j.out
#SBATCH --error=/scratch/dm1487/logs/bootbuild-%j.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${VSUMMARY:?}"; : "${OUT_H5:?}"
CKPT=${CKPT:-/scratch/dm1487/sage_outputs/scorer/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt}
/scratch/dm1487/envs/namo/bin/python -u scripts/sandbox/build_bootstrap_setup.py \
    --ckpt "$CKPT" --vsummary "$VSUMMARY" --start 0 --end 0 --out-h5 "$OUT_H5"
