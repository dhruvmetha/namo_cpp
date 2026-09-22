#!/bin/bash
# Train MORE's push network on CS iLab SLURM, one seed per task.
#
# This job is DATALOADER-bound, not GPU-bound. Each sample reads a 5x224x224 picture and
# rotates it, and roughly 97 samples share each state, so the loader needs cores more
# than the GPU needs FLOPs. Measured on a quiet arrakis: 587 samples/s with 8 workers
# against 753 the model alone can do. On a contended box it fell to 111. So prefer a
# high-core node (rlab7 has 256, ilab3/rlab1-4 have 96) and give the loader workers.
#
# The `unlimited` partition REJECTS --cpus-per-task, so worker count is set by --workers
# on the command line rather than by asking SLURM for cores.
#
# Usage (from the repo root, any CS box, shared filesystem so no copying):
#   H5=$NAMO_SCRATCH/more_pilot_20260921/more_pilot_chunked.h5 \
#   OUT=$NAMO_SCRATCH/more_pilot_20260921 \
#     sbatch --array=1-3 scripts/slurm/train_more.sbatch
#
#SBATCH --job-name=more-train
#SBATCH --partition=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${H5:?set H5 to the training corpus}"
: "${OUT:?set OUT to the run root}"
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
source env.ilab.sh
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

SEED=${SLURM_ARRAY_TASK_ID:-1}
WORKERS=${WORKERS:-12}
# Leave the pools alone here: unlike the collection arrays this job WANTS threads for
# decompression and the rotation, and it is the only thing on its GPU.
echo "MORE-TRAIN start $(date) host=$(hostname) seed=$SEED gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$NAMO_PYTHON" scripts/rl_loop/train_more.py \
  --h5 "$H5" --out "$OUT/model_s$SEED" \
  --epochs "${EPOCHS:-2}" --batch "${BATCH:-32}" --workers "$WORKERS" --seed "$SEED" \
  ${UNREACHABLE_CAP:+--unreachable-cap "$UNREACHABLE_CAP"} \
  ${DROP_UNSAMPLED:+--drop-unsampled}

echo "MORE-TRAIN done $(date) seed=$SEED"

# Held-out check immediately, so a diverged run is caught here rather than after an
# evaluation is already queued behind it. A falling loss is not evidence of learning.
LAST=$(ls "$OUT/model_s$SEED"/more_epoch*.pt | tail -1)
"$NAMO_PYTHON" scripts/rl_loop/verify_more_net.py \
  --h5 "$H5" --ckpt "$LAST" --states 300 --seed 0
