#!/bin/bash
#SBATCH --job-name=maskgen_dec2_2push_srcsplit
#SBATCH --output=/common/users/tdn39/Robotics/Mujoco/namo_cpp/logs/maskgen_dec2_2push_srcsplit_%j.out
#SBATCH --error=/common/users/tdn39/Robotics/Mujoco/namo_cpp/logs/maskgen_dec2_2push_srcsplit_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G

set -euo pipefail

# Raw 2-push training data (recursive; contains both 1- and 2-step solutions).
INPUT_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/raw_data/dec2/aug9_envs/2_push_train"

# Mask outputs (NPZ). With --split-difficulty enabled, this directory will contain
# easy/medium/hard subfolders, then source_subdir (e.g., 1/ or 2/), then task_id/episode_id.npz.
# NOTE: This is separate from the older `.../npz` output to avoid mixing old (overwritten) data.
OUTPUT_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/images/dec2/aug9_envs/2_push_train/npz_srcsplit"

# Consolidated HDF5 for training (single-step canonicalized deltas).
H5_OUTPUT_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/h5_files/se2/dec2/aug9_envs/2_push_train_srcsplit"
H5_OUTPUT_FILE="${H5_OUTPUT_DIR}/training_data.h5"
STATS_OUTPUT_FILE="${H5_OUTPUT_DIR}/stats_mean_std.json"
STATS_MAXABS_FILE="${H5_OUTPUT_DIR}/stats_max_abs.json"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${H5_OUTPUT_DIR}"

# NOTE: With `set -u`, the venv activate script can error if PYTHONPATH is unset.
PYTHONPATH="${PYTHONPATH:-}"
source /common/home/tdn39/.virtualenvs/mujoco/bin/activate

NAMO_ROOT="/common/users/tdn39/Robotics/Mujoco/namo_cpp"
SAGE_ROOT="/common/users/tdn39/Robotics/Mujoco/sage_learning"

mkdir -p "${NAMO_ROOT}/logs"

export PYTHONPATH="${SAGE_ROOT}:${NAMO_ROOT}/python:${NAMO_ROOT}/build_python:${PYTHONPATH:-}"

# Avoid OpenBLAS/MKL oversubscription with many workers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${NAMO_ROOT}/python"

python -m namo.visualization.mask_generation.batch_collection \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --local-only \
  --pattern "**/*_results.pkl" \
  --workers 50 \
  --filter-overlaps \
  --split-difficulty

python /common/users/tdn39/Robotics/Mujoco/sage_learning/scripts/convert_to_hdf5.py \
  "${OUTPUT_DIR}" \
  "${H5_OUTPUT_FILE}" \
  --minimal \
  --strict

python /common/users/tdn39/Robotics/Mujoco/sage_learning/scripts/compute_dataset_stats.py \
  --h5_file "${H5_OUTPUT_FILE}" \
  --output "${STATS_OUTPUT_FILE}" \
  --mode mean_std

python /common/users/tdn39/Robotics/Mujoco/sage_learning/scripts/compute_dataset_stats.py \
  --h5_file "${H5_OUTPUT_FILE}" \
  --output "${STATS_MAXABS_FILE}" \
  --mode max_abs

echo "Done."
