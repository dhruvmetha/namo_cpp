#!/bin/bash
# Tri-An's frozen one-keyhole problems: one array task per shard of the (problem, arm) work list.
# See scripts/pipeline/run_one_keyhole_frozen.py for the protocol.
#
# Usage (from the repo root on Amarel):
#   KH=$NAMO_SCRATCH/keyhole600_ablations_20260914 NSHARDS=480 [EXPECT_SOURCE_SHA256=<sha>] \
#     sbatch --array=0-479 --exclude=<nodes> scripts/slurm/one_keyhole_frozen.slurm
#SBATCH --job-name=kh600
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${KH:?set KH to the run root holding input/one_keyhole, config/, arms.json and post_restore_problem_ids.json}"
: "${NSHARDS:?set NSHARDS to the array size}"
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
source env.amarel.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

"$NAMO_PYTHON" scripts/pipeline/run_one_keyhole_frozen.py \
  --problems "$KH/input/one_keyhole" --config "$KH/config/namo_config.yaml" \
  --arms "${ARMS:-$KH/arms.json}" --post-restore-ids "$KH/post_restore_problem_ids.json" \
  --out "${OUT:-$KH/results}" --shard "${SLURM_ARRAY_TASK_ID:?}" --nshards "$NSHARDS" \
  --workers "${SLURM_CPUS_PER_TASK:?}" ${ONLY_INDEX:+$ONLY_INDEX} \
  ${EXPECT_SOURCE_SHA256:+--expect-source-sha256 "$EXPECT_SOURCE_SHA256"}
