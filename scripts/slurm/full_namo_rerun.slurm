#!/bin/bash
# Full NAMO rerun after the local rendering-goal fix: one array task per frozen400 scene,
# with HY5U seeds 1-3 (best-first search) and uniform Random seeds 7000-11000 side by side.
# Settings follow Tri-An's reference helper; see scripts/pipeline/run_full_namo_rerun.py.
#
# Time: Tri-An's timed runs of this exact population took at most 78 min per run on an
# exclusive node; 8 runs share this task's node, so allow roughly 3x.
#
# Usage (from the repo root on Amarel):
#   RERUN=$NAMO_SCRATCH/full_namo_rerun_20260914 CKPTS=<dir with HY5U_s{1,2,3}.ckpt> \
#     [EXPECT_SOURCE_SHA256=<code fingerprint of good rows>] sbatch --array=0-399 scripts/slurm/full_namo_rerun.slurm
#SBATCH --job-name=fnamo-rerun
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${RERUN:?set RERUN to the rerun root holding input/full_namo and config/}"
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
source env.amarel.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CKPTS=${CKPTS:?set CKPTS to the directory holding HY5U_s1.ckpt, HY5U_s2.ckpt and HY5U_s3.ckpt}
arms=()
for s in 1 2 3; do arms+=(--arm "HY5U_s${s}:model:7000:${CKPTS}/HY5U_s${s}.ckpt"); done
# RANDOM_SEEDS="" runs HY5U only; unset keeps all five Random seeds.
RANDOM_SEEDS=${RANDOM_SEEDS-7000 8000 9000 10000 11000}
for seed in $RANDOM_SEEDS; do arms+=(--arm "random_s${seed}:uniform:${seed}"); done

"$NAMO_PYTHON" scripts/pipeline/run_full_namo_rerun.py \
  --manifest-dir "$RERUN/input/full_namo" --config "$RERUN/config/namo_config.yaml" \
  --primitives data --out "$RERUN/results" --index "${SLURM_ARRAY_TASK_ID:?}" \
  ${EXPECT_SOURCE_SHA256:+--expect-source-sha256 "$EXPECT_SOURCE_SHA256"} "${arms[@]}"
