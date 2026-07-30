#!/bin/bash
# Build the targeted fill H5 and append only requested episode trees to the base GT.
# Required env: ALIGNMENT, COLLECT_DIR, BASE_H5, WORK_DIR, OUT_H5.
#SBATCH --job-name=build2p-gt
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build2p-gt-%j.out
#SBATCH --error=logs/build2p-gt-%j.err
set -euo pipefail
cd "${NAMO_REPO:?}"
source scripts/amarel/activate.sh >/dev/null 2>&1
cd "$NAMO_REPO"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1

: "${ALIGNMENT:?}"
: "${COLLECT_DIR:?}"
: "${BASE_H5:?}"
: "${WORK_DIR:?}"
: "${OUT_H5:?}"
mkdir -p "$WORK_DIR"

"$NAMO_PYTHON" scripts/pipeline/build_rung2_h5.py \
  --pkl-glob "$COLLECT_DIR/**/*_results.pkl" \
  --out "$WORK_DIR/targeted_fill.h5" --gamma 0.9
"$NAMO_PYTHON" scripts/pipeline/merge_targeted_rung2_gt.py \
  --base "$BASE_H5" --fill-glob "$WORK_DIR/targeted_fill.h5" \
  --alignment "$ALIGNMENT" --out "$OUT_H5"
echo "BUILD2P complete @ $(date -Is)"
