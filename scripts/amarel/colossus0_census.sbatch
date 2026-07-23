#!/bin/bash
#SBATCH --job-name=c0census
#SBATCH --account=general
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

: "${NAMO_REPO:?}"
: "${RESULT_ROOT:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO/scripts:$NAMO_REPO/scripts/pipeline:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

out=${CENSUS_OUT:-$RESULT_ROOT/census_225k.json}
pure_out=${PURE_OUT:-$RESULT_ROOT/pure2_225k.json}
expected_results=${EXPECTED_RESULTS:-225000}
actual_results=$(find "$RESULT_ROOT/collect" -type f -name '*_results.pkl' | wc -l)
if [[ "$actual_results" -ne "$expected_results" ]]; then
  echo "expected $expected_results result PKLs, found $actual_results" >&2
  exit 1
fi
if [[ -e "$out" || -e "$pure_out" ]]; then
  echo "refusing to overwrite census output" >&2
  exit 1
fi
"$NAMO_PYTHON" scripts/pipeline/build_2push_validset.py \
  --pkls-root "$RESULT_ROOT/collect" \
  --out "$out" \
  --pure-out "$pure_out" \
  --workers "$SLURM_CPUS_PER_TASK"
