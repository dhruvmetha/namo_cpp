#!/bin/bash
# Aggregate one complete canonical best-first arm after its evaluation array succeeds.
# Required env: EVAL_ROOT, SAGE_REPO, NAMO_BINDINGS. Optional: NAMO_REPO.
#SBATCH --job-name=aq0_agg
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

: "${EVAL_ROOT:?set EVAL_ROOT to the arm/seed evaluation directory}"
: "${SAGE_REPO:?set SAGE_REPO to the matching Sage checkout}"
: "${NAMO_BINDINGS:?set NAMO_BINDINGS to the canonical Amarel build_python directory}"

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?submit from the NAMO checkout or set NAMO_REPO}}
cd "$REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_BINDINGS:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${NAMO_PYTHON:-python}

"$PY" scripts/rl_loop/agg_search_eval.py \
  --eval-root "$EVAL_ROOT" \
  --onepush-dir "$EVAL_ROOT/1push_hmax2" \
  --twopush-dir "$EVAL_ROOT/2push" \
  --out "$EVAL_ROOT/aggregate.json" \
  --require-hmax 2 \
  --require-dedupe-noop \
  --require-prune-jam-depth

echo "AGGREGATE DONE $(date) root=$EVAL_ROOT"
