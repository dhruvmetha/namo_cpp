#!/bin/bash
# SEARCH-TRACE eval: the canonical search, recorded push by push for the viz pages.
#
# Same search as the deploy gate (hmax 2, combine=q, discount off, dedupe+jam on) but with
# --trace-out, so every episode also writes what the ranker TRIED, in what order, and the geometry
# plus region decomposition of the state each push reached. That is what the replay and gallery
# pages draw; the eval jsons it writes alongside are a by-product, not a new result.
#
# NOT the wall-clock protocol: tasks pack onto shared nodes here, so the times in these rows are
# NOT comparable with the eval_walltime campaign. Use it for pictures, never for seconds.
#
# Usage (one array per leg):
#   CKPT=$AMAREL_CKPT_DIR/HY5U_s1.ckpt \
#   OUT=$NAMO_SCRATCH/viz/hy5u_trace NSH=50 PUSH=2 \
#     sbatch --array=0-49 scripts/slurm/eval_trace.slurm
#SBATCH --job-name=evaltrace
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint>}"
: "${OUT:?set OUT=<output root>}"
SIM_BUDGET=${SIM_BUDGET:-900}
NSH=${NSH:-50}
PRIOR=${PRIOR:-model}
SEED_BASE=${SEED_BASE:-7000}
PUSH=${PUSH:-2}

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
SAGE=${SAGE_REPO:?set SAGE_REPO -- must match the code the ckpt was trained with}
BIND=${NAMO_BINDINGS:-$REPO/build_python}
cd "$REPO" || { echo "FATAL: REPO=$REPO is not a directory"; exit 1; }
source env.amarel.sh || { echo "FATAL: could not source env.amarel.sh"; exit 1; }
export SAGE_REPO="$SAGE"
export PYTHONPATH="$BIND:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE"
# One core per task, so pin every thread pool: an unpinned OpenBLAS spawns a thread per core of the
# whole node and the tasks trample each other.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=$(command -v python3)

ID=$SLURM_ARRAY_TASK_ID
if [ "$PUSH" = "1" ]; then
  LEG=1push_hmax2
  KEY=$("$PY" -m namo.eval_sets onepush_manifest)
  NEP=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
  KEYARG=(--key "$KEY")
else
  LEG=2push
  NEP=$("$PY" -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
  KEYARG=()
fi
mkdir -p logs "$OUT/$LEG" "$OUT/trace/$LEG"
S=$((ID * NEP / NSH)); EN=$(((ID + 1) * NEP / NSH))
echo "TRACE start $(date) host=$(hostname) leg=$LEG ID=$ID scenes[$S,$EN) budget=$SIM_BUDGET"
[ "$S" -ge "$EN" ] && { echo "empty shard"; exit 0; }

"$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget "$SIM_BUDGET" \
  --prior "$PRIOR" --agg mean5 --combine q --discount off --seed-base "$SEED_BASE" \
  "${KEYARG[@]}" --start "$S" --end "$EN" \
  --trace-out "$OUT/trace/$LEG" \
  --out "$OUT/$LEG/shard_${ID}.json" --leaf-out "$OUT/$LEG/shard_${ID}.jsonl"
echo "TRACE done $(date) ID=$ID traces=$(ls "$OUT/trace/$LEG" | wc -l)"
