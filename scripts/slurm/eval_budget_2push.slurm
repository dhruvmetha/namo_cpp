#!/bin/bash
# 2-push canonical eval at an ARBITRARY simulator budget (Amarel, CPU-only).
# [EXP-2026-08-09 § deep-budget, USER 2026-08-13: "increase the budget to 4000 sims"]
#
# Why a separate script from aquaman_eval_amarel.slurm: that one hardcodes --sim-budget 900 and
# --prior model, and always runs the 1push leg too. This one is 2push-only and parameterises the
# two knobs that matter here, so the model and the random baseline run under IDENTICAL settings
# (same key, same raw scale, same seeds) — the previous random baseline did NOT (it used
# pure2push_search_eval.json + the sigmoid scale, while model evals use the v3 key + raw), which
# is precisely why cross-era comparisons needed aquaman_agg_common.py.
#
# Usage (one array per model-seed):
#   CKPT=<ckpt> OUT=<dir> SIM_BUDGET=4000 N2SH=250 PRIOR=model RAW=1 \
#     sbatch --array=0-249 --time=6:00:00 scripts/slurm/eval_budget_2push.slurm
#   PRIOR=uniform SEED_BASE=7000 ... (CKPT still required: it is loaded but never scored)
#
# SIZING: a task's wall time is (episodes per task) x (worst-case budget x ~0.23 s/sim).
# At budget 4000 an unsolved episode is ~15 min, so 4 episodes per task worst-cases at ~62 min.
# Amarel's `main` QOS caps submitted jobs at 500 — submit in waves.
#SBATCH --job-name=eval_budget
#SBATCH --partition=main-redhat
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint path (loaded even when PRIOR=uniform)>}"
: "${OUT:?set OUT=<evaluation output root>}"
SIM_BUDGET=${SIM_BUDGET:-900}
N2SH=${N2SH:-250}
PRIOR=${PRIOR:-model}
SEED_BASE=${SEED_BASE:-7000}
RAW=${RAW:-1}

# SLURM copies this script out of the repo, so never derive the checkout from BASH_SOURCE.
REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
SAGE=${SAGE_REPO:?set SAGE_REPO -- it MUST match the code version the ckpt was trained with}
BIND=${NAMO_BINDINGS:-$REPO/build_python}      # env.amarel.sh does NOT export these two
cd "$REPO" || { echo "FATAL: REPO=$REPO is not a directory"; exit 1; }
source env.amarel.sh || { echo "FATAL: could not source env.amarel.sh from $REPO"; exit 1; }
export SAGE_REPO="$SAGE"
export PYTHONPATH="$BIND:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=$(command -v python3)
ID=$SLURM_ARRAY_TASK_ID
mkdir -p logs "$OUT/2push"

N2=$("$PY" -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
S=$((ID * N2 / N2SH)); EN=$(((ID + 1) * N2 / N2SH))
echo "EVAL start $(date) host=$(hostname) ID=$ID episodes[$S,$EN) budget=$SIM_BUDGET prior=$PRIOR seed=$SEED_BASE"
[ "$S" -ge "$EN" ] && { echo "empty shard, nothing to do"; exit 0; }

EXTRA=()
[ "$RAW" = "1" ] && EXTRA+=(--raw)

"$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget "$SIM_BUDGET" \
  --prior "$PRIOR" --agg mean5 --combine q --discount off --seed-base "$SEED_BASE" \
  --start "$S" --end "$EN" "${EXTRA[@]}" \
  --out "$OUT/2push/shard_${ID}.json" --leaf-out "$OUT/2push/shard_${ID}.jsonl"
echo "EVAL done $(date) ID=$ID"
