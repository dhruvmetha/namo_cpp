#!/bin/bash
# WALL-CLOCK eval (1push or 2push) — the timed protocol. [EXP-2026-08-09 § wall-clock, USER 2026-08-13]
#
# Why this differs from eval_budget_2push.slurm: wall-time is only meaningful if nothing else
# shares the node, so every task takes a WHOLE node (--exclusive) of one fixed CPU generation
# (--constraint=icelake) and runs single-threaded. Sims are the cross-box substrate; wall-times
# compare ONLY within one campaign on identical hardware [CLAUDE.md]. Timing is accumulated inside
# eval_bestfirst.solve_scene itself (perf_counter ~50 ns against a ~1 s sim), so the TIMED search
# is literally the canonical search — no separate timing fork (scripts/sandbox/time_bestfirst.py
# is RETIRED; its fork predated dedupe/jam-prune).
#
# Per-episode rows carry t_wall / t_sim / t_score / n_score, so the model's scoring overhead is
# separable from simulator time — the number that decides whether a sim-count win survives in
# seconds. Budget 900 to stay comparable with the registered walltime-nodiscount-hmax2-v1 rows.
#
# Usage (one array per arm):
#   CKPT=<ckpt> OUT=<dir> PRIOR=model|uniform|geometric SEED_BASE=7000 NSH=20 PUSH=2 \
#     sbatch --array=0-19 --constraint=icelake scripts/slurm/eval_walltime.slurm
# PUSH=1 evaluates the 1-push population (still hmax=2 search) into $OUT/1push_hmax2/.
# BOTH legs are pinned to the SAME constraint here so every arm of this campaign shares one CPU
# generation; never compare a 1push time against a 2push time regardless.
#SBATCH --job-name=walltime
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${OUT:?set OUT=<output root>}"
SIM_BUDGET=${SIM_BUDGET:-900}
NSH=${NSH:-10}
PRIOR=${PRIOR:-model}
CKPT=${CKPT:-}
[ "$PRIOR" != "model" ] || : "${CKPT:?set CKPT=<checkpoint> when PRIOR=model}"
SEED_BASE=${SEED_BASE:-7000}
PUSH=${PUSH:-2}

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
SAGE=${SAGE_REPO:?set SAGE_REPO -- must match the code the ckpt was trained with}
BIND=${NAMO_BINDINGS:-$REPO/build_python}
cd "$REPO" || { echo "FATAL: REPO=$REPO is not a directory"; exit 1; }
source env.amarel.sh || { echo "FATAL: could not source env.amarel.sh"; exit 1; }
export SAGE_REPO="$SAGE"
export PYTHONPATH="$BIND:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE"
# single-threaded is part of the PROTOCOL here, not just hygiene: a multi-threaded BLAS would make
# the model's scoring time depend on how many cores happened to be free.
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
mkdir -p logs "$OUT/$LEG"
S=$((ID * NEP / NSH)); EN=$(((ID + 1) * NEP / NSH))
echo "WALLTIME start $(date) host=$(hostname) cpu=$(lscpu | grep 'Model name' | head -1 | cut -d: -f2 | xargs) leg=$LEG ID=$ID episodes[$S,$EN) prior=$PRIOR seed=$SEED_BASE budget=$SIM_BUDGET"
[ "$S" -ge "$EN" ] && { echo "empty shard"; exit 0; }

"$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget "$SIM_BUDGET" \
  --prior "$PRIOR" --agg mean5 --combine q --discount off --seed-base "$SEED_BASE" \
  "${KEYARG[@]}" --start "$S" --end "$EN" \
  --out "$OUT/$LEG/shard_${ID}.json" --leaf-out "$OUT/$LEG/shard_${ID}.jsonl"
echo "WALLTIME done $(date) ID=$ID"
