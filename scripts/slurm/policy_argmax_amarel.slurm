#!/bin/bash
# POLICY MODE (zero search) on AMAREL, CPU-only: repeat[ argmax push -> simulate -> opened? ] up to
# KMAX times, object-constrained, region criterion. This is scripts/sandbox/eval_reactive_argmax.py,
# the same ranker and the same candidate pool the canonical best-first search uses, with the queue
# removed -- so the numbers sit directly beside the registered search rows on the SAME v3 population.
#
# ONE run at KMAX=10 answers hmax=5 AND hmax=10: each leaf records opened_at, so cumulative
# open@1..open@10 all come from the same rollout. Do NOT launch a second job for hmax=5.
#
# Layout: $OUT/{1push_policy,2push_policy}/shard_N.json(+.jsonl).
#
# Required env: CKPT, OUT. Optional: PRIOR (q=model argmax | uniform=random floor), SEED, KMAX,
# N1SH/N2SH (shard counts; array size must be N1SH+N2SH).
#   HY5U arm:   CKPT=.../HY5U_s1.ckpt OUT=... PRIOR=q       SEED=1
#   random arm: CKPT=<any loadable ckpt> OUT=... PRIOR=uniform SEED=7000   (uniform makes NO model call)
#
# ⛔ eval_reactive_argmax.py's --key DEFAULT is the v1 testset. Both legs pass --key explicitly.
# SAGE MUST MATCH THE CKPT (round-1 incident: 204 tasks died on an unknown action_motion_dim).
#SBATCH --job-name=pol_argmax
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint path>}"
: "${OUT:?set OUT=<evaluation output root>}"
PRIOR=${PRIOR:-q}
SEED=${SEED:-1}
KMAX=${KMAX:-10}

# ⛔ ${BASH_SOURCE[0]} IS NOT THE REPO UNDER sbatch -- SLURM spools the script to /var/lib/slurm.
# SLURM_SUBMIT_DIR is the correct fallback (see aquaman_eval_amarel.slurm for the full post-mortem).
REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
SAGE=${SAGE_REPO:?set SAGE_REPO -- it MUST match the code version the ckpt was trained with}
BIND=${NAMO_BINDINGS:-$REPO/build_python}
# STALE-BINDINGS GUARD: the .so is what actually simulates; a .so older than the newest C++ commit
# silently changes the physics under identical source.
if command -v git >/dev/null 2>&1 && [ -d "$REPO/.git" ]; then
  _so=$(ls "$BIND"/namo_rl*.so 2>/dev/null | head -1)
  if [ -n "$_so" ]; then
    _so_t=$(stat -c %Y "$_so" 2>/dev/null)
    _src_t=$(cd "$REPO" && git log -1 --format=%ct -- src include python/namo/cpp_bindings 2>/dev/null)
    if [ -n "$_so_t" ] && [ -n "$_src_t" ] && [ "$_so_t" -lt "$_src_t" ]; then
      echo "⚠⚠ STALE BINDINGS: $_so built $(date -d @$_so_t '+%F %T'), C++ changed $(date -d @$_src_t '+%F %T')."
    fi
  fi
fi
N1SH=${N1SH:-24}
N2SH=${N2SH:-24}
cd "$REPO" || { echo "FATAL: REPO=$REPO is not a directory"; exit 1; }
source env.amarel.sh || { echo "FATAL: could not source env.amarel.sh from $REPO"; exit 1; }
export SAGE_REPO="$SAGE"
export PYTHONPATH="$BIND:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=$(command -v python3)
ID=$SLURM_ARRAY_TASK_ID
mkdir -p logs

echo "POLICY start $(date) host=$(hostname) CKPT=$CKPT PRIOR=$PRIOR SEED=$SEED KMAX=$KMAX ID=$ID"
if [ "$ID" -lt "$N1SH" ]; then
  KEY=$("$PY" -m namo.eval_sets onepush_manifest)
  LEG=1push_policy; SH=$ID; NSH=$N1SH
else
  KEY=$("$PY" -m namo.eval_sets pure2push_manifest)
  LEG=2push_policy; SH=$((ID - N1SH)); NSH=$N2SH
fi
NX=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
S=$((SH * NX / NSH)); EN=$(((SH + 1) * NX / NSH))
mkdir -p "$OUT/$LEG"
"$PY" scripts/sandbox/eval_reactive_argmax.py --ckpt "$CKPT" --key "$KEY" \
  --max-pushes "$KMAX" --prior "$PRIOR" --seed "$SEED" --start "$S" --end "$EN" \
  --out "$OUT/$LEG/shard_${SH}.json" --leaf-out "$OUT/$LEG/shard_${SH}.jsonl"
echo "POLICY done $(date) ID=$ID leg=$LEG shard=$SH"
