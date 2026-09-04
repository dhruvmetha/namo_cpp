#!/bin/bash
# Aquaman canonical eval on AMAREL (CPU-only). Protocol copied verbatim from the round-0 arm-A/B
# shard headers so results stay directly comparable:
#   hmax=2 sim_budget=900 prior=model agg=mean5 combine=q discount=off
#   w0_mode=one raw-score scale dedupe_noop=true prune_jam_depth=true
# Layout: $OUT/{1push_hmax2,2push}/shard_N.json(+.jsonl).
#
# SHARDING (N1SH 1push leaf shards, N2SH 2push leaf shards):
#   default  N1SH=40 N2SH=32  --array=0-71    <- reproduces every pre-2026-08-08 eval byte-identically
#   fine     N1SH=128 N2SH=128 --array=0-255  <- ~4x less wall
#   bundled  WORKERS_PER_TASK=21 N1SH=378 N2SH=378 --array=0-35
#            <- 756 evaluator processes behind 36 scheduler jobs, with every requested CPU used
# Why fine helps: wall time is set by the SLOWEST shard, not the average. A 2push shard holding a
# few hard episodes at the full 900-sim budget (~3.5 min each) strands everyone else. Splitting the
# same work finer shortens the straggler. FLOOR: one 900-sim episode is ~3.5 min and cannot be
# split, so no sharding takes the eval below ~4 min.
# NOTE: 256 tasks x several models trips Amarel's submit limit -- submit in waves (see eval_wave2.sh).
#
# SAGE MUST MATCH THE CKPT (round-1 incident: 204 tasks died on an unknown action_motion_dim).
#SBATCH --job-name=aq0_eval
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint path>}"
: "${OUT:?set OUT=<evaluation output root>}"

# Box-agnostic: the checkout is $NAMO_REPO, else the directory the job was SUBMITTED from, else
# where this file lives. Sage and the built bindings are siblings of it (a fresh worktree has no
# build_python of its own -- point NAMO_BINDINGS at the checkout that does). Override any of the
# three by env; nothing here is a literal box path.
#
# ⛔ ${BASH_SOURCE[0]} IS NOT THE REPO UNDER sbatch. SLURM copies the batch script to
# /var/lib/slurm/slurmd/job<N>/slurm_script before running it, so deriving the checkout from it
# yields REPO=/var/lib/slurm -- the cd succeeds, `source env.amarel.sh` fails, and set -e kills
# the task with the error swallowed by the redirect below: 1 s runtime, 0-byte log, no clue.
# That cost ~30 min on 2026-08-08 and 432 dead tasks; it had only ever worked because callers
# happened to have NAMO_REPO exported and sbatch --export=ALL carried it in. SLURM_SUBMIT_DIR is
# the correct fallback -- it survives the spooling.
REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
SAGE=${SAGE_REPO:?set SAGE_REPO -- it MUST match the code version the ckpt was trained with}
BIND=${NAMO_BINDINGS:-$REPO/build_python}
# ⛔ STALE-BINDINGS GUARD. The .so is what actually simulates; the checkout only decides which
# python runs. On 2026-08-09 a smoke shard scored 89 sims on Amarel and 98 on CS with identical
# source -- the gap was entirely a .so built 2026-07-29, predating 5daaed5 ("set_full_state must
# restore ALL state"), which leaked ctrl/qacc_warmstart across restores and made a replayed board
# not the same board twice. Every registered number came from that binary. Silent, and worth ~10%
# of sims. So: if the bindings are older than the newest C++ commit, say so loudly.
if command -v git >/dev/null 2>&1 && [ -d "$REPO/.git" ]; then
  _so=$(ls "$BIND"/namo_rl*.so 2>/dev/null | head -1)
  if [ -n "$_so" ]; then
    _so_t=$(stat -c %Y "$_so" 2>/dev/null)
    _src_t=$(cd "$REPO" && git log -1 --format=%ct -- src include python/namo/cpp_bindings 2>/dev/null)
    if [ -n "$_so_t" ] && [ -n "$_src_t" ] && [ "$_so_t" -lt "$_src_t" ]; then
      echo "⚠⚠ STALE BINDINGS: $_so built $(date -d @$_so_t '+%F %T'), but C++ changed $(date -d @$_src_t '+%F %T')."
      echo "⚠⚠ Results will NOT match a current build. Rebuild with ./build_python_bindings.sh, or set NAMO_BINDINGS deliberately."
    fi
  fi
fi
N1SH=${N1SH:-40}
N2SH=${N2SH:-32}
WORKERS_PER_TASK=${WORKERS_PER_TASK:-1}
ONEPUSH_LIMIT=${ONEPUSH_LIMIT:-0}
TWOPUSH_LIMIT=${TWOPUSH_LIMIT:-0}
REFUSE_OVERWRITE=${REFUSE_OVERWRITE:-0}
# Protocol knobs, defaulted to the registered canonical values -- an existing caller that sets none
# of them reproduces its old run byte-identically. Overridden only to answer a different question,
# e.g. HMAX=10 SIM_BUDGET=10 for a depth-matched comparison against a policy rollout.
HMAX=${HMAX:-2}
SIM_BUDGET=${SIM_BUDGET:-900}
PRIOR=${PRIOR:-model}
SEED_BASE=${SEED_BASE:-7000}
cd "$REPO" || { echo "FATAL: REPO=$REPO is not a directory"; exit 1; }
# NOT silenced: a failure here used to be invisible. env.amarel.sh is chatty on success, so the
# noise is the price of ever seeing the failure again.
source env.amarel.sh || { echo "FATAL: could not source env.amarel.sh from $REPO"; exit 1; }
export SAGE_REPO="$SAGE"
export PYTHONPATH="$BIND:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$SAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=$(command -v python3)
ID=$SLURM_ARRAY_TASK_ID
mkdir -p logs

[ "$WORKERS_PER_TASK" -ge 1 ] || { echo "WORKERS_PER_TASK must be positive" >&2; exit 2; }
KEY=$("$PY" -m namo.eval_sets onepush_manifest)
N1=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
N2=$("$PY" -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
if [ "$ONEPUSH_LIMIT" -gt 0 ] && [ "$ONEPUSH_LIMIT" -lt "$N1" ]; then N1=$ONEPUSH_LIMIT; fi
if [ "$TWOPUSH_LIMIT" -gt 0 ] && [ "$TWOPUSH_LIMIT" -lt "$N2" ]; then N2=$TWOPUSH_LIMIT; fi
mkdir -p "$OUT/1push_hmax2" "$OUT/2push"

# A bundled task loads its checkpoint from node-local storage, avoiding up to 21 simultaneous reads
# of the same 40-50 MB file from the shared filesystem.
RUN_CKPT=$CKPT
LOCAL_CKPT=
if [ "$WORKERS_PER_TASK" -gt 1 ]; then
  LOCAL_CKPT=/tmp/namo_eval_${SLURM_JOB_ID}.ckpt
  cp "$CKPT" "$LOCAL_CKPT"
  RUN_CKPT=$LOCAL_CKPT
  EVAL_PARENT_PID=$BASHPID
  cleanup_local_ckpt() {
    [ "$BASHPID" != "$EVAL_PARENT_PID" ] || rm -f "$LOCAL_CKPT"
  }
  trap cleanup_local_ckpt EXIT
fi

run_leaf() {
  local GID=$1 LEG SH NSH N S EN OUT_JSON OUT_JSONL WORKER_LOG
  if [ "$GID" -lt "$N1SH" ]; then
    LEG=1push_hmax2; SH=$GID; NSH=$N1SH; N=$N1
  else
    LEG=2push; SH=$((GID - N1SH)); NSH=$N2SH; N=$N2
  fi
  S=$((SH * N / NSH)); EN=$(((SH + 1) * N / NSH))
  [ "$S" -lt "$EN" ] || return 0
  OUT_JSON=$OUT/$LEG/shard_${SH}.json
  OUT_JSONL=$OUT/$LEG/shard_${SH}.jsonl
  WORKER_LOG=$OUT/$LEG/shard_${SH}.log
  if [ "$REFUSE_OVERWRITE" = 1 ] && { [ -e "$OUT_JSON" ] || [ -e "$OUT_JSONL" ]; }; then
    echo "refusing to overwrite leaf shard $OUT/$LEG/shard_${SH}" >&2
    return 1
  fi
  if [ "$LEG" = 1push_hmax2 ]; then
    /usr/bin/time -v "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$RUN_CKPT" --key "$KEY" \
      --hmax "$HMAX" --sim-budget "$SIM_BUDGET" --prior "$PRIOR" --seed-base "$SEED_BASE" \
      --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
      --out "$OUT_JSON" --leaf-out "$OUT_JSONL" > "$WORKER_LOG" 2>&1
  else
    /usr/bin/time -v "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$RUN_CKPT" \
      --hmax "$HMAX" --sim-budget "$SIM_BUDGET" --prior "$PRIOR" --seed-base "$SEED_BASE" \
      --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
      --out "$OUT_JSON" --leaf-out "$OUT_JSONL" > "$WORKER_LOG" 2>&1
  fi
}

FIRST=$((ID * WORKERS_PER_TASK))
LAST=$((FIRST + WORKERS_PER_TASK))
TOTAL=$((N1SH + N2SH))
if [ "$LAST" -gt "$TOTAL" ]; then LAST=$TOTAL; fi
[ "$FIRST" -lt "$TOTAL" ] || { echo "empty bundle ID=$ID"; exit 0; }
echo "AQ0 EVAL start $(date) host=$(hostname) CKPT=$CKPT ID=$ID leaves=$FIRST..$((LAST - 1)) N1SH=$N1SH N2SH=$N2SH workers=$WORKERS_PER_TASK"
pids=()
for ((GID=FIRST; GID<LAST; GID++)); do
  run_leaf "$GID" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
[ "$failed" -eq 0 ] || { echo "one or more leaf evaluators failed" >&2; exit 1; }
echo "AQ0 EVAL done $(date) ID=$ID leaves=$FIRST..$((LAST - 1))"
