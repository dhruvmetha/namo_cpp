#!/bin/bash
# Aquaman canonical eval on AMAREL (CPU-only). Protocol copied verbatim from the round-0 arm-A/B
# shard headers so results stay directly comparable:
#   hmax=2 sim_budget=900 prior=model agg=mean5 combine=q discount=off
#   w0_mode=one raw=false dedupe_noop=true prune_jam_depth=true
# Layout: $OUT/{1push_hmax2,2push}/shard_N.json(+.jsonl).
#
# SHARDING (N1SH 1push shards, N2SH 2push shards; array size must be N1SH+N2SH):
#   default  N1SH=40 N2SH=32  --array=0-71    <- reproduces every pre-2026-08-08 eval byte-identically
#   fine     N1SH=128 N2SH=128 --array=0-255  <- ~4x less wall
# Why fine helps: wall time is set by the SLOWEST shard, not the average. A 2push shard holding a
# few hard episodes at the full 900-sim budget (~3.5 min each) strands everyone else. Splitting the
# same work finer shortens the straggler. FLOOR: one 900-sim episode is ~3.5 min and cannot be
# split, so no sharding takes the eval below ~4 min.
# NOTE: 256 tasks x several models trips Amarel's submit limit -- submit in waves (see eval_wave2.sh).
#
# SAGE MUST MATCH THE CKPT (round-1 incident: 204 tasks died on an unknown action_motion_dim).
#SBATCH --job-name=aq0_eval
#SBATCH --partition=main-redhat
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

echo "AQ0 EVAL start $(date) host=$(hostname) CKPT=$CKPT ID=$ID N1SH=$N1SH N2SH=$N2SH"
if [ "$ID" -lt "$N1SH" ]; then
  KEY=$("$PY" -m namo.eval_sets onepush_manifest)
  N1=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
  S=$((ID * N1 / N1SH)); EN=$(((ID + 1) * N1 / N1SH))
  mkdir -p "$OUT/1push_hmax2"
  "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --key "$KEY" --hmax 2 --sim-budget 900 \
    --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
    --out "$OUT/1push_hmax2/shard_${ID}.json" --leaf-out "$OUT/1push_hmax2/shard_${ID}.jsonl"
else
  I=$((ID - N1SH))
  N2=$($PY -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
  S=$((I * N2 / N2SH)); EN=$(((I + 1) * N2 / N2SH))
  mkdir -p "$OUT/2push"
  "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget 900 \
    --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
    --out "$OUT/2push/shard_${I}.json" --leaf-out "$OUT/2push/shard_${I}.jsonl"
fi
echo "AQ0 EVAL done $(date) ID=$ID"
