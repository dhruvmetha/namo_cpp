#!/bin/bash
# ============================================================================================
# CANONICAL NAMO GPU-TRAIN TEMPLATE. Encodes the fixes that bit us repeatedly on 2026-07-17 so
# they can't be forgotten — start every train from THIS, don't hand-roll an sbatch.
#
# Gotchas baked in (each cost us a failed full-scale launch):
#   (1) env.ilab.sh does NOT conda-activate -> bare `python` is MISSING in the non-interactive
#       sbatch shell ("python: command not found"). We resolve the interpreter explicitly (PY).
#   (2) CS ulimit -u=2000 + OpenBLAS defaulting to 64 threads/proc -> >1 train/node blows the
#       thread limit ("pthread_create failed"). Pin all BLAS/OMP pools to 1.
#   (3) --time: NEVER omit (CS main-redhat DEFAULT is 2 MINUTES -> instant kill) and NEVER use
#       partition MAX on a busy GPU queue (a 3-day request can't backfill -> may never start).
#       Rule: ~2-3x the EXPECTED runtime. See memory: feedback_no_slurm_time_limits.
#   (4) CS `unlimited` partition REJECTS --cpus-per-task -> omit it there (Amarel allows it).
#   (5) train_q2_rankaux HANGS on DataLoader teardown AFTER writing the ckpt -> poll the log for
#       the "[eval_scorer-load check]" marker, then kill the hung process.
#
# Usage (CS, one model):   H5=/path/x.h5 OUT=/path/outdir sbatch scripts/slurm/train.slurm
# Usage (gamma-sweep):     add `--array=0-3`; task->gamma via GAMMAS below (load-remap, one H5).
#   e.g.  H5=... OUT=/path/models sbatch --array=0-3 --gres=gpu:a100:1 scripts/slurm/train.slurm
# Amarel deltas: `--partition=gpu-redhat`; `--cpus-per-task=8` is allowed; NAMO_PYTHON is already
#   an absolute path there so (1) is a no-op; `source env.amarel.sh` instead of env.ilab.sh.
# Tunables via env: EPOCHS(12) BATCH(256) WORKERS(8) LR(3e-4) SEED(1) RANK_LAMBDA(0.1) RANK_TEMP(0.15)
# ============================================================================================
#SBATCH --job-name=namo_train
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
set -uo pipefail

: "${H5:?set H5=<training h5 path>}"
: "${OUT:?set OUT=<output dir (base dir for gamma-sweep)>}"

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO" && source env.ilab.sh >/dev/null 2>&1            # Amarel: source env.amarel.sh

# (1) resolve a WORKING interpreter — env.ilab.sh leaves NAMO_PYTHON="python", absent in sbatch
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=/common/users/dm1487/envs/mjxrl/bin/python
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO/build_python:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:${SAGE_REPO:-}:${PYTHONPATH:-}"

# (2) pin BLAS/OMP thread pools (CS ulimit -u=2000; GPU does the compute, CPU threads just thrash)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TMPDIR=/tmp/namo_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"   # loky DataLoader temp-dir collision fix
export RANK_LAMBDA=${RANK_LAMBDA:-0.1} RANK_TEMP=${RANK_TEMP:-0.15}

# gamma-sweep: array task -> gamma (load-time setup-label remap; all variants train from ONE H5)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  GAMMAS=(0.3 0.5 0.7 0.9); G=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
  export NAMO_GAMMA=$G; OUT="$OUT/g${G}"
fi
mkdir -p "$OUT"; LOG="$OUT/train.log"

echo "TRAIN start $(date) host=$(hostname) H5=$H5 GAMMA=${NAMO_GAMMA:-none} PY=$PY OUT=$OUT"
"$PY" scripts/rl_loop/train_q2_rankaux.py --h5 "$H5" --out-dir "$OUT" \
  --epochs "${EPOCHS:-12}" --batch-size "${BATCH:-256}" --num-workers "${WORKERS:-8}" \
  --lr "${LR:-3e-4}" --seed "${SEED:-1}" --split-seed 0 > "$LOG" 2>&1 &
PID=$!
# (5) hang-kill: ckpt written + "[eval_scorer-load check]" printed, then teardown hangs -> kill
while kill -0 $PID 2>/dev/null; do
  if grep -q "eval_scorer-load check" "$LOG" 2>/dev/null; then
    echo "train done (marker seen); killing hung teardown"; sleep 5; kill -9 $PID 2>/dev/null; break
  fi
  sleep 30
done
echo "TRAIN DONE gamma=${NAMO_GAMMA:-none} $(date)"; ls -lh "$OUT/checkpoints/" 2>/dev/null
