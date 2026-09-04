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
#   (3) --time: NEVER omit (CS main DEFAULT is 2 MINUTES -> instant kill) and NEVER use
#       partition MAX on a busy GPU queue (a 3-day request can't backfill -> may never start).
#       Rule: ~2-3x the EXPECTED runtime. See memory: feedback_no_slurm_time_limits.
#   (4) CS `unlimited` partition REJECTS --cpus-per-task -> omit it there (Amarel allows it).
#   (5) train_q2_rankaux HANGS on DataLoader teardown AFTER writing the ckpt -> poll the log for
#       the "[eval_scorer-load check]" marker, then kill the hung process.
#
# Usage (CS, one model):   H5=/path/x.h5 OUT=/path/outdir sbatch scripts/slurm/train.slurm
# Usage (gamma-sweep):     set NAMO_GAMMA_SWEEP=1 and add `--array=0-3`; task->gamma via GAMMAS below.
#   e.g.  H5=... OUT=/path/models NAMO_GAMMA_SWEEP=1 sbatch --array=0-3 --gres=gpu:a100:1 scripts/slurm/train.slurm
# Amarel deltas: `--partition=gpu`; `--cpus-per-task=8` is allowed; NAMO_PYTHON is already
#   an absolute path there so (1) is a no-op; `source env.amarel.sh` instead of env.ilab.sh.
# Tunables via env: EPOCHS(12) BATCH(256) WORKERS(8) LR(3e-4) SEED(1) PATIENCE(0) RANK_LAMBDA(0.1) LOWER_RANK_LAMBDA(0.05) RANK_TEMP(0.15)
# Optional action grounding: NAMO_ACTION_MOTION=1 enables plain crop-relative (dx,dy,dtheta) late
#   fusion. Add NAMO_ACTION_MOTION_SHARP=1 for Fourier features + learned depth identity. Motion
#   remains OFF by default; NAMO_ACTION_MOTION_ENCODING is only needed for explicit historical modes.
#   NAMO_ACTION_DEPTH_SELF_ATTN=1 additionally compares the five motion-grounded depths within each
#   contact using one local attention head; it never runs attention across all 300 actions.
# SPEED (measured 2026-07-17, test-small-first before trusting — see memory reference_training_speedup):
#   train is CO-bottlenecked (NFS reads ~= GPU compute). Two fixes TOGETHER: (a) stage the H5 to
#   node-local /dev/shm before the train line (4.3x dataloader); (b) torch.compile the net (1.78x).
#   Also: models overfit by epoch 3-4 -> EPOCHS=5 or add --patience (faster AND avoids overfit).
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
SAGE_OVERRIDE=${SAGE_REPO:-}
ENV_SCRIPT=${NAMO_ENV_SCRIPT:-env.ilab.sh}
cd "$REPO" && source "$ENV_SCRIPT" >/dev/null 2>&1
if [ -n "$SAGE_OVERRIDE" ]; then export SAGE_REPO="$SAGE_OVERRIDE"; fi

# (1) resolve a WORKING interpreter — env.ilab.sh leaves NAMO_PYTHON="python", absent in sbatch
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=/common/users/dm1487/envs/mjxrl/bin/python
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO/build_python:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:${SAGE_REPO:-}:${PYTHONPATH:-}"

# (2) pin BLAS/OMP thread pools (CS ulimit -u=2000; GPU does the compute, CPU threads just thrash)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TMPDIR=/tmp/namo_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"   # loky DataLoader temp-dir collision fix
export RANK_LAMBDA=${RANK_LAMBDA:-0.1} LOWER_RANK_LAMBDA=${LOWER_RANK_LAMBDA:-0.05} RANK_TEMP=${RANK_TEMP:-0.15}

# Gamma sweeping is explicit because many unrelated launchers also use SLURM arrays.
if [ "${NAMO_GAMMA_SWEEP:-0}" = "1" ]; then
  : "${SLURM_ARRAY_TASK_ID:?NAMO_GAMMA_SWEEP=1 requires a four-task SLURM array}"
  GAMMAS=(0.3 0.5 0.7 0.9); G=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
  export NAMO_GAMMA=$G; OUT="$OUT/g${G}"
fi
# SPEED: stage the H5 to node-local storage (4.3x vs NFS; see header notes).
#
# DEFAULT IS /dev/shm. Measured 2026-08-08 on rlab3, same node/data/code, only this differing:
#     /dev/shm  4.15 min/epoch          /tmp  ~15 min/epoch    (3.6x)
# because /tmp IS NODE-DEPENDENT: NVMe on ilab2 (/dev/nvme0n1p4) but the ROOT RAID array on rlab3
# (/dev/md2, no separate mount). Staging a fleet to /tmp there means every worker does random
# reads against spinning storage. An earlier edit today defaulted to /tmp on the untested
# assumption that node-local disk ~ RAM; it is not, and it cost ~40 min of wall time.
#
# /dev/shm caveats, both real but narrower than they looked:
#   * systemd RemoveIPC deletes it when the user's LAST session on the node closes -- this killed
#     jobs on rlab7. Other nodes ran a dozen /dev/shm jobs today without incident.
#   * it is RAM, but rlab3 shows 502 GB with 1.8 GB used, so exhaustion is not the usual case.
# Set NAMO_STAGE_DIR=/tmp on nodes where /tmp is genuinely NVMe, or to dodge RemoveIPC.
# Falls back to NFS silently if the copy fails -- correctness unaffected.
if [ "${NAMO_STAGE_SHM:-1}" = "1" ]; then
  SH5="${NAMO_STAGE_DIR:-/dev/shm}/namo_train_${SLURM_JOB_ID:-$$}.h5"
  if cp "$H5" "$SH5" 2>/dev/null; then
    H5="$SH5"; trap 'rm -f "$SH5"' EXIT
    echo "staged H5 to $SH5"
  else
    rm -f "$SH5" 2>/dev/null; echo "stage-dir copy failed; using NFS path"
  fi
fi
mkdir -p "$OUT"; LOG=${TRAIN_LOG:-$OUT/train.log}

echo "TRAIN start $(date) host=$(hostname) H5=$H5 GAMMA=${NAMO_GAMMA:-none} PY=$PY OUT=$OUT RESUME_FROM=${RESUME_FROM:-none} LOG=$LOG"
EXTRA=()
if [ -n "${POSTCHECK_LIMIT:-}" ]; then EXTRA+=(--postcheck-limit "$POSTCHECK_LIMIT"); fi
if [ -n "${RESUME_FROM:-}" ]; then
  [ -f "$RESUME_FROM" ] || { echo "missing resume checkpoint: $RESUME_FROM" >&2; exit 1; }
  EXTRA+=(--resume-from "$RESUME_FROM")
fi
RUNNER=()
if [ "${NAMO_USE_SRUN:-0}" = "1" ]; then RUNNER=(srun --nodes=1 --ntasks=1); fi
"${RUNNER[@]}" "$PY" "${TRAIN_SCRIPT:-scripts/rl_loop/train_q2_rankaux.py}" --h5 "$H5" --out-dir "$OUT" \
  --epochs "${EPOCHS:-12}" --batch-size "${BATCH:-256}" --num-workers "${WORKERS:-8}" \
  --lr "${LR:-3e-4}" --seed "${SEED:-1}" --split-seed 0 \
  --patience "${PATIENCE:-0}" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!
MARKER_SEEN=0
# (5) hang-kill: ckpt written + "[eval_scorer-load check]" printed, then teardown hangs -> kill
while kill -0 $PID 2>/dev/null; do
  if grep -q "eval_scorer-load check" "$LOG" 2>/dev/null; then
    echo "train done (marker seen); killing hung teardown"; MARKER_SEEN=1
    sleep 5; kill -9 $PID 2>/dev/null; wait "$PID" 2>/dev/null || true; break
  fi
  STATE=$(ps -o stat= -p "$PID" 2>/dev/null || true)
  if [[ "$STATE" == Z* ]]; then
    wait "$PID"; STATUS=$?
    echo "train exited before completion marker status=$STATUS"
    exit "$STATUS"
  fi
  sleep 30
done
if [ "$MARKER_SEEN" -eq 0 ]; then
  wait "$PID"; STATUS=$?
  if [ "$STATUS" -ne 0 ]; then echo "train exited before completion marker status=$STATUS"; exit "$STATUS"; fi
  if grep -q "eval_scorer-load check" "$LOG" 2>/dev/null; then
    MARKER_SEEN=1
  else
    echo "train process exited without the eval_scorer-load check marker"
    exit 1
  fi
fi
compgen -G "$OUT/checkpoints/*.ckpt" >/dev/null || { echo "train completed without a checkpoint"; exit 1; }
echo "TRAIN DONE gamma=${NAMO_GAMMA:-none} $(date)"; ls -lh "$OUT/checkpoints/" 2>/dev/null
