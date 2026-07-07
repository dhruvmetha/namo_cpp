#!/bin/bash
# CS-side generation: harvest Amarel pkls -> parallel H5 -> train pi+V -> dev eval (run_gen_cs.py).
# iLab `unlimited`, pinned to one GPU node (unlimited does NOT cgroup-slice CPUs, so one job/node;
# the parallel render + dataloader use all cores). Submit --gres=gpu:1 only (no --cpus-per-task).
#   Env in (required): ARM, GEN, RUNROOT, COLLECT, NSHARDS. Optional: CKPT, KEEP(0.1), WORKERS(48),
#     EPOCHS(40), POOL, SPLIT, EVALLIM(0).
#   Submit: ARM=A GEN=0 RUNROOT=.../armA COLLECT=.../armA/gen0/collect NSHARDS=64 \
#     sbatch -w rlab1 --gres=gpu:1 scripts/rl_loop/run_gen_cs.slurm
#SBATCH --job-name=rl_gen_cs
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/rl_runs/logs/gencs-%j.out
#SBATCH --error=/common/users/dm1487/scratch_namo/rl_runs/logs/gencs-%j.err
set -euo pipefail
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
cd "$MAIN"
source env.ilab.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/tmp                 # spawn-worker tempdirs off NFS (avoids .nfs busy cleanup noise)
PY=/common/users/dm1487/envs/mjxrl/bin/python
: "${ARM:?}"; : "${GEN:?}"; : "${RUNROOT:?}"; : "${COLLECT:?}"; : "${NSHARDS:?}"
POOL=${POOL:-/common/users/dm1487/scratch_namo/rl_runs/gen0_pool.json}
SPLIT=${SPLIT:-/common/users/dm1487/scratch_namo/rl_runs/gen0_split.json}
KEEP=${KEEP:-0.1}; WORKERS=${WORKERS:-48}; EPOCHS=${EPOCHS:-40}; EVALLIM=${EVALLIM:-0}; DLWORKERS=${DLWORKERS:-32}
mkdir -p /common/users/dm1487/scratch_namo/rl_runs/logs
"$PY" -u scripts/rl_loop/run_gen_cs.py \
  --arm "$ARM" --generation "$GEN" \
  --pool-key "$POOL" --split-file "$SPLIT" \
  --out-root "$RUNROOT" --pre-collected-dir "$COLLECT" --expected-shards "$NSHARDS" \
  --vhead-fail-keep-frac "$KEEP" --revalidate-fraction 0 --num-workers "$DLWORKERS" \
  --render-workers "$WORKERS" --max-epochs "$EPOCHS" --eval-limit "$EVALLIM" \
  ${CKPT:+--ckpt "$CKPT"}
