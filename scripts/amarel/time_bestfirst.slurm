#!/bin/bash
# Best-first SEARCH wall-clock timing on ONE node (same machine), CPU, single-threaded.
# Env: OUT (jsonl path). Opt: NPER (episodes/tier, default 50), BUDGET (sims/scene, default 30).
#SBATCH --job-name=bftimebench
#SBATCH --partition=main,main-redhat
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=logs/bftimebench-%j.out
set -euo pipefail
cd "${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"
source env.amarel.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${OUT:?}"
"$NAMO_PYTHON" -u scripts/sandbox/time_bestfirst.py --n-per-tier "${NPER:-50}" --budget "${BUDGET:-30}" \
  --hmax "${HMAX:-2}" --key "${KEY:-$NAMO_DATASETS/namo_testset_v1/labels/pure2push.json}" --out "$OUT"
