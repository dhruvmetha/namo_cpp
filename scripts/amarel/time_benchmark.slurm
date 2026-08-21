#!/bin/bash
# Wall-clock timing benchmark on ONE node (same machine for all models), CPU, single-threaded.
# Env: OUT (jsonl path). Opt: NPER (episodes per tier, default 100).
#SBATCH --job-name=timebench
#SBATCH --partition=main,main
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=logs/timebench-%j.out
set -euo pipefail
cd "${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"
source env.amarel.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${OUT:?}"
"$NAMO_PYTHON" -u scripts/sandbox/time_benchmark.py --n-per-tier "${NPER:-100}" --out "$OUT"
