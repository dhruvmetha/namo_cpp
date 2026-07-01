#!/bin/bash
# Bit-compare gate for the render speedup. MODE=capture (render N ref crops, save) or compare (vs ref).
# For capture: revert the render to ORIGINAL first. For compare: restore to the fast render first.
#SBATCH --job-name=render_equiv
#SBATCH --partition=main,main-redhat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --output=logs/render_equiv-%j.out
set -euo pipefail
cd "${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"
source env.amarel.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
echo "MODE=${MODE:?} N=${N:-200} on $(hostname)"
"$NAMO_PYTHON" -u scripts/sandbox/test_render_equiv.py --mode "${MODE:?}" --n "${N:-200}"
echo "DONE"
