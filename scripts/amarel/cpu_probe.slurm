#!/bin/bash
# Generic CPU runner for sandbox PROBE scripts (model runs on CPU; no GPU). Submit instead of srun-ing into an
# interactive alloc (which gets reclaimed + piggybacks the user's session). Required env: SCRIPT. Optional: ARGS.
#   SCRIPT=scripts/sandbox/check_h2_finish.py ARGS="--n 70" sbatch --job-name=h2probe scripts/amarel/cpu_probe.slurm
#SBATCH --job-name=cpu_probe
#SBATCH --partition=main,main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
: "${SCRIPT:?set SCRIPT=scripts/sandbox/...py}"
echo "probe: $SCRIPT ${ARGS:-} on $(hostname)"
"$NAMO_PYTHON" -u "$SCRIPT" ${ARGS:-}
echo "PROBE_DONE"
