#!/bin/bash
#SBATCH --job-name=c0final
#SBATCH --account=general
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/finalize_%j.out
#SBATCH --error=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/finalize_%j.err
set -euo pipefail

: "${NAMO_REPO:?}"
: "${ROOT:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO/scripts:$NAMO_REPO/scripts/pipeline:${PYTHONPATH:-}"

"$NAMO_PYTHON" scripts/pipeline/filter_geom_disjoint.py \
  --gen-xmls "$ROOT/gen" \
  --test-xmls "$ROOT/testset_xmls.txt" \
  --out-manifest "$ROOT/manifests/clean_all.txt" \
  --out-report "$ROOT/manifests/geometry_filter.json" \
  --workers "$SLURM_CPUS_PER_TASK"
"$NAMO_PYTHON" scripts/pipeline/build_colossus0_manifest.py \
  --clean-manifest "$ROOT/manifests/clean_all.txt" \
  --out "$ROOT/manifests/colossus0_1m.txt" \
  --report "$ROOT/manifests/colossus0_1m.json" \
  --aug9 600000 --feb 400000 --seed 42
test "$(wc -l < "$ROOT/manifests/colossus0_1m.txt")" -eq 1000000
