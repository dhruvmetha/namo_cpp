#!/bin/bash
#SBATCH --job-name=resweep_drift
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# Re-measure keyhole-1 difficulty on a stratified 200-scene sample of the 2-hop pool,
# under the push physics of a8f61fd (object-object and object-wall contact no longer
# aborts a push). Every tier label we hold was swept before that change, so the whole
# difficulty axis has to be re-read before anything is built on top of it.
#
# Sample: 50 scenes from each of the four old buckets (dead / hard / med / easy) on the
# hmax=2 union axis, seed 20260820. Manifest and the old labels it was drawn from live in
# $NAMO_SCRATCH/resweep_20260820/ (override with ROOT=).
#
# Identical protocol to the original keyhole-1 labelling run (kh_label.slurm, 2026-08-17)
# except for the binding it links: same config, same exhaustive depth-2 sweep, one
# worker per scene. That is the point -- the binding is the only variable.
#
#   sbatch --array=0-7 scripts/slurm/resweep_tier_drift.slurm      (submit from the repo root)
set -euo pipefail
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?set NAMO_REPO or submit from the repo}}
cd "$NAMO_REPO"
source env.amarel.sh
export PYTHONPATH="$PWD/build_python:$PWD/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

ROOT=${ROOT:-$NAMO_SCRATCH/resweep_20260820}
MANIFEST=$ROOT/manifest.txt
CFG=python/namo/data_collection/region_opening_exhaustive_2push_multihop_car.yaml

# HARD GATE -- on halk0018/0020/0028 this file once read back as NUL bytes and
# modular_parallel_collection only WARNED, fell through to argparse defaults, collected
# zero episodes and still exited 0 at "100% success". Fail loudly instead.
python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); assert c.get('region_exhaustive_mode') is True; assert c.get('region_max_chain_depth')==2; print('config OK', len(c),'keys')" "$CFG"

# The .so must be the one built from this tree, or the run measures the old physics again.
python -c "
import pathlib, subprocess
# splitlines, not split: values carry spaces (python=Python 3.11.14).
info = dict(l.split('=', 1) for l in pathlib.Path('build_python/BUILD_INFO').read_text().splitlines() if '=' in l)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert info['git_sha'] == head, 'stale .so: built from %s, HEAD is %s' % (info['git_sha'], head)
assert info['dirty_cpp'] == '0', 'C++ tree was dirty at build time'
print('binding OK', info['git_sha'][:8], info['built_at'])
"

CHUNK=${CHUNK:-25}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
OUT=$ROOT/shards/shard_$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
rm -rf "$OUT"; mkdir -p "$OUT"

echo "TASK $SLURM_ARRAY_TASK_ID idx [$START,$END) node $(hostname) start $(date)"
python -m namo.data_collection.modular_parallel_collection \
  --config-yaml "$CFG" --algorithm region_opening \
  --manifest "$MANIFEST" --output-dir "$OUT" \
  --start-idx "$START" --end-idx "$END" --workers "$CHUNK" \
  --primitive-data-dir "$PWD/data"
echo "TASK $SLURM_ARRAY_TASK_ID done $(date)"
