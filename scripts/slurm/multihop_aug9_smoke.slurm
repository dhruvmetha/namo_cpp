#!/bin/bash
# End-to-end Amarel smoke: generate exact two-hop aug9 car XMLs, then solve one with HY5U.
#
# Required:
#   OUT=/scratch/... CKPT=/cache/.../HY5U_s2.ckpt sbatch scripts/slurm/multihop_aug9_smoke.slurm
# Optional:
#   TEMPLATE_REL=set1/benchmark_5.xml START_SEED=640000000 NUM_ENVS=1 EVAL_LIMIT=1
#   NAMO_REPO=/path/to/namo ENV_CREATOR_REPO=/path/to/mujoco_env_creator
#SBATCH --job-name=mh2_smoke
#SBATCH --partition=main-redhat
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

: "${OUT:?set OUT to a new smoke output directory}"
: "${CKPT:?set CKPT to the HY5U checkpoint}"

NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
ENV_CREATOR_REPO=${ENV_CREATOR_REPO:-$(dirname "$NAMO_REPO")/mujoco_env_creator}
TEMPLATE_REL=${TEMPLATE_REL:-set1/benchmark_5.xml}
START_SEED=${START_SEED:-640000000}
NUM_ENVS=${NUM_ENVS:-1}
EVAL_LIMIT=${EVAL_LIMIT:-1}

cd "$NAMO_REPO"
source env.amarel.sh
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${NAMO_PYTHON:-python}

CONFIG="$NAMO_REPO/config/namo_config_complete_skill15_car_1x.yaml"
TEMPLATE_ROOT="$ENV_CREATOR_REPO/templates/aug9_car_v3"
mkdir -p "$OUT/generated" "$OUT/eval" "$NAMO_REPO/logs"

echo "SMOKE generation start $(date) host=$(hostname) template=$TEMPLATE_REL seed=$START_SEED"
"$PY" "$ENV_CREATOR_REPO/generate_envs.py" "$TEMPLATE_ROOT/$TEMPLATE_REL" \
  --namo-config "$CONFIG" \
  --num-envs "$NUM_ENVS" \
  --output-dir "$OUT/generated" \
  --num-workers 1 \
  --start-seed "$START_SEED" \
  --num-objects-json "$TEMPLATE_ROOT/num_objects.json" \
  --object-size-range 0.06 0.16 \
  --object-half-height 0.05 \
  --goal-size 0.02 \
  --clearance-radius 0.0 \
  --min-goal-distance 0.0 \
  --no-require-adjacent \
  --exact-hop 2 \
  --samples-per-pair 1 \
  --min-region-area 0.02

find "$OUT/generated" -type f -name '*.xml' ! -name '*_temp.xml' | sort > "$OUT/generated_xmls.txt"
head -n "$EVAL_LIMIT" "$OUT/generated_xmls.txt" > "$OUT/smoke_manifest.txt"
test -s "$OUT/smoke_manifest.txt"

echo "SMOKE HY5U start $(date) xml_count=$(wc -l < "$OUT/smoke_manifest.txt")"
"$PY" -m namo.solvability_runner \
  --manifest "$OUT/smoke_manifest.txt" \
  --path-length 2 \
  --config-file "$CONFIG" \
  --goal-strategy scorer \
  --region-max-chain-depth 2 \
  --primitive-data-dir "$NAMO_REPO/data" \
  --primitive-prefix 1x_car_d5_ \
  --region-success-min-reachable 20 \
  --goals-per-region 100 \
  --seed 42 \
  --simulation-budget 300 \
  --simulation-budget-scope keyhole \
  --local-search best_first \
  --best-first-prior model \
  --scorer-ckpt "$CKPT" \
  --ml-device cpu \
  --workers 1 \
  --output-dir "$OUT/eval"

"$PY" -c 'import json,sys; print("SMOKE result", json.load(open(sys.argv[1])))' "$OUT/eval/summary.json"
echo "SMOKE done $(date)"
