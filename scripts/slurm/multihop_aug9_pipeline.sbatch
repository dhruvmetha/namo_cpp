#!/bin/bash
# Dependency stages for sampled Aug9 independent certification and random-five evaluation.
# All machine paths and phase-specific settings come from CAMPAIGN_ENV.
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
set -euo pipefail
source "${CAMPAIGN_ENV:?set the campaign environment file}"
source "$NAMO_ACTIVATE_SH"
export PYTHONPATH="$NAMO_BINDINGS_DIR:$NAMO_REPO/python:$NAMO_REPO/scripts/pipeline:$RUN_ROOT/deps:${PYTHONPATH:-}"
sha256sum --check "$RUNTIME_HASHES"
phase_root="$RUN_ROOT/${PHASE:?}"
certifier="$NAMO_REPO/scripts/pipeline/certify_keyhole_horizons.py"
case "${1:?setup, prepare, certify, publish, or aggregate}" in
  setup)
    "$NAMO_PYTHON" -m pip install --no-deps --target "$RUN_ROOT/deps" shapely==2.1.2
    "$NAMO_PYTHON" - <<'PY'
import os, namo_rl, shapely, yaml
from pathlib import Path
config = yaml.safe_load(Path(os.environ["NAMO_CONFIG_FILE"]).read_text())
assert config["skill"]["check_robot_trajectory_collision"] is True
assert config["skill"]["stuck_check_stride"] == 150
assert str(Path(namo_rl.__file__).resolve()).startswith(os.environ["NAMO_BINDINGS_DIR"] + "/")
print("Ready:", namo_rl.__file__, "Shapely", shapely.__version__)
PY
    ;;
  prepare)
    "$NAMO_PYTHON" "$certifier" prepare-generated \
      --source "$phase_root/generated" --output "$phase_root/tasks.jsonl" \
      --witness-budget "$WITNESS_BUDGET"
    ;;
  certify)
    "$NAMO_PYTHON" "$certifier" run --tasks "$phase_root/tasks.jsonl" \
      --task-index "$SLURM_ARRAY_TASK_ID" --task-count "$SLURM_ARRAY_TASK_COUNT" \
      --output-dir "$phase_root/certification" --config "$NAMO_CONFIG_FILE"
    ;;
  publish)
    "$NAMO_PYTHON" "$certifier" aggregate-generated --tasks "$phase_root/tasks.jsonl" \
      --results-dir "$phase_root/certification" --output-dir "$phase_root/testbed"
    total=$(wc -l < "$phase_root/testbed/xmls.txt")
    if (( total == 0 )); then
      echo "No scenes in either certified group; nothing to evaluate."
      exit 0
    fi
    shards=$(( total < RANDOM_MAX_SHARDS ? total : RANDOM_MAX_SHARDS ))
    dependency=afterok
    jobs=()
    for seed in 7000 8000 9000 10000 11000; do
      job=$(sbatch --parsable --partition=main --account=general \
        --job-name="aug9-r${seed}" --time=08:00:00 --cpus-per-task=1 --mem=4G \
        --array="0-$((shards - 1))%$RANDOM_CONCURRENCY" \
        --output="$phase_root/logs/random_s${seed}_%A_%a.out" \
        --export="ALL,NAMO_REPO=$EVAL_REPO,N_SHARDS=$shards,WORKERS=1,PRIOR=uniform,SHUFFLE_SEED=$seed,MANIFEST=$phase_root/testbed/xmls.txt,OUT=$phase_root/raw/random_s$seed" \
        "$PIPELINE_REPO/scripts/slurm/multihop_aug9_eval.slurm")
      jobs+=("$job")
      dependency="$dependency:$job"
    done
    aggregate_job=$(sbatch --parsable --dependency="$dependency" \
      --job-name=aug9-r5-report --output="$phase_root/logs/random_aggregate_%j.out" \
      --export="ALL,NAMO_REPO=$PIPELINE_REPO" \
      "$PIPELINE_REPO/scripts/slurm/multihop_aug9_pipeline.sbatch" aggregate)
    "$NAMO_PYTHON" - "$phase_root/random_jobs.json" "$shards" "$aggregate_job" "${jobs[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"shards_per_seed": int(sys.argv[2]),
    "aggregate": sys.argv[3], "random_arrays": dict(zip(
        ["7000", "8000", "9000", "10000", "11000"], sys.argv[4:]))}, indent=2) + "\n")
PY
    ;;
  aggregate)
    total=$(wc -l < "$phase_root/testbed/xmls.txt")
    shards=$(( total < RANDOM_MAX_SHARDS ? total : RANDOM_MAX_SHARDS ))
    "$NAMO_PYTHON" "$RANDOM_ANALYSIS_SCRIPT" --run-root "$phase_root" \
      --source-manifest "$phase_root/testbed/manifest.jsonl" \
      --seeds 7000 8000 9000 10000 11000 --shards-per-seed "$shards" \
      --expected-scenes "$total"
    ;;
  *) echo "Unknown pipeline stage: $1" >&2; exit 2 ;;
esac
