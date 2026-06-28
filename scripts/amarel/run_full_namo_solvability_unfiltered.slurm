#!/bin/bash
# Run the exact-n Full NAMO solvability pipeline over the shared
# envs_unfiltered XML pool.
#
# Default behavior:
# - scans INPUT_DIR for XMLs
# - auto-discovers all exact shortest-path lengths n present in the pool
# - runs one exact_n_solvability_runner invocation per n
# - writes manifests under OUTPUT_ROOT/n_<n>/
#
# Submit:
#   sbatch scripts/amarel/run_full_namo_solvability_unfiltered.slurm
#
# Common overrides at submit time:
#   INPUT_DIR=/path/to/xmls
#   OUTPUT_ROOT=/path/to/output_root
#   N_VALUES="1 2 3"
#   FULL_NAMO_WORKERS=16
#   GOAL_STRATEGY=primitive
#   SIMULATION_BUDGET=200000
#
# Notes:
# - Uses the current runner defaults you asked for:
#   random_rollout, allow collisions, unified C++ snapshot, seed=42,
#   no outer FullNAMO iteration cap, simulation_budget=100000
# - Explicitly pins region_max_chain_depth=2 (max 2 pushes per hop)
#
#SBATCH --job-name=namo-full-solv
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fullnamo-solv-%j.out
#SBATCH --error=logs/fullnamo-solv-%j.err

set -euo pipefail
mkdir -p "$NAMO_LOGS"

NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"

INPUT_DIR=${INPUT_DIR:-/common/users/shared/robot_learning/dm1487/namo/full_namo_tests/envs_unfiltered}
OUTPUT_ROOT=${OUTPUT_ROOT:-/common/users/shared/robot_learning/dm1487/namo/full_namo_tests/envs_unfiltered/envs}
CONFIG_FILE=${CONFIG_FILE:-config/namo_config_complete_skill15.yaml}
PRIMITIVE_DATA_DIR=${PRIMITIVE_DATA_DIR:-data}

GOAL_STRATEGY=${GOAL_STRATEGY:-random_rollout}
REGION_MAX_CHAIN_DEPTH=${REGION_MAX_CHAIN_DEPTH:-2}
SIMULATION_BUDGET=${SIMULATION_BUDGET:-100000}
GOALS_PER_REGION=${GOALS_PER_REGION:-10}
REGION_SUCCESS_MIN_REACHABLE=${REGION_SUCCESS_MIN_REACHABLE:-1}
SEED=${SEED:-42}
FULL_NAMO_WORKERS=${FULL_NAMO_WORKERS:-$SLURM_CPUS_PER_TASK}

# Optional overrides
N_VALUES=${N_VALUES:-}
PRIMITIVE_PREFIX=${PRIMITIVE_PREFIX:-}
ROLLOUT_SAMPLES_PER_STATE=${ROLLOUT_SAMPLES_PER_STATE:-}
REGION_FRONTIER_BEAM_WIDTH=${REGION_FRONTIER_BEAM_WIDTH:-}
FULL_NAMO_MAX_ITERATIONS=${FULL_NAMO_MAX_ITERATIONS:-}
LIMIT=${LIMIT:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

export NAMO_REPO INPUT_DIR OUTPUT_ROOT CONFIG_FILE

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "repo:                 $NAMO_REPO"
echo "input_dir:            $INPUT_DIR"
echo "output_root:          $OUTPUT_ROOT"
echo "config_file:          $CONFIG_FILE"
echo "goal_strategy:        $GOAL_STRATEGY"
echo "region_max_chain_depth=$REGION_MAX_CHAIN_DEPTH"
echo "simulation_budget:    $SIMULATION_BUDGET"
echo "workers:              $FULL_NAMO_WORKERS"
echo "start:                $(date -Is)"

if [ -n "$N_VALUES" ]; then
    read -r -a PATH_LENGTHS <<< "$N_VALUES"
else
    mapfile -t PATH_LENGTHS < <(
        python - <<'PY'
import os
from pathlib import Path

from namo.environment_selection import analyze_environment_path_length, get_xml_files

repo_root = Path(os.environ["NAMO_REPO"])
input_dir = os.environ["INPUT_DIR"]
config_file = os.environ["CONFIG_FILE"]
config_path = Path(config_file)
if not config_path.is_absolute():
    config_path = repo_root / config_path

lengths = set()
for xml_path in get_xml_files(input_dir=input_dir):
    analysis = analyze_environment_path_length(
        xml_path,
        str(config_path),
        use_cpp_snapshot=True,
    )
    if analysis.selection_error is None and analysis.path_length_n >= 0:
        lengths.add(int(analysis.path_length_n))

for n in sorted(lengths):
    print(n)
PY
    )
fi

if [ ${#PATH_LENGTHS[@]} -eq 0 ]; then
    echo "No exact path lengths discovered in $INPUT_DIR; nothing to run."
    exit 0
fi

echo "path_lengths:         ${PATH_LENGTHS[*]}"

for n in "${PATH_LENGTHS[@]}"; do
    run_output="$OUTPUT_ROOT/n_${n}"
    mkdir -p "$run_output"

    cmd=(
        python "$NAMO_REPO/python/namo/scripts/exact_n_solvability_runner.py"
        --xml-dir "$INPUT_DIR"
        --path-length "$n"
        --output-dir "$run_output"
        --config-file "$CONFIG_FILE"
        --goal-strategy "$GOAL_STRATEGY"
        --region-max-chain-depth "$REGION_MAX_CHAIN_DEPTH"
        --region-allow-collisions
        --primitive-data-dir "$PRIMITIVE_DATA_DIR"
        --goals-per-region "$GOALS_PER_REGION"
        --region-success-min-reachable "$REGION_SUCCESS_MIN_REACHABLE"
        --seed "$SEED"
        --use-cpp-snapshot
        --simulation-budget "$SIMULATION_BUDGET"
        --workers "$FULL_NAMO_WORKERS"
    )

    if [ -n "$PRIMITIVE_PREFIX" ]; then
        cmd+=(--primitive-prefix "$PRIMITIVE_PREFIX")
    fi
    if [ -n "$ROLLOUT_SAMPLES_PER_STATE" ]; then
        cmd+=(--rollout-samples-per-state "$ROLLOUT_SAMPLES_PER_STATE")
    fi
    if [ -n "$REGION_FRONTIER_BEAM_WIDTH" ]; then
        cmd+=(--region-frontier-beam-width "$REGION_FRONTIER_BEAM_WIDTH")
    fi
    if [ -n "$FULL_NAMO_MAX_ITERATIONS" ]; then
        cmd+=(--full-namo-max-iterations "$FULL_NAMO_MAX_ITERATIONS")
    fi
    if [ -n "$LIMIT" ]; then
        cmd+=(--limit "$LIMIT")
    fi
    if [ -n "$EXTRA_ARGS" ]; then
        read -r -a extra_array <<< "$EXTRA_ARGS"
        cmd+=("${extra_array[@]}")
    fi

    printf 'running n=%s:' "$n"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"

    echo "summary for n=$n:"
    cat "$run_output/summary.json"
    echo
done

echo "end:                  $(date -Is)"
