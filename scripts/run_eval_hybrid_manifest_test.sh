#!/usr/bin/env bash
set -euo pipefail

# Long-running jobs on NFS can occasionally hit "Stale file handle" while bash is
# still reading this script. Re-exec from a local temp copy to avoid that class
# of failure.
if [[ "${NAMO_EVAL_LOCALIZED:-0}" != "1" ]]; then
  ORIG_SCRIPT="${BASH_SOURCE[0]}"
  TMP_SCRIPT="$(mktemp /tmp/run_eval_hybrid_manifest_test.XXXXXX.sh)"
  cp "$ORIG_SCRIPT" "$TMP_SCRIPT"
  chmod +x "$TMP_SCRIPT"
  exec env NAMO_EVAL_LOCALIZED=1 NAMO_EVAL_ORIG_SCRIPT="$ORIG_SCRIPT" "$TMP_SCRIPT" "$@"
fi

# Hybrid (ML-first -> primitive fallback) evaluation on the aug9_medium "manifest_test" suite.
#
# Notes:
# - RegionOpening runs one attempt per neighbour region.
# - Environment-level success is typically computed as:
#     opened_neighbours / total_neighbours > 0.5
#   (you can aggregate this later from the per-neighbour `*_results.pkl` files).
#
# This script:
# - Uses 1 worker (sequential_ml_collection.py), 1 GPU for inference, and 1-thread CPU settings.
# - Writes the same `*_results.pkl` files as the existing manifest_test evaluation pipeline.
#
# Defaults:
# - Manifest dir: /common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium
# - Manifest file (if a dir is given): manifest_test.txt
# - Checkpoint: vdiff_ms_cross_attn_coordgrid_max_abs_dec30 epoch=909 (user provided)
#
# Usage:
#   ./scripts/run_eval_hybrid_manifest_test.sh
#
# Override examples:
#   ./scripts/run_eval_hybrid_manifest_test.sh --gpu 1
#   ./scripts/run_eval_hybrid_manifest_test.sh --manifest /path/to/manifest_test.txt
#   ./scripts/run_eval_hybrid_manifest_test.sh --manifest /path/to/manifest_dir
#   ./scripts/run_eval_hybrid_manifest_test.sh --ckpt /path/to/other.ckpt
#   ./scripts/run_eval_hybrid_manifest_test.sh --start-idx 0 --end-idx 250

SCRIPT_PATH="${NAMO_EVAL_ORIG_SCRIPT:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
NAMO_CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$NAMO_CPP_DIR/.." && pwd)"

COLLECT_SCRIPT="$NAMO_CPP_DIR/python/namo/data_collection/sequential_ml_collection.py"

DEFAULT_MANIFEST_DIR="/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium"
DEFAULT_MANIFEST_FILE="$DEFAULT_MANIFEST_DIR/manifest_test.txt"
DEFAULT_CKPT="/common/users/tdn39/Robotics/Mujoco/sage_learning/outputs/2026-01-06/vdiff_ms_cross_attn_coordgrid_max_abs_dec30_20260106_104925/checkpoints/epoch=909-val_loss=0.0635.ckpt"

# This base matches the absolute paths in the aug9 manifests.
DEFAULT_XML_DIR="/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/medium"
DEFAULT_NAMO_CONFIG="config/namo_config_complete_skill15.yaml"
DEFAULT_EVAL_RESULTS_ROOT="$NAMO_CPP_DIR/eval_results/manifest_test_hybrid"

CKPT_PATH="$DEFAULT_CKPT"
MANIFEST_PATH="$DEFAULT_MANIFEST_FILE"
XML_DIR="$DEFAULT_XML_DIR"
NAMO_CONFIG_FILE="$DEFAULT_NAMO_CONFIG"
EVAL_RESULTS_ROOT="$DEFAULT_EVAL_RESULTS_ROOT"

GPU_ID="0"
START_IDX="0"
END_IDX=""
CHAIN_DEPTH="1"
ML_SEED="42"

# Budgets / caps (match manifest_test reference defaults unless overridden).
ML_MAX_TERMINAL_CHECKS="${ML_MAX_TERMINAL_CHECKS:-20000}"
ML_MAX_SOLUTIONS_PER_NEIGHBOR="${ML_MAX_SOLUTIONS_PER_NEIGHBOR:-1}"
ML_MAX_RECORDED_SOLUTIONS_PER_NEIGHBOR="${ML_MAX_RECORDED_SOLUTIONS_PER_NEIGHBOR:-1}"

EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $0 [options] [-- <extra args passed to sequential_ml_collection.py>]

Options:
  --ckpt PATH            Model checkpoint (.ckpt) (default: $CKPT_PATH)
  --manifest PATH        Manifest file, OR a directory containing manifest_test.txt (default: $MANIFEST_PATH)
  --xml-dir PATH         XML directory base (default: $XML_DIR)
  --config-file PATH     NAMO config YAML (default: $NAMO_CONFIG_FILE)
  --eval-results-root P  Root for outputs (default: $EVAL_RESULTS_ROOT)
  --gpu ID               CUDA_VISIBLE_DEVICES value (default: $GPU_ID)
  --start-idx N          Start index into manifest (default: $START_IDX)
  --end-idx N            End index into manifest (default: auto from manifest)
  --chain-depth N        RegionOpening chain depth (default: $CHAIN_DEPTH)
  --seed N               ML inference seed (default: $ML_SEED)
  --help|-h              Show this help

Env overrides:
  VENV_PATH=/common/home/tdn39/.virtualenvs/mujoco   (virtualenv to source)
  SKIP_VENV=1                                        (disable venv activation)
  ML_MAX_TERMINAL_CHECKS=20000
  ML_MAX_SOLUTIONS_PER_NEIGHBOR=1
  ML_MAX_RECORDED_SOLUTIONS_PER_NEIGHBOR=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt|--ml-goal-model)
      CKPT_PATH="${2:-}"; shift 2 ;;
    --manifest)
      MANIFEST_PATH="${2:-}"; shift 2 ;;
    --xml-dir)
      XML_DIR="${2:-}"; shift 2 ;;
    --config-file)
      NAMO_CONFIG_FILE="${2:-}"; shift 2 ;;
    --eval-results-root|--output-root)
      EVAL_RESULTS_ROOT="${2:-}"; shift 2 ;;
    --gpu)
      GPU_ID="${2:-}"; shift 2 ;;
    --start-idx)
      START_IDX="${2:-}"; shift 2 ;;
    --end-idx)
      END_IDX="${2:-}"; shift 2 ;;
    --chain-depth)
      CHAIN_DEPTH="${2:-}"; shift 2 ;;
    --seed)
      ML_SEED="${2:-}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$CKPT_PATH" || ! -e "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [[ -d "$MANIFEST_PATH" ]]; then
  MANIFEST_PATH="$MANIFEST_PATH/manifest_test.txt"
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: manifest not found: $MANIFEST_PATH" >&2
  exit 2
fi

if [[ -z "$END_IDX" ]]; then
  END_IDX="$(grep -cve '^[[:space:]]*$' "$MANIFEST_PATH")"
fi

# Single-thread CPU settings
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

# Optional venv activation (matches SLURM scripts; safe to override by exporting SKIP_VENV=1).
VENV_PATH="${VENV_PATH:-/common/home/tdn39/.virtualenvs/mujoco}"
if [[ "${SKIP_VENV:-0}" != "1" && -f "$VENV_PATH/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  if [[ -f "$VENV_PATH/bin/postactivate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/postactivate"
  fi
  set -u
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

export SAGE_LEARNING_PATH="${SAGE_LEARNING_PATH:-$ROOT_DIR/sage_learning}"
export PYTHONPATH="$NAMO_CPP_DIR/build_python:$NAMO_CPP_DIR/python:$NAMO_CPP_DIR/python/namo/visualization:$ROOT_DIR/sage_learning:${PYTHONPATH:-}"

# Output directory layout mirrors the SLURM manifest_test worker so downstream tools can reuse it.
MODEL_DIR="$(dirname "$(dirname "$CKPT_PATH")")"
MODEL_TAG="$(basename "$MODEL_DIR")"
CKPT_STEM="$(basename "$CKPT_PATH" .ckpt | tr '=.' '__')"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
START_PAD="$(printf '%06d' "$START_IDX")"
END_PAD="$(printf '%06d' "$END_IDX")"
OUT_DIR="$EVAL_RESULTS_ROOT/${MODEL_TAG}/${CKPT_STEM}/shard_0_start${START_PAD}_end${END_PAD}_job${RUN_TAG}"

mkdir -p "$OUT_DIR"

echo "=== Hybrid manifest-test evaluation ==="
echo "Checkpoint:       $CKPT_PATH"
echo "Manifest:         $MANIFEST_PATH"
echo "Indices:          [$START_IDX, $END_IDX)"
echo "Output dir:       $OUT_DIR"
echo "GPU:              CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Threads:          OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Chain depth:      $CHAIN_DEPTH"
echo "ML seed:          $ML_SEED"
echo "Max terminal chk: $ML_MAX_TERMINAL_CHECKS"
echo "======================================"

cd "$NAMO_CPP_DIR"

python3 "$COLLECT_SCRIPT" \
  --algorithm region_opening \
  --config-file "$NAMO_CONFIG_FILE" \
  --xml-dir "$XML_DIR" \
  --manifest "$MANIFEST_PATH" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --episodes-per-env 1 \
  --workers 1 \
  --output-dir "$OUT_DIR" \
  --keep-empty-action-successes \
  --goal-strategy ml_fallback \
  --ml-goal-model "$CKPT_PATH" \
  --ml-device cuda \
  --ml-samples 32 \
  --ml-seed "$ML_SEED" \
  --ml-k-nearest 5 \
  --ml-match-position-tolerance 999 \
  --ml-match-angle-tolerance 999 \
  --ml-match-angle-weight 1 \
  --ml-match-max-per-call 999 \
  --goals-per-region 10 \
  --region-allow-collisions \
  --region-selection-strategy ml_first \
  --region-max-chain-depth "$CHAIN_DEPTH" \
  --region-max-solutions-per-neighbor "$ML_MAX_SOLUTIONS_PER_NEIGHBOR" \
  --region-max-recorded-solutions-per-neighbor "$ML_MAX_RECORDED_SOLUTIONS_PER_NEIGHBOR" \
  --region-frontier-beam-width 1000 \
  --region-disable-edge-blacklist \
  --max-terminal-checks "$ML_MAX_TERMINAL_CHECKS" \
  "${EXTRA_ARGS[@]}"
