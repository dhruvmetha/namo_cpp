#!/usr/bin/env bash
set -euo pipefail

# Submits the full dec2 2-push pipeline:
#  1) Raw -> NPZ (with trajectory-suffix splitting) + NPZ -> H5 + stats
#  2) Train+eval all vector-model architectures (parallel, after conversion)
#
# Usage:
#   bash namo_cpp/scripts/submit_dec2_2push_pipeline.sh
#
# Optional overrides (applied to ALL sbatch submissions):
#   SBATCH_PARTITION=...  SBATCH_TIME=...  SBATCH_QOS=...  SBATCH_ACCOUNT=...
# Example:
#   SBATCH_TIME=48:00:00 bash namo_cpp/scripts/submit_dec2_2push_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${NAMO_ROOT}/.." && pwd)"
SAGE_ROOT="${ROOT}/sage_learning"

MASKGEN_SCRIPT="${SCRIPT_DIR}/run_mask_generation_and_convert_dec2_2push.sh"

TRAIN_SCRIPTS=(
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_film_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_film_dec2_2push_mean_std.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_cross_attention_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_multiscale_cross_attention_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_channelwise_cross_attention_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_fusion_cross_attention_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_film_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_cross_attention_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_multiscale_cross_attention_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_channelwise_cross_attention_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_fusion_cross_attention_coordgrid_dec2_2push.sh"
)

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node." >&2
  exit 1
fi

SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--partition="${SBATCH_PARTITION}")
fi
if [[ -n "${SBATCH_TIME:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--time="${SBATCH_TIME}")
fi
if [[ -n "${SBATCH_QOS:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--qos="${SBATCH_QOS}")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--account="${SBATCH_ACCOUNT}")
fi

if [[ ! -f "$MASKGEN_SCRIPT" ]]; then
  echo "ERROR: maskgen script not found: $MASKGEN_SCRIPT" >&2
  exit 1
fi

for script in "${TRAIN_SCRIPTS[@]}"; do
  if [[ ! -f "$script" ]]; then
    echo "ERROR: training script not found: $script" >&2
    exit 1
  fi
done

RAW_DIR="/common/users/shared/robot_learning/dm1487/namo/datasets/raw_data/dec2/aug9_envs/2_push_train"
MANIFEST="/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered.txt"
if [[ ! -d "$RAW_DIR" ]]; then
  echo "ERROR: raw data dir not found: $RAW_DIR" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: eval manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Preflight:"
echo "  Raw dir:   $RAW_DIR"
echo "  Manifest:  $MANIFEST"
echo "  PKLs:      $(find "$RAW_DIR" -type f -name '*_results.pkl' | wc -l | tr -d ' ')"
if ((${#SBATCH_EXTRA_ARGS[@]})); then
  echo "  sbatch:    ${SBATCH_EXTRA_ARGS[*]}"
fi
echo ""

mask_out="$(sbatch "${SBATCH_EXTRA_ARGS[@]}" "$MASKGEN_SCRIPT")"
mask_jobid="$(echo "$mask_out" | awk '{print $4}')"
if [[ -z "$mask_jobid" ]]; then
  echo "ERROR: failed to parse maskgen job id from: $mask_out" >&2
  exit 1
fi
echo "Submitted maskgen+convert job: ${mask_jobid}"

echo ""
echo "Submitting train+eval jobs (afterok:${mask_jobid}):"

for script in "${TRAIN_SCRIPTS[@]}"; do
  out="$(sbatch "${SBATCH_EXTRA_ARGS[@]}" --dependency=afterok:${mask_jobid} "$script")"
  jobid="$(echo "$out" | awk '{print $4}')"
  echo "  ${jobid}  $(basename "$script")"
done

echo ""
echo "Done. Conversion job: ${mask_jobid}"
