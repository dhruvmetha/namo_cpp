#!/usr/bin/env bash
set -euo pipefail

# Submits TRAIN+EVAL for the top-3 1-push architectures on dec2 2-push tasks.
#
# Assumes you already have:
#   /common/users/shared/robot_learning/dm1487/namo/datasets/h5_files/se2/dec2/aug9_envs/2_push_train_srcsplit/training_data.h5
# and ideally:
#   .../stats_max_abs.json
#
# If stats_max_abs.json is missing, this script submits a small stats job first and then
# submits the train+eval jobs with an afterok dependency.
#
# Evaluation uses the 2-push filtered manifest (with per-line skip rules) and
# region_max_chain_depth=2. By default it runs ML-only (no brute-force fallback).
# To enable hybrid fallback, submit with: EVAL_HYBRID=1 bash <this_script>
#
# Usage:
#   bash namo_cpp/scripts/submit_dec2_2push_top3_1push_archs.sh
#
# Optional overrides (applied to ALL sbatch submissions):
#   SBATCH_PARTITION=...  SBATCH_TIME=...  SBATCH_QOS=...  SBATCH_ACCOUNT=...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${NAMO_ROOT}/.." && pwd)"
SAGE_ROOT="${ROOT}/sage_learning"

H5_ROOT="/common/users/shared/robot_learning/dm1487/namo/datasets/h5_files/se2/dec2/aug9_envs/2_push_train_srcsplit"
H5_FILE="${H5_ROOT}/training_data.h5"
STATS_MAXABS="${H5_ROOT}/stats_max_abs.json"
MANIFEST="/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered.txt"

# Top-3 architectures based on prior 1-push runs (checkpoint names listed by user):
#   - multiscale_cross_attn (flow matching)
#   - vdiff_film_coordgrid (vector diffusion)
#   - vdiff_ms_cross_attn_coordgrid (vector diffusion)
TRAIN_SCRIPTS=(
  "${SAGE_ROOT}/job_scripts/train_eval_flow_matching_multiscale_cross_attention_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_film_coordgrid_dec2_2push.sh"
  "${SAGE_ROOT}/job_scripts/train_eval_vector_diffusion_multiscale_cross_attention_coordgrid_dec2_2push.sh"
)

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node." >&2
  exit 1
fi

EVAL_HYBRID="${EVAL_HYBRID:-0}"

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
# Ensure eval mode is explicit (prevents accidental EVAL_HYBRID=1 leakage from the submit shell).
SBATCH_EXTRA_ARGS+=(--export=ALL,EVAL_HYBRID="${EVAL_HYBRID}")

for script in "${TRAIN_SCRIPTS[@]}"; do
  if [[ ! -f "$script" ]]; then
    echo "ERROR: training script not found: $script" >&2
    exit 1
  fi
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: eval manifest not found: $MANIFEST" >&2
  exit 1
fi

if [[ ! -f "$H5_FILE" ]]; then
  echo "ERROR: missing H5 training file: $H5_FILE" >&2
  echo "Run: sbatch ${NAMO_ROOT}/scripts/run_mask_generation_and_convert_dec2_2push.sh" >&2
  exit 1
fi

dependency_arg=()
if [[ ! -f "$STATS_MAXABS" ]]; then
  echo "stats_max_abs.json missing; submitting stats job first..."
  stats_out="$(
    sbatch "${SBATCH_EXTRA_ARGS[@]}" \
      --job-name=stats_dec2_2push_maxabs \
      --output="${SAGE_ROOT}/job_scripts/slurm_logs/stats_dec2_2push_maxabs_%j.out" \
      --error="${SAGE_ROOT}/job_scripts/slurm_logs/stats_dec2_2push_maxabs_%j.err" \
      --time=02:00:00 \
      --cpus-per-task=4 \
      --mem=32G \
      --wrap "set -e; source /common/home/tdn39/.virtualenvs/mujoco/bin/activate; python ${SAGE_ROOT}/scripts/compute_dataset_stats.py --h5_file '${H5_FILE}' --output '${STATS_MAXABS}' --mode max_abs"
  )"
  stats_jobid="$(echo "$stats_out" | awk '{print $4}')"
  if [[ -z "$stats_jobid" ]]; then
    echo "ERROR: failed to parse stats job id from: $stats_out" >&2
    exit 1
  fi
  echo "Submitted stats job: ${stats_jobid}"
  dependency_arg=(--dependency="afterok:${stats_jobid}")
fi

echo ""
echo "Preflight:"
echo "  H5:        $H5_FILE"
echo "  Stats:     $STATS_MAXABS"
echo "  Manifest:  $MANIFEST"
echo "  Eval:      $([[ \"$EVAL_HYBRID\" == \"1\" ]] && echo \"hybrid\" || echo \"ml_only\")"
if ((${#dependency_arg[@]})); then
  echo "  Depends:   ${dependency_arg[*]}"
fi
if ((${#SBATCH_EXTRA_ARGS[@]})); then
  echo "  sbatch:    ${SBATCH_EXTRA_ARGS[*]}"
fi
echo ""

echo "Submitting top-3 architectures (train+eval on 2-push manifest):"
for script in "${TRAIN_SCRIPTS[@]}"; do
  out="$(sbatch "${SBATCH_EXTRA_ARGS[@]}" "${dependency_arg[@]}" "$script")"
  jobid="$(echo "$out" | awk '{print $4}')"
  echo "  ${jobid}  $(basename "$script")"
done

echo ""
echo "Done."
