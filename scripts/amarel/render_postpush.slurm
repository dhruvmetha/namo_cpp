#!/bin/bash
# Full-scale OFFLINE render of post-push s1 states (horizon-Q v2 OOD data).
# Slices a PKL manifest by SLURM_ARRAY_TASK_ID, then fans the shard out across the node's CPUs
# (render_postpush_from_state.py is single-process; we run one per CPU on a sub-slice). CPU job — the
# render is matplotlib, NO GPU, NO MuJoCo (reads the collector's byte-saved state_observation).
#
# Required env at submit:
#   PKL_MANIFEST   full list of v2 pkls (find .../pkls_2push_v2 -name '*_results.pkl')
#   SHARD_SIZE     pkls per array task (match --array=0-(ceil(N/SHARD_SIZE)-1))
#   OUTPUT_DIR     npz output root; each shard -> $OUTPUT_DIR/shard_$TASK/
# Optional:
#   MAX_PER_EPISODE  cap post-push s1 per episode (0=all; >0 bounds total volume) [USER tunable]
#   RP_WORKERS       per-shard parallel render procs (default $SLURM_CPUS_PER_TASK)
#SBATCH --partition=main,main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rppush-%A_%a.out
#SBATCH --error=logs/rppush-%A_%a.err

set -euo pipefail
mkdir -p "$NAMO_LOGS"

NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"
: "${MJ_PATH:?source env.<machine>.sh first}"
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$SAGE_REPO"

: "${PKL_MANIFEST:?must set PKL_MANIFEST}"
: "${SHARD_SIZE:?must set SHARD_SIZE}"
: "${OUTPUT_DIR:?must set OUTPUT_DIR}"
MAX_PER_EPISODE=${MAX_PER_EPISODE:-0}
nproc_shard=${RP_WORKERS:-$SLURM_CPUS_PER_TASK}

task_id=${SLURM_ARRAY_TASK_ID:-0}
start_line=$(( task_id * SHARD_SIZE + 1 ))
end_line=$(( (task_id + 1) * SHARD_SIZE ))
shard_dir="$OUTPUT_DIR/shard_${task_id}"
shard_list="$shard_dir/pkls.txt"
mkdir -p "$shard_dir"
sed -n "${start_line},${end_line}p" "$PKL_MANIFEST" > "$shard_list"
n_lines=$(wc -l < "$shard_list")

echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID:-NA}_${task_id} slice=[$start_line,$end_line] pkls=$n_lines"
echo "out=$shard_dir max_per_episode=$MAX_PER_EPISODE procs=$nproc_shard"
[ "$n_lines" -eq 0 ] && { echo "empty slice"; exit 0; }

# fan out: split the shard list into nproc_shard sub-lists, one renderer per CPU
split -n l/"$nproc_shard" -d --additional-suffix=.txt "$shard_list" "$shard_dir/sub_"
pids=()
for sub in "$shard_dir"/sub_*.txt; do
    [ -s "$sub" ] || continue
    python "$NAMO_REPO/scripts/pipeline/render_postpush_from_state.py" \
        --pkl-list "$sub" --output-dir "$shard_dir/npz" --max-per-episode "$MAX_PER_EPISODE" &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "shard done. npz count: $(find "$shard_dir/npz" -name '*.npz' | wc -l)  fail=$fail"
exit $fail
