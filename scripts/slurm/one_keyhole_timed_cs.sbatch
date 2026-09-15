#!/bin/bash
# Timed one-keyhole runs on one CS node. Starts SLOTS single-threaded run_one_keyhole_frozen.py
# processes, each pinned to the first CPU of its own L3 cache group, so no two share a cache.
# Every node in a campaign must have the same CPU model (EXPECT_CPU); the runner shuffles the
# (problem, arm) units with one seed and records host, CPU model, pinned CPU and load per unit.
#
# Usage, one job per node; slots are numbered across the whole campaign:
#   nodes=(rlab3 rlab4 ilab3)
#   for i in 0 1 2; do
#     sbatch --nodelist=${nodes[i]} --output=<out>/logs/%x_%j.out --export=ALL,NAMO_REPO=<pinned checkout>,\
#   OUT=<out>,ARMS=<arms.json>,PROBLEMS=<input/one_keyhole>,CONFIG=<namo_config.yaml>,\
#   POST_RESTORE=<post_restore_problem_ids.json>,EXPECT_SOURCE_SHA256=<sha>,\
#   EXPECT_CPU="AMD EPYC 7352 24-Core Processor",FIRST_SLOT=$((16*i)),SLOTS=16,NSLOTS=48 \
#       scripts/slurm/one_keyhole_timed_cs.sbatch
#   done
# EXTRA_ARGS (for example "--only-index 0 --only-index 450") passes through to the runner for smokes.
#SBATCH --job-name=kh600-timed
#SBATCH --partition=unlimited
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -uo pipefail
: "${NAMO_REPO:?}" "${OUT:?}" "${ARMS:?}" "${PROBLEMS:?}" "${CONFIG:?}" "${POST_RESTORE:?}"
: "${EXPECT_SOURCE_SHA256:?}" "${EXPECT_CPU:?}" "${FIRST_SLOT:?}" "${SLOTS:?}" "${NSLOTS:?}"
cd "$NAMO_REPO" && source env.ilab.sh >/dev/null 2>&1

model=$(lscpu | sed -n 's/^Model name: *//p')
[ "$model" = "$EXPECT_CPU" ] || { echo "wrong CPU on $(hostname): $model"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "checkout $NAMO_REPO is not clean"; exit 1; }
mapfile -t cpus < <(lscpu -p=CPU,CACHE | grep -v '^#' | awk -F, '{split($2, c, ":"); if (!(c[4] in seen)) {seen[c[4]] = 1; print $1}}')
[ "${#cpus[@]}" -ge "$SLOTS" ] || { echo "$(hostname) has only ${#cpus[@]} cache groups"; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT/slot_logs"
echo "start $(date) host=$(hostname) cpu=$model commit=$(git rev-parse HEAD) cpus=${cpus[*]:0:$SLOTS}"
pids=()
for ((i = 0; i < SLOTS; i++)); do
  slot=$((FIRST_SLOT + i))
  taskset -c "${cpus[$i]}" "$NAMO_PYTHON" scripts/pipeline/run_one_keyhole_frozen.py \
    --problems "$PROBLEMS" --config "$CONFIG" --arms "$ARMS" --post-restore-ids "$POST_RESTORE" \
    --out "$OUT/rows" --expect-source-sha256 "$EXPECT_SOURCE_SHA256" \
    --timed --workers 1 --shuffle-seed 20260915 --shard "$slot" --nshards "$NSLOTS" ${EXTRA_ARGS:-} \
    > "$OUT/slot_logs/slot_$(printf %02d "$slot").log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
echo "done $(date) failed=$failed"
exit "$failed"
