#!/bin/bash
# Sharded HY5U or uniform-random Full-NAMO evaluation for exact-N-hop XMLs.
# Budget scope defaults to keyhole; set SIM_BUDGET_SCOPE=full_problem for a shared cap.
#
# Example:
#   MANIFEST=/scratch/.../xmls.txt OUT=/scratch/.../eval N_SHARDS=20 PRIOR=model \
#     PATH_LENGTH=3 CKPT=/cache/.../HY5U_s2.ckpt \
#     sbatch --array=0-19 scripts/slurm/multihop_aug9_eval.slurm
#SBATCH --job-name=mh2_eval
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

if [[ -n "${CAMPAIGN_ENV:-}" ]]; then
  source "$CAMPAIGN_ENV"
fi

: "${MANIFEST:?set MANIFEST to the generated XML manifest}"
: "${OUT:?set OUT to the sharded evaluation output root}"
: "${N_SHARDS:?set N_SHARDS to the submitted array length}"
PRIOR=${PRIOR:-model}
checkpoint_args=()
if [[ "$PRIOR" == "model" ]]; then
  : "${CKPT:?set CKPT to the HY5U checkpoint}"
  checkpoint_args+=(--scorer-ckpt "$CKPT")
fi
SEARCH_SEED=${SEARCH_SEED:-42}
SHUFFLE_SEED=${SHUFFLE_SEED:-$SEARCH_SEED}
# Exact initial region-path length of the pool being evaluated. Defaults to 2 so every
# prior two-hop invocation reproduces unchanged; set to 3 or 4 for the deeper pools.
PATH_LENGTH=${PATH_LENGTH:-2}

NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
task_id=${SLURM_ARRAY_TASK_ID:-0}
total=$(wc -l < "$MANIFEST")
start=$((task_id * total / N_SHARDS))
end=$(((task_id + 1) * total / N_SHARDS))

if [ "$start" -ge "$end" ]; then
  echo "EVAL empty shard task=$task_id range=[$start,$end) total=$total"
  exit 0
fi

cd "$NAMO_REPO"
if [[ -n "${NAMO_ACTIVATE_SH:-}" ]]; then
  source "$NAMO_ACTIVATE_SH"
elif [ -f "$NAMO_REPO/env.ilab.sh" ] && [[ "$NAMO_REPO" == /common/* ]]; then
  source env.ilab.sh
else
  source env.amarel.sh
fi
export PYTHONPATH="${NAMO_BINDINGS_DIR:-$NAMO_REPO/build_python}:$NAMO_REPO/python:${PYTHONPATH:-}"
if [[ -n "${RUNTIME_HASHES:-}" ]]; then
  sha256sum --check "$RUNTIME_HASHES"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
PY=${NAMO_PYTHON:?environment did not define NAMO_PYTHON}
WORKERS=${WORKERS:-1}
CONFIG=${NAMO_CONFIG_FILE:-$NAMO_REPO/config/namo_config_complete_skill15_car_1x.yaml}
shard_out="$OUT/shard_$(printf '%04d' "$task_id")"
if [[ -e "$shard_out" ]]; then
  echo "refusing to overwrite output: $shard_out" >&2
  exit 1
fi
mkdir -p "$shard_out" "$NAMO_REPO/logs"
sed -n "$((start + 1)),${end}p" "$MANIFEST" > "$shard_out/input.txt"
audit_args=()
if [ "${AUDIT_NEXT_KEYHOLE_REACHABILITY:-0}" = "1" ]; then
  audit_args+=(--audit-next-keyhole-reachability)
fi
preserve_args=()
if [ "${PRESERVE_NEXT_KEYHOLE_ACCESS:-0}" = "1" ]; then
  preserve_args+=(--preserve-next-keyhole-access)
fi
limit_args=()
if [ -n "${INPUT_LIMIT:-}" ]; then
  limit_args+=(--limit "$INPUT_LIMIT")
fi

echo "EVAL start $(date) host=$(hostname) task=$task_id range=[$start,$end) total=$total prior=$PRIOR verifier_seed=$SEARCH_SEED shuffle_seed=$SHUFFLE_SEED hop=$PATH_LENGTH"
"$PY" -m namo.solvability_runner \
  --manifest "$shard_out/input.txt" \
  --path-length "$PATH_LENGTH" \
  --config-file "$CONFIG" \
  --goal-strategy scorer \
  --region-max-chain-depth 2 \
  --primitive-data-dir "$NAMO_REPO/data" \
  --primitive-prefix 1x_car_d5_ \
  --region-success-min-reachable 20 \
  --goals-per-region 100 \
  --seed "$SEARCH_SEED" \
  --shuffle-seed "$SHUFFLE_SEED" \
  --simulation-budget "${SIM_BUDGET:-900}" \
  --simulation-budget-scope "${SIM_BUDGET_SCOPE:-keyhole}" \
  --local-search best_first \
  --best-first-prior "$PRIOR" \
  "${checkpoint_args[@]}" \
  --ml-device "${ML_DEVICE:-cpu}" \
  "${audit_args[@]}" \
  "${preserve_args[@]}" \
  "${limit_args[@]}" \
  --workers "$WORKERS" \
  --output-dir "$shard_out"
"$PY" - "$shard_out" "${INPUT_LIMIT:-$((end - start))}" "${SIM_BUDGET:-900}" "${SIM_BUDGET_SCOPE:-keyhole}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected, cap = map(int, sys.argv[2:4])
summary = json.loads((root / "summary.json").read_text())
rows = [
    json.loads(line)
    for name in ("solved.jsonl", "unsolved.jsonl")
    for line in (root / name).read_text().splitlines()
    if line.strip()
]
assert summary["input_env_count"] == summary["selected_env_count"] == expected == len(rows)
assert len({row["xml_path"] for row in rows}) == expected
errors = [row for row in rows if row.get("failure_kind") == "runner_exception"]
assert not errors, f"runner exceptions: {errors[:1]}"
if sys.argv[4] == "full_problem":
    assert all(isinstance(row.get("simulation_budget_used"), int)
               and 0 <= row["simulation_budget_used"] <= cap for row in rows)
print(f"Verified {expected} results: {summary['solved_count']} solved")
PY
echo "EVAL done $(date) task=$task_id"
