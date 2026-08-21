#!/bin/bash
#SBATCH --job-name=rl_collect
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-63
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
# Collection fan-out for one RL-loop generation: one array task = one episode shard -> one pkl.
# Forward-only rollouts are CPU-bound; arm A (uniform pi0) needs no model. Real collection runs
# where the TRAIN xmls live (Amarel /scratch after a git sync) -> submit to a CPU partition there
# (main). On the CS estate submit to `unlimited` (one shard per node; see compute-resources).
#
# Submit:
#   NSHARDS=64 CONFIG=<gen_dir>/config.json OUTDIR=<gen_dir>/collect BUFFER=<run>/buffer.pkl \
#     sbatch --array=0-63 --partition=<part> scripts/rl_loop/collect.slurm
# Then harvest+train+eval:
#   python scripts/rl_loop/run_generation.py ... --pre-collected-dir <gen_dir>/collect --expected-shards $NSHARDS
set -euo pipefail

: "${REPO:=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
: "${ENV_FILE:=env.ilab.sh}"          # override with env.amarel.sh on Amarel
: "${NSHARDS:?set NSHARDS}"
: "${CONFIG:?set CONFIG (LoopConfig json)}"
: "${OUTDIR:?set OUTDIR}"
: "${BUFFER:=}"
: "${PY:=/common/users/dm1487/envs/mjxrl/bin/python}"

cd "$REPO"
# shellcheck disable=SC1090
source "$ENV_FILE"

"$PY" scripts/rl_loop/collect_shard.py \
  --config "$CONFIG" \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --n-shards "$NSHARDS" \
  --out-dir "$OUTDIR" \
  ${BUFFER:+--buffer "$BUFFER"}
