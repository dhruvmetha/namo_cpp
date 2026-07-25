#!/usr/bin/env bash
# ONE-COMMAND canonical testset eval (both tiers) for a pi ckpt.
#   scripts/rl_loop/eval_testset.sh <pi.ckpt> <out_prefix> [--no-wait]
# Submits 1push + 2push reactive arrays on the iLab cluster (shard grid derived from each
# key's episode count — never hand-sized), blocks until shards land, runs the KEYED aggregators,
# prints both tables. Frequent operations get tools, not rituals [USER 2026-07-08].
set -euo pipefail
CKPT=$1; PREFIX=$2; NOWAIT=${3:-}
R=/common/users/dm1487/scratch_namo/rl_runs
PY=/common/users/dm1487/envs/mjxrl/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ES() { PYTHONPATH="$REPO/build_python:$REPO/python:${PYTHONPATH:-}" "$PY" -m namo.eval_sets "$1"; }
ONEPUSH_KEY=$(ES onepush_manifest); PURE2PUSH_KEY=$(ES pure2push_manifest); DIVISIONS_KEY=$(ES pure2push_divisions)
EPS_PER_SHARD=70

submit() { # tier key maxp
  local tier=$1 key=$2 maxp=$3
  local n_eps; n_eps=$($PY -c "import json;k=json.load(open('$key'));print(sum(len(v) for v in k.values()) if isinstance(k,dict) else len(k))")
  local nsh=$(( (n_eps + EPS_PER_SHARD - 1) / EPS_PER_SHARD ))
  local out=$R/testset_${PREFIX}_${tier}
  mkdir -p "$out"
  local jid; jid=$(CKPT=$CKPT OUT_DIR=$out KEY=$key MAXP=$maxp SHARD=$EPS_PER_SHARD \
    sbatch --parsable --array=0-$((nsh-1)) "$HERE/testset_reactive.slurm")
  echo "$tier: job $jid ($nsh shards, $n_eps eps) -> $out" >&2
  echo "$out $nsh $tier"
}

read O1 N1 T1 <<< "$(submit 1p $ONEPUSH_KEY 1)"
read O2 N2 T2 <<< "$(submit 2p $PURE2PUSH_KEY 2)"
[ "$NOWAIT" = "--no-wait" ] && exit 0

for spec in "$O1 $N1" "$O2 $N2"; do
  read out nsh <<< "$spec"
  until [ "$(ls "$out"/shard_*.jsonl 2>/dev/null | wc -l)" -ge "$nsh" ]; do sleep 60; done
done
echo "== 1push =="; $PY "$HERE/agg_testset_onepush.py" --leaf-glob "$O1/shard_*.jsonl" --out "$O1/agg.json" | $PY -c "import json,sys;d=json.load(sys.stdin);print(d['by_division'])"
echo "== 2push =="; $PY "$HERE/agg_testset_reactive.py" --leaf-glob "$O2/shard_*.jsonl" --divisions $DIVISIONS_KEY --out "$O2/agg.json" | tail -1
