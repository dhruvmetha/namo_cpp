#!/usr/bin/env bash
# Exhaustive depth-2 region_opening labelling of a scene manifest, fanned across CS iLab boxes.
#
# All CS boxes share /common, so a shard is just a slice of the manifest and there is nothing to
# copy. Reads BOXES as "host:workers" pairs and splits the manifest in proportion to workers, so a
# 256-core box takes a bigger slice than a 48-core one and they finish together.
#
# SEARCH_TIMEOUT is 600, not the config's 1800. `region_selection_strategy: cost_first` exhausts
# every depth-1 push before expanding any depth-2 chain, so a timeout truncates only the 2-push
# part of the trial log, which build_2push_validset.py already records as censored. The
# solve_rate_1push that difficulty tiers are cut on stays exact. Measured on the 60-scene pilot at
# 1800: p50 finished in 1.2 min but p100 took 20, and 4 scenes never finished at all, so the tail
# was most of the wall time.
#
#   BOXES="rlab7:128 rlab5:64 rlab6:32 rlab4:48 ilab3:48" \
#     scripts/pipeline/label_scenes_sharded.sh <manifest.txt> <out-root>
set -euo pipefail

MANIFEST="${1:?usage: label_scenes_sharded.sh <manifest.txt> <out-root>}"
OUT_ROOT="${2:?usage: label_scenes_sharded.sh <manifest.txt> <out-root>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOXES="${BOXES:-rlab7:128 rlab5:64 rlab6:32 rlab4:48 ilab3:48}"
CONFIG="${CONFIG:-python/namo/data_collection/region_opening_exhaustive_2push_multihop_car.yaml}"
SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-600}"

N=$(wc -l < "$MANIFEST")
TOTAL_W=0
for b in $BOXES; do TOTAL_W=$(( TOTAL_W + ${b##*:} )); done
echo "manifest=$N scenes  boxes=[$BOXES]  total_workers=$TOTAL_W  timeout=${SEARCH_TIMEOUT}s"

mkdir -p "$OUT_ROOT"
off=0
for b in $BOXES; do
    host="${b%%:*}"; w="${b##*:}"
    take=$(( N * w / TOTAL_W ))
    [ "$take" -lt 1 ] && take=1
    end=$(( off + take ))
    [ "$end" -gt "$N" ] && end=$N
    shard="$OUT_ROOT/$host"
    mkdir -p "$shard/pkls"
    sed -n "$(( off + 1 )),${end}p" "$MANIFEST" > "$shard/manifest.txt"
    cnt=$(wc -l < "$shard/manifest.txt")

    # OpenBLAS spawns one thread per core per process; at 128 workers that blows CS's ulimit -u.
    ssh -o BatchMode=yes "$host.cs.rutgers.edu" "cd $REPO && source env.ilab.sh >/dev/null 2>&1 && \
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
               NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 && \
        PYTHONPATH=\"$REPO/build_python:$REPO/python\" nohup python -m namo.data_collection.modular_parallel_collection \
          --config-yaml $CONFIG --manifest $shard/manifest.txt --output-dir $shard/pkls \
          --start-idx 0 --end-idx $cnt --workers $w --search-timeout $SEARCH_TIMEOUT \
          > $shard/collect.log 2>&1 < /dev/null &" >/dev/null 2>&1
    echo "  $host: scenes $(( off + 1 ))-$end ($cnt) with $w workers -> $shard"
    off=$end
done
echo "launched. poll: find $OUT_ROOT -name '*.pkl' | wc -l   (target $N)"
