#!/bin/bash
# Config-driven cascade driver. Reads config/corpora/<CORPUS>.yaml via
# scripts/corpus.py for ALL paths + per-phase params (depth / K / mining /
# seed sweep). Waits for phase-1, then mines + runs phases 2 -> 3 -> 4 -> 5A-E
# inside one SLURM allocation (survives client disconnects).
#
# This script is corpus-agnostic: a new corpus = a new config/corpora/<id>.yaml,
# no new driver. (Filename keeps the v3_ prefix only because docs reference it.)
#
# Submit:
#   CORPUS=v3_aug9 PHASE1_JID=<phase-1 array id> sbatch scripts/amarel/v3_cascade_driver.slurm
# Print the plan without touching SLURM (no side effects):
#   CORPUS=v3_aug9 PHASE1_JID=0 DRYRUN=1 NAMO_PY=<python> bash scripts/amarel/v3_cascade_driver.slurm
#
# Phase-1 (exhaustive) is submitted separately and must write to PHASE1_OUT
# (run `corpus.py <id> paths` to see it).
#
#SBATCH --job-name=cascade-driver
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dm1487/logs/cascade-driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/cascade-driver-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
cd "$NAMO_REPO"

CORPUS=${CORPUS:-${CORPUS_TAG:-}}        # CORPUS_TAG still accepted (back-compat)
: "${CORPUS:?set CORPUS (corpus id; needs config/corpora/<id>.yaml)}"
: "${PHASE1_JID:?set PHASE1_JID (phase-1 array job id to wait on)}"
DRYRUN=${DRYRUN:-0}

[ "$DRYRUN" = 1 ] || source scripts/amarel/activate.sh
PY=${NAMO_PY:-python}
W=${SLURM_CPUS_PER_TASK:-16}
COLLECT="$NAMO_REPO/scripts/amarel/v3_cascade_collect.slurm"

# --- resolve everything from the corpus config (single source of truth) ---
eval "$("$PY" scripts/corpus.py "$CORPUS" paths)"       # ENVS, PHASE1_OUT..PHASE5_OUT, PHASE*_MANIFEST, LAYOUT
eval "$("$PY" scripts/corpus.py "$CORPUS" phasevars)"   # P2_DEPTH P2_K P2_MINE ... P5_SEEDS P5_REASONS

shards_for()  { "$PY" -c "import math;print(min(60,max(1,math.ceil($1/1500))))"; }
mine_dir()    { local v="${1^^}_OUT"; echo "${!v}"; }   # phase2 -> $PHASE2_OUT

wait_for_job() {
    [ "$DRYRUN" = 1 ] && { echo "  DRYRUN: would wait for job $1"; return; }
    local j=$1; while [ "$(squeue -j "$j" -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
}

# mine_single <out_manifest> <mine_phase> [extra args...]   (build_phase2_manifest.py)
mine_single() {
    local outman=$1 phase=$2; shift 2; local dir; dir=$(mine_dir "$phase")
    if [ "$DRYRUN" = 1 ]; then echo "  DRYRUN mine: phase2_manifest --phase1-dir=$dir $* -> $outman"; return; fi
    "$PY" scripts/build_phase2_manifest.py --phase1-dir "$dir" --output "$outman" --workers "$W" "$@"
}

# mine_union <out_manifest>   (build_phase4_manifest.py over P4_MINE = "phase1 phase2 phase3")
mine_union() {
    local outman=$1; set -- $P4_MINE; local d1 d2 d3
    d1=$(mine_dir "$1"); d2=$(mine_dir "$2"); d3=$(mine_dir "$3")
    if [ "$DRYRUN" = 1 ]; then echo "  DRYRUN mine: phase4_manifest --phase1=$d1 --phase2=$d2 --phase3=$d3 -> $outman"; return; fi
    "$PY" scripts/build_phase4_manifest.py --phase1-dir "$d1" --phase2-dir "$d2" --phase3-dir "$d3" \
        --output "$outman" --workers "$W"
}

# submit <manifest> <outdir> <depth> <K> <name> [GOAL_SEED]   (blocks until done)
submit() {
    local manifest=$1 outdir=$2 depth=$3 K=$4 name=$5 seed=${6:-}
    local n shards last jid
    if [ "$DRYRUN" = 1 ]; then
        echo "  DRYRUN submit: $name depth=$depth K=$K seed=${seed:-none} -> $outdir"; return
    fi
    n=$(wc -l < "$manifest")
    if [ "$n" -eq 0 ]; then echo "  [$name] empty manifest, skipping"; return; fi
    shards=$(shards_for "$n"); last=$(( shards - 1 ))
    echo "  [$name] envs=$n shards=$shards depth=$depth K=$K -> $outdir"
    jid=$(MANIFEST="$manifest" OUTPUT_DIR="$outdir" DEPTH="$depth" KROLL="$K" NUM_SHARDS="$shards" \
        ${seed:+GOAL_SEED="$seed"} \
        sbatch --array=0-"$last" --job-name="$name" "$COLLECT" | awk '{print $4}')
    echo "  [$name] job=$jid  $(date -Is)"
    wait_for_job "$jid"
}

echo "===== cascade driver: $CORPUS (layout=$LAYOUT, job ${SLURM_JOB_ID:-NA}, dryrun=$DRYRUN) ====="
echo "waiting for phase-1 ($PHASE1_JID) ..."; date -Is
wait_for_job "$PHASE1_JID"

# ---- Phase 2: mine phase-1 fails ----
echo "===== phase 2 ====="; date -Is
mine_single "$PHASE2_MANIFEST" "$P2_MINE"
submit "$PHASE2_MANIFEST" "$PHASE2_OUT" "$P2_DEPTH" "$P2_K" "${CORPUS}-p2"

# ---- Phase 3: mine phase-2 fails ----
echo "===== phase 3 ====="; date -Is
mine_single "$PHASE3_MANIFEST" "$P3_MINE"
submit "$PHASE3_MANIFEST" "$PHASE3_OUT" "$P3_DEPTH" "$P3_K" "${CORPUS}-p3"

# ---- Phase 4: union of partial-fail (p1,p2,p3) ----
echo "===== phase 4 ====="; date -Is
mine_union "$PHASE4_MANIFEST"
submit "$PHASE4_MANIFEST" "$PHASE4_OUT" "$P4_DEPTH" "$P4_K" "${CORPUS}-p4"

# ---- Phase 5A-E: seed sweep on phase-4 residual ----
echo "===== phase 5 ====="; date -Is
mine_single "$PHASE5_MANIFEST" "$P5_MINE" --include-reasons $P5_REASONS
letters=(A B C D E); i=0; jids=""
for s in $P5_SEEDS; do
    p=${letters[$i]}; i=$((i + 1))
    if [ "$DRYRUN" = 1 ]; then
        echo "  DRYRUN submit: ${CORPUS}-p5${p} depth=$P5_DEPTH K=$P5_K seed=$s -> ${PHASE5_OUT}${p}"
        continue
    fi
    N5=$(wc -l < "$PHASE5_MANIFEST"); [ "$N5" -gt 0 ] || { echo "  phase5 manifest empty, skipping"; break; }
    shards=$(shards_for "$N5"); last=$(( shards - 1 ))
    jid=$(MANIFEST="$PHASE5_MANIFEST" OUTPUT_DIR="${PHASE5_OUT}${p}" DEPTH="$P5_DEPTH" KROLL="$P5_K" \
        NUM_SHARDS="$shards" GOAL_SEED="$s" \
        sbatch --array=0-"$last" --job-name="${CORPUS}-p5${p}" "$COLLECT" | awk '{print $4}')
    echo "  phase5$p (seed=$s): job $jid"; jids="$jids,$jid"
done
if [ "$DRYRUN" != 1 ] && [ -n "$jids" ]; then
    jids=${jids#,}; echo "  waiting for phase-5 passes: $jids"
    while [ "$(squeue -j "$jids" -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
fi

echo "===== $CORPUS cascade COMPLETE @ $(date -Is) ====="
[ "$DRYRUN" = 1 ] || "$PY" scripts/corpus.py "$CORPUS" stamp || true   # self-describing lineage (meta.json)
echo "PKL counts:"
for v in PHASE1_OUT PHASE2_OUT PHASE3_OUT PHASE4_OUT; do
    d=${!v}
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | wc -l)"
done
for p in A B C D E; do
    d="${PHASE5_OUT}${p}"
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | wc -l)"
done
