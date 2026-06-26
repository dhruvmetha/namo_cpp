#!/bin/bash
# pull_from_amarel.sh — run ON ilab (it can ssh OUT to Amarel; Amarel can't reach ilab). Pulls the data to run
# eval (the gate) and the clean re-run. ONE rsync per dir — a grouped multi-path remote triggers rsync-3.2's
# "rejecting unrequested file-list name" error. Run AFTER `source env.ilab.sh`.
#   bash scripts/portability/pull_from_amarel.sh eval     # just the eval/gate data
#   bash scripts/portability/pull_from_amarel.sh train    # just the re-run training data
#   bash scripts/portability/pull_from_amarel.sh          # both (default)
set -euo pipefail
: "${NAMO_SCRATCH:?source env.ilab.sh first}"
REMOTE=${REMOTE:-dm1487@amarel.rutgers.edu}; SRC=/scratch/dm1487
pull(){ local rel="$1"; local dst="$NAMO_SCRATCH/$(dirname "$rel")"; mkdir -p "$dst"; echo "== $rel =="; rsync -avhP "$REMOTE:$SRC/$rel" "$dst/"; }
GROUP=${1:-all}
if [ "$GROUP" = eval ] || [ "$GROUP" = all ]; then        # --- EVAL / GATE (~2.7G) ---
  pull datasets/namo_testset_v1                            # 2.0G  test labels
  pull datasets/car_envs/v3/test                          # 130M  test scene XMLs (labels point here)
  pull manifests                                          # 477M  scene-list manifests
  pull mujoco/mujoco-3.2.7                                 # 4.5M  MuJoCo (also build-time)
  pull eval/exhaustive_pairmap_pure2.pkl                  # 42M   Stage-0 rank analysis (optional)
  pull sage_outputs/scorer/qfull_nohz_v3_v4hq_s1          # ~53M  NoHz-v3 gate-baseline ckpt
fi
if [ "$GROUP" = train ] || [ "$GROUP" = all ]; then       # --- RE-RUN TRAINING (~3.5G) ---
  pull h5/v4_hq_h2_scorer                                  # 1.6G  setup data (relabel target)
  pull h5/v4_hq_m2b_scorer                                 # 1.3G  1-push openers
  pull h5/v4_hq_onepush_h2_aug                             # 417M  aug
  pull h5/v4_hq_exit_finish_valid                         # 92M   v3 finish
  pull h5/v4_hq_exit_finish                                # 55M   v3 finish (orig)
  pull datasets/v4_hq_h2/labels_exhaustive_pure2push.json # relabel key (frac_first_push)
fi
echo "=== pull ($GROUP) done. NEXT: bash scripts/portability/rewrite_paths.sh ==="
