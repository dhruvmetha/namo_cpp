#!/bin/bash
# Retire the untracked per-campaign copies on Amarel, 2026-08-09.
# Runs ON amarel. Archives (never deletes) -- namo_postprune_eval holds the only surviving
# binary that produced every registered number.
set -u
# Amarel paths come from env, never baked in (portability guard):
#   AMAREL_ROOT  remote home holding aquaman0/   AMAREL_REPO  remote namo checkout
#   AMAREL_SAGE  remote sage checkout (must match the ckpt)
AMAREL_ROOT=${AMAREL_ROOT:?set AMAREL_ROOT=<remote home holding aquaman0/>}
AMAREL_REPO=${AMAREL_REPO:?set AMAREL_REPO=<remote namo checkout>}
AMAREL_SAGE=${AMAREL_SAGE:?set AMAREL_SAGE=<remote sage checkout>}
cd /cache/home/dm1487/projects/namo || exit 1
A=_archive_2026-08-09
mkdir -p "$A"
for d in namo_bfix sage_bfix namo_postprune_eval; do
  if [ -d "$d" ]; then mv "$d" "$A/" && echo "archived $d"; else echo "skip $d (absent)"; fi
done

cat > "$A/README.md" <<'EOF'
Archived 2026-08-09, after CS and Amarel were put on identical git checkouts and a smoke
shard reproduced bit-identically on both (35 episodes / 98 sims / 35 solved).

namo_bfix            Plain copy, no .git. Ran every aquaman/arjuna eval. Superseded by
                     namo_cpp (feat/horizon-q-redesign, same commit as CS).

sage_bfix            Plain copy, no .git. Superseded by sage_learning (feat/horizon-q @
                     db75913). This directory was DELETED by the 2026-08-08 cleanup while
                     jobs still depended on it, then restored by rsync -- 288 eval tasks
                     died on "No module named 'src'" in between.

namo_postprune_eval  DO NOT DELETE. Its build_python/*.so (built 2026-07-29) is the ONLY
                     surviving copy of the binary that produced EVERY registered number in
                     horizon_q_model_registry.md. It predates 5daaed5 ("set_full_state must
                     restore ALL state"), which leaked ctrl and qacc_warmstart across
                     restores so a replayed board was not the same board twice. Identical
                     source scores 89 sims on this .so vs 98 on a current build (~10%
                     fewer). Reproducing any pre-2026-08-09 registry row needs this binary.
                     Also holds ~6169 logs from the registered wall-clock campaign.

Live from here on: namo_cpp + sage_learning -- both git checkouts, both matching CS.
Do not create per-campaign plain copies again: untracked trees became load-bearing, one was
silently deleted, and another quietly supplied stale physics for two weeks.
EOF
echo "--- live checkouts remaining ---"
ls -d */ 2>/dev/null | grep -v _archive
