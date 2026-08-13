# DAgger round-4 collection spec (drafted 2026-08-12)

Status: DRAFT, awaiting user go. Written from the lessons of [EXP-2026-08-09-crossboard-ranking](log/EXP-2026-08-09-crossboard-ranking.md) (isolation 2×2 → hybrid → HY5U). Nothing here is launched.

## Why this round, and the one trap it must avoid

DAgger collects states where the *current* model fails, so the learner sees its own mistakes. The trap this project just paid for: **model-failure states are hard states, and hard states are opener-poor.** The measured relationship (card § POST-HOC CORRECTION) is that deploy 1-push tracks the fraction of root boards containing any verified opener — 49-55% in the family corpora (1p 21.8-24.7) versus 73-76% in old/hybrid (1p 38.1-42.1). A naive "collect where the model fails" corpus would reproduce exactly the difficulty skew that cost the 1-push crater.

So this round is **DAgger-with-a-mix-constraint**, not pure DAgger.

## Teacher

`HY5U_s2` (best seed of the best arm: 1p-h@1 46.7, 2p-h@5 38.1, @900 97.5). Rationale: a stronger teacher makes failures rarer and more informative — the states it still misses are the genuinely hard population, not noise. Checkpoint path in the card's mini-registry.

## Collection composition (the load-bearing part)

Three streams, mixed at collection time, with the opener fraction METERED PER WAVE rather than discovered afterwards:

| stream | share | source | purpose |
|---|--:|---|---|
| A. model-failure episodes | 40% | episodes where `HY5U` search failed or needed >100 sims on the canonical trace | the DAgger signal proper |
| B. opener-bearing episodes | 45% | fresh rooms, keep episodes whose root board has ≥1 verified opener | holds the 1-push ingredient; target overall opener fraction ~70% |
| C. child boards under failed pushes | (within A+B) | `region_stop_after_root_opener: false`, capped-12 finish sweeps | the V5 ingredient — proven harmless at 12.8% lie rate, so do NOT pay 3× for exhaustive |

**Pre-registered meter:** report opener-bearing root fraction after every wave. If it drops below 0.65, raise stream B's share before continuing. This is the "meter what loses exposure" law applied to the axis that actually bit us.

## Rooms

Use the ~45k pool rooms not yet consumed by any corpus (`/scratch/dm1487/antman0/gen/*`, set-diff against the union of old/family0/family1 room sets — all three are nested subsets, so the diff is well-defined). Every corpus to date re-mined the same rooms; new geometry is untested and is the only source of genuinely new diversity. Restart pool generation as filler behind the collection.

## Labels

- Root cells: standard sweep, tiers opener 1.0 / setup 0.9 / dead 0.0, untried masked.
- Setup tier stamped at 0.9 in the H5; the {0, 0.5, 1} ladder is applied at load time via `NAMO_GAMMA=0.5` (corpus stays ladder-agnostic).
- Child boards: capped-12 finish sweeps, misses recorded censored (`finish_sweep_censored`), never fake zeros.
- **Unreachable cells need no collection change** — `NAMO_UNREACH_W` derives them from `r_mask` at training time.

## Machinery (all exists, nothing new to write)

`scripts/slurm/family_collect.slurm` (wave ≤470 tasks × 350 rooms, 14 CPU / 12 workers, `--time` explicit) → `scripts/pipeline/build_rung2_h5.py --family-select` → `scripts/slurm/family_h5.slurm` (200-shard render) → `scripts/pipeline/pool_family_h5.py` (per-row chunks + lzf; gzip auto-chunk kills DataLoaders) → optionally `scripts/pipeline/build_hybrid_h5.py` to graft onto the hybrid roots.

Physics gate before launch: `scripts/check_box_sync.sh` + BUILD_INFO post-5daaed5.

## Pre-registered readings

- **Primary:** does 2p-hard@5 exceed HY5U's 39.5 on the common episode set (`aquaman_agg_common.py`)? That is the metric the campaign found hardest to move.
- **Guard:** 1p-hard@1 must not fall below 42 (the hybrid floor). If it does, the mix constraint failed and the opener meter should show it.
- **Diagnostic only:** V5. It anti-predicted deploy five times this campaign; it must not gate any decision.

## Sequencing note

Two cheaper experiments are running first (2026-08-12) because they may change what this corpus should contain: the `NAMO_UNREACH_W` dose sweep (0.1 / 0.3 / 1.0), and the same auxiliary applied to the OLD corpus (`AJ2U`). If the unreachable auxiliary proves corpus-general and dose-sensitive, its optimal setting should be fixed before spending compute on collection.
