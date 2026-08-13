---
type: experiment
status: idea
created: 2026-08-12
updated: 2026-08-12
metric: "gate: 2p-hard@5 > HY5U's 39.5 on the COMMON episode set (aquaman_agg_common.py); guard: 1p-hard@1 ≥ 42; V5 explicitly barred from gating."
tags:
  - experiment
  - data
  - dagger
---
# DAgger round-4 — mix-constrained collection from the HY5U lessons

Status: DRAFT, awaiting user go. Written from the lessons of [EXP-2026-08-09-crossboard-ranking](EXP-2026-08-09-crossboard-ranking.md) (isolation 2×2 → hybrid → HY5U). Nothing here is launched.

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

### CORRECTION (2026-08-13) — the selection criterion, sharpened

The stream-A description above ("episodes where the model failed") is too loose and would reproduce the family-corpus failure. Two kinds of hard board must be separated:

1. **No solution exists** — every push fails. Teaches "everything here is junk"; contains NO positive, so it gives the ranker nothing to rank toward. The family corpus over-collected these (opener-bearing roots 49-55% vs 73-76% in the old/hybrid corpora) and 1p-h@1 crashed 38 → 24.7.
2. **A solution exists but the model ranked it badly** — the simulator verifies an opener; the model placed it far down its ordering. A labeled positive sitting beside the model's confident wrong answers: the most informative example available.

**Only (2) is DAgger. (1) is merely harder data.** Stream A's filter is therefore **"model failed to find it within budget AND a verified opener exists"**, not "model failed". The opener-fraction meter stays as a symptom check, but the filter states the intent directly.

This also dissolves an apparent paradox in the evidence: the hybrid TRAINING corpus is markedly EASIER than the test set — P(opener | labeled) by depth is 11/18/23/28/31% on training roots versus 3.7/3.9/5.2/7.1/8.9% in the exhaustive test GT — yet it is the corpus that works. Easiness was never the problem; **absence of positives** was.

### ADDITION (2026-08-13) — fix the depth-selection bias at collection time

Measured on 104,420 GT boards: the model picks the DEEPEST push (index 4) **69.6%** of the time, against a 26.2% share of true openers and a 20% uniform baseline (mean depth: model 3.30, truth 2.24). Cost: its simulator calls are **1.3-1.7× more expensive** than random's, which is why the wall-clock speed-up (3.7× hard 1push) is roughly half the simulator-call speed-up (6.5×).

**⚠ CAUSE NOT ESTABLISHED — an earlier draft of this section blamed `BeamPlanner(first_depths=(4, 3, 2))` in `scripts/sandbox/scorer_beam.py`. That is WRONG: that default belongs to the EVAL/deploy-side beam planner, not to collection.** The collection planner `region_opening.py` sorts candidates by `(depth ascending, score descending)` or `(score descending, depth ascending)` (`_sort_candidates_sync`) — it does not favour deep pushes. Retracted 2026-08-13 on the user's challenge; do not propagate the (4,3,2) story.

What IS measured, and stands:
- Training corpus (hybrid roots): labeled coverage ~even across depths (19.9/20.2/20.2/20.0/19.7%), P(opener | labeled) rising 11.0 → 31.1%.
- Exhaustive test GT: labeled coverage shallow-heavy (26.0/22.5/19.2/16.7/15.7%), P(opener | labeled) rising 3.7 → 8.9%.
- So both agree deep is better by a similar RATIO (2.8× vs 2.4×), but training's absolute opener rates are 3-4× higher, and the two disagree on which depths get labeled at all.

The open question is why labeled coverage is even in training but shallow-heavy in exhaustive GT (note `r_mask` appears to be per-EDGE, marking all 5 depths reachable together, so the difference is in what was TRIED, not what was reachable). **Diagnose before designing a fix** — candidates include the goal generator's per-depth feasibility, `max_goals` truncation, and differing primitive availability. A wrong root cause here would produce a collection change that does nothing.

**Round-4 change (conditional on that diagnosis):** make the per-depth TRY distribution match feasibility, or record per-depth sampling weights so the loss can invert them. Pre-registered readout: model's top-1 depth histogram moves toward the true-opener histogram; mean seconds per simulator call falls toward random's ~0.25s; and top-1 accuracy on boards WITHOUT a depth-4 opener rises from 70.7% (2.07× random) toward the 79.3% the deep-answer boards get. [Figures corrected 2026-08-13: the earlier 7.0%/0.69× was an artifact of scoring over per-EDGE r_mask cells that include infeasible deep pushes — see the crossboard card's retraction.]

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
