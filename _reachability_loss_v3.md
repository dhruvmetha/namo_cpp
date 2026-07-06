---
type: experiment
status: idea
created: 2026-07-05
updated: 2026-07-05
metric: "TBD — NoHz-v3 confirmed trained with reachability loss OFF (no unreachable_k; bce_reachable_only=true is inert under head_mode=hl_gauss). Proposal: retrain NoHz-v3 with the M2c reachability supervision ON."
tags:
  - experiment
  - training
---
# Reachability loss for NoHz-v3 — was it on, and does turning it on help?

**Plain-English framing.** "Reachability loss" = teaching the model to give a *low* score to pushes the robot can't even reach, so it doesn't hallucinate value on impossible pushes. The question: was it on when we trained the deployed model (NoHz-v3), and if not, does turning it on help — especially the hard-1push rare-opener ranking?

## Finding (the check) [CLAUDE 2026-07-05]

**NoHz-v3 (`qfull_nohz_v3_v4hq`) was trained with the reachability loss OFF.** Verified against the exact recipe `sage_learning/scripts/train_steppen_arrakis.sh` — its header states it is the SAME recipe as the registered NoHz-v3, delta = the signed reward target only. Evidence:
- **`unreachable_k` not passed** → default `0` (off). Confirmed `unreachable_k: int = 0` in `sage_learning/src/data/scorer_data.py:25`. `unreachable_k` (M2c) is the only mechanism that supervises unreachable cells toward low value ("ADD k unreachable cells to the loss mask"); absent ⇒ the value head's loss is masked to *reachable* cells only, so unreachable pushes get zero training signal.
- **`bce_reachable_only=true` is set but INERT.** v3 uses `head_mode=hl_gauss`; in `classifier_module._compute_masked_loss` the hl_gauss branch returns `self._hl_gauss.loss(logits, labels, mask)` and never consults `bce_reachable_only` (that flag only gates the BCE-classifier branch). Even at face value `=true` means "BCE only on reachable" = reachability suppression off.

**Why unreachable cells are masked out by default (i.e. why we don't penalize them):** unlike untried cells (unknown → penalizing = false-negative "C15 poison"), unreachable cells are *known* true-negatives (`f_grid` stores 0, executability is a known fact) — so the false-negative argument does NOT apply here; masking them is a *choice*, not a necessity. The choice is made because (a) at deploy the search pool is pre-filtered to reachable pushes, so the model is never asked to score an unreachable one, and (b) reachability is already an *input* channel (reachable-region), so it is TOLD, not learned. Penalizing unreachable cells (`unreachable_k`) is therefore redundant for ranking — M2c confirmed it buys only deploy-robustness (unmasked output becomes trustworthy, mask-optional), no ranking gain ("the encoder already extracts reachability from robot_region").

## Hypothesis [USER 2026-07-05]

If NoHz-v3 was trained without the reachability loss (it was), retrain it with the reachability loss ON and measure whether it helps — motivated by the hard-1push finding that the model buries the lone opener among ~100 look-alikes on the rarest-opener rooms.

## Plan [CLAUDE] — NOT a clean flag-flip; two caveats to clear first

**What "reachability loss on" means concretely:** add the M2c supervision `+data.unreachable_k=K` (sample K unreachable cells per row and train the value head to score them low). M2c used K=20.

**Caveat 1 — the C15 landmine (correctness).** The current `unreachable_k` applies to ALL rows uniformly. The v3 data mix has **2-push rows** (`v4_hq_h2_scorer`, `v4_hq_onepush_h2_aug`), where a cell unreachable *now* can become reachable *after a setup push* — forcing those to 0 is a **false negative**, which `policy_framework_journal.md` proved is "poison" worse than having fewer labels (the **C15** result: C15 12.4 < even B5 14.1). So a naive `unreachable_k=20` on the full mix would poison the H=2 rows. **Fix required:** per-row gating — apply `unreachable_k` only to H=1-type rows where unreachable == never-valid (a small change in `scorer_data.py`; the M2c note flagged this exact gating as deferred).

**Caveat 2 — prior evidence says robustness, not ranking.** The last time reachability supervision was tested (M2c/M2d verdict, `horizon_q_build_journal.md` 2026-06-13): M2c **eliminated the hallucination zone** (a deploy-robustness win — lets you drop the reachability mask at deploy: dead-slice V 0.327→0.072) but gave **no ranking improvement** (hard@1 M2c 32.21 ± 1.48 vs M2b 32.86 ± 2.38 — no clear win; "the encoder already extracts reachability from robot_region; explicit signal adds no ranking"). Search already pools reachable-only, so the hallucination zone may not touch the pooled eval. Our hard-1push problem is a *ranking* problem (float the rare opener to #1) — so there is a real risk this retrain buys robustness we don't currently need, not the hard-slice ranking we're chasing. Counter-argument to still run it: M2c was tested on the older M-series H=1 F-classifier, not the v3 hl_gauss value head on the full H=2 mix — it has not been tried in this exact setting.

**Staged recipe:** base = `train_steppen_arrakis.sh` minus the `target_scheme=signed`/vmin/vmax lines (i.e. the plain NoHz-v3 recipe) + gated `unreachable_k`. 3 seeds, arrakis GPUs (as steppen did). Eval = reactive + best-first, 1push AND 2push, vs NoHz-v3, with the hard-1push rarity-slice breakdown (Q3b/Q3c) as the primary lens.

**Gate:** expensive 3-seed GPU retrain + a known correctness landmine (C15) + prior counter-evidence ⇒ hold the launch for USER confirm on scope (proceed despite "robustness-not-ranking"? K value? gating approach). Do NOT auto-fire a naive run.

## Discussion

**[Claude 2026-07-05] The reachability loss is likely the WRONG lever for the hard-1push problem — the real cause is a training-sampling gap.** Diagnosed in [[_1push_bottleneck]] Q3d, measured on the training H5s: the model buries the rare 1push opener because `sample_k=30` uniformly subsamples the loss and, on the rarest rows (~1 opener in ~58–90 reachable), the opener is dropped from the loss **59% of the time** — the starvation rate tracks the test solve@1 slice-for-slice. The opener is *reachable* and *labeled*; reachability supervision (`unreachable_k`) targets *unreachable* cells and would not touch it. **So for hard-1push the cheaper, better-targeted fix is positive-aware sampling** (always keep labeled openers in the loss). Reachability loss remains a *deploy-robustness* option (M2c: mask-optional), not a hard-1push ranking fix. For 2push the picture is different again — a collection-sparsity problem (61% of H=2 rows have zero setups labeled), not a loss-toggle at all. Recommendation: run the **positive-aware-sampling retrain** as the 1push-hard experiment; keep reachability-loss parked unless deploy-robustness becomes the goal.

## Run

_(not launched — awaiting confirm)_

## Result

_(pending)_
