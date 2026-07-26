# AUC reconciliation — one name, seven different measurements

**Status:** `[REF]` · reconciled 2026-07-26 · supersedes every loose "setup-vs-dead AUC" citation in the cards.

**The headline: the conflicting AUC numbers are not disagreements — they are seven different metrics that all got written down as "AUC".** Once each is named by what it actually compares, the whole spread from 0.47 to 0.93 is consistent and every number lands where it should. Two numbers are genuinely retired (stale labels), and one number in current use (`0.583`) does not reproduce.

Everything below for the current deploy checkpoint was recomputed in one pass from a single cached score file, so the variants differ **only** in metric definition — same model, same scores, same eval set: script `scripts/sandbox/auc_reconcile.py`, scores `round3/eval/gt_model_errors/values.npy` (ckpt `round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt`), eval `round2/h5/testset_gt.h5` (exhaustive root+finish GT, 66,456 nodes / 982 canonical scenes), tiers joined via `pure2push_divisions.json`.

## The one-paragraph plain-English version

Every AUC asks "does a good push score above a bad one?" — but "good", "bad", and "compared against what" were never held fixed across cards. Three axes moved silently: **which truth** (exhaustive simulation vs the incomplete `valid_first_push` label, which undercounts real setups ~2.4×), **which eval set** (the training-distribution dead-bank vs the canonical testset — separations are strongly distribution-bound), and **what the negative is** (another cell on the same board, a cell on some other board, or the single best-scoring cell out of ~70 on another board). The third axis is the one that produced the scary numbers: comparing one setup cell against another board's *maximum* is comparing a single draw against the best of seventy, so it reads near-chance even when the model is fine per-cell.

## Variant table — same model, same scores, same eval set

| # | what it compares | easy | med | hard | all |
|---|---|--:|--:|--:|--:|
| **V1** | root **cell-level, pooled** across boards: exact setup cells (0.9) vs exact root dead cells (0.0) | 0.809 | 0.825 | 0.814 | **0.829** |
| **V2** | same masks, **within board** (mean of per-board AUCs) | 0.811 | 0.840 | 0.810 | 0.822 |
| **V3** | cross-board, symmetric: root board-max vs dead post-push board-max | 0.663 | 0.613 | 0.574 | 0.614 |
| **V4** | cross-board, **cell vs cell**: best true setup cell vs all reachable dead post-push cells | 0.925 | 0.912 | 0.864 | **0.892** |
| **V5** | cross-board, as reported in the 07-24 card's D5: best true setup cell vs dead board-**MAX** | 0.609 | 0.559 | **0.469** | 0.545 |
| **V5m** | V5 restricted to moved (non-noop) dead boards | 0.707 | 0.626 | **0.515** | 0.608 |
| **V6** | board-level live vs dead (07-24 REPORT D2): live board-max vs dead board-max | 0.686 | 0.747 | 0.768 | 0.750 |

Median reachable cells per dead board: **70–75**. That number is the whole story of the V4-vs-V5 gap.

## What each historically-cited number actually was

| cited | where | metric | eval set | truth source | verdict |
|---|---|---|---|---|---|
| 0.93 "H2 setup AUC" | `horizon_q_build_journal.md` | setup-vs-**nonsetup** at H=2, Hz model | horizon-q era | old | **stale** — different model family, different label, pre-dates the single-ranker framing |
| ~0.46 held-out setup | `EXP-2026-07-11-curriculum-ladder` | setup-vs-rest with `pos_weight=6` | ladder-era | broken labels | **retired** — the card itself concludes "fix the data, not the loss" |
| 0.80 / 0.75 hard | `_ranker_bottleneck.md` Detector A | V1-family (root cell, pooled) | testset_v1 | `valid_first_push` (incomplete) | valid as written; **depressed** by label incompleteness |
| 0.805 / 0.799 / 0.762 | `_setup_value_check.md` Table 2 | V1-family per tier | testset_v1 | `valid_first_push` | same family, reproduces the above; **depressed** the same way |
| 0.716 / 0.720 / 0.733 | `EXP-…-marvel` line 287, `auc_r2_testset.json` | V1-family, but negatives = **tried-only** cells | testset_v1 | `valid_first_push` | **doubly depressed** — incomplete positives AND search-selected hard negatives. Do not cite as "the testset wall" |
| 0.745–0.799 | `EXP-…-marvel` line 322 | **V1** proper | `testset_gt.h5` | exhaustive | **canonical**; current deploy ckpt continues this line at 0.829 |
| 0.876–0.925 | `rankdiag_*.json`, colossus / exact-value cards | **V1** proper | `round2_eval.h5` | exhaustive | **canonical but a different distribution** — dead-bank / in-training-distribution rooms, not the canonical testset. Never compare it to a testset number |
| 0.750 / 0.768 hard | 07-24 `REPORT.md` D2 | **V6** | `testset_gt.h5` | exhaustive | reproduces exactly |
| 0.469 | 07-24 `REPORT.md` D5 | **V5** | `testset_gt.h5` | exhaustive | reproduces exactly — but see the order-statistic caveat below |
| **0.583** | 07-24 card ("orchestrator recompute"), used as the anchor in 07-25 | claimed V5 restricted to moved boards | `testset_gt.h5` | exhaustive | **does NOT reproduce** — V5m gives 0.515 hard. The 0.583 depends on undocumented join/subset choices; stop citing it |

## The three real conclusions

**1. Root setup separation is 0.81–0.83 and TIER-FLAT — it does not degrade on hard.** V1 is 0.809 / 0.825 / 0.814 across easy/med/hard. The long-standing "setup separation collapses on hard" reading (0.805 → 0.762, or 0.80 → 0.75) came from `valid_first_push` labels, which undercount true setups worst exactly where setups are rarest. Against exhaustive truth the effect disappears. This does not overturn the *ranking* story — hard top-1 placement is genuinely worse (hit@1 70.6 / 65.2 / 46.8 by tier, 07-24 D1) — separation and top-1 placement are different failures, and only the second is tier-dependent.

**2. "Within-board strong, pooled weak" is false at the root cell level.** V1 (pooled, 0.829) ≈ V2 (within-board, 0.822). Root cells are comparably ordered whether or not you mix boards. The cross-board weakness is real but lives specifically **between a root board and post-push dead boards** (V3 0.614), not in pooling per se.

**3. The flooding mechanism is an order statistic, not average mis-scoring.** V4 says a true setup cell outscores a random dead post-push cell **89%** of the time — the model is not confused about cells. V5 says the same setup cell loses to a dead board's *best* cell more than half the time on hard. The difference is entirely that each dead board gets ~70 draws to produce one high score. So the honest statement is: **the model's score distribution has a fat enough right tail on dead boards that ~70 draws routinely beat one true setup** — a tail/calibration problem about extremes. It is *not* "the model rates dead boards higher than live ones on average" (V6 says 0.75–0.77 in the right direction), and it is *not* "below chance discrimination".

For choosing what to fix, V5's near-chance reading is the deploy-relevant one — best-first really does pop each board's max first, so the order statistic is what floods the queue. But it must be quoted **as** an order-statistic comparison, never as "the model can't tell a setup from a dead push."

## Rules going forward

- **Name the variant, always.** Write "V1 root-cell pooled, exhaustive GT, testset" — not "setup-vs-dead AUC 0.83". Any AUC without positives/negatives/eval-set/truth-source stated is not citable.
- **Exhaustive GT only.** `valid_first_push` is completion-sampled; every AUC built on it is a lower bound of unknown tightness. Use `testset_gt.h5` / `round2_eval.h5`.
- **Never compare across eval sets.** `round2_eval.h5` (≈0.91) and `testset_gt.h5` (≈0.83) measure the same thing on different distributions. Marvel already established separations are distribution-bound; the numbers are not each other's baselines.
- **V1 is the default headline metric** for setup ordering (it is what `scripts/rl_loop/score_round2_eval.py` reports, and what the model registry tracks). Report V4 and V5 together whenever the claim is about cross-board comparability — V4 alone hides the flooding, V5 alone overstates it.
- **Tier the report** (easy/med/hard), same as every other result.

## Reproduce

```bash
source env.ilab.sh
python scripts/sandbox/auc_reconcile.py       # ~2 min, CPU only, uses the cached values.npy
```
