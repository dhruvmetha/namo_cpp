# AUC reconciliation — one name, seven different measurements

**Status:** `[REF]` `[LIVE]` · reconciled 2026-07-26 · supersedes every loose "setup-vs-dead AUC" citation in the cards.

**The headline: the conflicting AUC numbers were never disagreements — they were seven different metrics that all got written down as "AUC", computed by four scripts, two of which were never in git.** Once each is named by what it compares, the whole spread from 0.47 to 0.93 is consistent. Everything below is now produced by ONE tool over a full model × eval-set grid with seed bands, so nothing here has to be argued about again.

## How to get these numbers

```bash
source env.<box>.sh
python scripts/eval_auc.py --ckpt label:PATH.ckpt [--ckpt ...] --eval-set twopush_gt_h5 --out grid.json   # 2-push panel
python scripts/eval_scorer.py --live-canonical --network edge_crossattn --num-depths 5 --ckpt PATH --out one.json  # 1-push panel
python scripts/agg_auc_grid.py --canonical grid.json --deadbank db.json --onepush deploy=one.json > tables.md
```

`scripts/eval_common.py:mw_auc` is the single AUC definition, imported by both evals — same status as `match_episode` / `bin_of` / `floor_no_replacement`. Eval-set paths resolve through `config/eval_sets.yaml` (`namo.eval_sets`); see [`eval_set_registry.md`](eval_set_registry.md). Scores are cached per (ckpt, eval set), so re-running with new variants is free. **Do not add a fifth AUC code path.**

## The variant grammar

Every AUC asks "does a good push score above a bad one?" Three axes were moving silently: **which truth** (exhaustive sim vs the `valid_first_push` label, which undercounts real setups ~2.4×), **which eval set** (canonical testset vs dead-bank), and **what the negative is** (another cell on the same board, a cell on another board, or the best-scoring cell out of ~70 on another board). Name all three or the number is not citable.

| # | positives | negatives | pooling |
|---|---|---|---|
| **V1** | root exact setup cells (0.9) | root exact dead cells (0.0) | pooled across boards |
| **V2** | same | same | within board, then averaged |
| **V3** | root board-max | dead child board-max | board level, symmetric |
| **V4** | best true setup cell | all reachable dead child **cells** | cell vs cell |
| **V5** | best true setup cell | dead child board-**MAX** | 1 draw vs best-of-~70 |
| **V5m** | same | same, moved (non-noop) boards only | — |
| **V6** | live child board-max | dead child board-max | board level |
| **F1** | child exact opener cells (1.0) | child exact dead cells (0.0) | pooled across boards |
| **F2** | same | same | within board, then averaged |

Median reachable cells per dead board: **70–75**. That number is the whole story of the V4-vs-V5 gap.

## The grid — 12 checkpoints, canonical testset (`twopush_gt_h5`, 981 tiered episodes)

| metric | d20_base | exactv2_s1 | exactv2_s2 | exactv2_s3 | ctrl_s1 | ctrl_s2 | ctrl_s3 | deploy_s1 | deploy_s2 | deploy_s3 | colossus_openeronly | colossus_split |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| V1 | 0.779 | 0.757 | 0.775 | 0.778 | 0.754 | 0.770 | 0.768 | 0.829 | 0.825 | 0.827 | 0.825 | 0.831 |
| V2 | 0.775 | 0.767 | 0.783 | 0.781 | 0.743 | 0.748 | 0.756 | 0.822 | 0.817 | 0.821 | 0.817 | 0.826 |
| V3 | 0.594 | 0.604 | 0.579 | 0.631 | 0.614 | 0.575 | 0.610 | 0.615 | 0.601 | 0.611 | 0.669 | 0.675 |
| V4 | 0.875 | 0.893 | 0.877 | 0.899 | 0.887 | 0.864 | 0.883 | 0.893 | 0.887 | 0.897 | 0.919 | 0.912 |
| V5 | 0.490 | 0.497 | 0.484 | 0.525 | 0.482 | 0.453 | 0.491 | 0.547 | 0.535 | 0.547 | 0.593 | 0.624 |
| V6 | 0.747 | 0.782 | 0.793 | 0.773 | 0.779 | 0.784 | 0.769 | 0.750 | 0.771 | 0.735 | 0.681 | 0.702 |
| F1 | 0.862 | 0.865 | 0.860 | 0.851 | 0.882 | 0.870 | 0.870 | 0.837 | 0.852 | 0.840 | 0.820 | 0.796 |
| F2 | 0.917 | 0.917 | 0.906 | 0.917 | 0.922 | 0.909 | 0.920 | 0.913 | 0.902 | 0.911 | 0.908 | 0.894 |
| setup hit@1 | 51.9 | 49.8 | 51.7 | 50.7 | 45.1 | 46.8 | 47.3 | 59.7 | 59.9 | 59.9 | 59.4 | 63.8 |
| finish hit@1 | 70.0 | 70.0 | 70.4 | 70.6 | 68.5 | 68.5 | 69.6 | 69.4 | 70.7 | 69.2 | 68.4 | 68.6 |

Random floors over the same boards (hypergeometric, `eval_common.floor_no_replacement`): setup @1 **27.6**, finish @1 **15.4**.

Per tier, for the three reference models:

| model | tier | V1 | V5 | setup hit@1 | floor@1 |
|---|---|--:|--:|--:|--:|
| d20_base | easy / med / hard | 0.752 / 0.780 / 0.757 | 0.567 / 0.497 / 0.414 | 66.8 / 56.3 / 37.4 | 47.4 / 27.6 / 14.9 |
| deploy_s1 | easy / med / hard | 0.809 / 0.825 / 0.814 | 0.609 / 0.562 / 0.470 | 70.5 / 65.2 / 46.5 | 47.4 / 27.6 / 14.9 |
| colossus_split | easy / med / hard | 0.805 / 0.840 / 0.802 | 0.648 / 0.639 / 0.574 | 74.7 / 68.0 / 52.1 | 47.4 / 27.6 / 14.9 |

## The 1-push horizon (canonical `onepush_manifest`, `eval_scorer.py --live-canonical`, 1323 episodes)

Seed-mean per condition; opener AUC = valid vs invalid over the deploy-realistic candidate pool (the horizon-1 analogue of F1/F2).

| condition | AUC pooled e/m/h | within-episode e/m/h | hit@1 e/m/h | floor@1 e/m/h |
|---|---|---|---|---|
| d20_base | 0.849 / 0.856 / 0.859 | 0.885 / 0.895 / 0.896 | 98.3 / 82.2 / 40.7 | 62.8 / 15.4 / 2.5 |
| exactv2 (3 seeds) | 0.876 / 0.872 / 0.865 | 0.891 / 0.916 / 0.909 | 97.7 / 83.6 / 40.7 | " |
| ctrl (3 seeds) | 0.903 / 0.888 / 0.877 | 0.918 / 0.925 / 0.911 | 98.7 / 83.8 / 41.0 | " |
| deploy (3 seeds) | 0.844 / 0.840 / 0.837 | 0.872 / 0.899 / 0.893 | 97.9 / 84.9 / 39.2 | " |
| colossus_openeronly | 0.852 / 0.847 / 0.833 | 0.879 / 0.902 / 0.874 | 98.3 / 84.3 / **44.1** | " |
| colossus_split | 0.843 / 0.831 / 0.806 | 0.867 / 0.888 / 0.866 | 98.0 / 84.6 / **35.3** | " |

Opener separation is **tier-flat** too (d20_base even rises slightly on hard: 0.849 → 0.859). Seed spread on 1-push hard hit@1 is **±2.2 pts** (deploy 39.7 / 36.8 / 41.2) — much looser than the 2-push setup hit@1 band, and the reason single-seed hard-1-push comparisons keep flipping.

## The seed noise floor — measured, not guessed

Marvel pre-committed "AUC run-noise ≈ ±0.03". Three paired 3-seed conditions say it is **3× tighter than that** for the pooled metrics, and much looser for the cross-board ones. Mean ± half-range within a condition:

| metric | exactv2 | ctrl | deploy |
|---|--:|--:|--:|
| V1 | 0.770 ± 0.011 | 0.764 ± 0.008 | 0.827 ± 0.002 |
| V2 | 0.777 ± 0.008 | 0.749 ± 0.007 | 0.820 ± 0.003 |
| V3 | 0.604 ± 0.026 | 0.599 ± 0.020 | 0.609 ± 0.007 |
| V4 | 0.890 ± 0.011 | 0.878 ± 0.011 | 0.892 ± 0.005 |
| V5 | 0.502 ± 0.021 | 0.475 ± 0.019 | 0.543 ± 0.006 |
| V6 | 0.783 ± 0.010 | 0.777 ± 0.008 | 0.752 ± 0.018 |
| F1 | 0.858 ± 0.007 | 0.874 ± 0.006 | 0.843 ± 0.007 |
| F2 | 0.913 ± 0.005 | 0.917 ± 0.007 | 0.909 ± 0.005 |
| setup hit@1 | 50.7 ± 0.950 | 46.4 ± 1.100 | 59.8 ± 0.100 |
| finish hit@1 | 70.3 ± 0.300 | 68.9 ± 0.550 | 69.8 ± 0.750 |

**Use ±0.01 for V1/V2/F1/F2, ±0.025 for V3/V5, ±1.1 pt for hit@1.** A delta inside those bands is nothing.

## Five conclusions

**1. Dead-bank's famous separation advantage is a POOLING ARTIFACT.** Same 12 models, canonical vs dead-bank:

| metric | canonical | dead-bank | Δ |
|---|--:|--:|--:|
| V1 (pooled) | 0.793 | 0.913 | **+0.120** |
| V2 (within board) | 0.788 | 0.785 | **−0.003** |
| F1 (pooled) | 0.850 | 0.929 | +0.078 |
| F2 (within board) | 0.911 | 0.693 | **−0.218** |

**Within a board the model is identical on the two distributions.** The entire +0.12 is between-board variance: the dead-bank contains whole-dead-root boards, so pooling hands you a free board-level signal that has nothing to do with ranking skill. On the finish side the pooled number goes *up* while the within-board number falls **0.22** — the dead-bank finish boards are genuinely harder to order, and the pooled metric hides it. So dead-bank AUC absolutes (0.91–0.93) were largely measuring board composition. Model-vs-model deltas **on the same set** stay valid; the absolutes and any cross-set reading do not.

**2. Root setup separation is TIER-FLAT (~0.75–0.83), cross-board comparability is NOT.** V1 barely moves easy→hard for any model (deploy 0.809/0.825/0.814); V5 falls hard every time (deploy 0.609/0.562/**0.470**; d20_base 0.567/0.497/**0.414**). The long-standing "setup separation collapses on hard" reading came from `valid_first_push` labels, which undercount true setups worst exactly where setups are rarest. The real tier-dependent defect is cross-board score comparability.

**3. The flooding mechanism is an order statistic, not average mis-scoring.** V4 (cell vs cell) is **0.86–0.92** for every model — the model is not confused about cells. V5 puts the same setup cell against the dead board's *best* cell and lands near chance, because each dead board gets ~70 draws. Quote V5 **as** an order statistic. It is not "the model can't tell a setup from a dead push", and it is not below chance.

**4. AUC is UNDERPOWERED on the setup axis and UNINFORMATIVE on the opener axis.** Spearman across all 12 checkpoints:

| pair | ρ |
|---|--:|
| V1 vs setup hit@1 (2-push root) | **+0.94** |
| V5 vs setup hit@1 | +0.84 |
| F1 vs finish hit@1 (2-push child) | **+0.07** |
| 1-push opener AUC vs hit@1 — easy / med / hard | +0.39 / **−0.63** / +0.37 |

Two different failures, not one. On the **setup** axis AUC orders models correctly but has almost no dynamic range: the adopted split-budget loss beats its paired control by **+0.006 V1** (inside the ±0.01 noise band, i.e. invisible) while gaining **+4.3 pts setup hit@1** (4× its band). Same direction, unusable resolution. On the **opener/finish** axis AUC does not track ranking at all — `ctrl` has the best 1-push hard opener AUC of any model (0.877 seed-mean) and a completely ordinary hard hit@1 (41.0), while `colossus_openeronly` has one of the worst AUCs (0.833) and the **best** hard hit@1 (44.1). The asymmetry has an obvious cause: setup positives are rare (~2% base rate) so separation and top-1 nearly coincide, whereas openers are abundant, so a model can average well across ~70 candidates and still not put a winner first. **Headline rank metrics with their floor; use AUC as a diagnostic, and never as the opener-side gate.**

**5. The two axes trade, and the panel shows it.** `colossus_split` is best on the setup/cross-board axis (V5 0.624, setup hit@1 63.8) and **worst on finish separation** (F1 0.796) — which is exactly its deploy signature (fastest 2-push search, worst hard-1-push tail @5 64.2 vs deploy's 72.1). `deploy` is the mirror image. Reading only one half of the panel will pick the wrong model. (Correlational, n≈5 conditions, and `d20_base` is a partial counterexample — F1 0.862 with a mid tail. Treat as a lens, not a law.)

## What each historically-cited number actually was

| cited | where | metric | eval set | truth source | verdict |
|---|---|---|---|---|---|
| 0.93 "H2 setup AUC" | `horizon_q_build_journal.md` | setup-vs-**nonsetup** at H=2, Hz model | horizon-q era | old | **stale** — different model family and label, pre-dates the single-ranker framing |
| ~0.46 held-out setup | `EXP-2026-07-11-curriculum-ladder` | setup-vs-rest, `pos_weight=6` | ladder-era | broken labels | **retired** — the card itself concludes "fix the data, not the loss" |
| 0.80 / 0.75 hard | `_ranker_bottleneck.md` Detector A | V1-family | testset_v1 | `valid_first_push` | valid as written; **depressed**, and its tier slope is a label artifact |
| 0.805 / 0.799 / 0.762 | `_setup_value_check.md` Table 2 | V1-family per tier | testset_v1 | `valid_first_push` | same family; same artifact |
| 0.716 / 0.720 / 0.733 | marvel line 287, `auc_r2_testset.json` | V1-family, negatives = **tried-only** | testset_v1 | `valid_first_push` | **doubly depressed** — incomplete positives AND search-selected hard negatives. Not "the testset wall" |
| 0.745–0.799 | marvel line 322 | **V1** | `testset_gt.h5` | exhaustive | **canonical**; this grid continues the line (d20_base 0.779 → deploy 0.827 ± 0.002) |
| 0.876–0.925 | `rankdiag_*.json`, colossus / exact-value cards | **V1** | `round2_eval.h5` | exhaustive | correct for that set, but **pooling-inflated ~+0.12** (conclusion 1). Never compare to a canonical number |
| 0.750 / 0.768 hard | 07-24 `REPORT.md` D2 | **V6** | `testset_gt.h5` | exhaustive | reproduces exactly |
| 0.469 | 07-24 `REPORT.md` D5 | **V5** | `testset_gt.h5` | exhaustive | reproduces (grid: deploy_s1 hard V5 0.470) |
| **0.583** | 07-24 card, anchor of 07-25 | claimed V5, moved boards only | `testset_gt.h5` | exhaustive | **does NOT reproduce** — V5m gives 0.515 hard. Depends on undocumented join/subset choices. **Stop citing it** |

## Rules going forward

- **Name the variant, always.** "V1 root-cell pooled, exhaustive GT, canonical testset" — not "setup-vs-dead AUC 0.83".
- **Report V2 alongside V1, and F2 alongside F1.** The pooled-vs-within pair is what exposes a board-composition effect; V1 alone is how the dead-bank illusion survived for weeks.
- **Exhaustive GT only.** `valid_first_push` is completion-sampled; every AUC built on it is a lower bound of unknown tightness.
- **Never compare across eval sets.** Same metric, different distributions, +0.12 of pure composition.
- **Rank metrics are the headline; AUC is the diagnostic.** Always print the random floor next to hit@k. On the opener/finish axis AUC is not even a valid tiebreak (ρ≈0).
- **Noise bands to clear:** ±0.01 on V1/V2/F1/F2, ±0.025 on V3/V5, ±1.1 pt on 2-push setup hit@1, **±2.2 pt on 1-push hard hit@1**.
- **Tier everything** (easy/med/hard) and report both horizons (1-push / 2-push) — the tier slope lives in different variants than you'd guess.
