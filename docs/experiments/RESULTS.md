---
status: hub
tags:
  - results
updated: 2026-07-16
---
# Results — DAgger curriculum training framework

The maintained approach. A clean seed, then rounds of **{ generate fresh scenes → screen with the current model → keep its mistakes → label → accumulate → retrain → eval }**, laddering **1-push → 2-push → …** (Ant-Man → Beast → …). Each stage's model also produces the *next* stage's dataset for free (screened-dead scenes become the multi-push bank).

**Setting:** CAR robot, testset `namo_testset_v1`, region-opening criterion. Every result split by **difficulty (easy/med/hard)**. Difficulty is defined *per horizon* (compare within a horizon, not across): **1-push** = `solve_rate` fixed cuts (hard < 0.05 / med 0.05–0.30 / easy ≥ 0.30); **2-push** = number of solving first-pushes (`n_setups`). Full per-experiment detail lives in each card under `log/`.

**Archive:** the pre-curriculum line — horizon-Q / NoHz single-ranker, RL-only self-imitation, horizon-role probe, prior-work ledger — is preserved verbatim in [archive/RESULTS_pre_dagger_horizonq_2026-07.md](archive/RESULTS_pre_dagger_horizonq_2026-07.md).

---

## 1-push ladder (Ant-Man)  🔵 ACTIVE

Clean 50k seed, then five DAgger rounds screening fresh scenes with antman_{r-1} (best-first, budget 300), keeping not-top-5 mistakes. Full table, execution notes, open questions → [EXP-2026-07-14 card](log/EXP-2026-07-14-region-opening-curriculum-marvel.md).

| model | rows | easy@1 | med@1 | **hard@1** | all@1 | hard@20 |
|---|---|---|---|---|---|---|
| antman-0 (seed) | 50,000 | 92.7 | 63.7 | **23.0** | 72.7 | 85.8 |
| antman-2 | 90,700 | 95.6 | 73.2 | **28.4** | 78.1 | 91.2 |
| antman-3 | 120,657 | 96.3 | 77.2 | **32.8** | 80.4 | 91.7 |
| antman-4 | 151,218 | 96.7 | 81.2 | **39.2** | 82.9 | 94.6 |
| antman-5 | 167,655 | 97.1 | 78.9 | **42.6** | 82.9 | 92.6 |
| random | — | 62.6 | 19.2 | **1.5** | ~39.4 | 37.7 |

**Finding.** hard@1 **23.0 → 42.6** in five rounds (~28× random), no plateau; gains all at low k (better *ordering*, sim verifies for free). easy saturated, med@1 live, hard the headroom. The climb tracks data VOLUME (50k → 168k rows); keep-rate falls (5.5% → 3.8%) and per-row efficiency rises — a first hint DAgger targeting matters, but **the targeting-vs-volume control has not been run**. Round 5 (undersized 449k) was a redistribution, not a lift (hard@1 +3.4 but med@1 −2.3 / hard@20 −2.0). NEXT: redo round 5 at 700k + fold in the volume control. Free byproduct: **72,521 labeled 2-push (Beast) episodes** + ~865k unlabeled leads.

## 2-push ladder (Beast)  ⏳ pending

Dataset accumulating for free from every Ant-Man round (screen-dead → bank). Not trained yet. See card for the bank contents (`(xml, object_id, robot_goal)` triples).
