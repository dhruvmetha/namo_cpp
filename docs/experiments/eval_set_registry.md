# Eval-set registry

The authoritative list of every NAMO **eval / test / GT** artifact — what it is, its coverage, distribution, and whether it is EVAL-ONLY.
Companion to the [model registry](horizon_q_model_registry.md) (which logs trained models). Do not reconstruct eval paths by glob — read here.
All paths under `/common/users/dm1487/scratch_namo/`.

## ⭐ Single source — `config/eval_sets.yaml`

**All eval code resolves test-set paths from `config/eval_sets.yaml` via `namo.eval_sets`.** This doc is the human-readable truth; the yaml is the machine truth (one file, both must agree). Change the test set = edit the yaml only; every reader follows.
- Python: `from namo import eval_sets` → `eval_sets.PURE2PUSH` / `.ONEPUSH` / `.DIVISIONS` / `.TWOPUSH_SOURCE` / `.TWOPUSH_GT_H5` (resolved absolute Paths, box-portable via `namo.paths`). The historical sampled tiers and dead-bank H5 live under `noncanonical_files`, not the canonical registry surface.
- Shell/slurm: `python -m namo.eval_sets pure2push_manifest` prints the resolved path; `--list` prints all names.
- Guard: `python/tests/test_eval_sets.py` asserts every path resolves to an existing file with the expected counts (1322 / 1012 / GT tiers 385·488·137 + 2 unknown / 2341 / 68,393), and verifies all three canonical random artifacts use the registered search. Run it before trusting a config edit.
- Migrated: **all** committed eval entrypoints (incl. `eval_bestfirst.py` / `time_bestfirst.py`) + agg scripts + slurm launchers. No committed eval code hardcodes a `namo_testset_v1/labels` path any more.

## Canonical test manifests (testset_v1) — USE THESE

The canonical eval distribution. Answer keys (which episodes + difficulty), verified by live sim at eval time.

| artifact | path | size | what it is |
|---|---|---|---|
| **1push manifest** | `datasets/namo_testset_v1/labels/onepush_search_eval.json` | **1322 episodes** | Search-eligible 1-push answer key (`valid`/`tried`/`solve_rate`). Fixed per-episode tiers: hard <0.05 (**204**), medium <0.30 (**421**), easy otherwise (**697**). |
| **2push manifest** | `datasets/namo_testset_v1/labels/pure2push_search_eval.json` | **1012 episodes** | Search-eligible genuine-2push answer key (1push-unsolvable ∧ 2push-solvable) with shared search-queue failures and zero-GT-setup disagreements excluded. |
| **2push tiers** | `datasets/namo_testset_v1/labels/pure2push_gt_divisions_search_eval.json` | same 1012 + `division` | Fixed exhaustive-GT setup density: hard <5% (**137**), medium 5–30% (**488**), easy ≥30% (**385**), unmatched GT root (**2 unknown**). THE tier source (join by xml,object,region). |
| **legacy sampled 2push tiers** | `datasets/namo_testset_v1/labels/pure2push_divisions_search_eval.json` | same 1012 + `division` | Historical incomplete-manifest setup-count bins: easy 238 / medium 408 / hard 366. Retained only to reproduce earlier tables; do not use for new headline results. |
| **SOURCE (root)** | `datasets/namo_testset_v1/labels/twopush.json` | 2341 episodes | the **one depth-2 exhaustive collection pass** (`build_2push_validset.py → twopush.json`, keyed by realpath). `onepush_episodes.json` and `pure2push.json` are **DERIVED from this** (via `derive_onepush_from_2push.py`). This is the master; the two above are the consumed splits. Also read by `summarize_2push.py`. |

The unfiltered canonical-path `onepush_episodes_canonical.json` (1323) and `pure2push.json` (1018) remain source artifacts. The registered search views exclude one 1push and two 2push episodes where learned plus all three random seeds exhaust a search queue that does not realize the exhaustive answer. The 2push view additionally excludes four hard episodes whose sampled manifest claims a solution but whose fully exhaustive root contains zero genuine setups; every exclusion is an exact `(xml, object, region)` record and remains in the untouched source for audit.

**INVARIANT:** the unit is per **(xml, object, region)** region-opening instance, never per room. One xml holds many instances with different tiers.

## Canonical random-search baseline — USE THIS

The baseline is **one three-seed measurement**, not three separate baselines: uniform-random push ordering with seeds **7000/8000/9000**, reported as mean ± sample standard deviation at every simulator budget. It uses the same search as the learned ranker on both registered horizons: `hmax=2`, budget 900, `combine=q`, confidence discount τ=0.15, no-op dedupe on, and jam-depth pruning on. The learned arm is deterministic and therefore runs once. “Tight” below means solve@1 for 1push and solve@2 for 2push.

The machine-readable entry is `baselines.random_search_hmax2` in `config/eval_sets.yaml`; its three budget-900 aggregates are `eval/postprune_hmax2/final35/agg_random_s{7000,8000,9000}.json` under `$NAMO_SCRATCH`. New comparisons must use their mean curve and sample-SD band; a single seed is only a debugging/reproduction view.

| horizon | tier | random tight | random @30 | random @900 | random solved-only calls |
|---|---|---:|---:|---:|---:|
| 1push | easy | 60.1±2.3 | 100.0±0.0 | 100.0±0.0 | 1.8±0.1 |
| 1push | medium | 12.7±0.5 | 97.9±0.5 | 100.0±0.0 | 6.4±0.2 |
| 1push | hard | 3.3±1.2 | 80.1±2.1 | 100.0±0.0 | 23.0±1.0 |
| 2push | easy | 6.3±2.1 | 73.4±2.8 | 100.0±0.0 | 28.6±1.1 |
| 2push | medium | 0.9±0.6 | 31.6±2.4 | 98.8±0.4 | 99.7±2.2 |
| 2push | hard | 0.0±0.0 | 9.2±1.5 | 76.9±2.8 | 285.6±36.4 |

The equal-budget hard-2push tails use the same three seeds with original per-episode RNG streams preserved: learned reaches 100% at 3,831 calls, while final random success is 99.3±0.7% and its mean curve reaches 95% at 3,456 calls versus 1,071 for learned. Full learned-versus-random results and plots are in [RESULTS.md](RESULTS.md).

## Exhaustive GT (offline analysis only — NOT used by solve@k)

The solve@k/sims eval uses the **live simulator** as verifier, NOT these. These are for offline ranking / "how buried is the true winner" analysis.

**1-push has no separate GT h5 — the manifest IS the exhaustive 1-push GT.** `onepush_episodes.json` already exhausts every **reachable** push per episode (`tried` mean 81.6/300, rest unreachable) and records **every** valid opener (`valid` mean 30.8). 1-push is single-ply so exhaustion is cheap; only 2-push needs a separate GT because its finish tree is deep. So: 1-push exhaustive GT = `onepush_episodes.json` itself; the **only canonical comprehensive 2-push GT artifact** = `testset_gt_plus35.h5`.

| artifact | path | size | what it is | ⚠ |
|---|---|---|---|---|
| **canonical 2push GT** | `curriculum2/beast/round2/h5/testset_gt_plus35.h5` | 68,393 nodes / **1152 roots** | REF full-exhaustive root+finish sweep plus 35 targeted missing roots. EVAL-ONLY, never train. | Covers **1016/1018** source episodes and **1010/1012** search-eligible episodes. |

**Canonicality rule:** this H5 is the only comprehensive 2-push GT test artifact for new results. `twopush.json` is the live-search answer key, not comprehensive per-candidate GT; `pure2push_divisions_search_eval.json` contains historical sampled tiers, not GT; and `round2_eval.h5` is a noncanonical dead-bank diagnostic distribution, not a test set. “Exhaustive” means every candidate under each included root was swept; it does not erase the two uncovered search episodes, which remain explicitly `unknown` and are excluded from tier-specific GT claims.

**testset_gt.h5 schema** (verified 2026-07-26 by opening the file):
```
chain_depth       (N,)         int8
contact_px        (N,60,2)     f32
ctx               (N,5,64,64)  f16
edges_agree       (N,)         int8
f_grid            (N,60,5)     f32
is_solution_node  (N,)         int8
n_reach_edges     (N,)         int32
n_tried           (N,)         int32
n_win             (N,)         int32
node_kind         (N,)         str   in {root, depth2, depth2_noop}
object_id         (N,)         str
parent_depth      (N,)         int16
parent_edge       (N,)         int16
r_mask            (N,60,5)     f32
value_mask        (N,60,5)     f32
value_target      (N,60,5)     f32
robot_goal        (N,3)        f32
setup_moved       (N,)         int8
xml               (N,)         str
```
Row counts by `node_kind`: root=1117, depth2=49622, depth2_noop=15717.
Join keys: a root row is identified by `(xml, object_id)`; a finish state (depth2 or depth2_noop) is identified by `(xml, object_id, parent_edge, parent_depth)`.
This file carries **no `ceiling_mask` dataset** — so its `0` cells are hard zeros from an exhaustive sweep, not ceiling-optimistic placeholders (contrast with H5s that do carry a ceiling_mask, where `0` can just mean "not swept").

The manifest↔GT alignment is the TRUTH below (not a file — the derived aligned artifacts were archived to `deprecated/`; this text is the record).

**GT alignment (finalized 2026-07-30):** the original H5 had benign build-version drift: 37 source-manifest episodes lacked a root and 136 roots were outside the manifest. A targeted exhaustive fill completed 35 of those 37 exact `(xml, object, goal region)` episodes and added 1,937 rows. Two unusually large sweeps were stopped by user decision and remain explicitly unknown; fixed-tier charts exclude those two rather than mis-bin them.

**Sweep provenance:** config `amarel:/scratch/dm1487/curriculum2/beast/round2/testset_finish_gt/ref_fullexhaust.yaml` (region_opening, `region_exhaustive_mode: true`, no early-stop) → driver `gt_build.sbatch` (100-way array) → H5 builder `scripts/pipeline/build_rung2_h5.py` → merged to `testset_gt.h5`.

**testset_gt.h5 ↔ pure2push.json `valid_first_push` agreement, cell level (verified 2026-07-26, 287 sampled pure-2push episodes):** this is a different check from the root-alignment above — it asks, for each individual first-push candidate, whether GT's green set (openers + setups at the root) and the manifest's `valid_first_push` agree, not just whether the episode is rooted in both.

- **Join key confirmed correct.** Correctly-paired episodes score mean Jaccard **0.443** between the two green sets; a deliberately SHUFFLED pairing scores **0.015** — a 30x separation, the proof that `(xml_realpath, object_id)` is the right join key.
- **GT is the more complete source.** Cells in both = 1605, GT-only = 3549, manifest-only = 107. GT is a strict superset of the manifest's valid set in **238/287 episodes (83%)**. Median green-set size: GT **11**, manifest **4**.
- **Cause, from trial counts:** median first pushes TRIED per episode is comparable at depth 1 — 55 (manifest) vs 65 (GT sweep). The gap opens at depth 2: the manifest only marks a first push valid if it *also* found a finishing push, and its finish sweep was not exhaustive, while GT's was (49,622 `depth2` + 15,717 `depth2_noop` rows over 1117 roots).
- **Practical guidance:** for any per-candidate truth badge or rank-of-first-good-push metric, use `testset_gt.h5`, not `pure2push.json`'s `valid_first_push` — the manifest would mis-score roughly two-thirds of genuinely good pushes as failures.
- **Honest caveat:** the 107 manifest-only cells are good pushes GT does NOT mark, about 6% of the greens the manifest knows about. Where that bites, a GT-based metric is slightly pessimistic about the model — the safe direction.
- **`robot_goal` gotcha (cost time to rediscover):** GT's `robot_goal` field is a SAMPLED POINT INSIDE the goal region, not the XML's designated goal site. They differ by 0.02-0.22 m on sampled episodes. Treating the two as equal produces a false mismatch alarm (2 of 115 "agreeing" cases were actually this).

**Design note, not a defect:** 85 rows carry a root cell of 0.0 whose own child grid contains an opener. All 85 are `node_kind == depth2_noop` with `setup_moved == 0`. This is deliberate per `scripts/pipeline/build_rung2_h5.py:182-206` — a setup push that did not move the object is not a genuine 2-push setup, so its "win" is really a recovered 1-push opener and is withheld from the gamma overlay on purpose.

## Non-canonical GT — USE WITH CAUTION

| artifact | path | size | what it is | ⚠ |
|---|---|---|---|---|
| dead-bank GT | `curriculum2/beast/round2/h5/round2_eval.h5` (`deadbank_gt_h5`) | 73,368 rows / 1609 roots, 940 rooms | exhaustive GT on the **dead-bank distribution**, NOT canonical. | Separations here are **distribution-bound** and do NOT replicate on the canonical set (marvel card: recall@20 67→90 gap was dead-bank-specific; V1 runs ~0.08 higher here than on the canonical set for the SAME model). Don't headline results off this. Its rooms are not in `pure2push_divisions.json`, so it has **no tiers** — all-only. |

## Who consumes these — the offline ranking panel

`scripts/eval_auc.py` is the single tool for separation (AUC) + rank metrics over the exhaustive-GT H5s, for BOTH sets above; `scripts/eval_scorer.py --live-canonical` is the 1-push counterpart on `onepush_manifest`. Both take their AUC definition from `scripts/eval_common.py:mw_auc`. Which variant means what, and which historical numbers are retired: [`auc_metrics_reconciliation.md`](auc_metrics_reconciliation.md). Do not add a fifth AUC code path.

## Deprecated / redundant (moved to `labels/deprecated/`)

| artifact | why deprecated |
|---|---|
| `pure2push_HARD.json` | byte-identical to the `division=="hard"` slice of `pure2push_divisions.json`. Use the divisions file. |
| `onepush_HARD.json` | undocumented derived tertile view (442/1323); not in the testset README. |
| `pure2push_aligned981.json` | pure2push.json filtered to the 981 GT-covered episodes — derived; alignment recorded above. |
| `pure2push_gt_alignment.json` | the 981/37/136 diff map — derived; alignment recorded above. |

(Historical `v3_test_validsets.json` — already gone; README still warns about it: had no `object_center`, not what `eval_scorer.py` consumes.)

## Training H5s (NOT eval sets)
Logged in the [model registry](horizon_q_model_registry.md). Disk-cleanup candidates flagged there, not here: `beast2_all.h5` (3.3 GB, v0 lineage-only), `round2_raw.h5` (4.08 GB, raw intermediate, no refs) — user decision, retained pending.
