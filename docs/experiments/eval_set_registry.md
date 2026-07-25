# Eval-set registry

The authoritative list of every NAMO **eval / test / GT** artifact — what it is, its coverage, distribution, and whether it is EVAL-ONLY.
Companion to the [model registry](horizon_q_model_registry.md) (which logs trained models). Do not reconstruct eval paths by glob — read here.
All paths under `/common/users/dm1487/scratch_namo/`.

## Canonical test manifests (testset_v1) — USE THESE

The canonical eval distribution. Answer keys (which episodes + difficulty), verified by live sim at eval time.

| artifact | path | size | what it is |
|---|---|---|---|
| **1push manifest** | `datasets/namo_testset_v1/labels/onepush_episodes.json` | 991 xml / **1323 episodes** | 1-push answer key (`valid`/`tried`/`solve_rate`). Consumed by `eval_scorer.py`, `time_bestfirst.py`. |
| **2push manifest** | `datasets/namo_testset_v1/labels/pure2push.json` | 983 xml / **1018 episodes** | genuine-2push answer key (1push-unsolvable ∧ 2push-solvable). Consumed by `eval_bestfirst.py` (`--key`). |
| **2push tiers** | `datasets/namo_testset_v1/labels/pure2push_divisions.json` | same 1018 + `division` | difficulty by solve_rate: **easy 238 / medium 409 / hard 371**. THE tier source (join by xml,object,region). |
| combined view | `datasets/namo_testset_v1/labels/twopush.json` | 2341 episodes | union of 1push+2push; used by `derive_onepush_from_2push.py`, `summarize_2push.py`. |

**INVARIANT:** the unit is per **(xml, object, region)** region-opening instance, never per room. One xml holds many instances with different tiers.

## Exhaustive GT (offline analysis only — NOT used by solve@k)

The solve@k/sims eval uses the **live simulator** as verifier, NOT these. These are for offline ranking / "how buried is the true winner" analysis.

| artifact | path | size | what it is | ⚠ |
|---|---|---|---|---|
| **canonical 2push GT** | `curriculum2/beast/round2/h5/testset_gt.h5` | 66,456 nodes / **1117 roots** | REF full-exhaustive root+finish sweep on the canonical set. EVAL-ONLY, never train. | Covers **981/1018** manifest episodes — see alignment below. |
| aligned-981 manifest | `datasets/namo_testset_v1/labels/pure2push_aligned981.json` | 981 episodes | pure2push.json ∩ testset_gt roots. Use this when an analysis needs BOTH manifest + exhaustive GT. |
| alignment sidecar | `datasets/namo_testset_v1/labels/pure2push_gt_alignment.json` | — | records the 981 aligned keys, the **37 manifest episodes with no GT**, and the **136 GT roots not in the manifest** (92 are 1push-solvable). |

**testset_gt.h5 ↔ pure2push mismatch (investigated 2026-07-25):** benign build-version drift, NOT corruption. testset_gt.h5 (Jul 21) rooted a slightly different per-scene object set than pure2push.json (Jun 10). 37 manifest episodes lack a GT root (mostly scenes where the sweep rooted fewer objects — the missing ones are above-average-solvable, not marginal); 136 GT roots aren't in the manifest (92 are 1push-solvable, correctly excluded from the *pure*-2push manifest). The sweep is exhaustively-correct on the 981 it rooted. Decision: **align by intersection (981)** — did NOT re-sweep (data not in doubt). To make GT 1:1 with the manifest later: re-run the sweep keyed to pure2push.json's object list.

**Sweep provenance:** config `amarel:/scratch/dm1487/curriculum2/beast/round2/testset_finish_gt/ref_fullexhaust.yaml` (region_opening, `region_exhaustive_mode: true`, no early-stop) → driver `gt_build.sbatch` (100-way array) → H5 builder `scripts/pipeline/build_rung2_h5.py` → merged to `testset_gt.h5`.

## Non-canonical GT — USE WITH CAUTION

| artifact | path | size | what it is | ⚠ |
|---|---|---|---|---|
| dead-bank GT | `curriculum2/beast/round2/h5/round2_eval.h5` | 73,368 rows / 1609 roots, 940 rooms | exhaustive GT on the **dead-bank distribution**, NOT canonical. | Separations here are **distribution-bound** and do NOT replicate on the canonical set (marvel card: recall@20 67→90 gap was dead-bank-specific). Don't headline results off this. |

## Deprecated / redundant (moved to `labels/deprecated/`)

| artifact | why deprecated |
|---|---|
| `pure2push_HARD.json` | byte-identical to the `division=="hard"` slice of `pure2push_divisions.json`. Use the divisions file. |
| `onepush_HARD.json` | undocumented derived tertile view (442/1323); not in the testset README. Derive from the manifest if needed. |

(Historical `v3_test_validsets.json` — already gone; README still warns about it: had no `object_center`, not what `eval_scorer.py` consumes.)

## Training H5s (NOT eval sets)
Logged in the [model registry](horizon_q_model_registry.md). Disk-cleanup candidates flagged there, not here: `beast2_all.h5` (3.3 GB, v0 lineage-only), `round2_raw.h5` (4.08 GB, raw intermediate, no refs) — user decision, retained pending.
