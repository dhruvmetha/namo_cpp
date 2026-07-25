# Eval-set registry

The authoritative list of every NAMO **eval / test / GT** artifact — what it is, its coverage, distribution, and whether it is EVAL-ONLY.
Companion to the [model registry](horizon_q_model_registry.md) (which logs trained models). Do not reconstruct eval paths by glob — read here.
All paths under `/common/users/dm1487/scratch_namo/`.

## ⭐ Single source — `config/eval_sets.yaml`

**All eval code resolves test-set paths from `config/eval_sets.yaml` via `namo.eval_sets`.** This doc is the human-readable truth; the yaml is the machine truth (one file, both must agree). Change the test set = edit the yaml only; every reader follows.
- Python: `from namo import eval_sets` → `eval_sets.PURE2PUSH` / `.ONEPUSH` / `.DIVISIONS` / `.TWOPUSH_SOURCE` / `.TWOPUSH_GT_H5` (resolved absolute Paths, box-portable via `namo.paths`).
- Shell/slurm: `python -m namo.eval_sets pure2push_manifest` prints the resolved path; `--list` prints all names.
- Guard: `python/tests/test_eval_sets.py` asserts every path resolves to an existing file with the expected counts (1323 / 1018 / 238·409·371 / 2341 / 66,456). Run it before trusting a config edit.
- Migrated: **all** committed eval entrypoints (incl. `eval_bestfirst.py` / `time_bestfirst.py`) + agg scripts + slurm launchers. No committed eval code hardcodes a `namo_testset_v1/labels` path any more.

## Canonical test manifests (testset_v1) — USE THESE

The canonical eval distribution. Answer keys (which episodes + difficulty), verified by live sim at eval time.

| artifact | path | size | what it is |
|---|---|---|---|
| **1push manifest** | `datasets/namo_testset_v1/labels/onepush_episodes.json` | 991 xml / **1323 episodes** | 1-push answer key (`valid`/`tried`/`solve_rate`). Consumed by `eval_scorer.py`, `time_bestfirst.py`. |
| **2push manifest** | `datasets/namo_testset_v1/labels/pure2push.json` | 983 xml / **1018 episodes** | genuine-2push answer key (1push-unsolvable ∧ 2push-solvable). Consumed by `eval_bestfirst.py` (`--key`). |
| **2push tiers** | `datasets/namo_testset_v1/labels/pure2push_divisions.json` | same 1018 + `division` | difficulty by solve_rate: **easy 238 / medium 409 / hard 371**. THE tier source (join by xml,object,region). |
| **SOURCE (root)** | `datasets/namo_testset_v1/labels/twopush.json` | 2341 episodes | the **one depth-2 exhaustive collection pass** (`build_2push_validset.py → twopush.json`, keyed by realpath). `onepush_episodes.json` and `pure2push.json` are **DERIVED from this** (via `derive_onepush_from_2push.py`). This is the master; the two above are the consumed splits. Also read by `summarize_2push.py`. |

**INVARIANT:** the unit is per **(xml, object, region)** region-opening instance, never per room. One xml holds many instances with different tiers.

## Exhaustive GT (offline analysis only — NOT used by solve@k)

The solve@k/sims eval uses the **live simulator** as verifier, NOT these. These are for offline ranking / "how buried is the true winner" analysis.

**1-push has no separate GT h5 — the manifest IS the exhaustive 1-push GT.** `onepush_episodes.json` already exhausts every **reachable** push per episode (`tried` mean 81.6/300, rest unreachable) and records **every** valid opener (`valid` mean 30.8). 1-push is single-ply so exhaustion is cheap; only 2-push needs a separate GT (`testset_gt.h5`) because its finish tree is deep. So: 1-push exhaustive GT = `onepush_episodes.json` itself; 2-push exhaustive GT = `testset_gt.h5`.

| artifact | path | size | what it is | ⚠ |
|---|---|---|---|---|
| **canonical 2push GT** | `curriculum2/beast/round2/h5/testset_gt.h5` | 66,456 nodes / **1117 roots** | REF full-exhaustive root+finish sweep on the canonical set. EVAL-ONLY, never train. | Covers **981/1018** manifest episodes — see alignment below. |
The manifest↔GT alignment is the TRUTH below (not a file — the derived aligned artifacts were archived to `deprecated/`; this text is the record).

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
| `onepush_HARD.json` | undocumented derived tertile view (442/1323); not in the testset README. |
| `pure2push_aligned981.json` | pure2push.json filtered to the 981 GT-covered episodes — derived; alignment recorded above. |
| `pure2push_gt_alignment.json` | the 981/37/136 diff map — derived; alignment recorded above. |

(Historical `v3_test_validsets.json` — already gone; README still warns about it: had no `object_center`, not what `eval_scorer.py` consumes.)

## Training H5s (NOT eval sets)
Logged in the [model registry](horizon_q_model_registry.md). Disk-cleanup candidates flagged there, not here: `beast2_all.h5` (3.3 GB, v0 lineage-only), `round2_raw.h5` (4.08 GB, raw intermediate, no refs) — user decision, retained pending.
