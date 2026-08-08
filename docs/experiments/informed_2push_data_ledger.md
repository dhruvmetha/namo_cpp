---
status: frozen
tags: [experiment]
thread: scorer-search
updated: 2026-06-09
---

# Informed 2-push — DATA & MANIFEST LEDGER

Running record of every data file, manifest, checkpoint, config, and output I create or refer to in the informed-2-push effort. Companion to [informed_2push_journal.md](informed_2push_journal.md). Update on every new artifact.

## PINNED SEARCH / SIM SETTINGS (default ON — [USER] instruction)
- **Collisions ALLOWED**, only robot↔non-target disallowed. In the beam: `COLLISIONS_OFF=True` → `env.set_collision_checking(False)` → `check_object_collision_=False`. Controller: robot collisions ALWAYS abort (ungated, `namo_push_controller.cpp:609-611`); pushed-object collisions allowed.
- **Target-region-goal**: search targets the goal region (`robot_goal` = goal region; `is_robot_goal_reachable`). Collection: `region_allow_collisions: true` + `--target-goal-region` (all committed configs already do this).
- ⚠ `DATA_COLLECTION_GUIDE.md` example yaml shows `region_allow_collisions: false` — MISLEADING, ignore it; use `true`.

## MANIFESTS (referred to)
| path | what | status |
|---|---|---|
| `$NAMO_MANIFESTS/test_pure2push_combined.txt` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | 985 GENUINE depth-2 scenes (per-episode `_pair_` xmls); beam baseline 16%@1→56%@2 | **USING** (leaf diagnostic) |
| `$NAMO_MANIFESTS/test_pure2push_hardness.csv` | per-scene hits/depth2_hits + bucket (easy/med) | reference |
| `$NAMO_MANIFESTS/test_2push_solvable_combined.txt` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | 1186 scenes, collisions-allowed-defined | **REJECTED for leaf diag** — 1-push-contaminated (smoke: every leaf rank-0) |
| `$NAMO_MANIFESTS/test_1push_solvable_combined.txt` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | 539 scenes | not yet used |
| `…/test_{feb,aug9}_2push_solvable.txt`, `…_1push_solvable.txt` | per-source splits | reference |
| ⚠ provenance | none of the `*_solvable*`/`pure2push` manifests have a committed builder (sandbox one-offs in /scratch) | recover/commit if used in a result |

## CHECKPOINTS / CONFIGS / PRIMITIVES
| path | what |
|---|---|
| `$NAMO_SCRATCH/sage_outputs/scorer/sharp_s1/namo-classifier/9yizg6i8/checkpoints/epoch017-val_loss0.2713.ckpt` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | champion `sharp` 1-push scorer (the leaf) |
| `config/namo_config_complete_skill15_car_1x.yaml` | CFG used by beam + live_scorer |
| `data/1x_car_d5_motion_primitives_15_{square,tall,wide}.dat` | car d5 primitives (60×5) |
| env XMLs | `$NAMO_DATASETS/car_envs/v3/test/{aug9_car,feb_car}/…/env_*_pair_*.xml` |

## FILES I CREATED
| path | what |
|---|---|
| `scripts/sandbox/diag_leaf_s1.py` | leaf-vs-search diagnostic, EXTENDED for H3 (logs per-leaf scalars + good/dead label JSONL; `--start/--end` shards). Reuses `BeamPlanner`, no fork |
| `scripts/sandbox/diag_fpv_array.slurm` | SLURM array (8 tasks×30 scenes) to collect H3 per-leaf data in parallel (main-redhat, collisions-allowed+target-region-goal) |
| `scripts/sandbox/diag_fpv_aggregate.py` | aggregate shard JSONLs → per-scalar AUC + recall + H3a verdict |
| `docs/experiments/informed_2push_journal.md` | the journal (hypotheses + accept/reject) |
| `docs/experiments/informed_2push_data_ledger.md` | THIS ledger |
| `$NAMO_SCRATCH/eval/diag_leaf_s1_pure2push.json` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | **H1/H2 result** (N=25 pure2push): H1 REJECTED (recall@10=.877), H2 ACCEPTED (AUC=.534) |
| `$NAMO_SCRATCH/eval/diag_fpv_shard{0..7}.{json,jsonl}` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | H3 per-leaf data, 240 pure2push scenes — PENDING (array `55815040`) |
| `$NAMO_SCRATCH/eval/diag_fpv_aggregate.json` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | H3a verdict — pending |
| `$NAMO_SCRATCH/eval/diag_fpv_jobid.txt` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | array job id (55815040) |
| `$NAMO_LOGS/diag_fpv-55815040_*.{out,err}` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | array task logs |
| `$NAMO_SCRATCH/eval/diag_leaf_s1_smoke.json`, `diag_fpv_smoke.{json,jsonl}` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | smoke outputs (disposable) |

## SLURM JOBS
| job | what | status |
|---|---|---|
| `55815040` (array 0-7) | H3 first-push-value data, 8×30 pure2push scenes, main-redhat | DONE (5/8 finished; tasks 1,2,3 hit 2h wall → ~7 scenes lost; 4486 leaves / 233 scenes total) |

## RESULTS (this session)
| file | finding |
|---|---|
| `$NAMO_SCRATCH/eval/diag_leaf_s1_pure2push.json` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | H1 REJECTED (leaf recall@10=.877), H2 ACCEPTED (V=maxP AUC=.534) |
| `$NAMO_SCRATCH/eval/diag_fpv_shard{0..7}.jsonl` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | 4486 per-leaf (s0,a1)→good/dead records + scalars (H3b seed data) |
| `$NAMO_SCRATCH/eval/diag_fpv_aggregate.json` — **⚠ ARTIFACT GONE (verified 2026-08-06)** | H3a: best single scalar `mean_top5` AUC .73 (full) |
| (inline numpy, journaled) | held-out-room AUC: `mean_top5` .796, 6-scalar combo **.817**, maxP .69 — TRAINING-FREE first-push ranker |

## NEW DATASET HOME (policy+value line)
`$NAMO_DATASETS/policy_value_v1/` (Amarel only) — clean root for the search-generated POLICY+VALUE data (`pkls/ → npz/ → h5/`, `manifests/ logs/ stats/`). Full schema + provenance + non-exhaustive caveat in its `README.md`. Separate from the exhaustive 1-push `v3_scorer_1push`. Collection NOT started; prior = champion `sharp` ckpt.

## TODO data (only if a hypothesis calls for it)
- `s1` leaf data (exhaustive 1-push at post-first-push states) — would extend the exhaustive-1-push collection; MUST use collisions-allowed + target-region-goal; held out BY ROOM; per-episode unit.
- 2-push trajectory pkls — via the 5-phase cascade (`CORPUS_TAG=v3_{aug9,feb} sbatch scripts/amarel/v3_cascade_driver.slurm`).
