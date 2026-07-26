---
name: namo-data-pipeline
description: Use BEFORE building, filtering, evaluating, splitting, or labeling any NAMO training/test data, and before writing or editing ANY script that groups, dedups, buckets, matches, or splits samples by scene / episode / object / difficulty (H5/NPZ building, validsets, eval harnesses, train/val splits, difficulty stratification, informative/hard manifests). Forces reuse of existing pipeline scripts before writing new ones, enforces the per-episode (not per-room) data invariants, fixes bugged scripts in place, and keeps docs current.
---

# NAMO data & eval pipeline guard

You are about to touch NAMO data, eval, or training-data tooling. Work this checklist top-to-bottom **before writing code**. The whole point: reuse what exists, don't re-make a known bug, leave docs true.

## 0. Read the invariants first
`docs/pipeline/multi_episode_rooms.md` — the root rule and its three past failure modes. **One room (`xml`) has MANY episodes (different pushed object + goal region each).** Almost every bug in this pipeline is some version of treating the room as the unit.

## 1. Reuse before rewrite — check the inventory
Search `scripts/` (committed) and `scripts/sandbox/` (gitignored one-offs) for a script that already does this. **Edit/extend it; do not fork a new copy.** Current load-bearing pieces:

| script | does | invariant it must honor |
|---|---|---|
| `pipeline/build_episode_validsets.py` | pkls → per-episode answer key `{xml:[episodes]}` (object_center, solve_rate, valid, tried) | one record PER EPISODE, keyed by pushed-object pose |
| `pipeline/build_scorer_dataset.py` (+`add_contact_px.py`) | scorer H5 (`v3_scorer_e4`): JOIN DiT masks (`v3_balanced_1to1`) + exhaustive 1-push `f_grid`(60×5)/`r_mask`, then add `contact_px`(60,2) | reuse masks; labels = per-episode key; held out BY ROOM |
| `build_test_divisions.py` | pkls → hard/med/easy manifests + validsets (hard<5/med<30) | difficulty is per-episode |
| `build_filtered_train_h5.py` | de-leak a train H5 to a solve_rate band | keep rows by MATCHED episode sr |
| `emit_informative_manifest.py` | pkls whose episode solve_rate ∈ band | ⚠ selects whole PKL → dilutes; for a FILTERED set, filter per-episode at SOURCE |
| `build_h5.slurm` / `convert_to_hdf5.py` | NPZ dir → packed H5 | — |
| eval scripts + `resolve_robust.sh` | scorer eval, diffusion eval, and the verdict layer | → see **Eval structure** below |

If nothing fits, say so out loud, then write the new script in **committed** `scripts/`, not sandbox. If a sandbox script has now been reused ≥twice, promote it to `scripts/`.

### Eval structure — which to run when (after training a model)
Three layers, one rulebook shared so the scorer and diffusion are graded apples-to-apples:

```
eval_common.py     RULEBOOK   match_episode / bin_of / floor / mw_auc — imported, never re-defined. Not runnable.
eval_auc.py        PANEL      offline separation (AUC) + rank over exhaustive-GT H5s (2-push). ← the ONLY AUC tool
agg_auc_grid.py    TABLES     renders an eval_auc grid (models × eval sets) into markdown + seed bands
eval_scorer.py     REFEREE    one checkpoint → full diagnostic panel.        ← run this after training
resolve_robust.sh  VERDICT    runs the referee over all ckpts × 3 seeds, paired compare. ← run this to DECIDE
eval_grounding.py  (diffusion counterpart of eval_scorer — same rulebook; its mask-decode helpers = eval_feasibility.py)
```

- **Quick look at one scorer checkpoint** → `eval_scorer.py`:
  ```bash
  CUDA_VISIBLE_DEVICES="" /scratch/dm1487/envs/namo/bin/python scripts/eval_scorer.py \
    --ckpt <.../checkpoints/epochNNN-val_lossX.ckpt> --network edge_crossattn --num-depths 5 \
    --out /scratch/dm1487/eval/<name>.json
  ```
(`--network dit_classifier` only for the old global-readout E0; arch variants auto-detected from the ckpt.)
- **Decide if a change helped** (the real verdict — hard@1 carries ±3–4 ckpt noise) → add the run's group to `GRPS=()` in `resolve_robust.sh`, then run it. It averages per-seed and compares paired across seeds. (It parses `divisions.hard.scorer_realistic.@1` from each JSON — keep that key name stable.)
- **`eval_common.py` is a library, not a script** — running it does nothing; both evals `import` it.
- **Any AUC / setup-rank / opener-rank number** → `scripts/eval_auc.py` (2-push, exhaustive GT) or `eval_scorer.py --live-canonical` (1-push). Never write a new AUC: seven definitions once drifted across four scripts. Name the variant (V1/V4/V5/F1…) when quoting — grammar in `docs/experiments/auc_metrics_reconciliation.md`.

## 2. Enforce the invariants (hard gate)
- [ ] Unit of work = **(pushed object, goal region)**, never `xml` alone.
- [ ] Match a sample to its episode by **`object_center` (~0 mm)** + `gt ∈ valid` (not the stored/first episode).
- [ ] **Difficulty is per-episode** — never inherit a file / pkl / division label.
- [ ] Train/val/test **holdout splits group by ROOM (xml)**, never by row.
- [ ] A difficulty-**filtered** dataset filters **per episode at the source**, not per pkl.

## 3. If an existing script violates a rule, FIX IT IN PLACE (don't fork)
Correct it, then re-verify with the checks below, then note the fix in the doc.

## 4. Verify (numbers that must hold)
- `gt_in_valid_frac ≈ 1.0` and `bad_object_match = 0` after episode matching.
- Room-leakage = **0%** across any train/val/test split (group-by-xml).
- Difficulty composition matches intent (run `train_difficulty_composition.py`).

## 5. Close out
- Update the affected doc: `docs/experiments/model_comparison_report.md`, `docs/pipeline/multi_episode_rooms.md`, and this inventory.
- Delete superseded scripts (e.g. anything replaced by `build_episode_validsets.py`).
- If a NEW gotcha emerged, add it to `multi_episode_rooms.md` **and** a one-line auto-memory.
