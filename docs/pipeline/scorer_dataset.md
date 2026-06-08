# Scorer data & lineage (read this before touching scorer/DiT data)

There are **two datasets for two different models** built from the **same v3 corpora (feb + aug9)**.
They get confused constantly — they are NOT the same thing.

| | **F-scorer (`sharp`, Q₁)** | **DiT (diffusion goal model)** |
|---|---|---|
| dataset | **`v3_scorer_e4_data`** (`config/datasets/v3_scorer_e4.yaml`) | **`v3_balanced_1to1`** (`config/datasets/v3_balanced_1to1.yaml`) |
| kind | **discriminative** | **generative** |
| label | `f_grid` (60×5): 1=opens / 0=tried&failed / NaN=unreachable — **exhaustive 1-push** | first push `a1` of a *solution* (`se2_target_a1`, `goal_mask_a1`) + `solution_depth` ∈ {1,2} |
| has negatives? | **yes** (all 60×5 labeled) | **no** (positives = solution actions only) |
| push depth | **chain-1 only** | **single + multi-push** (1:1 single:multi *intent*; built H5 ≈ 107k:36k) |
| what it answers | "P(this push opens the goal region)" | "what's a good first push toward the goal?" |
| trained | the champion **`sharp`** (EdgeCrossAttn + Fourier PE + per-edge Embedding) | the cropped cross-attn DiT |

## How the F-scorer data is built (the lineage that was undocumented)
```
region-opening pkls (exhaustive chain-1)
   └─ scripts/pipeline/build_episode_validsets.py
        → per-episode answer key {xml:[episodes]}  (object_center, valid, tried, solve_rate)
scene masks (local_tight 5ch)  ←  LIFTED from the DiT H5 (v3_balanced_1to1)  [no re-render]
   └─ scripts/pipeline/build_scorer_dataset.py
        → JOIN masks + exhaustive 1-push f_grid(60×5) + r_mask     → ctx / f_grid / r_mask
   └─ scripts/pipeline/add_contact_px.py
        → add contact_px(60,2) per-edge contact pixels             → v3_scorer_e4_data  (N=98,387)
```
So: **the F-scorer reuses the DiT's masks and bolts on exhaustive 1-push labels.** Masks shared; labels different.

## The 2-push consequence (why this matters)
- We **cannot** build an exhaustive 2-push `f_grid` (all first×second push combos = combinatorial blow-up).
- So there is **no discriminative 2-push scorer dataset.**
- Multi-push capability therefore comes from **SEARCH over the 1-push Q₁** (`sharp`), with the **simulator as the model** — not from a learned 2-push scorer. (n-push = Bellman backup of Q₁; see `docs/experiments/scorer_hacman_journal.md`.)
- The only multi-push *training* signal we have is the DiT's `solution_depth=2` first-pushes in `v3_balanced_1to1` (generative, non-exhaustive) — usable as a first-push *proposer*, not a discriminative label.

## Provenance note
The scorer build scripts lived in gitignored `scripts/sandbox/` until 2026-06-08; promoted to
`scripts/pipeline/` and registered (`config/datasets/v3_scorer_e4.yaml`) so the scorer data is
documented as cleanly as the DiT data. Experiment results live in
`docs/experiments/scorer_hacman_journal.md`; this doc is the **data** record.
