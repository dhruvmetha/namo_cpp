---
type: experiment
status: live
created: 2026-07-05
updated: 2026-07-05
metric: "Root-cause check: do the v3 TRAINING labels under-count setups? Labels come from a sampled search (~14 of ~60 follow-up moves tried per first push; stamped 'never opens' if all sampled fail), so real setups whose finish is a needle get mislabeled worthless. Directly re-simulate a sample of 0-labeled first pushes and measure the true mislabel rate, split by difficulty, vs the eval-set 23%. Not yet run."
tags:
  - experiment
  - diagnostic
---
# Setup-label quality — are the v3 training labels under-counting setups?

**Sibling to [[_ranker_bottleneck]] and [[_setup_value_check]].** The bottleneck card showed the model is blind to setups. When we asked *why the model can't learn setups*, the leading cause was: the training labels themselves call many real setups "worthless." We measured a **23% under-count on the eval set** (pure2push). This card measures it directly on the **training data** — if it's real there, it's the confirmed #1 root cause, and the fix is "better labels," not a reward tweak.

**Plain idea.** A "setup" is only labeled a setup in training if the data collection actually *found* a finish after it. But collection is sampled — it only tries a handful of follow-up moves. If it misses the finish, a real setup gets stamped "never opens." We check how often that happens by re-simulating and searching for the finish properly.

## Hypothesis [CLAUDE — for USER review]

The v3 2-push training labels under-count setups, because collection samples only a fraction of the follow-up moves. A large share of first pushes labeled "never opens" (target 0) are actually valid setups — a finish exists if you search exhaustively. If confirmed, the model was trained to call real setups worthless — the direct cause of its setup-blindness.

## Plan [CLAUDE]

Pure re-simulation of existing training rows — no training, no GPU, no model. Directly measure: of the first pushes the training data labeled "never opens" (0), what fraction actually have a finish?

**The data (found by recon).** `v4_hq_h2_scorer/data.h5` (311,324 rows; the (scene, first-push)→{0, 0.9, 1} table the NoHz-v3 scorer trained on). Each row carries the scene `xml`, `object_center` (which object was pushed), and a 60×5 grid of candidate first-push actions — each cell is one `(edge, depth)` push with its own label and its own `frac_tried`/`frac_succ` sampled-rollout counts. Scene XMLs live on Amarel at `/scratch/dm1487/datasets/v4_hq_h2/pkls_2push_s30/` (64 shards).

**The label mechanism (confirmed from `configs/sampled_depth2_k30.yaml`) — why an under-count is expected.** Collection samples up to `k=30` (×3 restarts) reachable cells *per level*. So for each tried first push, only a sample of its follow-up (second-push) moves is tried — mean `frac_tried ≈ 13.5` of ~60 reachable. A first push is stamped **0** ("never opens") iff *all* its sampled follow-ups fail. A real setup whose finish is 1-of-~60 is found only if that finish lands in the ~14 sampled → likely missed → mislabeled 0. Same sampling mechanism already implicated in the eval-set 23% under-count.

**Method.**
- Sample first-push cells labeled **0** (`f_grid==0`, `r_mask==1`), stratified across scenes and difficulty (target a few hundred; enough for a tight fraction).
- For each: load the `xml`, identify the object by `object_center`, apply the exact `(edge, depth)` first push in sim → `s1`, then **exhaustively** search the second-push moves for a finish. Reuse the built pipeline (`scripts/pipeline/exit_collect.py::exhaustive_a2` / `scripts/sandbox/check_h2_finish.py`) — same protocol as the eval-set 23% measurement, so the numbers are comparable.
- Run on **Amarel** (the XMLs live there; no model or GPU needed — physics only). If XMLs must move, rsync only the shards the sample touches.

**Measure, split by difficulty:**
1. **Mislabel rate** — fraction of "never opens" first pushes that actually have a finish (= real setups mislabeled worthless). Compare to the eval-set 23% (easy 36% / med 22% / hard 11%).
2. **Mechanism check** — does the mislabel rate rise as `frac_tried` falls (fewer follow-ups sampled → more missed finishes)? A clean negative correlation confirms it's the sampling, not something else.

**Pre-registered interpretation:**
- **Large mislabel rate (double digits)** → the training labels *are* bad → confirmed #1 root cause → the fix is better labels (exhaustive/bootstrapped re-labeling), not reward or loss tweaks alone.
- **Small mislabel rate** → labels aren't the main issue → rarity + ranking-loss dominate; re-weight the fix.
- **frac_tried correlation** → confirms (or not) that the sampling budget is the lever — i.e. more follow-ups per first push during collection would recover setups.

**Deliverables:** the mislabel-rate table (by difficulty, vs eval 23%), the frac_tried correlation plot, and a plain-English verdict on whether bad labels are confirmed as the root cause. Owned files: this card, `assets/setuplabel_*.png`, `scripts/sandbox/setup_label_quality.py`. Physics-verify finishes under the same config as eval; flag any offline↔online mismatch.

## Run
_(Claude, auto)_

## Result
_(Claude, auto)_

## Discussion
_(you ↔ Claude — newest at the bottom.)_
