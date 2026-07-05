---
type: experiment
status: done
created: 2026-07-05
updated: 2026-07-05
metric: "Root-cause check: do the v3 TRAINING labels under-count setups? Labels come from a sampled search (~30 of ~55 reachable follow-up moves tried per first push; stamped 'never opens' if all sampled fail), so real setups whose finish is a needle get mislabeled worthless. RESULT: 16.7% of 'never opens' first pushes actually have a finish overall (600 re-simmed cells) — but this is 41.8% inside scenes that have any setup, and rises to 40-43% for the least-sampled cells (frac_tried<15 / coverage<0.4) vs ~1% for well-sampled ones. Sampling budget drives it (corr coverage↔mislabel -0.51). CONFIRMED: bad training labels are the #1 root cause of setup-blindness; fix = better labels, not a reward tweak."
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
_(Claude, auto — 2026-07-05, Amarel)_

**What ran.** Pure physics re-simulation of a stratified sample of first-push cells from the v3 training scorer H5
(`v4_hq_h2_scorer/data.h5`; 155,662 H=2 episodes = the 2-push-budget rows that carry the setup labels). No model,
no GPU. Script: `scripts/sandbox/setup_label_quality.py` (`sample` → `resim` → `aggregate`).

**Sample** (seed 0, one random cell per random episode → de-correlated by room; 599/600 distinct scenes):
- 600 **zero** cells: `f_grid==0 & r_mask==1` (reachable first push stamped "never opens") — the target.
- 150 **setup09** cells: `f_grid==0.9` (KNOWN setups) — positive control.
- 50 **opener10** cells: `f_grid==1.0` (direct openers) — control.

**Protocol (identical to the eval-set 23% measurement).** For each cell: load the `xml`, match the pushed object by
`object_center` (exact — **0.00 mm** match), apply the first push `(edge,depth)` in sim → `s1`, then **exhaustively**
simulate *every* reachable 2nd push (`exit_collect.exhaustive_a2`) under the same car config
(`namo_config_complete_skill15_car_1x`, collisions-off = training-match) and the same region-open criterion
(≥20% of 100 s0-sampled goal points reachable). `mislabel := a finish exists (n_open>0)`.

**Compute.** Amarel SLURM array `57865062` (main-redhat), 16 tasks × 50 cells, 8–13 min wall, all COMPLETED.
Manifest + per-cell results: `/scratch/dm1487/tmp/setuplabel/{manifest.json,results_*.jsonl}` (mirrored to
`/common/users/dm1487/scratch_namo/tmp/setuplabel/`).

**Validation (sim is faithful — no offline↔online mismatch).**
- Positive control: **149/150 (99.3%)** known setups re-found a finish exhaustively.
- `n_open − frac_succ` mean **+16.6** (exhaustive finds ~17 more openers than collection sampled; never fewer).
- Direct-opener control: **50/50** open in one push.
- **Premise correction:** the per-first-push 2nd-push budget is `frac_tried` **median ≈ 30** (= the `k=30`
  config), **not** the ~13.5 the Plan assumed (that avg was dragged down by zero-budget H=1 and direct-opener
  cells). Collection still tried only **~53% (median)** of the *reachable* 2nd pushes — the coverage gap that
  drives the mislabels.

## Result
_(Claude, auto)_

### Headline — the "never opens" label is wrong 1-in-6 times overall, ~2-in-5 where it matters

| population | mislabel rate (0-cells with a real finish) | n | 95% CI |
|---|---|---|---|
| **all "never opens" cells** | **16.7%** | 100/600 | 13.9–19.9 |
| — dead-row episodes (0 labeled setups) | **0.8%** | 3/368 | 0.0–2 |
| — live-row episodes (≥1 labeled setup) | **41.8%** | 97/232 | 36–48 |
| _eval-set reference (pure2push, live-only)_ | _23%_ | — | — |

The overall 16.7% is **diluted by dead episodes** (61% of the sample), whose "never opens" labels are almost always
correct (0.8% wrong) — collection covered those small/blocked scenes well, so the model can trust "this scene is
dead." The damage is concentrated **inside solvable scenes**: there, **42%** of the first pushes labeled worthless
are actually real setups. That is *higher* than the eval-set 23% on the comparable (live-only) population — the
training labels under-count setups at least as badly as the eval set does.

### Mechanism — it's the sampling budget, cleanly and monotonically

`assets/setuplabel_fractried.png`, `assets/setuplabel_coverage.png`.

| frac_tried (2nd pushes sampled) | mislabel | | coverage (frac_tried / reachable) | mislabel |
|---|---|---|---|---|
| <15 | **40.2%** (70/174) | | <0.4 | **43.1%** (88/204) |
| 15–24 | 15.8% (19/120) | | 0.4–0.6 | 7.4% (9/122) |
| 25–34 | 5.3% (10/187) | | 0.6–0.8 | 2.3% (2/87) |
| ≥35 | **0.8%** (1/119) | | ≥0.8 | **0.5%** (1/187) |

`corr(frac_tried, mislabel) = −0.35`; `corr(coverage, mislabel) = −0.51`. When collection tried <40% of the
reachable 2nd pushes, 43% of "never opens" labels are wrong; when it tried ≥80%, essentially none are. The
under-count **is** the under-sampling — not a criterion or physics artifact (the positive control rules that out).

### Difficulty (labels-based solve-rate tertiles among live rows — matches the eval direction)

| band (solve-rate proxy) | mislabel | eval-set ref |
|---|---|---|
| easy (sr~0.68) | **84.9%** (62/73) | 36% |
| med (sr~0.19) | 32.1% (26/81) | 22% |
| hard (sr~0.05) | 11.5% (9/78) | 11% |

Same ordering as the eval set (**easy > med > hard**): easy scenes have many finishes, so a "0"-labeled push is very
likely a missed setup; hard scenes have few, so a "0" is more often truly 0. Caveat: this difficulty axis is
labels-based (same under-counting labels) and among live rows only — a stratifier, not ground truth; the direction,
not the exact magnitude, is the comparable claim.

### Pre-registered questions — answered

1. **Are the training labels confirmed bad (root cause)? → YES.** 16.7% of "never opens" first pushes are actually
   setups overall, and **~42%** inside scenes that have any setup / **40–43%** for the least-sampled cells. The model
   was trained to call a large share of real setups worthless. This is the **confirmed #1 root cause** of its
   setup-blindness. The fix is **better labels** (more follow-ups per first push during collection, or exhaustive /
   bootstrapped re-labeling), **not** a reward or loss tweak alone.
2. **Does the sampling budget drive it? → YES, decisively.** Monotonic in both `frac_tried` (40%→0.8%) and coverage
   (43%→0.5%), `corr(coverage, mislabel) = −0.51`. More follow-ups per first push during collection would recover
   the setups — the sampling budget is the lever.

**One-line verdict:** Bad training labels are confirmed as the #1 driver of setup-blindness — 1-in-6 "never opens"
first pushes overall (2-in-5 inside solvable scenes) are actually real setups, and it's driven entirely by the
sampled-search budget (fix = re-label with more/exhaustive follow-ups, not a reward tweak). Dead-scene labels are
trustworthy; the leak is missed setups inside solvable scenes.

## Discussion
_(you ↔ Claude — newest at the bottom.)_
