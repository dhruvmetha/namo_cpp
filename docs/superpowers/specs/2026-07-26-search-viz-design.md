---
status: spec
tags: [visualization, search, eval, 2push]
updated: 2026-07-26
---

# Search visualization for the 2-push test set

## Goal

A local static web page that replays, **per episode**, the value-guided greedy best-first search on the pure-2push test set, so we can see two things at once: **how good the ranker's output is** (its 60×5 score grid against exhaustive ground truth) and **how that output steers the search** (the priority queue, pop by pop, sim by sim).

Audience: the user, at the desk and over screen-share with advisors. Not a paper figure, not an aggregate dashboard.

## Non-goals

- No aggregate statistics page. Per-episode only; the index is a browsable catalog of all episodes, not a summary.
- No random-prior arm. `--prior uniform` traces are out of scope for v1.
- No MuJoCo-rendered frames. Geometry is drawn as SVG from state vectors, because the story is about the pushes the search did **not** try, which rendered frames cannot show.
- No hosted/shareable build in v1. A curated self-contained export is a possible follow-up.

## Scope of v1

Two models, both search strategies — four trace arms.

| axis | v1 values | source |
| --- | --- | --- |
| model | **ceiling** `d20_plus_setup_only_splitloss` — `round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt`; **HARD** `setup_split_HARD_seed1` — `round3/models/setup_split_HARD_seed1/checkpoints/epoch011-val_loss0.8787.ckpt` | model registry |
| search strategy | `off` (plain best-first) and `conf τ=0.15` (adopted failure-discount) | `eval_bestfirst.py --discount` |
| difficulty | easy 238 / medium 409 / hard 371 | `datasets/namo_testset_v1/labels/pure2push_divisions.json` |

The two models are the same data and architecture under **different label regimes** (`build_rung2_h5.py:102-109`). Ceiling refuses to call a searched-but-unresolved push dead: because the sweep stopped at depth 2, "found nothing" only means unresolved, so those cells get the best value they could still have — `γ²=0.81` at a root, `γ=0.9` at a depth-2 node — flagged in `ceiling_mask`, with capped/unresolved children masked out entirely. HARD instead labels every reachable cell two-sided and exact, dead and unknown alike to `0`, with no fence and nothing masked. This is the pair that makes the visualization worth building: HARD wins the deep 1-push tail but regresses 2-push ordering (sims-to-solve 46.0 → 53.8), and nobody has seen *where* in the grid that trade happens.

Because the regimes differ, magnitudes are not comparable across the two models — the ceiling model puts `0.81` where HARD puts `0`. Zone D sidesteps this entirely by working in **rank space within a grid**, so a cross-model comparison is always "where did each model rank the green cells", never "whose number is bigger". A mask toggle still shows or hides the ceiling model's fenced cells.

Adding further models means dropping another trace directory in and appending one line to a manifest. No code change.

## Data model

Episode key is `(xml_realpath, object_id)` throughout — matching `pure2push.json`, which is nested `{xml_path: [episode_dict, ...]}` with `object_id` inside each episode dict. There is no separate episode id field. On disk that key becomes a filename as `<xml basename without extension>__<object_id>.json`; `manifest.json` holds the mapping back to the full realpath so nothing has to parse filenames.

### Ground truth is a badge, not a number

The model's cell values only ever determine an **ordering**. Nothing in this page compares a model magnitude to a GT magnitude, which is what makes the two label regimes safe to show side by side: the ceiling model was trained with `γ²=0.81` on searched-dead root pushes where GT says `0`, and that offset is label-regime bookkeeping, not model error.

So GT collapses to **green / not green**, read off `value_target` at the candidate's `(edge, depth)` (`scripts/pipeline/build_rung2_h5.py:95-113`):

| GT cell | meaning | shown as |
| --- | --- | --- |
| `1.0` | **opener** — this push merges robot and goal | solid green |
| `0.9` | **setup** — its subtree contained a verified win | hollow green |
| `0` | searched exhaustively, nothing found — dead | red |
| `-1` | unreachable edge | never appears; see below |
| masked (`value_mask == 0`) | reachable but untried — no signal | grey |

`0.9` only ever appears on root rows; at a depth-2 (finish) board the only green is `1.0`. That is correct for the search — at a finish board a "setup" means nothing, you need the merge — so green semantics track board depth automatically with no special-casing.

Unreachable (`-1`) cells never enter the priority queue at all, because candidates come from `rank_first_pushes_h2`, which enumerates reachable pushes only. They are drawn in the grid for completeness and are absent from the queue by construction.

**Rank of best green** is the page's headline quantity: the queue position of the highest-ranked green candidate. Model good means green at rank 1-3; model bad means green at rank 22 and twenty-one red pops burn before it.

Two offline artifacts per (episode, arm):

### `trace/<model>/<strategy>/<ep_key>.json` — produced by a patched `eval_bestfirst.py`

Records the search as it ran:

- `pops`: ordered list, one per simulated push — `{t, board_id, obj, edge, depth, q, bp, w, se, opened}`. `t` is the sim counter; `bp` is `priority(q, V, combine)`; `w` is the board weight at pop time; `se = bp * w` is the effective priority.
- `boards`: one record per board — `{board_id, depth, parent_edge, parent_depth, n_candidates, w0, free_strikes, pool}` where `pool` is the board's full candidate list `[{obj, edge, depth, q}]` (every candidate, including the ones never popped — these are the point of the visualization).
- `grids`: per board, the model's full 60×5 `P` grid at that board's state (`live_scorer.py:186-192`).
- `scene`: geometry needed to draw the room — wall/static rects, movable object pose, goal region, robot pose, and the object's 60 contact points in world coordinates.
- `result`: `{solved, sims, plan_len, end}` where `end ∈ {solved, budget, exhausted}`.

The board parent link (`parent_edge`, `parent_depth`) is what lets the page attribute a finish board to the setup push that spawned it, and is the same key the GT joiner uses.

### `gt/<ep_key>.json` — produced by a new joiner over `testset_gt.h5`

- root grid: `value_target[60,5]` from the `node_kind == 'root'` row matching `(xml, object_id)`.
- finish grids: for each `node_kind ∈ {depth2, depth2_noop}` row matching `(xml, object_id)`, its own `value_target[60,5]`, keyed by that row's `(parent_edge, parent_depth)`.
- `valid_first_push`, `tried_first_push`, `solve_rate_first_push`, `is_2push_solvable` copied from `pure2push.json`.

The action space matches exactly on both sides: `edge_idx ∈ [0,60)` (contact point index) and `depth ∈ [0,5)` (push-step level). The model's `P` and the GT's `value_target` are the same shape on the same axes, which is what makes the side-by-side heatmap meaningful.

Coverage: `testset_gt.h5` roots 981 of the 1018 manifest episodes (Jun-10 manifest vs Jul-21 sweep object-set drift, recorded at `docs/experiments/eval_set_registry.md:39`). The remaining 37 episodes still get a trace and a replay; they are flagged `no GT` in the index and their truth badges and heatmap-B panel are suppressed rather than faked.

## Trace generation

Additive, flag-gated patch to `scripts/sandbox/eval_bestfirst.py`: a `--trace-out DIR` option that writes one JSON per episode. With the flag unset, behavior is byte-identical to today — verified by running a smoke set with and without the flag and diffing `--out`/`--leaf-out`.

Cost: 1018 episodes × roughly 28 sims × about 1 s ≈ 8 CPU-hours per arm, embarrassingly parallel across episodes. Four arms (2 models × 2 strategies) ≈ 32 CPU-hours, which is well under an hour on a 64-core SLURM slice. Follow the `scaled-run` skill for the launch.

## The page

Static files served by `python -m http.server` from the data root: `index.html`, `episode.html`, `app.js`, `style.css`, plus `trace/` and `gt/` directories and a small `manifest.json` listing available arms and episodes. No build step, no external dependencies, no CDN — episode JSON is fetched lazily so the 1018-episode catalog stays responsive.

### Index

Three dropdowns — model, search strategy, difficulty — then a sortable table of matching episodes with columns: scene, object, tier, solved, sims used, **rank of best green** in the root ordering, what the model's top-1 push actually was (`opener` / `setup` / `dead`), and a `no GT` flag where applicable.

Default sort: rank of best green, worst first. That column is the ranker's job stated directly — the answer sitting at rank 12 instead of rank 1 is exactly the failure we are hunting. For the 37 no-GT episodes the column falls back to the rank of the push that actually solved the episode (known from the trace) and is marked as a fallback.

### Episode view

One clock `t` (the sim counter), four linked zones:

- **A. Scene** — SVG top-down room from `scene`: walls, goal region, robot, the movable object, and its 60 contact points. Point color encodes the model's `q`; an outline ring encodes GT truth. Already-popped candidates are dimmed and numbered with their pop order.
- **B. Priority queue at `t`** — the unsimulated candidates ordered by `se`, one row each: board tag, `(edge, depth)`, and a bar split into its `bp` and `×w` components. Rows are colored by board, so a child board being demoted by the failure discount is visible as a whole block sinking at once. Each row carries its green / not-green badge, and a persistent marker tracks **best green currently at #k** across the whole scrub — watching that number sink as the failure discount demotes a green candidate's board is the mechanism made visible.
- **C. Timeline** — sims 1..N as ticks marked pass/fail, with a scrubber. Moving it re-sorts B, updates the dimming in A, and switches D to the board the popped entry belonged to.
- **D. Model ranking vs truth** — rank space, not value space. Left: the current board's 60×5 model grid colored by the model's own rank or percentile within that grid. Right: the same grid with green cells only. There is no raw-value difference panel; the comparison is summarized by the single number *rank of best green*, the same quantity the index sorts on.

Cross-highlighting is global: hovering a heatmap cell in D lights the corresponding contact point in A and the corresponding row in B, and vice versa.

## Side deliverables

- `docs/experiments/horizon_q_model_registry.md`: add a consistent `train_h5:` field to every entry. Entries that never recorded their training H5 get `train_h5: unrecorded` rather than an inferred value.
- `docs/experiments/eval_set_registry.md`: add a schema block for `testset_gt.h5` — keys, shapes, dtypes, the `node_kind` row counts, and the two join keys — so the 214 MB file never has to be opened again to answer this question.

## Verification

- `--trace-out` unset produces output byte-identical to the unpatched script on a smoke set.
- Trace and GT join: for a sampled episode, the pops in the trace all appear in that board's `pool`, and every `(edge, depth)` in `pure2push.json`'s `tried_first_push` has a corresponding nonzero cell in the GT root grid.
- Episode counts: index shows 1018 episodes for the model arm, 981 with GT, tier counts 238/409/371.
- The page loads and scrubs an episode end to end with no console errors, served from a plain `http.server`.
