# Gotcha: one room (xml) has MANY episodes — never key on the room

**The invariant, stated once:** a single scene file (`xml_file`) is run by the planner as **multiple independent episodes**. Each episode picks a *different* target object and a *different* goal region, and produces its *own* exhaustive 1-push label set (`tried` / `valid`) and its *own* data sample. In the v3 test set, **402 of 1228 rooms (≈33%)** have >1 episode; in `v3_1push_le10` training, **6207 of 21073 rooms (≈30%)** contribute 2–4 samples each.

So `xml_file` is **not** a unique key for a problem. The unique key is the **(pushed object, goal region)** — in practice the **pushed object's initial pose** (`local_tight_object_center`, which matches the episode's object pose to ~0 mm) plus the goal region.

Treating the room as the unit of work is the single mistake behind a whole class of bugs. It bit us in three places (June 2026); all three are fixed, and the rules below keep them fixed.

---

## The three failure modes (and the rule that prevents each)

### 1. Eval scored a sample against the WRONG episode's answer key
A per-`xml` validset stored only one episode's `valid`/`tried`. A test sample that actually pushed a *different* object in the same room got graded against that one stored episode → a **correct** prediction was logged as `not_reachable` (a failure). The "correctness check" `gt_in_valid_frac` sat at ~75% instead of ~100%.

> **Rule:** the answer key is **per episode**, keyed by the pushed object. At eval time, match each
> sample to the episode whose `object_center` is nearest the sample's `object_center` (must be
> ≤ 0.01 m) **and** whose `valid` contains the sample's GT push.
> Enforced by: `build_episode_validsets.py` (emits `{xml: [episode, ...]}`) and
> `eval_grounding.py::match_episode`.

### 2. Difficulty buckets were assigned per ROOM, not per sample
`build_test_divisions.py` binned a *room* hard/med/easy by one episode's solve_rate, then NPZ-gen dumped samples for **all** of that room's episodes into that one division's H5. Result: the "hard" H5 was **24% non-hard** samples (easy/med episodes that rode along). Per-division metrics were meaningless at face value, and the random floor on "hard" read 14% instead of the true ~3%.

> **Rule:** difficulty is a property of the **episode**, not the file it landed in. Re-bin every
> sample by its **matched episode's** solve_rate (`hard < 0.05`, `med < 0.30`, else `easy`); do not
> trust the H5 filename/division. Pool all division H5s and dedup samples that repeat across files
> (a pkl selected into two buckets emits its samples twice).
> Enforced by: `eval_grounding.py::aggregate` (pools, re-bins by true solve_rate, dedups by
> `(xml, object_center, gt)`).

### 3. Train/val split scattered same-room siblings
`se2_data_cropped.py` split the H5 90/10 by shuffling **rows**, ignoring rooms. Sibling samples (same scene, different target object) landed on both sides → **42% of val samples shared their room with a train sample** → `val_loss` was optimistic (model "validated" mostly on rooms it trained on). The held-out **test** set was unaffected (separate pkl pool, 0 room overlap), so model-quality conclusions stood; only `val_loss` / checkpoint selection were biased.

> **Rule:** split train/val (and any held-out set) by **room (xml)**, never by row. A scene is wholly
> in train or wholly in val. Group rows by `xml`, shuffle the *rooms*, fill train to the target
> fraction of *samples*.
> Enforced by: `se2_data_cropped.py` (room-grouped split, June 2026).

---

## The general rule for any new analysis or split

1. **Unit of work = (pushed object, goal region), not the xml.** If your code groups, dedups, splits, or labels by `xml` alone, it is probably wrong for multi-episode rooms.
2. **Match samples to episodes by object pose** (`object_center` ~0 mm), then confirm the goal via `gt ∈ valid`.
3. **Carry difficulty per sample/episode**, never inherit it from a file or pkl.
4. **Hold out by room**, so a scene never appears in both train and eval.

A good example of doing it right *before* this came up: the 2-push eval (`docs/pipeline/difficulty_stratification.md`) already keys problems by `(env, region_label::object_id)` — object + region, exactly the unit above.

## Latent version not yet triggered

`emit_informative_manifest.py` selects a **whole pkl** if *any* episode falls in a solve-rate band, on the stated (false) assumption "~1 episode per pkl." `v3_1push_le10` is fine because it's an all-difficulty set used uniformly. **But** building a difficulty-*filtered* training set this way would dilute it with the pkl's other (out-of-band) episodes — the exact analog of failure mode #2. If you build a filtered set, filter **per episode** at NPZ-gen, not per pkl.

## Failure mode #4: name-based disjointness checks (the test-set trap)

Train and test reference the **same physical rooms under incompatible path schemes** — train via `outputs/v3_*_phase1/...run_NNNN_env_NNNN_pair...` SYMLINKS, the test pool via `car_envs/v3/test/...` real paths, and `run_NNNN` even *repeats across shards* for unrelated rooms. So a name-based "0 overlap with train" check is **meaningless** and silently passes on a leaky split. **Always verify test/train disjointness by ROOM GEOMETRY** — `md5(sorted wall pos/size/euler + sorted obstacle pos/size/euler)`, goal + robot excluded (committed: `scripts/pipeline/verify_geom_disjoint.py`). The canonical car test set built this way is `namo_testset_v1` (`docs/pipeline/canonical_testset.md`): 2173 scenes, geometry-proven 0-leak.

Related gotcha: the canonical 1-push eval key is **`v3_test_episodes.json`** (per-xml LIST of episodes WITH `object_center`, the thing `eval_scorer.py --episodes` consumes). `v3_test_validsets.json` is a simpler 1-per-xml form with NO `object_center` — **not** the eval key; don't confuse them.

## Failure mode #5: basename joins on the multi-hop pools

The multi-hop Full-NAMO scene pools (`multihop_aug9_hy5u/`) have only **800 unique basenames across 2,535 scenes**, because generation writes `set{1,2}/benchmark_{1..5}/run_XXXX/env_XXXX_pair_YYY.xml` and `run_XXXX` repeats across templates. A join keyed on basename therefore collides massively and silently.

Measured 2026-08-17: the same join scored **0/9 matches on basename and 9/9 on `realpath`**. Left unnoticed it would have mislabeled ~68% of the corpus — every scene would inherit some other scene's difficulty tier.

**Always join these pools on `os.path.realpath`.** This is the same family as failure mode #4 (name-based reasoning about rooms), but it bites joins rather than disjointness checks, and it bites within a single pool rather than across train/test.

## Note on script locations

`build_episode_validsets.py` and the scorer-data builders (`build_scorer_dataset.py`, `add_contact_px.py`) are now **committed under `scripts/pipeline/`** (promoted from sandbox 2026-06-08); the scorer dataset is registered at `config/datasets/v3_scorer_e4.yaml` with lineage in `docs/pipeline/scorer_dataset.md`. `eval_grounding.py` still lives under `scripts/sandbox/` (gitignored) — promote next. This doc is the durable record of the *rules*; the training-side fix is in the committed `sage_learning` repo (`src/data/se2_data_cropped.py`).
