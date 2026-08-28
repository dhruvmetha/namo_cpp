# Full NAMO Held-Out Population Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, reproducible builder for a fresh exact-two-boundary, structurally valid, geometry-disjoint Full NAMO evaluation population and prepare its committed generation campaign.

**Architecture:** A single committed pipeline script consumes the existing generator manifest, zero-simulation topology probe, and registered training-room references. It reuses the existing topology drop rules and geometry signatures, then emits a deterministic frozen population plus a complete audit for the already implemented `full_namo_sim_exp` runner.

**Tech Stack:** Python 3.12, standard library, existing NAMO pipeline helpers, pytest, SLURM generation/probe scripts

**Engineering Standards:** Follow `plan-coding-standards`. Keep joins on canonical realpaths, reuse existing topology and geometry helpers, preserve room-level holdout and floorplan clustering, use named domain constants, produce contextual failures, write outputs only after all validation succeeds, commit each coherent stage, and require focused plus Full NAMO regression tests before launch.

---

### Task 1: Specify the population builder with failing tests

**Files:**
- Create: `python/tests/test_build_full_namo_population.py`

- [ ] **Step 1: Add reusable XML and probe fixtures**

Create minimal MuJoCo-shaped XML fixtures containing named wall and movable-obstacle geoms, a candidate manifest, and one JSONL probe row per candidate. Import `scripts/pipeline/build_full_namo_population.py` through the existing script-directory test pattern.

- [ ] **Step 2: Add the acceptance, structural-drop, and train-leak test**

The test must supply one valid scene, one scene with `no_pushable_blocker=true`, and one scene whose full geometry matches a training XML. Assert that only the valid scene enters `population.json`, its cluster is `floorplan:<walls-signature>`, and `dropped_scenes.jsonl` reports explicit `no_pushable_blocker` and `training_geometry_leak` reasons.

- [ ] **Step 3: Add fail-closed contract tests**

Add focused tests that assert contextual errors for a missing probe row, an extra probe row, duplicate manifest paths after `realpath`, duplicate probe paths after `realpath`, and any pre-existing output file.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest python/tests/test_build_full_namo_population.py -v
```

Expected: collection or import failure because `build_full_namo_population.py` does not exist.

- [ ] **Step 5: Commit the test contract**

```bash
git add python/tests/test_build_full_namo_population.py
git commit -m "test: specify Full NAMO population builder"
```

### Task 2: Implement the deterministic population builder

**Files:**
- Create: `scripts/pipeline/build_full_namo_population.py`
- Test: `python/tests/test_build_full_namo_population.py`

- [ ] **Step 1: Implement canonical input loading**

Add functions that load a line manifest and probe JSONL keyed by `os.path.realpath`, reject duplicate identities, require exact candidate/probe key equality, and compute SHA-256 hashes of both source files.

- [ ] **Step 2: Reuse structural and geometry rules**

Import `DROP_RULES` from `probe_static_topology` and `geom_sig`/`load_xmls` from `verify_geom_disjoint`. Convert each truthy drop flag plus `goal_in_free_space == false` into explicit structural reasons. Build the union of full training-room signatures from every repeated `--train-xmls` input.

- [ ] **Step 3: Build accepted and dropped records**

For every candidate in sorted realpath order, record structural reasons first, then `unparseable_geometry` or `training_geometry_leak` as applicable. Accept all remaining scenes without consulting any outcome field and assign `cluster_id = "floorplan:<walls-signature>"`.

- [ ] **Step 4: Write all frozen outputs without overwrite**

Write deterministic `population.json`, `accepted_scenes.txt`, `dropped_scenes.jsonl`, and `population_audit.json`. The audit must include source hashes, input/accepted/dropped totals, reason counts, unique full-room and floorplan counts, room-variant counts, train-reference counts, exact scene leaks, and floorplan overlap. Refuse to run if any target already exists.

- [ ] **Step 5: Expose the CLI**

Provide required `--manifest`, `--probe-jsonl`, repeated `--train-xmls`, `--name`, `--expect-hop`, and `--out-dir` flags. Log each output path and a concise accepted/input/leak summary.

- [ ] **Step 6: Run focused and regression tests and verify GREEN**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest python/tests/test_build_full_namo_population.py -v
"$NAMO_PYTHON" -m pytest full_namo_sim_exp/tests python/tests/test_build_full_namo_population.py -q
```

Expected: every test passes with no warnings or failures.

- [ ] **Step 7: Commit the builder**

```bash
git add scripts/pipeline/build_full_namo_population.py
git commit -m "feat: build held-out Full NAMO populations"
```

### Task 3: Document and register the new pipeline stage

**Files:**
- Modify: `.claude/skills/namo-data-pipeline/SKILL.md`
- Modify: `full_namo_sim_exp/README.md`

- [ ] **Step 1: Add the builder to the data-pipeline inventory**

Document that `build_full_namo_population.py` converts generated exact-hop scenes plus the zero-simulation probe and geometry leak references into the frozen Full NAMO population, preserving complete-scene identity and floorplan clusters.

- [ ] **Step 2: Add the population-build command to the experiment README**

Show the exact CLI shape before `pipeline validate` and state that final generation must use fresh seeds and registered training references. Retain the existing rule that no success filtering happens after freezing.

- [ ] **Step 3: Verify documentation and commit**

Run `git diff --check`, then:

```bash
git add .claude/skills/namo-data-pipeline/SKILL.md full_namo_sim_exp/README.md
git commit -m "docs: register Full NAMO population build"
```

### Task 4: Prepare and launch the fresh exact-two-boundary generation campaign

**Files:**
- Create: `docs/experiments/log/EXP-2026-08-27-full-namo-heldout-testset.md`

- [ ] **Step 1: Resolve immutable experiment inputs**

Read the evaluated-artifact registry and final model lineage to record the exact HY5U checkpoint, complete training-corpus room references, generator commit, `EXACT_HOP=2`, a previously unused seed range, templates, requested population size, and output root. Do not infer any path from a directory nickname.

- [ ] **Step 2: Write the live experiment card**

Record the user's exact-two-boundary paper decision, the zero-simulation structural rules, geometry-disjointness gate, no-success-filtering rule, generation/probe/build commands, and the fact that the old two-hop and characterized three-hop pools are excluded from the final claim.

- [ ] **Step 3: Run all tests and commit before launch**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest full_namo_sim_exp/tests python/tests/test_build_full_namo_population.py -q
git diff --check
```

Commit the experiment card and stamp its resulting SHA into the card with a follow-up metadata commit before submission.

- [ ] **Step 4: Launch generation only on the registered compute host**

Use `scripts/slurm/multihop_aug9_generate.slurm` with the committed card's `OUT`, `NUM_ENVS`, `SEED_BASE`, and `EXACT_HOP=2`. Record the scheduler job ID immediately in the card. Do not launch if the repository is dirty, the seed range is unresolved, or the training references needed for the later geometry gate are not registered.

- [ ] **Step 5: On generation completion, probe, build, and freeze**

Create the complete generated manifest by realpath, run `probe_static_topology.py --expect-hop 2`, invoke `build_full_namo_population.py`, review `population_audit.json`, and stop if any unexplained drop or leak appears. Then point a copied `full_namo_sim_exp/experiment.example.json` at the frozen population and run `python -m full_namo_sim_exp.pipeline validate` to produce the immutable experiment lock.
