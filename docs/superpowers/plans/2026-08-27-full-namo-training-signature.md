# Full NAMO Training-Geometry Signature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export HY5U's private training-room geometry into a compact, reproducible signature artifact and consume it in the held-out Full NAMO population builder.

**Architecture:** `verify_geom_disjoint.py` remains the single owner of canonical geometry signatures and gains validation for the versioned compact reference. A focused exporter produces the artifact from existing H5/TXT/JSON inputs, while `build_full_namo_population.py` unions direct and compact references without changing candidate selection semantics.

**Tech Stack:** Python 3.12, standard library, existing `h5py` loader path, pytest, canonical NAMO geometry helpers

**Engineering Standards:** Follow `plan-coding-standards`. Reuse `geom_sig`, `load_xmls`, and `sig_map`; hash sources once; sort deterministic sets once; fail before output on incomplete geometry; use contextual validation errors; avoid environment-specific paths in source; add only the four behavior tests needed for reproducibility and fail-closed verification; end each coherent stage with a commit.

---

### Task 1: Specify the compact exporter contract

**Files:**
- Create: `python/tests/test_export_geom_signatures.py`
- Create later: `scripts/pipeline/export_geom_signatures.py`

- [ ] **Step 1: Add one deterministic round-trip test**

Create two minimal XML fixtures and a training TXT manifest, call the wished-for `export_signatures(train_specs=[manifest], out_path=artifact, workers=1)`, and assert the schema, source SHA-256, measured counts, sorted canonical full signatures, and sorted wall signatures.

- [ ] **Step 2: Add one incomplete-export test**

Reference one valid and one unparseable XML, assert `ValueError` names the unparseable count, and assert the output file was never created.

- [ ] **Step 3: Verify RED**

Run `source env.robotlearning.sh && "$NAMO_PYTHON" -m pytest python/tests/test_export_geom_signatures.py -v`. Expected: import failure because `export_geom_signatures.py` does not exist.

- [ ] **Step 4: Commit the test contract**

Run `git add python/tests/test_export_geom_signatures.py && git commit -m "test: specify training geometry signature export"`.

### Task 2: Implement deterministic export and strict loading

**Files:**
- Create: `scripts/pipeline/export_geom_signatures.py`
- Modify: `scripts/pipeline/verify_geom_disjoint.py`
- Test: `python/tests/test_export_geom_signatures.py`

- [ ] **Step 1: Add the shared schema loader**

Define `SIGNATURE_REFERENCE_SCHEMA = "namo-room-geometry-signatures-v1"` and `load_signature_reference(path)` in `verify_geom_disjoint.py`. Validate the schema, required dictionaries/lists, sorted uniqueness, lowercase 32-character hexadecimal MD5 values, nonempty signature sets, and equality between list lengths and the corresponding unique-signature counts; return full signatures, wall signatures, and the parsed provenance object.

- [ ] **Step 2: Implement the focused exporter**

Implement `export_signatures(*, train_specs, out_path, workers)` by resolving each source path, hashing each source file once, loading every XML through `load_xmls`, canonicalizing and deduplicating paths, calling `sig_map`, rejecting empty or partially unparseable inputs, and writing deterministic indented JSON with exclusive-create mode only after validation succeeds. Expose repeated required `--train-xmls`, required `--out`, and optional `--workers` CLI flags, and log measured path/signature counts plus the output path.

- [ ] **Step 3: Verify GREEN**

Run `source env.robotlearning.sh && "$NAMO_PYTHON" -m pytest python/tests/test_export_geom_signatures.py -v`. Expected: both tests pass.

- [ ] **Step 4: Commit the exporter**

Run `git add scripts/pipeline/export_geom_signatures.py scripts/pipeline/verify_geom_disjoint.py && git commit -m "feat: export training geometry signatures"`.

### Task 3: Consume compact references in the population builder

**Files:**
- Modify: `python/tests/test_build_full_namo_population.py`
- Modify: `scripts/pipeline/build_full_namo_population.py`

- [ ] **Step 1: Add one builder integration test**

Create a valid candidate and an exact-geometry training fixture, export the compact artifact with the real exporter, build using `signature_specs=[artifact]`, and assert the leaked candidate is dropped, a nonleaking candidate remains, and `population_audit.json` records the artifact path, artifact SHA-256, and embedded source provenance.

- [ ] **Step 2: Add one corrupted-reference test**

Change the artifact's declared `unique_room_signatures` count without changing its list, assert a contextual `ValueError`, and assert no population output exists.

- [ ] **Step 3: Update the existing CLI contract test**

Keep the existing no-training-reference assertion but require the error to state that at least one of `--train-xmls` or `--train-signatures` is mandatory.

- [ ] **Step 4: Verify RED**

Run `source env.robotlearning.sh && "$NAMO_PYTHON" -m pytest python/tests/test_build_full_namo_population.py -v`. Expected: failures because `signature_specs` and `--train-signatures` are unsupported.

- [ ] **Step 5: Implement compact-reference union and audit provenance**

Extend `_training_signatures` to merge direct and compact full/wall signature sets, retain direct XML counters, and record direct-reference count, compact-reference count, merged unique signature counts, compact artifact SHA-256, and embedded source provenance. Add optional `signature_specs=()` to `build_population`, add repeated `--train-signatures`, and call `parser.error` unless at least one direct or compact reference is supplied.

- [ ] **Step 6: Verify GREEN and regressions**

Run `source env.robotlearning.sh && "$NAMO_PYTHON" -m pytest python/tests/test_build_full_namo_population.py python/tests/test_export_geom_signatures.py -v`, followed by `"$NAMO_PYTHON" -m pytest full_namo_sim_exp/tests python/tests/test_build_full_namo_population.py python/tests/test_export_geom_signatures.py -q`. Expected: every test passes without warnings or failures.

- [ ] **Step 7: Commit builder integration**

Run `git add python/tests/test_build_full_namo_population.py scripts/pipeline/build_full_namo_population.py && git commit -m "feat: consume training geometry signatures"`.

### Task 4: Document, verify, and prepare the private export

**Files:**
- Modify: `.claude/skills/namo-data-pipeline/SKILL.md`
- Modify: `full_namo_sim_exp/README.md`
- Modify: `docs/experiments/log/EXP-2026-08-27-full-namo-heldout-testset.md`

- [ ] **Step 1: Document exact cross-account commands**

Document owner-side `export_geom_signatures.py --train-xmls ...hybrid_train_v1.h5 --out ...json`, SHA-256 verification, transfer to `/scratch/tdn39/full_namo_heldout_v1/artifacts/`, and builder use through `--train-signatures`. Replace the card's inaccessible checkpoint path with the verified staged checkpoint and hash.

- [ ] **Step 2: Run final verification**

Run the complete focused and Full NAMO suites, `"$NAMO_PYTHON" -m py_compile scripts/pipeline/export_geom_signatures.py scripts/pipeline/build_full_namo_population.py scripts/pipeline/verify_geom_disjoint.py`, `git diff --check`, and confirm no uncommitted files outside the intended documentation changes.

- [ ] **Step 3: Commit and push documentation**

Run `git add .claude/skills/namo-data-pipeline/SKILL.md full_namo_sim_exp/README.md docs/experiments/log/EXP-2026-08-27-full-namo-heldout-testset.md docs/superpowers/plans/2026-08-27-full-namo-training-signature.md && git commit -m "docs: register training signature handoff"`, then push `feat/full-namo-heldout-v1`.

