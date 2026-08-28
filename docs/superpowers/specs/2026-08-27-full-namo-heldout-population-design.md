# Full NAMO Held-Out Population Design

## Goal

Build and freeze a fresh, geometry-disjoint, exact-two-boundary Full NAMO simulation population for the final HY5U-versus-Random paper experiment.

## Population definition

One population item is one complete XML scene with one XML goal. At the initial unmodified state, the deterministic shortest robot-region-to-goal-region path must cross exactly two boundaries (`hop_count == 2`). This is a two-boundary Full NAMO task, not a local two-push episode; each boundary opener may itself search chains up to two pushes.

The final population must be generated from a new seed range after the method and protocol are frozen. The previously evaluated two-hop pool and the already characterized three-hop pool are not eligible for the final held-out claim.

## Validity and filtering

Generation and validation happen before the population is frozen. Reuse `scripts/slurm/multihop_aug9_generate.slurm` with `EXACT_HOP=2` and reuse `scripts/pipeline/probe_static_topology.py` for the zero-simulation structural audit.

Reject only structural defects already defined by `probe_static_topology.DROP_RULES`: probe errors, no initial path, hop mismatch, no blocking object on the first boundary, no reachable first blocker, no pushable first blocker, or an XML goal outside initial free space. Do not run HY5U, Random, exhaustive opening search, or any success oracle while selecting scenes.

After freezing, no scene is filtered for any reason. Runtime errors invalidate a campaign shard instead of shrinking the denominator, as already enforced by `full_namo_sim_exp`.

## Leakage and identity

Reuse `scripts/pipeline/verify_geom_disjoint.py::geom_sig`. Train/test leakage is equality of the complete room signature over walls and initial movable-obstacle geometry; filenames and generator seeds are not evidence of disjointness.

The builder accepts one or more registered training-corpus references and excludes every candidate whose full room signature appears in any reference. It also records floorplan overlap without treating it as scene leakage, matching the canonical region-opening test-set rule.

All joins use `os.path.realpath`; basenames are forbidden because the multi-hop pools contain extensive basename collisions. Duplicate candidate paths or duplicate probe rows are errors rather than silently deduplicated records.

Every accepted scene receives `cluster_id = "floorplan:<walls-signature>"`. Multiple generated tasks may share a floorplan, and the existing Full NAMO statistics must resample these clusters rather than treating every variant as independent.

## Builder and outputs

Add `scripts/pipeline/build_full_namo_population.py`. Inputs are the generated scene manifest, static-probe JSONL, one or more training XML/H5/JSON/TXT references, the expected hop count, a population name, and an output directory.

The builder requires exact candidate/probe population equality, verifies every structural decision from the probe row, computes geometry signatures, audits training leakage, and writes deterministically ordered outputs:

- `population.json`: the immutable input consumed by `full_namo_sim_exp`, containing the name and `{xml_path, cluster_id}` scenes;
- `accepted_scenes.txt`: one canonical realpath per accepted scene;
- `dropped_scenes.jsonl`: every structurally invalid, unparseable, or training-leaked candidate with explicit reasons;
- `population_audit.json`: input, acceptance, structural-drop, leakage, floorplan, and geometry-duplication counts plus SHA-256 hashes of the source manifest and probe.

The builder refuses to overwrite any existing output file so a frozen population cannot be mutated accidentally.

## Verification and launch gate

Focused tests cover exact probe matching, structural rejection, geometry leakage, floorplan clustering, realpath identity, deterministic output, and overwrite refusal. The complete existing `full_namo_sim_exp` suite must remain green.

Before generation launches, create and commit an experiment card containing the new seed range, exact generator command, expected hop count, structural rules, registered HY5U checkpoint, complete training references, and output roots. Run the builder only after generation and probing complete, inspect the audit, then cryptographically freeze the experiment through `full_namo_sim_exp.pipeline validate`.
