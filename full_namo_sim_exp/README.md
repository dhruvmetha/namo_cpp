# Full NAMO simulation experiment

This directory is the complete experiment-local path for the final held-out Full NAMO comparison between the fixed Sage Hybrid ranker and uniform-random search ordering. The core NAMO planner and simulator remain library dependencies; every experiment-specific launcher, timer, validator, aggregator, statistical analysis, and publication plot lives here.

## What was missing

- The existing Full NAMO launch, aggregation, and one-random-seed comparison code was scattered across `scripts/slurm` and `scripts/pipeline`.
- The existing runner did not persist whole Full NAMO planning time per scene.
- Separate arm jobs did not guarantee matched hardware for wall-clock comparisons.
- The aggregator silently deduplicated rows and did not require exact equality with a frozen population.
- No analysis combined exactly five Random seeds while respecting the fact that their `5N` rows reuse the same `N` scenes.
- No paired scene/room-cluster confidence interval existed for the terminal Full NAMO success difference.
- The final green, log-axis, no-band figure existed only in the paper repository.

## Data contract

The frozen population is a JSON object with a name and a `scenes` list. A scene may be a path string or `{"xml_path": "...", "cluster_id": "..."}`. Use `cluster_id` for all scenes derived from the same base room or template so the bootstrap does not treat correlated variants as independent.

Generation validity and train/test leakage checks happen before this manifest is frozen. Once frozen, the experiment performs no success-based or exhaustive-success filtering: every scene must appear exactly once in all six arms and remains in every denominator.

The experiment runner also bypasses the base exact-path-length selector. `protocol.path_length` is declared population metadata written to each row; it is never recomputed or used to include/exclude a scene. Invalid files or runtime exceptions invalidate the campaign shard instead of silently shrinking the denominator.

Copy `experiment.example.json` outside Git, set the frozen population, run root, checkpoint, exact protocol, five Random seeds, pinned SLURM partition/CPU constraint, predeclared reporting cutoffs, and bootstrap seed. One config represents one exact path-length population; use separately frozen configs if multiple hop populations will be reported separately.

## Building the held-out population

Generate the final scenes only after the method and protocol are frozen, using a fresh seed range with `scripts/slurm/multihop_aug9_generate.slurm`. Run `scripts/pipeline/probe_static_topology.py` over every generated XML with the same exact hop count, then build the population from the complete generated manifest, probe JSONL, and every registered training-room reference:

```bash
"$NAMO_PYTHON" scripts/pipeline/build_full_namo_population.py \
  --manifest "$GENERATED_MANIFEST" \
  --probe-jsonl "$STATIC_PROBE_JSONL" \
  --train-xmls "$FINAL_TRAIN_XML_REFERENCE" \
  --name full-namo-two-boundary-heldout-v1 \
  --expect-hop 2 \
  --out-dir "$NAMO_MANIFESTS/full_namo_two_boundary_heldout_v1"
```

Repeat `--train-xmls` when the final model's rooms come from multiple registered corpora. The builder requires exact manifest/probe equality, applies only the probe's zero-simulation structural rules, rejects full-room geometry leaks, assigns floorplan cluster IDs, refuses overwrite, and writes `population.json`, `accepted_scenes.txt`, `dropped_scenes.jsonl`, and `population_audit.json`. Review the audit before configuring a run; no result-producing method belongs anywhere in this build step.

## Pipeline

Activate the target machine environment from the repository root, then export `HY5U_CHECKPOINT` to the exact registered final checkpoint:

```bash
source env.amarel.sh
```

Validate and cryptographically freeze the complete configuration before launch. Both NAMO and the sibling Sage scorer repository must be clean. This writes `experiment.lock.json` with hashes of the experiment, population, every scene XML, every matching primitive-profile file, checkpoint, NAMO config, NAMO/Sage commits, loaded `namo_rl` extension, linked MuJoCo library, Python executable, and relevant package versions. An existing different lock is never overwritten:

```bash
"$NAMO_PYTHON" -m full_namo_sim_exp.pipeline validate --experiment /path/to/experiment.json
```

Print the SLURM launch command, including the campaign's predeclared partition and CPU constraint:

```bash
"$NAMO_PYTHON" -m full_namo_sim_exp.pipeline launch-command --experiment /path/to/experiment.json
```

Each array task evaluates the same shard sequentially on Sage Hybrid and all five Random seeds. Arm order rotates by shard, workers remain one, and all six measurements occur on the same node without contention. `runner.py` times the complete `FullNAMOPlanner.search` call, which includes every region-opening call and replanning step but excludes environment construction and one-time model loading.

`protocol.evaluation_seed` is shared by every arm and fixes snapshot, goal-sampling, and model randomness. The model seed and five `random_seeds` alter only the best-first edge-ordering RNG. This separation is required for paired scene-level comparisons: changing a Random ordering seed must never change the task geometry being evaluated.

After every shard finishes, aggregate, validate, compute statistics, and render the figure:

```bash
"$NAMO_PYTHON" -m full_namo_sim_exp.pipeline all --experiment /path/to/experiment.json
```

Outputs are written under the configured run root:

- `raw/shard_*/<arm>/`: atomically published, immutable per-shard runner outputs. Interrupted attempts remain under `staging/` and can be inspected without blocking a clean retry.
- `aggregate/<arm>/`: exact-population `solved.jsonl`, `unsolved.jsonl`, and summary.
- `analysis/full_namo_statistics.json`: final counts, five Random seed rates and sample SD, paired percentage-point effect, room-cluster bootstrap 95% CI, Holm-adjusted per-seed McNemar sensitivity tests, and predeclared cutoff results.
- `plots/full_namo_success_vs_cost.{pdf,png}`: wall time and cumulative simulator calls on logarithmic axes, positive-green model line, gray five-seed Random mean, no band, and compact terminal fractions.

The pooled Random tail fraction `X/(5N)` is descriptive. The paired bootstrap resamples frozen population clusters and compares the model result on scene `i` with the mean of Random's five outcomes on that same scene; it never treats `5N` as independent test examples.

## Verification

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest full_namo_sim_exp/tests -q
```
