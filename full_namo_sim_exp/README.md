# Full NAMO simulation experiment

## Reproduce the frozen-300 difficulty figures

The current paper plots compare **PAVE (HY5U S2 search)** with **Random (median per environment across five uniform seeds)**. Use `--layout separate` to export six independent figures: simulator budget and wall-clock budget for each of easy, medium, and hard. Every figure has only two smooth, solid curves, with difficulty identified in the title. All three simulator axes share the measured campaign cap of 9,000 pushes, and all three time axes share the same range of measured wall-clock planning seconds. Each difficulty has its own denominator of 100 environments, including failures. Policy and Geometric are not plotted in this view.

The shared project palette is in `figure_style.py`: PAVE green `#41C95A`, PAVE Policy blue `#4C78A8`, Random gray `#7A7A7A`, and Geometric red `#D62728`. Reuse `METHOD_COLORS` and `DIFFICULTY_LINESTYLES` in future figures.

From the `namo_cpp` repository root on dhruv-linux:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m full_namo_sim_exp.plot \
  --layout separate \
  --out-dir full_namo_sim_exp/plots/frozen300_separate_20260913 \
  --smooth-anchors 32 --dpi 600
```

Choose a new output directory for each rerun; an existing directory is rejected. `--snapshot /path/to/normalized_outcomes.json` can select another compatible frozen snapshot. The bundled default is `data/frozen400_timed_20260913.json`, SHA-256 `1887e5227715bf92f6d8bc7894d1e2799e69f739b1be2c5f1fbffbe4b0e8078e`, copied from the paper's `figures/data/full_namo_frozen400_timed_20260913/normalized_outcomes.json`. It contains the completed Amarel Icelake campaign `full_namo_frozen400_icelake_20260912_v1`; measurements use Intel Xeon Platinum 8358 CPUs. Rendering these saved measurements on dhruv-linux does not perform or retime any simulations.

`results.load_frozen_results` requires matched problem IDs and XML hashes in every required arm, then selects the existing 100 easy, 100 medium, and 100 hard labels. It excludes exactly the source's 100 frozen unresolved-label tasks; failures within the selected 300 remain included. Random seeds are 7000, 8000, 9000, 10000, and 11000. Failed runs have infinite cost to success, so a Random median is successful by a budget only when at least three of the five seeds solve that environment by that budget. Push and wall-time medians are computed independently before building success curves.

With `--layout separate`, outputs are `success_vs_simulator_budget_{easy,medium,hard}.{png,pdf}` and `success_vs_wall_clock_budget_{easy,medium,hard}.{png,pdf}` (six 600-DPI rasters and six vector PDFs, each containing one plot), `exact_curves.csv`, `display_curves.csv`, `per_environment_costs.csv`, and `metadata.json`. No combined figure is generated. Metadata records the source hash, included/excluded geometry IDs, code revision and hashes, package versions, palette, layout, line styles, denominator counts, smoothing anchors, and original measurement provenance. Smooth curves use monotone PCHIP interpolation through exact rates at 32 common log-spaced anchors, supplemented with integer budgets 1–10. Intermediate display values are interpolations; use `exact_curves.csv` for numerical claims.

The previous two-figure view remains available with `--layout overlaid` (the backward-compatible default): each metric has all three difficulties, using easy solid, medium dashed, and hard dotted lines. Its filenames end in `_by_difficulty.{png,pdf}`. Both layouts use identical underlying costs and exact/display CSV values.

The existing `namo312` interpreter already has the required packages. For a fresh plotting-only Python 3.12 environment, install `python -m pip install -r full_namo_sim_exp/requirements-plot.in`; no simulator bindings or model checkpoint are needed. Pins were verified with Python 3.12.13. The original interleaved experiment pipeline below remains available for its original data format.

## Original interleaved experiment pipeline

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

Every arm receives one shared 20,000-simulator-call budget for the complete Full NAMO problem. The budget is not reset at keyhole boundaries, and reaching it is recorded as a censored failure.

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
