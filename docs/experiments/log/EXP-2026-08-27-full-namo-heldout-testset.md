---
type: experiment
status: planned
created: 2026-08-27
thread: full-namo-heldout
robot: car
commit: pending-launch-stamp
metric: held-out Full NAMO success versus total simulator calls and whole-process wall time
tags: [experiment, full-namo, heldout-testset, exact-two-boundary, hy5u, random-baseline, amarel]
---

# Fresh exact-two-boundary Full NAMO held-out test set

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a learned ranker inside simulator-verified local region-opening search; this experiment measures the composed Full NAMO planner, not a standalone success classifier or multi-hop predictor.

## Paper decision

The final simulation population uses complete scenes whose initial deterministic shortest robot-region-to-XML-goal path crosses exactly two region boundaries (`hop_count == 2`). This matches the two-boundary real-robot setup while remaining nontrivial because each local boundary opener may search a chain of up to two pushes, so a nominal two-boundary scene can require up to four pushes and replanning can change the executed boundary count.

The old exact-two-hop pool under `/scratch/dm1487/multihop_aug9_hy5u/scale_20260817_0000` and the already characterized three-hop pool are excluded from the final held-out claim. The population below is generated only after the method, denominator, and evaluation protocol are frozen.

## Frozen method and denominator

One population item is one complete generated XML scene and its XML goal. Generation and selection use no HY5U score, Random score, exhaustive opening result, planner success, or runtime outcome.

Before freezing, reject only zero-simulation structural defects from `probe_static_topology.DROP_RULES`: probe error, no initial path, exact-hop mismatch, no blocking object on boundary zero, no reachable boundary-zero blocker, no pushable boundary-zero blocker, or XML goal outside initial free space. Reject any candidate whose complete room geometry signature over walls and initial movable obstacles appears in HY5U's training corpus.

After freezing, every accepted scene remains in every denominator. A runtime error invalidates the affected campaign shard instead of removing a scene. Floorplan reuse is recorded as `cluster_id = floorplan:<walls-signature>` and the paired confidence interval resamples these clusters.

## Immutable inputs

| field | value |
|---|---|
| HY5U checkpoint | `/cache/home/dm1487/aquaman0/ckpts_bfix/HY5U_s2.ckpt` |
| model choice | HY5U seed 2, the registered best seed of the registered best deployable HY5U arm |
| complete training reference | `/common/users/dm1487/scratch_namo/aquaman/round0/hybrid_train_v1.h5` |
| training-reference meaning | 1,302,659 rows: 257,409 old-corpus roots plus 1,045,250 family0 child rows; the H5 `xml` column is the complete HY5U room reference |
| generator | `scripts/slurm/multihop_aug9_generate.slurm`; implementation last changed at `ea660909879fa332671074895f17d7daccf01ef6` |
| templates | all ten `mujoco_env_creator/templates/aug9_car_v3/{set1,set2}/benchmark_{1..5}.xml` templates |
| exact-hop gate | `EXACT_HOP=2` |
| new seed base | `827000000` |
| generation shape | array `0-399`, `NUM_ENVS=10`: 4,000 requested layouts, 400 per template |
| derived start-seed range | `827000000` through `836390000`; each task requests ten consecutive generator seeds |
| generated root | `/scratch/dm1487/full_namo_heldout_v1/generation` |
| population root | `/scratch/dm1487/full_namo_heldout_v1/population` |
| evaluation root | `/scratch/dm1487/full_namo_heldout_v1/evaluation` |

The 4,000 request is a generation quota, not a target accepted count. Every generated XML enters the static probe and geometry gate; no template balancing, success balancing, or accepted-count truncation occurs afterward.

Before submission, verify on Amarel that the output root does not exist, the full derived seed interval is absent from prior generation records, the checkpoint is readable, and the committed launch checkout is clean. Record the checkpoint and training-reference hashes below when the two source machines are reachable.

## Generation, probe, and freeze commands

Run generation from the launch commit on Amarel's Piscataway `main` partition:

```bash
cd "$NAMO_REPO"
source env.amarel.sh
OUT=/scratch/dm1487/full_namo_heldout_v1/generation \
NUM_ENVS=10 \
SEED_BASE=827000000 \
EXACT_HOP=2 \
sbatch --array=0-399 scripts/slurm/multihop_aug9_generate.slurm
```

After every generation task completes successfully, construct the complete canonical manifest without filtering:

```bash
ROOT=/scratch/dm1487/full_namo_heldout_v1
mkdir -p "$ROOT/population"
find "$ROOT/generation" -type f -name '*.xml' -print0 \
  | sort -z \
  | while IFS= read -r -d '' xml; do realpath "$xml"; done \
  > "$ROOT/population/generated_scenes.txt"
```

Run the zero-push structural probe over the entire manifest:

```bash
"$NAMO_PYTHON" scripts/pipeline/probe_static_topology.py \
  --manifest "$ROOT/population/generated_scenes.txt" \
  --out "$ROOT/population/static_probe.jsonl" \
  --config config/namo_config_complete_skill15_car_1x.yaml \
  --expect-hop 2 \
  --workers 32
```

Build the immutable population from the entire manifest, exact probe, and complete HY5U training-room reference. If the H5 is not mounted on Amarel, first export its complete `xml` column on the CS estate into a hashed, immutable manifest and copy that manifest together with all referenced XML geometry needed by the gate; do not replace it with a subset or directory nickname.

```bash
"$NAMO_PYTHON" scripts/pipeline/build_full_namo_population.py \
  --manifest "$ROOT/population/generated_scenes.txt" \
  --probe-jsonl "$ROOT/population/static_probe.jsonl" \
  --train-xmls /common/users/dm1487/scratch_namo/aquaman/round0/hybrid_train_v1.h5 \
  --name full-namo-two-boundary-heldout-v1 \
  --expect-hop 2 \
  --out-dir "$ROOT/population/frozen"
```

Review `population_audit.json` and stop on any unexplained structural drop, unparseable training or candidate XML, duplicate identity, or training-geometry leak accounting error. Geometry leaks are expected to be explicitly dropped and counted; they are not grounds to regenerate or tune the population.

## Final Full NAMO campaign

Evaluate exactly two methods: deterministic HY5U seed 2 once and uniform Random with ordering seeds `7000, 8000, 9000, 10000, 11000`. Both use identical scenes, evaluation seed 42, `hmax=2` per local keyhole, `1x_car_d5_` primitives, 900 simulator calls per keyhole, raw `q`, discount off, no-op deduplication on, jam-depth pruning on, 100 target-region points, and the 20-point opening threshold.

For each scene, total simulation count is summed over every region-opening call and replanning step until the XML goal is reachable or the planner terminates. Wall-clock time covers the whole `FullNAMOPlanner.search` process and excludes environment construction and one-time checkpoint loading.

Use `full_namo_sim_exp`: one array task runs HY5U and all five Random arms sequentially on the same frozen shard and physical node, with arm order rotated by shard. Plot cumulative success against logarithmic total simulations and logarithmic whole-process wall time, with HY5U in positive green, Random as the five-seed mean in gray, no uncertainty band, and compact terminal success fractions. The caption reports the five Random terminal rates and sample standard deviation.

The model tail is `X/N`; the Random curve pools observed outcomes as `X/(5N)` only for its descriptive terminal fraction. Inferential comparison remains scene-paired: model outcome on scene `i` versus the mean of Random's five outcomes on scene `i`, with a floorplan-cluster bootstrap and per-seed McNemar sensitivity tests.

## Launch record

| item | value |
|---|---|
| committed launch SHA | pending |
| metadata-stamp SHA | pending |
| checkpoint SHA-256 | pending verified Amarel access |
| training reference SHA-256 | pending verified CS-estate access |
| seed/output collision audit | pending verified Amarel access |
| generation job ID | pending |
| generated manifest SHA-256 | pending |
| static probe SHA-256 | pending |
| frozen population SHA-256 | pending |
| accepted/input scenes | pending |
| exact training-geometry leaks dropped | pending |
| experiment lock SHA-256 | pending |

## Launch gate

- [ ] Repository and sibling scorer checkout are clean on the launch host.
- [ ] This card is committed and its launch SHA is stamped by a follow-up commit.
- [ ] Amarel host key is independently verified before updating SSH trust.
- [ ] Checkpoint path and SHA-256 are verified on Amarel.
- [ ] Complete hybrid training reference and SHA-256 are verified on the CS estate.
- [ ] Seed interval and all output roots are confirmed unused on Amarel.
- [ ] Focused builder tests and the complete `full_namo_sim_exp` regression suite pass at the launch SHA.
- [ ] All generation array tasks complete successfully before probing.
- [ ] Population audit is reviewed before `full_namo_sim_exp.pipeline validate` freezes the final experiment.

