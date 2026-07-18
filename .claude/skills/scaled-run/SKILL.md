---
name: scaled-run
description: Use BEFORE launching any scaled-up / multi-hour job — a SLURM collection, a training run or array, an eval sweep, or a big data build — and BEFORE reading code/docs, asking clarifying questions, or running any command toward such a launch (pre-flight comes first). Trigger on "launch the full collection/train/eval", "run the full X", "kick off the big run/sweep", "launch the N-way array", "scale this up to all scenes", or going from a tested pilot to production scale. Fires IN ADDITION to namo-data-pipeline when the scaled job is data work (labeling/collection/eval) — invoke both; this one owns smoke-test, time calibration, SLURM sizing, and monitoring.
---

# Scaled-run pre-flight

**The rule: never launch a full-scale run blind.** On 2026-07-17, skipping this cost an evening — the train alone failed 3× at full scale (python-PATH missing, thread-limit blown) on bugs a 2-minute smoke would have caught, and every time estimate was ~3× optimistic (which also caused too-short walltimes that killed jobs).

## Do these IN ORDER, before the full run

**1. SMOKE 1 unit end-to-end on the TARGET box.**
Run the real command on ONE item (1 scene / 1 epoch / 1 shard) on the exact box+partition the full run will use — not "some box." Confirm the log shows real progress (e.g. `[q2 setup]`, an epoch tick, pkls landing), not `command not found`, `pthread_create failed`, or a traceback. Env/PATH/thread/cgroup bugs are box-specific and only surface here.

**2. CALIBRATE the estimate from the smoke — do not guess.**
Measure the per-unit wall time from the smoke and multiply. Report a range, lean pessimistic. Never quote a single optimistic number (mine were ~3× low). This number also sets the SLURM `--time` (below) and tells the user a real ETA.

**3. PILOT small for EXPLORATORY rounds.**
If the run's job is to answer a yes/no question ("does model X beat baseline Y?"), run a small pilot first (a few-thousand scenes / one config / fewer epochs) to get the signal cheap, THEN scale only if it pays. Don't run a 6-hour full pipeline to get a first look. (Full scale is for rounds where you've already decided to invest.)

**4. Size the SLURM job right.**
- `--time` = ~2-3× the calibrated estimate. NEVER omit (CS main-redhat default is **2 minutes** → instant kill) and NEVER partition-MAX on a busy GPU queue (a 3-day request can't backfill → may never start). See memory `feedback_no_slurm_time_limits`.
- Pin thread pools: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` (CS `ulimit -u=2000`; OpenBLAS spawns 64 thr/proc). For collection, cv2 also needs an in-code `cv2.setNumThreads(1)`.
- CS `unlimited` partition REJECTS `--cpus-per-task` (omit it); Amarel allows it.
- Resolve the interpreter explicitly on CS (`env.ilab.sh` does NOT conda-activate → bare `python` is absent in sbatch).
- **Just start from the tested templates** — `scripts/slurm/train.slurm` bakes all of the above in.

**5. Monitor by the RELIABLE signal, not buffered logs.**
SLURM block-buffers task stdout, so logs lag. Poll the real artifact — pkl/ckpt COUNT, `squeue` state — and NEVER declare "stuck" from "0 log output" before the calibrated first-output time. NEVER declare "it's working" before the artifact actually appears.

## Related
Templates: `scripts/slurm/train.slurm`. Compute placement: `compute-resources` skill. Model/effort for subagents: `model-selection` skill. Memory: `feedback_no_slurm_time_limits`, `reference_beast_execution_recipe`.
