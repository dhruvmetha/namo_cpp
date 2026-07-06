# RL-only self-imitation loop — runbook

Implements Phase 1 of [EXP-2026-07-06-rl-only-self-imitation](../../docs/experiments/log/EXP-2026-07-06-rl-only-self-imitation.md). Code lives in `python/namo/rl_loop/`; this dir is the CLI + SLURM layer.

Forward-only rollouts (no branching, no mid-rollout resets) are the data engine. Solved trajectories train a filtered-BC pi head; Monte-Carlo returns (incl. zeros from failures) train a V head. Both reuse the sage EdgeCrossAttn scorer, deployed greedily. CAR robot, `namo_config_complete_skill15_car_1x.yaml`.

## One-time: freeze the split

```
python scripts/rl_loop/build_split.py --pool-key <episode_key.json> \
  --out <run>/split.json
```
80/10/10 by ROOM (xml), seed 42, 0% room leakage asserted. Do this once before gen-0.

## Per generation

Gen 0 is arm A (uniform pi0, no ckpt); gen N (arm B) conditions on the previous generation's pi ckpt.

```
# gen 0 (uniform), in-process collection with 8 workers:
python scripts/rl_loop/run_generation.py --arm A --generation 0 \
  --pool-key <episode_key.json> --split-file <run>/split.json \
  --out-root <run> --n-workers 8

# gen N (policy-conditioned):
python scripts/rl_loop/run_generation.py --arm B --generation N \
  --pool-key <episode_key.json> --split-file <run>/split.json \
  --out-root <run> --ckpt <run>/genN-1/ckpts/pi/checkpoints/<best>.ckpt --n-workers 8
```

One command does: collect (train rooms) -> harvest+filter into the persistent `<run>/buffer.pkl` -> render training H5 -> train pi + V -> greedy open@1/2/5/10 AND setup-hit@1/2/4/8 on dev rooms (both stratified by difficulty x horizon) -> print/save a report row and check the pre-registered kill signals.

## SLURM collection fan-out (scale / Amarel)

Real collection runs where the TRAIN xmls live (Amarel `/scratch` after a git sync). Fan out with the array job, then harvest:

```
NSHARDS=64 CONFIG=<run>/genN/config.json OUTDIR=<run>/genN/collect BUFFER=<run>/buffer.pkl \
  ENV_FILE=env.amarel.sh sbatch --array=0-63 --partition=main-redhat scripts/rl_loop/collect.slurm

python scripts/rl_loop/run_generation.py --arm A --generation N \
  --pool-key ... --split-file ... --out-root <run> --pre-collected-dir <run>/genN/collect
```
(`config.json` is written by `run_generation.py` on its first call for that generation, or hand-authored via `LoopConfig`.)

## Kill signals (checked + printed every generation)

1. gen-1 hard-episode positive coverage < 50% -> collection redesign.
2. gen-1 held-out hard-2push greedy open@10 < 35% -> forecast falsified.
3. hard unique-solve coverage flat across two consecutive generations -> RL-only coverage wall.

## Notes

- The model-scoring/render path (LiveScorer + sage visualizer `fast_scorer=True`) runs on the CS estate now — the "Amarel-only" note in CLAUDE.md is stale (verified on arrakis 2026-07-06).
- Setup-ranking reuses the Phase-0 oracle-finish logic (`phase0_ksweep.py`, committed 6368c45); anchor to beat: setup-hit@1 = 54.0, @8 = 82.5 (hard 36.8 / 70.4).
