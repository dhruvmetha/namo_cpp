---
type: experiment
status: done
created: 2026-08-22
commit: 00dca4e
metric: 29 accepted at 0.30 m; 31 at 0.20 m
tags: [experiment, multihop, data, scale]
---
# Fixed-template keyhole composition scale pilot

## Hypothesis

_(you, from chat)_ Since the environments use fixed templates, find different canonical keyhole scenarios and merge them where K1, K2, … do not interfere; scale medium-medium, medium-hard, hard-medium, and hard-hard two-hop cases using Amarel's resources.

## Plan

_(Codex)_ Run a target-box smoke, then a 40-task Amarel CPU array covering all ten Aug9 templates crossed with the four ordered medium/hard donor-tier pairs. Each task samples up to 100 blocker pairs separated by at least 0.30 m, asks for at most five accepted scenes, requires exact ordered two-hop static topology, and forward-replays the canonical donor openers against pinned component goals so K1 must open only C2 and K2 must remain solvable. Report attempted, static rejection, replay rejection, accepted count, and wall calibration separately by ordered tier pair and template before deciding production size.

## Run

_(Codex)_ Target-box smoke `60751008_9` ran on Amarel `main` at commit `de75c14`, task MM × `set2/benchmark_5`, with `LIMIT=1`, `MAX_ATTEMPTS=100`, and the production script. It completed successfully in 39 s, wrote the accepted XML plus `manifest.jsonl` and `summary.json`, and accepted 1 scene after 6 candidates; the other 5 were rejected statically. Artifacts: `$NAMO_SCRATCH/eval/keyhole_modules_scale_smoke_20260822/`.

_(Codex)_ Calibration: a direct five-scene extrapolation is 195 s per task; allow roughly 3–8 minutes because templates and ordered tier pairs can have different static and replay yields. The pilot requests 15 minutes per task, 4.6× the direct extrapolation, and only 40 single-CPU tasks, within the ≤200-CPU background policy. Production remains gated on this pilot's measured yield and tail.

_(Codex)_ The 0.30 m pilot was Amarel array `60751013`, 40/40 tasks completed with exit code 0 in at most 3:33, and all 40 summary artifacts landed under `$NAMO_SCRATCH/eval/keyhole_modules_scale_pilot_20260822/`. The complete aggregate is `aggregate.json`; results were also pulled to the matching CS `$NAMO_SCRATCH` path. Because the forward replay is the authoritative interference test, run one otherwise identical 0.20 m pilot before production to measure whether the conservative center-distance gate is discarding valid scenes.

_(Codex)_ The 0.20 m follow-up was Amarel array `60751209`, 40/40 tasks completed with exit code 0 in at most 6:11, and all 40 summaries landed under `$NAMO_SCRATCH/eval/keyhole_modules_scale_pilot20_20260822/aggregate.json`. It sampled at most 100 pairs per tier-template cell under the relaxed distance gate.

## Result

### Pilot at 0.30 m minimum blocker-center separation

| ordered donor tiers | attempted pairs | exact static two-hop | replay accepted | accepted / attempted | accepted / static |
|---|---:|---:|---:|---:|---:|
| medium-medium | 268 | 18 | 7 | 2.61% | 38.89% |
| medium-hard | 200 | 10 | 6 | 3.00% | 60.00% |
| hard-medium | 182 | 9 | 8 | 4.40% | 88.89% |
| hard-hard | 167 | 11 | 8 | 4.79% | 72.73% |
| **all** | **817** | **48** | **29** | **3.55%** | **60.42%** |

_(Codex)_ All 29 accepted scenes came from `set2/benchmark_3` (9) and `set2/benchmark_5` (20); the other eight templates yielded zero. The dominant loss is static geometry: 769/817 candidates never formed the intended exact ordered two-hop chain. Once that topology existed, donor-action continuation survived in 29/48 cases. There were zero duplicate ordered donor sequences in the accepted manifests.

### Pilot at 0.20 m minimum blocker-center separation

| ordered donor tiers | attempted pairs | exact static two-hop | replay accepted | accepted / attempted | accepted / static |
|---|---:|---:|---:|---:|---:|
| medium-medium | 351 | 38 | 9 | 2.56% | 23.68% |
| medium-hard | 299 | 19 | 6 | 2.01% | 31.58% |
| hard-medium | 296 | 22 | 8 | 2.70% | 36.36% |
| hard-hard | 212 | 11 | 8 | 3.77% | 72.73% |
| **all** | **1,158** | **90** | **31** | **2.68%** | **34.44%** |

_(Codex)_ Relaxing 0.30 → 0.20 m bought only two additional accepted scenes while adding 341 sampled pairs and 42 statically plausible chains. Replay failures rose from 19 to 59, and the slowest task grew from 3:33 to 6:11. The failure is therefore not a shortage of nearby pairs: closer modules more often interfere dynamically after they pass the static graph check.

## Verdict

**REJECT scaling the random-pair stitcher as the production generator, and reject the 0.20 m relaxation.** The mechanism is real, but candidate generation is backwards: it samples blockers and inherited endpoint poses, then hopes they accidentally form the desired three-room sequence. At 0.30 m, 94.1% fail static topology; at 0.20 m, the extra static candidates mostly fail replay. Pivot to the user's simpler construction: choose a room-like three-component layout and robot/goal endpoint rooms first, then fill doorway slot 1 and doorway slot 2 with easy canonical one-push modules before reintroducing medium/hard donors.
