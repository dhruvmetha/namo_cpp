---
type: experiment
status: live
created: 2026-08-22
commit: 00dca4e
metric: pending
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

## Result

Pending.

## Verdict

Pending.
