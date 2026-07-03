---
type: experiment
status: incomplete idea, do not do
created: 2026-07-03
metric:
commit:
tags:
  - experiment
---
# Step penalty
Dont do this yet, its incomplete

We want to train new model no-horizon model (with "v3" data). However, we change the target "q(s, a)" to 1 for opening (for when it immediately opens), q(s, a) = 0 if it does not open now but will open in the future and q(s, a) = -1 never opens now or in the future. 

Retrain model with this new scheme and compare against the random (1push and 2push)

## Hypothesis
_(you)_ The falsifiable claim — what we're testing and the expected direction.

## Plan
_(Claude)_ Code / data / config + the exact run command.

## Run
_(Claude, auto)_ job id · commit · config · date.

## Result + Verdict
_(Claude, auto from run output)_ Numbers — accept/reject **on numbers only**.

## Next
What this implies; the follow-up experiment.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
