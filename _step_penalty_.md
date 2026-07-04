---
type: experiment
status: live
created: 2026-07-03
updated: 2026-07-04
metric: "retrain −1/0/1 vs 0/0.9/1; 3-way eval running (sims/reactive near-done, 1push+2push timing on Amarel)"
commit:
tags:
  - experiment
---
# Step penalty

We want to train new model no-horizon model (with "v3" data). However, we change the target "q(s, a)" to 1 for opening (for when it immediately opens), q(s, a) = 0 if it does not open now but will open in the future and q(s, a) = -1 never opens now or in the future. 

At the moment we have trained with successful finish 1, successful setup (future finish exists) 0.9 and no success ever is 0. Change this to what we have described.

Retrain model with this new scheme and compare report the results (1push and 2push), on both search and reactive modes.  We will later compare this with random and no-horizon v3. after we finish. (After the results of random  and no-horizon v3 search are also computed (which they are being done simultaneously in parallel))

## Hypothesis
This "reward" scheme is better suited for ranking (search) than the older scheme (our other parallel experiment thread)

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
