---
type: experiment
status: compilation
created: 2026-07-03
metric:
commit:
tags:
  - experiment
---
# Full Search

Do the full best first search evaluation with 10 random seeds of random (random selection of what to expand next (what action to take)). Could be breadth, could be depth, we never know. 
Compare the results against No-horizon v3. 

Our objective is to record success v/s sim and success v/s time -- and if we are recording success v/s time as well. We need to make sure that the RO instances being solved by both methods are on the same base-system so that time comparison is fair. Yes? 

Budget at 900 sims total per RO instance per seed.

I want plots (with variance bands) for the above two things we measured and tables at different time (s) and sims.  
#### Random
For each random seed, we run the "random" best-first search over all the test RO problem. 
#### No-Horizon v3
Same thing, but we 3 seeds of trained models. Use that to do best first search by using the predicted q-values as the ranker. (We already have infra). You can also add some more aggregation data as how many times do we just go breadth expansions instead of depth? Are we not deep-diving into the expanded new state? Also measure other metrics to tell us if the model is doing a poor job in ranking performance.

Interesting part is what will we aggregate here, and we can do this live in chat.

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
