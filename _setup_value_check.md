---
type: experiment
status: live
created: 2026-07-05
updated: 2026-07-05
metric: "Free check: does ranking first pushes by the finish they enable un-bury the true setup and solve the hard/missed rooms? Compares 4 rules — current q, best-finish V(s1)-max, aggregate V(s1)-agg, and GT-oracle — on where the true setup lands + whether the room then solves. Built to teach the SHAPE of setup-value, not just yes/no. Not yet run."
tags:
  - experiment
  - diagnostic
---
# Setup-value check — can we un-bury the setup by scoring it by the finish it enables?

**Sibling to [[_ranker_bottleneck]].** That card showed the 2push search fails because the true SETUP push (opens nothing yet) is buried in the model's ranking. This card tests the cheapest fix idea — and, more importantly, is built to TEACH us what "setup value" actually looks like, not just give a yes/no.

**Plain idea.** A setup is defined by what it LEADS to (a finish). So score each first push by "after this push, how good are the finishing moves available?" The model is already good at spotting finishes, so we lean on its strength instead of its weakness. The honest doubt (from [[_1push_bottleneck]]): the model's finish-spotting also goes fuzzy on the rare hard rooms — exactly the ones we care about — so the cheap version may only help where we don't need it. This experiment is designed to expose that, not paper over it.

## Hypothesis [USER]

If we rank the first pushes by the finish they enable — instead of by whether they open the goal now — the true setup climbs toward the top and the hard/missed rooms solve. I want to SEE this happen (or not), understand WHY, and compare a "best single finish" score against an "aggregate finish" score so I can conclude what is really going on.

## Plan [CLAUDE]

Re-rank the first pushes for the **371 hard rooms + the 21 always-missed rooms** under four scoring rules, and measure (a) where the TRUE setup lands and (b) whether the room then solves. Everything on the exhaustively-solved pure2push set (n=1018), so we always know the true answer.

**The four ranking rules** (each scores a first push by looking at `s1` = the state right after that push):

- **q (baseline)** — the model's current direct score. We already know the setup sinks under this.
- **best-finish `V(s1)-max`** — the single highest finish-score among the second pushes available at `s1`. "Is there at least one great finish here?"
- **aggregate `V(s1)-agg`** — a summary of the finish-scores at `s1`, computed three ways: mean, top-3 mean, and count-above-the-opener-threshold. "How good are the finishes overall, not just the best one?"
- **GT-oracle `V(s1)-GT`** — does a TRUE finish actually exist at `s1` (ground truth from pure2push)? Perfect finish-detection — the upper bound the learned rules are chasing.

**What we measure, per rule, split by difficulty (easy/med/hard):**

1. **Where the true setup lands** — p@1 (is it the #1 pick?), median rank, fraction in the top 5. Under GT it is #1 by construction.
2. **The separation picture** — the score distribution for true setups vs junk first-pushes (histogram + AUC). Do the two piles separate, and by how much? This makes the mechanism visible — it directly answers "do we actually see high-for-setup / low-for-junk?"
3. **Downstream solve** — force the first push to the true setup, run the dive (second push) as normal, and check whether the room opens within budget. This isolates the key unknown: is setup-ranking the WHOLE bottleneck, or does the dive ALSO fail on these hard rooms?

**Pre-registered interpretation — what each outcome MEANS (write this before looking):**

- **GT-oracle solves (most of) the rooms** → setup-ranking IS the bottleneck; a good setup-value signal would fix it → green-light the retrain. **GT-oracle leaves rooms unsolved** → the dive also fails on the hardest rooms; setup-value alone won't reach 100%; the plan must widen beyond setups.
- **best-finish ≈ GT on easy/med but falls off on hard** → the cheap lookahead works where finishes are easy to spot and fails on the rare hard tail (finish buried there too) → the cheap mechanism is a partial fix; the learned retrain is needed for the tail. **best-finish ≈ GT everywhere** → the cheap lookahead is enough; maybe skip the retrain.
- **best-finish vs aggregate** → the SHAPE of a setup. If max wins and mean is diluted, a setup is "one clear finish" (so a false-positive can spoil max, and top-3/count should help). If aggregate wins, a setup is "several decent finishes." This directly informs how to build the setup-value training target.

**Data / cost.** Offline re-ranking reuses the existing classifier re-score data (child finish-scores at `s1`, from [[_ranker_bottleneck]]'s `cls/`) where it covers these rooms; a small CPU re-score fills the rest (simulate each first push once, score its children). The downstream-solve check is a small online best-first run on the hard+miss set. No training, no GPU. Watch the offline↔online label caveat from [[_1push_bottleneck]] (a stale-label room can look like a failure that isn't the ranker's fault) — flag any room where the offline "true setup" doesn't reproduce online.

**Deliverables:** the four-rule setup-rank table (by difficulty), the setup-vs-junk separation plots (per rule), the downstream-solve table (GT-oracle and best-finish), and a one-line verdict on each pre-registered question. Owned files: this card, `assets/setupval_*.png`, `scripts/sandbox/setup_value_check.py`.

## Run
_(Claude, auto — job / commit / data / date.)_

## Result
_(Claude, auto — the tables, plots, and the verdict on each pre-registered question.)_

## Discussion
_(you ↔ Claude — ask here; newest at the bottom.)_
