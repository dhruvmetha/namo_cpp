---
status: running
thread: rl_loop
robot: car
updated: 2026-07-11
---

# EXP-2026-07-11 — Curriculum ladder: exhaustive 1-push → self-collected clean 2-push

One model, built up like a curriculum. Learn a rock-solid **1-push** pusher on **complete** data; use it to collect **clean 2-push** data (try every first move, let the model finish); keep feeding **harder** scenes so it's forced to get good at the hard ones. No ranking objective — the **data** carries the ranking.

## Why this experiment (the diagnosis that motivated it)

Q2 (the γ^k value ranker, [EXP-2026-07-10](EXP-2026-07-10-exit-search-loop.md)) failed on **data**, not objective — verified three ways:
- **NoHz and Q2 are the same recipe.** Same `hl_gauss` value head, same target (opener=1 / setup=0.9 / dead=0), same no-horizon, same direct-score deploy. The ONLY material difference is the training data. So the setup-#1 gap (NoHz 50.9% vs Q2 21.6%) is a data-quality story, not architecture/objective.
- **The noise is false-negative setups, and it's structural.** At depth-1 a setup is indistinguishable from a dead-end (neither opens in one push), so rung-1 stamps every setup `0`. Rung-2 only *rescues* (→0.9) the setups its **Q1-steered** beam expands — but Q1 is an *opens-now* detector with no signal for setup selection, so it steers the setup search blind → many true setups never get a finisher → stay `0`.
- **Calibration ruled the objective out.** `pos_weight=6` lifted setups out of the dead band **in-sample** (pred 0.16→0.57) but held-out setup AUC stayed **~0.46 (near-chance)**. The head *can* represent setup-value; the labels won't let it generalize. ⇒ **fix the data, not the loss.**

## The bar (definition of winning)

Every bucket **easy / med / hard** × horizon **1push / 2push**: **≥95% success in far fewer sims than random.** Reported **with-search AND reactive (no-search)**, difficulty per-episode, held out by room.

## Design decisions (locked by USER)

- **NO ranking objective, ever** — the data carries ranking. (Adding one would confound "did the data get better?")
- **Exhaustive 1-push collection** (`region_sample_k: 0` → all reachable first-pushes, not a 25-sample).
- **Depth-2 = exhaustive first-ply + guided finish** (near-oracle finisher). Cost O(N·k), not O(N²) — the finish-sim budget is *reallocated* from "20 blind samples on 15 blind setups" to "a few guided finishes on all setups."
- **One model does open-now (depth-1) AND finish (depth-2 second push)** — same policy. Depth-2's finisher-misses feed back as depth-1 training data → both improve together.
- **Harder scenes via rejection sampling** (keep the hard tail) — the interim hardness knob.
- **pure2push exhaustive testset = finish diagnostic, EVAL ONLY** (never train; canonical held-out).

## Stages (plain)

- **Stage 1 — rock-solid 1-push model.** Collect exhaustive 1-push → train one value model (opens-now) → test all buckets vs the bar + grade its finishing on the exhaustive 2-push testset.
- **Stage 1b — make it harder (rejection sampling).** Drop easy scenes / keep the hard tail (and the ones it still fails) → retrain until the **hard** bucket clears the bar.
- **Stage 2 — collect clean 2-push.** Try **every** first push (1 sim each) → from each new state let the 1-push model pick the finisher (guided) → a first push is a **setup** iff a finish opens. Full first-ply coverage ⇒ no missed setups. Finisher-misses → new hard depth-1 examples.
- **Stage 3 — train the ≤2-push model** on the clean data (same value head, now also ranks setups) → test vs bar + random.
- **Stage 4 — climb.** Rejection-sample harder → recollect with the better model → retrain. Better model → cleaner+harder data → better model.

## Hypotheses [USER to confirm/edit]

- **H1** — With complete 1-push labels + rejection-sampled hard data, one value model clears **≥95% all buckets on 1push well before random**, with **no ranking loss**.
- **H2** — Exhaustive-first + guided-finish cuts the false-negative-setup rate **sharply** vs the Q1-steered beam, at **comparable sim cost**.
- **H3** — The same clean data lets the ≤2-push model **beat NoHz's setup-#1 (50.9%)** — because the wall was data, and we removed it.

## Plan / Run / Result

**Stage 1 — baseline first (near-zero cost).** Before any fresh collection, measure the current-best depth-1 model (NoHz-v3, the incumbent trained on exhaustive-ish 1-push data) on the bar: 1push @k-by-bucket (search + reactive) vs random, + finish-by-bucket on exhaustive pure2push. This gap defines how much Stage-1b hardening we need.

_(run log appended below as stages complete)_
