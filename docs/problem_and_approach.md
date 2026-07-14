---
status: hub
tags: [problem, approach, canonical, live]
updated: 2026-07-06
---

# The problem & our approach (NAMO region-opening)

> The durable, plain-English north-star: **what we're solving** and **the shape of how we solve it**. Kept abstract on purpose — the ranker's form, how it's trained, and all results are method detail that lives in [experiments/](experiments/) (the cards + [RESULTS.md](experiments/RESULTS.md)). Read this first so that detail lands in the right frame.

## 1. The problem

A car robot needs to reach a goal region in a room, but a movable object is **blocking the way**. It fixes this by **pushing the object out of the way** — sometimes one push is enough, sometimes it takes a short chain of two.
The unit of work is one **episode = (scene, target object, goal region)**: the blocking object is given, and the job is to find pushes that open the way to the goal.

**One opening, not a traversal [USER — do not muddle this].** This is a SINGLE region-opening: the **robot region** and the **goal region** are separated by the one given **blocking object**, and the job is to **merge those two regions** with a push on that object. **1-push** = one push of the object merges robot+goal. **2-push** = it takes two pushes to merge that *same* pair — a *setup* push (doesn't merge yet) then a *finish* push (merges). It is emphatically NOT "the goal is two regions away, open one region then the next" — we do **not** chain multiple region-openings in one problem. A scene where the goal region is **not adjacent** to the robot region (more than that one object between them) is **out of scope**, not a 2-push.

## 2. The approach

**The problem is solvable by search.** At each state there are ~50 reachable candidate pushes, and a simulator tells us exactly what any push does — it opens the way, or it doesn't, or it sets up a follow-up push. Trying pushes in a search (expand a push, simulate, expand again) reliably finds a solution.
**But search is expensive.** Every candidate push the search tries costs one simulator call (~1 second of physics), and a blind search burns a great many of them per problem before it stumbles onto the answer.
**So the research goal is to make search cheap by learning a good ranking function** — a heuristic (a "ranker") that orders which pushes (and which resulting states) the search should try *first*, so it reaches a solution after a handful of tries instead of hundreds. Two things make this the right frame — the canonical statement is [experiments/horizon_q_search_redesign_journal.md §0](experiments/horizon_q_search_redesign_journal.md): (a) the model is **a ranker, not a simulator** — it only orders the pushes; the sim still executes and checks them; (b) because that simulator is a **perfect, free verifier**, the model needs the right **order**, not calibrated probabilities. In one line: **learn the ordering that turns an expensive search into a cheap one** (minimize simulator calls to a solution).

## 3. The objective (the success bar)

**Beat the random ranker.** The baseline is a search that tries candidate pushes in random order. Our learned ranker has to solve the test-set problems with **far fewer simulator calls** (less search) than that random ordering — and it has to do so **across the board**, on every difficulty tier, not just on average.
The headline is a curve — **solve-rate vs. cost** (simulator calls, or wall-clock time). Our ranker's curve should **dominate** random's: reach any given solve-rate at a fraction of the cost. The measured version of exactly this comparison (learned value vs. uniform-random ordering, by difficulty, in both sims and wall-time) lives in [experiments/RESULTS.md](experiments/RESULTS.md).

---

**Method + experiment detail** — the ranker's form, how it is trained and labeled, and every result — lives in [experiments/](experiments/): the results sheet [RESULTS.md](experiments/RESULTS.md) and the cards ([`_full_search.md`](../_full_search.md), [`_reactive_search.md`](../_reactive_search.md), [`_ranker_bottleneck.md`](../_ranker_bottleneck.md), the live hypothesis [policy_value_search_hypothesis.md](experiments/policy_value_search_hypothesis.md)).
