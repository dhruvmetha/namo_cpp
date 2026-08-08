---
status: hub
tags: [problem, approach, canonical, live]
updated: 2026-08-01
---

# The problem & our approach (NAMO region-opening)

> The durable, plain-English north-star: **what we're solving** and **the shape of how we solve it**. Kept abstract on purpose — the ranker's form, how it's trained, and all results are method detail that lives in [experiments/](experiments/) (the cards + [RESULTS.md](experiments/RESULTS.md)). Read this first so that detail lands in the right frame.

## 1. Problem setup — how one region-opening episode comes from a room

A room contains the car robot, its goal, fixed walls, and movable objects. With every obstacle present, the backend builds a robot-size-aware free-space map: each connected patch of free space is a **region**, including the region containing the robot and the region containing the goal.

The backend then temporarily removes each movable object's robot-inflated footprint to determine which regions that object separates. The current region-opening scope keeps cases where the robot region and goal region are **immediate neighbours across one movable object**; that object is the boundary blocker. This produces one **episode = (scene, target object, goal region)**, with the robot region fixed by the robot's current pose.

If the goal region is not an immediate neighbour of the robot region—meaning more than that one object lies between them—the scene is out of scope. It is not a 2-push problem.

## 2. The problem

Given that episode, find the push or short push chain on the **given blocking object** that merges the robot region with the goal region. The task is one local opening, not navigation through a sequence of regions.

**1-push** means one push merges the two regions. **2-push** means two pushes on the same object merge the same pair: a **setup** push that does not open the region yet, followed by a **finish** push that does.

## 3. Success condition — what counts as open

Before search, the evaluator samples 100 fixed points in the goal region. After each simulated push, it rebuilds robot-size-aware reachability from the robot's current region; the episode succeeds when at least 20 of those points (20%) are reachable.

This tests whether the push created a meaningful robot-sized opening. The robot does not have to physically navigate to the goal during evaluation, and moving the object, exposing only a narrow sliver, or reaching only the single XML goal point is not enough.

## 4. The approach

**The problem is solvable by search.** At each state there are ~50 reachable candidate pushes, and a simulator tells us exactly what any push does — it opens the way, or it doesn't, or it sets up a follow-up push. Trying pushes in a search (expand a push, simulate, expand again) reliably finds a solution.
**But search is expensive.** Every candidate push the search tries costs one simulator call (~1 second of physics), and a blind search burns a great many of them per problem before it stumbles onto the answer.
**So the research goal is to make search cheap by learning a good ranking function** — a heuristic (a "ranker") that orders which pushes (and which resulting states) the search should try *first*, so it reaches a solution after a handful of tries instead of hundreds. Two things make this the right frame — the canonical statement is [experiments/horizon_q_search_redesign_journal.md §0](experiments/horizon_q_search_redesign_journal.md): (a) the model is **a ranker, not a simulator** — it only orders the pushes; the sim still executes and checks them; (b) because that simulator is a **perfect, free verifier**, the model needs the right **order**, not calibrated probabilities. In one line: **learn the ordering that turns an expensive search into a cheap one** (minimize simulator calls to a solution).

## 5. The objective (the research success bar)

**Beat the random ranker.** The baseline is a search that tries candidate pushes in random order. Our learned ranker has to solve the test-set problems with **far fewer simulator calls** (less search) than that random ordering — and it has to do so **across the board**, on every difficulty tier, not just on average.
The headline is a curve — **solve-rate vs. cost** (simulator calls, or wall-clock time). Our ranker's curve should **dominate** random's: reach any given solve-rate at a fraction of the cost. The measured version of exactly this comparison (learned value vs. uniform-random ordering, by difficulty, in both sims and wall-time) lives in [experiments/RESULTS.md](experiments/RESULTS.md).

---

**Method + experiment detail** — the ranker's form, how it is trained and labeled, and every result — lives in [experiments/](experiments/): the results sheet [RESULTS.md](experiments/RESULTS.md) and the cards ([`_full_search.md`](../_full_search.md), [`_reactive_search.md`](../_reactive_search.md), [`_ranker_bottleneck.md`](../_ranker_bottleneck.md), the live hypothesis [policy_value_search_hypothesis.md](experiments/policy_value_search_hypothesis.md)).
