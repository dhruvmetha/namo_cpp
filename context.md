# NAMO Region-Opening

## What We're Doing

A robot and its goal are blocked by a **single movable object**. We learn a ranker (action-ordering heuristic) that decides which push to attempt first on that object. Success is fewer simulator calls than random ordering, measured separately by difficulty tier (easy/medium/hard) and search depth (1-push / 2-push variants).

## Current State

**What works:** MuJoCo physics simulator, controller-grounded push primitives for point and car robots, Python pipeline collecting search traces.

**What's open:** The ranker model itself—architecture, handling censored labels from failed searches, the contribution of learning vs. better search strategy.

## Judge Papers On

- **Problem scope:** Single blocking object (our scope) vs. multi-object NAMO / continuous navigation / broader interactive systems?
- **What's learned:** Push-order heuristic (what we need), end-to-end control policy, trajectory optimizer, or value function?
- **Verifier:** Perfect cost-free simulator or learned dynamics / affordances / value estimates?
- **Metric:** Simulator calls to success, task completion time, path length, or control cost?
- **Role:** Algorithmic baseline for comparison, must-cite for problem framing, or context for related work?
