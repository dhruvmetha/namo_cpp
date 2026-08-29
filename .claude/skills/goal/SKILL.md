---
name: goal
description: State what the current NAMO work is actually FOR, and the test for whether a piece of work serves it. Use when asked "/goal", "what is the goal", "what are we doing this for", "why are we collecting this", or when about to spend real time on something whose payoff is not obvious. Read it before launching a collection, adding a field to a dataset, or fixing pipeline plumbing, so effort lands on the question being asked instead of on machinery around it.
---

# What this work is for

## The standing frame

The model is a **ranker**. It orders which push the search tries first, so the search solves in a handful of simulator calls instead of hundreds. The simulator is a perfect free verifier, so what matters is the ORDER, never a calibrated probability. Success is beating the random ranker on far fewer simulator calls, reported by difficulty tier AND by 1-push versus 2-push, never aggregate-only.

Full framing: [docs/problem_and_approach.md](../../../docs/problem_and_approach.md). That file is the frame; this one is the current push and the test for staying on it.

## Current push, opened 2026-08-29: multi-movable interaction

Build and label real-table scenes with **more than one movable object in the doorway**, still a single region-opening, where **pushing the target drives it into its neighbour**.

⛔ **THE INTERACTION IS THE POINT, not a side effect.** 593 single-movable scenes already exist in `handoff/real_scene_build_sheets_v2/`. In every one of them the pushed object cannot touch another movable, so the contact rate is structurally zero. If a new pool does not measure what happens when the target hits a neighbour, it answers a question the existing 593 already answered and is not worth its compute.

Three things only a multi-movable pool can express:
- one push moves two objects, which is why a scene where neither object alone clears the route can still open in a single push
- the ranker must choose WHICH object to push, not only where on the one object
- a two-push chain can switch objects, setup on one and finish on the other

## The test, apply it to any piece of work

**Does this get us closer to knowing how the ranker behaves when objects interact?** If the honest answer is "it makes the pipeline work," that is plumbing. Plumbing on the critical path is fine, say so out loud and keep it short. Plumbing dressed as progress is the failure mode.

Two things that have already gone wrong this way and are worth pattern-matching against:
- Treating `movable_collisions` as a sanity check rather than the measurement. It is the field that justifies the pool existing.
- Measuring interaction at one end of a chain. The setup push recorded its contacts and the finish push threw its result away, so the chains where interaction matters most were the ones half-measured.

## What a result has to say

Report by difficulty AND horizon, per the standing rule. On top of that, a multi-movable result has to answer:
- how often a solving push actually made the blocks touch, split by scene flavour
- what the second object changes versus the single-movable baseline, since that is the whole claim

⛔ A boolean "did they touch" cannot say whether the contact HELPED. Displacement per push is what answers that, and it is not currently recorded. Do not quote a contact rate as if it measured usefulness.

## Known holes in the current data, do not report around them

Both live pools carry these. Say them out loud with any number drawn from them.

- Roughly two thirds of the domino-flavour scenes get silently skipped, because at the start state the goal region is too tight for the sampler to place the car anywhere in it. Any domino statistic is therefore a biased sample of the LOOSEST third. See [[reference_region_goals_vs_region_labels]].
- The finish push's contacts are missing from anything labelled before commit `ebc7f63`.
- `finish[0]`, the object the finish landed on, undercounts cross-object chains: the sweep stops at the first finish that works and walks the objects in list order, so a same-object finish always wins the race where both exist.
