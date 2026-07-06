---
status: hub
tags: [workflow, orchestration]
updated: 2026-07-06
---
# Orchestration — running the parallel-experiment fleet

> How the agent fleet is run: one orchestrator (the main loop) + forked experiment agents. Claude reads this when running experiments in parallel; pointed to from CLAUDE.md and [WORKFLOW.md](WORKFLOW.md).

## One experiment = one forked subagent
The orchestrator (main loop) pulls, forks each active experiment (Opus, xhigh), tracks status, gates expensive steps (retrains) on the user, then merges each agent's card/plot edits back and owns the shared files (RESULTS.md, commits) so parallel agents don't race.

## Isolation — where an agent's edits land
- A **freshly-forked** `experiment-runner` (its def sets `isolation: worktree`) gets an isolated **locked worktree** at `.claude/worktrees/agent-<id>/`, branched from HEAD-at-fork-time — its edits do NOT appear in the main checkout.
- A **resumed** agent (continued via SendMessage from transcript) runs in the **shared checkout**.

## Safety rule — file-partition
Give each parallel agent **disjoint files** (its own `_card.md` + its own eval dirs), and agents **NEVER commit** (the orchestrator owns all commits). **Never fork two agents that write the same files.** Disjoint files ⇒ merges never conflict.

## Merging back
- **Shared-checkout (resumed) agents:** edits are already on the branch → `git add <their files> && commit`.
- **Worktree agents:** edits are isolated → because they never commit, copy their **OWNED files only** from `.claude/worktrees/agent-<id>/` into the main checkout, then commit. The worktree branched from an older HEAD and still holds since-deleted files — **never copy the whole tree.** Prune after: `git worktree remove`.

## Status & live job state
Each card's `status` frontmatter (`idea→live→done`) drives [DASHBOARD.md](DASHBOARD.md) / `experiments.base` — **the single source of truth; there is no separate status board.** Live compute-job state (SLURM ids, queue status) lives in the relevant card's **Run** section, not a side board.

## Tiering (cost)
- `scout` (sonnet/medium) — recon/mechanical fan-outs: node/queue checks, reuse-scans, rsyncs, aggregation.
- `experiment-runner` (opus/xhigh) — real experiment reasoning.
- Don't fork an agent for a trivial one-liner — inline Bash.
- Both the orchestrator and an experiment-runner may spawn scouts.
