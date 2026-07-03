---
type: board
updated: 2026-07-03
tags:
  - experiment
  - board
---
# Experiments Board

Orchestrator-maintained status of every experiment thread. Claude (orchestrator) forks each **active**
experiment to its own subagent (Opus 4.8, xhigh reasoning), tracks status here, and relays results.
Each row links to its card; full Plan/Run/Result lives in the card, one-line ledger in
[docs/experiments/RESULTS.md](docs/experiments/RESULTS.md).

## ✅ Done
| exp | one-line | headline result | date |
|---|---|---|---|
| [[_reactive_search]] | reactive random floor vs NoHz-v3, 1push+2push, easy/med/hard (car) | NoHz-v3 ≫ random every cell — 2push 42.1 vs 4.7 · 1push 82.3 vs 37.0 | 2026-07-03 |

## 🔄 Running (forked to subagents)
| exp | one-line | agent | notes |
|---|---|---|---|
| [[_full_search]] | best-first search: random(10 seeds) vs NoHz-v3(3), success-vs-sims + success-vs-time, budget 900, + aggregations a/b/c/d | opus/xhigh | **campaign LAUNCHED on Amarel (icelake-pinned), ~2.5h ETA**, watchers armed; agg+plots on landing |
| [[_step_penalty_]] | retrain NoHz with −1/0/1 target (never/future/immediate) vs current 0/0.9/1; report search+reactive (1push+2push) | opus/xhigh | **SCOPED + smoke PASSED; GATED** on your CAR-vs-point data call + Amarel confirm before the ~10h×3 retrain. Change = 2 additive edits (target_scheme=signed; HLGauss range→[−1,1]) |

## 📋 Pending
_(none)_

## 🧊 Frozen / not-doing
_(none)_
