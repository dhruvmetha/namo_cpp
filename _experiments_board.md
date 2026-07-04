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
| [[_full_search]] | best-first **search**: random(10) vs NoHz-v3(3), sims + wall-time, budget 900, + a/b/c/d | solve@900 95.3 vs 91.0 but **~3× fewer sims**; #1-pick-wins 50.9% vs 14.9%; both deep-dive; 13% "solved-but-slow" tail (H1/H2 scale) | 2026-07-04 |

## 🔄 Running (forked to subagents)
| exp | one-line | agent | notes |
|---|---|---|---|
| [[_step_penalty_]] | retrain NoHz with −1/0/1 target (never/future/immediate) vs current 0/0.9/1; report search+reactive (1push+2push) | experiment-runner (opus/xhigh/worktree) | **TRAINING DONE** — 3 seeds best-val s1 0.686/s2 0.694/s3 0.688 (iLab, epoch~12). **EVAL forked** — search+reactive, 1push+2push, 3-way vs NoHz-v3 + random; resolving fast_scorer (sync to shared FS vs Amarel) since ckpts are on shared FS |

## 📋 Pending
_(none)_

## 🧊 Frozen / not-doing
_(none)_
