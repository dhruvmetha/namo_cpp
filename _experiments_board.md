---
type: board
updated: 2026-07-04
tags:
  - experiment
  - board
---
# Experiments Board

Orchestrator-maintained status of every experiment thread + the compute jobs behind them. Full Plan/Run/Result
lives in each card; one-line ledger in [docs/experiments/RESULTS.md](docs/experiments/RESULTS.md). Reporting
default: **every result split by difficulty (easy/med/hard) AND horizon (1push/2push)**.

## ✅ Done
| exp | one-line | headline | date |
|---|---|---|---|
| [[_reactive_search]] | reactive floor vs NoHz-v3, 1push+2push, easy/med/hard (car) | NoHz-v3 ≫ random every cell — 2push 42.1 vs 4.7 · 1push 82.3 vs 37.0 | 2026-07-03 |
| [[_full_search]] | best-first search: random(10) vs NoHz-v3(3), sims + wall-time, budget 900, + a/b/c/d | solve@900 95.3 vs 91.0 (~3× fewer sims). **By difficulty: model's win concentrated in HARD (+11.5pt @900); easy/med = efficiency-only (both ~98%). Time: hard model 26s vs 46s, easy curves CROSS (7.1 vs 6.3s)** | 2026-07-04 |

## 🔄 Running

### [[_step_penalty_]] — retrain NoHz −1/0/1 target vs 0/0.9/1 · 3-way (random / NoHz-v3 / step_penalty) · search+reactive · 1push+2push
Training ✅ (3 seeds, iLab, best-val s1 .686 / s2 .694 / s3 .688). Eval (experiment-runner) **near done**:
- **sims + reactive** (machine-independent, headline): s1/s2 ✅ 100%; **s3 991/1018** (hard tail on ilab3); reactive ✅ 100% all 3 → **aggregation imminent**.
- **2push timing**: Amarel job `57845891` (+backfill `57846177`), sapphirerapids-exclusive, ~27% (822/1018), ~hours.
- **1push timing**: Amarel job `57846712`, sapphirerapids-exclusive, launched (PD, backfilling; fast).
- Next: full 3-way tables/plots × difficulty × horizon (sims now, time as each timing run lands).

### [[_full_search]] — difficulty×horizon retrofit (augmenting the done card)
- **2push**: sims + time by easy/med/hard ✅ **DONE + validated** (time = emeraldrapids-exclusive). → **embedding into card now.**
- **1push**: sims running on rlab7 (`bf1push` jobs); **time = PENDING a separate exclusive run** (rlab7 t_wall invalid — iLab co-tenanted).

## 📋 Pending
- **full_search 1push timing** — needs its own exclusive run; user OK'd sims-only on that card for now (deferred).

## 🧊 Frozen / not-doing
_(none)_

## Infra built this session (compounding)
- Agent defs: **experiment-runner** (opus/xhigh) + **scout** (sonnet/medium tier) · CLAUDE.md conventions: verify-before-bet,
  drive-don't-wait, **time-consistency** (single-arch exclusive), **stratified splits** (always tier×horizon), **tiering**,
  file-partitioning isolation (worktree isolation doesn't engage for bg subagents in 2.1.201). Compute skill: full node
  inventory + iLab-login fallback + timing→Amarel-only.
