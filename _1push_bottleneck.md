---
type: experiment
status: live
created: 2026-07-05
updated: 2026-07-05
metric: "TBD — filled on run"
tags:
  - experiment
  - diagnostic
---
# 1push bottleneck — why the sub-100% floor, and does the model reach the ceiling before random?

**Sibling to [[_ranker_bottleneck]] (which is 2push-only).** 1push has no *setup* move, so the 2push "setup under-ranking" story cannot apply — the diagnostic must find the *different* reason 1push tops out below 100%, and whether the learned ranker reaches that ceiling ahead of the random floor.

## Hypothesis [USER]

The 1push best-first search plateaus at ~99.7% (hard ~99.2%), not 100% — just like 2push fails to reach 100%. Why do these 1push cases also fail? And does the model reach 100% (its ceiling) *before* random does as the sim budget grows — i.e. is the win a higher ceiling, or just getting to the same ceiling faster?

## Plan [CLAUDE]

Pure **re-analysis of existing leaves** — no new sims, no GPU. Leaves: `$NAMO_DATA_ROOT/eval/fullsearch_1push/{nohz_s1..3, rand_s0..9, steppen_s1..3}/shard_0.jsonl` (one JSONL record per episode, 1323 each; verified on arrakis). Each record carries `solved`, `n_sim`, and **`solve_ranks`** = the 0-indexed rank at which the first opening push was found — this is exactly the per-episode signal all three questions need.

Reuse (do NOT re-derive): `scripts/sandbox/agg_fullsearch_bydiff.py::div_1push()` + `load_seed()` — the **strict positional** episode↔leaf join (sorted-xml then per-xml order, verified 1323/1323, zero mismatch). Difficulty = `onepush_episodes.json` `solve_rate` tertiles (hard<0.169 / med<0.533). Extend the join to also carry each episode's GT record (`solve_rate`, `valid`, `tried`, `region`, `object_center`). **Never join by xml basename — basenames collide across shards.**

Three questions:

- **Q1 — the sub-100% floor (the "why do they also fail").** Isolate every `solved=false` episode per seed. Since budget 900 ≫ the ≤~35-push pool, every candidate is tried, so a miss = *no run-candidate opened the goal*. Test: is the unsolved SET identical across the model seeds and the random seeds (recon says yes — the same 4 episodes with identical `n_sim` in nohz_s1 and rand_s0)? If identical → the floor is a **shared pool/exhaustion property, not the ranker**. Then positionally join each miss to its GT `solve_rate`/`valid`/`tried` and classify: is it a genuine long-tail (near-zero `solve_rate`) or a **contradiction** (GT `solve_rate` high yet online never opens). **Chase the recon lead:** a basename match flagged one miss at GT `solve_rate=1.0` — resolve it with the CORRECT positional episode; if the contradiction survives, characterize it (candidate-pool mismatch between offline enumeration and online search, or a sim/nav determinism gap).

- **Q2 — does the model reach the ceiling before random?** Reconstruct the full success-vs-sim-budget curve from `solve_ranks` (solved-by-budget-B ⇔ `solved ∧ solve_ranks[0] < B`), model (3-seed mean) vs random (10-seed mean), per tier. Check **stochastic dominance** — model CDF ≥ random CDF at every B — and flag any crossover (expect a tie/near-tie at the top). Report **sims-to-ceiling** per tier. Expected answer to the literal question: **neither hits 100%**; both plateau at the same ceiling; the model reaches it in ~⅓ the sims. Reproduces the RESULTS @1/@2/@5/@10/@20/@900 ladder as a self-check.

- **Q3 — the ranking headroom on hard (the 1push analog of "setup at rank 38/70").** For SOLVED hard episodes, the distribution of `solve_ranks` (0-indexed rank of the opener): median, mean, fraction at rank 0 (=solve@1), fraction ≤4, fraction ≥10. Best possible = rank 0 (a valid push always exists). Compare model's actual rank to the **random-ranker expectation** `(n_tried − n_valid)/(n_valid + 1)` per episode — how much better than chance — and to optimal (0). Does the model's rank of the opener track pool rarity (GT `solve_rate`)? This says whether the headroom is a shallow calibration nudge (rank 2–5) or a real signal gap (rank 10+).

**Deliverables:** fill Run + Result in this card with (i) the floor characterization table + the resolved `solve_rate=1.0` contradiction, (ii) the dominance/sims-to-ceiling table + success-vs-sims plot (hard + all-tier panel), (iii) the hard `solve_ranks` distribution table + histogram, and a one-line verdict splitting the 1push gap into unfixable pool/label floor vs solvable-but-mis-ranked headroom. Split every table by difficulty. Owned files only: this card, `assets/1push_bottleneck_*.png`, `scripts/sandbox/agg_1push_bottleneck.py`.

## Run

_(filled on run)_

## Result

_(filled on run)_
