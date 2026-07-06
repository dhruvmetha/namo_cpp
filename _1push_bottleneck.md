---
type: experiment
status: done
created: 2026-07-05
updated: 2026-07-05
metric: "The 1push sub-100% floor is a shared pool/label property, NOT the ranker: every one of 16 seeds/models misses the same 3-episode core (+2 knife-edge episodes). 4 of 5 are single-opener long-tail pools (GT solve_rate ≤0.02) the online sim never re-opens; 1 is a genuine offline↔online contradiction (pos 953, sr=1.0 — all 45 candidates open offline, none open online; n_sim==n_tried==n_valid==45 so it's a sim/nav/label-provenance gap, not a pool-size or ranking failure). Neither model nor random reaches 100%; they share the SAME ceiling (all 99.7%, hard 99.2%, easy 99.8%, medium 100%). The model's win is reaching that shared ceiling in ~⅓ the sims (3–5× fewer at the 90–95% coverage band), not a higher ceiling; it stochastically dominates random over the entire informative range (B≤127), the only ≤0.13pp 'crossover' being plateau noise from 2/10 random seeds opening one knife-edge episode. On hard, 55% of solved episodes already sit at opener rank 0 (mean 5.5 vs random 18.3 vs optimal 0), but a rarity-correlated tail (corr −0.30; 17% at rank ≥10) holds the remaining headroom — step-pen already shaves the mean to 4.5, proving that tail is reachable."
tags:
  - experiment
  - diagnostic
---
# 1push bottleneck — why the sub-100% floor, and does the model reach the ceiling before random?

**Sibling to [[_ranker_bottleneck]] (which is 2push-only).** 1push has no *setup* move, so the 2push "setup under-ranking" story cannot apply — the diagnostic must find the *different* reason 1push tops out below 100%, and whether the learned ranker reaches that ceiling ahead of the random floor.

**Bottom line (plain English).** To reach a goal blocked by one movable object, the robot picks from ~35–130 candidate shoves; the model scores them and the planner simulates them best-score-first until one opens the goal. The question isn't *can* a room be solved — with enough tries almost every room opens — it's *how fast the model's top pick is the right one*, and *why a tiny handful of rooms never open at all*. Three answers: (1) the "never open" rooms are the **same ~5 of 1323 for the model and for plain random** — rooms where the single working push in a big pile just doesn't re-open when actually simulated (a data/physics quirk, not the model's fault); (2) the model does **not** solve more rooms than random given unlimited tries — it reaches the **same ceiling, just ~3× faster**; (3) on the hard rooms the model's top pick is already right over half the time, and where it's wrong the working push is usually genuinely rare in the pile — though a tweaked model (step_penalty) already finds it a bit faster, so some of that gap is recoverable.

## Plain-language key

| Term | What it means |
|---|---|
| **room** (episode) | one problem: the robot must reach a goal that a movable object is blocking |
| **push** | one action — the robot shoves the target object in some direction |
| **pool** / candidates | all the pushes considered for a room (~35–130 here); the enumerated list is `tried` |
| **opener** | a push that actually opens the goal in a single shot |
| **1push room** | a room solvable by *one* push (this whole experiment) — there is **no "setup" move**, unlike 2push |
| **solve_rate** | fraction of the pool that are openers — low = a needle in a haystack |
| **best-first search** | the planner tries candidates in model-score order, simulating each until one opens |
| **sim** | one physics rollout of a candidate push — the expensive unit of work |
| **budget** (B) | max sims allowed per room (900). It is **≫ the pool**, so every candidate gets tried |
| **rank** | where the opener sits in the model's sorted list — **rank 0 = the model's #1 pick** (we count from 0) |
| **solve@1 / solve@B** | did the room open using only the top-1 / top-B picks |
| **ceiling** | solve-rate given the whole budget — i.e. "does *some* push open it" |
| **floor** / miss | rooms that never open even after every candidate is tried |
| **easy / medium / hard** | rooms binned by `solve_rate` (fewer openers = harder) |
| **NoHz-v3** | the current deployed model. **step-pen** = a variant trained with a −1/0/1 reward. **random** = no model (uniform pick from the pool) |
| **seed** | an independently-trained model copy; we average 3 model / 10 random seeds for error bars |
| **stochastic dominance** | the model's success curve sits at-or-above random's at *every* budget (never worse anywhere) |
| **pp** | percentage points |
| **positional join** | matching each result row to its true room by fixed order, not by filename (filenames collide across shards) |

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

- **Box:** arrakis (`arrakis.cs.rutgers.edu`). Pure re-analysis — no new sims, no GPU, no training.
- **Python:** `/common/users/dm1487/envs/mjxrl/bin/python` (numpy 2.3.2, matplotlib 3.10.0, Agg backend). The CLAUDE.md `/scratch/dm1487/envs/namo/bin/python` is the Amarel interpreter and does not exist on arrakis.
- **Commit:** 982cf26 (branch feat/horizon-q-redesign).
- **Script:** `scripts/sandbox/agg_1push_bottleneck.py` (owned by this card). Reuses the strict positional episode↔leaf join from `agg_fullsearch_bydiff.py` (`div_1push` / `load_seed`), extended so every leaf row carries its TRUE GT episode record (`solve_rate`, `valid`, `tried`, `region`, `object_center`, `object_id`, `xml`).
- **Leaves:** `/common/users/dm1487/scratch_namo/eval/fullsearch_1push/{nohz_s1..3, rand_s0..9, steppen_s1..3}/shard_0.jsonl` — 1323 records each (one per episode). Seeds used: NoHz-v3 = 3, random = 10, step-pen = 3 (16 total).
- **Labels:** `/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/onepush_episodes.json` (991 xml keys → 1323 episodes). Difficulty = `solve_rate` tertiles: **hard < 0.1692 ≤ med < 0.5333 ≤ easy** (n=1323; matches `agg_fullsearch_bydiff.div_1push`).
- **Join integrity gates (hard-asserted in the script, all passed):** every seed has exactly 1323 rows; positional `object_id` match = **1323/1323, zero mismatch** for all 16 seeds; and `n_sim == solve_ranks[0]+1` for **every** solved row (so the n_sim-based and rank-based CDFs are identical). NEVER matched by xml basename (they collide across shards).

**Correctness gate — reproduces the RESULTS ladder EXACTLY (all-tier, from these leaves):**

| ranker | @1 | @2 | @5 | @10 | @20 | @900 |
|---|---|---|---|---|---|---|
| NoHz-v3 (3 seeds) — reproduced | 82.3 | 85.8 | 90.6 | 94.0 | 96.7 | 99.7 |
| NoHz-v3 — RESULTS target | 82.3 | 85.8 | 90.6 | 94.0 | 96.7 | 99.7 |
| random (10 seeds) — reproduced | 37.3 | 51.9 | 69.8 | 80.8 | 89.1 | 99.7 |
| random — RESULTS target | 37.3 | 51.9 | 69.8 | 80.8 | 89.1 | 99.7 |

Gate passes to the decimal — the positional join and CDF definition are correct, so the new numbers below are trustworthy.

## Result

### Q1 — the sub-100% floor: a shared pool/label property, not the ranker

Only **5 distinct episodes** are ever missed across all 16 seeds/models. Three are missed by **every** seed (the hard core); two are knife-edge cases opened by a lucky subset. The unsolved set is NOT bit-identical across seeds (the two knife-edge episodes flip), but the **core {922, 953, 1154} is missed by all 16** — and critically, **random misses the same episodes the model does**, so the floor is not a ranking failure.

| pos | tier | GT solve_rate | n_valid | n_tried | run n_sim | # seeds miss (of 16) | class |
|---|---|---|---|---|---|---|---|
| 922 | hard | 0.008 | 1 | 125 | 125 (all seeds) | **16/16** | single-opener long-tail |
| 953 | **easy** | **1.000** | **45** | **45** | **45 (all seeds)** | **16/16** | **CONTRADICTION** |
| 1154 | hard | 0.013 | 1 | 75 | 75 (all seeds) | **16/16** | single-opener long-tail |
| 420 | hard | 0.020 | 1 | 50 | 49–50 (miss) / 4 (2 seeds solve) | 14/16 | knife-edge long-tail |
| 1133 | hard | 0.014 | 1 | 70 | 70 (miss) / 2–67 (11 seeds solve) | 5/16 | knife-edge long-tail |

Reading: the floor lives in **hard** (4 episodes) and **easy** (1 episode); **medium tops out at exactly 100%**. Ceilings: easy 99.78%, medium 100.0%, hard 99.24% (model) / 99.34% (random), all 99.67%.

**Long-tail cases (4 of 5):** each has a single valid opener in a 50–125-candidate pool (GT solve_rate 0.008–0.020). For the 3 all-seed-and-knife-edge ones, `n_sim == n_tried` (the online search exhausted the whole enumerated pool and opened nothing), so the lone offline opener simply is not reproduced by the online car push. The two knife-edge episodes (420, 1133) *are* opened by a few seeds — 2/10 random seeds hit 420's single opener (at rank 3, `n_sim=4`), and 11/16 seeds solve 1133 — pure sampling luck on a 1-in-50-to-70 pool, unrelated to which ranker is used.

**The contradiction (pos 953), resolved with the POSITIONAL episode:** GT `solve_rate = 1.0`, `n_valid = n_tried = 45` — offline, **all 45** enumerated candidates open the goal. Yet **all 16 seeds** run `n_sim = 45` and open **none**. Since `n_sim == n_tried == n_valid == 45`, the online search tried the *same 45-candidate pool* and exhausted it — this is **not** a pool-size / candidate-generation mismatch. It is an offline-GT ↔ online-eval determinism gap. **One concrete hypothesis (UNVERIFIED — needs a re-run to confirm):** this episode's offline `valid`/`solve_rate=1.0` label was produced under a different push/nav condition than the current car online best-first eval (e.g. a different robot-start / settle / push-velocity, or a stale label from a prior collection config), so the "always opens" offline outcome does not transfer to the online car dynamics for this specific (object, goal). It is a label-provenance / sim-config gap, not a search or ranking limitation. **Proposed follow-up (for the orchestrator to gate — NOT executed here):** re-run this one episode under the exact eval config with a nav/qpos dump to check whether any of the 45 offline-valid pushes actually open online; if none do, the offline label for 953 is stale and should be corrected.

**Q1 verdict:** the 1push sub-100% floor is an **unfixable pool/label property**, not a ranker gap — 0.4% of the testset (5/1323 episodes), of which 4 are rare single-opener pools and 1 is a stale-label/sim-determinism contradiction. No ranker (model or random) clears it.

### Q2 — ceiling vs random: same ceiling, reached in ~⅓ the sims

Share of rooms solved within a sim budget B — the success-vs-budget curve — NoHz-v3 (3-seed mean) vs random (10-seed mean), per tier. A room counts as solved by budget B if its opener sits at rank below B (i.e. the planner reaches it within B tries):

| tier | ranker | @1 | @2 | @5 | @10 | @20 | @900 (ceiling) |
|---|---|---|---|---|---|---|---|
| easy | NoHz-v3 | 98.7 | 99.2 | 99.6 | 99.8 | 99.8 | 99.8 |
| easy | random | 72.4 | 90.6 | 99.0 | 99.7 | 99.8 | 99.8 |
| medium | NoHz-v3 | 94.0 | 96.2 | 98.3 | 99.6 | 99.9 | 100.0 |
| medium | random | 33.1 | 52.7 | 82.9 | 96.4 | 99.8 | 100.0 |
| hard | NoHz-v3 | 54.2 | 62.0 | 73.9 | 82.7 | 90.3 | 99.2 |
| hard | random | 5.9 | 12.0 | 27.1 | 46.3 | 67.6 | 99.3 |
| all | NoHz-v3 | 82.3 | 85.8 | 90.6 | 94.0 | 96.7 | 99.7 |
| all | random | 37.3 | 51.9 | 69.8 | 80.8 | 89.1 | 99.7 |

**Same ceiling.** Model and random converge to the identical plateau on every tier (all 99.67 vs 99.71; hard 99.24 vs 99.34; easy 99.78 vs 99.78; medium 100.0 vs 100.0). The ≤0.1pp differences are seed noise from the knife-edge episodes (random's hard ceiling is *higher* only because 2/10 random seeds opened episode 420 — not a random advantage). **Neither reaches 100%** except on the medium tier.

**Stochastic dominance holds over the entire informative range.** Model ≥ random at every budget **B ≤ 127**. The only budgets where random ≥ model are **B ≥ 128** at the fully-converged plateau, gap ≤ **0.13pp** (hard) / **0.04pp** (all) — the "773 crossovers" the script counts are all this sub-0.13pp plateau wobble from episode 420, not a real crossover. On easy, model ≥ random at literally every B.

**Budget-to-coverage — the model reaches the shared ceiling in a fraction of the sims:**

| tier | reach 90% | reach 95% | reach 99% |
|---|---|---|---|
| all | model B=5 / rand B=23 (**4.6×**) | 13 / 39 (**3.0×**) | 50 / 82 (1.6×) |
| hard | 20 / 49 (**2.5×**) | 33 / 67 (**2.0×**) | 101 / 116 (1.1×) |
| medium | 1 / 7 (**7.0×**) | 2 / 9 (**4.5×**) | 8 / 14 (1.8×) |
| easy | 1 / 2 (2.0×) | 1 / 3 (3.0×) | 2 / 5 (2.5×) |

The ratio is 3–7× at practical coverage (80–95%) and narrows to ~1× only at the very top, where both are gated by the same handful of long-tail episodes.

![[1push_bottleneck_success_vs_sims.png]]

**Q2 verdict:** the model's win is **NOT a higher ceiling** — it reaches the **same sub-100% ceiling ~3× faster** (3–5× fewer sims across the 90–95% coverage band, up to 10× at low budgets), and stochastically dominates random everywhere the curve is still rising.

### Q3 — hard-tier rank headroom: median already optimal, headroom in a rarity-correlated tail

Distribution of the opener's 0-indexed rank on **SOLVED hard** episodes (mean ± std across seeds), vs the analytic random-ranker expectation `(n_tried − n_valid)/(n_valid + 1)` and optimal (0):

| metric | NoHz-v3 (3) | step-pen (3) | random | optimal |
|---|---|---|---|---|
| median rank | 0.0 | 0.0 | 13.6 (analytic) | 0 |
| mean rank | 5.53 ± 0.13 | 4.54 ± 0.34 | 18.34 ± 1.03 (empirical) · 18.32 (analytic) | 0 |
| frac @ rank 0 (= hard solve@1) | 54.6% | 57.2% | — | 100% |
| frac ≤ 4 | 74.5% | 77.7% | — | 100% |
| frac ≥ 10 | 16.7% | 13.9% | — | 0% |

(The analytic random expectation 18.32 matches the empirical random mean 18.34 to 0.02 — the formula is validated.)

**The median hard episode is already optimal (rank 0)** — 55% of hard-solved episodes have the opener at rank 0, and 75% within the top 5. The mean (5.5) is dragged up by a **17% tail at rank ≥10**. That tail is **rarity-correlated**: `corr(opener rank, GT solve_rate) = −0.30` on hard-solved — the rarest pools (solve_rate ~0.01–0.02) carry the deepest ranks (binned mean ~12), while the less-rare hard pools (solve_rate ~0.10–0.16) already sit at rank ~0–2. So much of the tail is intrinsic pool difficulty, not blind mis-ranking. **step-pen** ranks the opener **shallower** across the board (mean 5.5 → 4.5, rank-0 frac 54.6% → 57.2%, frac ≥10 16.7% → 13.9%) — consistent with its +2.5pp hard solve@1 edge, and direct evidence that the tail is at least partly **reachable** rank headroom, not a hard wall.

![[1push_bottleneck_rank_hist_hard.png]]

![[1push_bottleneck_rank_vs_solverate.png]]

**Q3 verdict:** the hard-tier headroom is **NOT a uniform shallow nudge and NOT a total signal wall** — it is bimodal: the median opener is already rank 0, and the residual gap lives in a rarity-correlated ~17% tail (rank ≥10) where the single opener is genuinely rare in the pool. It is closer to a **real signal gap concentrated in rare-pool episodes** (rank 5–120 tail) than a cosmetic rank-2–5 calibration miss — but step-pen shows part of it is recoverable.

### Q3b — which slice of hard actually struggles: the rarest-opener rooms

"Hard" is not one thing. Splitting it by how rare the opener is (`solve_rate` = share of the candidate pool that opens the goal) shows the ranking deficit is confined to the bottom slice — everything from `solve_rate` 0.05 up is already near-optimal. Model = NoHz-v3 (3-seed mean), per fixed `solve_rate` bin within hard (n=441):

| slice (GT solve_rate) | rooms | pool size | openers in pool | model solve@1 | model mean opener rank | random E[rank] | rank / pool |
|---|---|---|---|---|---|---|---|
| [0.006, 0.020) — rarest | 70 | 100 | ~1.2 | 18.6% | 14.8 | 44.8 | 0.15 |
| [0.020, 0.050) | 134 | 84 | ~2.6 | 39.3% | 6.8 | 22.3 | 0.08 |
| [0.050, 0.100) | 118 | 83 | ~6.0 | 67.8% | 3.2 | 11.0 | 0.04 |
| [0.100, 0.169) | 119 | 72 | ~9.2 | 78.4% | 1.1 | 6.2 | 0.02 |

Reading: from `solve_rate` 0.05 up (the top ~54% of hard), the model is already near-optimal — top pick opens 68–78%, median opener rank 0. The whole ranking deficit lives in the bottom two slices, and the rarest one (`solve_rate` < 0.02, ~1 opener in a ~100-push pile, n=70) is where solve@1 collapses to **18.6%** and the opener sits at mean rank **~15**. **44% of the entire hard deep tail (rank ≥ 10) is in this one bin; ~80% is in the bottom half of hard.** (Within-hard quartiles tell the same story: Q1 solve@1 20.9% / mean rank 12.7 vs Q4 79.3% / 1.1.)

Is this a closable ranking gap or intrinsic difficulty? Both — but not a wall. The model still floats the lone opener into the top ~15% of the pile (`rank/pool` 0.15) vs random's middle (0.43) — **3× better than chance** — so it has real signal, just not enough to pin one needle among ~100 look-alikes. Evidence it is partly reachable: **step-pen cuts the mean rank in this exact worst bin from 14.8 → 10.5** (−4.4, ~30% shallower) with solve@1 barely moving (+1.0pp) — its known +2.5pp hard edge is concentrated here as a "finds the needle faster once the top pick misses" effect, not a "wins outright more often" one. **Lever for hard-1push: a sharper ranking signal on the rarest-opener rooms specifically, not a broad hard-tier fix.**

### Q3c — the same slices in wall-clock TIME (same-HW, sapphirerapids-exclusive)

The sim story above *is* a time story: every buried-opener sim costs wall-clock seconds, so the rarest-opener slice is also where the seconds pile up. Interleaved timing run `eval/fullsearch_time_1push/tri1_s1` (1 timing seed; all 3 rankers hit each episode back-to-back on one `--exclusive` sapphirerapids node, so they are same-HW comparable; NoHz-v3's search reproduces the sim anchor bit-for-bit and its avg t_wall reproduces the published 0.70 s all / 1.43 s hard). Avg wall-time per room and solve-within-T, per hard rarity slice:

| slice (GT solve_rate) | ranker | avg t_wall (s) | solve@1 s | @2 s | @5 s |
|---|---|---|---|---|---|
| [0.006, 0.020) — rarest | NoHz-v3 | **3.62** | 34% | 49% | 79% |
| | step-pen | **2.63** | 40% | 63% | 86% |
| | random | **7.78** | 7% | 13% | 34% |
| [0.020, 0.050) | NoHz-v3 | 1.48 | 61% | 80% | 96% |
| | step-pen | 1.51 | 69% | 81% | 93% |
| | random | 3.80 | 22% | 40% | 72% |
| [0.050, 0.100) | NoHz-v3 | 0.92 | 84% | 89% | 96% |
| | step-pen | 0.79 | 89% | 92% | 98% |
| | random | 2.12 | 32% | 63% | 91% |
| [0.100, 0.169) | NoHz-v3 | 0.58 | 92% | 97% | 100% |
| | step-pen | 0.62 | 92% | 95% | 99% |
| | random | 1.07 | 63% | 89% | 98% |

Reading: within "hard," wall-time spans a **6× range** — from **0.58 s** on the least-rare slice (basically solved, 92% within 1 s) to **3.62 s** on the rarest, where even the model opens only **34% within 1 s** (random: 7%). The rarest slice (n=70) *is* the entire hard-tier time cost — 6× slower than the easiest hard slice for the model — and it is where the model's absolute time-advantage over random is largest: **3.62 s vs 7.78 s, a 4.2 s / 2.15× gap** that shrinks monotonically to 0.49 s on the least-rare slice. This is Q3b restated in seconds: bad ranking on the rarest-opener rooms → many sims → seconds; sharpen the rank and the seconds collapse. **step-pen is the proof** — its whole hard-tier wall-time edge (all-hard 1.25 vs 1.43 s) is almost entirely this one slice, where it saves **~1.0 s** (3.62 → 2.63 s) by floating the opener ~30% shallower; in the other three slices it is within ±0.13 s.

Caveats: single timing seed → point estimates (the 3-seed solve-rate variance lives in the sim tables above); in these timing leaves the `model` field labels the NoHz-v3 slot `"Hz"` and the step-pen slot `"NoHz"` (slot naming from `time_bestfirst.py`), anchor-verified by the 1.43 s-hard / 0.70 s-all match.

### Q3d — ROOT CAUSE: training-time sampling starves the rare opener (1push) vs sparse setup collection (2push)

Measured directly on the training H5s, replicating the deterministic `sample_k=30` loss subsample over the 3 deployed seeds (`v4_hq_m2b_scorer` = 1push-exhaustive rows; `v4_hq_h2_scorer` H==2 = 2push rows). The 1push and 2push misses have **different root causes**.

**1push = a sampling problem (the opener is in the data; the loss just doesn't see it).** 1push rows carry the exhaustive `f_grid` (all ~58–90 reachable pushes labeled), so the opener IS a labeled positive. But `sample_k=30` uniformly picks 30 reachable cells and computes the loss only on those (deterministic per seed). For a rare-opener row (~1 positive) the opener lands outside the 30 most of the time — and the starvation rate falls across the rarity slices **in lockstep with the test-time failure**:

| pos_rate slice | training starvation (opener dropped from loss) | test solve@1 |
|---|---|---|
| < 0.02 (rarest) | **59%** | 19% |
| 0.02–0.05 | 24% | 39% |
| 0.05–0.10 | 5% | 68% |
| 0.10–0.17 | 0.5% | 78% |
| ≥ 0.17 | 0% | (solved) |

Left column (opener never enters the loss) and right column (top pick wrong at test) rise together — the fingerprint that the rare-opener miss is **training starvation, not model capacity**. `sample_k=30` is the right PU-masking trick for genuinely-sampled rows, but wrong for these exhaustive 1push rows where it discards a *known* positive. **Fix = positive-aware sampling** (always keep labeled openers in the loss, fill the rest of the 30 with sampled negatives). Small change in `scorer_data.py` + retrain; no new collection. Note this makes the earlier reachability-loss idea ([[_reachability_loss_v3]]) the wrong lever for 1push-hard — that supervises *unreachable* cells; the problem is *reachable* openers being under-sampled.

**2push = a collection problem (the setup often isn't in the data at all).** The H==2 rows are sparse, not exhaustive: median **27** labeled first-pushes/row (54% below the sample_k=30 cutoff, so for most rows sampling drops nothing), and **61% of H==2 rows have zero valid setups labeled**. Sampling starvation on the rare-positive H=2 rows is only ~6% — negligible; you can't drop a positive that isn't there. The dominant issue is that valid setups were never *discovered*: confirming a setup at H=2 needs a follow-up search per first-push, which collection can't run exhaustively (the no-GT-at-scale constraint), so most rows teach "no setup here." Consistent with [[_ranker_bottleneck]] (setup buried at rank 38/70 on hard) — the model can't rank a setup it saw few positive examples of. **Fix = collection / exploration** (search+bootstrap / ExIt), not sampling — the harder problem. (Caveat: the per-row "27" reads from `r_mask`; the robust, unambiguous facts are the 61%-zero-setup rows and the ~6% sampling starvation.)

### Overall verdict — splitting the 1push gap

- **Unfixable pool/label floor (~0.3% all-tier, ~0.7% hard):** 5 episodes no ranker clears. 4 are single-opener long-tail pools (solve_rate ≤0.02) whose lone offline opener the online sim doesn't reliably reproduce; 1 (pos 953, sr=1.0) is a stale-label / sim-determinism contradiction. Random misses the same set — this is the ceiling, not a model deficit.
- **Solvable-but-mis-ranked headroom (hard tier):** the median opener is already at rank 0 (55% at solve@1), so the lever is the rarity-correlated ~17% tail at rank ≥10 (mean 5.5 vs optimal 0). It tracks pool rarity (corr −0.30), so it is partly intrinsic difficulty — but step-pen already pulls the mean 5.5 → 4.5, proving reachable rank headroom remains.
- **The literal question answered:** the model does **not** reach 100% before random — **neither reaches 100%** (both plateau at the same ~99.7% all / ~99.2% hard ceiling). The model's advantage is **reaching that shared ceiling in ~⅓ the sims** and dominating random at every budget the curve is still climbing.
