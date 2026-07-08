---
status: hub
tags:
  - results
updated: 2026-07-06
---
# Results

Paper-style compilation: for each experiment, the **main table + main figure + a tight key finding**. Full detail and verbose analysis live in each experiment's card (`_*.md`), linked per section. **Setting:** CAR robot, testset `namo_testset_v1`, region-opening criterion. Every result is split by **difficulty (easy/med/hard) × horizon (1push / 2push)**. ⚠ Difficulty is defined *per horizon* and is not the same scale across them: **2push** difficulty = number of solving first-pushes (`n_setups` → `division`); **1push** difficulty = `solve_rate` tertiles. So "hard" = *few solving setups* for 2push, *few opening pushes* for 1push — compare within a horizon, not across.

**Contents** — 1. [Reactive vs floor](#1-reactive-control-learned-value-vs-the-random-floor) ✅ ·
2. [Best-first search vs floor](#2-best-first-search-learned-value-vs-the-random-floor) ✅ (incl. 1push floor diagnostic) ·
3. [Step-penalty (−1/0/1)](#3-step-penalty-101-reward) ◑ softened reject ·
4. [The setup bottleneck & the fix](#4-the-setup-bottleneck-why-search-stalls-and-the-fix) ✅ ·
5. [Reactive MPC depth 5](#5-reactive-mpc-to-depth-5-more-pushes-dont-buy-back-the-search-gap) ✅ budget ≠ search · [Prior work](#prior-work-seeded-ledger)

---

## 1. Reactive control: learned value vs the random floor

The main model **NoHz-v3** (learned value, argmax setup → argmax finish) vs a **uniform-random** push of the labelled object (no model), under the forced-dive reactive protocol. Random = 10 seeds, NoHz-v3 = 3 seeds; mean ± std of region-opening rate (%). → card: [[_reactive_search]].

**Table 1a. 2push** — does the region open within **2 pushes** (metric = open@2, %)? Pure-2-push set, n = 1018.

| difficulty | random | **NoHz-v3** | lift (pt) |
|---|---|---|---|
| easy | 9.7 ± 1.6 | **61.2 ± 2.4** | +51.5 |
| medium | 4.4 ± 0.9 | **44.3 ± 2.9** | +40.0 |
| hard | 1.8 ± 0.6 | **27.5 ± 2.0** | +25.7 |
| *overall* | *4.7 ± 0.6* | ***42.1 ± 1.7*** | *+37.4* |

**Table 1b. 1push** — does it open within **1 push** (metric = open@1, %)? One-push set, n = 1323.

| difficulty | random | **NoHz-v3** | lift (pt) |
|---|---|---|---|
| easy | 71.7 ± 2.1 | **98.7 ± 0.4** | +27.1 |
| medium | 32.6 ± 3.1 | **93.9 ± 0.5** | +61.3 |
| hard | 6.2 ± 1.6 | **54.3 ± 0.4** | +48.0 |
| *overall* | *37.0 ± 1.1* | ***82.3 ± 0.2*** | *+45.3* |

![[react_search.png]]

**Finding.** The learned value beats the random floor in **every cell, both horizons** — ~9× on 2push (42.1 vs 4.7) and +45pt on 1push (82.3 vs 37.0). The 2push lift shrinks with difficulty (easy +51.5 → hard +25.7): random almost never cracks hard 2push (1.8%), and the model also drops most there (27.5%) — hard 2push is where the headroom is. The random band is tight (±0.6–3.1) because each seed averages ~1000 episodes, so its seed-to-seed std is just the binomial SE of a proportion — verified, not noise.

---

## 2. Best-first search: learned value vs the random floor

Greedy best-first search, budget **900 sims/instance**, combine=q: NoHz-v3's predicted Q ranks expansions vs uniform-random ordering. 10 random seeds, 3 model seeds. → card: [[_full_search]] (full per-cutoff tables, a/b/c/d diagnostics, time-by-difficulty).

**Table 2a. 2push wall-time** — % solved within wall-clock *t*, and avg t_wall per instance (emeraldrapids-exclusive, model & random same hardware). Pure-2-push set, n = 1018. *(This is the lead: where does the time actually go?)*

| difficulty | ranker | @1 s | @5 s | @30 s | avg t_wall (s) |
|---|---|---|---|---|---|
| easy | random | 21.1 | 69.0 | 96.8 | 6.3 |
| easy | NoHz-v3 | 62.9 | 85.9 | 94.3 | 7.1 |
| medium | random | 12.6 | 47.0 | 84.3 | 17.3 |
| medium | NoHz-v3 | 50.0 | 74.6 | 90.5 | **10.6** |
| hard | random | 6.7 | 25.9 | 55.7 | 46.5 |
| hard | NoHz-v3 | 35.6 | 57.3 | 77.1 | **26.3** |
| *all* | *random* | *12.4* | *44.5* | *76.8* | *25.4* |
| *all* | *NoHz-v3* | *47.8* | *70.9* | *86.5* | ***15.5*** |

![[fullsearch_success_vs_time_bydiff_2push.png]]

**Table 2b. 2push sims** — the *same* runs by sim count, to see **where that time goes**: % solved within budget B and avg sims to solve.

| difficulty | ranker | @2 | @30 | @900 | avg sims |
|---|---|---|---|---|---|
| easy | random | 6.6 | 66.3 | 99.8 | 43 |
| easy | NoHz-v3 | 55.0 | 85.7 | 98.9 | 38 |
| medium | random | 4.1 | 43.6 | 97.0 | 123 |
| medium | NoHz-v3 | 40.7 | 75.4 | 97.8 | 62 |
| hard | random | 1.6 | 22.0 | 78.7 | 344 |
| hard | NoHz-v3 | 26.1 | 56.1 | **90.2** | 165 |
| *all* | *random* | *3.7* | *41.1* | *91.0* | *185* |
| *all* | *NoHz-v3* | *38.7* | *70.8* | ***95.3*** | *94* |

![[fullsearch_success_vs_sims.png]]

**1push** — sub-second either way (avg **0.70 s** NoHz-v3 vs **1.35 s** random; hard 1.25 vs 1.43 s). Both reach the same ~99.7% solve ceiling, so the only difference is rank: the model's first pick opens the goal **82.3%** vs random **37.3%** (hard 54.2 vs 5.9). Full 1push table in the card.

**1push — why it tops out below 100% (full diagnostic → [[_1push_bottleneck]]).** The shared ~99.7% ceiling is an **unfixable pool/label floor, not a ranker gap** — only **5 / 1323** episodes are ever missed and **the model and random miss the identical set**, so **neither reaches 100%** and the model's win is reaching that same ceiling **~3× faster**, not a higher one. On hard the median opener already sits at **rank 0** (55% solve@1); the residual headroom is a small rare-pool ranking tail. **Not** the 2push setup-under-ranking story (1push has no setup). Per-episode floor breakdown, dominance/budget tables, and the rank distribution live in the card.

![[1push_bottleneck_success_vs_sims.png]]

**Finding.** In **wall-time** the model's win is real but lives **entirely on hard 2push** — **26 s vs 46 s**, and **48% solved in the first second** vs random's 12%. On **easy** the curves **cross** (**7.1 s vs 6.3 s**) — *not* from NN cost (scoring is only 3% of wall-time), but because the model's individual sims are costlier (178 vs 146 ms) and outweigh its small sim savings when an instance is already trivial. **The sims table shows the real story:** the model reaches the solution in ~half the sims (all 94 vs 185), but on easy/medium random eventually brute-forces the same ~98% ceiling — so the model buys **speed on hard**, not new solutions. **Where it's stuck** (full diagnostic → [[_ranker_bottleneck]]): both the sims tail and the 95%→100% ceiling gap trace to the **first push (the setup) being under-ranked** — the myopic q-head buries setups (which open nothing *yet*), so search dives wrong branches before reaching the true setup (sim-cost correlates **0.79** with first-push rank; on the 21 robust misses the setup sits at **median rank 38/70**). The earlier `dive_bonus` / H1-H2 idea is **refuted** — the dive is the *stronger* ranker (mean rank 2.05 vs the first push's 3.28); the fix is to rank first-pushes by **setup value**, not myopic q.

---

## 3. Step-penalty (−1/0/1 reward)

**Verdict: softened reject (horizon-split).** We retrain the no-horizon q-scorer on a *signed* target (+1 immediate-open / 0 valid-setup / −1 never-opens) and test whether it ranks pushes better for best-first search than the incumbent 0/0.9/1. 3-way vs random and NoHz-v3, mean ± std across seeds. → card: [[_step_penalty_]].

**Table 3a. Fair 3-way wall-time** (interleaved, sapphirerapids-exclusive; avg t_wall per instance, seconds) — the lead. Pooling verified: the timed NoHz-v3 reproduces the full-search sim anchor bit-for-bit (0/2341 mismatch) and matches its emeraldrapids wall-times within ~5%.

| horizon | random | NoHz-v3 | step-pen |
|---|---|---|---|
| 2push | 26.7 | 16.0 | 15.6 |
| 1push | 1.35 | 0.70 | 0.63 |

![[steppen_time_bydiff.png]]

**Table 3b. 2push search — sims** (combine=q, budget 900, n = 1018) — the ranking diagnostic behind the time.

| difficulty | ranker | @2 | @30 | @900 | sims-to-solve |
|---|---|---|---|---|---|
| easy | NoHz-v3 | 55.0 | 85.7 | 98.9 ± 0.4 | 28 |
| easy | step-pen | 54.9 | 87.0 | 99.2 ± 0.0 | 26 |
| medium | NoHz-v3 | 40.7 | 75.4 | 97.8 ± 0.8 | 43 |
| medium | step-pen | 39.4 | 74.5 | 97.4 ± 0.6 | 44 |
| hard | NoHz-v3 | 26.1 | 56.1 | 90.2 ± 0.8 | 88 |
| hard | step-pen | 22.6 | 54.1 | 90.7 ± 0.8 | 97 |
| *all* | *random* | *3.7* | *41.1* | *91.0 ± 0.8* | *115* |
| *all* | *NoHz-v3* | *38.7* | *70.8* | ***95.3 ± 0.6*** | *55* |
| *all* | *step-pen* | *36.9* | *70.0* | ***95.4 ± 0.5*** | *58* |

![[steppen_bestfirst_sims_2push.png]]

**Table 3c. 1push search — sims** (n = 1323). solve@1 = does the #1-ranked push open in one sim; @900 ties because the one-push pool is tiny (both eventually solve everything), so the contest is entirely front-loaded rank.

| difficulty | ranker | solve@1 | solve@900 | sims-to-solve |
|---|---|---|---|---|
| easy | NoHz-v3 | 98.7 | 99.8 | 1 |
| easy | step-pen | 98.5 | 99.8 | 1 |
| medium | NoHz-v3 | 94.0 | 100.0 | 1 |
| medium | step-pen | 94.7 | 100.0 | 1 |
| hard | NoHz-v3 | 54.2 | 99.2 | 7 |
| hard | **step-pen** | **56.7** | 99.1 | 6 |
| *all* | *random* | *37.3* | *99.7* | *8* |
| *all* | *NoHz-v3* | *82.3* | *99.7* | *3* |
| *all* | *step-pen* | ***83.3*** | *99.6* | *3* |

![[steppen_bestfirst_sims_1push.png]]

Reactive open-rate (secondary, success only), Δ = step-pen − NoHz-v3: 2push open@2 all **−2.5** (hard −3.5); 1push open@1 all **+1.0** (hard **+2.5**) — the same 1push-hard signal the search shows.

**Finding.** In **wall-time** (Table 3a) step-penalty is a **non-event**: 2push **15.6 vs 16.0 s** (tie), 1push **0.63 vs 0.70 s** — no horizon where the signed target costs or clearly saves time; both far below random (26.7 s, gap on hard ~27 vs 48 s). **The sims diagnostic explains the (lack of) difference — and the one real signal.** On **2push** search ranking it's a wash (solve@900 95.4 vs 95.3, marginally *worse* at low budget @2 36.9 vs 38.7, where a sharper ranker should win). On **1push** it earns a small, real edge in the ranking metric — **solve@1 83.3 vs 82.3 (+1.0 all, +2.5 hard)** — reproduced *exactly* in reactive open@1, so it's real. **Call: 0/0.9/1 stays incumbent** — it wins the harder 2push axis and the 1push edge is small and time-invisible — but the hypothesis is **not cleanly false**: the signed target sharpens the *open-now* decision, not the *setup* decision. Natural follow-up: apply −1/0/1 to the open-now (H1) head only, keep 0/0.9/1 for setup.

---

## 4. The setup bottleneck: why search stalls, and the fix

Three diagnostics close one loop: the search stalls because the model buries the **setup** — a first push that opens nothing yet but sets up a finish — and we now know *why* it buries it and *what* to do. → cards: [[_ranker_bottleneck]], [[_setup_value_check]], [[_setup_label_quality]].

**Table 4a. The setup is the bottleneck** (2push, hard tier). How the first push is ranked → is the true setup the top pick, and do the hard rooms solve?

| ranking rule | true setup is #1 | true setup median rank | hard rooms solved |
|---|---|---|---|
| current — "does it open now?" (q) | 18.9% | 5 | 90.2% |
| peek-ahead — "what finish does it enable?" | 33.7% | 2–3 | — |
| perfect setup (oracle) | 100% | 0 | **98.1%** |

Handing the search a perfect setup lifts hard-room solve **90.2 → 98.1%** and rescues **16 of the 21** always-missed rooms — and the finish move holds up once the setup is right (solves 97.6%). So setup-ranking is the dominant bottleneck. Ranking a first push by the finish it enables un-buries the setup (and, notably, helps *most* on hard) but isn't sharp enough to pin it #1 alone.

![[setupval_separation.png]]

**Table 4b. Why the model never learned setups** — the training labels under-count them. Of the first pushes the training data stamped "never opens," how many are actually real setups (a finish exists on exhaustive re-search)?

| where | "never opens" labels that are actually setups |
|---|---|
| dead scenes (no setup exists) | 0.8% — labels are fine |
| solvable scenes (a setup exists) | **41.8%** |
| — collection tried <40% of follow-ups | 43% |
| — collection tried >80% of follow-ups | 0.5% |
| *overall* | *16.7%* |

The model was trained to call **~2 of every 5 real setups** (inside solvable scenes) worthless — driven purely by the collection's sampling budget (correlation of coverage vs mislabel = −0.51). Dead scenes are labeled correctly, so the model can trust "this scene is dead"; the leak is missed setups inside solvable ones.

![[setuplabel_fractried.png]]

**Finding.** The whole story in one line: the search buries the setup because the model was *taught* to (bad labels), and un-burying it works (perfect setup → 98% hard). So the fix is **better labels** — more follow-up moves per first push during collection, or bootstrapped re-labeling with the current model — plus a **trained setup-value target** shaped as the *top-few* finishes a move enables (top-3 ties the single best; a plain average or count lags). This is *not* a reward or loss tweak: step-penalty (§3) already proved that changing the target *number* does nothing, because the number was never the problem. Ceiling: the ~2% gap to 100% is **fixable plumbing, not a floor** — of the 13 "impossible" rooms, **0 are genuine** (2 have an inconsistent open-criterion between collection and eval, 9 are push-controller jams/under-pushes, 2 are just eval noise), so the true achievable ceiling is ~100% once the criterion is unified and the controller is fixed. → [[_offline_online_gap]]. That's a *separate* fix from the label-fix (which sharpens ranking): these raise the ceiling. ⚠ Flagged there too: the eval sim is **non-deterministic at ~0.3 mm** (MuJoCo warmstart), enough to flip near-threshold rooms — so single-run "never opens" numbers near the ceiling are noisy.

---

## 5. Reactive MPC to depth 5: more pushes don't buy back the search gap

Can the no-search regime close the gap to search by simply **executing more pushes** (MPC: argmax, push, re-look, repeat — no undo)? Same forced-dive protocol as §1 extended to a ≤5-push loop (labeled object only, early-stop on open). NoHz-v3 = 3 best-val ckpt-seeds; random = 10 seeds. → card: [[EXP-2026-07-06-reactive-mpc-depth5]].

**Table 5a. 2push** — cumulative % of episodes whose region opens within k pushes (open@k). Pure-2-push set, n = 1018; open@1 = 0 by construction. Search reference: best-first solve@900 = **95.9**.

| difficulty | ranker | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|
| easy | random | 8.0 ± 1.4 | 23.0 ± 2.6 | 35.5 ± 3.8 | 46.2 ± 2.9 |
| easy | NoHz-v3 | **59.8 ± 3.6** | 66.8 ± 3.0 | 67.4 ± 3.6 | 67.9 ± 3.8 |
| medium | random | 4.5 ± 0.8 | 12.6 ± 1.9 | 21.6 ± 1.9 | 31.6 ± 2.4 |
| medium | NoHz-v3 | **42.5 ± 1.6** | 55.5 ± 2.5 | 57.6 ± 2.0 | 58.1 ± 2.0 |
| hard | random | 2.1 ± 0.5 | 7.0 ± 1.1 | 13.3 ± 2.1 | 21.3 ± 2.4 |
| hard | NoHz-v3 | **26.3 ± 0.7** | 43.3 ± 2.4 | 46.3 ± 1.6 | 47.3 ± 1.4 |
| *all* | *random* | *4.5 ± 0.5* | *13.0 ± 1.0* | *21.8 ± 1.6* | *31.3 ± 1.7* |
| *all* | *NoHz-v3* | ***40.7 ± 0.2*** | ***53.7 ± 0.9*** | ***55.8 ± 0.5*** | ***56.5 ± 0.4*** |

**Table 5b. 1push** — same, open@1..5. One-push set, n = 1323.

| difficulty | ranker | open@1 | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|---|
| easy | random | 72.6 ± 1.9 | 91.3 ± 0.9 | 96.3 ± 0.6 | 97.8 ± 0.6 | 98.4 ± 0.5 |
| easy | NoHz-v3 | **98.7 ± 0.4** | 99.4 ± 0.3 | 99.4 ± 0.3 | 99.4 ± 0.3 | 99.4 ± 0.3 |
| medium | random | 33.0 ± 2.2 | 60.4 ± 2.1 | 74.8 ± 2.5 | 82.9 ± 2.2 | 87.2 ± 1.8 |
| medium | NoHz-v3 | **93.9 ± 0.5** | 96.8 ± 0.2 | 96.9 ± 0.1 | 96.9 ± 0.1 | 97.0 ± 0.2 |
| hard | random | 6.4 ± 1.4 | 21.5 ± 1.8 | 35.3 ± 1.7 | 47.5 ± 1.6 | 56.7 ± 1.7 |
| hard | NoHz-v3 | **54.3 ± 0.4** | 73.0 ± 1.5 | 76.6 ± 1.7 | 77.4 ± 1.1 | 77.6 ± 1.1 |
| *all* | *random* | *37.5 ± 1.1* | *57.9 ± 1.0* | *68.9 ± 1.1* | *76.1 ± 0.9* | *80.8 ± 0.7* |
| *all* | *NoHz-v3* | ***82.3 ± 0.2*** | ***89.7 ± 0.5*** | ***91.0 ± 0.5*** | ***91.3 ± 0.3*** | ***91.3 ± 0.4*** |

![[react_mpc_d5.png]]

**Finding.** **The model plateaus by push 3 — extra budget closes only ~29% of the reactive-vs-search gap; the other ~71% needs simulate-and-undo.** On 2push·all, open@k goes 40.7 → 53.7 → 55.8 → 56.5: the per-push gains collapse (+13.0 / +2.1 / +0.7), leaving a **~39pp gap** to best-first search (95.9) even at 5 real pushes. Hard 2push caps at **47.3%** (more than half of hard scenes never open reactively, vs ~96 under search), and even 1push scenes carry a ~9% reactive-irrecoverable tail (saturates at 91.3). Read: **greedy mistakes are largely irreversible** — you can't push your way out of a bad first push; you have to be able to take it back, which is exactly what search buys. Secondary: random *doesn't* plateau (2push 4.5 → 31.3 by push 5), eating the model's lift from +36.2 to +25.2 — blind persistence slowly accumulates what early greedy commitment forgoes, but stays ~25pp behind. Anchor check passed (open@1/@2 reproduce §1 within seed noise; the §1 2push mean of 42.1 traced to a non-best-val ep011 s3 ckpt in the reused legacy leaves — the registry-consistent ep012 number is **40.7 ± 0.2**). Pre-registered caveat: not compute-matched — search spends *sims*, MPC spends *real pushes*; this measures what falls to zero-simulation control.

---

## 6. RL-only self-imitation loop: it learns, it doesn't transfer — forecast falsified at gen-1

Can pure RL — forward rollouts only, filtered-BC self-imitation, greedy deploy, no search anywhere — climb from a random policy toward search-level performance? Two arms (A = from scratch, B = guided collection via NoHz-v3 at calibrated T), pool = 5000 v4_hq_h1 episodes / 903 rooms (room-held-out, geometry-verified disjoint from the testset), R=16 depth-10 rollouts/episode/gen, ~3.5×10⁵ sims/arm/gen. Registered external forecast (GPT-5.5 xhigh, pre-run): 2push-all ~70 / hard ~53 by gen-5. → card: [[EXP-2026-07-06-rl-only-self-imitation]] (incl. the Phase-0 oracle-decomposition gate: wrong-setup = 74.6% of greedy failures; a finishable setup sits in the model's top-8 for 82.5% of episodes — mis-ranked, not missing).

**Table 6a. The climb test — canonical 2push testset (pure2push n=1018, greedy reactive open@2), gen0→gen1:**

| arm | easy | medium | hard | all |
|---|---|---|---|---|
| A (scratch) | 24.8→27.3 | 13.4→13.7 | 7.0→6.7 | 13.8→14.3 |
| B (guided) | 27.3→25.6 | 13.7→12.5 | 8.9→7.5 | 15.1→13.8 |
| *NoHz-v3 baseline* | *64.7* | *40.8* | *25.3* | *40.8* |

**Table 6b. Own-family dev (501 eps, 1push-pool episodes), gen0→gen1:**

| arm | med greedy open@2 | hard greedy open@2 | hard setup-hit@1 | hard key-hit@8 |
|---|---|---|---|---|
| A | 50.9→**59.3** | 15.1→13.9 | 25.3→24.7 | 50.0→50.0 |
| B | 66.5→65.3 | 14.5→16.3 | 25.9→27.7 | 77.8→**83.3** |

**Table 6c. The mechanism test — testset 1push tier (n=1323, gen-1 π, greedy open@1, solve-rate tertiles 438/444/441):**

| policy | easy | med | hard | all |
|---|---|---|---|---|
| random floor | 71.7 | 32.6 | 6.2 | 36.8 |
| A (scratch) gen-1 | 82.0 | 64.2 | **27.7** | 57.9 |
| B (guided) gen-1 | 85.2 | 65.5 | 22.7 | 57.7 |
| *NoHz-v3* | *98.7* | *93.9* | *54.3* | *82.3* |

**Finding.** **Kill-signal-2 fired (hard 7.5/6.7 ≪ 35) — the forecast is falsified — but by a failure mode the forecast didn't model.** The loop *learns*: one generation took a uniform policy to a strong in-distribution ranker (dev setup-hit@1 99/93/26 easy/med/hard — ~5× over uniform on hard), and the collection flywheel demonstrably turns (hard coverage rising 0.58→0.62, +1,711 unique hard solves across arms, 60–68% of solved-hard episodes bank a ≤2-push solution — so this is **NOT** the pre-registered coverage-failure branch). What failed is (1) **slope** — gen-0→gen-1 (~13% more data) moved dev metrics barely (arm A med +8.4 the only real climb) — and (2) **transfer to the testset's task composition** — flat-to-down on pure2push. **CORRECTION [2026-07-07, USER-caught]:** the initial "different room-generator family" attribution was WRONG — the testset README proves both corpora descend from the same feb_car+aug9_car templates (33% shared wall floorplans; disjointness is per-scene by design), and the collector's region-open bar matches the testset's (≥20%). The real shift is **episode composition**: pure2push episodes are exhaustively verified F=∅ (no single push opens — every episode demands setup→finish), while the training pool is 1-push-manifest episodes dominated by direct-push solutions (genuine setup-chains only in the thin hard tail, ~4.6k/25k buffer solves). The policy mastered the skill its data taught (direct pushes, dev near-ceiling) and barely practiced the only skill pure2push grades. **Table 6c is the confirming measurement:** on the SAME testset rooms, the gen-1 π scores well above the random floor on its practiced 1push task in EVERY tier (med 2×, hard 4×) while collapsing on 2push — a skill gap, not a rooms gap. The secondary wall is also quantified there: ~25pt below NoHz-v3 on every 1push tier (competent-but-weaker ranker, consistent with ~7× less training data). Pretraining's measured worth: nothing on easy, +12.6pt med greedy, hard setup-surfacing key-hit@8 77.8 vs 50.0 (gen-0, B vs A). V-head never trained (hl_gauss-specific hang; evidence + repro in the card; disabled by default). Highest-leverage next test per the card: pool from the testset's own room family — does the climb appear when pool family = eval family?

## Prior work (seeded ledger)

Compact history, pre-loop. Detail in the [model registry](horizon_q_model_registry.md).

| Date | Experiment | Metric | Verdict |
|---|---|---|---|
| 2026-06-29 | Render speedup (`fast_scorer`) | 2019→101 ms · render-equiv 158/158 | ✅ accept, no retrain |
| 2026-06-27 | NoHorizon vs Horizon @2 | reactive 40.7 / best-first 37.8; NoHz ≥ Hz | ~ tie (NoHz ≥ Hz) |
| 2026-06-15 | M2b (+ dead-ends), 1-push | hard@1 32.86 ± 2.4 · 2-push e2e 61.9% | ✅ best 1-push model |
