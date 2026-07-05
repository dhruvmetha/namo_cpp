---
type: experiment
status: done
created: 2026-07-04
updated: 2026-07-05
metric: "The search ranker stalls at 95% (90% on hard) because it can't rank SETUP pushes — moves that open nothing yet. On hard rooms its #1 pick is a valid setup only 19% of the time; on rooms it never solves, the true setup sits at rank ~38 of ~70. It HAS signal (detector AUC 0.80) just not sharp enough to put the setup first. Fix = train for setup value (leads-to-a-solution), not immediate opening. NOT horizon-conditioning (verified no help), NOT a dive bonus."
tags:
  - experiment
  - diagnostic
---
# Ranker bottleneck — why the search ranker can't finish the job

**Bottom line.** To reach a blocked goal the robot usually needs **two pushes**: a *setup* (shove an obstacle aside — this opens nothing by itself) and then a *finish* (this one actually clears the path). The model scores every candidate push and the planner tries them highest-score-first. It is **excellent at spotting the finish** push but **bad at spotting the setup** — because it was trained to predict *"does this open the goal now?"*, and a setup opens nothing now. So it buries the good setup, wastes its search budget on wrong pushes, stalls at ~95% (90% on the hard rooms), and is slow exactly where the problem is hard. Every claim below is measured against the true answer. **The fix is to train the model to value a *setup* (a move that leads to a solution), not just an immediate opening.**

---

## Plain-language key

| Term | What it means |
|---|---|
| **room** (episode) | one problem: a robot must reach a goal that movable objects are blocking |
| **push** | one action — the robot shoves the target object in some direction |
| **setup** (first push) | a push that opens nothing yet but clears the way for a second push that does |
| **finish** / **dive** (second push) | the push that actually opens the goal, made after a setup. "Dive" = the search exploring a first push's follow-on pushes |
| **opener** | any second push that opens the goal |
| **2-push room** (`pure2push`) | a room needing exactly two pushes; our exhaustively-solved test set, so we know the true answer |
| **best-first search** | the planner tries candidate pushes in order of the model's score — always the highest-scored untried push next — simulating each until one solves |
| **sim** | one physics rollout of a candidate push; the expensive unit of work |
| **budget** | max sims allowed per room (here, 900) |
| **q / score** | the number the model gives each candidate push; the planner ranks by it (higher = more promising) |
| **value** `V(s)` | how solvable a *state* looks — as opposed to `q`, the promise of a single *action*. The proposed fix ranks by this |
| **rank** | where the winning push sits in the model's sorted list — **rank 0 = the model's #1 pick** |
| **needle** | a room where only 1–2 of ~40–70 candidate pushes is a valid setup (a needle in a haystack) — the hard rooms |
| **easy / medium / hard** | rooms binned by how many first pushes work (more working setups = easier) |
| **NoHz-v3** | the current deployed model. **Hz** = a variant told how much budget is left. **step_penalty** = a variant trained with a −1/0/1 reward |
| **seed** | an independently-trained copy of the model; we use 3 and average (for error bars) |

**Detector terms** — we grade the model's score as a yes/no detector of *"is this push good?"* against the known true answer:

| Term | Plain meaning |
|---|---|
| **recall** | of the *truly good* pushes, how many the model scores high enough to try (does it *catch* them?) |
| **precision** | of the pushes the model *flags as good*, how many actually are (does it avoid false alarms?) |
| **FPR** (false-positive rate) | of the *truly bad* pushes, how many the model wrongly flags as good |
| **AUC** (ROC-AUC) | one number for how well the score separates good from bad at *all* thresholds: 0.5 = coin-flip, 1.0 = perfect |
| **p@1, p@k** | is the model's **top pick** (or top-k) actually good? This is what search cares about — it tries the top first |
| **base rate** | the fraction of candidates that are actually good (how sparse the needle is) |
| **op** (operating point) | the score threshold the search effectively uses |
| **ground truth (GT)** | the true answer — which pushes really solve — known because `pure2push` is exhaustively solved |

---

## What the search does, and what we tested

Best-first search works on the target object's candidate pushes. The model scores all of them in **one pass**; the planner repeatedly takes the highest-scored untried push, simulates it, and — if it doesn't open the goal — explores the pushes available *after* it (the "dive"). It stops when a push opens the goal or the 900-sim budget runs out.

We nailed down two facts first, because they rule out two tempting explanations:

1. **The model has a single scoring "head."** It scores the first push and the dive with the *same* function on the *same* scale — so "two heads on mismatched scales" is impossible here.
2. **The score is squashed through a sigmoid** into a narrow band (~0.50–0.70), but a sigmoid preserves order, so it changes *which push ranks first* not at all. A red herring for the search.

**What we're testing:** we want the learned ranker to solve ~100% of rooms *faster* than blind random search. Today it stalls at **95.3%** (90.2% on hard), and on easy rooms it's even a hair *slower* than random in wall-clock time.

---

## What we found

### 1. It's the first push (the setup) that's mis-ranked — not the finish

Where does the *winning* push land in the model's sorted list? (rank 0 = its #1 pick.)

| push | picks it #1 | in top-3 | buried (rank ≥5) | median rank | mean rank |
|---|---|---|---|---|---|
| **setup (1st push) — model** | 50.9% | 69.8% | 22.0% | 0 | **3.28** |
| **finish (dive) — model** | 70.0% | 82.8% | 12.5% | 0 | **2.05** |
| setup — random baseline | 14.9% | 39.6% | 43.6% | 4 | 5.95 |

And by difficulty — the setup ranking **collapses on hard**, while the finish holds up:

| difficulty | setup #1 | setup median | setup mean | setup buried ≥5 | finish #1 | finish mean |
|---|---|---|---|---|---|---|
| easy | 65.2% | 0 | 2.10 | 14.9% | 81.6% | 1.03 |
| medium | 52.3% | 0 | 2.79 | 19.0% | 70.8% | 1.80 |
| **hard** | **39.1%** | **1** | **4.68** | **30.6%** | 61.0% | 3.06 |

**Read:** the finish push is ranked *well* — once the search reaches a real setup's follow-on state, the model spots the goal-opening second push and picks it #1 **70%** of the time. It's the **setup** the model can't float to the top, and on hard rooms its top pick is the true setup only **39%** of the time (vs 65% on easy). ![[rankdiag_rank_h1_h2.png]]

### 2. Why it never reaches 100%: the misses are *buried setups*, not unsolvable rooms

21 of 1018 rooms are missed by all 3 model copies (that's the 95.3% ceiling). **Every one is solvable** — and in 97% of them the solver is *inside* the search tree, just ranked past the 900-sim budget. They're almost all hard "needle" rooms (19 of 21 hard; typically only ~2 working setups).

Where does the true setup sit in the model's ranking? It degrades steadily with difficulty, and on the missed rooms it's **buried at rank ~38 of ~70** — never the #1 pick:

| rooms | true setup's median rank | mean | picks setup #1 | buried ≥5 | # working setups |
|---|---|---|---|---|---|
| easy | **0** | 3.0 | 57% | 21% | 13 |
| medium | 3 | 8.2 | 33% | 39% | 4 |
| hard | 4 | 11.3 | 20% | 49% | 1 |
| **the 21 misses** | **38** | 39.5 | **0%** | **90%** | 2 |

The score gap between the model's top pick and the true setup is a tiny **0.016** — it simply can't *separate* the one good setup from ~38 useless pushes. ![[rankdiag_unsolved.png]]

### 3. Where the search wastes its time: chasing wrong setups

Most rooms solve almost instantly (**median 4 sims**), but a tail of hard rooms blows up into the hundreds:

| rooms | count | mean sims | sims on 1st pushes | sims on dives | winning setup's rank |
|---|---|---|---|---|---|
| fast half | 1499 | 2.3 | 1.1 | 1.2 | 0.1 |
| slow tail (top 10%) | 291 | **386** | 19.6 | **366.6** | **16.3** |

The slow rooms spend 95% of their sims *diving* — but **not** because diving is broken. They dive into ~20 **wrong** setups' follow-ons first, because the *right* setup is ranked ~16th. Proof it's the first push and not the dive: sim-count correlates **0.79** with the *first-push's* rank, but only **0.29** with the dive's rank. ![[rankdiag_simtail.png]]

### 4. Is the neural network the slow part? No.

Scoring (rendering the scene + the neural net) is only **3% of wall-clock time**; simulation is ~95%. A faster or batched net buys ~0.5%. So "make suggestions fast" is **already solved** — the lever is doing *fewer sims* (better ranking), not a faster net.

(The one place the model loses on wall-clock — easy rooms, **7.05 s vs random's 6.33 s** — is because its individual sims cost a bit more, not the net. On trivial rooms, ranking can't save enough sims to pay for itself.) ![[rankdiag_nncost.png]]

### 5. The mechanism: the model is blind to setups

Not a two-heads problem (single head) and not the sigmoid (order-preserving) — both ruled out above. The real reason: **the model predicts near-immediate solvability, so a setup — which opens nothing yet — scores barely above a useless push.** Setup rank degrades easy → medium → hard → missed (**0 → 3 → 4 → 38**).

We checked the obvious alternative fixes; they don't help:
- **Telling the model how much budget is left (Hz): no help** — its setup ranking matches NoHz (#1 49% vs 51%), and it uses *more* sims.
- **A "dive bonus": would make it worse** — it would dive into wrong setups even more.

![[rankdiag_mechanism.png]]

---

## The evidence: scoring the model as a detector

To ground everything above in hard numbers, we treated the model's score as a **detector** and graded it against the true answer (`pure2push` is exhaustively solved). Two detectors: *"is this first push a valid setup?"* and *"does this second push open the goal?"*

**How we got the true labels** (no guesswork):
- **Setup label** = the exhaustive "valid first push" set the test already grades on. *Measured caveat:* that set under-counts setups (simulating from "wrong" first pushes finds an opener 23% of the time), so setup **precision** against this label is a **lower bound** (recall is clean). We also report a fully-simulated clean-label view alongside.
- **Opener label** = simulated directly — push the object again from a real setup's state and check whether the goal opens (the collection's own criterion). Physics only, seed-independent.

Data: 68,345 first-push candidates over all 1018 rooms; 49,540 second-push candidates over a 180-room sample; 3 model seeds.

### Evidence ledger — each claim → the number that grounds it

| Claim | Grounding number (this pass) | Status |
|---|---|---|
| Setup ranking is THE bottleneck | setup **p@1 = 0.32** overall / **0.19 hard** (top pick is a valid setup only 19–32%); catches only 34% in its top 5; AUC 0.80 → **0.75 hard** | ✔ but it's a **top-of-list** failure, not zero recall |
| The finish/dive recognizes winners | opener **AUC 0.87**, recall 0.78, **p@1 = 0.62** (vs setup 0.32) — the finish is the stronger detector | ✔ confirms §1 |
| It can't reject dead ends | opener **precision 0.37** (0.29 hard); per wrong subtree ~9–16 dead pushes outscore the good setup | ✔ measured |
| "No signal at all" | AUC 0.80/0.87 = real signal, but the good-vs-bad score overlap is heavy (13%) | ↺ signal exists, **top-of-list ordering is fragile** |
| "~18 dead dives per wrong subtree" | **8.6** dead pushes score above the opener threshold; **15.7** score above the correct setup | ✔ real number ≈ 9–16 |
| Diving is 94% of sims; NN cost negligible | unchanged (§3 dives = 94%, §4 NN = 3% of wall) | ✔ (from §3/§4) |
| step_penalty shares the bottleneck | unchanged (see Discussion) | ✔ |

*(Column key for the detector tables below: **base** = base rate; **op** = the threshold the search effectively uses; all others are defined in the Plain-language key.)*

### Detector A — the setup: "is this first push a valid setup?"

| difficulty | n | base | AUC | prec@op¹ | recall@op | FPR@op | **p@1** | catch top-3 | catch top-5 |
|---|---|---|---|---|---|---|---|---|---|
| overall | 68345 | 0.090 | **0.805** | 0.204 | 0.789 | 0.304 | **0.324** | 0.229 | 0.336 |
| easy | 18840 | 0.191 | 0.799 | 0.374 | 0.799 | 0.316 | 0.483 | 0.114 | 0.191 |
| medium | 26210 | 0.076 | 0.784 | 0.163 | 0.777 | 0.326 | 0.357 | 0.230 | 0.353 |
| **hard** | 23295 | **0.023** | **0.752** | **0.046** | 0.783 | 0.392 | **0.187** | 0.301 | 0.409 |

¹ against the under-counting label → a **lower bound**. On clean simulated labels (720 pushes), true setup precision is **0.65 overall / 0.53 hard** (not 0.05–0.20), and the AUC agrees (~0.75–0.80): the separation is **moderate, not sharp**.

**Read:** the model doesn't fail at a *loose* threshold — its recall is a healthy **0.79**. It fails at the **top of the list**: its #1 pick is a valid setup only **19% of the time on hard**, where the base rate is just **2.3%** (≈1 needle in 43). An AUC of 0.75 simply isn't sharp enough to float that lone needle to first place, so the search tries wrong pushes first. ![[clsdiag_roc_pr.png]] ![[clsdiag_qhist.png]]

### Detector B — the finish: "does this second push open the goal?"

| difficulty | n | base | AUC | **prec@op** | recall@op | **FPR@op** | p@1 | catch top-5 |
|---|---|---|---|---|---|---|---|---|
| overall | 12870 | 0.125 | **0.867** | **0.373** | 0.780 | **0.190** | 0.619 | 0.428 |
| easy | 4260 | 0.165 | 0.861 | 0.529 | 0.708 | 0.130 | 0.701 | 0.368 |
| medium | 3810 | 0.124 | 0.885 | 0.360 | 0.840 | 0.213 | 0.620 | 0.471 |
| **hard** | 4800 | 0.089 | **0.866** | **0.290** | 0.786 | 0.192 | 0.534 | 0.448 |

**Read:** the finish is the *stronger* detector (AUC **0.87 > 0.80**, confirming §1) and catches winners well (recall 0.78). Its weak spot is **precision**: **0.37 overall, 0.29 on hard** — at its operating point about **two-thirds of the pushes it flags as openers are actually dead**. That's the "can't reject a dead end" problem, measured.

### Detector B, cost side — dead-end false alarms

A *truly dead* wrong subtree = a first push that isn't a setup, whose follow-on pushes open the goal **zero** times (416 such subtrees, ~67 candidate pushes each). How many of those dead pushes does the model score high?

| difficulty | # dead subtrees | subtree size | dead pushes above the opener threshold | dead pushes scored above the correct setup |
|---|---|---|---|---|
| overall | 416 | 67 | **8.6** (median 4, max 89) | **15.7** (median 10, max 109) |
| easy | 115 | 71 | 8.9 | 13.7 |
| medium | 140 | 65 | 7.2 | 16.7 |
| **hard** | 161 | 67 | 6.9 | 16.2 |

So in each wrong subtree the model deems **~9 pushes "opener-grade"** and **~16 better than the correct setup** — all dead. That's the measured cost: the model **can't tell a dead subtree is dead**, so it grinds through ~9–16 false alarms before backing out. ![[clsdiag_deadend.png]]

### How much signal is there at all?

The one-number separation is real: setup **AUC 0.80** (0.75 hard), finish **AUC 0.87** (0.87 hard) — so it's **not** "no signal." But the scores of good vs bad setups **overlap heavily**: 13% of useless pushes score at or above the median good setup, and the top-pick-vs-setup gap is a razor-thin **0.016**. So there *is* signal — it's just not sharp enough to survive at the **top of the list on hard, sparse rooms**, which is exactly where best-first search lives. ![[clsdiag_summary.png]]

---

## What to optimize (the verdict, on the numbers)

Three candidate explanations, resolved:

- **"No signal" — rejected.** Setup AUC 0.80, finish AUC 0.87. The model is not signal-starved.
- **Setup ranking, top of the list — the PRIMARY problem.** Its #1 pick is a valid setup only 0.32 overall / **0.19 on hard**; catches only 34% in its top 5. This gates the **ceiling** (misses are buried setups) *and* the **slow tail** (a buried setup means many wrong dives). The number that makes it primary: sim-cost correlates **0.79** with the first-push rank, only 0.29 with the dive rank.
- **Dive precision / rejecting dead ends — the SECONDARY problem.** The finish detector's precision is 0.37 (0.29 hard), and ~9–16 dead pushes per wrong subtree outrank the good setup. This sets the *cost* of each wrong dive, compounding the setup burial into the heavy tail.

**⇒ Fix the setup first: train the model to score a push by whether it *leads to a solution* (a "value" target), not by whether it opens the goal right now** — so a setup that opens nothing yet still scores high (lifting its top-of-list rate on hard). **Then** add a "this subtree looks dead" signal to the finish head, so a wrong subtree is abandoned in ~2 tries instead of ~9–16. Explicitly **not** horizon-conditioning (Hz, verified no help) and **not** a dive bonus (it makes the dead-end problem worse).

**Verified here:** all the detector numbers above. **Still a hypothesis:** that a value/multi-horizon target actually raises the setup's top-of-list rate, and that a precision signal helps the dive without hurting its recall — that's the cheap `V(s1)` test in *Next*.

## Next

Test the setup-value idea cheaply, before any retraining: re-run best-first but rank the first pushes by a 1-step lookahead value `V(s1)` (simulate each first push once, then score how solvable the resulting state looks) on the 371 hard rooms + the 21 misses. Read whether the true setup's rank climbs toward the top and whether hard-room solve-rate lifts above 90.2%. Caveat baked in: this lookahead *costs sims* (it simulates every first push), so it's a **diagnostic** — if it ranks better, the deployable fix is a *trained* setup-value model that gets the same ranking without paying the lookahead. If the lookahead wins, it argues squarely for a setup-value target in the next model.

## Provenance

Scripts: `scripts/sandbox/rankdiag_analyze.py` (record stats), `rankdiag_rescore.py` (re-score the true-setup rank), `rankdiag_classifier.py` (the detector re-score + simulated labels, 3 seeds), `rankdiag_cls_agg.py` (metrics + `clsdiag_*` plots), `rankdiag_plots.py` (`rankdiag_*` plots). Data: `…/scratch_namo/eval/fullsearch/rankdiag/` (records) and `…/rankdiag/cls/cls_metrics.json` (detector metrics). Model scored raw; the sigmoid is order-preserving so it changes no ranking.

## Discussion
_(you ↔ Claude — ask here; newest at bottom.)_

**[Claude 2026-07-04] Does step_penalty share this bottleneck? YES — verified** (same method; the NoHz numbers reproduce exactly, so the comparison is sound). Its −1/0/1 reward did **not** touch the setup-ranking problem, which is why it was a wash on 2-push search.

| setup ranking (rank 0 = model's #1 pick) | picks #1 | mean rank | buried ≥5 |
|---|---|---|---|
| overall — NoHz / step_pen | 50.9 / 49.3 | 3.28 / 3.41 | 22.0 / 22.6 |
| **hard** — NoHz / step_pen | **39.1 / 34.8** | **4.68 / 5.14** | 30.6 / 32.9 |
| finish/dive, overall — NoHz / step_pen | 70.0 / 69.4 | 2.05 / 2.01 | — |

Robust misses (missed by all 3 seeds): NoHz **21**, step_penalty **26** (16 shared — a *superset* of NoHz's failures on hard, not a different set). So step_penalty's setup ranking is about the same overall and **worse on hard** (the tier that gates 100%); the finish stays the stronger detector for both. **Both models are blind to setups** — the fix has to make the model value a setup (leads-to-a-solution), not relabel the open-now signal.
