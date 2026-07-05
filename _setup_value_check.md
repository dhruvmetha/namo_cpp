---
type: experiment
status: done
created: 2026-07-05
updated: 2026-07-05
metric: "Scoring a first push by the finish it enables un-buries the true setup exactly on the hard/missed rooms (hard setup rank 5→3, miss rank 54→9; hard p@1 18.9→33.7) but is too soft to pin it #1 (top pick a real setup only ~33% on hard; junk gets spurious finish-scores). A PERFECT setup opens 98.1% of hard rooms (real search 90.2%) and 16/21 misses — setup-ranking is the dominant bottleneck; a ~2%-hard offline↔online dive gap is the rest. A setup's shape = a few good finishes (top-3≈max ≫ mean/count). ⇒ green-light a trained setup-value target on the top-few finishes; won't reach 100% alone (label/sim tail)."
tags:
  - experiment
  - diagnostic
---
# Setup-value check — can we un-bury the setup by scoring it by the finish it enables?

**Sibling to [[_ranker_bottleneck]].** That card showed the 2push search fails because the true SETUP push (opens nothing yet) is buried in the model's ranking. This card tests the cheapest fix idea — and, more importantly, is built to TEACH us what "setup value" actually looks like, not just give a yes/no.

**Plain idea.** A setup is defined by what it LEADS to (a finish). So score each first push by "after this push, how good are the finishing moves available?" The model is already good at spotting finishes, so we lean on its strength instead of its weakness. The honest doubt (from [[_1push_bottleneck]]): the model's finish-spotting also goes fuzzy on the rare hard rooms — exactly the ones we care about — so the cheap version may only help where we don't need it. This experiment is designed to expose that, not paper over it.

## Hypothesis [USER]

If we rank the first pushes by the finish they enable — instead of by whether they open the goal now — the true setup climbs toward the top and the hard/missed rooms solve. I want to SEE this happen (or not), understand WHY, and compare a "best single finish" score against an "aggregate finish" score so I can conclude what is really going on.

## Plan [CLAUDE]

Re-rank the first pushes for the **371 hard rooms + the 21 always-missed rooms** under four scoring rules, and measure (a) where the TRUE setup lands and (b) whether the room then solves. Everything on the exhaustively-solved pure2push set (n=1018), so we always know the true answer.

**The four ranking rules** (each scores a first push by looking at `s1` = the state right after that push):

- **q (baseline)** — the model's current direct score. We already know the setup sinks under this.
- **best-finish `V(s1)-max`** — the single highest finish-score among the second pushes available at `s1`. "Is there at least one great finish here?"
- **aggregate `V(s1)-agg`** — a summary of the finish-scores at `s1`, computed three ways: mean, top-3 mean, and count-above-the-opener-threshold. "How good are the finishes overall, not just the best one?"
- **GT-oracle `V(s1)-GT`** — does a TRUE finish actually exist at `s1` (ground truth from pure2push)? Perfect finish-detection — the upper bound the learned rules are chasing.

**What we measure, per rule, split by difficulty (easy/med/hard):**

1. **Where the true setup lands** — p@1 (is it the #1 pick?), median rank, fraction in the top 5. Under GT it is #1 by construction.
2. **The separation picture** — the score distribution for true setups vs junk first-pushes (histogram + AUC). Do the two piles separate, and by how much? This makes the mechanism visible — it directly answers "do we actually see high-for-setup / low-for-junk?"
3. **Downstream solve** — force the first push to the true setup, run the dive (second push) as normal, and check whether the room opens within budget. This isolates the key unknown: is setup-ranking the WHOLE bottleneck, or does the dive ALSO fail on these hard rooms?

**Pre-registered interpretation — what each outcome MEANS (write this before looking):**

- **GT-oracle solves (most of) the rooms** → setup-ranking IS the bottleneck; a good setup-value signal would fix it → green-light the retrain. **GT-oracle leaves rooms unsolved** → the dive also fails on the hardest rooms; setup-value alone won't reach 100%; the plan must widen beyond setups.
- **best-finish ≈ GT on easy/med but falls off on hard** → the cheap lookahead works where finishes are easy to spot and fails on the rare hard tail (finish buried there too) → the cheap mechanism is a partial fix; the learned retrain is needed for the tail. **best-finish ≈ GT everywhere** → the cheap lookahead is enough; maybe skip the retrain.
- **best-finish vs aggregate** → the SHAPE of a setup. If max wins and mean is diluted, a setup is "one clear finish" (so a false-positive can spoil max, and top-3/count should help). If aggregate wins, a setup is "several decent finishes." This directly informs how to build the setup-value training target.

**Data / cost.** Offline re-ranking reuses the existing classifier re-score data (child finish-scores at `s1`, from [[_ranker_bottleneck]]'s `cls/`) where it covers these rooms; a small CPU re-score fills the rest (simulate each first push once, score its children). The downstream-solve check is a small online best-first run on the hard+miss set. No training, no GPU. Watch the offline↔online label caveat from [[_1push_bottleneck]] (a stale-label room can look like a failure that isn't the ranker's fault) — flag any room where the offline "true setup" doesn't reproduce online.

**Deliverables:** the four-rule setup-rank table (by difficulty), the setup-vs-junk separation plots (per rule), the downstream-solve table (GT-oracle and best-finish), and a one-line verdict on each pre-registered question. Owned files: this card, `assets/setupval_*.png`, `scripts/sandbox/setup_value_check.py`.

## Run

- **Box / compute:** iLab SLURM `unlimited`, node **rlab1** `--exclusive` (96 cores, shared FS — no sync), 64 single-core shards by room. Pure re-analysis + small CPU re-score; **no GPU, no training.** (arrakis was 85% saturated; rlab1 was fully idle.) Job 171557.
- **Python:** `/common/users/dm1487/envs/mjxrl/bin/python`. `cv2.setNumThreads(0)` per process (one process = one core; else 64×render-threads → EAGAIN storm).
- **Model:** NoHz-v3, **3 ckpt-seeds**, CPU inference, `score_ctx(..., raw=True)` — identical scale + code path to `rankdiag_classifier`. First-push / second-push candidate pool = `pool()` (reachable edge × valid primitive × depth<5), restricted to the labeled `object_id` (same-object 2-push, matching how `valid_first_push` is defined).
- **GT:** positional join over `sorted(pure2push.json)` (basenames collide across shards). Setup label = `(edge,depth) ∈ valid_first_push ∪ valid_1push` (pure2push ⇒ `valid_1push` empty). Opener = **simulated** (`goal_open_pts`, the collection's own criterion; seed-independent physics). Difficulty = per-episode `division` in `pure2push_divisions.json`.
- **Opener threshold** (for the "count > op" aggregate): `op = 0.0925`, the Youden-J point on the reuse dive data's (child-q, is_opener) pairs (n=49540, 2776 openers, J=0.677).
- **Miss set:** positional all-3-NoHz-seed miss = **21 rooms** (19 hard) — reproduces the ranker card exactly; hard-asserted `object_id` match per seed.
- **Scope:** ranking re-score on **all 1018** rooms (full easy/med/hard coverage for the per-tier question); downstream-solve on **hard ∪ miss** (373 rooms), per the plan.
- **Reuse:** `cls/setup_shard*.jsonl` (q-baseline `is_setup` for all 1018 — cross-checked) and `cls/dive_shard*.jsonl` (the opener threshold + child-score cross-check). New compute = per first push: sim it once → score its children (3 seeds) → aggregate; downstream = force GT setups / best-finish#1 → dive.
- **Script:** `scripts/sandbox/setup_value_check.py` (owned). Data out: `…/scratch_namo/eval/fullsearch/rankdiag/setupval/`.

## Result

**Headline (plain).** Scoring a first push by "the best finish it enables" (the cheap 1-step peek) does un-bury the true setup on exactly the rooms where the model is blind — the hard and always-missed ones — pulling it from rank ~5 to ~3 on hard and from rank ~54 to ~9 on the misses. But the peek is not sharp enough to put the setup at #1 (its top pick is a real setup only ~33% on hard), because junk first pushes sometimes get a spuriously high finish-score. And the deeper answer is the important one: if we hand the search a PERFECT setup, it opens **98.1% of hard rooms** (vs the real search's 90.2%) and **16 of the 21 always-missed rooms** — so setup-ranking is the dominant bottleneck, but a small hard tail (~2% of hard, ~24% of the misses) fails for a different reason: the dive doesn't reproduce online even from a true setup (an offline↔online gap, the same class as [[_1push_bottleneck]]'s pos-953). A setup's shape is "a few good finishes" (top-3 ≈ max, both beat mean and count) — that's what the trained setup-value target should predict.

All numbers below: NoHz-v3, 3 seeds, on pure2push (n=1018). "Setup" = a first push GT says a 2-push solution runs through (`valid_first_push`); "junk" = every other first push. rank 0 = the rule's #1 pick.

### Table 1 — where the true setup lands under each rule (by difficulty)

p@1 = the rule's #1 pick is a valid setup; medRank = rank of the best-placed true setup (0 = #1); top5 = a valid setup is in the top 5. Higher p@1 / lower rank / higher top5 = better. GT-oracle is #1 by construction (it IS the true-setup label) — its point is the downstream solve below.

| tier | rule | p@1 % | medRank | meanRank | top5 % |
|---|---|---|---|---|---|
| **easy** (n=238) | q (baseline) | **51.7** | 0 | 4.74 | 74.8 |
| | best-finish max | 45.8 | 1 | 5.24 | 73.1 |
| | agg top-3 mean | 47.9 | 1 | 5.24 | 73.9 |
| | agg mean | 42.9 | 1 | 5.98 | 67.2 |
| | agg count>op | 42.4 | 1 | 5.24 | 73.1 |
| **medium** (n=409) | q (baseline) | 39.6 | 2 | 5.88 | 67.0 |
| | best-finish max | 37.2 | 1 | 6.10 | 69.4 |
| | agg top-3 mean | 36.9 | 2 | 6.28 | 69.2 |
| | agg count>op | 39.1 | 1 | 5.97 | 67.2 |
| **hard** (n=371) | q (baseline) | 18.9 | 5 | 12.34 | 47.4 |
| | best-finish max | 33.2 | 3 | 10.78 | 58.8 |
| | **agg top-3 mean** | **33.7** | **2** | 10.77 | **59.3** |
| | agg mean | 30.7 | 3 | 11.08 | 55.8 |
| | agg count>op | 29.9 | 4 | 12.79 | 52.3 |
| **the 21 misses** | q (baseline) | 0.0 | 54 | 46.76 | 9.5 |
| | best-finish max | **23.8** | **9** | 22.05 | **42.9** |
| | agg top-3 mean | 19.0 | 9 | 21.71 | 38.1 |
| | GT-oracle | 100.0 | 0 | 0.00 | 100.0 |

**Read.** The peek helps exactly where q fails and is neutral-to-slightly-worse where q already wins. On **easy**, q already floats setups (p@1 51.7) and the peek slightly hurts (45.8) — easy rooms have many finishes, so a junk push easily gets one high finish-score that spoils the top pick. On **hard**, q's top pick is a setup only 18.9% of the time; the peek nearly doubles that to ~33% and cuts the median setup rank 5→2-3. On the **misses** the effect is largest: q buries the setup at median rank 54 (never #1); the peek pulls it to rank 9 and top-5 42.9%. So the peek is a real, targeted lift on the hard tail — but ~33% p@1 is still far from the #1 we need. ![[setupval_setuprank.png]]

### Table 2 — do the setup and junk score-piles actually separate? (AUC, offline label)

AUC = how well the score separates setup from junk across all candidates (0.5 = coin-flip, 1.0 = perfect).

| rule | easy | medium | hard | all |
|---|---|---|---|---|
| q (baseline) | 0.805 | 0.799 | 0.762 | 0.816 |
| best-finish max | 0.694 | 0.758 | **0.764** | 0.768 |
| agg top-3 mean | 0.692 | 0.757 | 0.763 | 0.767 |
| agg mean | 0.689 | 0.754 | 0.756 | 0.762 |
| agg count>op | 0.631 | 0.715 | 0.729 | 0.724 |
| base rate (setup/all) | 0.191 | 0.076 | 0.023 | 0.090 |

(q reproduces the ranker card's Detector A — easy 0.799 / med 0.784 / hard 0.752 — validating the whole pipeline; my q re-score matches the reuse `setup_shard` q **bit-for-bit**, max|Δ|=0.)

**Read — the apparent paradox, resolved.** On hard the peek's global AUC (0.764) barely beats q (0.762), yet its top-of-list p@1 improves a lot (18.9→33). These measure different things: AUC is average separation over ALL candidates; p@1 is per-room "does the best setup reach the top." The peek gives a true setup a **distinctive high finish-score** (the green spike near 1.0 in the plot) that floats it above the low junk *in its own room* — improving the top of the list — while it ALSO hands high scores to some junk, keeping the average separation flat. **Important caveat:** the offline setup label under-counts setups by ~23% (a "junk" first push opens the goal 23% of the time when simulated). The peek's "false positives" are disproportionately these mislabeled-junk-that-are-really-setups (they have a real finish), so the offline-label AUC penalizes the peek *harder* than it penalizes q — 0.764 is a lower bound; the clean-label separation is better. ![[setupval_separation.png]]

### Table 3 — downstream solve: force the setup, run the dive, does the room open online?

The key test. "GT-oracle" forces each true setup and dives (second push best-first, sim until an opener); the room counts solved if any true setup's dive opens it online. "best-finish#1" forces the peek's single top pick and dives. Reference: the real online 2-push search solves **90.2%** of hard rooms (and never the 21 misses). Openers simulated exactly as the reuse dive data (100% agreement on 60 overlap rooms).

| tier | GT-oracle solve | offline↔online gap | best-finish#1 solve | bf#1 is a real setup | if bf#1 is a setup → solve |
|---|---|---|---|---|---|
| **hard** (n=371) | **98.1%** | 7 rooms | 72.0% | 33.2% | **97.6%** |
| **the 21 misses** | **76.2%** (16/21) | 5 rooms | 28.6% | 23.8% | 40.0% |

**Read.** Hand the search a perfect setup and it opens **98.1%** of hard rooms — up from the real 90.2%, closing ~80% of the gap to 100%. On the 21 always-missed rooms, a perfect setup rescues **16 of 21**. So the misses are overwhelmingly buried-setup failures, not unsolvable rooms — confirming the ranker card. best-finish#1 as a committed selector solves 72% of hard: its failures are almost entirely because its top pick isn't a true setup (only 33% are) — **when best-finish#1 IS a true setup, the dive solves 97.6%.** So the dive is not the problem when the setup is right; the problem is picking the setup.

**The offline↔online gap (the caveat, flagged).** A small set of rooms fail even with a perfect setup — the dive opens nothing online despite offline GT saying a 2-push solution exists. Hard: 7 rooms `[8, 23, 60, 345, 401, 869, 884]` (1.9%). Misses: 5 rooms `[8, 60, 345, 664, 869]` (4 also in the hard-gap list). This is the same class as [[_1push_bottleneck]]'s pos-953 stale-label / sim-determinism gap — the offline "true setup" doesn't reproduce under the current online car dynamics. It reconciles the ranker card's "97% of misses have the solver in the tree" (offline) with my 76% (online): the ~24% difference IS this gap. These rooms are a label/physics floor, not a ranking problem — no setup-value model can fix them.

### The three pre-registered questions — verdicts (plain English)

**1. Does un-burying the setup actually solve the rooms, or does the dive also fail?**
**It mostly solves them — un-burying is the dominant fix, but a small tail also has a broken dive.** A perfect setup opens 98.1% of hard rooms (real search: 90.2%) and 16 of the 21 misses. The dive is reliable when the setup is real (solves 97.6%). The remaining failures are a ~2%-of-hard offline↔online gap where the dive can't reproduce the opener online — setup-value alone won't reach 100%; that tail needs a label/sim fix, not a ranker fix.

**2. Is the cheap best-finish peek enough, or only on easy/medium (falls off on the hard tail)?**
**The peek is not enough as a committed pick anywhere — but, against the pre-registered worry, it helps MOST on hard, not least.** The honest doubt was that finish-spotting goes fuzzy on hard; it doesn't — best-finish's hard AUC (0.764) holds up and its hard p@1 nearly doubles q's. The peek's real weakness is at the very top of the list: junk first pushes get spurious high finish-scores (partly the 23% under-counted real setups), so its #1 pick is a true setup only ~33% on hard and its committed-solve is 72% vs the oracle's 98%. So the peek is a genuine partial lift concentrated on the hard tail, but a *trained* setup-value model is needed to actually pin the setup at #1 — the lookahead diagnostic argues squarely for that retrain.

**3. best-finish vs aggregate — what shape is a setup (one clear finish, or several)?**
**A few good finishes — not one, not many.** top-3 mean matches the single max on hard (p@1 33.7 vs 33.2, AUC 0.763 vs 0.764, median rank 2 vs 3), while plain mean (diluted by junk children) and count-above-threshold (the coarsest, worst AUC 0.729) both lag. If a setup were "one single clear finish," averaging the top 3 would dilute it and lose to max — it doesn't. If it were "many decent finishes," count would win — it's the worst. So a true setup opens up a small handful (~2-3) of good finishing moves. **Build the setup-value target on the top-few finishes** (as robust as the max but slightly better-ranked), not the single best (false-positive-fragile) and not the full count (junk-diluted).

### Provenance / sanity gates

- Script `scripts/sandbox/setup_value_check.py` (owned); data `…/eval/fullsearch/rankdiag/setupval/` (64 shards, 1018 rank rows, 373 downstream, 0 errors, 11 min on rlab1). Summary `setupval_summary.json`.
- **Gate 1:** q0 re-score = reuse `setup_shard` q bit-for-bit (max|Δ|=0.00000 over 9705 overlap first-pushes; 0 setup-label mismatches).
- **Gate 2:** q setup-AUC (0.805/0.799/0.762) reproduces the ranker card's Detector A (0.799/0.784/0.752); q setup-rank p@1 (easy 51.7 / hard 18.9 / miss 0.0) reproduces its §2 (57/20/0).
- **Gate 3:** online opener simulation agrees 100% with the reuse dive data on 60 overlap rooms; positional miss set = 21 (object_id-asserted), real hard solve = 90.2% (both match the ranker card exactly).

## Discussion
_(you ↔ Claude — ask here; newest at the bottom.)_
