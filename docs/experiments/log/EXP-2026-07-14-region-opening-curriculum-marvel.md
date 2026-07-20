---
status: active
thread: rl_loop
robot: car
updated: 2026-07-16
supersedes: EXP-2026-07-12-opener-curriculum-loop (buggy lineage — retracted, see below)
---

# EXP-2026-07-14 — Region-Opening curriculum, clean restart (Marvel lineage)

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a **ranker** that orders pushes so **search** solves region-opening cheaply (beat random, fewer sim calls, every tier). This card is the *reproducible framework* for building that ranker; it does not restate the problem.

## The one sentence

Learn the ranker via a **curriculum ladder** (1-push → 2-push → …), running **pure DAgger within each stage**, on a **clean, correctly-labeled, in-sync-generated** data pipeline — starting from a **balanced bootstrap seed**.

## Why this design (the motivation, rung d / d-1 / …)

**Why a curriculum at all: SEARCH COST.** Region-opening is tree search over pushes, and depth costs `branching^depth` (~300 pushes/node, so 2-push ≈ 300², 3-push ≈ 300³). Exhaustive deep search does not scale (the "no exhaustive ground truth at scale" constraint). The curriculum keeps us at **effective depth 1 at every rung**: the rung-(d-1) model collapses rung-d's search from `b^d` to ~`b·k` (rank the b first-pushes, verify each continuation with the previous rung's top-k instead of a full sub-search). Exponential-in-depth becomes ~linear-per-rung, paid one rung at a time. It breaks a real chicken-and-egg: you can't get deep value labels without deep search, and you can't afford deep search without the value function. This is Expert Iteration with the depth made explicit and controllable rather than left to emerge from self-play.

**It is ONE ranker, not one model per stage.** The model is a single value/ranker that orders the pushes available at whatever state it's in; "1-push vs 2-push" is only how deep the search expands using that same ranker. A "setup" and a "finish" are the same ranker queried at a parent vs a child node, not two networks. (Ant-Man / Beast are curriculum *stages*, not separate models.)

**Discounting is what lets one ranker order setups correctly.** Value of a push ≈ `γ^(pushes-to-open)`: a direct opener ≈ γ, a setup (opens in 2) ≈ γ², a push that never opens = 0. For any γ<1 this yields `direct > setup > dead` automatically, which does two needed things at once: (a) lifts setups above dead pushes, which *is* the fix for the old "setup bottleneck" (a myopic opens-now model scores a setup 0 and buries it), and (b) keeps direct openers preferred so search tries to open in one push first. It generalizes to any depth (`γ^d`), so **no horizon conditioning / no Q(s,a,H) is needed** — γ encodes depth-to-solution in one scalar. γ is the only real knob (it sets the setup-vs-open margin; the old hand-set target 0/0.9/1 is just γ≈0.9 for a one-step setup). Collecting rung-d labels = collecting the `γ^d` value targets for that depth.

**Labeling = cheap-verify + exhaustive-fallback, and recycle the fallbacks.** For a rung-d problem: rank candidate first-pushes (setups), simulate, and check the continuation with the rung-(d-1) model's top-k. This is sim-confirmed, so PRECISION is perfect and the only error mode is a *miss*. A first-push whose continuation shows up in top-k is a confirmed-good setup, cheaply. Only when top-k finds nothing do you fall back to exhaustive search, which resolves the ambiguity ("no continuation exists" vs "the model just missed it") into a TRUE label. That avoids the setup under-counting bug (41.8% of the old "never opens" labels in solvable scenes were actually real setups). The fallbacks are dual-purpose: "model missed but a solution exists" cases are hard examples that improve rung-(d-1)'s recall (DAgger at depth d-1), and the resolved labels train rung d with true, not model-guessed, targets.

**Why per-rung quality matters (the real reason to "cap" a rung before moving up).** The rung-(d-1) model's recall@k directly sets rung-d's cost: every miss drops to the exhaustive fallback, which is exactly the `b^d` cost the curriculum exists to avoid. So making a rung good before climbing is not perfectionism, it is what keeps the next rung tractable and its labels clean. The right cap criterion is therefore **recall@k on real post-first-push states** (is the continuation reliably inside the top-k), not the deploy-time @1 headline.

## Why we restarted (the old lineage is dead)

model_0/1/2/3 are **discarded**. Three bugs, all found 2026-07-14, invalidated the whole EXP-2026-07-12 lineage:
1. **Labels measured the wrong thing.** `region_opening.target_goal_region` defaults False and was never set, so every label counted opening *any neighbour region*, not *the task goal* — verified by a faithful re-run (99.9% of "budget-hit" scenes are genuine 2-push) + a 15/15 replay. All prior training data is neighbour-opening.
2. **Generator out of sync with labeler.** `--require-adjacent` filters on an idealized static-adjacency test ("would two regions connect if this one object *vanished*") computed on obstacles-only geometry, while the labeler uses a real wavefront BFS with the object *pushed* — so ~22% of generated scenes are multi-hop and get dropped. The safety net `_runtime_validate_adjacency` was dead code (`return True`).
3. **Composition starvation.** The `75% hard` subsample + neighbour-label skew left model_3 with ~4% easy → regressed.

**Prerequisite gate for THIS card:** the generator fix (re-enable `_runtime_validate_adjacency` to use the labeler's own `get_region_snapshot` with robot/goal placed) must be **verified in-sync** (generate-with-fix → labeler drops ~0) before any data is generated.

## The fixed, reproducible pipeline

- **Generator:** `mujoco_env_creator/generate_envs.py`, `--require-adjacent` + the re-enabled per-placement re-check → gen "1-hop" == the labeler's real wavefront. One robot region, one goal region, one blocking object.
- **Labels:** goal-opening only — `region_opening_rung1_exhaustive_car.yaml` **+ `--target-goal-region`** (never the default). Exhaustive 1-push over reachable pushes → the 60×5 opener grid.
- **Difficulty — FIXED cuts, never tertiles** (`sr = openers/reachable`): **hard `sr<0.05`, medium `0.05–0.30`, easy `sr≥0.30`**.

## Lineage & naming (Marvel, alphabetical; stage = character, `-N` = iteration)

- **1-push stage → ANT-MAN:** `antman-0` (balanced seed) → `antman-1`, `antman-2`, … (DAgger rounds)
- **2-push stage → BEAST:** `beast-0` → `beast-1`, …
- (Cyclops, Daredevil, … for later stages.)

## The algorithm

All generation is **60% aug9_car_v3 / 40% feb_car** [USER — aug9 is denser/more interesting; worth its higher retry cost].

```
BOOTSTRAP  antman-0:
  generate a fresh batch (fixed gen, 60/40 aug9/feb) → label (goal-opening)
  → build a 50k-episode seed (train+val, split BY ROOM ~90/10), difficulty-LEANED, hard RELAXED:
       easy ~25k / med ~20k / hard ~all-available (~2-5k).
     [see PILOT FINDING: hard is the RARE bin, not easy — 20k hard would cost ~1M scenes/~4h]
  → train antman-0.

PURE DAgger  antman-r (r = 1,2,…):
  1. GENERATE  fresh scenes (fixed gen, 60/40 aug9/feb)
  2. SCREEN    best-first with antman_{r-1}:
       solved & opener in top-5  → DROP (model nails it)
       solved & opener NOT top-5 → KEEP (the model's 1-push mistake — THE signal)
       UNSOLVED (no 1-push opener) → 2-push → phase2_bank/ (Beast stage), NOT this stage
  3. LABEL     exhaustive goal-opening on the KEPT scenes; add up to 50k episodes this loop
  4. TRAIN     accumulate ALL clean SOLVABLE data + retrain antman_r
       (NO dead in training [USER — no negatives]; ALL dead → phase2_bank for Beast — it IS the 2-push material)
  5. EVAL      solve@k by fixed-cut difficulty vs random
  6. LOOP/STOP  improved → r+1 ; keep-yield→0 / plateau → stage done

LADDER: 1-push loop to plateau → start the 2-push stage (beast-*) on the banked post-setup scenes, same loop.
```

## Beast (2-push) — concrete plan [2026-07-16, from the design discussion]

**ONE ranker R, two query points.** `R(board, push) → γ^(pushes-to-open)`: opener/finish ≈ 1, setup ≈ 0.9, dead = 0. "antman-5" = R after the 1-push stage; "beast-r" = the SAME R after folding in 2-push labels — not a separate network. FINISH = R queried at a post-setup board ("which push opens now?"); setup-ranking = R at the root. One evolving R, used at parent vs child node.

**`LABEL(scene)` — exhaustive setups, model-ORDERED finish + exhaustive fallback.** Enumerate ALL reachable first-pushes (no beam, no sampling). Direct opener → 1. Else run FINISH on the post-setup state: walk EVERY candidate finish **in R's ranked order**, stop at the first that opens G. Opens within top-k → cheap; opens only beyond top-k → still found, but log a recall-miss (hard example that sharpens R); the full ordered sweep opens nothing → TRUE dead. Setup-with-a-finish → 0.9; exhaustively-no-finish → 0. **R never declares dead — only the exhausted sweep does; R only ever saves sims** (the fallback follows R's order too, so even it tries the likely finishes first). Cost ≈ b·k when recall@k is high, → b² as it drops.

**beast-0 bootstrap = RELABEL the 178,364 solvable antman scenes** (NOT "collect fresh dead scenes first"). The 1-push `0`s are unreliable — **41.8% are real setups** — so rerun LABEL over the 178k: openers stay 1, setups flip 0 → 0.9, true dead-ends stay 0. This both fixes the reuse-contradiction (one value scale, no contradictory 0-vs-0.9) AND seeds beast-0. It IS a full depth-2 collection (a sim per push; the finish-tries dominate, so cached states wouldn't save much). Off-distribution worry is a NON-issue: opener(1) outranks setup(0.9), so R uses the opener wherever one exists and a setup only on dead scenes — the "wasteful" setup is simply never selected (card's own point: discounting keeps direct openers preferred).

**[2026-07-17 SCOPE — USER]: round-0 relabels a 40k REPRESENTATIVE SUBSET of the 178k, not the full 178k.** 178k is too slow for a first pass (~4.5× wall-time); 40k gets beast-0 trained fast so we can answer the deploy question (beast-0 vs antman-5) before investing more. The 40k spans aug9 set1/set2 (all benchmarks) + feb_car templates — diverse, not benchmark-skewed. **TWO PHASES [USER 2026-07-17]: (A) train beast-0 on the RELABELED SOLVABLE scenes — 40k FIRST (178k too slow up front), extend toward the full 178k as the training bootstrap; THEN (B) DAgger from the 1push-DEAD envs** (line 85): screen dead scenes with beast_{r-1} → keep its 2-push mistakes → exhaustively LABEL → retrain beast_r. So the relabel (A) is the solvable-scene bootstrap (40k→178k); DAgger (B) is the subsequent data-collection growth from the dead bank. Round-0 deploy eval (beast-0 vs antman-5) gates B. Manifest: `/scratch/dm1487/curriculum2/beast/round0_manifest.txt` (40,000 episode-XMLs). Run: Amarel job `58255581` (160 tasks, pinned), ETA ~15:30.

**Then DAgger on the dead bank.** Sources: ~1.04M xml-only screen-dead leads (`phase2_bank/screen_dead_scenes.txt`) + 30,052 labeled-dead (`phase2_bank/labeled_dead_r*.h5`, full identity, exhaustively 1push-dead). Screen fresh dead scenes with beast_{r-1}: solved-in-2 & winning setup in top-k → DROP; beast's 2-push mistake → KEEP → LABEL (exhaustive) → accumulate → retrain beast_r. Unsolvable-in-2 → bank for 3-push (Cyclops).

**Cap = recall@k on post-setup states** (R reliably puts the finish in its top-k) AND keep-yield → 0 → climb to 3-push, same R one level deeper. **Two feedback loops per round, one R:** (A) finish recall-misses sharpen R's finish-finding (fewer fallbacks → cheaper labeling); (B) newly labeled setup-mistakes sharpen R's setup-valuing. Both are examples into the same network; γ ties them onto one scale.

### beast-0 ROUND-0 experiment spec [2026-07-16, USER]

**Round 0 ONLY** — relabel the 178k → train → eval; **NO DAgger rounds yet.** Reason: whether to *deploy* beast-0 vs antman-5 (always beast-0? route by depth / by "is there a 1-push opener"?) is an open question the eval answers *before* we invest in the ladder.

**γ-sweep at BUILD time.** The collection is **γ-agnostic** — the tree records depth-to-open per push; γ is applied only in `build_rung2_h5 --gamma`. So it's ONE expensive collection → N cheap builds → N trains. Sweep **γ ∈ {0.3, 0.5, 0.7, 0.9}** (low γ ≈ "setup buried near dead", high γ ≈ "setup ≈ opener"), named **beast-0-g{γ}**, **1 seed each** (add seeds only if one γ clears eval/seed jitter ~±1-1.5). Flat sweep ⇒ γ doesn't matter, lock 0.9.

**Eval = best-first, f = model score only** (`combine=q`, no cost term) on **BOTH** `namo_testset_v1` (1-push) AND `pure2push` (2-push, n=1018), vs **antman-5 / NoHz-v3 / random**. Two reads: (1) did beast-0 keep antman-5's 1-push opener skill? (2) did it gain 2-push setup skill? — antman-5 **fails 2-push best-first by construction** (buries setups at ~0), so it's on the 2-push table on purpose as the "you need this stage" line.

**Storage (non-overwriting):** everything under `curriculum2/beast/round0/` (`collect/` h5/ models/ eval/); antman data (178k h5, antman-5 ckpt, dagger_orchestrator) is **read-only input**. **Code:** `region_label_mode` in region_opening.py (exhaustive setups + early-stop finish + score/rank log + cost-prune disabled), config `region_opening_beast_relabel_car.yaml`, `build_rung2_h5 --gamma`. Scene XMLs (the 178k) live on **Amarel** → collection runs there.

### beast-0 round-0 execution retro — lessons [2026-07-17]

Round-0 collect→build→train→eval ran end-to-end but execution was bumpy (an evening of firefighting). Root causes + durable fixes, so the next round is smooth:

**H→E→V: the data format was too heavy (the #1 root cause).** H: a rendered 40KB ctx crop per tree-node is fine. E: 40k scenes → 2.47M rows → **107GB**, which OOM'd the merge (32G), was slow to move Amarel→CS, blew /scratch quota, and made A100 training slow (~12 steps/s). V: uncompressed REJECTED — `build_rung2_h5` now lzf-compresses ctx (`fa1e8cc`, **MEASURED 18× → ~6GB**), fixing the OOM/transfer/quota (a SIZE win). **SEPARATE measured finding (don't conflate): the slow training is CO-BOTTLENECKED** — NFS round-trip reads (~2980 rows/s), GPU compute (~2828), and observed (~3072) all converge within 10%, so compression does NOT speed it. Real speed levers = **node-local /dev/shm staging (4.3×) + `torch.compile` (1.78×), together**; and all 4 γ overfit by epoch 3-4 → early-stop (fewer epochs = faster). See memory `reference_training_speedup`. Deeper lever: distinct crop per node vs share/render-on-the-fly.

**Smoke every stage on the TARGET box.** The 4-train full launch failed 3× — python-PATH (`env.ilab.sh` doesn't conda-activate → bare `python` absent in sbatch), then OpenBLAS thread-limit (CS `ulimit -u=2000`, 64 thr/proc) — all catchable by a 2-min 1-epoch smoke. Fixed: `scaled-run` skill (pre-flight checklist) + `scripts/slurm/train.slurm` (all fixes baked in).

**Estimates were ~3× optimistic** (collection 1.25h→3h; walltimes too short → 26 build shards + a merge job killed). Fixed: calibrate from the smoke; walltime = 2-3× measured, never omit (CS default is 2min) or partition-MAX (blocks GPU backfill) — memory `feedback_no_slurm_time_limits`.

**Strategic — pilot small for exploratory rounds.** Round-0 only needed to answer "does beast-0 beat antman-5 on 2-push?" yet ran a full 40k→107GB→4γ pipeline (~6h) for a first look. Next exploratory round: few-k-scene pilot, one γ, fewer epochs → signal in ~1h, then scale only if it pays.

### beast-0a — censored labels, root boards only [2026-07-18, USER-driven redesign]

**The diagnosis behind it (from round-0's post-mortem):** labels are depth-conditioned censored observations, not values. A depth-k search either finds an opening (d exact → V=γ^(d-1)) or proves only d>k (→ **ceiling V ≤ γ^k**); "truly dead" is never observable. Round-0 wrote ceilings as hard 0s → 91% of the loss was false zeros + a 50× post-setup-state flood → the 1-push regression. Three research sweeps (survival-analysis lit, gpt-5.6-sol second opinion, combinatorial-search lit) converged: censored NLL for ceilings; the field never exhaustively proves negatives; ordering beats calibration (Chrestien 2023 — the repo's own "order, not calibration" is a theorem). Reading list: queue.md "Labels from search" section.

**beast-0a experiment:** ONE model, root/start-state boards ONLY (no post-setup rows at all — bets that "does this push open from this board" generalizes to post-setup boards). Data = antman 1-push set (178,364 rows; 0s → **ceiling 0.9**) ∪ beast-0 root rows (46,314 unique; setups → **exact 0.9**, exhausted deads → **ceiling 0.81**), dedup key (xml, object_id), beast wins overlap (39,373) → ~183.8k boards, `beast0a_train.h5` (lzf). Build: `scripts/pipeline/build_beast0a_h5.py`. Loss: exact cells → HL-Gauss CE (+ y=vmax one-hot endpoint fix); ceiling cells → censored NLL `-log P(V≤c)` (fractional-bin cut, group-mean, `NAMO_CENS_WEIGHT`=1): `sage_ext/hl_gauss_censored.py`, wired via `ceiling_mask` (commit 57e3581, unit-tested incl. monotone tightening). Train: rankaux recipe, γ=0.9 labels only (no sweep), ~5 epochs/early-stop (round-0 overfit by ep3-4). Eval: both axes vs the brackets antman-5 (1-push recovery = must-pass) / beast-0 (2-push hold) / random.

**Readout:** (1) 1-push recovers AND 2-push ≥ beast-0 ⇒ the 2.4M-row flood was waste; future collection = root labels only (~50× less data, no b² dead-proofs — unresolved scenes just wait for the next round). (2) 1-push recovers, 2-push → antman-5 level ⇒ add back a *sampled* slice of post-setup exact openers only. (3) 1-push doesn't recover ⇒ loss/weighting bug — short suspect list. Deferred (stackable later): B-vs-C mask-ablation, certain-pairs ranking aux, ranking-primary arm (Chrestien/LevinTS), hindsight mining of failed sweeps for other-pair openers.

### Round-1 (2026-07-19 overnight): full-rich coverage + the ceiling A/B — CHAMPION beast-1-c081

**Pipeline (all pilot-gated):** sweep-ranker pilot (200 scenes × 3 rankers + n=2.41M counterfactual from round-0's logged ranks) picked **antman-5c @ k=15** (97.7% retention, ~70% cost cut; 99.86% cross-ranker agreement on solvable/dead — the ranker changes cost, never answers). k-cap knob `region_label_topk` (243c6c7). Collection: 124,568/124,568 scenes, zero timeouts, ~2.5h at 320×12 pinned. Build: per-run-dir globs (recursive `**` glob × 128 workers = NFS livelock — TotalCPU=0 is the tell). Merge: dual variants for the fail-ceiling A/B [USER's Bayesian point: post-k15-failure ⇒ ~96% dead; strict honesty says ≤0.9, posterior says ≤0.81 at ~4%×0.09 label optimism].

**Results (count-asserted; 1push e/m/h/all@1 | 2push solve/avg/@30 | hardh2 s2s):** beast-0c 98.1/85.3/45.6/85.9 | 93.7/117/65.9 | 10.3 · beast1_c09-strict 97.3/83.8/38.7/84.0 | 94.5/104/67.6 | 11.3 · **beast1_c081-posterior 97.9/86.9/48.5/86.8 | 95.1/93/69.4 | 8.5**.

**Verdicts:** (1) **posterior ceilings WIN** (+2.8 all@1, +9.8 hard@1 on identical data); strict lost to beast-0c despite 4× rich data → loose ceilings squander data; recipe rule = tightest ceiling the posterior supports. (2) **beast-1-c081 = best model of the project on every axis** (vs antman-5: all@1 83.5→86.8, hard@1 40.7→48.5, 2push 154→93 avg sims). (3) Density ablation (same night): labels-per-board is load-bearing (full/30/15/8 → hard@1 45.6/40.2/35.3/25.5) — don't sample setups. (4) Hard-1push tier is a depth-1 artifact: random@depth-2 solves 98.5% (177/201 via 2-push plans); models beat random only under ~30-sim budgets → tight-budget ordering is the battleground.

**The label grammar (the week's core artifact — "forward-ness"):** every cell is a forecast of the future search compressed into one reactive scalar: **1** = opens now (sim-verified, always) · **exact 0.9** = pure setup, a verified finish exists after it (sim-verified, always) · **≤0.9** = won't open in one, beyond that unknown (one sim) · **≤0.81** = even two pushes don't get there (proven for round-0's exhaustive cells; ~96%-posterior for round-1's k15 cells — writing the posterior bound anyway is what the c081 A/B validated). Positives are never uncertain; uncertainty lives only on the negative side, which is why aggressive negative bounds are cheap (worst case: a 0.09-understated setup). Scales by construction: rung-3 adds exact 0.729 + tightens ceilings a notch.

**2-push @2 = "perfect play"** (pure2push needs ≥2 pushes, so @2 means first-choice setup then first-choice finish, zero waste): random 3.1 / antman-5c 26.3 / beast-0c 27.0 / **beast-1-c081 32.4**. Sharpest discriminator found — antman-5c ≈ beast-0c here (25%-rich never improved perfect play; its gains were mid-budget search efficiency), only 100%-rich moved it. Recommend @2 as a standing registry column. antman-5c 2-push finalized at n=1018: 91.5%/138.1 avg/@30 61.4.

**Bonus-episode ledger (VERIFIED 2026-07-19, agent-audited to the row):** beast-1's 191,703 = **46,314 old-rich + 121,750 round-1-recovered + 23,639 extras** (reconciles exactly). Corrections to earlier in-chat quotes: extras are **23,639 = 12.3%** of the set (not ~7-13k/~7%), and they are **HARDER, not easier** (opener-rate 19.0% vs 99.9% on recovered boards; more ceilings, fewer setups) — they're additional blocking-object episodes for the same task-goal adjacency, carrying no survivorship bias. **Leakage = 0** (exact-path ∩ both test keys = 0/0/0/0; basename-stem collisions DO occur (41-377) but are naming-collision artifacts — generation batches reuse index ranges; content-diff proved different rooms. ⚠ never key rooms by basename). Legitimacy: 20/20 sampled extras have sane labels; 11/11 locatable xmls parse with genuinely-movable objects (generator invariant covers the rest). **New open item:** 17,461 of A's target episodes were NOT recovered by round-1 (likely robot-already-at-goal / goal-not-in-snapshot skip paths — beast-1 never trained on them; worth a later look). Net: extras safe, distribution slightly toughened, champion's result stands strengthened.

**Open queue:** M1 (50/50 root/finish boards) + M2 (all roots + 1-2 finish boards/scene, sampled from round-0's dense 2.41M + round-1's k15 depth2 rows — on disk, no sims) → dense-finish masking gate → only then decide exhaustive finish recollection (~28k wh). Confirm-seed of beast1_c081. Registry/RESULTS rows (include @2).

### Post-round-1 diagnostics (2026-07-20 overnight) — model forensics + the round-2 data recipe

Three studies, all reusing round-1 artifacts (no new collection). Champion checkpoint = `round1/models/beast1_c081/checkpoints/epoch015-val_loss1.7523.ckpt`. Single seed throughout (as the champion is) — the recurring caveat below.

**1. What has beast-1-c081 learned? (report: `round1/analysis/beast1_c081/report.md` + 8 plots).**
Score-forensics over the test sets, scores in the raw grammar space [0,1] (deployed sigmoid squash is monotone → rankings invariant).
- **Openers mastered, setups NOT.** True openers score median **0.973** (target 1.0; AUC 0.894 vs dead); verified setups score median **0.583** (target 0.9) — collapsed toward the dead pile (setup-vs-dead AUC only 0.721). This single representational gap is the root cause of buried setups (first-setup rank median 3/6/14 easy/med/hard) and the 32.4 perfect-play ceiling. **The #1 actionable defect for round-2.**
- **Forward-ness is REAL** (not just ordering): a board's MAX score separates 1push-solvable from 2push-only at **AUC 0.892**; board-max ≥0.95 flags "opener exists" at FPR 0.046. One-sided/tier-dependent (hard-1push AUC 0.770 — hard openers score low enough to overlap 2push-only maxes).
- **Hard-1push misses = "bad pushes scored high," not "opener scored low":** on every missed board 100% of the pushes out-ranking the true opener are proven non-openers; the opener still scores ~0.893. The ranker's weakness is discrimination AMONG high-scoring candidates.
- **vs antman-5c: net +37 outright 2push solves (34 hard)** — the clearest capability gain. 1push flips are churny (net +22 hard, but 134 boards flip either way). Shape bias: gains on square objects (aspect 1.21), loses on elongated (1.44).
- **Unreachable floor clean:** median 0.0098 over 526k unreachable cells, 0.9% leak >0.5.

**2. k-policy for round-2 finish sweeps (same report, §E) — TWO corrections + a recipe.**
- **The finish-layer live/dead base rate is 74/26, NOT the 54/46 the round-1 note claimed** (measured on 1.86M dense depth-2 finish boards). ⚠ caveat: that population is Q1-*expanded* nodes (promise-biased) and train-lineage/in-distribution — the round-2 exhaustive-root pool will be more dead-heavy.
- **At the 74/26 prior, NO sweep depth k reaches 95% posterior-dead** on the (conservative) offline numbers — declaring "dead" needs finish-recall ≥ **0.982**, and the champion's measured recall is a strict LOWER bound (every live finish board on disk was early-stopped at ONE known finisher; zero exhaustively-swept live boards exist → the LB gap can't be closed offline). LB recall@k: 0.51@5, 0.77@15, 0.90@30.
- **Recommended round-2 finish policy:** sweep **k≈20 with a ≥0.95-score early-exit** (catches ~71% of live top-1s in 1-few sims; dead false-alarm 5.8%), write a **~0.85 ceiling on a failed sweep (not 0.81)**, and run a **1–2k exhaustive-finish CALIBRATION batch** early in round-2 to measure TRUE recall, then tighten toward 0.81 only if it confirms recall≥0.98. **TENSION worth noting:** c081's 0.81 posterior ceilings empirically WON the round-1 A/B (+9.8 hard@1) — so the recipe worked even though the posterior justification is now shakier than believed; the win may owe more to "aggressive negative bounds are cheap" than to a calibrated 96%-dead claim.

**3. Finish-board gates M1/M2/masking (report: `round1/mix_arms/REPORT.md`) — round-2 data recipe settled.**
All arms built from on-disk labels (zero sims), champion recipe, count-asserted 1323/1018/204.

| model | 1p hard@1 | 1p all@1 | 2p solve | 2p avg | **2p @2** | 2p @30 | hardh2 s2s |
|---|---|---|---|---|---|---|---|
| champion beast1_c081 | 48.5 | 86.8 | 95.1 | 93.0 | 32.4 | 69.4 | 8.5 |
| **M2-k15** (roots + k15 finish) | 43.6 | 86.1 | 95.0 | 89.4 | **34.7** | **72.4** | 9.8 |
| M2-dense (roots + dense finish) | 43.1 | 85.0 | 95.8 | 89.1 | 29.4 | 72.2 | 8.5 |
| M1 (50/50 root/finish swap) | 35.8 | 82.7 | 94.2 | 111.7 | 27.4 | 65.3 | 14.1 |

- **Gate 2 (masking): PASS → the ~28k-wh exhaustive finish recollection is CANCELLED.** k15 finish labels ≥ dense on every axis (@2 34.7 vs 29.4) at 58% fewer finish ceilings. Dense finish ceilings actually DILUTE perfect-play.
- **Gate 1: finish boards belong in round-2 — ADDED to full roots, in k15 form.** M2-k15 = best 2-push yet (@2 32.4→34.7, @30 69.4→72.4, avg 93→89). **Never substitute:** M1 (roots swapped out) lost on every axis — roots carry the 1-push skill.
- NOT a contradiction of the density ablation: that varied ROOT setup density (keep dense); this varies FINISH ceiling density (want k15-sparse).

**4. Extras attribution — beast-1-c081-noextra (results: `round1/eval/noextra_results.md`).**
Champion recipe retrained on the champion's H5 minus the extras (literal filter: drop 20,931, keep **170,772**; the verifier's 23,639 is not exactly reproducible — ~2.7k fuzz in 4,447 old-rich-duplicate-key rows; conclusion robust to the boundary).

| axis | champion (191,703) | noextra (170,772) | Δ |
|---|---|---|---|
| 1p med/hard/all@1 | 86.9/48.5/86.8 | 83.4/45.1/85.3 | −3.5/−3.4/−1.5 |
| 2p solve/@2/@30 | 95.1/32.4/69.4 | 95.4/28.0/69.3 | +0.3/−4.4/−0.1 |
| hardh2 s2s | 8.5 | 7.6 | better |

**Verdict: extras mildly load-bearing, keep them** (already collected, cost nothing). Dropping ~21k of the HARDER episodes cost ~3 hard@1 / ~4 @2, left solve/@30 flat — the antman "volume is the lift" pattern. **Seed nuance:** champion's 48.5 hard@1 now looks like a HIGH seed — noextra 45.1 + M2 arms 43.6/43.1 all cluster at 43–45. On 2push@2 (champion not an outlier) noextra is genuinely lowest (28.0) — the real evidence the extras helped. Champion stays champion (this was attribution). **The 45/43/43-vs-48.5 clustering is a second vote for a confirm-seed before locking round-2.**

**⇒ Round-2 recipe (fully specified):** source = dead-bank (~1M antman-failed leads; true-2push-only supply) + fresh scenes → exhaustive ROOT setup sweeps (dense labels) + champion-ordered finish sweeps (k≈20, ≥0.95 early-exit, ~0.85 ceilings, 1–2k calibration batch first) → train on dense roots + k15 finish boards ADDED. Fix target = the setup-anchor gap (setups scored 0.583 not 0.9). Scene-selection stays dumb (3-arm control proved targeting ≈ volume). Confirm-seed the champion/M2-k15 before committing.

## Decisions locked [USER 2026-07-14]

1. **Fresh restart** — discard the buggy lineage; new Marvel-named lineage.
2. **Pure DAgger** (mistake-targeting) after the seed — NO forced balance beyond the bootstrap.
3. **Bootstrap = 50k SOLVABLE episodes, difficulty-LEANED, hard RELAXED, NO dead** [USER 2026-07-14, post-pilot — SUPERSEDES the earlier 20k/20k/20k]. Cap easy, take all med + all-available hard; train+val split by room. The pilot FALSIFIED "easy is rare": easy DOMINATES (73–77% of solvable), genuine-1-push-hard is the RARE bin (3.5% in both templates), so 20k hard would cost ~1M scenes. No dead-as-negatives — dead banks for Beast. All generation **60/40 aug9/feb** [USER — aug9 more interesting]. See Run log pilot entry.
4. **Goal-opening labels** (`target_goal_region` ON) — always.
5. **In-sync generator** (the `_runtime_validate_adjacency` fix, verified) — always.
6. **Fixed-cut difficulty**, never tertiles.
7. **Curriculum ladder** 1-push → 2-push, DAgger within each stage.

## Eval

Solve@k by difficulty (easy/med/hard, fixed cuts) on the held-out `namo_testset_v1`, vs the random ranker — both **reactive @1** and **with-search @k**. Bar: beat random on every tier; the with-search curve dominates random's (any solve-rate at a fraction of the sim calls).

## Run log

**Gen↔label sync fix — VERIFIED (2026-07-14, `mujoco_env_creator@55badcb`).** Re-enabled `_runtime_validate_adjacency` to re-check each placement against the labeler's own live `get_region_snapshot` (robot/goal placed). Proof: label-time `goal_region_not_in_snapshot` drops **77.8% (fix OFF) → 0.0% (fix ON)** on a matched 80-scene sample; ~4.5 ms/sample. Generator now emits only true 1-hop scenes.

**Pilot (2026-07-14) — the premise FLIP.** ~560 scenes gen (fixed pipeline) + goal-opening labels, aug9_car_v3 & feb_car. "Easy is rare" is FALSE. Among SOLVABLE episodes: easy/med/hard = **73/23/3.5%** (aug9) and **77/20/3.5%** (feb) — easy dominates, **genuine-1-push-hard is the rare bin (3.5% both)**. Solvable/dead: aug9 42%/48%, feb 72%/25% (dead = 1-hop-adjacent goal that needs 2 pushes = free Beast material). Per-scene easy/med/hard/dead yield: aug9 0.39/0.13/0.019/0.62, feb 0.65/0.17/0.030/0.30. Gen accept-rate (runtime-validate % kept): aug9 27%, feb 90% (feb sparser → fewer multi-hop rejects). Cost to harvest 20k easy ≈ 31–51k scenes (<0.5 h @2k cores); 20k *solvable*-hard ≈ 0.7–1M scenes (~4–8 h) — the flipped bottleneck. sr from `algorithm_stats["primitive_trial_log"]` (openers/tried). Artifacts: `scratch_namo/tmp/antman0_pilot/SUMMARY.json`.
**Decisions from the pilot [USER]:** relax hard; seed = 50k SOLVABLE (no dead); lean composition; 60/40 aug9/feb; dead → Beast bank.

### RESULTS — 1-push ladder (Ant-Man), rounds 0–5 [2026-07-15/16]

Solve@1 by fixed-cut tier on held-out `namo_testset_v1` (1,323 eps; hard 204 / med 421 / easy 698), best-first hmax1 budget300 NOHZ base. Screener for round r = antman_{r-1}.

| model | screened | kept | keep% | train rows | easy@1 | med@1 | **hard@1** | all@1 | hard@20 |
|---|---|---|---|---|---|---|---|---|---|
| antman-0 (seed) | — | — | — | 50,000 | 92.7 | 63.7 | **23.0** | 72.7 | 85.8 |
| antman-1 | 9,033 | 493 | 5.46 | 50,528 | 93.7 | 67.7 | **24.0** | 74.7 | 86.3 |
| antman-2 | 715,432 | 38,266 | 5.35 | 90,700 | 95.6 | 73.2 | **28.4** | 78.1 | 91.2 |
| antman-3 | 686,951 | 28,950 | 4.21 | 120,657 | 96.3 | 77.2 | **32.8** | 80.4 | 91.7 |
| antman-4 | 682,577 | 30,660 | 4.49 | 151,218 | 96.7 | 81.2 | **39.2** | 82.9 | 94.6 |
| antman-5 | 449,284 | 16,959 | 3.78 | 167,655 | 97.1 | 78.9 | **42.6** | 82.9 | 92.6 |
| _random ranker_ | — | — | — | — | 62.6 | 19.2 | **1.5** | ~39.4 | 37.7 |

**Headline: hard@1 23.0 → ~39 over five rounds, ~26× random, then PLATEAU.** (The antman-5 row's 42.6 was undersized-449k-run noise; the full-scale 3-seed redo lands at 39.1 and plateaus — see RESOLVED below.) Gains concentrated at low k (the ranker finds openers *sooner*; sim verifies for free). Beats random on every tier at every k, and beats the exhaustive NoHz-v3 baseline (+8.2 hard@1) with cheap sampled data.

**Findings:**
- Round 1 (+528 rows) = noise (+1.0, within ~0.3 mm eval jitter). Rounds 2–5 are real (+4.4/+4.4/+6.4/+3.4).
- **Volume tracks the climb**, but keep-rate falls (5.46% → 3.78%) as the model eats its own error distribution, and per-row efficiency *rises* (round 5 best: ~2.0e-4 hard@1/row on the fewest rows). First hint DAgger targeting matters, n=2, not proven.
- Tiers: easy SATURATED (97@1, high-k maxed). med@1 still live (peaked 81.2 @a4, dipped 78.9 @a5). hard = main headroom.
- **Round 5 was a REDISTRIBUTION, not a lift:** gained hard@1 (+3.4), hard@2 (+6.4) but dropped med@1 (−2.3) and hard@20 (−2.0). Sharpened the top of the ranking at a cost to the tail. Also confounded: 449k vs ~700k for rounds 2–4.

**RESOLVED — round-5 REDO at full 737k + 3-arm control [2026-07-16]** (all eval 3-seed mean±std, same hmax1 budget300 harness; every arm = base 151,218 + 27,146 rows, size-matched, differ ONLY in the added rows' selection):

| arm | hard@1 | med@1 | all@1 | Δhard |
|---|---|---|---|---|
| base (through-r4, 0 delta) | 37.3±1.0 | 79.1±1.2 | 82.1±0.6 | — |
| **mistakes** (targeted, = antman-5-redo) | 39.1±1.4 | 81.0±1.1 | 83.1±0.5 | +1.8 |
| iid-volume (random iid) | 40.5±1.5 | 81.4±1.6 | 83.4±0.7 | +3.2 |
| diff-match (difficulty-matched iid) | 40.5±0.9 | 80.6±0.9 | 83.3±0.6 | +3.3 |

1. **PLATEAU — answered YES.** Full-scale antman-5-redo = **hard@1 39.1±1.4**, med restored to 81.0. The old 449k run's 42.6-hard/78.9-med "redistribution" was small-sample NOISE (the redo undid both). Ladder tops out ~39.
2. **TARGETING vs VOLUME — answered: NO advantage.** The 3 delta arms TIE — mistakes (+1.8) is no better than, if anything behind, random iid (+3.2) and difficulty-matched iid (+3.3). The ~+2-3 lift is pure VOLUME; composition is irrelevant. DAgger's hard-mistake selection bought nothing at this scale.
3. **vs exhaustive NoHz-v3 (same fixed-cut harness):** NoHz-v3 = hard@1 **30.9** / all 82.3; **antman-5 beats it +8.2 on hard** with cheap sampled data. (Archived "NoHz hard 54.2" = TERTILE binning; all@1 82.3 matches exactly, binning-invariant — see [[pipeline_1push_binning_mismatch]].)

**Beast (2-push) dataset accumulated for FREE:** 72,521 labeled-dead episodes with full `(xml, object_id, robot_goal)` identity (seed 54,268 + r1-r5 ~18,253), plus ~865k unlabeled screen leads (`phase2_bank/screen_dead_scenes.txt`, xml-only). ~42% of every screen is dead, model-stable across rounds.

**Execution notes (bugs found + fixed, so we don't repeat them):**
- **orch wait bug:** `orchestrator.sh` called undefined `wait_amarel_jobid` (typo for `wait_amarel_jobs`) → command-not-found → no wait → halted r1 at "rows=0". FIXED: defined `wait_amarel_jobid <jobid>` in `lib.sh` (blocks via `squeue -j`, +10s submit-race guard).
- **node damage:** `MUJOCO_GL=egl` in `build_array.sbatch` wedged **25 halk nodes** (unkillable GL init on GPU-less nodes; slurmstepd "not ending with signals"; 2 NODE_FAILs). GUARDED with `--exclude=halk[0001-0159]`; **root cause still OPEN** (build renders via matplotlib/Agg + cv2 + reads many pkls; halk-only, not CPU/heat; screens never wedged anything). Paul (admin) flagged it; email drafted.
- **count truncation:** `timeout N find|wc -l` truncates mid-count → 5× scene undercounts + a duplicate ledger row. Never trust a timed-out count. [[feedback_check_process_owner]] neighbourliness: pool gen capped 128 cores; Amarel fair-use ≤200 background / bursts ≤5h.

**NEXT: 2-push (Beast).** 1-push ladder DONE — saturated ~39 hard@1, above the exhaustive NoHz-v3 bar (30.9), no data-selection strategy climbs it further. Beast = the discounted value-ranker trained on the accumulated post-setup/dead scenes (72,521 labeled-dead w/ full identity + ~865k leads), with antman-5 as the rung-1 ranker that makes setup-labeling + search cheap: enumerate setups → top-k finish-verify with antman-5 → exhaustive only on the misses → recycle misses as dual-purpose (harder 1-push examples + true 2-push labels). recall@k on post-setup states is the cap criterion.

**Round-5-redo + control execution notes:**
- **CS `unlimited` rejects `--cpus-per-task`** — omit it (job rejected otherwise). `train_cs.sbatch` fixed.
- **loky DataLoader TMPDIR collision:** two trains co-scheduled on ilab1 crashed their DataLoader workers (`FileNotFoundError` on shared `/loky-*` temp dirs) → 0 ckpts. FIXED with per-job `export TMPDIR=/tmp/namo_$SLURM_JOB_ID` in the sbatch (matches the orchestrator's node-local-tmp pattern).
- **zsh arrays are 1-indexed** — `${arr[0]}` is empty; pass GPU ids explicitly in eval fan-out, don't index a bash-style array.
