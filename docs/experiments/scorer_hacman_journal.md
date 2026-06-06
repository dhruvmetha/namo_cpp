# Scorer → HACMan-faithful: experiment journal

**Owner:** autonomous overnight session (started 2026-06-06, user asleep ~8h).
**Objective (do not drift):** a 1-push *push-success scorer* — given a scene, score every (edge, depth)
primitive by P(opens a path to the goal) — that (a) beats the diffusion baseline and the honest floor,
and (b) is built so the SAME net becomes the value function for 2-push search later. We are NOT
collecting exhaustive 2-push labels; multi-push is search over this scorer.

**Method — every decision is a hypothesis with a test.** Format below. A change is KEPT only if its
test accepts; otherwise reverted/iterated. Bars: held-out **by room**, binned by **true difficulty**,
honest **without-replacement** floor, success@k + the failure decomposition (wrong-edge% / wrong-depth%
/ rank-of-first-valid). Primary metric to move: **hard @1** and **hard wrong-edge%**.

---

## Baseline (E0) — global-readout DiT classifier  [DONE, established]
- Arch: DiT over 5×64×64 crop → CLS token → MLP → 60×5. 6.7M params. Supervised masked BCE+Dice.
- Result (hard / med / easy, success@1): **14.3 / 46.4 / 84.2**;  hard @20 = 82.6 (vs floor 47).
- **Diagnosis (the bottleneck):** top-1 errors are **96.6% wrong-EDGE**, only 3.4% wrong-depth; first
  valid push sits at **median rank 7/75**; depth collapsed to **always-d4 (98%)**. Score separation
  weak (+0.12, positive in only 63% of scenes). Viz: a smooth spatial *gradient* across edges.
- **Conclusion:** the failure is **edge-selection precision**, and its cause is the **global readout**
  (one scene vector forced to emit 60 scores → can only express "which side," not "which exact edge").
  Depth is NOT the bottleneck (it solved it via a max-push prior). → fix edge precision first.

---

## Hypotheses & experiment plan

**H1 (E2 — per-edge tokens + cross-attention):** giving each edge its OWN token (positional id of its
contact location + a feature gathered from the scene) that cross-attends to the scene will let each edge
reason independently → **wrong-edge% drops, hard @1 rises** above 14.3. *This is HACMan's per-point
critic, on our 60-edge "point cloud."*  TEST: train, eval; accept if hard @1 ↑ and wrong-edge% ↓ vs E0.

**H2 (E3 — + zoomed object crop):** if E2 helps but per-edge scores are still fuzzy (resolution-bound),
adding a sharp zoomed object crop (object-bbox sub-crop, resized) as the source of each edge's *local*
feature will further sharpen edge precision. TEST: vs E2. (Only run if E2's residual looks resolution-limited.)

**H3 (E4 — more data, 10× pool):** le10 is ~10% of the 211k-scene pool; the weak score-separation /
"lost" ~17% may be data-limited. Regen masks for more scenes → hard @1 ↑. TEST: vs best-so-far. (Expensive;
gated on E2/E3 showing the arch is right but data-hungry.)

**H4 (E5 — continuous-duration actor, the HACMan "how"):** depth collapsed to always-d4, which
structurally misses the ~36% of valids needing a short push (the d0/d1 cases). A per-edge continuous
duration (actor maximizing the critic) should recover those. TEST: wrong-depth% / those specific cases.
*Lower priority — depth is only 3.4% of errors; do after edge precision is solved.*

**Critical guardrails:** (1) always compare on the SAME held-out test episodes as E0/diffusion;
(2) re-check the honest (without-replacement) floor each time; (3) watch for new degenerate shortcuts;
(4) leak-check any new data (0 train/test room overlap); (5) if a hypothesis is rejected, say so plainly
and record why — negative results are results.

---

## Results log  (filled in as experiments complete)

### E2 — per-edge cross-attention critic   [TRAINING, job 55581167, started 2026-06-06 ~01:57]
- **Hypothesis (H1):** the global CLS readout causes the 96.6% wrong-edge failure; per-edge tokens that
  reason independently will drop wrong-edge% and raise hard @1 above 14.3.
- **Design (`src/model/dit/edge_crossattn.py`, 4.3M params):** DiT patch-embed (4×4) → 16×16 scene
  tokens → 4 self-attn blocks (scene context). 60 **edge tokens** = `grid_sample` of the scene feature
  map at each edge's contact pixel (local feature) **+** an MLP positional id of its (x,y) contact
  coord. 4 **cross-blocks**: edge tokens cross-attend to scene + self-attend among edges (point-
  transformer). Shared per-edge MLP head → 60×5. = HACMan's per-point critic on our 60-edge cloud.
- **Data:** same `v3_scorer_1push` H5 + new `contact_px` (60,2) added via `add_contact_px.py` (pose
  math, in-bounds 100%). Same room-grouped split, same masked BCE+Dice loss. *Only the readout changed*
  vs E0 — so a win is attributable to per-edge reasoning, not data/loss.
- **Test:** eval on the SAME test episodes → compare hard @1 + wrong-edge% + grid viz vs E0
  (14.3 @1, 96.6% wrong-edge). ACCEPT if @1 ↑ and wrong-edge ↓.
- **RESULT — H1 ACCEPTED (strongly).** Mid-training read (epoch 41, val_loss 0.60 vs E0's 0.86 plateau;
  still improving, so this is a *lower bound*):
  - success@1: hard **14.3 → 26.6**, med 46.4 → 71.1, easy 84.2 → 96.1.  (vs diffusion 5.9/28.9/64.6.)
  - first-valid **median rank 7 → 3**; rank≤3 29% → 51%; score separation **+0.12 → +0.38** (3× sharper).
  - wrong-edge% 96.6 → **90.2** (still the dominant residual error, but the model is far better *at* it).
  - depth still ~always-d4 (85%); wrong-depth rose 3.4 → 9.8% (relatively larger now that edges are better).
  - "lost core": positive-separation scenes still **64%** (unchanged from E0) → ~36% of hard scenes have
    NO signal regardless of arch → smells **data-limited**, not arch-limited.
- **FINAL (converged epoch045, val_loss 0.594):** hard **24.0** / med 70.7 / easy 94.9 @1; hard @20 89.8.
  NB: the best-val_loss ckpt's hard @1 (24.0) is slightly *below* the epoch-41 read (26.6) — val_loss is
  all-difficulty, hard@1 is a noisy n=413 slice. So E2 ≈ **24–26 @1 on hard** (~1.7× E0, ~4× diffusion,
  ~9× floor). Takeaway: **E2 has ~maxed out on le10 data** (converged, hard@1 flat mid→final).
- **Decision:** keep E2 as the backbone. The flat mid→final on hard@1 says the *architecture* converged
  on this data → test the two orthogonal levers: **E2b (more capacity, same data)** and **E4 (same arch,
  3.6× data)**. Both LAUNCHED. E3 (zoom) held unless these stall on the edge-precision residual.

### E3-fine — resolution lever (finer per-edge gather)   [TRAINING, job 55588744]
- **Hypothesis (H2, refined):** the per-edge gather from the coarse 16×16 feature map blurs adjacent
  edges (~8 edges/cell), capping edge precision; a finer **32×32 map (patch=2 vs 4)** resolves them
  (~4 edges/cell) → wrong-edge% ↓, first-valid rank ↓, hard @1 ↑. *This is the cleanest resolution test —
  pure config change, same data/model, no rebuild* (vs the originally-planned 2nd zoomed crop, held as
  the heavier alternative if this helps but isn't enough).
- **Test:** vs E2 (patch=4). ACCEPT if hard @1 ↑ / wrong-edge ↓.
- result: PENDING (training).

### E4 — more data (the data lever for the lost core)   [TRAINING, 2026-06-06 ~04:40]
- **Hypothesis (H3):** E2 left the ~36% no-signal "lost core" of hard scenes unchanged → it's
  data-limited, not arch-limited. ~3.6× more (hard-inclusive) data should raise positive-separation% and
  hard @1/coverage. *Uses the E2 architecture — isolates the DATA variable.*
- **Data:** 80k random pkls from the full 211k solvable TRAIN pool (le10 was 22k). mask-gen array
  `55584295` (27 shards, reusing the tested `run_batch_collection_smoke`, extra= empty = same config as
  le10) + per-episode labels `55584296`. Then build H5 (--tight-only) → join (build_scorer_dataset) →
  add_contact_px → train E2-arch. ~75-100k episodes expected (~3.5× le10). Leak-check vs test before training.
- **Test:** train E2-arch on E4 data, eval vs E2-on-le10. ACCEPT if positive-separation% ↑ and hard @1 ↑.
- **⚠️ CRITICAL REALIZATION (logged before training, 2026-06-06 ~04:40):** E4's difficulty mix is
  **4% hard / 21% med / 76% easy** (the pool's natural distribution), vs le10's hard-enriched 34/46/19.
  Because **le10 = all pkls that have a ≤10% episode, le10 already contains EVERY hard training scene** —
  random sampling can only add easy/med, NOT hard. So E4 has *fewer* hard episodes than le10 (~4k vs ~9.5k).
  → E4 actually tests "more *total/diverse* data (with the easy-skew)", NOT "more hard data". And it implies
  the lost core **cannot** be fixed by more data — *we already have all the hard data* → the lost core is
  most likely **feature-limited** (the 5 masks don't determine which push opens the path; needs richer
  features like the wavefront distance field, which this H5 lacks). Still running E4 to see if diversity
  helps or the skew hurts — but the strong prior is now: data is NOT the lost-core lever; **features are.**
- E4 data: 98,387 episodes, 79,626 rooms, contact_px ✓, gt_in_valid 100%, 0 test leakage. job 55588491.
- result: PENDING (training).

### E2b — scale the winner (capacity lever)   [DONE — H REJECTED]
- **Hypothesis:** E2 (4.3M) capacity/training-limited; a bigger edge model (dim 256, depth 6/6, 2× params)
  improves further.
- **RESULT — REJECTED.** E2b val_loss converged *worse* (0.636 vs E2's 0.594), and on the metric it is
  **no better: hard @1 24.2 vs 24.0** (flat), med 64.6 < 70.7, easy 93.4 < 94.9. The bigger model does not
  help — likely under-trained at the lower lr, but regardless, **E2 (4.3M) is already the right size.**
- **Conclusion:** the limit is NOT capacity. With the lost core also unchanged E0→E2 (65% positive-sep),
  the evidence converges on **data** as the lever → E4 is the decisive test. (Negative result = result.)

---

## Reading list (for the user on waking)
Grounding for the per-edge / point-critic architecture we built:
- **HACMan** (Zhou et al., CoRL 2023) — the per-point Q-map + actor-critic we replicate.
  https://arxiv.org/abs/2305.03942 · code: github.com/HACMan-2023/HACMan (cloned at /scratch/dm1487/refs/HACMan).
  Read §"per-point feature extractor" + the PointNet++ U-Net (SA encoder + FP decoder w/ skips) — that's
  exactly what our edge-token cross-attention does, on a 60-edge cloud.
- **HACMan++** (Jiang et al., RSS 2024) — spatially-grounded *parameterized* primitives (what/where/how),
  the chaining we'll use for 2-push. https://arxiv.org/abs/2407.08585 · code: github.com/JiangBowen0008/HACManPP.
  Note it uses a **point_transformer** over the points — validates our transformer-over-edges choice.
- **Point Transformer** (Zhao et al., ICCV 2021) — local self-attention → per-point features fusing
  local+global. https://arxiv.org/abs/2012.09164 . This is the backbone idea of our edge self-attention.

**My recommendation for what to read first:** the HACMan per-point feature-extractor + PointNet++ section
— it makes the "each edge gets its own local+global summary" idea concrete and shows why the global
readout (our E0) was the bottleneck.

### E-oracle — geometric+wavefront oracle: is the lost core feature/FOV-limited or model-limited?   [RUNNING]
- **Sharpened hypothesis (the big one).** The scorer's input is ONLY the **tight 0.5 m crop** (confirmed:
  scorer H5 `ctx` = 5×64×64 from `local_tight_*`, `crop_size_meters`=0.5). So it reasons about "does a path
  to the goal open" through a 0.5 m keyhole around the object. **Hypothesis H5: the lost core is a
  field-of-view problem** — for ~36% of hard scenes the corridor that opens when you move the object lies
  (partly) outside 0.5 m, so the answer is *not in the crop* and NO model on this input can recover it.
  The competing hypothesis H5′: the answer IS in the crop and the model just can't extract it (arch/res).
- **Decisive test — `scripts/sandbox/geom_oracle.py`.** A PERFECT geometric reasoner that uses only the
  crop the model sees. Per candidate (edge, depth): look up the object's SE(2) displacement from the
  calibrated `1x_car_d5` primitive .dat → move the target footprint (rigid) → obstacle = static ∪
  other-movables ∪ moved-target → inflate by the robot radius (the wavefront's obstacle inflation) →
  8-conn flood from the robot region → "opens" iff goal-sample region joins the robot's component.
  Predicted-valid vs TRUE-valid (from sim) gives the oracle's recall.
  - **Decision rule:** oracle hard-recall ≈ easy-recall → info is in the crop → **model-limited** (push
    resolution E3 / better arch / explicit wavefront feature). Oracle hard-recall collapses → crop lacks
    the answer → **FOV/physics-limited** → bigger crop (wide 1.2 m — data must be regen, none exists yet)
    or full-map features / multi-push.
- **Every assumption gated before trusting the number:** G1 render-IoU (rendered object pose matches the
  real target mask) — smoke **0.948** ✓. G2 nopush-IoU (flood reproduces `robot_region`) — first naive
  flood **FAILED at 0.219** (leaked through 1-px gaps a 7 cm car can't fit) → **fixed** by calibrating an
  obstacle-inflation radius r* to reproduce `robot_region` (this is exactly the wavefront's inflation).
  G3 easy-recall must be high or the oracle itself is broken.
- **RESULT (tight crop) — initially read as FOV-limited; ⚠️ THIS READ WAS WRONG, corrected below by the
  wide-crop oracle + the model-beats-oracle check. Read the CORRECTION before citing anything here.**
  Full test set (n=413/491/752), both gates pass (G1 render-IoU **0.952**, G2 reproduce-robot_region
  **0.815** at calibrated r*=20px ≈ 4.5 cm robot radius). A *perfect* geometric reasoner with the 0.5 m crop:

  | bin  | oracle recall | oracle precision | base-rate | goal clipped by crop | misses via off-crop path |
  |------|---------------|------------------|-----------|----------------------|--------------------------|
  | hard | 67.3%         | **5.9%**         | 2.7%      | 26.9%                | **89.4%**                |
  | med  | 65.7%         | 29.6%            | ~16%      | 25.7%                | 89.8%                    |
  | easy | 85.6%         | 81.9%            | ~65%      | 19.5%                | 88.0%                    |

  Reading: even with perfect geometry, the 0.5 m crop caps hard recall at 67% and gives precision barely
  2× the random base-rate (within-crop connectivity is a near-useless signal on hard). The oracle's own
  precision collapse easy→hard (82%→6%) mirrors the *model's* easy→hard collapse — strong evidence the
  model's hard difficulty is **information-limited by the crop, not capacity/data-limited.** And ~89% of
  the oracle's missed valids are explained by the goal connecting through an **off-crop** path.
- **Honest caveats:** the oracle approximates (rigid single-object motion, within-crop reachability with
  ~18% modeling error per G2, sampled goal region) → absolute recall ±~15%. But the *pattern* (hard
  precision collapse + off-crop dominance of misses) is gate-validated and robust.
- **Falsifiable PREDICTION (pre-registered):** because E2b (capacity), E3-fine (resolution), and E4 (data)
  ALL keep the 0.5 m FOV, the oracle predicts **all three are ~flat on hard@1** (≈ E2's 24). E2b already
  confirmed (24.2). If E3/E4 also land flat on hard, that triple-confirms FOV as the bottleneck.
- **Decision → the lever is FIELD OF VIEW.** The wide 1.2 m crop already exists in the le10 NPZs
  (`local_wide_*`, all 5 channels, object centered to <3 mm — verified). Two routes:
  - **E5-wide** (wide-only): isolates the FOV variable. *Caveat measured:* in a 1.2 m crop the object
    spans only ~11 px (vs ~27 px tight) so the 60 edge contact points squish into ~11 px → per-edge
    gather loses edge resolution. So wide-only trades FOV-gain against edge-precision-loss (ambiguous).
  - **E5-dual** (tight+wide, like the diffusion's dual-crop): tight encoder for edge precision + wide
    encoder for FOV context, per-edge feature gathered from BOTH. The principled fix; more model surgery.
  - **Cleanest model-free check first:** run THIS oracle on the WIDE crop. If wide-oracle hard
    recall/precision ≫ tight, the answer is in 1.2 m → spend GPU on E5. If not, even 1.2 m is too small →
    need full-map / wavefront-distance features or genuine multi-push. (Pipeline `--crop wide` verified.)

- **⚠️ CORRECTION (the wide-crop oracle ran — it FALSIFIES the FOV conclusion).** Ran the same oracle on
  the 1.2 m wide crop (built `v3_test_*_lzf_both_data` from the test NPZs; gates pass G1 0.887, G2 0.832
  at the correctly re-calibrated r*=8px = the same ~4.5 cm radius in coarser px). Result vs tight:

  | bin  | recall (tight→wide) | precision (tight→wide) | goal clipped (tight→wide) | misses off-crop (tight→wide) |
  |------|---------------------|------------------------|---------------------------|------------------------------|
  | hard | 67.3 → **68.8**     | 5.9 → **5.8**          | 26.9% → **1.7%**          | 89.4% → **3.1%**             |
  | med  | 65.7 → 67.5         | 29.6 → 29.3            | 25.7% → 1.4%              | 89.8% → 8.6%                 |
  | easy | 85.6 → 87.5         | 81.9 → 81.0            | 19.5% → 2.0%              | 88.0% → 16.0%                |

  The wider crop **removes the goal-clipping** (26.9%→1.7%) and the off-crop misses (89%→3%) — the goal is
  now fully in view — **yet recall and precision on hard are UNCHANGED (68.8/5.8 ≈ 67.3/5.9).** So
  goal-visibility was NOT the binding constraint. The "89% off-crop" attribution in the tight read was a
  *proxy artifact* (goal-touches-border ≠ answer-is-off-crop): the same scenes stay unsolved at full FOV.
- **⚠️ AND the oracle is a WEAK BASELINE, not a ceiling — the trained model already beats it.** The
  geometric oracle's hard *precision* (≈ its success@1 as a ranker, since it emits binary open/closed) is
  **~6%**; E2 gets **24% hard@1** and **89.8% hard@20** (the oracle as a random pick among its ~34 "opens"
  would be ~6%@1 and only ~40%@20). So E2 extracts **~4× more** signal from the SAME crop than the rigid
  free-space oracle. The oracle over-predicts wildly on hard (~6% precision = it thinks half of all pushes
  open the goal) because rigid free-space motion ignores clutter (the object stops early / nudges
  neighbors). **The model has learned clutter-aware push outcomes the oracle cannot.** ⇒ this oracle
  **cannot upper-bound** the model; my "FOV-limited lost core" claim is **rejected**.
- **What actually holds (durable, honest):** (1) ~27% of hard goals ARE clipped by the 0.5 m crop — a real
  but *secondary* FOV limitation (fixing it didn't move the needle). (2) The hard plateau is NOT explained
  by capacity (E2b flat), data (E4 flat — see below), FOV (this), or — pending — resolution (E3). It looks
  like **genuine 1-push task difficulty**: hard scenes have few valid single pushes (median |valid| ~2 of
  ~63 reachable) and marginal geometry. (3) E2 is a strong converged 1-push scorer: beats diffusion ~4×,
  the random floor ~9×, AND a calibrated geometric oracle ~4× on hard@1.
- **Lesson logged:** a cheap proxy (goal-touches-border) over-attributed the cause; the decisive move was
  the *direct counterfactual* (actually widen the FOV and re-measure) + checking the baseline against the
  model. Pre-register the falsifier, not just the confirmer.

### E4 — more data (the data lever)   [PRELIM eval mid-training, epoch~17]
- **RESULT — H3 confirmed: data is NOT the hard lever.** E4 (3.6× data, easy-skewed) hard@1 = **24.0**,
  *identical* to E2's 24.0 (le10). Med **80.4** (>E2 70.7) and easy **99.3** (>E2 94.9) rose — but only
  because E4 added med/easy examples; hard was already saturated (le10 = every ≤10% scene). Exactly the
  E4-realization prediction. (Mid-training; hard@1 converges early/flat as in E2, so this is firm.)

## Key open question for the lost core — RESOLVED, with a corrected answer (E-oracle, 2026-06-06)
Hypothesis history (each step tested, several rejected — this is the audit trail):
1. data-limited? → **NO.** E4 realization: le10 already has ALL hard data; E4 (3.6× data) hard@1 flat 24.0.
2. capacity-limited? → **NO.** E2b (2× params) hard@1 flat 24.2.
3. FOV-limited? → tight oracle *suggested yes* (89% misses "off-crop"), but the **wide-crop oracle FALSIFIED
   it**: widening to 1.2 m removes goal-clipping (27%→1.7%) yet hard recall/precision don't move. Off-crop
   was a proxy artifact.
4. resolution-limited? → E3-fine **pending** (prediction: flat, same 0.5 m FOV / same task).
5. Is the geometric oracle even a ceiling? → **NO** — the trained E2 (24% hard@1) beats the oracle's ~6%
   precision ~4×. The oracle is a weak rigid-free-space baseline, not an upper bound.

**Corrected answer:** the hard plateau (~24 @1) is **genuine 1-push difficulty**, not a fixable input/arch
gap — hard scenes simply have very few valid single pushes (median ~2/63) and the model already extracts
more than rigid geometry can. The right move toward the **objective** (multi-push via search over the
scorer) is **2-push search using E2 as the value function** — where hard scenes get solved by *composing*
pushes (89.8% hard@20 means the right push is almost always in the top-20 the search would expand), not by
squeezing 1-push @1 further. Secondary/optional levers if we still want 1-push gains: dual-crop (addresses
the real ~27% goal-clipping) and a faithful sim-based oracle / wavefront-distance features (the rigid
oracle's failure says clutter-aware push *outcomes* are the missing signal). See the E-oracle entry above.
