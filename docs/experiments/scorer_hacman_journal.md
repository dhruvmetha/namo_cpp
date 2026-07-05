---
status: frozen
tags: [experiment]
updated: 2026-06-07
---

# Scorer → HACMan-faithful: experiment journal

> ## ⭐ FINAL SYNTHESIS (2026-06-06 ~08:10, converged + multi-checkpoint numbers — READ THIS FIRST)
> Earlier entries below report **single-checkpoint** hard@1 point estimates that turned out to carry
> **±3-4 noise on n=413** (demonstrated: E2 across its 4 ckpts = 24.0/25.4/25.4/27.4). Two conclusions in
> the body were CORRECTED once I (a) evaluated *converged* checkpoints and (b) measured the per-model
> checkpoint spread. The verified picture (hard@1, multiple ckpts per model):
>
> | model | data / change | hard@1 (ckpt range) | hard@20 | verdict |
> |---|---|---|---|---|
> | E0 DiT global readout | le10 tight | 14.3 | 82.6 | baseline |
> | **E2 per-edge cross-attn** | le10 tight | **25.6** (24.0–27.4) | ~89.5 | **the big win (+11 over E0), solid** |
> | E2b 2× capacity | le10 tight | 24.2 | — | flat (within noise) → not a lever |
> | E3-fine patch=2 | le10 tight | 25.6 (25.2–25.9) | ~90 | flat (within noise) → not a lever |
> | **E4 same arch** | **3.6× data** tight | **28.8** (conv. 28.3–29.8) | ~87.5 | **modest REAL gain ~+3** (all 3 converged ckpts > E2's best) |
>
> **Robust takeaways:** (1) E2's per-edge architecture is the big win over E0 (+11, far beyond noise).
> (2) Capacity and resolution are flat → not levers. (3) **Data IS a modest lever (~+3 hard@1)** — even
> easy-skewed data helps via better *representation* (le10 already had all hard scenes), so data *scaling*
> (H3, 10×) is live; my earlier "data flat" was an artifact of evaluating E4 mid-training (epoch17=24.0,
> converged=28-30). (4) FOV is NOT the lever (wide-crop oracle falsified it) and the geometric oracle is a
> weak baseline the model already beats ~4× — not a ceiling. (5) Hard remains genuinely hard for 1-push
> (~2 valid pushes of ~63); the route to *solving* it is **2-push search over the best scorer (E4)**,
> while **data scaling** is the live 1-push lever. **Methodology lessons:** eval CONVERGED checkpoints;
> characterize checkpoint noise before reading ±3 differences as signal; pre-register the falsifier.
> *(Below this banner, treat any single "hard@1 = X" as ±3-4 and superseded by this table.)*

---

**Owner:** autonomous overnight session (started 2026-06-06, user asleep ~8h). **Objective (do not drift):** a 1-push *push-success scorer* — given a scene, score every (edge, depth) primitive by P(opens a path to the goal) — that (a) beats the diffusion baseline and the honest floor, and (b) is built so the SAME net becomes the value function for 2-push search later. We are NOT collecting exhaustive 2-push labels; multi-push is search over this scorer.

**Method — every decision is a hypothesis with a test.** Format below. A change is KEPT only if its test accepts; otherwise reverted/iterated. Bars: held-out **by room**, binned by **true difficulty**, honest **without-replacement** floor, success@k + the failure decomposition (wrong-edge% / wrong-depth% / rank-of-first-valid). Primary metric to move: **hard @1** and **hard wrong-edge%**.

**Decision attribution (convention, user request 2026-06-06):** every design choice is tagged **[USER]** (Dhruv's call) or **[CLAUDE]** (mine), so we can trace which decisions drove which results. Predictions are pre-registered (stated *before* the result) and tagged the same way.

---

## Baseline (E0) — global-readout DiT classifier  [DONE, established]
- Arch: DiT over 5×64×64 crop → CLS token → MLP → 60×5. 6.7M params. Supervised masked BCE+Dice.
- Result (hard / med / easy, success@1): **14.3 / 46.4 / 84.2**;  hard @20 = 82.6 (vs floor 47).
- **Diagnosis (the bottleneck):** top-1 errors are **96.6% wrong-EDGE**, only 3.4% wrong-depth; first valid push sits at **median rank 7/75**; depth collapsed to **always-d4 (98%)**. Score separation weak (+0.12, positive in only 63% of scenes). Viz: a smooth spatial *gradient* across edges.
- **Conclusion:** the failure is **edge-selection precision**, and its cause is the **global readout** (one scene vector forced to emit 60 scores → can only express "which side," not "which exact edge"). Depth is NOT the bottleneck (it solved it via a max-push prior). → fix edge precision first.

---

## Hypotheses & experiment plan

**H1 (E2 — per-edge tokens + cross-attention):** giving each edge its OWN token (positional id of its contact location + a feature gathered from the scene) that cross-attends to the scene will let each edge reason independently → **wrong-edge% drops, hard @1 rises** above 14.3. *This is HACMan's per-point critic, on our 60-edge "point cloud."*  TEST: train, eval; accept if hard @1 ↑ and wrong-edge% ↓ vs E0.

**H2 (E3 — + zoomed object crop):** if E2 helps but per-edge scores are still fuzzy (resolution-bound), adding a sharp zoomed object crop (object-bbox sub-crop, resized) as the source of each edge's *local* feature will further sharpen edge precision. TEST: vs E2. (Only run if E2's residual looks resolution-limited.)

**H3 (E4 — more data, 10× pool):** le10 is ~10% of the 211k-scene pool; the weak score-separation / "lost" ~17% may be data-limited. Regen masks for more scenes → hard @1 ↑. TEST: vs best-so-far. (Expensive; gated on E2/E3 showing the arch is right but data-hungry.)

**H4 (E5 — continuous-duration actor, the HACMan "how"):** depth collapsed to always-d4, which structurally misses the ~36% of valids needing a short push (the d0/d1 cases). A per-edge continuous duration (actor maximizing the critic) should recover those. TEST: wrong-depth% / those specific cases. *Lower priority — depth is only 3.4% of errors; do after edge precision is solved.*

**Critical guardrails:** (1) always compare on the SAME held-out test episodes as E0/diffusion; (2) re-check the honest (without-replacement) floor each time; (3) watch for new degenerate shortcuts; (4) leak-check any new data (0 train/test room overlap); (5) if a hypothesis is rejected, say so plainly and record why — negative results are results.

---

## Results log  (filled in as experiments complete)

### E2 — per-edge cross-attention critic   [TRAINING, job 55581167, started 2026-06-06 ~01:57]
- **Hypothesis (H1):** the global CLS readout causes the 96.6% wrong-edge failure; per-edge tokens that reason independently will drop wrong-edge% and raise hard @1 above 14.3.
- **Design (`src/model/dit/edge_crossattn.py`, 4.3M params):** DiT patch-embed (4×4) → 16×16 scene tokens → 4 self-attn blocks (scene context). 60 **edge tokens** = `grid_sample` of the scene feature map at each edge's contact pixel (local feature) **+** an MLP positional id of its (x,y) contact coord. 4 **cross-blocks**: edge tokens cross-attend to scene + self-attend among edges (point- transformer). Shared per-edge MLP head → 60×5. = HACMan's per-point critic on our 60-edge cloud.
- **Data:** same `v3_scorer_1push` H5 + new `contact_px` (60,2) added via `add_contact_px.py` (pose math, in-bounds 100%). Same room-grouped split, same masked BCE+Dice loss. *Only the readout changed* vs E0 — so a win is attributable to per-edge reasoning, not data/loss.
- **⚠️ Contact-point fidelity vs namo_cpp [CLAUDE simplification, surfaced by USER question 2026-06-06]:** compared my `contact_px` (sage) against C++ `generate_rectangular_edge_points` (`src/planning/namo_push_controller.cpp:120`). **Matches:** the 60-edge *ordering* (interleaved Top0,Bottom0,Top1,…, then Right0,Left0,… — even<30=Top/odd<30=Bottom, even≥30=Right/odd≥30=Left) and the along-face sampling (`sample_lin(-w,w,n)` with n=15, full half-extent span). This index↔label mapping is verified independently (`edge_align_err=0`). **Does NOT match:** the C++ pushes each contact point OUT along the face normal by `offset = robot_radius + push_offset_margin` (~5–6 cm ≈ ~7 px in the 64-px crop ≈ ~1.5 cells of the coarse 16×16 feature map) — that standoff is where the *robot* stands to push. My gather point sits ON the object face (no offset). **Why:** a deliberate simplification — I used the face point as the per-edge positional anchor and didn't replicate the robot standoff. **Does it matter?** For label alignment, no (the *index* mapping is what aligns predictions to labels, and that's exact). For the feature gather, it's a ~1.5-cell shift in *where* each edge reads the scene; train and eval use the *same* convention so the model is self-consistent — but the C++ standoff point (in the robot's approach space) is arguably a *more informative* gather location than a point on the object body. **Candidate test (not yet run):** regenerate `contact_px` WITH the C++ offset → re-eval E2 → does hard@1 move? Clean, cheap; logged as a follow-up.
- **⚠️ Contact-pixel ALIASING [measured on real H5, surfaced by USER question 2026-06-06]:** checked `contact_px` on n=200. (a) **The 4 corners are EXACTLY coincident** — edge pairs (0,59),(1,31),(28,58), (29,30) sit at **0.000 px** apart — a direct consequence of omitting the standoff offset (adjacent faces share their endpoint; the C++ offset would separate them). So **4 of 60 edges are geometric duplicates**. (b) **Within a face**, the 15 points span only ~12.5 px (the object is *tiny* in the 0.5 m crop) → ~**0.9 px apart**, while the E2 feature cell is **4 px** (patch=4) → **~5 contact points read the SAME feature cell.** ⇒ per-edge *local* features are near-identical among neighbours; the model separates edges mostly via the **positional-id MLP**, not the local gather. Mechanistic reason resolution was a candidate lever (E3-fine patch=2 → 2 px cells) and why it helped only marginally (even at 2 px, points are 0.9 px apart → aliasing persists). **Implication:** genuine per-edge local discrimination needs a finer map *at the object* or a **zoomed object crop** (the planned-but-unbuilt zoom) — not just patch=2 on the same 0.5 m field.
- **Test:** eval on the SAME test episodes → compare hard @1 + wrong-edge% + grid viz vs E0 (14.3 @1, 96.6% wrong-edge). ACCEPT if @1 ↑ and wrong-edge ↓.
- **RESULT — H1 ACCEPTED (strongly).** Mid-training read (epoch 41, val_loss 0.60 vs E0's 0.86 plateau; still improving, so this is a *lower bound*):
  - success@1: hard **14.3 → 26.6**, med 46.4 → 71.1, easy 84.2 → 96.1.  (vs diffusion 5.9/28.9/64.6.)
  - first-valid **median rank 7 → 3**; rank≤3 29% → 51%; score separation **+0.12 → +0.38** (3× sharper).
  - wrong-edge% 96.6 → **90.2** (still the dominant residual error, but the model is far better *at* it).
  - depth still ~always-d4 (85%); wrong-depth rose 3.4 → 9.8% (relatively larger now that edges are better).
  - "lost core": positive-separation scenes still **64%** (unchanged from E0) → ~36% of hard scenes have NO signal regardless of arch → smells **data-limited**, not arch-limited.
- **FINAL (converged epoch045, val_loss 0.594):** hard **24.0** / med 70.7 / easy 94.9 @1; hard @20 89.8. NB: the best-val_loss ckpt's hard @1 (24.0) is slightly *below* the epoch-41 read (26.6) — val_loss is all-difficulty, hard@1 is a noisy n=413 slice. So E2 ≈ **24–26 @1 on hard** (~1.7× E0, ~4× diffusion, ~9× floor). Takeaway: **E2 has ~maxed out on le10 data** (converged, hard@1 flat mid→final).
- **Decision:** keep E2 as the backbone. The flat mid→final on hard@1 says the *architecture* converged on this data → test the two orthogonal levers: **E2b (more capacity, same data)** and **E4 (same arch, 3.6× data)**. Both LAUNCHED. E3 (zoom) held unless these stall on the edge-precision residual.

### E3-fine — resolution lever (finer per-edge gather)   [TRAINING, job 55588744]
- **Hypothesis (H2, refined):** the per-edge gather from the coarse 16×16 feature map blurs adjacent edges (~8 edges/cell), capping edge precision; a finer **32×32 map (patch=2 vs 4)** resolves them (~4 edges/cell) → wrong-edge% ↓, first-valid rank ↓, hard @1 ↑. *This is the cleanest resolution test — pure config change, same data/model, no rebuild* (vs the originally-planned 2nd zoomed crop, held as the heavier alternative if this helps but isn't enough).
- **Test:** vs E2 (patch=4). ACCEPT if hard @1 ↑ / wrong-edge ↓.
- **RESULT — H2 mostly rejected; resolution is a MARGINAL lever.** Converged (early-stopped epoch042, val_loss **0.5883** < E2's 0.5936): hard @1 **25.9** / med **72.9** / easy **96.3**, hard@20 **90.3** — a *small but consistent* lift over E2 (24.0/70.7/94.9, @20 89.8) across all three bins + val_loss. (The mid-training prelim read 25.2; converged is 25.9.) So the finer 32×32 per-edge gather *does* help a bit (less contact-pixel aliasing: ~4 edges/cell vs ~8 at patch=4) — **E3-fine is now the best single model** — but it's ~flat vs E2 (within the ±3-4 ckpt noise). **Lever sweep (see ⭐ FINAL SYNTHESIS for the verified ranges): capacity 24.2 ≈ flat / resolution 25.6 ≈ flat / DATA 28.8 = modest real +3. Data is the only clear 1-push lever; the rest are within noise.** For 2-push search, prefer the E4 checkpoint.

### E4 — more data (the data lever for the lost core)   [TRAINING, 2026-06-06 ~04:40]
- **Hypothesis (H3):** E2 left the ~36% no-signal "lost core" of hard scenes unchanged → it's data-limited, not arch-limited. ~3.6× more (hard-inclusive) data should raise positive-separation% and hard @1/coverage. *Uses the E2 architecture — isolates the DATA variable.*
- **Data:** 80k random pkls from the full 211k solvable TRAIN pool (le10 was 22k). mask-gen array `55584295` (27 shards, reusing the tested `run_batch_collection_smoke`, extra= empty = same config as le10) + per-episode labels `55584296`. Then build H5 (--tight-only) → join (build_scorer_dataset) → add_contact_px → train E2-arch. ~75-100k episodes expected (~3.5× le10). Leak-check vs test before training.
- **Test:** train E2-arch on E4 data, eval vs E2-on-le10. ACCEPT if positive-separation% ↑ and hard @1 ↑.
- **⚠️ CRITICAL REALIZATION (logged before training, 2026-06-06 ~04:40):** E4's difficulty mix is **4% hard / 21% med / 76% easy** (the pool's natural distribution), vs le10's hard-enriched 34/46/19. Because **le10 = all pkls that have a ≤10% episode, le10 already contains EVERY hard training scene** — random sampling can only add easy/med, NOT hard. So E4 has *fewer* hard episodes than le10 (~4k vs ~9.5k). → E4 actually tests "more *total/diverse* data (with the easy-skew)", NOT "more hard data". And it implies the lost core **cannot** be fixed by more data — *we already have all the hard data* → the lost core is most likely **feature-limited** (the 5 masks don't determine which push opens the path; needs richer features like the wavefront distance field, which this H5 lacks). Still running E4 to see if diversity helps or the skew hurts — but the strong prior is now: data is NOT the lost-core lever; **features are.**
- E4 data: 98,387 episodes, 79,626 rooms, contact_px ✓, gt_in_valid 100%, 0 test leakage. job 55588491.
- result: PENDING (training).

### E2b — scale the winner (capacity lever)   [DONE — H REJECTED]
- **Hypothesis:** E2 (4.3M) capacity/training-limited; a bigger edge model (dim 256, depth 6/6, 2× params) improves further.
- **RESULT — REJECTED.** E2b val_loss converged *worse* (0.636 vs E2's 0.594), and on the metric it is **no better: hard @1 24.2 vs 24.0** (flat), med 64.6 < 70.7, easy 93.4 < 94.9. The bigger model does not help — likely under-trained at the lower lr, but regardless, **E2 (4.3M) is already the right size.**
- **Conclusion:** the limit is NOT capacity. With the lost core also unchanged E0→E2 (65% positive-sep), the evidence converges on **data** as the lever → E4 is the decisive test. (Negative result = result.)

### E6 — seed variance on the best setting + reachability-supervision ablation   [RUNNING, 2026-06-06 ~13:38]
- **Motivation [CLAUDE]:** hard@1 on n=413 has ±3-4 single-checkpoint noise (E2 ckpts 24.0/25.4/25.4/27.4), so the "E4 data +3" needs a real error bar (seeds), not point estimates.
- **[USER] decision:** run a few seeds of the best setting we had (E4 = edge_crossattn on 3.6× data) to get variance on the performance.
- **[USER] hypothesis (reachability):** supervising the model to predict reachability (BCE on *all* 300 cells, pushing unreachable→0) helps it interpret the robot-region channel and understand what a good push is — so removing that supervision should *hurt*.
- **[CLAUDE] design:** 3 baseline seeds (BCE-on-all) + 3 ablation seeds (BCE on *reachable-only*), all on E4 data, same fixed room split (seed 0), vary only model-init seed (1/2/3). Implemented via a `bce_reachable_only` flag (`classifier_module.py`) + configurable `seed` (`train_classifier.py`); unit-tested loss finite for both, Hydra overrides dry-run-checked. Jobs: baseline 55606688/90/92, ablation 55606689/91/93 (gpu,gpu-redhat, 1 GPU each, all RUNNING — no queue).
- **Pre-registered predictions (the bet, stated before results):**
  - **[USER]:** reachability supervision matters → ablation (reachable-only) is **clearly worse** (the "it helps the model read the robot region" position).
  - **[CLAUDE]:** **small effect**, |baseline − ablation| ≤ ~2 pts hard@1 either way (lean slightly "helps but small"); rationale — the reachable region is *already an input channel*, so the ablation only removes the *supervision* to predict it, not the information itself.
  - **Resolution rule:** mean±std hard@1 over 3 seeds each; "clearly worse" = baseline−ablation > 2 pts with non-overlapping spread.
- **Test:** eval all 6 converged ckpts on the fixed test set; report baseline vs ablation mean±std.
- **RESULT (2026-06-06 ~16:20, evaled at past-peak best ckpts — both groups overfitting, val rising, so best ckpts are final):**
  - baseline BCE-on-all: hard@1 **[31.2, 28.1, 28.6] → 29.3 ± 1.7**
  - ablation reachable-only: **[29.5, 26.2, 27.8] → 27.8 ± 1.7**
  - **⚠️ CORRECTED VERDICT (initial "wash" was wrong — [USER] caught it).** First I compared the two groups *unpaired* (ranges 28.1–31.2 vs 26.2–29.5 overlap → looked "not significant, ~neutral"). But the design is **PAIRED**: `seed-k-with` and `seed-k-without` are the SAME init + SAME data + SAME batch order — *only the reachability loss term differs*. So the correct test is paired, per seed:
    | seed | with-sup | without | gain |
    |------|----------|---------|------|
    | 1 | 31.2 | 29.5 | +1.7 |
    | 2 | 28.1 | 26.2 | +1.9 |
    | 3 | 28.6 | 27.8 | +0.8 |
**All 3 seeds: with > without, by +1.5 ± 0.6 (paired, t≈4.3).** → **reachability supervision gives a small but CONSISTENT +1.5 — it helps. [USER]'s intuition was right; [CLAUDE]'s "wash" was wrong.** The unpaired view masked it because the **seed-wobble (±1.7) is bigger than the effect (+1.5)** and cancels in the paired diff. (Caveat: 3 seeds, small effect → "likely real, small"; more seeds to be certain.)
  - **Methodology lesson [important]:** ALL our scorer experiments use **matched seeds** (only one knob differs) → always compare **PAIRED** (per-seed diff), never group-mean±std. Group/overlap tests hide any effect smaller than the seed-wobble. *(Fixing `resolve_all.sh` to report paired diffs.)*
  - **Two bonus results:** (1) **seed noise on hard@1 = ±1.7** (n=413) — quantifies the wobble that earlier bit the single-ckpt reads. (2) These baseline seeds = **E4's real error bar 29.3 ± 1.7**, all 3 clear E2's old ~25.6 (24–27.4) → **the "+3 from data" looks SOLID** (le10 matched-seed arm will confirm, paired). Note 29.3 > the single-ckpt E4 reads (28.3–29.8) — seed-averaging lands a touch higher.
- **⚠️⚠️ ROBUST RE-EVAL (2026-06-06 ~17:10 — the FINAL word; supersedes the +1.5 above).** The +1.5 used *single best-val ckpts*, but hard@1 wobbles **±4 within a run** across nearby epochs (measured: e4seed_s1 26.9→31.2 with flat val_loss). Seed 1's baseline landed on a lucky 31.2 ckpt → inflated the paired diff. Fix: average hard@1 over each seed's **3 saved ckpts**, then pair (`resolve_robust.sh`). Result:
  | group | per-seed (3-ckpt avg) | group |
  |-------|------------------------|-------|
  | le10 (E2-data) | 23.3 / 23.4 / 25.2 | **23.9 ± 1.1** |
  | E4 (more data) | 28.3 / 28.2 / 27.7 | **28.1 ± 0.3** |
  | reach-only BCE | 29.4 / 27.7 / 28.4 | **28.5 ± 0.8** |
  - **REACHABILITY: NEUTRAL** — BCE-all − reach-only = −1.0/+0.5/−0.7 → **−0.4 (mixed signs)**. The "+1.5 helps" was within-run ckpt noise; robust averaging removes it. **Final: reachability supervision neither helps nor hurts.** (Progression: wash→+1.5→neutral; the ckpt-averaged neutral is the truth.)
  - **DATA: SOLID +4.1** — E4 − le10 = +5.1/+4.8/+2.5, **all seeds, t≈5.2.** More data is a real, strong hard-lever (~+4). Error bars shrank ±1.7→±0.3 under ckpt-averaging — so 28.1 (not the noisy 29.3) is E4's true hard@1, and le10 is 23.9.
  - **METHOD LESSON (now standard):** hard@1 (n=413) needs BOTH **seed-averaging AND checkpoint-averaging** — single-ckpt (even paired) is too noisy (±4 within-run + ±1.7 cross-seed). `resolve_robust.sh` does both.
- decision: **DATA is the lever (+4, solid). Reachability supervision = neutral** (keep it, harmless, but it's not a lever). Next robust verdict: zoom (dual), pending its training.

### E7 — de-alias the per-edge gather (zoom, not raw 224)   [PLANNED]
- **[USER] hypothesis:** the contact-pixel aliasing (~5 pts/feature-cell, 4 corners coincident) is the 64×64 resolution; going to 224 will give each contact point its own pixel and help further.
- **[CLAUDE] analysis (the correction):** raw 224 on the *wide 0.5 m crop* does NOT cleanly fix it. The object is only ~12% of the crop, so "1 point per feature cell" needs a feature grid ≥ ~125 across (15 pts / 0.12) → ~15k tokens → compute-infeasible (E3-fine at 64/patch2 = 1024 tokens already OOM'd at batch 128). 224-input *does* sharpen the rendered object (27 px vs 12.5 px) — a real, untested effect — but 224+patch4 is still ~2 pts/feature-cell and ~3k tokens (heavy). **The efficient fix is to ZOOM so the object fills the frame:** at a ~2× object-bbox crop (~12 cm), even 64–96 px + patch 4 gives ~1 pt/cell (grid ≥ 2.5·crop_cm ≈ 30) — cheap (~256–1024 tokens). Best design = **dual input**: keep the 0.5 m crop feeding scene tokens (context via cross-attn) + gather per-edge *local* features from the zoom crop (sharp, de-aliased). Data is cheap — 224 masks are object-centered, so the zoom is a center-crop+resize, no re-simulation.
- **Pre-registered predictions:**
  - **[USER]:** it will help further (resolution is the lever).
  - **[CLAUDE]:** worth testing (real measured flaw + the *sharper-object* mechanism is untested — E3-fine only sampled the blurry 64 finer), but I bet a **small gain (~+1–3 hard@1)**, not a breakthrough — because (i) E3-fine (5→2.5 pts/cell) was ~flat, and (ii) the geom-oracle says the deeper bottleneck is clutter-aware push *outcomes* / task sparsity (~2 valid of 63), not local edge sharpness.
  - **Resolution rule:** zoom/dual-crop hard@1 vs E4 baseline, ≥3 seeds (noise is ±3-4); ACCEPT if >+3 beyond the seed spread.
- **[CLAUDE] build (2026-06-06 ~14:40) — DONE, gated.** Engineering choice **[CLAUDE]: a `use_zoom` TOGGLE in `EdgeCrossAttn`, not a new file** — reproducibility-gated (use_zoom=False is byte-identical: E2 ckpt loads 0/0, +0 params; use_zoom=True +1.1M). Verified config **[USER instinct → CLAUDE sized it]: zoom window 0.24 m @ 128 px** (not raw-224): sanity viz (`viz_zoom_matching.py`) showed wide↔zoom match 0.0000 px, 100% contacts in-bounds, de-alias **4.2 → 1.0 pts/feature-cell**, object 1.6%→17% of frame. Dual H5 via `add_zoom_crop.py` (center-crop the 224 masks → context_zoom 5×128×128 + contact_px_zoom; smoke 200 rows miss=0). End-to-end pipeline smoke passed (5.4M params, loss computes). Full dual H5 build = job 55608957 (CPU). Next: GPU-smoke for OOM/batch, then 3 dual seeds vs E4 baseline.
- **[CLAUDE] launch:** 3 dual seeds 55609443/44/45 submitted with `--dependency=afterok:55608957` (auto-fire when the build's gate-asserts pass — clean hands-off chain). **batch_size=64** (the 128-px zoom stem is heavier; 64 is safely under E3-fine's OOM line; minor confound vs E4-baseline-at-128 — re-run matched if the signal is promising). Bet stands: [USER] helps / [CLAUDE] small (+1–3).
- **[CLAUDE] ops note:** first dual-build (job 55608957, `main-redhat`) was **preempted at ~59%** → H5 incomplete (no `n_samples` attr) + the dependent seeds went `DependencyNeverSatisfied`. Fix: cancelled the stuck seeds, resubmitted the build on the **non-preemptible `main`** partition with `--requeue` (job 55610210), re-chained dual seeds 55610211/12/13. Evidence it was preemption: `sacct` State=PREEMPTED
  + the `.out` epilog line "CANCELLED ... DUE TO PREEMPTION". **Lesson [per USER]: use `main-redhat` as the PRIMARY CPU partition (big pool, starts instantly) + `--requeue` to survive preemption — do NOT switch to plain `main` (it can leave you queued). The current rebuild on `main` already got a CPU, so it stays; future CPU jobs → `main-redhat --requeue`.**
- result: REBUILDING (main) → dual seeds queued (dependency). **[SUPERSEDED by E8/E9 — the zoom was compute-bound (1024-token self-attn, no HACMan analog) AND the deeper question (do we need the local gather at all?) reframed it. Zoom parked; pursuing edge-differentiation instead.]**

### E8 — is the per-edge local GATHER needed at all?   [RUNNING, 2026-06-06 ~20:30]
- **Chain of thought (how we got here):** [USER] asked to compare *exactly* to HACMan. Reading their code: HACMan = PointNet++ U-Net over a goal-conditioned point cloud (SA local grouping → global max-pool → FP upsample w/ skips); points are explicit **coordinates** (no rasterization → no aliasing). [CLAUDE] realized our per-edge gather reads a **coarse 16×16 map where ~5 edges share a cell** (aliased), and HACMan never has this because it stays in coordinate space. [USER] pushed: *"isn't the gather important? HACMan has everything in one coordinate system."* → key insight: HACMan's global step (`max_pool`) is a **bottleneck that destroys local info**, so they MUST re-inject it via the SA gather; **our** global step (cross-attention) has **no bottleneck** — every scene token stays available — so the edge can reach its local content via cross-attention too. ⇒ the explicit gather may be **redundant**, not because local doesn't matter, but because cross-attn already provides it.
- **[USER] hypothesis:** the gather (local link) matters. **[CLAUDE] hypothesis:** cross-attn + coord recovers it → no-gather ≈ gather.
- **Design [CLAUDE]:** `use_local=False` (edge = positional-id + cross-attn only, no gather). 3 seeds on E4 data, paired vs e4seed (gather). `local_proj` omitted → no-gather ckpt auto-detected at eval.
- **Pre-registered:** [CLAUDE] within ±1.5 of gather (gather redundant); [USER] no-gather worse (gather needed). Jobs 55628182/83/84. result: PENDING.

### E9 — edge DIFFERENTIATION (sharp positional id): the dominant-error lever   [RUNNING, 2026-06-06 ~20:35]
- **Chain of thought:** [USER] asked *"is the positional-id enough to differentiate two nearby edges?"* [CLAUDE]: with the **raw-coord MLP** we use — probably NOT. (1) MLP **spectral bias**: nearby coords → near-identical encodings; (2) at each corner **two DISTINCT edges (different push faces) share an identical contact pixel** — pairs (0,59),(1,31),(28,58),(29,30), verified 0.000 px apart in ALL 98k training samples — so a coordinate-only id is provably unable to separate e.g. a top-face push from a left-face push at that corner → literally indistinguishable by coordinate. **MEASURED cost (20k episodes, 2026-06-07):** the two edges in each corner pair are byte-identical model inputs (same coord→same id, same pixel→same gather) AND have DIFFERENT push directions (paired-midpoint rule, edge_mapping.py L18-20), so they genuinely differ. When a corner contact is relevant (~24% of episodes), **~75% of the time EXACTLY ONE of the pair opens the path** → without `embed` the model MUST tie them and gets one wrong every time. But it's only 8/60 edges → `embed` fixes a real but BOUNDED slice; `fourier` (spectral bias) + `fine_stem` (gather aliasing) handle the other 52 distinct-but-nearby edges. Edge indexing == motion-primitive edge_idx (generate_rectangular_edge_points) — same edges we execute. And E2's residual error is **~88% wrong-EDGE** — so edge-differentiation is plausibly the real bottleneck, bigger than gather/zoom.
- **Literature [CLAUDE, fanned out 4 Sonnet agents — see Reading list]:** strong convergence — • positional encoding: **Fourier features** (Tancik 2020; NeRF) + **per-element `Embedding(60)`** (ViT) is the textbook fix; the embedding *guarantees* distinct ids (fixes the corners). • per-location action nets (HACMan, VAT-MART, Where2Act, Transporter, Spatial-Action-Maps) **all** add an explicit contact-point positional encoding for exactly this; "load-bearing" for nearby candidates. • aliasing fix = sample a **fine** feature map (RoIAlign/PointRend/Deformable) or drop the gather. • NAMO/search: a learned per-push scorer used as a **value function for multi-push search is a validated pattern** (Visual Foresight Trees, MORE, Bejjani RHP) — confirms our 2-push direction.
- **[USER+CLAUDE] hypothesis:** edge-differentiation is the dominant bottleneck → a **sharp id** (Fourier
  + per-edge embed) lowers wrong-edge% and raises hard@1.
- **Design [CLAUDE]:** a 2×2 (id ∈ {raw, sharp} × gather ∈ {on, off}) + component ablation, all 3 seeds, E4 data, paired + ckpt-averaged eval, lean gpu-redhat (max parallel per [USER]):
  | run | id | gather | tests |
  |-----|----|--------|-------|
  | e4seed (done) | raw | yes | baseline |
  | nogather (E8) | raw | no | gather needed? |
  | **fourier** | Fourier | yes | does Fourier alone help? |
  | **embed** | +embed | yes | does per-edge identity alone help? |
  | **sharp** | Fourier+embed | yes | the lit hybrid |
  | **sharpng** | Fourier+embed | no | sharp id *without* gather (the HACMan-true design) |
  | **finegather** | raw | yes (FINE 32×32) | de-aliased gather alone (aliasing-agent's fix) |
  | **sharpfine** | Fourier+embed | yes (FINE 32×32) | sharp id AND sharp gather together |
Implemented via `pos_fourier`/`use_edge_embed`/`use_local`/`fine_stem` flags (eval auto-detects all). Unit-tested all 7 variants + baseline-repro (default loads e4seed 0/0). Jobs 55630676–87 (12) + nogather 55628182–84 (3) + fine 55631655–60 (6) = **21 jobs**.
- **[CLAUDE] note on running fine-stem NOW (not as conditional phase-2):** [USER] said "use as many GPUs as possible, everything in parallel." The aliasing literature (RoIAlign/PointRend/Deformable-DETR) says a COARSE 16×16 gather bilinearly mixes neighbouring edges' content (aliasing) → the de-aliasing fix is to gather from a FINE map. Rather than wait for the sharpng-vs-sharp verdict to *decide* whether to test it, we run it in parallel — GPUs are free, so the 2×2 (id × gather-sharpness) all-at-once is strictly more informative than staged. `fine_stem`: one stride-2 conv → 32×32 feature map, gather at the same wide contact coord (+0.01M params).
- **Pre-registered predictions:** [CLAUDE] **sharp ≥ +2 hard@1 over baseline** and **wrong-edge% drops** (high confidence — lit + the 88% wrong-edge); embed alone likely helps most (fixes corners + identity); if **sharpng ≈ sharp** → gather redundant even with sharp id (→ adopt the simplest no-gather model); if **sharp > sharpng** → local matters → expect **finegather > e4seed** and **sharpfine ≥ sharp** (sharp gather beats aliased gather); if **finegather ≈ e4seed** → coarse-gather aliasing was NOT the limiter.
- **Accept/reject:** paired ckpt-avg hard@1 (3 seeds, ±1.7 noise; >+2 consistent = real) **+ wrong-edge%** from the failure decomposition (the mechanism check — the hypothesis is specifically about edges).
- result (PRELIM, 11/21 final but robust — paired ckpt-avg, t-stats high; baseline e4seed=28.1 hard@1):
  | lever | hard@1 | vs base | verdict |
  |---|---|---|---|
  | **sharp (fourier+embed, coarse gather)** | **33.2** | **+5.1 (t≈11)** | ✅ **CHAMPION** |
  | embed alone | 32.4 | +4.3 (t≈6.5) | ✅ the main driver (fixes corners+identity) |
  | fourier alone | 28.6 | +0.6 | ~neutral (only helps WITH embed) |
  | sharp+no-gather | 27.4 | sharp−sharpng=+5.8 | ❌ gather needed even with sharp id |
  | no-gather | 24.8 | −3.2 (t≈2.5) | ❌ gather IS needed |
  | finegather | 26.2 | −1.8 (t≈2.9) | ❌ de-aliasing HURTS vs coarse |
  | sharp+fine | 27.7 | sharpfine−sharp=−5.5 | ❌ fine stem hurts on sharp too |
  - **ACCEPT/REJECT vs pre-registration:** ✅ "sharp ≥ +2 & embed helps most" → +5.1 / embed +4.3, ACCEPTED. ❌ **[CLAUDE] hypothesis "cross-attn makes gather redundant" FALSIFIED** (no-gather −3.2); **[USER] was right, the gather matters.** ❌ **fine-stem de-aliasing hypothesis FALSIFIED** (finegather −1.8 vs coarse, −5.5 on sharp) — the gather matters but sharpening it via a stride-2 conv stem does NOT; coarse 16×16 patch features (which also carry context via the shared scene encoder) beat a fine map. Lesson: the win came from EDGE IDENTITY (embed/fourier), not from local-feature sharpness.
  - **CHAMPION for 2-push value fn = `sharp` (+network.pos_fourier=true +network.use_edge_embed=true).**
  - **MECHANISM CHECK (wrong_edge_compare, 2026-06-07) — pre-registered mechanism PARTIALLY REJECTED:** I predicted sharp/embed would LOWER the wrong-edge fraction. It does NOT. failure-analysis @1 / wrongE%: baseline 29.7 / 88.6 · embed_s1 31.7 / 87.6 · sharp 32.4 / 89.0. So @1 rises (+2-3) but the wrong-edge FRACTION of the residual stays ~88%. Interpretation: the edge-id levers make the model rank the correct push #1 MORE OFTEN, but WHEN it still fails it's just as wrong-edge as before — the win is "more exact hits," not "a different error character." The wrong-edge problem is largely IRREDUCIBLE at 1-push (median |valid|=2 of ~75 reachable; you must also reason WHICH edge opens the path, not just tell edges apart). **This REINFORCES 2-push search:** rank≤5 = 65%, rank≤10 = 78% → a beam that expands top-k catches the right first push ~78% of the time at k=10, then composes a 2nd push. **Design input: beam width ≈ 10.** (Caveat: wrong_edge_compare.sh aggregation dropped embed/1 sharp seed to NA — a transient race with mid-write ckpts; individual reruns confirm the pattern. Hard@1 win itself is solid via resolve_robust.)

---

## Reading list (for the user on waking)
Grounding for the per-edge / point-critic architecture we built:
- **HACMan** (Zhou et al., CoRL 2023) — the per-point Q-map + actor-critic we replicate. https://arxiv.org/abs/2305.03942 · code: github.com/HACMan-2023/HACMan (cloned at /scratch/dm1487/refs/HACMan). Read §"per-point feature extractor" + the PointNet++ U-Net (SA encoder + FP decoder w/ skips) — that's exactly what our edge-token cross-attention does, on a 60-edge cloud.
- **HACMan++** (Jiang et al., RSS 2024) — spatially-grounded *parameterized* primitives (what/where/how), the chaining we'll use for 2-push. https://arxiv.org/abs/2407.08585 · code: github.com/JiangBowen0008/HACManPP. Note it uses a **point_transformer** over the points — validates our transformer-over-edges choice.
- **Point Transformer** (Zhao et al., ICCV 2021) — local self-attention → per-point features fusing local+global. https://arxiv.org/abs/2012.09164 . This is the backbone idea of our edge self-attention.

**My recommendation for what to read first:** the HACMan per-point feature-extractor + PointNet++ section — it makes the "each edge gets its own local+global summary" idea concrete and shows why the global readout (our E0) was the bottleneck.

**E9 literature (fanned-out agents, 2026-06-06) — edge-differentiation + per-location scoring:**
- **Fourier Features Let Networks Learn High-Frequency Functions** (Tancik et al., NeurIPS 2020, arxiv 2006.10739) + **NeRF** positional encoding (arxiv 2003.08934) — the fix for the raw-coord MLP spectral bias that blurs nearby edges. Recipe: NeRF sin/cos, L≈8 bands.
- **ViT** (Dosovitskiy 2021, arxiv 2010.11929) — learned per-element embeddings for a fixed token set ≈ our per-edge `Embedding(60)` (guarantees distinct ids; fixes coincident corners).
- **VAT-MART** (arxiv 2106.14440), **Where2Act** (2103.15454), **Transporter Nets** (2010.14406), **Spatial Action Maps** (2004.09141) — per-location manipulation scorers; ALL add an explicit contact-point positional encoding; "load-bearing" for separating nearby candidates.
- Aliasing fix (if local matters): **RoIAlign** (1703.06870), **PointRend** (1912.08193), **Deformable DETR** (2010.04159), **Conv Occupancy Nets** (2003.04618) — sample a *fine* feature map, not a coarse grid.
- **2-push search direction is validated:** learned per-action scorer as a search value fn — **Visual Foresight Trees** (2105.02857), **MORE** (MCTS+self-supervised, 2202.01426), **Bejjani RHP** (1803.08100); learned-NAMO (Scholz IROS'16; Yang 2506.15380). Use beam/MCTS with the scorer as leaf value.
- Full agent reports (4) are in the session transcript task outputs.

## Deep-dive literature (6-agent fan-out, 2026-06-06) — "how do people learn such a thing?"   [USER asked]
Six Sonnet finders over distinct slices (NAMO; learn-1-step-search-multi-step; per-point affordance; nearby-location discrimination; pushing prediction; spatial-action-maps + value-fns). The through-line:

**The field's recurring recipe — and we already follow it (good news, validates the whole approach):**
1. **Spatial per-candidate scoring map in ONE forward pass**, grounded to the scene — HACMan critic map, VPG dense Q-maps (1803.09956), Spatial Action Maps (2004.09141), Transporter (2010.14406), Where2Act per-point, Contact-GraspNet (2103.14127). NOT scalar-then-index. Our 60×5 head = this. ✓
2. **Supervise by downstream task OUTCOME** (grasp succeeded / part moved / **path opened**), per-candidate BCE. Everyone else pays for the label with slow RL (HACMan, VPG, VAT-MART) or a learned image-predictor (DIPN 2011.04692, VFT 2105.02857). **We get it free from deterministic wavefront reachability — our one structural advantage; lean on it.**
3. **Use the 1-step scorer as a VALUE/HEURISTIC for search, not a one-shot policy** — VFT, MORE (2202.01426), Bejjani RHP (1803.08100), SoRB (1906.05253), Q*/DeepCubeA (2102.04518), SaIL (1707.03034). This is our 2-push plan, and it's a well-validated pattern. Classical root: Stilman's Scene-Feasibility-Graph — "does this push reconnect robot-region to goal-region?" is literally our label; his "artificial constraints" lookahead = our multi-push expansion.

**Our 88%-wrong-edge IS a localization problem; the keypoint/detection field has a deep playbook:**
- **Cross-validates E9 (running now):** explicit position features (CoordConv 1807.03247; Fourier/embed) ≈ our `fourier`/`embed` arms; anti-aliased sampling (BlurPool 1904.11486; RoIAlign; PointRend) ≈ our `fine_stem`. So the architecture levers we're testing tonight are the textbook fixes — not flailing.
- **NEW lever #1 — soft/Gaussian edge labels + focal down-weight** (CenterNet 1904.07850 β=4; Stacked Hourglass): replace one-hot CE with a label that tapers with **arc-distance** around the perimeter, so adjacent edges get partial credit and we STOP training near-neighbors hard to zero. Cheap (loss only), composes with whatever E9 winner emerges. Attacks the confusing gradient directly.
- **NEW lever #2 — two-stage coarse→fine: actionability GATE then fine score** (Where2Act 2101.02692; Graspness ICCV'21; Kloss "Contact Reasoning" 1911.03112). Gate the ~60 edges first; the fine head only has to rank the ~6 survivors → a far easier discrimination problem. Most direct attack on the 88%.
- **NEW lever #3 — continuous offset head / soft-argmax** (DSNT 1801.07372; DARK 1910.06278; Integral Pose): decouple "which edge" from "exact contact point along the face" — also mops up the ~11% wrong-DEPTH.
- Supporting: class-imbalance BCE (only ~1-5/300 positive → check pos_weight/focal γ=2); HRNet keep-high-res near the boundary; ATSS/OTA label-assignment for the shared-corner ambiguity.

**NEW lever #4 (orthogonal, big-ticket) — C₄ EQUIVARIANCE.** Our obstacle has 4 symmetric faces. An equivariant encoder (escnn, cyclic C₄) makes data from one face train all four ≈ **4× effective data** — and DATA is our one proven lever (+4.1). SO(2)/SE(2)-equivariant manipulation shows 10-100× sample efficiency (Wang ICLR'22 2203.04439; Zhu RSS'22 grasp 2202.09468; Q* invariance proof). Higher effort (new backbone) but principled and distinct from edge-id.

**For the 2-push search (the goal):** scorer = BOTH policy (top-k branching) AND value (leaf re-score), the AlphaZero pattern (MORE/VFT). We DON'T need a learned transition model (VFT/DIPN do) — we have MuJoCo: expand top-k edges → simulate → re-score from new state → check reachable. Bejjani: cap depth, price the tail with the scorer. SoRB/Q*: edge cost = −log P(scorer), log-probs ADD → principled multi-push combination; for A*-style use, a **ranking loss beats MSE** (search needs correct order, not calibrated magnitude). Goal-condition the scorer + **HER relabel** (1707.01495): a push that opened a *different* region is a free positive for that region — fills intermediate-subgoal data during search with zero new collection.

**Ranked next experiments (impact × cheapness), after E9 names a winner:**
1. soft Gaussian edge labels + focal (loss-only, composes with E9 winner) — cheapest shot at the 88%.
2. two-stage actionability gate → fine head — shrinks the discrimination set; biggest direct attack.
3. C₄-equivariant encoder — 4× data for free; ties to the proven data lever.
4. ranking/list loss (search-friendly) + continuous depth-offset head — better for the search use + the 11%.
5. then build 2-push search: top-k expand → MuJoCo sim → re-score → reachable, w/ −log P edge costs.

Full 6 agent reports are in the session transcript task outputs.

### E-oracle — geometric+wavefront oracle: is the lost core feature/FOV-limited or model-limited?   [RUNNING]
- **Sharpened hypothesis (the big one).** The scorer's input is ONLY the **tight 0.5 m crop** (confirmed: scorer H5 `ctx` = 5×64×64 from `local_tight_*`, `crop_size_meters`=0.5). So it reasons about "does a path to the goal open" through a 0.5 m keyhole around the object. **Hypothesis H5: the lost core is a field-of-view problem** — for ~36% of hard scenes the corridor that opens when you move the object lies (partly) outside 0.5 m, so the answer is *not in the crop* and NO model on this input can recover it. The competing hypothesis H5′: the answer IS in the crop and the model just can't extract it (arch/res).
- **Decisive test — `scripts/sandbox/geom_oracle.py`.** A PERFECT geometric reasoner that uses only the crop the model sees. Per candidate (edge, depth): look up the object's SE(2) displacement from the calibrated `1x_car_d5` primitive .dat → move the target footprint (rigid) → obstacle = static ∪ other-movables ∪ moved-target → inflate by the robot radius (the wavefront's obstacle inflation) → 8-conn flood from the robot region → "opens" iff goal-sample region joins the robot's component. Predicted-valid vs TRUE-valid (from sim) gives the oracle's recall.
  - **Decision rule:** oracle hard-recall ≈ easy-recall → info is in the crop → **model-limited** (push resolution E3 / better arch / explicit wavefront feature). Oracle hard-recall collapses → crop lacks the answer → **FOV/physics-limited** → bigger crop (wide 1.2 m — data must be regen, none exists yet) or full-map features / multi-push.
- **Every assumption gated before trusting the number:** G1 render-IoU (rendered object pose matches the real target mask) — smoke **0.948** ✓. G2 nopush-IoU (flood reproduces `robot_region`) — first naive flood **FAILED at 0.219** (leaked through 1-px gaps a 7 cm car can't fit) → **fixed** by calibrating an obstacle-inflation radius r* to reproduce `robot_region` (this is exactly the wavefront's inflation). G3 easy-recall must be high or the oracle itself is broken.
- **RESULT (tight crop) — initially read as FOV-limited; ⚠️ THIS READ WAS WRONG, corrected below by the wide-crop oracle + the model-beats-oracle check. Read the CORRECTION before citing anything here.** Full test set (n=413/491/752), both gates pass (G1 render-IoU **0.952**, G2 reproduce-robot_region **0.815** at calibrated r*=20px ≈ 4.5 cm robot radius). A *perfect* geometric reasoner with the 0.5 m crop:

  | bin  | oracle recall | oracle precision | base-rate | goal clipped by crop | misses via off-crop path |
  |------|---------------|------------------|-----------|----------------------|--------------------------|
  | hard | 67.3%         | **5.9%**         | 2.7%      | 26.9%                | **89.4%**                |
  | med  | 65.7%         | 29.6%            | ~16%      | 25.7%                | 89.8%                    |
  | easy | 85.6%         | 81.9%            | ~65%      | 19.5%                | 88.0%                    |

Reading: even with perfect geometry, the 0.5 m crop caps hard recall at 67% and gives precision barely 2× the random base-rate (within-crop connectivity is a near-useless signal on hard). The oracle's own precision collapse easy→hard (82%→6%) mirrors the *model's* easy→hard collapse — strong evidence the model's hard difficulty is **information-limited by the crop, not capacity/data-limited.** And ~89% of the oracle's missed valids are explained by the goal connecting through an **off-crop** path.
- **Honest caveats:** the oracle approximates (rigid single-object motion, within-crop reachability with ~18% modeling error per G2, sampled goal region) → absolute recall ±~15%. But the *pattern* (hard precision collapse + off-crop dominance of misses) is gate-validated and robust.
- **Falsifiable PREDICTION (pre-registered):** because E2b (capacity), E3-fine (resolution), and E4 (data) ALL keep the 0.5 m FOV, the oracle predicts **all three are ~flat on hard@1** (≈ E2's 24). E2b already confirmed (24.2). If E3/E4 also land flat on hard, that triple-confirms FOV as the bottleneck.
- **Decision → the lever is FIELD OF VIEW.** The wide 1.2 m crop already exists in the le10 NPZs (`local_wide_*`, all 5 channels, object centered to <3 mm — verified). Two routes:
  - **E5-wide** (wide-only): isolates the FOV variable. *Caveat measured:* in a 1.2 m crop the object spans only ~11 px (vs ~27 px tight) so the 60 edge contact points squish into ~11 px → per-edge gather loses edge resolution. So wide-only trades FOV-gain against edge-precision-loss (ambiguous).
  - **E5-dual** (tight+wide, like the diffusion's dual-crop): tight encoder for edge precision + wide encoder for FOV context, per-edge feature gathered from BOTH. The principled fix; more model surgery.
  - **Cleanest model-free check first:** run THIS oracle on the WIDE crop. If wide-oracle hard recall/precision ≫ tight, the answer is in 1.2 m → spend GPU on E5. If not, even 1.2 m is too small → need full-map / wavefront-distance features or genuine multi-push. (Pipeline `--crop wide` verified.)

- **⚠️ CORRECTION (the wide-crop oracle ran — it FALSIFIES the FOV conclusion).** Ran the same oracle on the 1.2 m wide crop (built `v3_test_*_lzf_both_data` from the test NPZs; gates pass G1 0.887, G2 0.832 at the correctly re-calibrated r*=8px = the same ~4.5 cm radius in coarser px). Result vs tight:

  | bin  | recall (tight→wide) | precision (tight→wide) | goal clipped (tight→wide) | misses off-crop (tight→wide) |
  |------|---------------------|------------------------|---------------------------|------------------------------|
  | hard | 67.3 → **68.8**     | 5.9 → **5.8**          | 26.9% → **1.7%**          | 89.4% → **3.1%**             |
  | med  | 65.7 → 67.5         | 29.6 → 29.3            | 25.7% → 1.4%              | 89.8% → 8.6%                 |
  | easy | 85.6 → 87.5         | 81.9 → 81.0            | 19.5% → 2.0%              | 88.0% → 16.0%                |

The wider crop **removes the goal-clipping** (26.9%→1.7%) and the off-crop misses (89%→3%) — the goal is now fully in view — **yet recall and precision on hard are UNCHANGED (68.8/5.8 ≈ 67.3/5.9).** So goal-visibility was NOT the binding constraint. The "89% off-crop" attribution in the tight read was a *proxy artifact* (goal-touches-border ≠ answer-is-off-crop): the same scenes stay unsolved at full FOV.
- **⚠️ AND the oracle is a WEAK BASELINE, not a ceiling — the trained model already beats it.** The geometric oracle's hard *precision* (≈ its success@1 as a ranker, since it emits binary open/closed) is **~6%**; E2 gets **24% hard@1** and **89.8% hard@20** (the oracle as a random pick among its ~34 "opens" would be ~6%@1 and only ~40%@20). So E2 extracts **~4× more** signal from the SAME crop than the rigid free-space oracle. The oracle over-predicts wildly on hard (~6% precision = it thinks half of all pushes open the goal) because rigid free-space motion ignores clutter (the object stops early / nudges neighbors). **The model has learned clutter-aware push outcomes the oracle cannot.** ⇒ this oracle **cannot upper-bound** the model; my "FOV-limited lost core" claim is **rejected**.
- **What actually holds (durable, honest):** (1) ~27% of hard goals ARE clipped by the 0.5 m crop — a real but *secondary* FOV limitation (fixing it didn't move the needle). (2) hard@1 sits mid-to-high 20s; capacity (E2b) and resolution (E3) are flat within noise, FOV is not the lever (this), but **data scaling (E4) gives a modest real ~+3** (see corrected E4 entry — my "E4 flat" was a mid-training read). Remaining difficulty is **genuine 1-push hardness** (median |valid| ~2 of ~63). (3) E2 is a strong converged 1-push scorer: beats diffusion ~4×, the random floor ~9×, AND a calibrated geometric oracle ~4× on hard@1. (Note: the "~6%" / "24%" / "89.8%" figures in the two bullets above are single-checkpoint reads; the per-model ranges are in the ⭐ FINAL SYNTHESIS at the top — the oracle-vs-model gap holds.)
- **Lesson logged:** a cheap proxy (goal-touches-border) over-attributed the cause; the decisive move was the *direct counterfactual* (actually widen the FOV and re-measure) + checking the baseline against the model. Pre-register the falsifier, not just the confirmer.

### E4 — more data (the data lever)   [DONE — H3 partially ACCEPTED: modest real gain]
- **⚠️ CORRECTED RESULT.** First prelim (epoch017, mid-training) read hard@1 = 24.0 and I wrongly called it "flat, data not the lever." Evaluating **converged** checkpoints overturns that: epoch025 **29.8**, epoch029 **28.3**, last **28.3** — converged E4 ≈ **28.8 hard@1**, and **all three converged ckpts exceed E2's best (27.4)**. So **3.6× data gives a modest REAL gain of ~+3 hard@1** over E2 (25.6). Med 80.4 / easy 99.5 also up. Caveat: E4's hard@**20** is ~87.5 (slightly *below* E2's ~89.5) — the data gain is in top-1 sharpness, not coverage. Mechanism: the gain is NOT from new hard scenes (le10 already had them all) but from a better learned **representation** off 3.6× more diverse (easy-skewed) data. ⇒ data *scaling* (H3, 10×) is a live lever worth pursuing; **lesson: I evaluated mid-training and called it flat — always eval converged checkpoints.**

## Key open question for the lost core — RESOLVED, with a corrected answer (E-oracle, 2026-06-06)
Hypothesis history (each step tested — this is the audit trail; see the ⭐ FINAL SYNTHESIS at top for the verified multi-checkpoint numbers that supersede the noisy point estimates below):
1. data-limited? → **partially YES (modest).** Converged E4 (3.6× data) ≈ 28.8 hard@1 vs E2's 25.6 — a real ~+3 via better representation. (My first read "flat 24.0" was a *mid-training* artifact — corrected.)
2. capacity-limited? → **NO.** E2b (2× params) hard@1 24.2 ≈ E2 (within noise).
3. FOV-limited? → **NO.** Tight oracle *suggested yes* (89% misses "off-crop"), but the **wide-crop oracle FALSIFIED it**: widening to 1.2 m removes goal-clipping (27%→1.7%) yet hard recall/precision don't move.
4. resolution-limited? → **NO.** E3-fine (patch=2) 25.6 ≈ E2 (within noise).
5. Is the geometric oracle even a ceiling? → **NO** — the trained model beats the oracle's ~6% hard precision ~4×. It's a weak rigid-free-space baseline, not an upper bound.

**Corrected answer:** hard@1 sits in the **mid-to-high 20s** and the only lever that clearly moves it is **data scaling** (modest ~+3 from 3.6×; 10× is worth trying). Capacity / resolution / FOV are flat. Hard is genuinely hard for 1-push (~2 valid pushes of ~63) and the model already extracts more than rigid geometry. The route toward the **objective** is **2-push search over the best scorer (E4)** — hard scenes get solved by *composing* pushes (~88% hard@20 = the right first push is almost always in the top-20 a search would expand). Parallel 1-push lever: **scale data** (the live one). Lower-priority: dual-crop (the real ~27% goal-clipping) and clutter-aware push-outcome features (the rigid oracle's failure mode).

---

## 🎯 2-PUSH (deployable) — overnight autonomous run [2026-06-07, USER asleep 8h]
**[USER] mandate:** "have a 2-push model that ALSO works on 1-push, deployable to a real robot tomorrow. Go crazy, parallelize, stick to hypothesis→result, use the research."

**[CLAUDE] framing (the thesis):** there is NO separately-trained 2-push model and NO exhaustive 2-push labels. "2-push" = **depth-≤2 search using the 1-push scorer as the value function**, with **MuJoCo as the (real) transition model** (we don't need a learned one — VFT/DIPN do because they lack a sim; we have one). Search checks depth-1 first ⇒ returns a 1-push plan when one suffices, 2-push otherwise ⇒ ONE system does both. Grounded in VFT (2105.02857), MORE (2202.01426), Bejjani RHP (1803.08100), Stilman SFG.

**Key algorithm trick (no 2-push labels):** the scorer predicts P(this push opens path to GOAL), so a 2-push chain's FIRST push scores ~0 (it doesn't open the goal alone). To rank first pushes, use a one-ply Bellman lookahead with the SAME scorer: **V(s₁) = max over second-push scorer(s₁)** after simulating the first push. Expand first pushes by V(s₁). Edge cost = −log P so probs compose (SoRB/Q* 2102.04518).

**Plan (each step hypothesis→result):**
1. **Live-scorer bridge** (GATING): env state → 5-ch tight crop + contact_px → scorer → (60,5) P. Validate vs H5 (crop MAE) AND functionally (does scorer top-k contain the known solving push; live recall@k ≈ H5 success@k). If the live crop ≠ training crop, everything downstream is poisoned — so this gates.
2. **scorer_beam planner** (BasePlanner): depth-1 (sim top-K1, terminal=is_robot_goal_reachable) → 1-push; else depth-2 (re-score at s₁, V(s₁), sim top-K2) → 2-push. Reuse region_opening enumeration + env.step.
3. **Eval**: % solved ≤1 push (baseline) vs ≤2 push on room-held-out test, per difficulty. **Pre-registered prediction [CLAUDE]:** depth-2 lifts HARD solve-rate substantially (88% hard@20 says the right first push is almost always in a small top-k a search expands); easy/med already near-solved at depth-1. Accept the 2-push lever if hard(≤2) ≫ hard(≤1).
4. **Deployable inference**: scene+goal → plan of (edge_idx, push_steps) — verified to map to the real motion primitives (1x_car_d5, 60 edges×5 depths, edge convention matched to generate_rectangular_edge_points).
5. **Research levers in parallel (compute is free):**
   - **E9 resolution** → champion 1-push scorer (the beam's value fn); swap champion in once known.
   - **Soft Gaussian edge labels + focal β=4** (CenterNet 1904.07850) — cheap loss-only lever, composes with E9 winner; 3-seed retrain, paired on hard@1 + wrong-edge%.
   - stretch: ranking loss (search-friendly value), C₄-equivariance (4× data, escnn), MORE self-distill.

**Risks logged:** (a) live-crop fidelity (gated in step 1); (b) wavefront warmup before get_reachable_edges/ is_robot_goal_reachable (must run a skill/snapshot first); (c) first-push ranking for 2-push needs the V(s₁) lookahead, not the raw s₀ scorer. result: IN PROGRESS.

### 2-push progress log [2026-06-07, autonomous]
- **E9 champion picked:** `sharp` (fourier+embed) 33.2 hard@1 (+5.1). Best ckpt sharp_s1 epoch017 (val 0.2713).
- **Step 1 — live-scorer bridge: GATE PASSED bit-for-bit.** scripts/sandbox/live_scorer.py renders the 5-ch tight crop from the LIVE env and runs the scorer. vs H5: crop 0.0 MAE on ALL 5 channels, contact_px 100% <0.5px, functional recall live==H5 bit-for-bit (hard 50/55/68, med 77/95/95, easy 100). CRITICAL config: `namo_config_complete_skill15_car_1x.yaml` (robot 0.035, motion_primitives_1x_car.dat) — NOT namo_config_car.yaml. Region masks via Python wavefront exporter → works at arbitrary mid-search states.
- **Step 2 — scorer_beam search: BUILDING** (scripts/sandbox/scorer_beam.py). Depth-≤2: depth-1 = top-K1 by scorer P, sim, terminal=is_robot_goal_reachable → 1-push; else depth-2 = reuse saved s1, V(s1)=max second-push P (1-ply Bellman), expand by V(s1), sim top-K2 → 2-push. K1=K2=10 (rank≤10=78% finding). Champion sharp as value fn. Eval on test_{1,2}push_solvable_combined manifests (539 / 1186 scenes): **prediction — depth-2 lifts the 2-push-solvable solve rate FAR above depth-1.** Verified-by-sim only.
- **Research lever in parallel:** soft-label retrain (sharp+CenterNet soft edge labels, σe=1.0/σd=0.7) 3 seeds training (55656001-03); flag-gated (off==identity, E9 reproducible); resolve_robust wired for the paired comparison vs sharp.
- E9 final resolver: 4 jobs left → confirms prelim.

### E9 FINAL (all 21 done) + soft-label caveat [2026-06-07]
- **FINAL resolver CONFIRMS the prelim exactly:** sharp 33.2 (+5.1, t≈11), embed 32.4 (+4.3), fourier +0.6 neutral, no-gather −3.2, finegather −1.8, sharp+fine −5.5. Champion = **sharp**, locked.
- **soft-label is INCONCLUSIVE (undertrained):** sharp+softlabel showed 21.9 (−11.3) BUT those jobs are at epoch 6-7 vs sharp's converged 17-21 — invalid apples-to-oranges. ~6 min/epoch (soft-target build adds overhead); converges in ~3-4h. Re-resolve on convergence. DO NOT conclude soft-label is bad yet.
- **[CLAUDE] methodological correction (important):** hard@1 is the WRONG metric to judge the beam's value function. The beam expands **top-k**, so what matters is **recall@k** (is the correct push in the top-10). Soft labels deliberately blur top-1 → may HURT hard@1 while HELPING recall@10 — exactly what search uses. So judge softlabel (and pick the beam's scorer) on **recall@10 / actual beam solve-rate**, not hard@1. The beam currently uses sharp (best hard@1 AND, from the mechanism check, rank≤10=78%); re-evaluate vs softlabel on recall@10 once it converges.

### 2-push search: BUG found → fixed [2026-06-07, autonomous]
- **First build (a038... agent): search runs but finds 0 two-push solutions.** Validation on ~30 2-push-solvable scenes: 0 verified depth-2 (it DID find depth-1 solutions env_0000/env_0119 at P=1.0, terminal check works; ~110 sims/scene, no crash). So mechanically correct, but the 2-push capability — the whole point — was absent.
- **Diagnosis (from the code, scorer_beam.py:140 original):** depth-1 took the top-K1 first pushes ranked by the s0 scorer P, and **depth-2 only reused those same K1**. But a 2-push chain's FIRST push opens nothing yet → its P≈0 → it sits at the BOTTOM of the P-sorted pool → never simulated. The beam was STRUCTURALLY blind to 2-push first moves. This is EXACTLY the risk pre-registered ("first-push ranking for 2-push needs V(s1), not the raw s0 scorer"). [USER hypothesis-style: confirm before fixing — done via code + the 0/30 validation.]
- **Fix [CLAUDE]:** decouple the two depths. Depth-1 still uses top-K1 by P (P IS predictive for 1-push solutions — fast path). Depth-2 now sweeps a **broad first-push budget that is NOT P-ranked** (`_first_budget`: all reachable obj × reachable edges × first_depths {4,3,2}, capped at max_first=60), simulates each, then ranks ALL simulated first pushes by **V(s1)=max second-push P** (one-ply Bellman), and verifies the top-N1 first pushes × top-K2 seconds. Verified-by-sim throughout.
- **Cost:** ~max_first first sims + a render per s1 (for V) + N1×K2 second sims → ~2-4 min/scene (vs ~1 min broken). Acceptable for deployment-class planning.
- **TEST (running, job 55661355):** fixed solver on 4 one-push + 10 two-push scenes (incl. ones UNSOLVED by the broken version). **Pre-registered prediction: now finds verified depth-2 solutions on the 2-push set.** If 0 again → escalate (check 2-push-solvability of these scenes by primitives; terminal check at s1).

### 2-push search WORKS + collision investigation [2026-06-07]
- **Quicktest (both fixes, 6 two-push-solvable scenes): 5/6 SOLVE at DEPTH-1** (P=1.0, ~3s, 1 sim each; obstacle_2/3/5/0 various edges). 1 unsolved (env_0087) had bestV(s1)=0.999 — the search FOUND a 2-push path whose 2nd push scores 0.999, but the sim-verify failed → scorer false-positive at the 2nd step, not a search bug. **The deployable search works** (verified-by-sim).
- **DOMINANT bug was the COLLISION setting, not (only) first-push ranking.** Scenes UNSOLVED in the broken run (env_0068/0093/0119/0171) now solve in ONE push once collisions match training. With collisions ON, scorer-recommended pushes aborted mid-execution.
- **Collision investigation (verified the agent's fix):** `modular_parallel_collection.py:1044` `--region-allow-collisions default=True` ("strict mode ... intended for evaluation, NOT data collection"). So v3 DATA = object-collisions ALLOWED during pushes (robot-traj collisions always abort). The search's `set_collision_checking(False)` correctly matches this. (NOTE: DATA_COLLECTION_GUIDE.md:61 says default false — STALE/misleading vs the code default True. Worth a doc fix.)
- **Deployment caveat [CLAUDE, important]:** "object collisions allowed" means a push may DISPLACE other objects — on a real robot you can't push an object THROUGH another, you'd shove it. So I run the eval BOTH ways: `--collisions off` (training-match, the scorer's native solve rate) AND `--collisions on` (STRICT = real-robot-faithful). **The off→on gap = the "push-through tax"** = solutions that rely on object overlap and won't transfer to the real robot. The STRICT number is the honest deployable one.
- Eval running: jobs 55665961 (off) + 55665962 (on), n=30 per manifest each, K1=10 K2=10 N1=5 max_first=40.
- **Open finding:** many "2-push-solvable" scenes are actually 1-push-solvable by the scorer (collisions off). The depth-2 LIFT (scenes where 1-push fails, 2-push works) is what the eval will quantify — may be modest if most are 1-push. Honest either way.

### Soft-label lever: REJECTED (converged) [2026-06-07]
- sharp+softlabel converged (6/6/5 ckpts): **25.9 hard@1, −7.3 vs sharp (t≈−9.3, all seeds).**
- **[CLAUDE] pre-registered salvage hypothesis FALSIFIED:** I expected soft labels might HURT @1 but HELP recall@k (the metric the beam actually uses). They do NOT — softlabel is worse at EVERY k: sharp @1/@5/@10/@20 = 33.2/64.3/77.6/89.4 vs softlabel 25.8/59.7/72.7/86.1. The CenterNet neighbor-blurring just degraded the signal everywhere for this problem (the target really is sharp: only ~2 of ~75 pushes work, and spreading credit to non-working neighbors is pure noise here).
- **Verdict: REJECT soft-label. `sharp` remains champion AND beam value function** (best @1 AND best @k).
- Side note: embed alone ≈ sharp on recall@k (@10 77.4 vs 77.6); fourier adds little to @k, mainly a marginal @1 bump. sharp still best overall.

### 2-PUSH EVAL — HEADLINE RESULT [2026-06-07] ✅
**Training-match (collisions OFF, matches scorer training), champion=sharp, K1=K2=10 N1=5 max_first=40:**
| manifest | n | %≤1 push | %≤2 push | LIFT | t/scene | sims/scene |
|---|---|---|---|---|---|---|
| test_1push_solvable | 30 | 73.3 | 76.7 | +3.3 | 88s | 27 |
| **test_2push_solvable** | 30 | 33.3 | **63.3** | **+30.0 pp** | 94s | 44 |
- depth hist 2push: {1:10, 2:9, None:11} → **9/30 solved by genuine verified depth-2 chains**; 1-push alone gets only 10/30. **The 2-push search NEARLY DOUBLES the solve rate on scenes that need two pushes.** Pre-registered prediction CONFIRMED. Clean depth-2 example env_0413: push1 P=0.000 (setup) → push2 P=1.000 (opener) — exactly the V(s1) mechanism working.
- 1-push manifest: depth-2 adds little (+3.3) — expected, they're 1-push problems.

**STRICT (collisions ON = real-robot-faithful; any object collision aborts the push):**
- **1-push: 43.3% ≤1** (vs 73.3% off) → **~30pp "push-through tax"** — ~30% of the scorer's 1-push solutions rely on shoving THROUGH an object (the scorer was trained collisions-allowed), invalid on a real robot where you'd displace it. depth-2 added 0 here (13→13).
- Strict is ~5× slower (~148s/scene — most scenes miss depth-1 and fall to the depth-2 sweep). The full n=30×2 strict run hit the 2h SLURM limit after finishing 1-push + 10/30 of 2-push. Re-running strict 2-push (n=20, job 55685381) for the deployable-strict 2-push number.
- **Honest framing for deployment:** OFF = optimistic upper bound (pushes ignore obstacles), STRICT = conservative lower bound (pushes abort on any touch). The REAL robot is between (it displaces objects). The strict number is the safe deployable floor; the off number is the scorer's native capability.

### STRICT (real-robot) verdict + the deployment blocker [2026-06-07] ⚠️
- **Strict 2-push (n=20): 10% ≤1, 10% ≤2 (ZERO depth-2), 18/20 unsolved.** vs training-match 33%/63%.
- **Confound RULED OUT:** skill15_car_1x config has `check_object_collision: true`, `check_robot_trajectory_collision: false`. So strict aborts only on OBJECT-OBJECT collision (true push-through), NOT on car-grazing. The 63%→10% collapse is the genuine push-through tax, not an artifact.
- **deploy_plan.py works** (env_0068: training-match SOLVES 1-push edge43 push_steps2 P=0.999; strict UNSOLVED — that P=0.999 solution is a push-through). The PIPELINE + METHOD are sound; the gap is the scorer's training regime.
- **ROOT CAUSE:** the scorer was trained on collisions-ALLOWED labels (v3 region_allow_collisions=True), so it rates push-through pushes high. On a real robot (object collisions matter) those abort → solve rate collapses. The 2-push chains are hit hardest (more pushes = more chances to push through something).
- **NUANCE (don't overstate the 10%):** the test_2push_solvable manifest was DEFINED under collisions- allowed, so some of the 18 unsolved may have NO physically-valid solution at all (their defining "solution" was a push-through). So 10% conflates (a) scorer mis-trained for strict + (b) scenes not strict-solvable. Need a strict-defined benchmark to isolate (a).
- **HONEST DEPLOYMENT VERDICT:** method + pipeline + deployable entrypoint = DONE and working; training- regime 2-push lift is large and real (+30pp). But the current scorer is NOT real-robot-ready as-is — its solutions are dominantly push-throughs. **#1 next step for real deployment: re-collect/relabel data with check_object_collision=true (strict), retrain `sharp` on physically-valid labels, re-define the test benchmark as strict-solvable. Then the SAME search/bridge/deploy_plan should carry over** (they're regime-agnostic — only the scorer's training labels need to change).

---
## ⭐ DEPLOYABLE 2-PUSH — FINAL STATUS & HANDOFF [2026-06-07 ~05:30]

**What was built (all working, verified):**
1. **Champion 1-push scorer** `sharp` (EdgeCrossAttn + Fourier PE + per-edge Embedding(60)): 33.2 hard@1, +5.1 vs E4 baseline (t≈11, all seeds). Ckpt: `/scratch/dm1487/sage_outputs/scorer/sharp_s1/namo-classifier/9yizg6i8/checkpoints/epoch017-val_loss0.2713.ckpt`
2. **Live-scorer bridge** `scripts/sandbox/live_scorer.py` — env state → 5-ch crop + contact_px → (60,5) P. Validated bit-for-bit vs the H5. Config: `namo_config_complete_skill15_car_1x.yaml`.
3. **Depth-≤2 scorer-beam search** `scripts/sandbox/scorer_beam.py` — 1-push scorer as the value function, MuJoCo as the transition model, NO 2-push labels. depth-1 = top-K1 by P; depth-2 = broad un-P-ranked first-push budget ranked by V(s1)=max 2nd-push P, verified-by-sim.
4. **Deployable entrypoint** `scripts/sandbox/deploy_plan.py` — scene → executable plan `(object, edge_idx, push_steps, target_se2)`. Defaults to STRICT (real-robot).

**Results (champion=sharp, K1=K2=10 N1=5 max_first=40):**
| eval | regime | %≤1 | %≤2 | note |
|---|---|---|---|---|
| 1push_solvable (n30) | train-match | 73.3 | 76.7 | depth-2 adds little (they're 1-push) |
| **2push_solvable (n30)** | **train-match** | 33.3 | **63.3** | **+30pp — method works** |
| 1push_solvable | STRICT | 43.3 | 43.3 | push-through tax ~30pp |
| 2push_solvable (n20) | STRICT | 10.0 | 10.0 | scorer's solutions are mostly push-throughs |

**THE blocker for real-robot deployment:** the scorer was trained collisions-ALLOWED (v3 region_allow_collisions=True), so its high-P pushes often pass THROUGH objects → abort under real physics → strict solve rate collapses. Confound ruled out (robot-traj checking off; strict = object-collision only). The 10% is also a confounded lower bound: test_2push_solvable was defined collisions-allowed, so some scenes may have no strict solution at all.

**HOW TO RUN:**
- plan one scene: `python scripts/sandbox/deploy_plan.py --xml <env.xml>` (strict) `[--no-strict]` `[--goal X Y TH]`
- eval: `python scripts/sandbox/scorer_beam.py --eval --n 30 --collisions {off|on} [--manifest <path>] [--only 1push|2push]`
- env=namo_rl.RLEnvironment(xml, config/namo_config_complete_skill15_car_1x.yaml, False)

**RANKED NEXT STEPS (for real-robot deployment):**
1. **Re-collect/relabel data STRICT** (`--no-region-allow-collisions`, check_object_collision=true) and retrain `sharp` on physically-valid labels. The bridge/search/deploy_plan are regime-agnostic — only the scorer's labels change. THIS is the deployment unblock.
2. **Re-define a STRICT test benchmark** (strict-solvable scenes) to cleanly measure real-robot quality (the current 10% conflates scorer-quality with scene-unsolvability). `test_pure2push_combined.txt` is a cleaner genuine-2-push set (eval running).
3. Then: bigger beam / better first-push heuristic for strict (the scorer mis-guides under strict); data-scaling the strict scorer (data was the one proven 1-push lever, +4.1).

**Negative results (cleanly recorded, don't retry):** soft-label (−7.3, worse at every k), fine-stem de-aliasing (−1.8), reachability ablation (neutral), dual-crop zoom (−15.5).

### PURE-2-PUSH eval — cleanest 2-push-value number [2026-06-07]
- **test_pure2push_combined (genuine 2-push-only, n=25, train-match): 16% ≤1 → 56% ≤2 (+40pp).** depth hist {1:4, 2:10, None:11} → **10/25 solved by verified depth-2 chains.** Depth-1 alone is low (16%) BECAUSE these scenes genuinely need 2 pushes — exactly where the search earns its keep. Strongest validation of the no-2-push-labels search thesis. (Train-match; strict would be lower, same push-through caveat.)
