# Horizon-Q Build Journal

> **STATUS [2026-06-11]: ACTIVE BUILD, un-parked.** This is the *implementation* journal. The *design*
> spec (37 grounded decisions, each with reason + citation) lives in
> [multipush_horizonQ_journal.md](multipush_horizonQ_journal.md) → "UN-PARKING BUILD SPEC". This file is the
> resume-able operational record: state, decisions, plan, progress log, and how to pick up mid-implementation.
>
> **HOW TO RESUME (if context was compacted):** read §1 (problem), §2 (locked decisions), §3 (car/config
> saga — RESOLVED), §7 (repo state + commits), §9 (progress log + NEXT ACTIONS). Everything needed to
> continue is in §9.

---

> **MODEL REGISTRY (all ckpt paths/numbers/eval dirs): [horizon_q_model_registry.md](horizon_q_model_registry.md)** — read it, do not glob.

## 1. The problem (one paragraph)
Region Opening (RO): a 7 cm diff-drive car opens a path to a goal region by pushing ONE blocking object in a
sequence of 1–3 pushes, choosing among **300 discrete pushes** (60 contact edges × 5 depths). The sim is a
**perfect but expensive oracle** (~1 s/push). We learn a **budget-conditioned horizon-Q** `Q(s,a,H)` =
"does this push open the region within H pushes (best play)" to amortize expensive online search into a cheap
forward pass, deployed reactively (1 fwd pass) or with shallow verify-search. H_max = 2.

## 2. Locked decisions (this session, 2026-06-11) [USER]
1. **H_max = 2** (extends to 3 later).
2. **Gamma discounting** target (prefer shorter): label = `1.0` (opens in 1) / `γ≈0.9` (opens in 2) / `0`.
3. **Binary per-push success** = the 20% reachable bar (FROZEN).
4. **One head** (map = policy = value; split only if calibration tension bites).
5. **Regenerate H=1** training data under 20% bar + KEEP dead-ends.
6. **Scenes:** reuse existing v3 feb+aug9_v3 pools, generate more dead-end/2-push-rich if needed.
7. **CPU/SLURM**, no MJX.
8. **Sim-first, reactive-extensible** (keep the no-verify reactive path viable for the real car later).
9. **HACMan-style online TD baseline: PARKED** (search-distilled MC targets are the method).
- **Robustness:** train on gamma; ALSO record success-fraction per first push (re-record each ExIt round — it's
  policy-conditioned). Revises the 2026-06-10 robustness-over-optimality call (now: prefer shorter).
- **control_steps_per_push = 550** (kept, see §3).

## 3. The car / config / primitive saga — RESOLVED [2026-06-11]
**The robot model changed and we chased a confound. Final state, with provenance:**
- **Car geometry:** wheels moved to **y = ±0.034** (inside chassis edge 0.035, outer face 0.0345) + chassis↔wheel
  **contact-exclude restored**. Chassis is now the widest part (wheels can't catch obstacles). Verified in MuJoCo
  (no protrusion, settles on wheels, diff-drive works). Synced across BOTH car copies:
  - `test_xml/little-car-modeling-package/assets/mjcf/little_car.xml` (used by the motion-primitive generator).
  - `mujoco_env_creator/generate_envs.py` `DIFF_DRIVE_CAR_BODY_XML` (injected into generated collection/eval envs).
- **control_steps_per_push = 550:** this is the push DURATION. History: 375→482 (May23, regen DB)→347 (May25,
  Tier-2)→**550 (May25, e198b47 "experimental tuning")**. The 550 commit explicitly says *"exploratory, should be
  reverted before production"* and its own sweep favored **520 (8/9) over 550 (7/9)**. [USER 2026-06-11: keep 550 for now.]
- **Primitives regenerated** at 0.034 + 550: `data/motion_primitives_1x_car_d5_{square,wide,tall}.dat` (300 prims/shape,
  0 non-finite, stable). Backup of the stale set in `data/_primitive_backup_pre0034/` (also in git history).
- **THE CONFOUND (resolved):** first regen diff showed ~14% "more push reach" and an agent called it SIGNIFICANT.
  **Wrong attribution.** Controlled diff (regen at OLD car 0.0375 AND new car 0.034, BOTH at 550) shows the **pure
  car effect is NEGLIGIBLE: ±0.5% reach, sub-3 mm, ≤0.02° at every depth.** The ~14% was the **482→550 config
  change**, not the wheels.
- **⇒ TEST SET IS REUSABLE AS-IS.** `namo_testset_v1` (collected at 0.0375 + 550) ≈ the new 0.034 + 550 world within
  eval noise. **No re-collection / no re-labeling needed.** (Saved ~6 h.) Eval keys stand:
  `labels/onepush_episodes.json` (20% bar, 1-push), `labels/pure2push.json` (2-push F1′).
- **Still to do for collection:** restore `controller_stuck_threshold` 1000000→**5** in the production collection
  config (the commit's own "revert before production"; the disabled auto-abort matters for multi-push collection).

## 4. What we're learning (the budget-Q — condensed)
`Q(scene-crop, action=(edge,depth), H)` → 60×5 map of gamma-discounted values for budget H.
- **Inputs:** 5-channel scene crop (robot · robot-goal · movable objects · static walls · reachable) + per-edge
  contact tokens (Fourier PE + per-edge embed) + **H embedding**.
- **Output:** 60×5 map; **policy = top-k**, **value = top-k-mean pool** (NOT raw max — H0b: max is fluke-dominated).
- **Arch:** EdgeCrossAttn (self-attn ON, validated H2) + H-embedding + **HL-Gauss classification value head**
  (Stop-Regressing) + gamma targets.
- **Recursion:** `Q(s,a,H) = [a opens?] OR V(T(s,a), H−1)`; budget decrements through the transition.
- **Two deploy regimes (always design for BOTH — [[feedback_search_nosearch_lens]]):** no-search (query at
  decrementing H, top-k, execute/verify; leans on the HIGH-H head) and search (prior→expand→leaf `Q(·,H−1)`→backup;
  leans on the LOW-H head on post-push states).
- Full 37-decision spec w/ citations: [multipush_horizonQ_journal.md](multipush_horizonQ_journal.md) "UN-PARKING BUILD SPEC".

## 5. The plan — Phases 0–6
Each: deliverable · milestone (M#) · reuse-vs-new · status.

- **Phase 0 — Unblock (gates).** 0.1 car/config/primitive [DONE: §3, test set reusable]. 0.2 `feat/horizon-q`
  branches + restore stuck-threshold→5 [PENDING]. 0.3 confirm region_opening emits gamma depth per first push [PENDING].
- **Phase 1 — H=1 data (model-free base).** Re-collect H=1 at 0.034/550, 20% bar, ~30 cells/scene uniform, masked,
  **KEEP dead-ends**. Reuse `region_opening` sampled collection + `modular_parallel_collection`. Deliverable `v4_hq_h1`.
  **M1:** plain 1-push scorer on `v4_hq_h1` reproduces champion hard@k. [Task #21, PENDING]
- **Phase 2 — Model: budget-Q @ H=1.** Extend `edge_crossattn`/`classifier_module`: H-embedding + HL-Gauss head +
  gamma targets + top-k-mean pool. Warm-start encoder from champion, re-init value/H head. **M2:** budget-Q(H=1) ≈
  champion; dead-ends→low V. [Task #20, PENDING]
- **Phase 3 — H=2 data (search-distilled, exploration-controlled).** Informative subset (no 1-push opener). Per first
  push: execute→s′; γ=1.0 if opens else verify k₂ second pushes→γ if any open + record success-fraction; else 0.
  Round-1 verify-heavy. Exploration: uniform/uncertainty (NOT confidence), floor ≥25%, **setup-state archive
  (Go-Exploit)**, disagreement acquisition. Harvest post-push H=1 free; tag negative types; keep H=2 dead-ends.
  Reuse tagged depth-2 (`region_opening` chain_depth/parent + `build_2push_validset`). **M3:** beat H0b 34.5%@1 @0 sims.
- **Phase 4 — Train full budget-Q (H=1+H=2).** Mixed batches (H=1 dominant + H=2, replay), gamma, HL-Gauss. **M4:**
  predicts setup-push values; reactive H=2 solve on held-out informative slice.
- **Phase 5 — Deploy + eval (BOTH regimes).** Reactive + search. Eval hit@k on `namo_testset_v1` (REUSED as-is) both
  regimes + post-push slice + dead-end slice + 2-push solve@k/sims. **M5 (headline):** solve@k/sims vs H0b 49-sim beam.
- **Phase 6 — ExIt iteration (2–3 rounds).** Re-collect w/ improved Q + exploration + setup archive; Reanalyze
  fractions; retrain; budget-escalate informative subset (Ferber). **M6:** per-round hit@k curve.

**Dependency:** 0→1→2→3(needs M2 informative subset)→4→5→6.

## 6. Comprehensive testing protocol
- **Eval rulebook (shared, do not re-define):** `scripts/eval_common.py` (match_episode / bin_of / floor).
- **Referee (one ckpt → full panel):** `scripts/eval_scorer.py --ckpt ... --network edge_crossattn --num-depths 5
  --out ...`. Default episodes = `namo_testset_v1/labels/onepush_episodes.json` (20% bar). hard@1 carries ±3–4 noise.
- **Verdict (decide if a change helped):** add group to `GRPS=()` in `resolve_robust.sh`; averages per-seed,
  paired compare across 3 seeds. Parses `divisions.hard.scorer_realistic.@1` — keep key stable.
- **Budget-Q-specific test slices (NEW, must add):** (a) hard@k BOTH regimes (reactive vs search); (b) **post-push
  slice** — eval Q on post-push states (H0b's OOD blind spot); (c) **dead-end slice** — does V say "low" on hopeless?;
  (d) **2-push solve@k / sims** vs the H0b 34.5%@1 / 49-sim-beam baselines; (e) per-difficulty (easy/med/hard).
- **Invariant gates before trusting any eval:** `gt_in_valid_frac ≈ 1.0`, `bad_object_match = 0`, room-leakage = 0%.
- **3 seeds minimum** per condition (hard@1 noise). Never compare single ckpts.

## 7. Repo state (commits / branches / paths)
- **namo_cpp** (branch `car-baseline`): `ea0f5ff` car 0.034 + regen primitives + d5 config + compare script;
  `2410564` un-parking 37-spec; `f5c5cbb` (superseded) the 0.0375 step. **Working tree clean** (little_car.xml at 0.034).
- **mujoco_env_creator** (branch `chore/archive-aug9-claudemd`): `c8144cc` car 0.034 in DIFF_DRIVE_CAR_BODY_XML;
  `6f54935` archive large aug9 + CLAUDE.md. `main` untouched.
- **sage_learning**: untouched so far (model work pending). Key files: `src/model/dit/edge_crossattn.py`,
  `src/model/classifier_module.py`, `src/data/scorer_data.py`.
- **Test set:** `/scratch/dm1487/datasets/namo_testset_v1/` — `manifests/canonical_scenes.txt` (2173, real XML paths),
  `labels/{onepush_episodes,pure2push,twopush}.json`, `pkls_2push_v2/`, `README.md` (datasheet). REUSED AS-IS.
- **Primitives:** `data/motion_primitives_1x_car_d5_{square,wide,tall}.dat` (0.034/550). Config:
  `config/namo_config_complete_skill15_car_1x_d5.yaml` (gen: max_push_steps=5). Collection config:
  `config/namo_config_complete_skill15_car_1x.yaml` (control_steps 550; stuck_threshold=1e6 → restore to 5).
- **Python env:** `/scratch/dm1487/envs/namo/bin/python`. Build: `./build_python_bindings.sh` (needs `MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7`).
- **Existing training data:** `v3_scorer_e4` (old bar, 0 dead-ends — to be superseded by v4_hq_h1).

## 8. Reuse scripts + per-episode invariants (from namo-data-pipeline skill)
**Reuse, don't fork:** `pipeline/build_episode_validsets.py` (pkls→per-episode key), `pipeline/build_2push_validset.py`,
`pipeline/build_scorer_dataset.py` (+`add_contact_px.py`), `amarel/testset_2push_collect.slurm` (depth-2 driver),
`data_collection/modular_parallel_collection.py`, `planners/opening/region_opening.py`, `eval_scorer.py`,
`resolve_robust.sh`, `eval_common.py`.
**Per-episode invariants (HARD gate):** unit = **(pushed object, goal region)**, never `xml`; match sample→episode by
**`object_center` (~0 mm)** + `gt ∈ valid`; **difficulty per-episode**; train/val/test holdout **grouped by ROOM (xml)**;
filtered datasets filter **per-episode at source**. Root doc: `docs/pipeline/multi_episode_rooms.md`.

## 9. Progress log + NEXT ACTIONS  ← **RESUME HERE**

### ⏩ AUTONOMOUS PIPELINE STATE [updated 2026-06-13 ~14:05 ET — USER AFK ~14:00–18:00, self-driving]
**GOAL (USER):** the 2×2 model matrix {Horizon, NoHorizon} × {v1 data, v2 data}, ALL trained + TESTED on the
test set, stats reported. Plus the v2 DATA FIX so H=2 encompasses H=1 (1-push@H2 augmentation + balanced
sampler — see the [USER DIRECTIVE 14:00] entry below for the why/how). Don't compromise; Slack each milestone.

**MATRIX STATUS [16:00 ET]:** Horizon-v1 `qfull_v4hq` ✅ ep16 (val .6517, still training 56015587).
NoHorizon-v1 `qfull_nohz_v4hq` ⏳ training (56025708, val .728). **Horizon-v2 `qfull_v2_v4hq` 56057675 +
NoHorizon-v2 `qfull_nohz_v2_v4hq` 56057676 ⏳ TRAINING (RE-LAUNCHED 17:15 w/ 14h walltime)** — L40S, array 9-11.
⚠⚠ **WALLTIME FAIRNESS FIX [17:12 ET]:** v1 + NoHorizon-v1 ran with **14h** walltime (v1 TIMEOUT'd at ep17, val
PLATEAUED ⇒ ep16 = its converged best, cell valid). But `launch_v2_training.sh` used the slurm DEFAULT **6h** —
and v2's 944k-row mix is ~74min/epoch ⇒ would TIMEOUT at ~ep5, SEVERELY under-trained vs v1's ep16 (invalid 2×2
comparison). `scontrol` extension is admin-only (denied) ⇒ CANCELLED the 6h v2 (only ep0, ~0 lost) + relaunched
with `--time=14:00:00` (committed to launch_v2). Now all 4 models train at matched 14h. First-launch 56055107/108
(6h) + the value_bins-bug 56054990/991 are dead. The 56055107/108 entry below is superseded.
v2 data_dir = 43 H5s; **[scorer setup] n=944,129 rows** (m2b 252k + h2 311k + aug 80k + postpush 300k),
train 850k/val 94k — **40% of the mix = the two OOD modes** (380k aug+postpush) ⇒ strongly over-represented.
⚠ FIRST LAUNCH 56054990/991 FAILED at model-instantiation: `value_bins` is a NETWORK param not model
(`+model.value_bins` → TypeError); fixed to `+network.value_bins=51` (head_mode stays `+model.`), relaunched OK.

**✅ v2 OOD MIX BUILT [15:55 ET] — all ingredients ready, launch recipe LOCKED:**
- `data_dir = m2b ; h2 ; onepush_h2_aug ; postpush(shard_0..3)` (';'-joined; ScorerDataModule joins them — l.137).
- 1-push@H2 aug: `v4_hq_onepush_h2_aug/data.h5` = **80,000 H=2 sparse-positive opener rows** (opener=1.0 @H2,
  loss masked to opener cells; from m2b 1-push-solvable rows, --max-rows subsample). VALIDATED (H=2, ctx 5×64×64,
  contact_px, r_mask==opener mask). The H4 dilution fix: lifts H=2 opener fraction ~16%→~46%.
- post-push: 1.87M rendered npz → **300k subsample** (good 64% / dead 36%, natural ratio) → shard-H5s in
  `v4_hq_postpush_v2/shard_*.h5`. The OOD s1 calibration data. ⚠ PACK THROUGHPUT: build_postpush_h5 is ~5 npz/s
  (per-row cv2.resize×5 + 60-edge contact_px loop dominate) — the first 4×75k pack (56054137) was ~4h ETA,
  CANCELLED + **re-sharded to 40×7.5k (56054645, all 40 RUNNING, ~25min)**. launch_v2 now globs shard_*.h5.
- MIX BALANCE [round-1 decision, CLAUDE]: ScorerDataModule has NO weighted sampler → proportions set by
  ON-DISK COUNTS + uniform sampling. m2b 252k + h2 ~280k + aug 80k + postpush ~280k ⇒ postpush ~33%, aug ~9%
  of the mix = the two OOD modes are OVER-REPRESENTED by counts (USER directive). Weighted sampler = a later lever.
- v2 launch = SAME slurm/recipe as v1 (array 9-11 = B30 × seeds 1-3). Horizon flags: `budget_cond=true
  budget_h=true head_mode=hl_gauss value_bins=51`. NoHorizon: same minus budget_cond, `budget_h=false`.

**📈 EVAL CURVE — FINAL [16:49 ET, n=1018, all 76 shards ✓]: see the HORIZON-V1 CELL block below for the full table.**
Headline: MODEL @900=**73.2** vs RANDOM(5-seed) **69.6**±.4; @2sim 17.7 vs 2.9 (6×); avg-sims-to-solve **61.6 vs
124.7** (2× efficiency). Model dominates the low/reactive budget, CONVERGES with random at 900 (value = efficiency
+ reactive regime, not ceiling). [The earlier n~150 "90%" was the easy shards finishing first — disregard.]

**PIPELINE DAG + NEXT ACTION on each completion (drive these as watchers fire):**
1. post-push RENDER (56025904, ~12 shards left, → /scratch/dm1487/datasets/v4_hq_h2/postpush_npz_v2, ~1.5M npz)
   → ON DONE: PACK to scorer H5: `build_postpush_h5.py --npz-dir <out> --out-h5 /scratch/dm1487/h5/v4_hq_postpush_v2/data.h5`
   (1.5M npz = shard the pack or subsample to ~300-400k; then DELETE npz to reclaim ~100GB — quota 617/1024GB).
2. PACK done → BUILD BALANCED v2 MIX H5(s): root (v4_hq_m2b_scorer H1 + v4_hq_h2_scorer H1/H2) + **1-push@H2
   AUGMENTATION** (relabel exhaustive 1-push openers as H=2 rows, opener=1.0 rest masked — the H2⊇H1 fix) +
   post-push. Sampler = WeightedRandomSampler keeping a DECENT OnePush fraction (not full balance). NEW small
   builder needed for the 1-push@H2 augmentation (sparse-positive H=2 rows from onepush data).
3. v2 MIX ready → LAUNCH Horizon-v2 (`qfull_v2_v4hq`, budget_cond+value_bins+hl_gauss+budget_h, array 9-11) +
   NoHorizon-v2 (`qfull_nohz_v2_v4hq`, value_bins+hl_gauss+budget_h=false, array 9-11). Same DATA_DIR (the v2 mix).
4. Any model ep~8/15 → snapshot feeler (eval_scorer_feeler.slurm + m3_key_feeler). NoHorizon-v1 done → test it.
5. ALL 4 trained → TEST ALL: eval_scorer H=1 & H=2 (onepush set), best-first pure2 & pure1 (model+random),
   key-graded m3. Assemble a 4-family stats table → Slack + journal + registry.

**LIVE JOBS + watchers [16:55 ET]:** EVAL FLEET ✅ DONE (n=1018; Horizon-v1 cell FINAL — see CELL block).
TRAINING: Horizon-v1 ✅ DONE (ep16 .6517, cell FINAL). **NoHorizon-v1 ✅ DONE [01:18 ET, ep14 .6732 converged]
— EVAL LAUNCHED 01:34** (rank 56091194/5, best-first 56091196; v1 ROW lands ~03:30, watcher `bc9ssymgs` → reduce_2x2).
Note NoHz-v1 val .673 > Horizon-v1 .652 = EXPECTED (NoHorizon has no H input → must fit H=1 AND H=2 labels for the
same state with one head → blurs; the conditioning lets Horizon fit both → lower loss; real test = ranking/solve).
**Horizon-v2 56057675 + NoHorizon-v2 56057676 still training (~ep5-6, finish ~07am)** — v2 chain re-armed (watcher
`b7ljxwker`, v2-only ids) → `eval_one_model.sh` (ranking H1/H2 + best-first
@900) → I arm the reduce. Random baseline is SHARED (bf900_uniform_s0..4, model-agnostic — don't recompute).
**2×2 HARNESS (committed):** `eval_one_model.sh <run>` (per-cell evals) + `reduce_2x2.py` (assemble matrix vs
shared random). Naming convention: solve=`bf900_<run>`, rank=`<run>_rank`; Horizon-v1's custom dirs symlinked in.
Curve reducer: `reduce_bestfirst_curve.py` (--avg-seeds for random mean±std).
**TOOLS (all committed):** eval_bestfirst.py, eval_rollout.py, eval_m3.py(--grade key), render_postpush_from_state.py,
build_postpush_h5.py, render_postpush.slurm, bestfirst_eval.slurm, m3_key_feeler.slurm, eval_scorer_feeler.slurm(EVAL_H).
**KEY FACTS to not re-fumble:** TEST SET is EXHAUSTIVE 2-push (exhaustive_depth2.yaml; full (a1,a2)→outcome in raw
pkls' primitive_trial_log) — k=30 sampling was TRAINING only. ckpts get pruned by save_top_k (use current best, not ep11).

> **⚠ SUSPECT BARS [2026-06-13 ~19:03]:** fpv_m2b **75.2@1** + old-champ **34.5@1** were computed on YET ANOTHER
> manifest (`pure2push_scenes.txt`, 787 scenes, only 640 overlap the key) — NOT object-constrained, first-push-
> graded, ~49 sims. THREE different 'pure-2' manifests existed (787/985/983-key), none aligned. ⇒ 75.2/34.5 are
> NOT comparable to the unified object-constrained key-driven solve numbers; RE-RUN fpv (M2b-as-leaf) on
> test_pure2_fromkey.txt before citing. All misaligned manifests archived to _archive_misaligned_2026-06-13.

### 🎯 EVAL METHODOLOGY — CANONICAL & UNIFIED (the non-confused version) [2026-06-13 ~15:20 ET, USER-directed]
This is the ONE eval story. Numbers computed before unification (75.2 / 34.5 / 22.9 / 62.4 / analytic-84.3) are
SUPERSEDED — do NOT cite them against these. Anything new gets graded exactly this way.

1. **What we grade (the unit).** Region Opening on the car: a path to a goal region is blocked by ONE labeled
   object; the planner must open it by pushing THAT object via ≤H pushes (H=2 here), choosing among the object's
   reachable (edge,depth) pushes (≤60 edges × 5 depths). The unit is one EPISODE = (scene-xml, object_id, goal),
   NEVER the xml — a scene hosts several episodes (CLAUDE.md GOTCHA).

2. **One key drives everything (key, not scene-list).** Pure-2-push set: **`pure2push.json`** (983 scenes / 1018
   episodes; EVERY episode is_1push_solvable=False & is_2push_solvable=True — needs a real setup-then-open).
   Key-derived manifest `test_pure2_fromkey.txt` (983). 1-push companion = `onepush_episodes.json` (991/1323).
   All scene-list manifests that didn't match the key are ARCHIVED (`_archive_misaligned_2026-06-13`). RULE: drive
   eval FROM THE KEY, per-record; the env-filename scene-lists are non-unique (651/983 overlap) and were the bug.

3. **Object-constrained (per-episode invariant).** The search may push ONLY `rec.object_id`. Unconstrained,
   best-first "solves" a pure-2 scene in 1 push via a DIFFERENT, easier object in the same room (~7% phantom
   @1sim). Constrained ⇒ @1sim solves = **0** = the honest 2-push problem (H13 ✅).

> **🔧 SUCCESS-CRITERION FIX [2026-06-14 ~04:20 ET, USER: "labels are correct, fix the eval"].** The best-first
> SOLVE eval graded success by `env.is_robot_goal_reachable()` = the SINGLE xml-`<site name="goal">` point (±5cm).
> But the LABELS grade by the COLLECTION criterion: **≥20% of the goal region reachable = ≥20 of 100 sampled
> goal-region points** (`region_opening._validate_opening`, goals_per_region=100 + fraction=0.2). DIFFERENT
> criteria ⇒ the label-vs-eval mismatch (273 "unsolved-but-2push-solvable" + 13 plan_len=1). FIX (eval_m3.py).
> ⚠ **FIRST fix attempt was BUGGY** (`goal_region_open` re-snapshotted POST-PUSH): after a push opens the path the
> goal region MERGES with the robot region + gets relabeled, so it sampled the WRONG region — matched labels only
> **20%** (on valid_1push openers) and gave a bogus ~41% solve. **CORRECT fix: `sample_goal_points(env)` ONCE at s0**
> (goal region still a distinct unreachable neighbour) → `goal_open_pts(env, pts)` counts those FIXED s0 points
> reachable ≥ ceil(0.2·100) after each push. VERIFIED **96%** match on labeled valid_1push openers (the ~4% = label
> noise). eval_bestfirst `--success region`(default)|point samples s0 pts per scene. ALL solve evals re-running
> CORRECTED [56115386-391 +s4]; old point dirs archived `bf900_*_POINT`; ranking (label-based) unchanged.
> ⇒ **all solve@K numbers in the cells below are SUPERSEDED by the corrected region re-run** (watcher bolef7st5).

> **✅ SCOPE OF THE SUCCESS-CRITERION BUG [2026-06-14, verified]:** it affected ONLY the live-SIM SOLVE evals
> (best-first solve@K via is_robot_goal_reachable). The M-SERIES (M1/M2a/M2b/M2c) + the v1-row RANKING numbers
> are `eval_scorer.py` = PURE LABEL-GRADING (topk_hit: top-k ranked candidate ∈ labels' `valid` set; NO env.step,
> NO is_robot_goal_reachable) ⇒ consistent with the ≥20%-region labels by construction ⇒ UNAFFECTED, verdicts
> STAND. Only solve@K was wrong. (The old suspect fpv_m2b 75.2 / champ 34.5 WERE single-point sim numbers, but
> already superseded for the manifest reason.) ROOT-CAUSE of the miss: had the label-vs-eval disagreement (273
> unsolved + 13 plan_len=1) and rationalized it as search-budget instead of auditing the success predicate.

4. **Metric = SOLVE-RATE vs SIMS, one curve (reactive→search on the same axis).** A sim = one real env push
   (~1s). solve@K = fraction of episodes opened within K sims. K=2 ≈ the reactive/0-search anchor (one setup + one
   open = the minimum 2-push cost); large K = full search. **EVERYTHING CAPPED AT 900 SIMS [USER].** Best-first
   explores in a budget-INDEPENDENT order, so ONE 900-cap run records the sim-index each episode solved at and
   yields the WHOLE curve {2,3,5,10,20,50,100,200,500,900} by post-processing the leaf jsonl (no separate per-budget
   runs). Reducer: `scripts/sandbox/reduce_bestfirst_curve.py` (--avg-seeds → random mean±std).

5. **MODEL = real sims, no shortcuts.** `eval_bestfirst.py --prior model`: value-guided greedy best-first ON THE
   LABELED OBJECT. Q(s,a,H) expands (which push to add); V=mean_top5(Q(s,·)) selects (which branch to chase);
   priority=blend(Q,V). Simulates every push it tries; grades by `env.is_robot_goal_reachable()` (includes the real
   post-push state s1 — the OOD target — so it can't over-credit like a first-push proxy). ckpt = Horizon-v1
   qfull_v4hq_s1 ep16 (val .6517). → `bf900_model_ep16`.

6. **RANDOM = real sims too, ≥5 seeds [USER: forget the analytic-over-the-map shortcut].** Same script
   `--prior uniform`: IDENTICAL loop + IDENTICAL candidate SET (the labeled object's reachable pushes), but RANDOM
   order and NO model — the forward pass is skipped entirely (the baseline must not touch the network; also makes it
   cheap on CPU). 5 seeds (SEED_BASE 7000..11000), report mean±std. → `bf900_uniform_s0..s4`. This is the
   brute-force floor the model must beat; the model−random gap = what the learned value buys at each sim budget.

7. **eval_scorer (M-series RANKING referee) is SEPARATE and still valid.** Offline hit@k over pre-rendered H5
   crops, object-MATCHED per crop, NO sim — "does the model rank the opener high?" (M1 +6.1pp; M2a/M2b verdicts
   STAND). Best-first answers the complementary "does the search OPEN the path?" Both object-constrained; they share
   the scoring core (live_scorer imports eval_scorer's loader/contact_px/match) ⇒ consistent, comparable.

### 🧩 EVAL ARCHITECTURE — two scripts, ONE shared scoring core [2026-06-13, verified — consistent, comparable]
- **`eval_scorer.py` = RANKING (hit@k)**: offline, reads PRE-rendered H5 crops, NO sim, object-matched per crop.
  The M-series referee (M1/M2a/M2b/M2c). "Does the model rank the opener high?"
- **`eval_bestfirst.py` = SOLVE (solve-rate / sims-to-solve)**: live env, runs the SEARCH, SIMULATES, object-
  constrained via --key. "Does the model's search open the path?" (also: eval_rollout, eval_m3 = same live path.)
- **SHARED CORE (so the two are consistent):** `live_scorer.py` imports `load_scorer, contact_px, match_episode`
  FROM eval_scorer + renders with the SAME `NAMODataVisualizer.generate_all_masks_highres` that built the H5
  crops. ⇒ model-load + contact_px + episode-match + crop-render are IDENTICAL across both. Can't merge (offline-
  static vs online-live I/O) but the scoring layer IS unified. TODO consistency check: eval_scorer hit@1 ≈
  best-first first-push top-1 on the same scenes (live_scorer has a rare wavefront-fallback path, last_fell_back).

> **⚠ TEST-SET HYGIENE [2026-06-13 ~14:55]:** the SCENE-LIST manifests (test_pure2push_combined.txt 985)
> and the KEY (pure2push.json 983) are DIFFERENT sets — only **651 overlap** by full path (env filenames
> are NOT unique: same run_NNNN basename under different set/benchmark dirs; 985 manifest = 410 unique basenames).
> ⇒ ALWAYS drive eval from the KEY, never the scene-list manifest. FIX: key-derived manifests
> test_{pure2,onepush}_fromkey.txt (983/991). The 671-episode object-constrained run used the OLD manifest
> (651 valid overlap); re-running on the key-derived 983 for the final table.

### 📋 CANONICAL TEST SETS / EVAL KEYS [2026-06-13, verified inventory — USE THESE, don't guess]
Dir `/scratch/dm1487/datasets/namo_testset_v1/labels/` (each JSON keyed by scene-xml → list of per-(object,goal) records).
- **1-PUSH eval → KEY `onepush_episodes.json`** (991 scenes / 1323 records; fields `valid`/`tried` = openers). eval_scorer
  default. ⚠ For best-first pure-1 I used `test_1push_solvable_combined.txt` (539 = aug9 159 + feb 380) — a SUBSET, not the
  full 991. STANDARDIZE 1-push eval on onepush_episodes.json [USER decision pending].
- **PURE-2-PUSH eval → KEY `pure2push.json`** (983 scenes / 1018 records; **ALL is_1push_solvable=False & is_2push_solvable=True**).
  **Manifest: `test_pure2push_combined.txt` (985 ≈ matches).** The clean pure-2 set — 0 one-push openers in the label. Used = correct.
  Difficulty bins: `pure2push_divisions.json` (same 983 + divisions).
- **SUPERSET (any ≤2-push, INCLUDES 1-push) → `twopush.json`** (1830 / 2341; 1323 also 1-push-solvable). NOT pure-2; overall 2-push F1'.
- **EXHAUSTIVE raw pkls** (full (a1,a2)→outcome map): `namo_testset_v1/pkls_2push_v2/shard_*/` (config exhaustive_depth2.yaml).
- ⚠ **OPEN ISSUE:** best-first eval pools ALL reachable objects + re-derives goal from xml ⇒ can "solve" a pure-2 scene in 1
  push via a DIFFERENT object (per-episode (object,goal) constraint NOT enforced — CLAUDE.md GOTCHA; ~7% @1sim). Verify + decide
  whether to constrain the eval to the labeled object. [USER decisions pending: (1) 1-push key = onepush_episodes? (2) constrain eval?]

### 📊 HORIZON-V1 CELL — FINAL results [16:49 ET, n=1018, all 76 shards ✓]
**SOLVE** (best-first @900, object-constrained pure2push, the unified eval; random = 5-seed real-sim mean±std):
| solve@ | 2 | 10 | 50 | 100 | 200 | 500 | 900 | avg-sims-to-solve |
|---|---|---|---|---|---|---|---|---|
| **MODEL** (Horizon-v1 ep16) | 17.7 | 39.5 | 56.7 | 62.6 | 66.7 | 70.7 | **73.2** | **61.6** |
| **RANDOM** (5-seed mean±std) | 2.9±.3 | 15.0±.9 | 38.0±2.1 | 47.2±1.2 | 55.6 | 64.9 | **69.6**±.4 | **124.7** |

→ Model ≫ random at LOW budget (6.1× @2sim, 2.6× @10) but **CONVERGES at 900** (73.6 vs 70.2, +3.4pp). **The win
is SIM-EFFICIENCY (60 vs 122 avg sims = 2×) + the reactive/low-budget regime, NOT the asymptotic ceiling** — with a
900-sim budget brute-force random nearly catches up on this object-constrained ≤2-push problem. Neither hits 100%
(pure2 is all 2-push-solvable) ⇒ best-first@hmax2 doesn't exhaust the hard tail within 900. This is the honest,
defensible horizon-Q story: amortize search to reach good solutions with far fewer sims; biggest gap where sims are
scarcest. [Earlier n~150 partial showed model@900=90 — that was the EASY shards finishing first; full set is 73.6.]

**RANKING** (eval_scorer hit@k, onepush key, scorer_realistic @1/@5/@10):
| division | H=1 | H=2 |
|---|---|---|
| hard | **34.4** / 69.3 / 80.4 | **12.2** / 30.2 / 41.3 |
| med | 84.2 / 95.2 / 97.7 | 59.6 / 76.6 / 84.2 |
| easy | 98.7 / 100 / 100 | 83.8 / 95.3 / 97.6 |

→ **H=2 DILUTION CONFIRMED on ep16** (hard@1 34.4→12.2; med 84→60; easy 99→84). This is the H2/H4 baseline the v2
1-push@H2 aug (80k opener=1.0@H2 rows) must lift back toward the H=1 numbers. (H=1 hard@1 34.4 > M2b 32.86 ⇒ H5
STILL holds at ep16; the earlier 38.4 was a different ckpt.) Note hard@1 H=2 failure_decomp: wrong_edge 71% — at
budget-2 the model picks a different edge (the setup-vs-opener confusion the aug targets).

> **🔁 SEED-2 ROBUSTNESS CHECK [2026-06-14 ~10:34 ET]:** the FULL 2×2 below is seed-1 only. Launched s2 evals for
> all 4 model families (56137094-105, corrected region eval) to confirm the qualitative findings (NoHz wins reactive,
> Horizon wins search) hold across training seeds — half the compute of a full 3-seed, cluster was idle. 2-seed
> compare lands ~12:30 ET (watcher bm4aq91a7). s3 only if s2 diverges or [USER] wants publication error bars.

### 🏁 FULL 2×2 MATRIX — FINAL [2026-06-14 ~08:26 ET, region criterion (corrected), n=1018 all cells; SEED-1]
| cell | rankH1 | rankH2 | s@2 | s@10 | s@50 | s@100 | s@900 | avg-sims |
|---|---|---|---|---|---|---|---|---|
| Horizon-v1 | 34.4 | 12.2 | 22.3 | 50.4 | 71.5 | 81.8 | 93.8 | 58.5 |
| NoHorizon-v1 | 21.2 | 21.2 | 28.7 | 46.8 | 63.1 | 71.4 | 89.0 | 85.1 |
| **Horizon-v2** | 36.0 | 30.7 | 24.2 | **55.3** | **76.3** | **82.6** | **94.9** | **54.6** |
| **NoHorizon-v2** | 31.7 | 31.7 | **32.6** | 52.0 | 67.5 | 74.0 | 91.6 | 76.7 |
| RANDOM (5-seed) | — | — | 3.3 | 19.8 | 51.0 | 63.6 | 90.8 | 113.6 |

**THE COMPLETE STORY (both deploy regimes, [[feedback_search_nosearch_lens]]):**
- **REACTIVE (@2sim, no-search): NoHorizon wins in BOTH data versions** (v1 28.7>22.3, v2 32.6>24.2). ⚠ **The
  reactive-flip prediction FAILED** — fixing Horizon-v2's H=2 ranking (12.2→30.7) did NOT flip reactive; the gap
  even WIDENED (6.4→8.4) because v2 data helped NoHz's reactive MORE (+3.9 vs +1.9). The unconditioned single
  "goodness" head is just better single-shot; budget-conditioning splits capacity and stays a reactive handicap.
- **SEARCH (@50-900): Horizon wins in BOTH versions** (+8-9pp @50-100, 1.4-1.5× efficiency; @900 Horizon>NoHz>random,
  NoHz≈/<random). The H=1 head guides the 2nd-push lookahead; horizon's value IS the search regime.
- **v2 data helps BOTH models' solve** (Horizon @50 71.5→76.3, NoHz 63.1→67.5; both +efficiency) AND fixes the
  H=2 ranking — broadly beneficial.
- **BEST DEPLOY: reactive → NoHorizon-v2 (32.6@2); search → Horizon-v2 (94.9@900, 54.6 sims).** No single winner;
  it's regime-dependent. Random @900=90.8 ⇒ at the ceiling only the Horizon models beat brute force.
- Random caveat: @900 random (90.8) > NoHorizon-v1 (89.0) — unconditioned guidance is net-negative at the ceiling.

### 📊 RANKING 2×2 — ALL 4 MODELS [2026-06-14 ~07:34 ET, eval_scorer hard@1, converged ckpts; LABEL-graded so correct]
| model | rankH1 | rankH2 |
|---|---|---|
| Horizon-v1 | 34.4 | **12.2** |
| NoHorizon-v1 | 21.2 | 21.2 |
| **Horizon-v2** | **36.0** | **30.7** |
| **NoHorizon-v2** | **31.7** | **31.7** |

**(1) v2 data FIXES Horizon's H=2 dilution: 12.2 → 30.7** (H1 unregressed 36.0 ≥ 34.4) — H4/H12 ✓ on the converged
ckpt (matches the ep7 feeler 31.2). **(2) v2 data ALSO lifts NoHorizon a lot (21.2 → 31.7)** — the OOD postpush+aug
mix helps the unconditioned model's ranking too. ⇒ at H=2 Horizon-v2 ≈ NoHorizon-v2 (30.7 vs 31.7, tie); the
horizon's REMAINING ranking edge is the H=1 specialization (+4.3). **H10 sharpened: given GOOD data, the horizon's
net value = H=1 ranking; at H=2 the unconditioned single-head matches it.** Solve@K (the reactive-flip test:
does Horizon-v2 @2 now beat NoHz-v2?) lands ~09:30 via bolef7st5.

### ✅ V2 H=2-DILUTION FIX CONFIRMED [2026-06-14 ~03:36 ET — Horizon-v2 ep7 FEELER, under-trained]
| model | H=1 hard@1 | H=2 hard@1 |
|---|---|---|
| Horizon-v1 ep16 (broken H2) | 34.4 | **12.2** |
| NoHorizon-v1 (budget-blind) | 21.2 | 21.2 |
| **Horizon-v2 ep7** (1-push@H2 aug) | **36.0** | **31.2** |

**The 80k 1-push@H2 augmentation did exactly its job: H=2 hard@1 12.2 → 31.2 (+19pp) at ep7 ALREADY** — past
NoHorizon's 21.2 (+10) and nearly matching its own H=1 (36.0). H=1 did NOT regress (36.0 ≥ v1's 34.4) ⇒ the aug
added H=2 opener signal for free. This is under-trained (v2 → ~ep11), so 31.2 is a LOWER bound. **H4 ✅ ACCEPTED
(preview).** Next: the SOLVE test — does Horizon-v2's reactive solve@2 now BEAT NoHorizon-v1's 22.5 (the broken-H2
reactive liability flips)? Full v2 cells ~09:00 via b7ljxwker. med/easy H2 also up (76.6/95.1 vs v1 59.6/83.8).

### 🥊 V1 ROW — CORRECTED (region criterion) [2026-06-14 ~06:34 ET; v1 final, random partial ~72/76; bolef7st5→full]
| model | rankH1 | rankH2 | s@2 | s@10 | s@50 | s@100 | s@900 | avg-sims |
|---|---|---|---|---|---|---|---|---|
| **Horizon-v1** | 34.4 | 12.2 | 22.5 | **50.7** | **71.7** | **82.2** | **94.1** | **58.4** |
| **NoHorizon-v1** | 21.2 | 21.2 | **28.7** | 46.8 | 63.1 | 71.4 | 89.0 | 85.1 |
| RANDOM (5-seed) | — | — | 3.4 | 20.2 | 51.9 | 64.3 | 91.3 | 111.5 |

**vs the (inflated/inconsistent) single-point numbers: everything is ~+20pp** (≥20%-region is EASIER than reaching the
exact site point) — Horizon @900 73.2→**94.1**, NoHz 70.2→89.0, random 69.6→91.3. **The QUALITATIVE story is ROBUST to
the fix:** (1) NoHorizon STILL wins reactive @2 (28.7 vs 22.5 — Horizon's broken H=2 still sabotages budget-2);
(2) Horizon STILL wins search (+9-11pp @50-100, 1.5× efficiency 58 vs 85 sims). **NEW sharp point: random @900 (91.3)
BEATS NoHorizon (89.0)** — NoHorizon's budget-blind guidance is WORSE than random ordering at the ceiling (it commits
to wrong branches deterministically), while Horizon (94.1) still edges random ⇒ horizon's guidance has positive value
in search, NoHorizon's is net-negative at high budget. [point-criterion v1 row archived below as superseded.]

### 🥊 V1 ROW — Horizon-v1 vs NoHorizon-v1 [SUPERSEDED — single-point criterion, 2026-06-14 ~02:50 ET]
| model | rankH1 | rankH2 | s@2 | s@10 | s@50 | s@100 | s@900 | avg-sims |
|---|---|---|---|---|---|---|---|---|
| **Horizon-v1** (ep16) | **34.4** | 12.2 | 17.7 | **39.5** | **56.7** | **62.6** | **73.2** | **61.6** |
| **NoHorizon-v1** (ep14) | 21.2 | **21.2** | **22.5** | 37.0 | 48.5 | 55.0 | 70.2 | 92.9 |
| RANDOM (5-seed) | — | — | 2.9 | 15.0 | 38.0 | 47.2 | 69.6 | 124.7 |
**⭐ NoHorizon's @900 (70.2) ≈ RANDOM (69.6)** — at the search ceiling its guidance adds ~nothing; Horizon (73.2) is
+3.6 over random. So Horizon is the more valuable SEARCHER (higher ceiling + 1.5× efficiency), NoHorizon the better
REACTIVE picker. Both crush random reactively (22.5/17.7 vs 2.9).

**NoHorizon's ranking is budget-BLIND by construction (H=1≡H=2≡21.2 — no H input).** The interplay:
- **H=1 ranking: Horizon ≫ NoHorizon (34.4 vs 21.2).** Budget-conditioning sharply helps the 1-push opener ranking.
- **H=2 ranking: NoHorizon > Horizon (21.2 vs 12.2).** Horizon's H=2 head is the BROKEN/diluted one; the
  unconditioned single "goodness" head is more robust at budget 2.
- **SOLVE — REACTIVE (@2sim): NoHorizon WINS (22.7 vs 17.7).** best-first at budget-2 queries H=2, and Horizon's
  broken H=2 mis-orders the first push ⇒ the conditioned model is WORSE than unconditioned in the 0-search regime.
- **SOLVE — SEARCH (@50-100): Horizon wins (+6-7pp) + 1.5× more sim-efficient (61.6 vs 92.7);** @900 ~tie (73.2
  vs 71.5). Once search expands to the post-push state (queried at H=1, where Horizon=34.4≫21.2), Horizon's better
  H=1 ranking finds the 2nd push faster.

**H10/H11 partial verdict:** the horizon's value is REAL but currently LOPSIDED — it buys a big H=1 ranking gain +
mid-budget search efficiency, but its broken H=2 head SABOTAGES the pure-reactive regime (loses @2 to the simpler
unconditioned model). **SHARPENED v2 PREDICTION:** fixing Horizon's H=2 (the 1-push@H2 aug) should lift Horizon-v2's
reactive solve@2 ABOVE NoHorizon's 22.7 and its rankH2 above 21.2 — IF v2-Horizon-rankH2 stays ≤ NoHorizon's 21.2,
the horizon isn't worth it and the unconditioned model wins. This is now a crisp, falsifiable v2-cell test.

### 🔎 PLATEAU / FAILURE ANALYSIS [2026-06-13 ~21:50 ET, on the Horizon-v1 solve curve, n=1018]
WHY the solve curve plateaus ~73% (model) / ~70% (random) instead of 100% on an all-2-push-solvable set:
- **273/1018 (27%) unsolved @900 are ALL `is_2push_solvable=True`** → NOT a goal/label problem, NOT unsolvable
  scenes. It's a **needle-in-haystack search-budget** limit: unsolved scenes have median **3 valid setups among
  ~60 candidate first-pushes** (rarity 10.5% vs 15.2% solved); the (a1,a2) space is ~60×60 with few openers.
- **MECHANISM = the H=2 ranking collapse.** At H=1 the first valid setup sits at median rank **2/60**; at H=2 it
  craters to median rank **15/60** (hard@1 34.4 → 12.2). best-first is greedy on that ranking, so on a rare-needle
  scene it expands ~15 wrong first-pushes (each spawning 2nd-push sims) before reaching a setup → blows past 900.
- **Model = EFFICIENCY, not COVERAGE.** model solved 745; random **5-seed UNION solved 777** (model-only 7,
  random-only 39). The model is deterministic (one fixed, H2-mis-ranked order) → fails the SAME way every time on
  its blind spots; 5 random orderings have diversity that stumbles into buried needles. Per single search the model
  dominates (2× fewer sims, 6× @2); pooled, its bias costs coverage. **The win concentrates on the HARDEST scenes:
  rarest-needle quartile model +10.9pp over random (62.2 vs 51.3); common-needle quartile only +2pp.**
- **FALSIFIABLE v2 PREDICTION:** the plateau's root cause IS the H=2 dilution the v2 1-push@H2 aug targets. If v2
  lifts H=2 ranking toward H=1 (setup back near rank 2-3), then (a) a chunk of the 273 flips to solved, (b) the
  deterministic-coverage gap to random shrinks, (c) **reactive solve@2 (now 17.7%, capped by the 12.2 H2 rank@1)
  rises**. Check on the v2 cells.
- **⚠ 13/745 model solves were `plan_len=1`** on `is_1push_solvable=False` scenes = **~1.7% label-vs-eval
  reachability mismatch** (eval's `is_robot_goal_reachable()` opens in 1 where the label said no). Small but real;
  this is the drift a goal-condition recompute would clean up — [USER asked 2026-06-14 about redoing the test set
  "with new goal conditions"; AWAITING the goal spec. If only goal/reachability changed → offline re-grade of the
  stored states, hours; if scenes/pushes changed → full re-collect. Output to namo_testset_v2, never overwrite v1.]

### 🔬 HYPOTHESIS LEDGER [USER 2026-06-13: run EVERYTHING as Observation→Hypothesis→Prediction→Verdict; accept/reject ON NUMBERS ONLY, nothing else. Add a new H# for every new design choice/problem; fill Verdict when numbers land.]

> **✅ EVAL CORRECTION DONE [2026-06-13 ~14:50, object-constrained pure-2, n=671 episodes, budget 100]:**
> @1sim=**0** both (cross-object 1-push shortcut GONE ✓ H13). solve-rate-vs-sims CURVE:
> MODEL 21/28/35/42/49/57/**62**% @ sims {2,3,5,10,20,50,100}; UNIFORM 3/4/7/14/23/36/**46**%. Model >> uniform
> at every budget (7× @2sim → 1.3× @100); model solves in **14.6 avg-sims vs uniform 30.8** (~2× more efficient).
> CORRECTED reactive 2-push solve (@2sim) = **~21%** (old all-objects 22.9 was barely inflated, +~2pp). Verdicts
> stamped: **H9 ✅ ACCEPT** (value-guided best-first ≫ uniform), **H13 ✅ ACCEPT** (object constraint = true 2-push,
> 0 one-sim solves). NOT affected, STAND: M1/M2a/M2b/M2c, budget-Q@H1 +5.5pp (H5), H2-dilutes (H2), dead-end +3.2pp.
> ⚠ CAVEAT: n=671 of ~1018 — **334 manifest scenes failed xml-key match** (path-convention mismatch; FIX for full
> coverage — the 671 matched are valid). [TODO: flat-H1 object-constrained for H6 foresight refresh.]

- **H1 — Budget-conditioning works (the core bet). VERDICT: ✅ ACCEPTED.**
  Obs: a setup push is worthless with 1 push left, valuable with 2. Hyp: conditioning Q on remaining budget H
  lets ONE net value the same push differently per H. Predict(accept iff): H=2 query ranks setups ≫ H=1 query
  on pure-2-push. Numbers: H=2 hit@1=19.8 (5.5× floor 3.6); H=1=3.4 (AT floor). Reactive solve@1=22.9 (8.4× floor 2.7). → ACCEPT.
- **H2 — H=2 subsumes H=1 (opener=1.0 at H=2). VERDICT: ❌ REJECTED.**
  Obs: opens-in-1 ⊆ opens-in-2. Hyp [CLAUDE]: H=2 holds on 1-push scenes (opener 1.0 > setup 0.9). Predict: H=2-on-onepush
  ≈ H=1-on-onepush. Numbers: H=2-on-onepush hard@1=13.7 vs H=1=38.4 (−25pp; even easy 99→87). → REJECT. H=2 ≠ superset.
- **H3 — H=2 dilution cause = MISSING 1-push data at H=2. VERDICT: ❌ REJECTED [USER caught].**
  Hyp [CLAUDE]: H=2 rows are deadend-only. Predict: ~0% of H=2 rows are 1-push-solvable. Numbers: 16.2% ARE 1-push-solvable. → REJECT.
- **H4 — H=2 dilution cause = IMBALANCE (16% too few). VERDICT: ✅ ACCEPTED [preview, Horizon-v2 ep7].**
  Obs: 16% 1-push@H2 but still dilutes (dominated by 84% setup/dead). Hyp: rebalance/augment 1-push@H2 → dilution ↓.
  Numbers: 80k opener=1.0@H2 aug rows ⇒ Horizon-v2 ep7 H=2 hard@1 = **31.2 vs Horizon-v1's 12.2** (+19pp, past
  NoHorizon 21.2), H=1 unregressed (36.0). The imbalance WAS the cause; injecting opener signal fixes it. → ACCEPT
  (confirm at final ep + check the reactive-solve flip).
- **H5 — Budget-Q@H1 ≥ M2b (no 1-push regression). VERDICT: ✅ ACCEPTED.**
  Predict: budget-Q@H1 hard@1 ≥ M2b 32.86. Numbers: 38.4 (ep15), +5.5pp. → ACCEPT.
- **H6 — Foresight helps end-to-end (reactive rollout). VERDICT: ✅ ACCEPTED (modest).**
  Predict: reactive-Q (1st push @H2) > reactive flat-H1. Numbers: 22.7 vs 19.4 (+3.3@1, +7.2@10). → ACCEPT but modest
  (rollout forgives a mis-ranked 1st push via retries/2nd push).
- **H7 — Learned prior beats brute search at matched sims (amortization). VERDICT: ⏳ RE-TESTING.**
  Old flawed-beam numbers: Q-search 30.3 vs brute 11.2 (2.7×); reactive-Q 22.7 > brute-search 11.2. Beam was wrong-design →
  re-testing with value-guided best-first (H9). Predict: best-first(model) ≫ best-first(uniform).
- **H8 — mean_top5 > max as the state/leaf value. VERDICT: ✅ ACCEPTED (H0b prior).**
  Obs: max is fluke-dominated on OOD states. Numbers: mean_top5 34.5 vs maxP 24.6 @1. → ACCEPT (use mean_top5 for selection).
- **H9 — The search is value-guided GREEDY BEST-FIRST (Q expands, mean5-V selects; min sims), NOT MCTS/beam.
  VERDICT: ✅ ACCEPTED [CANONICAL 900-cap, n=1018 FINAL, 5-seed random].** solve@K MODEL vs RANDOM(mean): @2sim
  17.7 vs 2.9 (6.0×), @10 39.5 vs 15.0 (2.6×), @100 62.6 vs 47.2 (1.3×), **@900 73.2 vs 69.6 (1.05×)**; avg-sims-
  to-solve 61.6 vs 124.7 (2× efficiency). **REFINEMENT: the guidance buys SIM-EFFICIENCY + the reactive/low-budget
  regime, NOT the asymptotic ceiling — at 900 sims brute-force random nearly catches up** (both ~70-74% on the
  object-constrained ≤2-push problem; best-first@hmax2 doesn't exhaust the hard tail). Greedy best-first (no
  MCTS/PW) confirmed effective; the OLD n=671 budget-100 numbers (62 vs 46) match @100 here (63.2 vs 47.8).
- **H10 — Do we even NEED the horizon? VERDICT: ✅ RESOLVED [full 2×2] — REGIME-DEPENDENT, no single winner.**
  Horizon WINS SEARCH in both data versions (@50-100 +8-9pp, 1.4-1.5× efficiency, @900 beats random where NoHz
  ≈/<random) and the H=1 ranking (+4-13pp). NoHorizon WINS REACTIVE in both (@2 28.7>22.3 v1, 32.6>24.2 v2). The
  H=2 fix in v2 (12.2→30.7) did NOT flip reactive (gap widened 6.4→8.4) ⇒ budget-conditioning is intrinsically a
  reactive handicap (capacity split) but the search asset. DEPLOY: reactive→NoHorizon-v2, search→Horizon-v2.
- **H11 — In SEARCH the horizon is REDUNDANT; horizon only helps REACTIVE. VERDICT: ❌ REJECTED [v1, the OPPOSITE].**
  Predicted horizon helps reactive, redundant in search. DATA [FINAL n=1018]: horizon LOSES the reactive @2 (17.7 <
  NoHz 22.5, its broken H=2 sabotages budget-2) and WINS search (@50-100 +7-8pp, 1.5× sim-efficiency 61.6 vs 92.9,
  @900 73.2 > 70.2 where NoHz≈random 69.6). So in v1 the horizon's net value is SEARCH (higher ceiling + efficiency),
  and it's currently a reactive LIABILITY. The v2 H=2 fix is predicted to flip the reactive sign (Horizon-v2 @2 > 22.5).
- **H13 — Eval MUST be object-constrained / per-episode (push the LABELED object only). VERDICT: ⏳ PENDING [USER design].**
  Obs: unconstrained best-first solved ~7% of "pure-2-push" scenes in 1 sim — because scenes have MULTIPLE reachable objects
  (verified: env_0177 has obstacle_1 AND obstacle_3) and the search opened the path via a DIFFERENT (easier) object than the
  labeled 2-push one. Hyp [USER]: restrict the search to rec.object_id (one-to-one w/ GT key) ⇒ evaluates the TRUE k-push
  problem on the labeled object; same object-matched eval for ALL models (M1/M2a/M2b/M2c via eval_scorer, which is already
  object-matched). VERDICT: ✅ ACCEPTED — object-constrained @1sim=**0** (cross-object 1-push shortcut gone); the honest
  2-push curve = MODEL 21→62%. Impl: rank_first_pushes_h2(restrict_obj=), eval_bestfirst --key + per-record loop.
  ⚠ FOLLOW-UP: 334/985 manifest scenes failed xml-key match (only 671 episodes graded) — fix path matching for full coverage.
- **H12 — v2 OOD data (post-push + 1-push@H2) fixes the OOD failures. VERDICT: ✅ ACCEPTED (ranking) [2026-06-14].**
  Numbers: Horizon-v2 H=2 hard@1 = 30.7 vs Horizon-v1 12.2 (+18.5, dilution fixed, H1 unregressed 36.0). BONUS: v2
  data also lifts NoHorizon (21.2→31.7) — the OOD mix is broadly beneficial. Solve-side confirm pending (bolef7st5).

**Done (2026-06-11, autonomous session):**
- Car 0.034 + exclude across both copies, MuJoCo-verified, committed (ea0f5ff namo_cpp, c8144cc env_creator).
- Primitives regenerated at 0.034/550, committed. Backup in `data/_primitive_backup_pre0034/`.
- **Controlled diff → pure car effect NEGLIGIBLE (±0.5%); the ~14% was the 482→550 config. TEST SET REUSABLE AS-IS.**
- 3-agent lit sweep → closest neighbors (MORE, Bejjani, HACMan, Go-Exploit, Soemers, DeepCubeA, Ferber, SAVE);
  reading list Slacked. 37-decision grounded spec committed.
- **feat/horizon-q branch across all 3 repos.** Restored `controller_stuck_threshold`→5 (committed dee0b59).
- **Budget-Q model scaffold (committed df198f0, sage_learning feat/horizon-q):** EdgeCrossAttn `budget_cond`
  (H-embedding) + `value_bins` (HL-Gauss head) opt-in flags (default OFF=unchanged); `src/model/hl_gauss.py`;
  `scripts/smoke_budget_q.py` PASSES (forward, masked CE, grads, value∈[0,1], pool, backward-compat).
- **H=1 collection VALIDATED + LAUNCHED:** 6-scene smoke + dead-end scene both collect cleanly at 0.034/550 + 20%
  bar; dead-ends RECORDED (all-fail trial_log → H0b bug is in the BUILDER not collection). Driver parameterized
  (GOALS_PER_REGION, committed). **SLURM job 55944720** (array 0-59, 250k feb, goals 100) → `/scratch/dm1487/outputs/v4_hq_h1`. RUNNING.
- **Datasheets** (committed): `docs/pipeline/horizon_q_datasets.md` + canonical_testset.md reuse banner.
- **H=1 collection FINISHED** (~242k pkls). **Validset built: 284,406 episodes / 219,881 scenes — 76.1% solvable,
  23.9% DEAD-ENDS (67,959)** via `--keep-dead-ends`. H0b fixed in the data (`/scratch/dm1487/datasets/v4_hq_h1/episodes_deadends.json`).
- **⚠ MASK-SOURCE CORRECTION (important):** the H5 builder LIFTS masks from the DiT `v3_balanced` H5 — but that's
  on DIFFERENT scenes (aug9/v3_phase2; **0 path overlap** with my feb-250k v4_hq_h1 labels). So masks CANNOT be
  lifted — they must be **RENDERED** from the v4_hq_h1 pkls. Render path validated (`run_mask_generation.py batch`
  → npz with `local_tight_*` 224x224 + `object_center` + `edge/depth_a1`). **Render LAUNCHED: SLURM job 55949895**
  (array 0-20 → `/scratch/dm1487/outputs/v4_hq_h1_masks`). NOTE: the mask renderer ALSO drops dead-ends (same H0b
  filter as the validset) — so the rendered masks = SOLVABLE only; dead-end masks need a `batch_collection` fix (task #23).


- **MASKS RENDERED:** 213,789 npz (solvable) → SLURM 55949895 done. **npz→H5 pack LAUNCHED: SLURM 55951271**
  (`convert_to_hdf5 --minimal` → `/scratch/dm1487/h5/v4_hq_h1_masks/data.h5`). Then: `build_scorer_dataset`
  (join masks + f_grid, SAME scenes) → `add_contact_px` → train. **M1 uses the EXISTING champion recipe**
  (edge_crossattn + pos_fourier + use_edge_embed + sigmoid_bce) on the v4_hq_h1 H5, graded on namo_testset_v1 —
  no budget-Q wiring needed for M1 (that's M2+).

**CORRECTED H5 pipeline (supersedes §9.1's mask-lift):** collect ✓ → validset ✓ → **render masks from v4_hq_h1
pkls (job 55949895)** → npz→H5 (`convert_to_hdf5`/`build_h5`) → `build_scorer_dataset` join (masks + f_grid, now
SAME scenes) → `add_contact_px` → train. For M1: solvable scenes; dead-end masks (task #23) for the value.

**NEXT ACTIONS (in order):**
1. **Monitor job 55944720** → when done, count pkls per shard, spot-check a dead-end + a solved pkl.
2. [Task #22, H0b] **Fix `build_scorer_dataset.py` to KEEP dead-ends** (all-zero f_grid retained). Validate on the
   dead-end pkl at `/scratch/dm1487/hq_deadend/`. CRITICAL before building the H5.
3. **Build the H5** from v4_hq_h1 pkls (join DiT masks from v3_balanced_1to1 + f_grid + r_mask + contact_px),
   room-grouped split. Then **M1**: plain 1-push scorer reproduces champion hard@k on v4_hq_h1.
4. **Wire budget-Q training:** `classifier_module.py` training_step (gamma targets + H + HL-Gauss loss via
   `hl_gauss.py`) + `scorer_data.py` (emit H + gamma label + keep dead-ends). Then train budget-Q(H=1) → **M2**.
5. [Phase 3] H=2 search-distilled collection on the informative subset (see §5).
- Smoke artifacts (delete when done): `/scratch/dm1487/hq_smoke/`, `/scratch/dm1487/hq_deadend/`.

**Constraints/judgment for autonomous work:** do NOT launch a big training run on unverified/incomplete data.
Smoke-test before scaling. Keep this §9 log current so a compaction can resume. Slack the user at each milestone.


- **⚠ COMPOSITION FIX [USER catch]:** v4_hq_h1 was 100% feb (reused v3_feb_top250k manifest, unreasoned).
  Test set = 59% feb / 41% aug9; champion = ~0 feb. A feb-only model is OOD on 41% of the eval. FIX: collect
  aug9 H=1 (SLURM 55954585, 100k aug9 -> v4_hq_aug9_h1). **Killed the premature feb-only pack** (would double-pack).
  Revised M1 path: aug9 collect -> render aug9 -> pack feb+aug9 npz TOGETHER (one src-h5) -> validset
  (feb+aug9, --keep-dead-ends) -> join -> M1, target ~60:40 (subsample feb 250k at PACK so src-h5 carries the
  ratio; validset stays full; join only emits rows present in src-h5). aug9 H=2 queued after aug9 H=1.

- **⚠ H=2 SEQUENCING [USER 2026-06-12]:** KILLED the running feb-only H=2 collection (was 55953042). Reasons:
  (1) it was composition-wrong (feb-only; H=2 needs the same ~60:40 as H=1/test), (2) it was stealing CPUs from
  the M1-critical aug9 H=1, (3) sunk cost tiny (~8k pkls, recollected in the unified pass anyway). NEW H=2 ordering:
  aug9 H=1 done -> render aug9 -> aug9 validset (gives aug9 dead-end scenes) -> MERGE feb dead-end scenes
  (`v4_hq_h1_deadend_scenes.txt`, 63,892) + aug9 dead-end scenes into ONE manifest -> a SINGLE H=2 pass over the
  union = composition-correct H=2. H=2 is now OFF the M1 critical path entirely (M1 needs only H=1); M1 pack/join
  proceeds in parallel with the unified H=2 collection.

- **aug9 H=1 DONE [2026-06-12]:** job 55954585, all 60 shards COMPLETED, **exactly 100,000 pkls** →
  `/scratch/dm1487/outputs/v4_hq_aug9_h1`. Manifest: `/scratch/dm1487/manifests/v4_hq_aug9_h1_pkls.txt`.
  Auto-kicked: **aug9 mask render 55955792** (array 0-19, SHARD_SIZE=5000, same driver/args as feb →
  `/scratch/dm1487/outputs/v4_hq_aug9_h1_masks`) + **aug9 validset 55955793** (`--keep-dead-ends` →
  `/scratch/dm1487/datasets/v4_hq_h1/episodes_deadends_aug9.json`; its dead-end scenes feed the unified H=2 manifest).
- **CODE-REGRESSION CHECK [USER ask, 2026-06-12]:** decided NO full retrain on old data (±3-4pp ckpt noise ⇒
  single run can't detect regression; violates never-retrain-baselines; diff inspection of df198f0 shows
  `budget_embed` constructed ONLY if budget_cond, head unchanged when value_bins=0, training stack untouched ⇒
  defaults-off code path is bit-for-bit the champion's). INSTEAD: **inference regression** — job 55955816 re-runs
  `eval_scorer` on the recorded champion arm **h5samp_B30_s1** (ckpt in final_verdict_snapshot, recorded hard@1=23.8,
  newbar_verdict JSON) with feat/horizon-q code; deterministic eval ⇒ numbers must match EXACTLY.
  **RESULT: EXACT MATCH** — full JSON identical (all divisions/@k, 1179 eps, hard@1=23.8). Inference path certified.
  Remaining gap = TRAINING stack (loader/loss/opt/env); plan [CLAUDE proposal, user not yet confirmed]: 1-epoch
  training smoke on the OLD H5, same seed/config, compare epoch-1 loss vs champion's wandb curve (~1 GPU-h).
  Full 3-seed retrain REJECTED (GPU nondeterminism ⇒ 1 run uninformative vs ±3-4pp noise; registry already has
  the 3-seed champion distribution; never-retrain-baselines). Last-resort control arm only, if M1 fails + data
  bisects come back clean.
  Also confirmed [USER ask]: M1 recipe = self-attn ON (`edge_self_attn=True` default; H2 verdict re-validated on
  the 20% test in 9cb5468) + pos_fourier + use_edge_embed + sigmoid_bce.
- **M1 SCOPE [USER ask, 2026-06-12]:** M1 = SOLVABLE-ONLY (no dead-ends) — keeps the gate a controlled
  data-factory test vs champion (dead-ends carry no within-scene ranking signal + masks for them need task #23
  anyway). Dead-ends enter at **M2b**: M2a = budget-Q head on the SAME solvable H5 (isolates arch), M2b = +dead-ends
  (isolates data, checks dead-ends→low V). PRE-REGISTERED PREDICTION for M1: hard@k **at or slightly above**
  champion (train bar now MATCHES test bar — the old mismatch cost ~5pp; sparse-30/composition should be neutral).
  If clearly below ⇒ factory bug; bisect via exhaustive-subset / feb-only / old-mask cells.

- **aug9 H=1 RESULTS + COMPOSITION CORRECTION [2026-06-12]:** validset (55955793) + render (55955792) done.
  **aug9 is much harder than feb:** 100k pkls → 55,676 scenes / 65,102 tried episodes; **40.3% solvable / 59.7%
  dead-ends** (feb: 76.1/23.9). **25,938 solvable npz rendered.** The "missing" ~44% of pkls = **pre-trial setup
  failures** (taxonomy: `success=False, validation_method='connectivity'`, mostly 0 region goals sampled) — the
  car-scale rooms often can't stage an episode; no trials → no labels → correctly excluded; BENIGN env property,
  not a pipeline bug. aug9 dead-end episodes (38,859 / 37,227 scenes) feed the unified H=2 manifest.
  **CORRECTION: pack target is 65:35 feb:aug9, not 59:41** — the 59:41 was scene-level; the test set's
  EPISODE-level split is 855 feb : 468 aug9 = 65:35, and episodes are what we train/eval on.
  **Sizing [CLAUDE, measured]:** pack-now = 25.9k aug9 → ~74k total @65:35 (−25% vs champion's 98k);
  collect REMAINING ~65k aug9 scenes (~1h, idle CPUs) → ~43k aug9 → **~123k total @65:35 (+25% vs champion)**
  → removes "less data" as an M1 confound (data scaling = the proven lever, E4). DECISION: collect the rest first.
  Manifest `v4_aug9_rest.txt` = full pool − used 100k − canonical test (65,008 scenes). **LEAK GATE PASSED:**
  used∩test=0, pool∩test=0, feb250k∩test=0 (test set path-disjoint from training pools, as designed).
  **LAUNCHED: job 55956248** (60 shards → `/scratch/dm1487/outputs/v4_hq_aug9_h1_rest`, goals=100). On completion:
  render rest-masks + extend validset → pack feb+aug9 @65:35 (~123k) → join → M1.

- **[USER GO, 2026-06-12] GPU training smoke + unified H=2 both approved.**
  (a) **Training-stack regression smoke LAUNCHED: job 55956678** = `train_h5_sampling.slurm` array idx 9
  (B30 cond, seed 1) with `SMOKE=1` (2 epochs, run name `h5samp_B30_s1_smoke`) on the OLD H5
  (`v3_scorer_e4_data`) with feat/horizon-q code. VERDICT RULE: epoch-1/2 train+val losses match the recorded
  `h5samp_B30_s1` wandb curve (same seed+sample_seed ⇒ identical data order; GPU nondeterminism ⇒ near-match,
  not bit-match). Catches loader/loss/optimizer/env drift that the (passed) inference regression can't see.
  **RESULT (2026-06-12): TRAINING STACK CERTIFIED.** First smoke was LR-horizon-confounded (cosine over
  max_epochs: 2 vs 200) ⇒ reran schedule-matched (job 55957256, SMOKE_EPOCHS=200 + 55-min wall cap).
  val_loss ep0/1/2: original 0.8947/0.7186/0.6677 vs smoke200 0.9087/0.7097/0.6493 — ±1-3%, ALTERNATING
  sign, identical trajectory shape = GPU nondeterminism (+different card), NOT a systematic regression.
  Full cert: inference byte-exact + training curves interleave within noise + git shows stack untouched.
  (b) **Unified H=2 GO:** after rest-validset → merge dead-end scene manifests (feb 63,892 + aug9-b1 37,227 +
  rest ~24k expected) → ONE `testset_2push_collect.slurm` pass (killed feb-only job's pattern:
  `sbatch --array=0-63 --job-name=v4-h2`, env MANIFEST/HOME_DIR/PKL_SUBDIR). Runs parallel to M1 pack/train.

- **[2026-06-12] ALL H=1 DATA COMPLETE + H=2 UNIFIED LAUNCHED + STACK CERTIFIED:**
  (a) **aug9-rest done** (55956248, 65,008/65,008 pkls) → render 55957921 (17,206 npz) + validset 55957922
  (42,720 eps: 17,298 solvable / 25,422 dead) both done. **Full H=1 inventory: feb 213,789 + aug9 43,144
  solvable npz; validsets feb + aug9 + aug9_rest** (all `--keep-dead-ends`).
  (b) **Unified H=2 LAUNCHED: job 55958028** — merged manifest `v4_hq_h2_deadend_scenes_unified.txt`
  (**125,494 dead-end scenes** = feb 63,892 + aug9 61,602; 51/49) → `/scratch/dm1487/datasets/v4_hq_h2/
  pkls_2push_unified` (64 shards; killed feb-only partial kept aside in pkls_2push/, 15,126 pkls, do not mix).
  H=2 TRAINING-row composition gets set later at dataset build (match the 2-push test slice).
  (c) **TRAINING-STACK CERTIFIED:** schedule-matched smoke (SMOKE_EPOCHS=200, killed @5 epochs) tracks the
  original B30_s1 wandb curve within ±1.6–2.7% with SIGN FLIPS (ep0 +1.6%, ep1-4 slightly better) = GPU
  nondeterminism, not regression. With the byte-exact inference regression ⇒ full-retrain question CLOSED.
  `train_h5_sampling.slurm` gained SMOKE_EPOCHS (committed in sage_learning).
  (d) **Dead-end pipeline COMPLETE (task #23):** renderer `--include-dead-ends` (namo_cpp 8a73945; all 5
  consumed channels render-time derivable — `robot_region` is the model's reachability channel, the
  reachable-OBJECTS list only fills the unused global mask ⇒ NO H=2 backfill dependency, M2b unblocked) +
  scorer join dead-aware matching/dedup/gating + `dead` column (3a8b59a). Decision [CLAUDE]: SKIP the
  region_opening reachable-recording edit — no longer needed, and H=2 must run byte-identical collector
  code to H=1 for homogeneity.
  (e) **M1 pack LAUNCHED: job 55958342** — `v4_hq_m1_npz_65_35.txt` (123,269 npz = feb 80,125 seed-42 sample
  + aug9 43,144; 65:35 by episode, matching the test set's 855:468) → `/scratch/dm1487/h5/v4_hq_m1_65_35/data.h5`
  via convert_to_hdf5 `--npz-list` (NEW flag, sage 8fc589e) `--minimal --tight-only --compression lzf`
  (champion src-H5 convention). **Validset merge: job 55958266** → `episodes_deadends_all.json` (feb+aug9+rest,
  overlap-asserted). NEXT when both land: `build_scorer_dataset --src-h5 .../v4_hq_m1_65_35/data.h5
  --episodes .../episodes_deadends_all.json --out-h5 /scratch/dm1487/h5/v4_hq_m1_scorer/data.h5` → gates
  (gt_in_valid>0.99, bad_match≈0, dead=0 rows in M1 since npz are solvable-only) → **add_contact_px** →
  M1 train (champion recipe, 3 seeds, GPU). NOTE [parallel session]: budget-Q training wiring landed via
  sage 770cc9c (head_mode=hl_gauss + H passthrough + budget_h flag; extended smoke PASSES; default-off) —
  task #20 wiring DONE, M2a can start right after M1.

- **⚠ TWO-SESSION OWNERSHIP SPLIT [2026-06-12 ~02:40, session A]:** two Claude sessions drive this build;
  coordinate HERE (both read this file). **B owns the M1 chain** (m1-pack 55958342 → join → add_contact_px →
  M1 train 3 seeds) and unified H2 (55958028) — claimed in the entry above. **A owns the M2b data path**:
  dead-end mask render **55958356** (feb+aug9-b1 pkls, `--dead-ends-only` → `v4_hq_de_masks`; rest pkls to
  follow) → de-pack → multi-src join (88ff98d: `--src-h5` now takes MULTIPLE h5s w/ cross-file dedup —
  backward-compatible, B's single-src call unaffected; mini end-to-end test passed: 240 rows, 40 dead,
  gt=100%, align_err=0) → M2a/M2b configs. **RULE: before ANY sbatch, check squeue/sacct for an equivalent
  job; concurrent build_scorer_dataset runs must use DIFFERENT --out-h5 paths (same-path = corrupt H5).**

- **⚠ [USER CONSTRAINT, 2026-06-12] EXHAUSTIVE COLLECTION BEYOND 1-PUSH IS NOT SCALABLE FOR US.**
  The running unified H2 (55958028, exhaustive depth-2 over the 125k dead-end scenes) is the **LAST
  exhaustive collection beyond 1-push** — kept (CLAUDE recommendation, user not opposed) because it is the
  one-time calibration asset: (a) certified depth-2 dead-ends (an absence claim needs the swept tree) for the
  dead-end EVAL slice, (b) exact success-fractions, (c) the complete answer key any sampling scheme can be
  carved from offline (B30-from-Aexh pattern). ALL FUTURE H>=2 collections (ExIt rounds 2+, H=3) are
  SAMPLED AT ALL LEVELS [USER 2026-06-12: "Sample all levels"]: sampled first pushes AND sampled (budget-k2)
  second pushes; a tried first-push cell whose sampled follow-ups all fail is labeled LOW — occasionally a
  false zero on a single-scene basis, but ACROSS environments E[label|cell] = the fraction of working
  follow-ups (graded by ROBUSTNESS, not the certified OR), which BCE at scale converges to and which matches
  the budgeted-attempts deployment reality better than best-play OR [USER argument, CLAUDE concurs — this
  SUPERSEDES the "level-2 must be swept" rule in the entry above]. Dead-ends arise NATURALLY as all-low
  grids — "if the network picks up by scale of different environments that the entire f-grid is low-Q,
  that is what it should pick up" [USER]. The exhaustive H2 run's swept trees remain useful as the
  certified EVAL slice + offline sampling simulator, NOT as a training requirement. H=3: search-distilled only.

- **SAMPLED H2 COLLECTION — IMPLEMENTED + SMOKED [2026-06-12, overnight]:** [USER killed exhaustive H2
  55958028 @14,635 pkls — KEPT as ordinary training rows; certified eval = namo_testset_v1 (exhaustive both
  depths), training data is NEVER exhaustive beyond depth-1 again]. Implementation (committed):
  `region_sample_k` (uniform random k-subset of reachable (edge,depth) per chain level — ONE cap point, all
  node expansions flow through the same candidate build in region_opening) + `region_sample_restarts`
  ([USER]: up to 3 attempts with fresh subsets ONLY while no chain found; trial logs MERGED ⇒ union mask;
  early-stop on success = adaptive compute). Plumbed via modular_parallel_collection + CONFIG_YAML override
  in testset_2push_collect.slurm. Config: `v4_hq_h2/configs/sampled_depth2_k30.yaml` (k=30, restarts=3,
  enumerate-all-sampled + record-all-solutions ON). **SMOKE #1 (k30, no restarts) PASSED:** root ≤30 ✓,
  ≤30/child ✓, levels tagged (`chain_depth`, `parent_edge`/`parent_depth` ⇒ per-level masks reconstructable),
  ~850 trials/dead-instance (vs 1,860 exhaustive). SMOKE #2 (restarts=3, 10 feb scenes, job 55959912) in
  flight — verify early-stop on 2p-solvable + 3x merged logs on dead; THEN scale relaunch on
  `v4_hq_h2_s30_remaining.txt` (~110k scenes, PKL_SUBDIR=pkls_2push_s30). ⚠ ETA caveat measured in smoke #1:
  restarts trigger on every truly-dead instance (3x ≈ 2,500 trials > exhaustive 1,860) — net cost depends on
  the 2p-solvable fraction; measure rate from a pilot before resizing shards if needed.
  **SMOKES #2 + #3 PASSED → SCALED [2026-06-12 ~05:15]:** #2 (10 feb scenes): 26/26 episodes solved, ALL
  early-stopped at attempt 1 (root ≤30) ✓. #3 (3 known-dead aug9 scenes): restarts fired 3x with fresh draws
  + MERGED logs (root trials 87/76/48; the 48 = only 16 reachable cells x3) ✓; dead cost ~1.0-2.3k sims as
  predicted. **FULL LAUNCH: job 55960285** — 110,824 remaining scenes (125,494 − 14,670 exhaustive-done),
  64 shards, `pkls_2push_s30` (kept separate from exhaustive `pkls_2push_unified` for the datasheet).
  M1 chain same night: pack DONE (123,269 npz → 4.7 GB src-h5) → join 55960012 RUNNING → cpx 55960013
  (afterok) → gate check → M1 train (RUN_PREFIX=m1_v4hq DATA_DIR=v4_hq_m1_scorer, array 9-11 = B30 x s1-3).
  M2b packs 55959968/9 running (105,864 + 23,672 = 129,536 dead-end npz).

- **M1 TRAINING LAUNCHED [2026-06-12 ~05:55]: job 55961140** (`m1_v4hq_s{1,2,3}`, B30 champion recipe,
  array 9-11). Data chain ALL GATES GREEN: join 123,269/123,269 unique episodes (solvable-only ✓),
  bad_match=0, gt_in_valid=100.00%, edge_align_err=0; contact_px N=123,269 miss=0 in-bounds=1.000;
  **composition 65.0:35.0 EXACT** (feb 80,125 / aug9_b1 25,938 + aug9_rest 17,206). Final H5:
  `/scratch/dm1487/h5/v4_hq_m1_scorer/data.h5`. ⚠ probe gotcha for the record: H5 `xml` paths are
  collection-shard symlinks (`outputs/v4_hq_*/shard_*/envs/...`) — discriminate feb/aug9 by OUTPUT ROOT
  (`/v4_hq_h1/` vs `/v4_hq_aug9_h1[_rest]/`), NEVER by 'feb_car'/'aug9_car' substrings (always absent).
  M1 verdict protocol: eval_scorer per ckpt → resolve_robust-style 3-seed paired compare vs champion 23.8
  hard@1 (pre-registered: at-or-above). Snapshot-feelers at ~ep8/~ep15 per [[feedback_periodic_feelers]].

- **H2 LABEL BUILDER READY [2026-06-12 ~07:45] (commit 95017b4):** `build_2push_validset.py` extended:
  (a) `frac_first_push` = [pe,pd,n_succ_2,n_tried_2] per expanded first push over UNIQUE child cells
  (restart-union; denominator is part of the label under sampling — 1/30 brittle vs 6/22 robust);
  (b) dead-end episode recovery via `_pose_from_xml` (the obs-only pose lookup DROPPED dead episodes —
  H0b pattern again; smoke-dead: 0→3 episodes, tried unions match smoke #3 exactly). Gamma targets derive
  downstream (1.0/γ/0 from valid_1push/valid_first_push/tried) — γ stays a tunable, not baked in.
  Validated on both restart smokes (dead + solvable). Ready to run on `pkls_2push_s30` + the kept
  exhaustive `pkls_2push_unified` when collection lands. M1 ep5 feeler (job 55961584) in flight.
- **M1 FEELER @ep5 [2026-06-12 ~08:00]: hard@1 = 25.4 — ABOVE champion 23.8** (m1_v4hq_s1
  epoch005-val_loss0.5833, eval JSON `/scratch/dm1487/eval/m1_feeler_s1_ep5.json`). Pre-registered
  at-or-above prediction CONFIRMING at 5/200 epochs; single-ckpt ±3-4 noise ⇒ preview only, M1 verdict =
  end-of-training 3-seed paired compare. Next feeler ~ep15.

- **✅ M1 PASSED [2026-06-12 ~12:10] — hard@1 = 29.40 ± 1.50 vs champion 23.27 ± 1.38 (+6.1pp, ALL seeds
  positive: +4.1/+7.8/+6.6).** Protocol: 9 top-val ckpts (3 seeds × ~3), full eval panel each
  (`/scratch/dm1487/eval/m1_verdict/`), per-seed means vs the registered newbar_verdict B30 numbers.
  Training: all 3 seeds early-stopped by recipe at best ep18-19 (~3.3h) — SAME behavior as the original
  champion runs (B30_s1 ~ep20, B30_s2 ep39, wandb-verified) ⇒ no confound. Attribution [CLAUDE]: train/test
  bar match (predicted ~5pp) + 25% more data (123k vs 98k, the proven lever) + 65:35 composition — all
  aligned, same direction. **CONSEQUENCES: (a) v4 data factory CERTIFIED end-to-end; (b) m1_v4hq = the new
  1-push baseline — M2a's "≈ champion" gate now reads "≈ 29.40 ± 1.50 (m1_v4hq)"; BASELINE REGISTRY +=
  m1_v4hq_s{1,2,3} (do not retrain); (c) M2a warm-start source = best m1_v4hq ckpt (s3 ep19 31.2 is the
  single best; use per-seed bests).** NEXT: M2a (budget_cond=true value_bins=51 head_mode=hl_gauss
  budget_h=true, H≡1, SAME H5) + M2b (same flags on v4_hq_m2b_scorer with dead rows) — M2a/M2b per the
  two-cell split (arch vs data isolation).

- **M2a LAUNCHED [2026-06-12 ~12:25]: job 55964116** (`m2a_v4hq_s{1,2,3}`, SAME M1 H5 + B30 recipe, flags:
  budget_cond=true value_bins=51 head_mode=hl_gauss budget_h=true, H≡1). **DEVIATION from the 37-spec
  [CLAUDE]: FROM-SCRATCH, not warm-start** — the spec chose warm-start when training was assumed expensive;
  measured cost is ~3.3h to early-stop, and from-scratch keeps the M2a cell to exactly ONE change vs
  m1_v4hq (head/conditioning), same init protocol. Warm-start remains the fallback if from-scratch
  underperforms. GATE: hard@k ≈ m1_v4hq 29.40 ± 1.50 (rankings via E[bin], monotone-invariant).

- **SAMPLED H2 COLLECTION COMPLETE [2026-06-12 ~14:20]: job 55960285 — 110,824/110,824 scenes, 64/64
  COMPLETED, ~7h wall** (vs 24h+ exhaustive trajectory; the [USER] k30+restarts recipe at scale).
  Total H2 inventory: `pkls_2push_s30` (110,824 sampled) + `pkls_2push_unified` (14,670 exhaustive remnant).
  **Label builds LAUNCHED:** 55967155 (sampled → labels_s30.json + pure-2push view) + 55967156
  (exhaustive remnant → labels_exhaustive.json + pure view) via the extended build_2push_validset
  (γ-derivable, frac_first_push robustness, dead-end XML-anchor recovery). M2a ep17+/M2b ep2+ training.

- **H2 LABELS DONE [2026-06-12 ~14:35] — SAMPLED ≈ EXHAUSTIVE AT THE POPULATION LEVEL (the [USER]
  ensemble-statistics argument empirically CONFIRMED):** sampled tree 140,062 eps = 16.2% 1p-solvable /
  **28.5% 2-push-only (39,871 setup episodes)** / 55.4% unsolved-≤2; exhaustive remnant 18,143 eps =
  15.6% / **28.5%** / 56.0% — identical composition incl. the 2-push discovery rate ⇒ k30+restarts lost
  nothing distributionally. Artifacts: `labels_s30.json` (+`labels_s30_pure2push.json`, 38,771 scenes /
  39,871 eps = the M3 informative subset), `labels_exhaustive.json` (+pure view). Note: 1p-solvables on
  "dead-end scenes" = OTHER episodes (object/goal pairs) than the H1-dead one — per-episode invariant as
  always. NEXT (Phase 3/4): post-push H=1 harvest + H2 H5 build (gamma + frac labels via the new fields,
  mixed-H batches) — after M2a/M2b verdicts.

- **✅ M2a PASSED [2026-06-12 ~15:10] — hard@1 = 29.62 ± 0.93 vs m1_v4hq 29.40 ± 1.50 (+0.2pp, per-seed
  +2.1/+0.9/−2.3 = within noise).** The budget-Q architecture (H-embedding + HL-Gauss 51-bin head,
  from-scratch, same H5/recipe as M1) preserves 1-push ranking exactly — the conditioning/value machinery
  is FREE. Evals: `/scratch/dm1487/eval/m2a_verdict/` (eval_scorer gained budget-Q ckpt support: auto-detect
  from state_dict, H=1 forward, E[bin] ranking — committed). M2b (same flags + 129,536 dead rows) training;
  its verdict = ranking ≈ M1 AND dead→low-V probe (top-k-mean V on dead vs solvable val rows).

- **⚙ STANDING RULE [USER 2026-06-12]: MAXIMIZE PARALLELISM — never serialize without a true data
  dependency.** At every state change, re-derive the dependency graph and launch everything whose inputs
  exist: machine-time steps submit immediately; code is written + smoke-validated WHILE compute runs;
  known-invocation chains use sbatch --dependency=afterok; verdicts gate ONLY their own downstream.
  (Also in auto-memory: feedback_maximize_parallelism.) Exceptions stand: no full training on unverified
  data; never touch sessions/allocations.

- **DEAD-SLICE CONTROL (M2a, ZERO dead training rows) [2026-06-12 ~17:25] — H0b PARTIALLY PRE-SOLVED:**
  probe (eval_dead_slice.py, 500v500, V=top5-mean, H=1): candidate-pool V_dead=0.31 vs V_solv=0.90
  (AUC 0.959); all-cells V_dead=0.71 vs 0.93 (AUC 0.907). Even without dead-end data, per-cell ZEROS on
  solvable scenes generalize to mostly-low maps on dead states — the OLD H0b "always believes something
  works" was a property of the old data/bar, not inherent. **BUT the untried-cell optimism leak is REAL
  and now MEASURED: +0.40 value inflation on dead states when pooling over all 300 cells (0.71) vs the
  candidate set (0.31).** ⇒ M2b's sharpened gate: close the all-cells gap (V_dead well below 0.71) —
  the direct test of whether dead-row masked supervision suppresses untried-cell optimism. Deploy note
  either way: pool V over the CANDIDATE set when reachability is known.

- **[USER 2026-06-12] MEASUREMENT RULE: value pooling + all eval ALWAYS over the candidate pool
  (r_mask=1 / wavefront-reachable at deploy; edge-level — a reachable contact point makes all 5 depths
  candidates).** r_mask=0 cells are inexecutable; scoring them is meaningless. The all-cells pool is kept
  ONLY as a no-mask robustness diagnostic. Restated M2b gate accordingly: (1) ranking ≈29.4 (primary),
  (2) candidate-pool dead separation improves (V_dead < 0.31 / AUC > 0.959 vs the M2a control),
  (3) all-cells gap closure = robustness bonus, not a gate. Note: even M2a (zero dead rows) separates
  dead states on the candidate pool — per-cell zeros generalize; M2b widens, not creates.

- **[USER 2026-06-12] M2c/M2d PARKED as representation-only experiments.** Deployment ALWAYS post-filters
  to reachable contact points (candidate-pool rule) — so unreachable-cell supervision (M2c: 30+20 mask)
  and the reachability input-flag (M2d) have NO deployment consequence; their only possible value is
  "learn better" (auxiliary signal for the encoder), judged by ranking metrics. Key semantic findings
  from the discussion: (a) "would it open if executed" is a well-defined COUNTERFACTUAL on all cells —
  M2c changes the question (operational = affordance x outcome), not completes it; (b) the robot_region
  input channel IS the wavefront rendered — M2c teaches a lossy pixel re-derivation of a bit M2d can
  hand over exactly (same source ⇒ NO independent robustness, correcting an earlier CLAUDE claim);
  (c) the one surviving M2c case = dense connectivity-reading supervision might transfer to the outcome
  task (the 20% bar is itself a connectivity question). Run on idle GPUs only, if at all.
  Reachability recording (bd54571) stands regardless — it serves any future variant + post-push rows.

- **M2c/M2d LAUNCHED as the reachability-signal ablation [USER hypothesis, 2026-06-12 ~18:10]:**
  "giving it reachability (M2d) or it learning reachability (M2c) sharpens scene understanding ⇒ better
  values / hard@k." PRE-REGISTERED: accept iff hard@1 > M2b's by the 3-seed paired compare (same H5
  v4_hq_m2b_scorer, same recipe, ONE change each). Side-quest on training-signal quality — deploy
  post-filters reachable regardless. Cells: **M2b** = base (in flight) · **M2c (55971858)** =
  +unreachable_k=20 (S30∪S20 mask, known-0s on unreachable; smoke: S30 byte-identical + 20 exactly
  on-unreachable) · **M2d (55971859)** = +reach_flag_input (per-edge bit embedding; the wavefront bit
  handed over instead of pixel-re-derived). All committed 56e44b0; secondary readouts: dead-slice probe
  (both pools), HL-Gauss bimodality on boundary cells (M2d), val_loss. Note val split scores full-R for
  all cells (comparable); M2c's mask change is train-only.

- **M2b FEELER [2026-06-12 ~18:45] (best ckpt s3 ep13, single-ckpt preview): ALL THREE READOUTS PASS.**
  hard@1 = 29.6 (M2a 29.62 / M1 29.40 ⇒ 51% dead rows cost NOTHING on ranking); dead-slice
  candidate-pool V_dead = **0.065 vs control 0.313 (5x lower)**, AUC 0.987 vs 0.959 — the H0b gate
  closing decisively; all-cells V_dead **halved (0.710 → 0.359)**, AUC 0.955 vs 0.907 — dead-row
  supervision generalizes to UNTRIED cells of the same state (whole-scene hopelessness), pre-empting
  part of M2c's hypothesized role WITHOUT any unreachable supervision. JSONs:
  /scratch/dm1487/eval/{m2b_feeler,dead_slice_m2b_feeler}.json. Full 3-seed verdict after wall-kill.

- **✅ M2b VERDICT [2026-06-12 ~19:00]: PASSED — dead-end data IMPROVES ranking. hard@1 = 32.86 ± 2.38**
  (s1 31.4 / s2 35.6 / s3 31.6) vs M2a 29.62 ± 0.93, M1 29.40 ± 1.50; @5 = 65.44 ± 1.25 vs 60.9;
  @10 = 80.07; @20 = 90.87. Dead-slice (per-seed best ckpts): cand-pool V_dead 0.061/0.071/0.065,
  AUC 0.989/0.983/0.987 (control 0.313/0.959); all-cells V_dead 0.26-0.36 (control 0.71). Evals:
  /scratch/dm1487/eval/m2b_verdict/. INTERPRETATION [CLAUDE]: 51% dead rows acted as massive negative
  supervision — junk families pushed down ⇒ fewer impostors at top-k (the @5 +4.5pp) AND top-1 gains.
  **M2 PHASE CLOSED: arch free (M2a) + dead data = pure gain (M2b). m2b_v4hq = the new best 1-push
  model / the warm-start + comparison baseline for Q-full (M4).** BASELINE REGISTRY += m2b_v4hq_s{1,2,3}.
  Note for the H0b record: the old champion's post-push saturation left only 2nd-3rd-decimal ordering
  (the 34.5 sliver, ≈floor by @10) — M2b's 13x dead/solvable value ratio is the corrected contrast.

- **[USER 2026-06-12 ~19:24] CONDITIONAL FOLLOW-UP LOGGED: if M2c/M2d show good results, RE-COLLECT the
  H=2 data so it carries reachability too** (the sampled H2 run pkls_2push_s30 predates the
  reachability_log recording fix bd54571 — its rows have no recorded reachable-edge sets, root or
  post-push). Notes for the decision when the verdicts land (~21:00 local): (a) cost is known-affordable
  now — the full 110k-scene sampled collection took ~7h; (b) a CHEAPER alternative exists: derive
  reachability retroactively from the npz robot_region channel + contact pixels (validatable against H=1
  rows where tried==reachable), avoiding re-collection entirely; (c) an ExIt round-2 re-collection is
  planned regardless — folding the reachability-carrying recollect into round 2 may get it for free.
  Decision deferred until the M2c/M2d verdicts say whether the signal is worth it.

- **📋 BASELINE PLAN FOR THE RESULTS WRITE-UP [logged 2026-06-12, execute when M3/M4/M5 results are in]:**
  No public benchmark for this setting ⇒ external comparison = re-implement algorithmic alternatives on
  OUR benchmark. **Internal (banked, never retrain):** random 11.8@1 · sims-matched random · geometric
  oracle (~6% hard) · champion B30 23.27 · M1 29.40 · M2a 29.62 · M2b 32.86 (+ dead-slice controls) ·
  old-champ+49sims 34.5 · M2b+49sims (tonight) · h1_pol policy-distillation arms (=BC baseline, registry).
  **Must add (cheap):** (a) Q-full's OWN H=1 head queried at the root, 0 sims — THE internal control for
  M3 (if H=2 head ≤ own-H=1 ranking, H-conditioning learned nothing); (b) marginal-prior ranker (global
  per-cell success frequency, no scene input — measures dataset-prior leakage in all scores); (c) expert-
  at-budget curve (region_opening at capped sims = the classical Stilman-lineage anchor for the M5 plot,
  far-right oracle ~750 sims). **One real new training:** IQL on the SAME data (TD vs our distilled-MC —
  the offline-RL reviewer baseline; spec pre-registers MC>TD at our horizon via TD-or-not-TD). **Adapted/
  positioning only:** Bejjani value-RHP (≈M2a+search, label as adapted protocol) · HACMan (arch absorbed;
  delta isolated by our own M1→M2a→M2b→M2c/d ladder) · MORE (different sub-question, no number).
  DECISION DEFERRED [USER]: queue IQL after Q-full vs park until paper push.

- **🚀 M2b+SEARCH (fpv protocol, preliminary 49/50 shards) [2026-06-12 ~22:25]: the 34.5 BASELINE
  DOUBLED.** Per-scene first-push pick (mean_top5, n=561 scenes with ≥1 good leaf, protocol identical to
  H0b incl. the good-leaf filter): **75.2 @1** / 83.2 @3 / 85.7 @5 / 89.3 @10 / 94.3 @20 (old champion:
  34.5/52.9/63.4/72.6/90.3 — and old was ≈floor by @10; new has signal at every depth). Leaf-level:
  48,171 first-push sims, 8,697 good leaves (18%); scalar AUC good-vs-dead 0.856 (maxP≈mean_top5).
  **Second stage on OOD post-push states (the rank_succ question): model's #1 second push opens 69.8%
  (median rank 0 = first try), @3 83.9, @10 95.2** — the old champion's post-push saturation failure is
  FIXED by generalization alone (M2b never trained on a post-push state). Dead-training did NOT flatten
  within-dead-state ordering. **END-TO-END 2-push solve (new metric, old system couldn't): top-1 first
  push + model-ranked second pushes = 61.7% at exactly 2 executed pushes; 69.0% ≤3 tries; 75.2% ceiling.**
  ⇒ the SEARCH regime with M2b is a working 2-push agent today (~49+verify sims/scene); this is the M5
  reference line. M3's registered gate stays 34.5 @ 0 sims; the honest modern bar for amortization is
  this 75.2-with-sims line. Files: /scratch/dm1487/eval/fpv_m2b/ + diag_fpv_aggregate.json. FINAL 50/50 confirmed identical: 75.2@1 / e2e 61.9 / 2nd-stage 69.8 (48,209 leaves).

- **PRE-REGISTERED Q-full diagnostic — the H-BIFURCATION PROBE [2026-06-12 ~22:40]:** take the SAME
  H=1-dead initial states from the dead-slice probe (where M2b correctly reads 0.065 at H=1; NOTE the
  population is "dead at H=1" — ~28.5% are 2-push-solvable) and query Q-full at H=2. PREDICTION: values
  BIFURCATE by ground-truth 2p-solvability — the 2p-solvable subset rises (setup cells toward γ=0.9),
  truly-deeper-dead stays at the floor; report the two sub-population means + AUC(2p-solvable vs dead-at-2)
  + the H=1-vs-H=2 delta per state. This is the H-conditioning's semantic test in its purest form, and the
  mechanism exhibit to sit next to M3's headline. Implementation: eval_dead_slice.py --h 2 + a split of the
  dead sample by the H2 labels key.

- **⏰ OVERNIGHT STATE MACHINE [2026-06-12 ~22:45 — READ THIS FIRST AFTER ANY COMPACTION]:**
  User AFK, full autonomy granted. RUNNING: h2-chain2 55972757 (packs ~done → twopush join ~2.5h → cpx;
  monitor bya9ik6d3) · m2c-train 55971858 + m2d-train 55971859 (6h walls ~00:10, monitor bvmvrbahb).
  **ON CHAIN2 COMPLETION → check gates in its log (unique-episodes/bad_match/gt_in_valid/edge_align_err;
  expect ~280k rows = 2 per episode, dead rows present) → LAUNCH Q-FULL:**
  `cd sage_learning && RUN_PREFIX=qfull_v4hq DATA_DIR='/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5;/scratch/dm1487/h5/v4_hq_h2_scorer/data.h5' EXTRA_OVERRIDES="+network.budget_cond=true +network.value_bins=51 +model.head_mode=hl_gauss +data.budget_h=true" sbatch --export=ALL,RUN_PREFIX,DATA_DIR,EXTRA_OVERRIDES --array=9-11 --time=14:00:00 --job-name=qfull-train scripts/train_h5_sampling.slurm`
  (multi-H5 datamodule f82ac6d handles the ';' list; 14h wall — epochs ~2x M2b's; ~530k rows total).
  Feeler at ~ep5 (Slack it). ON M2c/M2d WALLS → verdict evals (xargs pattern, m2c_verdict/m2d_verdict
  dirs) + dead-slice probes; verdict = 3-seed vs M2b 32.86 ± 2.38 [USER hypothesis pre-registered; if
  good → H2-recollect-with-reachability decision per 4b8b36f]. ON Q-FULL DONE (morning) → verdict suite:
  (a) eval_scorer panel ×9 ckpts (H=1 parity vs M2b 32.86); (b) M3 = zero-sim setup-pick on the pure2push
  slice via the fpv protocol with --scalars-only?? NO — M3 eval = rank first pushes by Q(s0,·,H=2) map
  directly (NO sims): needs a small eval (reuse eval_dead_slice-style load + the pure2push key valid_first_push;
  ~30 lines) — bars: 34.5 (registered) / 75.2-with-sims (tonight's line, c9d0ec5); (c) H-BIFURCATION probe
  (525ea31): same dead states, --h 1 vs --h 2, split by 2p-solvability; (d) dead-slice + post-push
  calibration check (0.549 → toward 0.065). Slack each verdict. Hourly cron continues. NEVER kill
  sessions/allocations; check squeue before any submit (idempotency).

- **2-PUSH DIFFICULTY DIVISIONS BUILT [USER ask, 2026-06-12 ~22:55]:** keyed on SETUP-CELL density from
  the canonical exhaustive key (1-push thresholds don't transfer — setups are rarer): **hard = 1-2 setup
  cells / medium = 3-8 / easy = >8** (371/409/238 of 1,018 episodes) →
  `namo_testset_v1/labels/pure2push_divisions.json` (per-episode `division` + `n_setups` fields).
  Tonight's M2b+search per division (n=429 matched; 345 leaves unmatched on (realpath,obj) join — likely
  multi-episode (xml,obj) region ambiguity + manifest⊃key, investigate at M3 time): hard 76.8@1 / med
  78.8 / easy 86.2; e2e@2pushes 59.9/62.2/73.4. **GRADIENT NEARLY FLAT ⇒ PRE-REGISTERED PREDICTION:
  Q-full's ZERO-SIM M3 gradient will be much steeper (recognizing a simulated child is easy; imagining a
  needle is hard) — the hard-division reactive-vs-search gap = the cleanest measure of what simulation
  still buys.** M3 verdicts MUST be reported per these divisions.

- **🔬 M2c/M2d VERDICT [2026-06-13 ~00:25]: [USER] hypothesis REJECTED on ranking; "TEACH BEATS TELL"
  discovered on robustness.** hard@1 3-seed: M2c 32.21 ± 1.48 / M2d 34.20 ± 2.09 vs M2b 32.86 ± 2.38 —
  neither clears the pre-registered bar (M2d +1.3 mean but paired deltas +1.0/−1.9/+4.9 = mixed-sign).
  ⇒ the encoder already extracts reachability from robot_region; explicit signal adds no ranking.
  **Dead-slice all-cells V_dead: M2b 0.327 / M2c 0.072 / M2d 0.621.** M2c (learning from 20 sampled
  unreachable zeros) ELIMINATED the hallucination zone — unmasked == masked value (0.072 ≈ 0.061): the
  model is self-contained, the deploy mask becomes optional rather than load-bearing. M2d (given the
  bit) DEGRADED unmasked behavior — legality outsourced to the flag is never learned by the net.
  CONSEQUENCES: (a) the [USER] conditional H2-recollect-with-reachability is NOT triggered (4b8b36f);
  reachability_log recording stays for other uses; (b) M2c's unreachable_k=20 = a FREE robustness
  upgrade — fold into round-2 recipes for rows where tried==reachable ONLY (H=1-type rows; naive
  application to sampled H=2 rows would be the C15 bug — needs per-row gating, deferred); (c) Q-full
  recipe UNCHANGED tonight (one-change discipline). Evals: /scratch/dm1487/eval/{m2c,m2d}_verdict/.

- **📌 STANDING RULE [USER 2026-06-13]: EVERY trained model goes in [horizon_q_model_registry.md](horizon_q_model_registry.md)
  the moment it trains** — its 3 best-val ckpt paths (the wandb-hash dirs are unrecoverable by glob), data
  H5, headline number, and eval dir. The main journal links to it; do NOT scatter ckpt paths in §9 prose.
  NEXT to register: Q-full (qfull_v4hq) on launch; fpv_m2c results when 56008453 lands.

- **[USER PRINCIPLE 2026-06-13] M2c-logic and the main pipeline are DECOUPLED TRACKS — never block each
  other.** IF fpv_m2c (56008453) shows M2c+search > M2b+search (intrinsic legality travels OOD, worth
  having) → recollect 2-push data with reachability_log baked into pkls (bd54571 records it going
  forward; OR derive geometrically from robot_region — cheaper, validatable on H=1 rows). BUT: the
  critical path (Q-full tonight, M3/M4/M5, early ExIt rounds) KEEPS ITERATING on the EXISTING M2b-style
  H2 data with NO M2c logic — do not stall the thesis for the robustness upgrade. When the
  reachability-carrying data is ready (days later is fine), fold M2c's unreachable-supervision into a
  LATER training round; it catches up to the main line, never gates it. TECHNICAL WHY the recollect is
  the gate: H2 rows are SAMPLED, so the complement of "tried" mixes unreachable (safe-zero) +
  reachable-but-unsampled (the C15 poison) — only reachability_log disambiguates, so M2c-on-H2 cannot be
  a mere flag on current data. (M2c on H=1-type rows where tried==reachable needs no recollect.)

- **fpv_m2c FINAL (50/50) [2026-06-13 ~01:05]: M2c ≈ M2b, edges to M2b — hypothesis REJECTED, no recollect.**
  @1 79.8 vs 80.6 · @5 88.4 vs 90.1 · e2e@2push 61.6 vs 63.0 · 2nd-push@1 71.7 vs 69.8 (M2c's only +, noise) ·
  OOD value margin 0.097 vs 0.106 (M2c THINNER). Intrinsic legality did NOT travel OOD as performance —
  mechanism: search pools reachable-only, where M2b/M2c were already equal. Per [USER] decoupling
  (4379917): NO 2-push reachability-recollect triggered; M2c stays a deploy-robustness feature only.
- **POST-M3 FORK [USER question, causality]: the residual hard-case failure is CONTACT-POINT (edge)
  selection — 60% of hard@1 misses are wrong-edge, ~90% of all misses; depth is solved (82% given right
  edge), reachability solved.** Diagnosis: Q(s,a) is a black-box (scene,push)->value with NO explicit
  forward model — it implicitly composes push->object-displacement->connectivity. Two failure kinds:
  (a) CONSEQUENCE-BLINDNESS (can't tell which edge sets up better w/o modeling what it produces) =
  causal/forward-model gap; (b) genuine ALIASING (scenes indistinguishable at 64x64) = resolution, not
  causality. **THE DECIDING MEASUREMENT: Q-full reactive-vs-search gap on the SAME wrong-edge hard
  scenes** — if search >> reactive there, it's consequence-blindness → add an EFFECT-PREDICTION auxiliary
  head (predict post-push reachable region / object pose; data already rendered as _step rows; HACMan-
  flow / model-based-lite). If reactive ≈ search, it's aliasing → resolution/FOV, not causality. Levers
  ranked: (1) effect-prediction aux head [cheap, data ready]; (2) H=2 distillation [in progress = Q-full];
  (3) latent world model TD-MPC2 [big]; (4) counterfactual/contrastive pairs for causal-feature isolation.

- **🔌 RESUME CHECKLIST [if the session/srun was killed & restarted — DO THESE FIRST, 2026-06-13 ~01:53 ET]:**
  In-memory monitors + the hourly Slack cron DIE with the session; SLURM jobs + journal + git survive.
  ON RESUME: (1) `squeue -u dm1487` + `sacct -j 56013237,56013312 -X` — check Q-full (56013312) state;
  if still RUNNING re-arm its monitor (confirm-alive + ep5-feeler + completion, see below); if DONE run
  the verdict suite. (2) Re-create the hourly Slack cron (CronCreate "7 * * * *" → squeue + journal §9 →
  Slack U07N1DR8S94, short, ET times). (3) Q-FULL VERDICT SUITE when it finishes (per a967c31 + registry):
  eval_scorer panel ×9 ckpts (H=1 parity vs M2b 32.86); M3 = zero-sim setup-pick on pure2push (rank by
  Q(s,·,H=2) map, NO sims) vs 34.5 (registered) & 75.2-with-sims (fpv_m2b) + per-division
  (pure2push_divisions); H-bifurcation probe (525ea31, eval_dead_slice --h 1 vs --h 2 on dead states split
  by 2p-solvability); post-push calibration (0.549→?). Slack + journal + REGISTER in
  horizon_q_model_registry.md. (4) fpv_qfull (2-push search, reuse diag_leaf_s1 --ckpt qfull) — the
  reactive-vs-search gap = the causality decomposition. (5) OPTIONAL pending [USER greenlight]: aliasing-floor
  measurement A (input-neighbor GT-disagreement on test crops; free; isolates aliasing vs consequence-blindness).
  Q-full launch cmd (if needs relaunch): RUN_PREFIX=qfull_v4hq DATA_DIR='...m2b_scorer/data.h5;...h2_scorer/data.h5'
  (Hydra-quoted in train_h5_sampling.slurm now) EXTRA_OVERRIDES="+network.budget_cond=true +network.value_bins=51
  +model.head_mode=hl_gauss +data.budget_h=true" array=9-11 time=14:00:00.

- **[USER directive 2026-06-13 ~02:20] POST-PUSH HARVEST — BUILD NOW, v2 NOT GATED ON v1.** Launch Q-full-v2
  (root + post-push) as soon as the post-push data is ready, regardless of v1's M3/M4 outcome. Datasheet
  the H5 when built (docs/pipeline/horizon_q_datasets.md). SOURCING (decided after npz inspection — good-s1
  npz has edge_idx_a1=opener + object_center but NO parent-a1, so its FULL grid is unrecoverable from the
  free render; design accordingly):
  • **GOOD post-push (~150k): FREE from the 781,881 rendered _step_1 npz** — sparse-positive label
    (cell edge_idx_a1=1, single opener; valid masked row = solvable post-push state). Sample ≤3/episode.
  • **DEAD post-push (~150k): REPLAY** — load XML→s0, execute the expanded-dead-a1 (from trial_log), render
    s1 via --include-dead-ends, full all-zero grid from kids[a1] (~48 tried a2, all fail). 782k validates replay.
  • Total ~300k post-push pool, tagged (horizon=1, state=post-push, outcome). TRAIN ratio = stratified
    WeightedRandomSampler (tunable), NOT concat — protect H=2 (~30-40%/batch), post-push ~15-20%.
  • Rationale: dead post-push (the 0.549-calibration fix) gets FULL supervision; good post-push (redundant
    w/ existing solvable) rides free sparse. Sample-don't-enumerate: cap per-episode, ~300k not ~3M.
  • COLLECTOR FUTURE-FIX (separate): persist expanded-node states so ExIt/H=3 never drop them (no replay next time).

- **⏰ AUTONOMOUS STATE [2026-06-13 ~03:05 ET — READ AFTER COMPACTION]:** Q-full **v1 = job 56015587** (16 workers/16 CPUs for dataloader throughput; superseded 56015450/56015450@8w)
  (3 seeds L40S, training, dataloader-bound ~12-14h, realpath-hang FIXED; monitor bevx43njy → ep5 feeler +
  completion). The dead 56013237/56013312 are CANCELLED false-starts. v1 verdict suite = a967c31 +
  registry gates. **[USER directive] BUILD POST-PUSH v2 DATA NOW; LAUNCH Q-full-v2 AS SOON AS ITS SMOKE
  PASSES (not gated on v1's M3/M4).** Datasheet the H5 (docs/pipeline/horizon_q_datasets.md).

  **v2 BUILD STEPS (post-push harvest, plan in the [USER directive 02:20] entry above):**
  1. GOOD post-push (~150k, FREE): sample from the 781,881 `_step_1` npz (manifest
     v4_hq_h2_postpush_npz.txt; cap ~3/episode); SPARSE label = cell (edge_idx_a1, depth_idx_a1)=1, the
     single recorded opener, loss_mask = just that cell. Tag H=1, state=post-push, dead=0. NEW small
     builder (npz → src-h5 via convert_to_hdf5 → sparse-label scorer H5 + add_contact_px).
  2. DEAD post-push (~150k, REPLAY): from trial_log, per episode pick expanded-dead-a1 (parent in
     depth-2 entries, ALL children fail); replay `env.step(Action(obj,a1.edge,a1.depth))` from XML→s0
     (namo_rl, deterministic ⇒ reproduces collector s1); render s1 via batch_collection --include-dead-ends
     (s1 = dead-end H=1 row anchored at post-push pose); label = all-0 over the ~48 tried a2 from kids[a1].
     SMOKE FIRST: replay a GOOD a1, verify its render ≈ the existing _step_1 npz for that scene (proves
     replay correctness). Only scale if smoke clean; else journal blocker, do NOT launch a broken v2.
  3. SAMPLER: WeightedRandomSampler in scorer_data keyed (H, state-type, dead), tunable; default protect
     H=2 ~30-40%/batch, post-push ~15-20%. NOT concat.
  4. LAUNCH v2: same recipe as v1 + the post-push H5(s) in the ';'-joined DATA_DIR + sampler config.

  **[USER ANALYSIS REQUEST — why success% plateaus (hard@1 ~33, 60% of hard misses = WRONG-EDGE)] —
  THE MOTION-EFFECT GAP (user hunch, CLAUDE concurs it's the strongest lead):** the model gets contact
  LOCATION (contact_px = Fourier PE of the contact pixel + edge_embed) but NOT the MOTION VECTOR each
  (edge,depth) induces — i.e. *which direction & how far the object moves*. That Δ (se2_target, object-
  local, from the primitive DB = "nominal effect in free space", perturbed by clutter) IS in the data
  (npz se2_target_a1) but is NEVER fed as a per-cell INPUT feature; depth is only an OUTPUT axis. So the
  model must re-learn (edge,depth)→object-motion from scratch. This maps directly onto the edge-selection
  failure: picking the right edge = knowing which push DIRECTION clears the corridor, which needs the
  motion vector. HACMan feeds the motion vector (continuous actor); we feed only WHERE, drop the HOW.
  **CONCRETE EXPERIMENT (post-v2, cheap, opt-in flag like budget_cond): add se2_target (Δx,Δy,Δθ object-
  frame) as a per-(edge,depth) input token feature; predict edge-selection improves (wrong-edge ↓).**
  OTHER DIRECTIONS to investigate + why: (a) effect-prediction aux head (predict s1 reachable region from
  scene+action — consequence-modeling, the causal route; data = post-push renders); (b) [USER] supervise
  the OPENED/terminal state (an "is-open" / H=0 connectivity head — grounds WHAT "open" means explicitly
  vs implicit-via-reward); (c) aliasing-floor measurement A (input-neighbor GT-disagreement — is the
  residual even reducible, or is 64x64 the ceiling? prior: resolution/FOV were FLAT levers ⇒ leans
  reducible); (d) v2 post-push (OOD calibration fix, in progress). RANK: motion-effect feature (a-hunch)
  > effect-pred head > opened-state aux > aliasing-floor (cheap, run first to bound the rest).

- **⛔ DEAD POST-PUSH REPLAY IS BLOCKED [2026-06-13 ~03:30, USER check caught it].** Validated replay vs
  GROUND-TRUTH recorded post-states (solution-path post_action_state_observations; smoothing RULED OUT —
  smoothing_stats None, original_* empty = raw). PERFECT correlation: **free-space pushes 298/298 replay
  EXACT (<5mm); collision pushes 47/47 DIVERGE (50-160mm).** env.step reproduces the free-space primitive
  deterministically, but collision dynamics don't reproduce (robot approach/contact differs slightly ⇒
  different collision ⇒ different s1). So replayed dead s1 does NOT match the s1 whose a2-labels are in the
  trial log, for the ~50% of pushes that collide (often the cluttered/interesting ones). REPLAY-BASED DEAD
  POST-PUSH = INVALID. (replay_postpush.py kept as artifact, marked free-space-only/blocked.)
  **PIVOT — this REVERSES the earlier replay>recollect call (which assumed replay correct):** the CORRECT
  way to get dead post-push states is to SAVE them at COLLECTION time (the collector future-fix: persist
  every expanded-node state observation, extend reachability_log bd54571 to full state) + RE-RUN the H2
  collection. A fresh sampled collection gives a different tree, so rebuild root + good-postpush + dead-
  postpush all from the same coherent new trees. This is a re-collection (~7h H2 + render), NOT a repack.
  **GOOD post-push (the 781,881 SAVED renders, sparse-positive) is UNAFFECTED — those are the collector's
  real s1, valid.** v2 STATUS: HELD. good-only v2 would add solvable post-push but NOT fix the dead-leaf
  calibration (0.549, the main point) — marginal. Recommend: HOLD v2 for correct dead states via
  re-collection [USER decision — significant compute + collector change]; v1 (root-only, 56015587) +
  its M3/M4 verdicts are the near-term result. DO NOT autonomously launch a re-collection or a partial v2.

- **[ANALYSIS, developed 2026-06-13 ~03:35] WHY success% plateaus — the motion-effect decomposition,
  sharpened by tonight's collision finding.** The residual failure is edge (contact-point) selection (60%
  of hard@1 misses; depth 82% solved, reachability solved). The push EFFECT = nominal free-space primitive
  motion (deterministic fn of edge,depth = se2_target) PERTURBED by collisions (tonight: free-space 298/298
  exact, collision 47/47 diverge — the effect is genuinely collision-dependent). The model gets contact
  LOCATION (contact_px) but NOT the motion effect. So edge-selection failure splits cleanly:
    (A) NOMINAL-DIRECTION blindness — the model doesn't know "edge 7 pushes the object NW, edge 23 pushes
        it E" without re-deriving it. FIX: feed se2_target (Δx,Δy,Δθ object-frame, from the primitive DB,
        already in npz se2_target_a1) as a per-(edge,depth) INPUT token feature. Free, opt-in flag like
        budget_cond. Predicts wrong-edge ↓. THE TOP LEAD (user hunch).
    (B) COLLISION-PERTURBATION blindness — for cluttered pushes the effect deviates from nominal; the model
        must read the scene to predict the deviation. This is the consequence/forward-model part → effect-
        prediction aux head (predict post-push reachable region from scene+action). Harder.
  Tonight's collision result QUANTIFIES the split: ~50% of pushes collide, so (B) is ~half the problem and
  (A) the other half. Test order: add se2_target feature (A, cheap) FIRST — if wrong-edge drops, nominal-
  direction was the gap; the residual after is (B)/aliasing. Then aliasing-floor measurement (is the (B)
  residual reducible at 64x64, or observation-limited? prior: resolution/FOV FLAT ⇒ leans reducible ⇒
  effect-pred head). [USER also raised: supervise the OPENED/terminal state at "H=0" — an is-open
  connectivity head grounding WHAT open means; speculative, lower priority than A/B.] All gated after
  v1's M3/M4 (which tell us if foresight distills at all before adding features).

- **✅ v2 COLLECTOR STATE-SAVING VERIFIED [2026-06-13 ~04:05]** (commit: region_opening node
  state_observation). Smoke1 (3 scenes, 46 eps, 625 post-push nodes): state+object_id present 100%;
  post-push captured for BOTH good (183) and dead (442) first pushes; 625/625 label-aligned (each s1 has
  its a2 outcomes from the trial log = correct-by-construction, the replay-collision-divergence bug CANNOT
  occur since we save the actual collector state); root s0 captured 46/46. The 51/625 "non-moved" post-push
  states all explained: 4 stuck/collision + 47 NO-EFFECT pushes (replay-confirmed: env.step also moves 0mm
  → push genuinely engages nothing → s1==s0 faithful, NOT a capture bug). ⇒ render path must FILTER
  no-effect post-push states (object pose == s0; redundant with root). Smoke2 (10 scenes, 56017972): ALL GREEN (58 eps, 1418 post-push, good 298 + dead 1120, 100% aligned). FULL v2 RE-COLLECTION LAUNCHED: job 56018429 (125,494 scenes, 64 shards, pkls_2push_v2, ~7h). Monitor b1xdohzj4.
  **FULL v2 RE-COLLECTION (ready, gated on smoke2):** MANIFEST v4_hq_h2_deadend_scenes_unified.txt (125,494
  scenes) → PKL_SUBDIR=pkls_2push_v2, CONFIG sampled_depth2_k30.yaml, 64 shards, 24h wall (~7h). Then
  NEW render-from-saved-state path (reads node_log state_observation, renders each post-push s1 + label
  from kids[parent_a1]; good=openers, dead=all-0; FILTER no-effect; adapt replay_postpush.py MINUS env.step)
  → root + post-push + dead H5 (tagged H/state-type/dead) → stratified sampler → Q-full-v2. v2 NOT gated on
  v1 [USER]. v1 (56015587) training; **ep5 H=1 feeler: hard@1=29.1 @5=61.9** (M2b 32.86/65.4) — healthy/early, H1 head sharing capacity w/ H2, expected to climb by ep15+. NOTE: H=1 sanity only, NOT the M3 foresight number (that needs the zero-sim H=2 setup-pick eval, built before completion).

- **📊 ep15 H=2 EVAL SUITE — FORESIGHT CONFIRMED, FLOOR-ANCHORED [2026-06-13 ~10:48 ET] (pure2push n=985).**
  Two evals built this session: (A) **key-graded first-push hit@k** (`eval_m3.py --grade key`, zero-sim,
  vs exhaustive `pure2push.json` setups + random floor); (B) **reactive ROLLOUT** (`eval_rollout.py`, the
  FAITHFUL deploy-matched number — model drives, sim only committed pushes, grade by reachability, NO key).
  • **(A) key-graded (3 seeds): H=2 hit@1 = 20.6/19.4/19.5 (~19.8), H=1 = ~3.4, FLOOR = 3.6.** ⇒ H=1 is AT
    chance (3.4≈3.6; at @10 it's 14.5 < floor 23.6 = ANTI-correlated, it promotes "opens-in-1" pushes that
    are NOT setups); **H=2 = 5.5× floor.** Budget token flips the SAME net from chance/anti-correlated to
    5.5×-chance at finding setups. The foresight claim, anchored.
  • **(B) reactive rollout Q (3 seeds, foresight-on, budget-aware): solve@1 = 22.8/22.7/23.3 (~22.9),
    @10 ~40.1, ~14 sims/scene.** TIGHT across seeds (0.6pp). Slightly ABOVE the key proxy (19.8) because
    the key is SAMPLED (k≈30) and misses setups the rollout actually simulates — so rollout is the more
    faithful number, NOT a ceiling violation. This is THE honest 2-push reactive solve rate.
  • Eval methodology (the frame, [USER] "understand this first"): decision-sims vs grading-sims; first-push
    PROXY (key) over-credits (ignores real 2nd push from OOD s1 = v2 target) but ALSO under-counts (sampled
    key) — rollout resolves both. MUST-BEAT bars: random-policy rollout (floor, ~11:00) + flat-H1 rollout
    (foresight-off). Search (Q-beam w2=10 vs uniform-beam) = the amortization ceiling/curve (~11:40).
    Tools: eval_m3.py(grade=key,+floor), eval_rollout.py(--prior q|uniform,--flat-h1,--w2), m3_key_feeler +
    rollout_eval slurms. ckpts = ep15 best (s1 0.6534/s2 0.6518/s3 0.6543; save_top_k prunes — transient).
  • CAVEAT: H=1 sim bars (fpv 75.2 / champ 34.5) are SEARCH (49 sims) + first-push-graded — NOT comparable
    to the 0-sim rollout; they bound the search ceiling, a different axis.

- **🎯 [USER DIRECTIVE 2026-06-13 ~14:00 ET] — 2×2 MODEL MATRIX + H=2-MUST-ENCOMPASS-H=1 DATA FIX.**
  TRAIN 4 models: {Horizon, NoHorizon} × {v1 data, v2 data}. Done/in-flight: Horizon-v1 = qfull_v4hq (ep15);
  NoHorizon-v1 = qfull_nohz_v4hq (training 56025708). TODO when v2 data ready: **Horizon-v2 + NoHorizon-v2**
  (3 seeds each). Then **TEST ALL 4 on the test set + report stats** (eval_scorer H1&H2, best-first pure2&pure1,
  key-graded). **DATA FIX [USER, the core ask]: make H=2 ENCOMPASS H=1** — Q(s,a,H=2)="value given 2 pushes
  left", and solvable-in-1 ⊆ solvable-in-2, so a 1-push opener MUST be 1.0 at H=2. It isn't now (only 16% of
  H=2 rows are 1-push-solvable → dilution 38→14 on 1-push). FIX in the v2 mix: (1) AUGMENT H=2 with 1-push
  scenes — relabel exhaustive 1-push openers as H=2 rows (opener=1.0, rest masked; FREE, no new collection);
  (2) BALANCE the sampler so a DECENT fraction of OnePush samples flow in (not fully balanced — keep 2-push +
  post-push, but enough 1-push that H=2 sees openers); (3) keep post-push OOD (dead-leaf calibration). GOAL
  [USER clarified]: **OPTIMIZE FOR OOD** — the two OOD failure modes: (a) post-push states s1 (dead-leaf), (b)
  1-push scenes queried at H=2 (OOD for the H=2 head, 16% seen → dilution). v2 mix over-represents BOTH
  (post-push data + 1-push@H2 augmentation), not full balance = "decent amount of OOD samples from OnePush".
  EXEC: render→pack→build balanced v2 mix→train H-v2 + NoH-v2→test all 4→Slack stats. v2 NOT gated on v1.

- **🧪 NO-HORIZON ABLATION LAUNCHED [2026-06-13 ~11:15 ET, job 56025708, [USER] "do we even need the horizon?"].**
  [USER hypothesis: a single model on all 1+2-push data, NO horizon, just learns "what's a good push" and
  that may suffice.] IDENTICAL recipe to Q-full v1 (same ';'-joined m2b+h2 data, HL-Gauss value head,
  B30 sample_k=30, 3 seeds) EXCEPT budget_cond OFF + budget_h=false → predicts the gamma-valued f_grid
  (opener=1.0/setup=0.9/fail=0) WITHOUT the H token. Run `qfull_nohz_v4hq_s{1,2,3}`. The clean control for
  "does the horizon pull its weight." WHAT THE HORIZON BUYS (honest, the only 3 things): (1) budget-honesty
  (knows what's achievable in the budget you HAVE; can say "unsolvable-in-1" / "1 push suffices" — the no-hz
  model can't); (2) a BOOTSTRAPPABLE value for deeper horizons (Q(s,a,H)=[opens]∨V(T(s,a),H−1) → distill to
  3/4-push, the ExIt arc) — flat goodness has no remaining-depth notion; (3) distance-to-goal gradient
  (γ^steps) = better search heuristic. The hz may NOT be needed for raw FIRST-PUSH RANKING — a pooled
  goodness model (opener=1.0>setup=0.9>fail) ranks openers first when present, setups when not, AND avoids
  the per-horizon imbalance that made our H=2 head dilute (so it could be MORE robust across 1/2-push).
  CAVEAT in the data: a setup push has CONFLICTING labels across rows (0 in its H=1 row, 0.9 in its H=2 row)
  → no-hz model averages to ~0.45; ranking order (opener>setup>fail) preserved, absolute values muddied.
  COMPARE on: rollout (reactive+search) + onepush (H-equiv) + pure2push. Same seeds → paired.

- **⚠️ KEY FINDING — H=2 does NOT subsume H=1; budget-matched/cascade deploy required [2026-06-13 ~10:53 ET,
  jobs 56025283/4, eval_scorer --h].** [USER] intuition "H=2 captures 1push too" tested on the ONEPUSH set
  (3 seeds, ep15). **budget-Q @H=1: hard@1 = 36.5/40.2/38.6 → MEAN 38.4 (vs M2b 32.86 = +5.5pp), easy@1 ~98.8
  — one model BEATS the 1-push specialist when queried at the right budget (no regression, an improvement).
  budget-Q @H=2 on the SAME 1-push scenes: hard@1 = 11.1/15.3/14.8 → MEAN 13.7, easy@1 ~86.8 — DILUTES HARD
  (−25pp hard, −12pp even on EASY).** [CLAUDE prediction "H=2 holds because opener target 1.0 > setup 0.9"
  = FALSIFIED.] The model learned H=1/H=2 as DISTINCT modes, not nested: at H=2 it hunts setups and
  deprioritizes the 1-push opener. **CAUSE [CORRECTED — [USER] caught my wrong "deadend-only" claim; VERIFIED
  in v4_hq_h2_scorer]: NOT absence, IMBALANCE.** The H=2 rows ALREADY include 1-push-solvable scenes —
  **16.2% have an opener cell =1.0 (~25k rows), 28.7% setup-only (0.9), 55.1% dead.** The head HAS seen
  "opener=1.0 at H=2" but it's a 16% minority vs 84% setup/dead → dominated by hunt-setups/it's-dead → under-
  ranks openers on the all-1-push test set. (Reconciles both facts: H=2 nails setups on pure2push = 5.5×
  floor where setups ARE the answer; dilutes on 1-push where openers are, and are the minority signal.)
  IMPLICATIONS: (1) DEPLOY = budget-matched query or CASCADE (H=1 first → opener? else H=2), NOT a universal
  H=2. (2) LEVER for v2 [USER decision] = REBALANCE not re-collect: up-weight the 16% 1-push-solvable H=2
  rows in the WeightedRandomSampler (cheap, testable, no new data; maybe also up-weight HARD 1-push — the 16%
  may skew easy, unverified). (3) combined-set rollout must use cascade ranking (max over budgets), not
  rank-first-push-at-H=B (underperforms on easy). [CLAUDE over-claimed "deadend-only" TWICE before checking.]

- **🔧 RENDER-FROM-SAVED-STATE BUILT [2026-06-13 ~06:45 ET] — `scripts/pipeline/render_postpush_from_state.py`** (autonomous, while v2 collects). OFFLINE renderer (NO env, NO MuJoCo step ⇒ collision-divergence bug structurally impossible — renders the collector's byte-saved state). Spec VALIDATED against real v2 pkls (job 56018429 output `/scratch/dm1487/datasets/v4_hq_h2/pkls_2push_v2`): per episode, object constant ⇒ key kids by (pe,pd); region-goal duplicate post-push nodes are byte-identical (Δpose=0.00000) ⇒ dedup keeps one; no-effect filter = SE(2) (xy<5mm AND |Δθ|<3°), NOT xy-only (rotation-only is a real effect). Each post-push s1 self-carries its label: pp_open_ed/dp (f_grid=1), pp_tried_ed/dp (the ~k sampled a2 = r_mask/loss_mask, rest UNKNOWN/masked — no C15 bug), pp_dead, pp_H=1, pp_parent_edge/depth, pp_reach_edges. **CAUGHT BUG before it bit: `save_episode_data` persists only a HARDCODED whitelist of `metadata` keys — pp_* in `meta` would be silently dropped (replay_postpush.py had this latent too). FIX: inject pp_* into the `masks` dict (save_dict=dict(masks) copies every key).** Smoke (5 pkls) in flight to confirm pp_* land in npz. NEXT (mechanical, after smoke green): `build_postpush_h5.py` = trimmed build_scorer_dataset reading raw npz → scorer H5 (ctx = local_tight_* 5ch resized 224→64 INTER_AREA; f_grid from pp_open; r_mask=pp_tried per existing tried≡r_mask convention; contact_px via add_contact_px's contact_px(); tag state_type=postpush, dead, H=1). **DEFERRED to USER (data-balance calls they own): final v2 H5 COMPOSITION + sampler ratios (existing root-H2 + H1 + new post-push good/dead; "protect H=2 ~30-40%", "don't blow up data") + LAUNCH.** Do NOT pick ratios or launch autonomously.

- **🎯 v1 ep11 H=1 FEELER CLEARS THE BAR [2026-06-13 ~07:43 ET, job 56022469, eval_scorer_feeler.slurm].**
  3 seeds best ckpts (ep11, val_loss 0.657–0.658, still dropping). hard@1 / @5 / med@1 / easy@1:
  s1 34.4/65.1/83.3/98.3 · s2 41.3/67.2/84.7/98.6 · s3 30.7/68.3/84.2/98.9 · **MEAN 35.5 / 66.9**.
  **vs M2b (H=1-only champion) 32.86 / 65.4 ⇒ +2.6pp @1, +1.5pp @5 — the budget-Q model MATCHES/BEATS
  the H=1 champion AT H=1.** Climb from ep5 (29.1/61.9). ⇒ H-conditioning + H=2 data does NOT tax H=1
  capacity (the capacity-sharing worry FALSIFIED); it slightly helps. Seed spread 30.7–41.3 = the known
  ±3–4 hard@1 noise; mean is the signal. **DIAGNOSTIC: wrong-edge/miss ≈88.8% mean, depth-acc|rightEdge
  99.2%, rank-1st-valid median 1.0, score-sep margin 0.144 (99.5% positive)** ⇒ model is SHARP; the entire
  residual hard miss is contact-EDGE selection (not depth, not reachability, not ranking) = the MOTION-
  EFFECT GAP (se2_target feature = confirmed top post-v2 lever). NOTE: H=1 sanity ONLY — the M3 zero-sim
  foresight number (eval_m3.py at completion) is the headline. Registered in horizon_q_model_registry.md.

- **✅ POST-PUSH PIPELINE VALIDATED END-TO-END + COMMITTED [2026-06-13 ~07:30 ET, commit 6155032 + render slurm].** Fixed render (pp_* in masks dict): all 10 pp_* keys land. 5-pkl smoke: 598 rendered (good 308 / dead 290), 132 no-effect filtered, 0 skipped. Packer (`build_postpush_h5.py`) on 203 npz → 203 rows (good 66 / dead 137), 0 bad. **Coherence ALL CLEAN: f_grid>0 outside r_mask=0; positives in dead rows=0; good rows mean 4.23 positives (sparse); contact_px 100% in-bounds [21,43]; H all=1, postpush all=1.** Real `ScorerDataModule(budget_h=True)` consumes it → batch dict {context,f_labels,loss_mask,r_mask,cp_reachable,ratio,H,contact_px} = exactly ClassifierModule's expected input. Schema is a SUPERSET of the root H5s (m2b_scorer 252k + h2_scorer 311k) ⇒ `;`-joins cleanly; xml grouping keeps each scene's post-push s1 on the same split side as its root (no leak). **Full-scale render driver READY (not submitted): `scripts/amarel/render_postpush.slurm`** — CPU main/main-redhat, slices PKL manifest by array task, bash-fans across 32 CPUs (renderer is single-proc), MAX_PER_EPISODE env [USER tunable — bounds total post-push volume; K≈3 over 125k scenes ≈ ~375k balanced pool, the journal's ~300k target without blowing up vs 564k root]. **WHEN v2 COLLECTION LANDS (~9:10 ET):** build PKL manifest from pkls_2push_v2 → submit render_postpush.slurm (decide MAX_PER_EPISODE) → build_postpush_h5 per shard → merge → datasheet → present composition/sampler to USER → launch Q-full-v2.

## 9.1 READY-TO-RUN when collection (job 55944720) finishes
```bash
# 1. manifest of v4_hq_h1 pkls
find /scratch/dm1487/outputs/v4_hq_h1 -name '*_results.pkl' > /scratch/dm1487/manifests/v4_hq_h1_pkls.txt
# 2. per-episode validset WITH dead-ends (H0b) — the --keep-dead-ends flag is the fix added this session
/scratch/dm1487/envs/namo/bin/python scripts/pipeline/build_episode_validsets.py \
   --manifests /scratch/dm1487/manifests/v4_hq_h1_pkls.txt \
   --out /scratch/dm1487/datasets/v4_hq_h1/episodes_deadends.json --keep-dead-ends --workers 32
# 3. scorer H5 = JOIN masks (src-h5) + the new f_grid labels (episodes). NOTE: masks lifted from DiT solution
#    data have NO dead-end scenes (task #23) -> this H5 = SOLVABLE scenes only until masks are rendered for dead-ends.
/scratch/dm1487/envs/namo/bin/python scripts/pipeline/build_scorer_dataset.py \
   --src-h5 /scratch/dm1487/h5/v3_balanced_1to1_lzf_tight_data/data.h5 \
   --episodes /scratch/dm1487/datasets/v4_hq_h1/episodes_deadends.json \
   --out-h5 /scratch/dm1487/h5/v4_hq_h1_scorer/data.h5
# then add_contact_px.py (contact_px 60x2), then train.
```
**Training wiring (sage_learning `classifier_module.py` + `scorer_data.py`):** add `head_mode="hl_gauss"`
(uses `src/model/hl_gauss.py` on gamma targets + `loss_mask`); pass `H` through `forward` (currently calls
`network(x, contact_px, ...)` — add H); `scorer_data` emits `H` (=1 for H=1 rows) + gamma target (= f_grid in
{0,1} for H=1). For M1/M2: train budget-Q with `budget_cond=True, value_bins=51, H≡1`, gamma=f_grid; verify ≈
champion hard@k via `eval_scorer.py` (default episodes = `namo_testset_v1/labels/onepush_episodes.json`).

## 10. Open tunables (pin by experiment)
γ exact value · k₂ (2nd-push breadth) · verify→bootstrap schedule · informative-subset threshold · dead-end ratio ·
#ExIt rounds · recall-tilt timing · one-head-vs-split.

## 11. References (closest neighbors — full reading list Slacked)
MORE (arXiv:2202.01426) · Bejjani RHP (1803.08100) · HACMan (2305.03942) · Go-Exploit/Targeted Search Control
(2302.12359) · Soemers ExIt-distribution (2006.00283) · DeepCubeA (Nature MI'19) · Ferber budget-escalated heuristic
(ICAPS'22) · SAVE (1912.02807) · ExIt (1705.08439) · MuZero Reanalyze (2104.06294) · Stop-Regressing (2403.03950) ·
CQL/IQL (2006.04779/2110.06169) · Pathak disagreement (1906.04161) · TD-or-not-TD (1806.01175). **Novelty:** no prior
work combines ExIt + disagreement-acquisition-for-setups + continuous manipulation + verify-before-bootstrap leaf.

<!-- M3-EVAL SCOPING (build fresh, runway until v1 completes ~early afternoon ET) -->
- **M3 ZERO-SIM FORESIGHT EVAL — scoped, build fresh:** reuse the fpv/LiveScorer infra (scorer_beam.py
  LiveScorer already handles budget-Q forward, extended earlier), NOT a from-scratch eval. Per pure2push
  scene (manifest test_pure2push_combined.txt): render s0 LIVE, query Q(s0,·,**H=2**) — ZERO sims (no
  first-push simulation, unlike fpv which simulated) — rank reachable first pushes, grade hit@k vs the
  pure2push key's valid_first_push (setups). Per-division (pure2push_divisions: hard≤2/med 3-8/easy>8).
  Bars: 34.5 (registered, old champ + 49 sims) / 75.2-with-sims (fpv_m2b first-pick). Also run the
  H-BIFURCATION probe (525ea31) + H=1 parity (eval_scorer, done at ep5=29.1). This is THE headline; verify
  ctx render matches fpv's (the LiveScorer renders s0 the same way). Run on v1's per-seed best ckpts at
  completion → full M3/M4 verdict.
- **M3 EVAL TOOL: scripts/sandbox/eval_m3.py** (force-added, sandbox gitignored). Zero-sim H=2 foresight: rank first pushes by Q(s0,.,H=2), verify top-k by sim. Run per-seed at v1 completion: --start 0 --end 985 --h 2 --topk 10. --h 1 = reactive-1push control. Smoke 56021488 (5 scenes, ep7).
- **M3 EVAL SMOKE PASSED [2026-06-13 ~06:15 ET]:** eval_m3.py runs end-to-end (5 scenes, ep7 ckpt, no errors, hit@1=60 — NOT meaningful, n=5+early, tool-validation only). Ready for full run at v1 completion: per-seed best ckpts, --start 0 --end 985 --h 2 (+ --h 1 control), per-division.

### 🔬 DEEP FAILURE ANALYSIS [2026-06-14 ~11:15 ET, corrected region eval, seed-1; scripts/sandbox/analyze_failures.py]
**HEADLINE: failures are RANKING failures, NOT intrinsic hardness.** Of 1018 episodes, only **16 (1.6%) are unsolved
by ALL 4 models** (truly unsolvable within 900 sims); **155 are model-specific** (one model fails, another solves).
So the 5-11% failure tails are the model's Q failing to rank a FINDABLE needle into the top-900, not the scene being
too hard.
- **Needle EXISTS in the failures:** unsolved (Hz-v2) have median **3 solving (a1,a2) pairs** (vs 13 for solved) /
  0.1% density (vs 0.8%) — rarer, but only **8/52 are true ≤1-pair needle-in-haystack**. The other 44 have a
  findable needle the ranking missed.
- **Failures concentrate in the HARD division:** Hz-v2 hard 89% vs med/easy 98%; NoHz-v2 hard 82% vs 97%. Horizon >
  NoHorizon in every bin; v2 > v1 in every bin.
- **92% of NoHorizon-v1's failures are solvable by RANDOM ordering** (103/112); Hz-v2 83% (43/52). ⇒ deterministic
  guidance steers AWAY from findable needles on its blind spots — random diversity beats it there. **Actionable:
  ε-exploration / stochastic restarts / ensembling would recover most of the tail** (the needle is reachable, the
  greedy ranking just doesn't surface it).
- **REACTIVE gap (NoHz>Hz @2) is on EASY/MEDIUM, not HARD:** Hz-v2 @2 easy35/med23/hard18 vs NoHz-v2 easy53/med34/
  hard19. The unconditioned head's robustness wins the EASY single-shot pick (many setups, pick a working one fast);
  the horizon's capacity-split costs reactive sharpness exactly there — hard reactive is ≈ (both ~18-19%, rare setups).
- **Mechanism summary:** Horizon trades reactive-easy sharpness for search efficiency + fewer ranking blind spots
  (8 unsolved-only vs NoHz 42). The remaining failure tail is recoverable via exploration, not bigger budget.

### ✅ 2-SEED ROBUSTNESS — all 2×2 findings HOLD [2026-06-14 ~11:35 ET, s1 vs s2, corrected region eval]
| cell | seed | rH1 | rH2 | s@2 | s@50 | s@100 | s@900 | avgSim |
|---|---|---|---|---|---|---|---|---|
| Horizon-v1 | s1/s2 | 34.4/40.2 | 12.2/15.3 | 22.3/25.9 | 71.5/71.2 | 81.8/79.3 | 93.8/91.7 | 58.5/59.7 |
| NoHorizon-v1 | s1/s2 | 21.2/25.4 | 21.2/25.4 | 28.7/29.2 | 63.1/64.9 | 71.4/71.8 | 89.0/89.2 | 85.1/81.8 |
| Horizon-v2 | s1/s2 | 36.0/35.4 | 30.7/25.4 | 24.2/24.8 | 76.3/75.0 | 82.6/81.3 | 94.9/94.1 | 54.6/55.2 |
| NoHorizon-v2 | s1/s2 | 31.7/29.1 | 31.7/29.1 | 32.6/33.4 | 67.5/69.7 | 74.0/76.9 | 91.6/92.2 | 76.7/66.0 |

**EVERY key finding holds with CONSISTENT SIGN across both seeds (no flips):** (1) NoHz wins reactive @2 in both
data versions both seeds; (2) Horizon wins search @50-100 + efficiency both versions both seeds; (3) v2 fixes Hz
H=2 (30.7/25.4 vs v1 broken 12.2/15.3); (4) Horizon > NoHz @900 both seeds. Seed variance ~2-6pp per cell, no sign
flips ⇒ the single-seed 2×2 conclusions are ROBUST. (s3 available if publication error bars wanted.) ⇒ **2×2 DONE
+ seed-confirmed; deploy = reactive→NoHorizon-v2, search→Horizon-v2.**

### 🎯 Q-VALUE AUDIT — the bottleneck is H=1 second-push value [2026-06-14 ~11:55 ET, n=20→150; analyze_qvalue.py]
Traced top-H=2 a1 → sim → top-H=1 a2 vs the exhaustive (a1,a2) GT, Hz-v2 & NoHz-v2. **CORE: H=2 setup-selection is
GOOD; H=1 second-push value is the FAILURE.**
- **H=2 picks setups (good):** setup-vs-nonsetup AUC=0.86 (Hz); first GT-setup at median rank 1.5; H2(a1) corr 0.59
  with max-H1(s1) ⇒ coherent handoff. The FIRST push is not the problem.
- **H=1 does NOT translate (the failure):** H=1 Q is COMPRESSED to [0.4,0.8] (never >0.8). Calibration: Q .4-.6→4%
  open, Q .6-.8→31% open (signal exists, 8× enrichment, but weak/under-confident). On a confirmed setup the top-H1
  a2 opens only 40%(Hz)/20%(NoHz); first opener at median rank 3-4.5. ⇒ model reaches s1 then mis-picks the opener
  60-80% of the greedy first try = the gap between 94% search-solvable and ~25% reactive.
- **Search:** sound but V=mean_top5(H1) is built on the compressed H1 ⇒ can't separate real vs fake setups ⇒ greedy
  starves the needle (the 83-92%-random-solvable result). Fix H1 calibration ⇒ the greed becomes justified.
- **NoHz>Hz reactive:** reactive is gated by the 2nd push (H1), BOTH models' weakness ⇒ horizon's H2 edge buys
  nothing reactively; the single head is marginally sharper on the easy 2nd-push pick. Not "horizon worse" — reactive
  is decided where both are weakest + single head is crisper there.
- **ACTIONABLE LEVER: H=1 value calibration on post-setup s1** (HL-Gauss compressed, can't express high confidence).
  Sharpen it (temperature/recal, value head that reaches ~1, post-push s1 data targeting 2nd-push openers) → lifts
  BOTH reactive AND search + justifies the greedy search. v2 postpush helped RANKING but not CALIBRATION. [n=150 refines.]

### 🔧 Q-VALUE AUDIT — n=150 REFINEMENT (confirms core, corrects two claims) [2026-06-14 ~12:40 ET]
| metric (Hz-v2 / NoHz-v2) | n=150 |
|---|---|
| H2 setup-vs-nonsetup AUC | **0.93 / 0.85** (Hz strong; CONFIRMED H2 good) |
| H1 calibration Q .4-.6 / .6-.8 → open | 4% / **29%** (Hz), 4% / 26% (NoHz) — CONFIRMED compressed [0.4,0.8], 2700+ samples |
| H1 top-1 a2 opens (on a setup) | **0.40 / 0.28** |
| H1 first-opener rank (med) | **3.0 / 4.5** |
| H2(a1) vs max-H1(s1) corr | 0.29 / 0.14 |
| greedy reactive 2push success | 0.14 / 0.09 |

**CONFIRMED (robust):** H=1 value COMPRESSED/miscalibrated (the bottleneck); H=2 setup-selection EXCELLENT (AUC 0.93).
**CORRECTION 1:** the H2→H1 handoff correlation is **0.29, not 0.59** (n=20 over-stated) — modest, not strong.
**CORRECTION 2 (important):** my greedy setup→opener handoff shows **Hz BETTER than NoHz** (top-1 opener 40% vs 28%,
AUC 0.93 vs 0.85) — which CONFLICTS with the full-eval reactive @2 (NoHz 33 > Hz 24). ⇒ **NoHz's reactive win is NOT
a better handoff** (Hz's is better); it's a SEARCH-PRIORITY / easy-scene effect (the blend priority + flat single head
surfaces a working pair within 2 pops more often on easy scenes per the division breakdown). My earlier "single head
sharper on the 2nd push" mechanism was WRONG. Pinning it down needs a per-decision best-first @2 trace [open].

### ✅ REACTIVE MECHANISM RESOLVED — cross-head scale mismatch [2026-06-14 ~13:50 ET, trace_reactive.py n=150]
Best-first @2 = P(sim2 DIVES into a child) × P(that child opens). Decomposed (matches full-eval @2 24/33):
| model | V0 (H2 firstpush) | V_s1 (H1 child) | V0−V_s1 | dive-rate | open|dive | @2 |
|---|---|---|---|---|---|---|
| Hz-v2 | 0.592 | 0.576 | **+0.016** | **41%** | **66%** | 0.27 |
| NoHz-v2 | 0.539 | 0.590 | **−0.051** | **78%** | 41% | 0.32 |

**NoHz wins reactive by DIVING 2× more** (78% vs 41%), even though Horizon's handoff is far better (child opens 66%
vs 41%). **ROOT CAUSE: cross-head SCALE MISMATCH.** Hz's H=2 head values first-pushes (V0=.592) ≥ its H=1 head
values the resulting children (V_s1=.576) ⇒ the blend priority keeps shopping fresh first-pushes instead of
committing to a good setup. NoHz uses ONE head ⇒ the post-setup state genuinely scores higher (V_s1 .590 > V0 .539)
⇒ it dives. **So Horizon's reactive deficit is NOT a value-quality issue — it's that the H1/H2 heads aren't on the
same scale, so a good setup's child doesn't outrank a fresh first-push.**
**ACTIONABLE (striking):** if Horizon dove at NoHz's rate (78%) with its OWN better handoff (66% open), its @2 would
be ~**0.51** — far above NoHz's 0.32. ⇒ normalize/calibrate H1 & H2 onto a common scale (or use V_s1 from the H1
head only for the dive decision) and Horizon should WIN reactive too. Combine this with the H=1 compression fix
(§Q-VALUE AUDIT) → lifts reactive AND search. This corrects the earlier (wrong) "single head sharper" mechanism.

### 🧪 FORCED-DIVE CEILING — fix #1 validated [2026-06-14 ~14:55 ET, trace_reactive_forced.json n=150]
Re-ran @2 forcing the search to ALWAYS dive (take the top setup's best opener at sim2):
| model | @2 actual | @2 FORCED-DIVE | Δ |
|---|---|---|---|
| Hz-v2 | 0.267 | **0.393** | **+0.126** |
| NoHz-v2 | 0.320 | 0.333 | +0.013 (already dives 78%) |
**Forced-dive Horizon (0.393) BEATS NoHorizon (0.33)** ⇒ fix #1 (cascade / bias-to-dive at low budget, NO retrain)
recovers + reverses the reactive gap. WHY clean: in the cases Hz chose NOT to dive, forced child still opens 21%,
a fresh first-push opens 0% (pure-2 can't solve in 1 more push) ⇒ at low budget DIVING STRICTLY DOMINATES.
⚠ SELF-CORRECTION: I earlier estimated the ceiling ~0.51 (0.78 dive × 0.66 open) — WRONG, that 0.66 is selection-
biased (Hz dives when the child looks good). Honest unconditional ceiling = 0.39. (3rd over-stated back-of-envelope
this session: cf. handoff 0.59→0.29, preview "90%". Pattern: small-sample/biased mental math — verify before citing.)
**FIX PLAN restated:** #1 cascade/depth-bonus search (free, +12.6pp reactive, test now) ; #2 bootstrap H2←γ·maxH1
(TD, ties heads to one scale = the principled horizon-Q recurrence we skipped) ; #3 de-compress H1 value head (HL-Gauss
sharpening + post-setup s1 data) for calibration. Root cause: H1/H2 trained as INDEPENDENT supervised heads — good
rankings, bad magnitudes (uncalibrated + not cross-budget comparable); the mixed-queue search lives on magnitudes.

### 🔬 STEP 0 — sigmoid double-squash CONFIRMED + finish-ranking genuinely weak [2026-06-14 ~17:20 ET, n=120 setups]
Scored H=1 finishing pushes on post-setup states, RAW E[bin] vs deployed sigmoid(E[bin]), GT openers vs non:
| | openers mean | non-openers mean | separation | range |
|---|---|---|---|---|
| RAW E[bin] | 0.37 | 0.097 | **0.273** | [.03,.88] openers |
| SIGMOID (deployed) | 0.589 | 0.524 | **0.065** | [.50,.71] all |
**(1) DOUBLE-SQUASH REAL:** live_scorer.score_ctx runs sigmoid on the HL-Gauss E[bin] (already in [0,1]) → crushes
[0,1]→[0.5,0.73], compressing opener↔non separation 4× (0.273→0.065). Monotone ⇒ rankings/M-series/2×2-ranking
UNAFFECTED; only MAGNITUDES (search V/blend, calibration) hit. FIX: score_ctx(raw=True) returns E[bin] for HL-Gauss
(wired, head-type-aware so sigmoid_bce ckpts unaffected). Expected: un-mushes the search's dive-vs-restart → fewer
restarts. (Does NOT change argmax picks ⇒ won't change which finish is chosen.)
**(2) FINISH RANKING GENUINELY WEAK (sigmoid-invariant):** even RAW, openers mean only 0.37 (vs ~0.9 ideal), p10=0.03
— many real openers scored as low as non-openers. This is the ~40%-top-1 weakness; needs RETRAIN (calibrate/sharpen
H1), independent of the squash. ⇒ TWO separate levers: drop-sigmoid (free, fixes search restart, won't raise ceiling)
+ retrain-finish (raises ceiling). NEXT: re-run best-first with raw=True, measure realized @2 + curve gain.

### 📊 TRAINING DATA SPLIT + SUCCESS/FAIL RATIO [2026-06-14 ~17:30 ET] — the data root of the mushy finish
Split: ScorerDataModule 90/10 room-grouped; v2 mix n=944,129 → train 849,717 / val 94,412. SUCCESS = push opens (≥20%):
| source | rows | tried/row | success% | fail:succ |
|---|---|---|---|---|
| m2b (1-push H=1, initial states) | 252,805 | 73.5 | 43.5% | 1.3:1 |
| h2 (2-push SETUP data, H=2) | 311,324 | 30.6 | 5.0% | 18.9:1 |
| aug (1push@H2 fix) | 80,000 | 34.4 | 100% | 0:1 |
| **postpush (the FINISH, H=1 on s1)** | 300k | 14.7 | **19.5%** | **4.1:1** |
**ROOT OF THE MUSHY FINISH (data side):** the H=1 head calibrates well on m2b (balanced 43.5%, initial states) but the
FINISH happens on post-setup s1 states covered ONLY by postpush — which is **4:1 failure-skewed + a minority** → head
hedges toward the ~20% base rate, can't push real openers high (matches Step0 raw openers mean 0.37). Setup data (h2)
is even more skewed (19:1) yet H2 selection is great (AUC 0.93) ⇒ imbalance alone isn't fatal; the finish is worse
because it's judged on OOD s1 states. **FIX (concrete): mint a large, balanced finish set from the exhaustive
(a1,a2)→opens map (exact, on-distribution s1 labels) + up-weight the opener class.** Hz-v2 RAW (de-squash) re-run
running (56206502) for the search-side gain in parallel.
