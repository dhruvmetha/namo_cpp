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
