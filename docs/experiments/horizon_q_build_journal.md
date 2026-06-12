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
