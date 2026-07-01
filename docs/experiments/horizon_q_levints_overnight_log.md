# Horizon-Q — LevinTS OVERNIGHT WORK LOG

> **2026-06-29 night. Branch `feat/horizon-q-levints`** (forked off `feat/horizon-q-redesign`).
> Durable record for the overnight autonomous push — survives context compaction; the chat does not.
> Read with: [horizon_q_levints_search_design.md](horizon_q_levints_search_design.md) (design) +
> [horizon_q_levints_implementation_plan.md](horizon_q_levints_implementation_plan.md) (plan) +
> [ILAB_RESUME.md](ILAB_RESUME.md) + [horizon_q_model_registry.md](horizon_q_model_registry.md).

## MISSION [USER, going AFK]
Work through the night, autonomously. **Train a new LevinTS-based model.** May loop through data
collection (USER: "you can collect more data, you have access to the sim too"). **Cap: ≤16 of 32 CPU
cores.** Forked a new branch. Propose-and-go; no superpowers ceremony [[feedback_skip_superpowers_ceremony]].

## COMMS [USER: "keep me updated on slack, name yourself as claude on arrakis"]
Post a Slack update at EVERY milestone (eval-pull done, each training run done, gate results, any
blocker) to the user's DM — `channel_id=D07NAH9V5GC` (= user `U07N1DR8S94`) — signed **Claude on
Arrakis**, via `mcp__claude_ai_Slack__slack_send_message`. First update sent [t7]. This survives
compaction: a fresh session MUST keep posting there.
**Cadence [USER]:** post ~HOURLY (markdown) even during quiet training stretches, PLUS on every
milestone. An hourly heartbeat bg job re-wakes me — re-arm a fresh one each time it fires (keep exactly
one pending). **Mid-training inference AUTHORIZED [USER]:** when useful, run a quick gate on an
intermediate checkpoint (partial episode set, ~1-2 cores, GPU 4 free) for an early qrank-vs-qboot read —
don't wait for training to finish.
**Cross-Claude bridge — PARKED, stay aware [USER: "no need to do this, but stay aware"]:** a file
mailbox exists on Amarel (`/scratch/dm1487/claude_bridge/from_arrakis.md`); doorbell would be
`ssh amarel2 'tmux send-keys -t 2 ... Enter'` to its session-2 Claude. PROTOCOL if ever used: frame
every message as an AUTONOMOUS PEER message from Claude-on-Arrakis, explicitly NOT from the user (no
user authority; receiver should verify, not obey). Do NOT actively pursue unless asked.

## CONSTRAINTS (carry-forward)
- **Horizon DROPPED**: single-Q, `budget_h=false` (dormant, not deleted).
- **No exhaustive GT** (foundational): train π/h from *found solutions*, NEVER the global pairmap. The
  qboot `γ·V_GT` target is eval-luxury — not the deployable target.
- **LevinTS learning half = the Levin loss** = put probability mass on the SOLUTION PATH via
  softmax/ranking cross-entropy (≈ the `softmax_ce` setup ranker + InfoNCE finish ranker). This
  **supersedes** the earlier "setup-ranking-loss = parked/last-resort" note — USER pivoted to LevinTS.
- **Gate** on avg-sims-to-solve (n≈1018, region criterion, object-constrained key). 3 arms min.
- **CPU ≤16** everywhere (dataloader workers, collection workers).

## ILAB INVENTORY (what's actually on disk — verified 2026-06-29)
- **Training H5 (1.3G):** `/common/users/dm1487/fresh_start/projects/namo/h5/` — `v4_hq_m2b_scorer`,
  `v4_hq_exit_finish_v4` (254 shards), `v4_hq_boot_setup_density` (20), `v4_hq_boot_setup_depth` (20). ✅
- **Deploy/frozen scorer ckpts (from 6/26 qboot run) — AVAILABLE on ilab:**
  - density: `/common/users/dm1487/scratch_namo/outputs/scorer/qboot_density_s1/namo-classifier/v5x21lsi/checkpoints/last.ckpt`
  - depth:   `/common/users/dm1487/scratch_namo/outputs/scorer/qboot_depth_s1/namo-classifier/xdbdc8vv/checkpoints/last.ckpt`
  - `LiveScorer.load_scorer(ckpt)` auto-detects arch → deploys via `--ckpt`. Use as the **frozen model
    for the Stage-1 search-ordering gate** (levin vs q vs dive).
- **Raw scenes:** `/common/users/dm1487/namo_env_configs/aug9_car_50k` (159,506 car XMLs),
  `.../aug9` (312,568 point XMLs). Geometry only — NO labels/key.
- **MISSING on ilab:** all `datasets/`, `manifests/`, labels JSON (eval key + train relabel key). →
  `NAMO_SCRATCH=/common/users/dm1487/scratch_namo` has only `outputs/`.

## BLOCKERS
- **B1 — eval/gate data on Amarel (IN PROGRESS).** Need `test_pure2_fromkey.txt` + `pure2push.json` +
  test scene XMLs (+ NoHz-v3 baseline ckpt). Delivery: ilab PULLS (`pull_from_amarel.sh eval`, ~2.7G).
  ilab→Amarel SSH had no key auth → FIXED ilab side: wrote `~/.ssh/config` (Host amarel →
  `id_ed25519_amarel`). Remaining: USER relays a message to **Amarel Claude** to append the ilab pubkey
  (`ssh-ed25519 AAAA…amarel-rsync`) to Amarel `~/.ssh/authorized_keys`. A background job polls SSH and
  auto-pulls on success (see log). **Gate is blocked until this lands.**
- **B2 — levin-search collection NO-GO tonight (code gap + pull).** (1) Nothing consumes the
  `solution`/`sol_logpi` jsonl (`eval_bestfirst.py` writes it; no converter exists) → jsonl→H5 is
  NEW non-trivial code (re-render+re-label to `ScorerH5Dataset` format). (2) `exit_collect.py` writes
  trainer H5 directly but needs the TRAIN labels key (`labels_exhaustive_pure2push.json`, Amarel `train`
  pull) and is the OLD finish-head collector (not the levin product). Base model is NOT a blocker.

## PLAN (revised after Agent-B)
- **P1 — TRAIN the LevinTS-objective ranker (PRIMARY, no Amarel dep).** Single-Q NoHorizon, ranking
  loss (softmax_ce setup + InfoNCE finish) on the local H5. Multi-seed. *Exact command pending Agent-A
  (training-stack investigator, running).* Starts as soon as Agent-A returns.
- **P2 — Stage-1 search-ordering GATE.** Once B1's eval data lands: A/B `combine=levin` (τ sweep) vs
  `q` vs `q+dive_bonus`, frozen model = a qboot ckpt above. Then gate the P1 model too.
- **P3 — collection (deferred).** Levin round needs the jsonl→H5 converter (write it) + a train key.
  Fallback "Path B" = `pull_from_amarel.sh train` then `exit_collect.py` sharded ≤16 — but that's the
  OLD finish collector, not the levin product. Only pursue if P1/P2 are healthy and time remains.

## DECISIONS
- **D-ON-1:** PRIMARY = ranking-loss model on EXISTING data (no collection dependency) = lowest-risk
  "new LevinTS model". Collection is stretch.
- **D-ON-2:** Gate the Stage-1 ordering with a **qboot ckpt** (already on ilab); do NOT block the gate
  on the NoHz-v3 Amarel ckpt (it's only a baseline number).
- **D-ON-3:** Do NOT run the deprecated `rewrite_paths.sh` after the pull — paths remap at load via
  `namo.paths.resolve()` (CLAUDE.md). Verify key/xml resolution at gate time instead.

## EXECUTION LOG (append-only)
- **[t0]** Stage-1 built + unit-tested (10/10) + committed `23c5199` on `feat/horizon-q-levints`
  (force-added past `scripts/sandbox/.gitignore`). `levin_cost.py` + `combine=levin`/`--tau` +
  solution-path logging in `eval_bestfirst.py`.
- **[t1]** Probed ilab: bindings import OK; eval data MISSING; no NoHz ckpt local. Confirmed training
  H5 present (4 mixes). Wrote `~/.ssh/config` for Amarel. Generated/found `id_ed25519_amarel` pubkey.
- **[t2]** Dispatched Agent-A (training command) + Agent-B (collection feasibility). Gave USER the
  Amarel-Claude key-authorization message.
- **[t3]** Agent-B DONE → collection NO-GO (B2); found qboot ckpts on ilab (deploy model available);
  found raw aug9 scenes. Wrote this journal.
- **[t4]** Agent-A delivered the training recipe: **`softmax_ce` IS the ready Levin ranking loss**
  (policy CE over the masked reachable set = mass on solution cells; enable `+model.head_mode=softmax_ce`,
  DROP `value_bins`). InfoNCE/D2 is NOT in code (minimal diff is in Agent-A's report, for a later step).
  Launched 2 arms (qrank_density GPU2 / qrank_depth GPU3, softmax_ce, sample_k=30, 8 workers).
  **VALIDATED:** clean start — 4.3M-param EdgeCrossAttn, 305k rows (train 274k / val 30k, room-grouped),
  `train_loss≈3.37≈log(30)` (correct softmax-ranker baseline), ~2145 steps/epoch (~12-16 min/epoch).
- **[t5]** USER returning + moving me to a persistent **tmux** session with Amarel info. **STOPPED both
  arms cleanly (SIGTERM)** — epoch 0, no real progress lost; recipe proven. GPUs 2/3/4 free, machine
  clean. Handoff below.
- **[t6]** GREEN LIGHT from Amarel Claude: ilab key authorized (`ssh amarel` → SSH_OK / amarel2), all 4
  eval sources verified (testset 2.0G, car_envs/v3/test 130M/18,270 files, manifest 110K, NoHz-v3 ckpt
  218M). **EXECUTING the handoff:** launched (bg) — eval pull `pull_from_amarel.sh eval` (task bslx7h0kn),
  `qrank_density_s1` GPU2 (b6fdswf0m), `qrank_depth_s1` GPU3 (b01etu5fj). Verified both arms progressing
  (density step 8, train_loss 3.37→3.33 falling; depth starting). GPUs 2/3 active, 4 free, CPU≈16.
  **NEXT (auto on completion):** eval-pull done → SMOKE-gate (few rooms) with a qboot ckpt to validate the
  eval pipeline; training done → FULL gate (qrank softmax_ce vs qboot hl_gauss, density+depth) on avg-sims.
  Gate sims are CPU-heavy → run AFTER training (or low parallelism) to respect the 16-core cap.
- **[t7]** Slack comms live (DM `D07NAH9V5GC` as "Claude on Arrakis"). **Eval pull DONE** — verified key
  (1.6M), 983-room manifest, test scenes, NoHz-v3 baseline ckpt
  (`qfull_nohz_v3_v4hq_s1/.../epoch012-val_loss0.6896.ckpt`). **Smoke gate PASSED**: `combine=levin` on
  NoHz-v3, rooms 0-5 (all benchmark_5/hard) — pipeline works end-to-end (Amarel→ilab path remap at load,
  real sims, `solution`/`sol_logpi` logging). **20% solve / 26 sims = PLUMBING CHECK ONLY** (n=5, hard),
  NOT a verdict. **Training:** both arms epoch 0 ~86%, **~27 min/epoch**, loss 3.37→~3.1, first ckpt at
  epoch-0 end; ~6 h projected to converge.
  **NEXT:** first MEANINGFUL mid-training gate ~epoch 5 (model converged enough); FULL gate at completion.
  Gate arms: `--combine {q, q+dive_bonus, levin --tau {0.5,1,2}}`, baseline ckpt = NoHz-v3 epoch012;
  then qrank density/depth best-val ckpts vs qboot (already on ilab).
- **[t8]** T+1h heartbeat: both arms **Epoch 2** (~24 min/epoch); ckpts saving (epoch001 + last.ckpt).
  Early val — qrank_density top1 **0.234** / top5 **0.420** (val_loss 3.91); qrank_depth top1 **0.250** /
  top5 **0.456** (val_loss 3.96). (val_loss>train is the sample_k normalization difference — val over full
  reachable, train over k=30 — NOT overfitting; top1/top5 are well above chance.) Launched a VALIDATION
  gate (qrank_density epoch-1, combine=levin, rooms 0-10, GPU4) to confirm a `softmax_ce` ckpt loads+scores
  in eval_bestfirst BEFORE the real epoch-5 gate. Re-armed hourly heartbeat. ckpt dirs:
  `outputs/scorer/qrank_density_s1/.../z7ax3oj1/`, `qrank_depth_s1/.../x5cujg88/`.
- **[t9]** Validation gate DONE (qrank_density epoch-1, combine=levin, 10 hard rooms): RC=0 — **softmax_ce
  ckpt loads + scores correctly in eval_bestfirst**, solution-path logging works (solved ex:
  `solution=[{edge12,d4},{edge16,d4}], sol_logpi=-5.745` = log path-π; d/π≈625 → levin_cost machinery
  confirmed end-to-end; the future Levin-loss step WILL have its training data). 10%/27 sims = plumbing only
  (epoch-1 model, hard rooms). **BOTH gate paths de-risked** (NoHz-v3 baseline + qrank softmax_ce). Idle
  until ~epoch 5 → real mid-training gate (heartbeat `b91n2ms2w` ~1h). TODO at gate: check manifest tier
  layout (rooms 0-10 were all benchmark_5/hard) so the gate sample is tier-representative, not all-hard.
- **[t10]** T+2h heartbeat: **Epoch 4** (~24 min/ep), train_loss 3.37→2.94. **⚠️ val_top1/top5 DROPPED**
  ep1→ep3: density 0.234/0.420 → **0.091/0.155**; depth 0.250/0.456 → 0.122/0.183 — while train+val LOSS
  both improved (val_loss 3.91→3.78). Hypotheses: (a) metric artifact — softmax_ce matching the smooth
  γ·V_GT setup-target shifts the argmax; (b) mixed sparse-opener + dense-γ·V_GT targets muddy the top-1
  ranker. Val metric ≠ objective → gate it. **Manifest tiers INTERLEAVED** (aug9 bench_1-5 + feb_car
  straightX) ⇒ contiguous slice is representative. Launched mid-training gate (rooms 0-100, GPU4, bg
  bacv0268w/btr2xp3wj): qrank_density ep3 combine=q vs qboot_density combine=q (clean loss A/B; qboot ckpt
  `qboot_density_s1/.../v5x21lsi/last.ckpt`). Re-armed heartbeat. **Watch:** if qrank ≫ worse sims than
  qboot, softmax_ce-on-mixed-targets is suspect → consider sample_k=0 / sparse-only ranker next.
- **[t11]** MID-TRAINING GATE (rooms 0-100, combine=q, budget 30): **qboot (hl_gauss, converged last.ckpt)
  69.9% / 5.68 sims** vs **qrank_density (softmax_ce, EPOCH 3) 54.4% / 9.09 sims** — qrank BEHIND.
  Caveats: (1) ep3 vs converged = unfair; (2) real worry = val_top1 DROPPING with training (0.234->0.091)
  => may not catch up. (Absolute solve% > registry ~38% because budget=30 vs tight @2; A/B internally
  valid — same budget+combine.) **HYPOTHESIS:** pure softmax_ce REPLACES BCE, loses per-cell calibration;
  better LevinTS loss = **BCE + lambda*InfoNCE rank** (KEEPS value, ADDS ranking) — Agent-A diff for
  `classifier_module._compute_masked_loss` sigmoid_bce branch ([t4]); enable via
  `+model.rank_weight=.. +model.rank_tau=..`. **PLAN:** re-gate qrank ~epoch 8 (next heartbeat); if still
  behind + val_top1 still falling -> implement+launch `qrank_infonce` variant, freeing compute by killing
  the qrank_depth arm (both current arms share the softmax_ce flaw). NOT thrashing on ep3 — confirm first.
- **[t12]** T+3h heartbeat (Epoch 6): **val_top1 RECOVERED** — density 0.234(ep1)→0.091(ep3)→**0.313(ep5)**
  (now ABOVE ep1, still climbing); depth →0.217. The ep3 dip was **TRANSIENT** (LR warmup / noisy val),
  NOT degradation ⇒ **earlier "softmax_ce degrading" alarm RETRACTED; InfoNCE variant NOT needed now**
  (good call not to thrash on ep3 data). val_loss 3.91→3.73, train 2.86 — still improving, ~7 epochs left.
  GPUs **0,1 FREED** (other user done) → 0,1,4 free. Re-gating qrank_density ep5 (last.ckpt) combine=q
  (bp9uls71j) + combine=levin (bcomcgfrr), rooms 0-100, vs qboot 69.9%/5.68. Re-armed heartbeat.
- **[t13]** T+4h heartbeat (Epoch 8): val_top1 climbing — density 0.313(ep5)→**0.374**; depth
  0.217→**0.473** (depth ranks SHARPER than density). qrank ep5 re-gate **combine=q = 55.3% / 7.68 sims**
  (vs ep3 54.4%/9.09: sims improving, solve ~flat; still behind qboot 69.9%/5.68). **combine=levin arm
  SLOW** (~1h+, still running) — likely budget-exhaustion (weak/not-converged ranker → 30 sims/episode)
  and/or GPU contention; watching, will report or kill. ~5 epochs left. Re-armed heartbeat.
  READ: softmax_ce learning but trailing qboot value head; if it plateaus < qboot at convergence, the
  **BCE+InfoNCE** variant (additive to the working value head ⇒ should be ≥ qboot) is the motivated next
  run — launch as the density run frees GPU2 (or kill qrank_depth to start sooner).
- **[t14]** ⭐ **KEY FINDING — combine=levin (τ=1) is CATASTROPHIC vs combine=q** (qrank ep5, rooms 0-100):
  **levin 5.8% / 23.3 sims vs q 55.3% / 7.68**. Root cause: `d/π` penalizes DEPTH — a 2nd-push node (d=2)
  costs `2/(π_setup·π_finish)` ≫ root (d=1) `1/π_setup` when π is diffuse ⇒ search expands ALL ~30-50
  first-pushes before EVER diving to a 2nd push ⇒ can't solve 2-push problems within budget 30. **LevinTS
  d/π REQUIRES a confident/sharp tree policy — its value is the Levin-LOSS peaking π on solution paths, NOT
  the ordering bolted onto an arbitrary model.** ⇒ Stage-1 "swap ordering on frozen model" FAILS; deploy
  ordering for these models = **combine=q**. The "LevinTS model" deliverable = the **softmax_ce-trained
  ranker deployed with combine=q**. Testing fix: LOW τ (sharp π → π_path→1 → child cost→2 beats diffuse
  roots → dives) at τ=0.5/0.2 (rooms 0-50, bnwjn1rm1). (Refines the pre-registered note: within-node
  levin≈q; cross-depth ANTI-dives — worse than the predicted "≈".)
- **[t15]** PER-TIER tool `scripts/sandbox/tier_breakdown.py` (joins leaf jsonl × `pure2push_divisions.json`
  `division` by (xml,object_id,region); per-episode; 100% match). Full set = hard 371/med 409/easy 238 =1018.
  Rooms 0-100, combine=q: **qrank ep5** easy/med/hard = 57.1/58.8/48.4 (ALL 55.3); **qboot** = 81.0/74.5/54.8
  (ALL 69.9). qrank trails on ALL tiers, WORST on EASY (57 vs 81 — should near-saturate ⇒ under-fit at ep5).
- **[t16]** T+5h (Epoch 10): val_top1 density **0.437** / depth **0.600** (depth ranks MUCH sharper ⇒ likely
  the real qrank contender). **Low-τ levin CONFIRMS dive mechanism**: τ=0.2 = **32.0%** (rooms0-50) vs τ=1
  5.8% — sharp π restores diving; still < q 55% ⇒ levin trails q even sharpened → **deploy combine=q**.
  TIMING CORRECTION: ~24 min/epoch (8 workers) ⇒ best-val ckpt ~epoch 13-15 (~1.5h); full early-stop
  ~epoch 38 (≫later, unnecessary). PLAN: once val plateaus, KILL training → free 16 cores → full 1018
  per-tier gate (parallel slices) for converged qrank density+depth vs qboot, combine=q.
- **[t17]** Levin τ-sweep COMPLETE (qrank ep5): τ=1 **5.8%** → τ=0.5 **12.0%** → τ=0.2 **32.0%** (MONOTONE:
  sharper π ⇒ more diving ⇒ more solves — mechanism confirmed) but ALL < combine=q **55.3%**. ⇒ **d/π search
  ordering = CONFIRMED DEAD-END for H=2 with these models.** The only LevinTS lever here is the TRAINING
  half (qrank softmax_ce ranker), deployed with combine=q. **Levin-ordering investigation CLOSED.** (NOTE:
  Stage-1 design spec gate = REJECTED on numbers — `combine=levin` ≪ `combine=q`; update the design spec §6
  verdict in the morning.)
- **[t18]** Training STOPPED (TaskStop; val_loss plateaued ~3.69-3.71 since ep8, best-val saved). Best-val
  ckpts: qrank_density **ep10** (vl3.6932), qrank_depth **ep9** (vl3.7055), qboot_density **ep12** (vl0.7152),
  qboot_depth **ep14** (vl0.7192). (qrank vl~3.7 vs qboot vl~0.72 NOT comparable — softmax_ce~log(30) vs
  hl_gauss value scale.) Launched **DEFINITIVE full-1018 per-tier gate** `scripts/sandbox/full_tier_gate.sh`
  (biojdyigx): 4 models × 4 manifest slices = 16 parallel eval_bestfirst, combine=q, GPUs 1-4, ~45 min;
  merges per model → `tier_breakdown.py`. Verified 16 procs up, no errors. Re-armed heartbeat (hang-fallback).
- **[t19]** ⭐⭐ **DEFINITIVE FULL-1018 PER-TIER GATE** (combine=q, best-val, 100% matched). solve% easy/med/hard/ALL:
  - **qboot_depth (value)   79.4 / 69.2 / 48.0 / 63.9**  ← BEST
  - qboot_density (value)  76.5 / 66.3 / 47.2 / 61.7
  - qrank_depth (rank)     71.0 / 62.3 / 45.6 / 58.3
  - qrank_density (rank)   71.0 / 62.8 / 43.7 / 57.8
  **VERDICT: qboot (hl_gauss VALUE) BEATS qrank (softmax_ce RANK) on EVERY tier, both summaries, by ~4-6pp;
  qrank also higher sims_to_solve. depth>density for both.** The softmax_ce ranking loss UNDERPERFORMS the
  value head. **Root-cause read:** softmax_ce normalizes per-state ⇒ loses CROSS-STATE calibration that
  combine=q (orders nodes across states/depths by raw score) needs; the value head keeps it. Combined with
  [t17] (d/π dead-end) ⇒ **naive LevinTS (rank loss + d/π ordering) does NOT beat the value baseline.**
  NEXT (code change + design choice, WITH USER): **hl_gauss VALUE + λ·rank AUXILIARY** (keep calibration, ADD
  ranking — NOT replace). Did NOT launch unattended (thrash/design risk). Overnight **train+gate COMPLETE**.
  Leaf jsonls: `$NAMO_SCRATCH/eval/full_tier/{model}_full.jsonl`. TODO morning: registry + design-spec §6
  verdict (levin REJECTED).

## HANDOFF — tmux session (do these in order; CPU ≤16, pick FREE GPUs via `nvidia-smi`)

**0.** You're on `feat/horizon-q-levints`. `cd repo && source env.ilab.sh`. Stage-1 search code is
committed (`23c5199`). Training recipe is VALIDATED (ran clean, then stopped for this handoff).

**1. RELAUNCH the 2 training arms** (the new LevinTS-ranking model; NO Amarel dep). For each arm swap
`<GPU>` (a free one) and `density`→`depth` + `RUN=qrank_depth_s1`:
```bash
export WANDB_MODE=offline WANDB_DIR=$NAMO_SCRATCH/wandb
H5=/common/users/dm1487/fresh_start/projects/namo/h5
M2B=$H5/v4_hq_m2b_scorer/data.h5
EXIT=$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5|sort -V|paste -sd';' -)
BOOT=$(ls $H5/v4_hq_boot_setup_density/shard_*.h5|sort -V|paste -sd';' -)   # depth arm: v4_hq_boot_setup_depth
RUN=qrank_density_s1; OUT=$NAMO_OUTPUTS/scorer/$RUN; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/src/train_classifier.py" \
  --config-name=train_scorer_edge name="$RUN" data_dir="'$M2B;$EXIT;$BOOT'" output_dir="$OUT" \
  max_epochs=200 num_workers=8 +seed=1 +data.sample_seed=1 +data.sample_k=30 \
  +model.bce_reachable_only=true +network.pos_fourier=true +network.use_edge_embed=true \
  +data.budget_h=false +model.head_mode=softmax_ce > "$NAMO_LOGS/$RUN.out" 2>&1
```
In tmux this can run in a window directly (persistent). Sanity-check `$NAMO_LOGS/$RUN.out`: expect
`train_loss` falling from ~3.37. ~6-10h to early-stop. Add to registry when best ckpt lands.

**2. PULL the eval/gate data** (THIS is the only Amarel-dependent step): once Amarel Claude authorized
the ilab key — `ssh amarel true` should now succeed — run `bash scripts/portability/pull_from_amarel.sh
eval`. **Do NOT** run `rewrite_paths.sh` (deprecated; paths remap at load via `namo.paths.resolve()`).
Verify: `ls $NAMO_SCRATCH/datasets/namo_testset_v1/labels/pure2push.json`.

**3. STAGE-1 GATE** (search ordering; frozen model = a qboot ckpt already on ilab — see ILAB INVENTORY).
Once eval data is present, run `scripts/sandbox/eval_bestfirst.py` arms: `--combine q`, `--combine q
--dive-bonus 1.0`, `--combine levin --tau {0.5,1,2}` (--hmax 2 --sim-budget 30, default key+manifest).
Gate = avg-sims-to-solve. NOTE: the gate sims are CPU-heavy — don't run concurrently with 16-core
training; either wait for training or drop training workers.

**4. GATE the qrank models** (P1) the same way once trained; compare qrank (softmax_ce) vs qboot
(hl_gauss) on avg-sims, both density+depth. Record every number here + in the registry.

**Honest caveat to carry:** these qrank models still train their SETUP ranking on the boot_setup
`γ·V_GT` targets (eval-luxury, pairmap-derived). That is fine for a first LevinTS-loss model but is NOT
the de-oracled target — the scale-legal version trains setup ranking on FOUND-solution targets (needs
the jsonl→H5 converter, a code gap; see B2). Don't ship qrank as the deployable de-oracled model.
