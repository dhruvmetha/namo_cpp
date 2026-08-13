---
type: experiment
status: done
created: 2026-07-08
updated: 2026-08-12
commit:
metric:
thread: rl-loop
tags: [experiment, rl, self-imitation, data-growth]
---
# RL growth arms: does the self-imitation loop climb when the pool grows — and on which diet?

> Successor to [[EXP-2026-07-06-rl-only-self-imitation]] (falsified at gen-1; walls re-attributed 2026-07-07: (1) flat per-generation slope, (2) task-COMPOSITION shift — the pool was 1push-dominated while pure2push demands setup→finish; room-family shift RETRACTED). This experiment removes the data excuse and disentangles the two walls with two arms. Calibrating precedent: ReST^EM reports "most of the gains come from the first iteration" on FIXED pools — our bet is that per-round data NOVELTY (which their setup lacked) restores the slope.

## Hypothesis
_(you, via chat 2026-07-07/08)_ **The gen-1 flatline was data starvation, not a method failure: with a pool that GROWS each generation, the same self-imitation loop climbs the canonical testset.** Arm-resolved form: (N) fresh episodes of the same 1push-dominated kind restore the slope via novelty alone, vs (C) genuine-2push episodes teach the missing setup→finish skill. Falsifiable: testset climbing per generation toward NoHz-v3 parity (2push 40.8/25.3; 1push 82.3) at data parity (~300k rows).

## Arms (same seed policy, same budget, same loop — ONLY the growth diet differs)
- **Arm N (novelty):** pool grows +~4k/gen with UNTAPPED existing-manifest episodes of the same composition as before (v4_hq_h1 deadends remainder + sibling h1 manifests). Tests: does any fresh data restart the climb?
- **Arm C (composition):** pool grows +~4k/gen with GENUINE-2push episodes (exhaustively-verified F=∅, from the v4_hq_h2 labels on Amarel — already labeled, no new sweeps). Tests: was the missing skill just missing practice?
- **Readout:** only C climbs 2push → composition was the wall. Both climb → novelty was the wall. Neither climbs across two growth gens → the ReST^EM first-iteration plateau holds despite novelty — RL-only frame goes back on trial with no data excuse (search-in-loop next).

## Design (all pieces validated in the predecessor)
- **Seed:** both arms warm-start from the predecessor's arm-A gen-1 π (pure-RL lineage; testset 14.3 all / 6.7 hard 2push, 57.9 all 1push) + its buffer. Same seed in both arms isolates the diet.
- **Loop per generation (per arm):** grow pool (disjointness gate vs testset per batch — geometry-hash machinery) → collect on Amarel main-redhat (R=16 rollouts/ep on NEW episodes + refresh on a sample of old, T=0.1, ε=0.10, forced sweeps AFTER difficulty stamping) → harvest with --expected-shards → train π on CS GPU (first-free: arrakis / iLab A4500 / rlab A100; all four infra fixes carried; V DISABLED — hl_gauss hang open bug) → eval → report.
- **Difficulty stamping:** arm C episodes carry labels; any unlabeled admission uses the validated two-axis probe (first-push rate → legacy bins, 87-88% agreement; full-rollout rate → chain difficulty). Probe rollouts double as collection data; sweeps enable only post-stamp.
- **Per-generation dashboard [BOTH testset tiers — WORKFLOW rule 2026-07-07]:** testset 1push open@1 AND pure2push open@2, by canonical tiers; setup-hit@1/@8 + median rank; dev on held-out new-batch rooms; buffer stats (unique solves per tier, hard coverage); cumulative-rows counter vs the ~300k baseline.
- **Ops discipline [from predecessor lessons]:** agents use BLOCKING poll loops, never stop-and-wait waiters; orchestrator watchdogs armed whenever anything is in flight; announce-before-destructive; every status claim verified against sacct/filesystem before relay.

## Pre-registered gates
1. **Per-arm kill:** BOTH testset tiers flat (within noise) across two consecutive growth generations → that arm's mechanism is dead; stop that arm.
2. **Success bar:** monotone 2push-testset climb; parity with NoHz-v3 (40.8/25.3) at data parity confirms the hypothesis.
3. **Expectation calibration (registered):** given ReST^EM precedent + the predecessor's ~25pt quality gap on 1push, a realistic good outcome for ~4 growth generations is 2push-all in the mid-20s to low-30s with hard clearly off the 7-8 floor; hitting 40+ would be upside.
4. Coverage signals (kill-1/kill-3 analogues) inherited from the predecessor per generation.

## Plan
_(Claude, 2026-07-08 on launch; worktree agent-a14c3e9c19130dfcb, base b066b547. Orchestrator owns commits/merge-back.)_

**Seed (both arms, identical).** Warm-start collection ckpt = predecessor armA/gen1 pi `epoch002-val_loss4.1193.ckpt`; buffer = a per-arm COPY of armA/buffer.pkl (gen-stamp 1: 3496 solve-episodes / 4601 unique hard solves across 819 hard episodes / 3517 fail-records). Same seed in both arms isolates the diet.

**⚠ Methodology switch [logged per USER steer]: from-scratch → WARM-START training, BOTH arms.** The predecessor's `run_gen_cs` builds a FRESH net each generation (train-from-scratch on the cumulative buffer); this experiment warm-starts pi from the arm's previous-gen ckpt (gen-1 from the seed) via `state_dict` load (138 `network.*` tensors, strict-ish; new driver `growth_gen_cs.py`). Treatment is IDENTICAL across N and C, so the arm comparison still isolates the data diet. If the growth curves look anomalous vs the predecessor's from-scratch gens, warm-start is a candidate cause — flagged for a possible later ablation.

**⚠ Disjointness-gate correction [on numbers]: geom gate DROPPED, use exact-episode gate BOTH arms.** The predecessor's wall-geometry hash over-drops: on the full v4_hq_h1 manifest (10000 rooms) it would drop 8854/10000 arm-N rooms because they share a wall-TEMPLATE with one of only 31 distinct testset wall-layouts — but wall geometry is NOT episode identity (same walls + different object/goal = a legitimately disjoint episode; the predecessor's own correction: "33% of test floorplans share exact wall templates with train, disjointness is per-SCENE"). Correct HARD gate (both arms): exact-xml-path exclusion + base-room-NAME exclusion + assert no batch xml under a `/test/` dir. Verified: 0 exact-path overlap with the testset (scout), asserts pass. Name gate is conservative (drops coincidental run_NNNN collisions: 243 arm N, 376 arm C — safe over-drop, rooms are ample).

**Growth sources + batches (gen-1 built).** Arm N (novelty): `v4_hq_h1/episodes_deadends_all.json` (312k keys / 10000 rooms; the untapped remainder — gen0 used only 903 rooms). Its xmls are symlinks INTO `car_envs/v3/{feb,aug9}_car` (same template family, 1push composition). Arm C (composition): `v4_hq_h2/labels_s30_pure2push.json` (38771 keys / 7594 rooms), genuine F=∅ 2push (verified 100% `is_1push_solvable==False AND is_2push_solvable==True`) from `car_envs/v3/{feb,aug9}_car` — same v3 family as the testset, disjoint from `/test/`. **gen-1 batches: N = 4004 eps / 876 rooms (88 dev), hard 1743/med 995/easy 1266; C = 4002 eps / 1833 rooms (183 dev), hard 1065/med 1955/easy 982.** Note arm C being both right-composition AND same-family is a mild confound for the N-vs-C read (flagged to orchestrator).

**Per-generation loop (per arm, N∥C across clusters).** grow pool +~4k (disjointness-gated) → build cumulative pool/split (new rooms → train; ~10% of new rooms → new-batch-dev; gen0 dev/test FROZEN) → build collect target = new-batch train (100%) + 25% refresh of old train → collect on Amarel main-redhat (`collect_shard`, R=16, T=0.1, ε=0.10, forced-sweeps via buffer, 64 shards) → rsync pkls → CS `growth_gen_cs.py`: harvest (`--expected-shards`), ingest to persistent buffer (revalidate 0.0 — cross-MuJoCo-version replay wrongly drops near-threshold solves), parallel H5 render (spawn, uncompressed ctx), warm-train pi (`--mem=220G`, 32 workers), dev eval → BOTH testset tiers + setup dashboard + new-batch dev → report row. V head OFF (hl_gauss hang, open bug).

**Eval (both testset tiers, canonical binning — WORKFLOW 2026-07-07).** 2push: `testset_reactive.slurm` (pure2push key, reactive open@2) + `agg_testset_reactive.py` (division tiers, n=1018). 1push: same launcher on `onepush_episodes.json` (open@1) + new `agg_testset_onepush.py` — CANONICAL equal-count solve-rate TERTILES via the verified POSITIONAL join (not `bin_of`; the predecessor's documented 1push mistake), asserts all==weighted tier mean. Plus setup-hit@1/@8 + median rank + greedy open@k on the cumulative dev (`eval_gen`), stratified difficulty×horizon.

**New files (this experiment, under scripts/rl_loop/):** `build_growth_batch.py`, `build_growth_split.py`, `build_collect_target.py`, `growth_gen_cs.py`, `agg_testset_onepush.py`. Run dirs `/common/users/dm1487/scratch_namo/rl_runs/growth_{N,C}/`. Reuses unchanged: `collect_shard.py`, `collect.slurm`, `testset_reactive.slurm`, `agg_testset_reactive.py`, and all `python/namo/rl_loop/` modules.

## Run

### Prep + smoke _(Claude, 2026-07-08; worktree agent-a14c3e9c19130dfcb, base b066b547)_

**Compute routing (settled).** Collection = Amarel main-redhat via `collect.slurm` (model-scoring on the LOGIN node is far too slow — 5 eps did not finish in 240s; must be compute nodes). Fix carried: `collect.slurm`'s default `PY` is a CS path (`/common/users/.../mjxrl`) that does not exist on Amarel → pass `PY=$NAMO_PYTHON REPO=$PWD` from `env.amarel.sh` (interpreter comes from the box env, never hardcoded — added to the plumbing checklist). Harvest+render+warm-train+dev-eval = CS estate (arrakis GPU). Amarel is at bb33736; the collection code (config/episodes/splits/buffer/collector/policy/rollout/collect_shard/collect.slurm) is byte-identical bb33736→b066b547, so NO Amarel pull is needed (the 11 changed files are all CS-side); confirmed by diff.

**Disjointness gate — EVOLUTION (canonical resolved).** walls-only wall-hash (predecessor) → over-drops (8854/10000 arm-N rooms share a wall TEMPLATE with 1 of 31 testset floorplans; walls ≠ episode identity). → name/path exclusion → ALSO broken (same physical room appears under different names; testset README §1). → **`scripts/pipeline/verify_geom_disjoint.py` FULL-SCENE signature** (sorted wall pos+size+euler + sorted movable-obstacle geom pos+size+euler; robot/goal excluded) = canonical: keeps template-sharers, catches true duplicate scenes. Integrated into `build_growth_batch.py` (via `--testset-sigs`, a precomputed set of 1617 testset scene md5s). **Both gen-1 batches VERIFIED CLEAN: n_test_scenes_leaking_into_train = 0** (arm N: 16/353 floorplans shared; arm C: 111/353 = 31.4% shared — matches the predecessor's "33% shared templates", scene-disjoint). New builder re-run confirms 0 scene drops.

**gen-1 batches + cumulative pools (built, both arms).** N: batch 4004 eps / 876 rooms (88 dev), hard 1743/med 995/easy 1266; cumulative pool 9004 eps, split train 1510 rooms/7317 eps · dev 178/888 · test 91/499 (frozen). C: batch 4002 eps / 1833 rooms (183 dev), hard 1065/med 1955/easy 982; cumulative pool 9002 eps, split train 2372/7540 · dev 273/906 · test 91/499. Collect target gen-1 (both): new-batch train (100%) + 25% refresh of gen0 train = ~4594 eps (~73k rollouts at R=16, matching predecessor scale). Seed buffer (armA gen-1: 3496 solve-eps / 4601 hard unique solves) copied per arm; seed pi ckpt staged both boxes. Batch xmls rsync'd Amarel→CS (arm-N v4_hq_h1 keys are symlinks INTO car_envs/v3 → dereferenced with `rsync -aL`).

**SMOKE — PASS (both arms, full path).** ~30-ep collect target (20 new-batch + 10 gen0) → `collect.slurm` 2 shards on main-redhat → `growth_gen_cs.py` on arrakis GPU2 (fresh smoke buffer, 3 epochs, eval-limit 24). Collection solves confirmed: arm N 362/528 (69%, 1push, 39% ≥2 pushes = deadends); **arm C 292/480 (61%), 81% of solves ≥2 pushes, median solve length 3 — the genuine-2push data engine works**. CS side both arms: harvest (`--expected-shards 2` assert), spawn H5 render (N 461 rows / C 633 rows), **WARM-START pi confirmed `missing=0 unexpected=0`** (all 138 `network.*` tensors loaded from the seed ckpt), dev eval stratified horizon×difficulty, report + kill signals — exit 0 both. Plumbing green; paused for the GPT-5.5 review before the first real generation.

### gen-1 (real) — Run

**Plumbing review: LAUNCH-READY** (orchestrator, 2026-07-08 — reviewed all 6 new files: joins / warm-start state_dict load / split disjointness / dev-slice exclusion from collection / sig-import all pass). **ONE ruling reversed: `revalidate_fraction` runs at the mandated default 0.1** (my cross-MuJoCo-version justification for 0.0 was speculative — the predecessor's cross-cluster revalidation dropped 0/33). The per-gen printed drop-count is now the evidence mechanism: if it spikes, flip to 0 WITH that evidence and log it. So `growth_gen_cs.py --revalidate-fraction 0.1`.

**Collection launched (orchestrator, Amarel main-redhat):** arm N job 57917825, arm C job 57917826 — 64 shards each, my staged `collect_gen1_config.json` (R=16, T=0.1, ε=0.10, seed pi ckpt), OUTDIR `/scratch/dm1487/rl_growth/{N,C}/gen1_collect`, staged buffers, `PY=$NAMO_PYTHON ENV_FILE=env.amarel.sh`. Note: runtime ran long (2h+ vs the predecessor's ~33 min) — a heavy slow tail of hard/2push shards on co-tenanted main-redhat nodes.

## Result + Verdict
_(auto)_ — H→E→V, verdict on numbers only; every table by difficulty × BOTH horizons.

## Next
_(tbd)_

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_

### Growth gen-1 — Result (Claude, 2026-07-08; orchestrator-driven post-spend-cap)
Collection 94-95% coverage both arms (pathological-scene tail cut per announce; artifacts + per-tier coverage in Run). Kill-signal-1 PASS both (C hard coverage 0.694, buffer 8,018 hard unique solves — ~2x seed). Revalidate@0.1 evidence: C dropped 102/~4.3k sampled (~2.4%, jitter-scale — ruling upheld). Warm-start both arms; N 67,533 BC rows / C 93,591 (+39% — multi-push rows are denser, as the diet predicts).
**TESTSET (both tiers, keyed-join agg after the positional-join defect — 917 mismatches — was root-caused and fixed):**
| open@2 (2push) | easy | med | hard | all |
|---|---|---|---|---|
| seed (armA gen1) | 27.3 | 13.7 | 6.7 | 14.3 |
| N growth-gen1 | 26.9 | 14.4 | 7.8 | 14.9 |
| C growth-gen1 | 28.2 | 14.7 | 7.5 | 15.2 |

| open@1 (1push) | easy | med | hard | all |
|---|---|---|---|---|
| seed | 82.0 | 64.2 | 27.7 | 57.9 |
| N | 81.9 | 67.8 | 25.9 | 58.6 |
| C | 83.2 | 66.4 | 26.8 | 58.9 |

**Read: FLAT both arms, both tiers (+0.6-1.0 all; hard within noise). Neither diet moved the testset in one growth generation — 3rd consecutive near-flat generation across 3 data treatments (ReST^EM first-iteration-plateau signature). Gate 1 of 2: gen-2 flat ⇒ per-arm kills fire + method verdict. Dev dashboards pending.**

### SEARCH AUDIT (Experiment 0) — the RL pi was never graded on the project's own metric
_(Claude, 2026-07-09)_ The growth-arms "flat" verdict was measured GREEDY (reactive open-rate). But the north-star is sims-to-solve UNDER SEARCH — never run on the RL pi (V-head hung, deploy was greedy-only). Audit: RL growthC-gen1 pi as the `combine=q` ranker in best-first (`time_bestfirst.py`, hmax2, budget900), interleaved 3-way with NoHz-v3 + random on identical nodes (983/1018 scenes, 83/100 shards; 17 hard-heavy shards FAILED — hard n=237, a floor). Sims machine-independent (primary); wall-time NOT pooled (mixed iLab microarchs, unpinned) so reported paired only.

**Sims-to-solve (the objective = efficiency = intelligence):**
| tier | metric | NoHz-v3 | RL-pi | random |
|---|---|---|---|---|
| hard (n=237) | solve@900 | 92% | **90%** | 78% |
| | median sims | **10** | 34 | 118 |
| | mean sims | 90 | 138 | 215 |
| all (n=763) | median sims | 4 | 22 | 38 |
| | mean sims | 50 | 108 | 113 |

**Verdict [on numbers]:** (1) the greedy plateau was a MIS-GRADE — under search the RL pi nearly matches NoHz-v3 on solve-rate (hard 90 vs 92) and beats random decisively (hard median 34 vs 118 sims). The RL line is NOT falsified as a ranker. (2) BUT on the metric that IS the project — efficiency — RL-pi is only partway: 3.4× NoHz-v3's median sims on hard (34 vs 10), and its tail-heavy mean (all 108) sits at ~random (113), from a minority of scenes where it confidently commits to a DEAD setup and search must exhaust it. That dead-setup tail is the exact missing-negatives signature: NoHz-v3's median-10 efficiency IS its 300k dead-labels. **The efficiency gap 34→10 on hard is a direct price tag on dead-setup knowledge → motivates Experiment 1 (mine search-verified negatives → supervised recipe; target: RL-pi hard median 34→~10).** Note also RL-pi < random on EASY sims (mean 98 vs 27) — overconfidence on easy scenes, same mechanism.

## Status reconciliation (2026-08-12)

**Closed as `done` — dead-ended, not superseded.** Only generation 1 ran (both arms flat); the kill-gate required two flat generations, so the experiment never reached its own decision point. No registry rows, no RESULTS.md entry, and no later card cites it. **Dangling:** generation 2 for both arms was never launched. Its search-audit finding (the price of dead negatives) informally seeded the curriculum-ladder framing, but no card carries that link explicitly.
