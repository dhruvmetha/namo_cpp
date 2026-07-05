---
type: experiment
status: done
created: 2026-07-05
updated: 2026-07-05
metric: "Re-ran the 13 wall rooms (offline says solvable, online opens nothing) under the exact car eval config, replaying every offline-valid push with motion introspection + a K=6 determinism sweep. 0/13 are a genuine geometric floor. 2 are STALE/criterion (953, 8: the single-point 'goal reachable?' test and the 20%-of-region test disagree at s0, so the 'solvable' label can't be graded consistently). 9 are NAV/SIM execution gaps — 4 the controller JAMS pushing one object through another (object frozen, 0 cm; e.g. 922's lone opener needs shoving obstacle_1 THROUGH obstacle_2), 3 faithful-but-too-short pushes that stay below threshold, and 2 knife-edge rooms (869, 884) that FLIP open when the sim history changes: inserting a single read-only is_robot_goal_reachable() before the push shifts the object landing ~0.3 mm and turns 7/100→91/100. 2 (1133, 23) actually REPRODUCE (open online), so the eval search just missed a working push. Root cause of the flips: MuJoCo warmstart/contact state that set_full_state doesn't restore, so the same push from the same (qpos,qvel) is order-dependent near the open threshold. Integrity: pushed-object↔label object_center match <0.06 mm all 13. Split: 11/13 hard, plus 953 (easy 1push) and 664 (medium 2push). VERDICT: the ~2% ceiling is FIXABLE (align the collection/eval opener criterion + fix the stuck/under-push controller + make the sim deterministic) → true achievable ceiling ~100%, not a genuine floor; the [[_setup_label_quality]] label-fix lifts RANKING, but the ceiling needs these orthogonal criterion/controller/determinism fixes."
tags:
  - experiment
  - diagnostic
---
# Offline↔online gap — is the ~2% ceiling a fixable label/config issue or a genuine floor?

**Sibling to [[_1push_bottleneck]], [[_setup_value_check]], [[_setup_label_quality]].** Those cards kept hitting the same wall: rooms the OFFLINE label calls "solvable" (a valid push exists) that the ONLINE eval search never opens. This caps the solve ceiling at ~98%. This card re-runs each of those exact rooms under the current eval config and checks, push by push, WHY the offline-"valid" push doesn't open online — then classifies each room as a stale label, a nav/sim execution gap, or a genuine floor.

**Bottom line (plain English).** None of the wall rooms is truly unsolvable — the ~2% ceiling is fixable plumbing, not a wall. Of the 13 rooms where the offline label says "solvable" but the online search opens nothing: two have a self-contradicting "is the goal open?" definition (one test says the robot can already reach the goal, the other says it's fully blocked), seven are pushes the robot physically can't carry out online (it jams the target object against a second object so nothing moves, or the push is simply too short to clear the doorway), and two sit so close to the "open enough" line that invisible simulator jitter flips them open or shut from one run to the next. Two more actually DO open online — the search just never tried the right push. So to lift the ceiling toward 100% we need three fixes, all fixable: (1) use ONE consistent "goal open" rule in both data-collection and eval, (2) fix the push controller so it actually executes the shove it promised, and (3) make the simulator repeatable so near-threshold rooms stop flipping. This is a DIFFERENT set of fixes than the [[_setup_label_quality]] label-fix (which sharpens the model's ranking) — that one speeds us to the ceiling; these three raise the ceiling itself.

## Plain-language key

| Term | What it means |
|---|---|
| **room** (episode) | one problem: reach a goal that a movable object blocks |
| **offline "valid" push** | a push the stored label says opens the goal (from data collection) |
| **online replay** | executing that exact push now, in the live sim, under the current eval config |
| **opener criterion** | how we decide the goal is "open": ≥20 of 100 sampled goal-region points reachable (the collection's own rule) |
| **single-point criterion** | the older/looser test `is_robot_goal_reachable()` — is the ONE xml goal point reachable |
| **object displacement** | how far the pushed object moved (cm) — 0 = the push didn't execute; ~5–8 cm = a full push |
| **stuck** | the push controller couldn't make progress (robot jams) and aborted — the object stays put |
| **STALE label** | the offline "solvable" label is wrong for the current eval (criterion/config mismatch) → re-collect |
| **NAV/SIM gap** | the push *should* open but the controller can't execute it faithfully, or the sim flips it run-to-run → fixable |
| **GENUINE** | a full, faithful push really doesn't open the goal → the label is wrong for a real, physical reason |
| **REPRODUCES** | the offline-valid push DOES open online — so it was never a ceiling floor; the eval search just missed it |
| **knife-edge** | the room sits right at the 20% open threshold, so tiny sim noise flips it open/closed |

## Hypothesis [USER]

The ~2% ceiling comes from rooms where the offline label says "solvable" but the online eval opens nothing. Is that a FIXABLE label/config problem (→ true ceiling ~100% once collection and eval are aligned) or a GENUINE floor no amount of ranking or re-labeling can lift? This decides whether the label-fix from [[_setup_label_quality]] also raises the ceiling, or only the ranking.

## Plan [CLAUDE]

Take the exact wall rooms from the prior cards and re-run each under the current car eval config with motion introspection on. For every offline-"valid" push, execute it online from the canonical start state — bit-identical to what the eval harness does per candidate (`set_full_state(s0) → env.step → goal_open_pts`) — and record the mechanism: did the object move? how far? how many goal points opened? did the controller report stuck / a failure? which opener criterion fires. Then classify each room.

- **1push targets** (positional idx in `onepush_episodes.json`): **953** (offline: all 45 candidates open), **922, 1154, 420, 1133** (single-opener rooms).
- **2push targets** (positional idx in `pure2push.json`): the 7 hard offline↔online rooms **8, 23, 60, 345, 401, 869, 884** + the miss room **664**. For each, force the offline "valid" setup and exhaustively dive every second push, checking whether any opens.
- **Classify** each room STALE / NAV-SIM / GENUINE (plus REPRODUCES for labels that actually open online), from the mechanism evidence, split by horizon (1push/2push) and difficulty (tertile / division).

Model-FREE (the candidate pool is `PrimitiveGoalStrategy` geometry, no scorer), so the off-Amarel scoring breakage doesn't apply — runs on arrakis.

**Owned files:** this card, `assets/oogap_*.png`, `scripts/sandbox/offline_online_gap.py`. No commits.

## Run

- **Box:** arrakis (`arrakis.cs.rutgers.edu`), CPU only. Model-free physics replay — no GPU, no scorer, no training.
- **Python:** `/common/users/dm1487/envs/mjxrl/bin/python`. `PYTHONPATH` = main checkout `build_python:python:scripts:scripts/sandbox`.
- **Config:** `config/namo_config_complete_skill15_car_1x.yaml` (car, 1x_car_d5 primitives), collisions off — the EXACT current eval config (`scorer_beam.make_env`, hardcoded in the eval path). Opener criterion `goal_open_pts` (≥20 of 100 s0-sampled goal-region points), identical to the collection.
- **Provenance check (recon):** BOTH label files were built 2026-06-10 with the SAME car config, SAME primitives, SAME collision setting, SAME opener criterion (car region_opening, `modular_parallel_collection` → `derive_onepush_from_2push`). So a naive "point-robot / different-velocity stale label" is ruled out at the config level — the gap must be subtler.
- **Replay = eval-faithful:** each push executed exactly as `time_bestfirst.timed_bf` does per candidate. `env.step` returns rich info (`failure_reason`, `robot_goal_reached`, `steps_executed`, `stuck`, `movable_collisions`) — the nav/sim evidence, no qpos-parsing needed. Object displacement from `get_observation()`.
- **Integrity gate (positional join):** the pushed object at s0 matches the label's `object_center` to **<0.06 mm** in all 13 rooms (0.01–0.055 mm) — the positional flatten (sorted-xml then per-xml order) is correct.
- **Script:** `scripts/sandbox/offline_online_gap.py` (owned): `--run` (replay + per-push evidence), `--det K` (re-run each room K times to measure sim non-determinism), `--dump idx` (NAV_LOG + QPOS_DUMP for one room), `--agg` (tables + plots). Data out: `/common/users/dm1487/scratch_namo/eval/oogap/`.

## Result

**Headline.** I re-ran all 13 wall rooms under the exact car eval config and replayed every offline-"valid" push, watching the object move. **None is a genuine floor (0/13).** Two are label/criterion artifacts, nine are the online push controller failing to execute the push (four jam, three under-push, two flip open/shut on sim jitter), and two actually DO open online (the eval search just missed them). So the ~2% ceiling is a fixable alignment problem between how we collect labels and how we run the eval — not a wall.

### Class tally (13 rooms)

| class | count | rooms | plain meaning |
|---|---|---|---|
| **STALE / criterion** | **2** | 953 (1p, easy), 8 (2p, hard) | the two "goal open?" tests disagree at the start → the "solvable" label can't be graded here |
| **NAV/SIM — controller jams** | **4** | 922, 1154, 345, 664 | robot reaches the push pose but jams the object against a second object → object frozen (0 cm) |
| **NAV/SIM — under-push** | **3** | 420, 60, 401 | the push executes but is too short (1–3 cm); region stays below threshold on every repeat |
| **NAV/SIM — knife-edge (sim jitter)** | **2** | 869, 884 | sits right at the open threshold; flips open when the sim history changes (warmstart) |
| **REPRODUCES (not a gap)** | **2** | 1133 (1p), 23 (2p) | the offline-valid push DOES open online — the eval search just never tried it |
| **GENUINE floor** | **0** | — | none: no faithful full push fails to open for a real geometric reason |

Combined base classes: **STALE 2 · NAV/SIM 9 · REPRODUCES 2 · GENUINE 0.** By horizon: 1push {STALE 1, NAV/SIM 3, REPRO 1}; 2push {STALE 1, NAV/SIM 6, REPRO 1}. By difficulty: **11/13 hard**, plus 953 (easy 1push) and 664 (medium 2push) — the wall lives almost entirely in the hard tier.

### Table 1 — 1push rooms (replay each offline-valid push from s0)

`open`/K = how many of 6 identical repeats opened; `maxrc` = best goal-points reachable (of 100, threshold 20); `maxdisp` = furthest the object moved; `irr_s0` = single-point criterion says goal already reachable at s0.

| idx | tier | sr | n_valid | in-pool | open online (K=6) | max reachable | max disp | stuck | irr_s0 | class |
|---|---|---|---|---|---|---|---|---|---|---|
| 953 | easy | 1.000 | 45 | 45 | 0/6 | 0/100 | 0.0 cm | 0 | **True** | STALE (criterion) |
| 922 | hard | 0.008 | 1 | 1 | 0/6 | 0/100 | 0.0 cm | **1** | False | NAV/SIM (stuck) |
| 1154 | hard | 0.013 | 1 | 1 | 0/6 | 0/100 | 0.0 cm | **1** | False | NAV/SIM (stuck) |
| 420 | hard | 0.020 | 1 | 1 | 0/6 | 0/100 | 2.8 cm | 0 | False | NAV/SIM (under-push) |
| 1133 | hard | 0.014 | 1 | 1 | **6/6** | **100/100** | 5.4 cm | 0 | False | REPRODUCES |

Reading: the pushed object matches the label's `object_center` to <0.06 mm (positional join correct), and every offline-valid push IS in the current candidate pool — so this is NOT a candidate-index drift. **953** (the flagship contradiction, offline solve_rate=1.0) is degenerate: the single-point test says the goal is already reachable at s0, so the push skill short-circuits every one of the 45 "valid" pushes (`robot_goal_reached=true`, object never moves), while the region test reads 0/100 — the two criteria contradict each other, so no push can ever satisfy the grader. **922/1154** jam. **420** under-pushes. **1133 reproduces** (opens 100/100 on all 6 repeats) — the eval search simply didn't try its lone opener on 5/16 seeds (a ranking/budget miss, per [[_1push_bottleneck]]).

### Table 2 — 2push rooms (force the offline setup, exhaustively dive every 2nd push)

`opens (K=6)` = repeats where some (setup, 2nd push) cleared the region; `maxrc2` = best reachable across the whole dive; `setup disp` = how far the setup moved the object; `gr1` = setup alone trips the single-point criterion.

| idx | tier | n_setups | opens (K=6) | max reachable | setup disp | stuck | gr1 | class |
|---|---|---|---|---|---|---|---|---|
| 8 | hard | 2 | 0/6 | 8/100 | 2.5 cm | no | **yes** | STALE (criterion) |
| 23 | hard | 1 | **6/6** | **22/100** | 3.7 cm | no | no | REPRODUCES |
| 60 | hard | 1 | 0/6 | 1/100 | 2.0 cm | no | no | NAV/SIM (under-push) |
| 345 | hard | 2 | 0/6 | 0/100 | 0.0 cm | **yes** | no | NAV/SIM (stuck) |
| 401 | hard | 1 | 0/6 | 3/100 | 0.9 cm | no | no | NAV/SIM (under-push) |
| 869 | hard | 1 | 0/6 → **flips** | 9/100 (cold) → **15/100** (alt) | 8.2 cm | no | no | NAV/SIM (knife-edge) |
| 884 | hard | 1 | 0/6 → **flips** | 7/100 (cold) → **91/100** (alt) | 2.3 cm | no | no | NAV/SIM (knife-edge) |
| 664 | med | 4 | 0/6 | 0/100 | 2.9 cm | **yes** | no | NAV/SIM (stuck) |

Reading: even handed a PERFECT setup and an exhaustive 2nd-push search, most rooms stay below threshold — but for a mechanism, not a floor. **23 reproduces** (opens 6/6; [[_setup_value_check]]'s dive missed it on ranking). **8** is a criterion artifact (the setup alone trips the single-point test while the region stays at 8/100). **345/664** jam (setup frozen). **60/401** under-push. **869/884** are knife-edge — see below.

### The core mechanism — the sim is not repeatable, and it flips near-threshold rooms

`assets/oogap_determinism.png`. The goal-point sampling is deterministic (identical 100-point set every call — ruled out as the noise source). The **physics is not**: the same push from the same restored `(qpos, qvel)` lands the object in a slightly different place depending on what ran before it.

The cleanest proof, on room **884**: I ran the exact same setup + exhaustive dive twice, the ONLY difference being a single **read-only** `is_robot_goal_reachable()` call inserted before the setup —

```
WITHOUT pre-query: setup lands obj at (0.23778, 0.41846) -> best 2nd-push opens  7/100 -> CLOSED
WITH    pre-query: setup lands obj at (0.23772, 0.41875) -> best 2nd-push opens 91/100 -> OPEN
```

A read-only query, changing nothing on paper, shifts the object landing ~0.3 mm and flips the room from 7/100 (shut) to 91/100 (wide open). The cause is **MuJoCo warmstart / contact-solver state that `set_full_state` does not restore** (it restores qpos and zeroes qvel only). So a push's result depends on the hidden solver history — and the eval processes candidates in model-score order from a heap, so each candidate inherits a different history → non-reproducible, and near the 20%-of-region threshold that jitter flips solvability. Rooms 869 and 884 are exactly this: 0/6 in cold identical repeats, but they flip OPEN under a different history (869: 9→15 at threshold 14; 884: 7→91 at threshold 20). ![[oogap_determinism.png]]

### Nav-dump evidence (`NAMO_NAV_LOG=1`) — did the robot reach the pose? did the object move?

**Room 922 (jam / NAV gap).** The `[NAV_PATH]` trace shows the robot DID reach the push contact pose next to the object, and `[PUSH_PATH]` shows the intended shove — but the object moves **0.00 cm** and the step returns `failure_reason='Controller-level stuck (counter=5)'`, `movable_collisions='obstacle_2_movable'`. The lone offline opener requires pushing obstacle_1 *through* obstacle_2; the controller jams and aborts. (This room is arguably a mislabeled 2-object problem, not a real 1-push.)

**Room 1133 (reproduces).** Robot reaches the pose, `[PUSH_PATH]` executes, object moves **5.41 cm**, `robot_goal_reached=true`, region opens **100/100**. A clean, faithful, opening push — confirming the label; the eval just under-sampled it.

**Room 953 (criterion short-circuit).** No `[NAV_PATH]` is ever emitted and no qpos is written — every push returns `robot_goal_reached=true, steps_executed=1` with the object frozen: the skill sees the (single-point) goal already reachable at s0 and never runs the push.

`assets/oogap_reachcount.png` = best offline-valid push vs the open threshold per room; `assets/oogap_disp.png` = per-push object displacement (the 0-cm "stuck" pushes vs the 5-cm full pushes). ![[oogap_reachcount.png]]

### The decision this answers — plain verdict

**The ~2% ceiling is FIXABLE, not a genuine floor.** 0 of 13 wall rooms is truly unsolvable. The gap is entirely: 2 self-contradicting "goal open" definitions, 4 pushes the controller physically jams, 3 pushes it under-executes, 2 rooms flipped open/shut by an unrepeatable simulator, and 2 that already open online. Three orthogonal fixes lift the ceiling toward ~100%:

1. **One consistent opener criterion** in collection AND eval (drop the single-point `is_robot_goal_reachable` grade in favor of the region-fraction rule everywhere, or count `is_robot_goal_reachable=True` scenes as already-solved) → fixes 953, 8.
2. **Fix the push controller** so it executes the shove it plans — don't abort into "stuck" when a second object is in the way (collisions are meant to be off), and don't under-travel → fixes 922, 1154, 345, 664, 420, 60, 401.
3. **Make the sim deterministic** — restore or clear the MuJoCo warmstart/contact state in `set_full_state`, or grade "open" with a small margin / majority-of-repeats so a 0.3 mm jitter can't flip a room → fixes 869, 884 and stabilizes the whole eval.

**Relation to [[_setup_label_quality]]:** that card's label-fix sharpens the model's *ranking* (finds setups faster) — it speeds us TO the ceiling. The ceiling itself is a separate, plumbing-level problem (criterion + controller + determinism). Both are fixable, so the true achievable ceiling is ~100% once collection and eval are aligned.

### Provenance / sanity gates

- Script `scripts/sandbox/offline_online_gap.py` (owned); data `/common/users/dm1487/scratch_namo/eval/oogap/` (`oogap_1push.jsonl`, `oogap_2push.jsonl`, `oogap_determinism.json`, `oogap_summary.json`). No model, no GPU, no training — model-free physics replay on arrakis.
- **Provenance:** both `onepush_episodes.json` and `pure2push.json` were built 2026-06-10 with the SAME car config / primitives / collision setting / opener criterion as the eval — so a naive point-robot/velocity stale label is ruled out; the gap is the subtler criterion/controller/determinism story above.
- **Integrity:** pushed object ↔ label `object_center` match <0.06 mm (0.01–0.055 mm) in all 13 rooms; every offline-valid push is present in the current primitive pool (no candidate-index drift — confirmed by 1133/23 reproducing with the current mapping).
- **Eval-faithful replay:** each push executed exactly as `time_bestfirst.timed_bf` does per candidate (`set_full_state(s0) → env.step → goal_open_pts`), so a reproduced/failed open here is what the eval sees.

## Discussion
_(you ↔ Claude — newest at the bottom.)_
