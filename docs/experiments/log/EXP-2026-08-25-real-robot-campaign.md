---
type: experiment
status: live
created: 2026-08-25
commit: 96667f6
metric: hardware solve rate by tier and horizon, search vs reactive, on physically buildable scenes
tags: [experiment, real-robot, sim2real, hardware, campaign, orchestration]
---
# Real-robot campaign

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is still one local raw-Q ranker that orders which pushes the search tries. This campaign asks what survives contact with a physical table.

⛔ **This card is the ONE place campaign state lives.** It holds only what cannot be derived. Branches, worktrees and who is ahead of whom come from `scripts/campaign_state.sh`, never from prose here, because prose about mutable facts goes stale and then lies. If you want to know where a branch is, run the script.

## The question

Sim says these scenes are solvable. Does the robot solve them, and does the ranker's ordering advantage survive the sim-to-real gap? Split by tier and horizon, never aggregate.

## Standing decisions

| decision | date | who |
|---|---|---|
| ⛔ NEVER tune sim friction, mass or any physics constant to match hardware. The gap is a result to characterise, not to close. Every label came from the current constants; retuning invalidates 2.4M sims | 2026-08-26 | USER |
| Physics frozen on BOTH sides for the study's duration. Hardware's tier floors and corridor data came from these constants too, so this protects their study as much as ours | 2026-08-26 | both |
| Build search AND reactive execution. Nobody knows which handles the gap better, so the table decides | 2026-08-26 | USER |
| v1 build sheets stay untouched; v2 ships to its own directory. Hardware validated v1 and nothing should build against a moved target | 2026-08-25 | CLAUDE |
| Corridor width is measured and reported, never used as a filter. The threshold comes from a function whose name lied once already | 2026-08-25 | CLAUDE |

## What shipped

593 build sheets in `handoff/real_scene_build_sheets_v2/`, 100 per tier per horizon except hmax2/hard at 93, which is every qualifying scene in the pool. Hardware validated 593 of 593 with their own geometry and checksum checker.

Tiers come from pure depth-2 enumeration over 1478 scenes, 170,095 first pushes, ~2.4M sims. The old search sweep left 1965 of 44,278 first pushes unexpanded, so its solve rates were measured against whatever the search happened to touch.

Alongside the sheets: `post_push_clearance.csv`, `marker_retarget.csv`, `ranker_contact_offset.csv`, `uniform_offset_baseline.csv`, `uniform_offset_distributions.csv`. The README in that directory explains each.

## Findings so far

**Rotation is the sim-to-real gap, and it is interface geometry, not a constant.** A real block pushed off-centre keeps rotating, 2.126 deg/cm at 3.0 cm offset and 2.416 at 3.5, R^2 0.980 and 0.972. Near the face centre it turns ~3 deg in 2 cm then stays flat. Our sim self-squares at every offset.

`physics` settled the cause (branch `physics/rotation-probe`, `887003b`, writeup at `probe/FINDINGS.md`). Torsional friction is falsified twice over: 0.0001 gives bit-identical results to 0.005, and raising it REDUCES rotation. No constant can work, analytically: push force and floor yaw resistance both scale with mu*m*g, so coupling reduces to d/c^2 and neither mass nor friction appears. A 10,368-run sweep over the full product of friction, torsional, mass, pusher friction, solref, cone, condim and impratio gives median |coupling| 0.006 deg/cm. The 31 combos that do clear 2.126 are numerical accidents: zero at every offset then a cliff at one, where hardware grows in proportion. Zero of 10,368 clear 1.5 deg/cm at both 3.0 and 3.5.

The ground model is fine. MuJoCo's block-floor yaw friction radius is 8.28 cm against ~9.0 implied by hardware. The discrepancy is entirely at the pusher: the car's flat 7 cm face and the block's flat end face form a perfect planar mate, 3-4 loaded contact points across 4.3 cm, which is a kinematic constraint on relative yaw. The block's heading tracks the car's to within 0.3 deg. Real faces are flat to maybe 0.1 mm and touch at high spots, so they have far lower angular stiffness and cannot hold a block square.

Swapping the car's face for a cylinder produces rotation but fails the centre-contact case, giving -18.0 deg where hardware self-squares to about +1. The real interface behaves like a point contact at corner offsets and a flat mate near centre, and no single geometry tested switches between the two.

**Practical read:** solve rates that turn on WHICH push opens a region stay trustworthy, and translation is unaffected. Anything depending on the block's final HEADING after a corner push is systematically wrong, and wrong in one direction, since sim always under-rotates. That is why chained plans drift and why re-observing between pushes matters.

**The ranker mildly prefers centre contacts, and about half of that is a longer-push preference.** Spearman -0.414 over 200 scenes, falling to -0.232 once simulated travel is partialled out. Centre contacts travel further (-0.362), so the flattering "it implicitly avoids bad physics" reading is not supported. A prediction that top scores would be degenerate was falsified: the median scene has one candidate within 0.001 of the top out of 125.

**Corridor width after the push was never checked by anything.** The generator's margin test only looks at the route with the movable deleted. 98 of 593 scenes have a best route under 11 cm. Hardware's own reading is that a corridor the car can spin in needs 9.9 cm, not the 8.0 our wavefront assumes, so their calibration ladder settles a number that shifts every tier label on both sides if it comes back high.

## Trial matrix: CROSSED, 56 runs [USER 2026-08-26]

Execution mode crosses with the model-vs-uniform arm: 14 scenes x 2 arms x 2 modes = **56 hardware runs**, ~10 hours of table time against ~5, on a 2026-09-15 deadline.

Both `real_robot` and CLAUDE recommended NOT crossing, and running reactive as a separate comparison after the primary 28. USER decided otherwise, on the grounds that nobody knows whether planning or reactive handles the sim-real gap better and the table is what settles it. Given that the gap attacks lookahead specifically, that is defensible.

⚠ **THIS REQUIRES A DATED AMENDMENT TO THE PRE-REGISTRATION, WRITTEN BEFORE ANY TRIAL RUNS.** `docs/ICRA_REAL_ROBOT_STUDY.md` states the design was locked before collection. With 0 of 28 matrix rows collected this is an amendment, not a violation, but only if it is recorded now with the date and the fact that no matrix data existed. Written after the first row lands it is something else. `real_robot` owns that doc and is drafting it. No trial runs before it exists.

If the time budget forces a cut, the study doc's cut order sacrifices easy and medium first, and the sign test's 8 hard pairs are the floor. Escalate with numbers rather than letting collection fail partway.

## The 9 marker-fail scenes stay, flagged [USER 2026-08-26]

Nine of the 593 shipped scenes leave the goal marker unreachable on every measured solution, so they will strict-fail or retarget on the table. Five are hmax2/hard, four are 1push/hard, all named in `marker_retarget.csv`.

They stay. They open the region by the simulator's own rule, so they are genuine solves under the criterion the labels use, and dropping them would take 1push/hard to 96 and hmax2/hard to 88 in the tier that already ships short of 100.

⚠ **THIS IS ABOUT THE 593-SCENE SHIPPED POOL, NOT THE 14-SCENE HARDWARE MATRIX.** CLAUDE relayed it as "they stay in the matrix", which is a different sentence about a different set, and `real_robot` caught it. Verified: all 14 matrix scenes are verdict=strict with retarget 0.0 and there is zero overlap with the nine. No second amendment is needed and the hardware matrix is untouched.

Also corrected: "report as retarget-or-fail" is wrong. Under hardware's success semantics all nine collapse to plain FAIL. Seven have no reachable cell recorded at any distance; the two that do sit beyond the 12.0 cm cap, `1push/hard_036` at 15.0 and `1push/hard_097` at 14.5.

Related and worth reading together: **51 of the 593 have a best corridor under 9.9 cm**, the width hardware calculates a car needs to spin in. If the calibration ladder puts the real threshold up there, those 51 become expected failures too. The `failure_cause` column is what will separate them from marker failures, which is another reason it lands before collection.

**None of that reaches the matrix either.** All 14 matrix scenes have a best corridor of 13.74 or better, 13 of 14 at the 30 cm measurement ceiling. But best corridor is the BEST route, not the only one: `easy_002`'s worst measured route is 9.95 cm, `med_077` and `easy_001` are both 11.2. If the planner picks a poor route on those three the run can still land near the threshold, which is exactly the case `failure_cause=corridor_too_tight` has to separate from `marker_unreachable`.

## Table-time floor, not ceiling [real_robot 2026-08-26]

56 runs is 10 to 11 hours, but **the rebuild dominates, not the trial**: four cells per scene means four rebuilds per scene, since a push disturbs the scene and the protocol rebuilds between arms. Costing this as "56 runs" understates it, which CLAUDE did.

Calendar is not the constraint. 20 days remain and data should land by ~9/8 to leave a week for analysis, so 10 hours over 13 days is under an hour a day. **Per-session reliability is the constraint.** The 3-scene pilot produced two runtime bugs, a battery swap mid-run, a radio blip and a 55 s deadband stall.

So the number to defend is **32**: 8 hard scenes fully crossed, holding both sign tests at n=8. Amendment 1 forbids cutting a single cell, because that breaks the pairing for both tests at once.

## ⚠ trials.csv has a schema hole, do not patch half of it

Reported by `real_robot` 2026-08-26. The failure-taxonomy column **does not exist**. The fixed vocabulary lives in prose at `docs/ICRA_REAL_ROBOT_STUDY.md:71` and `docs/REAL_ROBOT_TRIALS.md:119` and nowhere in code or in the file. Current header:

    trial_id,build_id,tier,axis,started_at,ended_at,command,planner_outcome,user_verdict,notes,scene_checksum,log_path

Worse, which ARM ran is not a column either. It lives inside the free-text `command` string as the substring `model HY5U_s2`. That is the same class of bug `68362d0` fixed at the flag layer and the CSV still has at the record layer: a misrouted flag now raises, but a run recorded under the wrong arm still reads as ordinary text.

So when mode lands, add `arm`, `exec_mode` and `failure_cause` in ONE pass, each with a fixed vocabulary and each parsed rather than free text. Adding `exec_mode` alone splits the matrix rows across two schemas. Both mode and arm are per-TRIAL, not per-session: the protocol rebuilds between arms and randomises arm order per scene, so a session mixes arms by construction, and the analysis is paired within scene so the grouping key must sit on the row.

**Decided [USER 2026-08-26]: the header lands BEFORE the first matrix run**, not when reactive is finished. `policy` proposes all three columns in one pass, USER approves, `real_robot` lands it. Only the 4 pilot rows exist so the cost is near zero now and rises the moment collection starts. The arm-in-free-text problem gets fixed in the same pass: with 56 runs across two crossed factors there are now four ways to mislabel a row instead of two.

## Open questions

- What corridor width does the real car actually clear? Hardware predicts 8.4 to 11 cm. Their ladder measures it.
- Does the coupling scale with contact offset as a constant, or is there a sharp flip where pushes stop self-squaring? Hardware pre-registered three outcomes before running.
- Search or reactive on the table? Genuinely open, which is why both get built.
- ~1.8% of scenes carry label noise where the simulator disagrees with its own recorded verdict on the same push, same config, same sequence. Cause unidentified, concentrated in 2-push chains.

## Landed 2026-08-26 late

`real_robot` pushed: `origin/real-robot` moved `4ecfac4 -> 3a393a5`, 29 commits. The study brief, the runbook, the three amendments, the corridor note and the nav-failure/goal-retarget runtime work are all public. The pre-registration and the code it constrains are no longer on one disk. Authorisation came from USER typing directly into that session, after it declined two relayed forms; the amendments carry a provenance note recording how each decision arrived.

Our side: reactive rule merged at `e5467c2`, parity-invariance merge at `634e841`. The parity anchor is now a property of one pool read twice (26 cases, deployed ckpt), not of the combine default.

**CHAIN COMPLETE 2026-08-27: reactive mode is on the robot.** `origin/real-robot` at `54908bb`, r2 merged --no-ff after review-as-new-work, 358 passed on the merged tree, namo reconciliation in as `bee94b6`. The runbook's NOT RUNNABLE YET block is replaced with the earned warning that an empty reactive plan is a wiring fault, not a hard scene — three separate defects today shared that exact symptom.

Sharpened seed-bug consequence, verified by `policy` repro + `real_robot` AST: the HELD path has NO retry (`_generate_plan` has the 5-attempt machinery, `_generate_plan_holding_target` has none). Under Amendment 2 both arms run held, so pre-fix ALL 56 runs would have returned empty plans logged as `exception` — a stationary robot reading as hard scenes. The bee94b6 fix was load-bearing for the matrix, not housekeeping. (The pilot ran the UNHELD path with the fix already in-tree, so pilot pacing still stands as measured.)

The matrix is UNBLOCKED. Remaining before it runs: the trials.csv header. Still open, non-blocking: tracking `real_trials/`, the robot-side `.gitignore` symlink hole, the startup refusal, the two-hop test path, recorder dedup after the matrix.

Still USER's, none blocking: trials.csv header, tracking `real_trials/` + the `.gitignore` symlink hole on the robot side, the startup refusal, the two-hop test fix.

## Seed bug found in reconciliation, shipped artifacts clean [2026-08-26]

`real_robot`'s d6ddb67 reconciliation (the "~30-commit divergence" was actually 1 ahead / 49 behind; nobody had run rev-list and all three sessions repeated the unmeasured number) landed one commit, `bee94b6`: `int(params.get("shuffle_seed", ...))` raised on an explicitly-passed None, killing the FIRST planning attempt of every run through `BestFirstRegionOpeningPlanner`; the retry carried a real seed and worked, so it read as a flake. Unpushed since 2026-08-23.

**Blast radius, traced file by file: zero shipped artifacts affected.** Only `planning_service.py` and `full_namo_planner.py` construct that planner. The exhaustive labels, tiers, marker verdicts, corridor numbers, uniform baselines and the ranker study all run env.step / numpy / BeamPlanner paths that never touch it. Exposed in principle: the DEPLOY path (namo_planner.py:1244 passes shuffle_seed=None as a present key, the exact trigger). **But the pilot was NOT exposed** — `real_robot` checked timestamps rather than accepting my inference: the fix was authored 2026-08-23 00:56, fifty minutes before the first pilot trial, sitting unpushed in the working tree the robot ran from. Zero tracebacks in all four pilot logs; the one retry-looking trial was a `NO SUBGOALS` empty return with NAMO_SCRATCH unset (already logged FAILED_ENV), not a raise. **Pilot pacing is therefore valid as measured and the 10-11h budget for 56 runs stands; do not adjust it.** My "the pilot's flakiness WAS this bug" was an unverified inference and wrong. The neat part: anyone reproducing the pilot from public history before tonight WOULD have hit the bug, since the fix lived on one disk — the exact gap the push closed. The matrix runs on the fixed tip.

## Paste-to-build shipped [2026-08-27]

Star scenes in the gallery, copy, paste the JSON to `real_robot`, their `resolve_shortlist.py` (their 95d8fd9) joins on xml and prints v2 id, tier, marker verdict and corridor numbers, or refuses with a reason. Export carries `xml` + `gallery_id` + `dataset`; stale browser stars refresh on page load; the round-trip is enforced at card-build time (`build_real_scene_cards.py` fails if any index row's xml does not resolve). Resolver never joins on bare id, since three id namespaces name overlapping scenes. Known quirk their tests surfaced: the sheet CSVs are CRLF, so naive readers get `\r` on the last column — noted in the v2 README.

## Who owns what

`namo-a1` orchestrates, owns every merge, and is the only session that commits to `feat/horizon-q-redesign`.

`physics` probes MuJoCo capability on `physics/rotation-probe`. Nothing it finds gets adopted. Rotation probe complete.

`policy` adds reactive execution across the interface, on `policy/reactive-service` in namo and `policy/reactive-deploy` in robot_control. The robot_control branch reaches hardware as a proposal, never a push to their working branch.

`real_robot` owns the hardware repo and the table. Runs the calibration ladder and the 14-scene paired matrix.

## Protocol notes that are easy to get wrong

v1 and v2 build ids do NOT correspond. `easy_007` in v2 is a different table layout. Join on `xml`, never `build_id`.

Edge indices are not comparable across the boundary without canonicalising yaw. Hardware plans on a camera capture; a symmetric block placed 180 degrees from the sheet reads 180 off and swaps face parity, so their even index below 30 is our odd one. Canonicalise yaw mod 180 and flip parity within each 30-block.

Success is 20% of the goal REGION's sampled points, not the XML marker. The two disagree often, which is what `marker_retarget.csv` records.

The paired matrix compares within scene, never pooled. The uniform baseline ranges 1.25 to 5.89 cm across scenes from geometry alone, which swamps the effect several times over.
