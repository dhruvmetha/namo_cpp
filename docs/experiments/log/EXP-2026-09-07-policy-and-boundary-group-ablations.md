---
status: complete
tags: [experiment, ablation, policy, region-opening]
---

# Policy ablations and boundary-group generalization

## User request

Evaluate the trained ablations as policies on `(room, target object, target region)` episodes, and inspect a separate `(room, fixed blocker group, target region)` generalization cohort. Use only 5 mm simulator checks.

## Plan

Single-object evaluation reuses the selected three-seed checkpoints for HY5U, no-unreachable HY5, no-family, regression-only, independent contacts, global readout, no local feature, and no learned edge identity. Run the seven ablations as closed-loop policies for up to ten executed pushes per episode. Every attempted push counts; no simulated alternatives are screened before commitment. State-local no-op and jam feedback affects subsequent choices. Preserve the original object and initial sampled region target. Report open@1/@2/@5/@10 by source horizon and easy/medium/hard, with three-seed mean and sample SD. These are policy-only comparisons; the existing hmax=2 search curves do not isolate execution mode beyond two calls.

Reuse registered HY5U/random K=10 policy controls only after reproducing their prefix outcomes under the pinned 5 mm configuration. Their common eligible population is checked by exact episode keys, not adjusted to accept missing shards. Canonical manifests contain 1,328/992 episodes; the historical policy sampler excludes empty target samples, yielding a smaller eligible population. Model-dependent omissions are not accepted. Unexpected initialization errors fail the task. Existing full search ablations are reused without repetition.

The second experiment starts with a zero-push census of both canonical manifests. Deduplicate physical initial boundaries by resolved room, initial source/target regions, and sorted blocker IDs. Keep source episode membership, fixed sampled target points, and geometry-derived singleton/alternative/joint-blockage categories. Group markers do not certify dynamic solvability or minimum push count. Select at most 24 eligible multi-object groups deterministically, before outcomes are observed. If no meaningful groups exist, report that limitation instead of constructing arbitrary object sets. Otherwise run HY5U and Random with policy K=2 and best-first hmax=2/budget=900 on the same fixed groups; source labels remain explicitly source labels.

## Execution safeguards

An isolated checkout preserves unrelated active edits. Jobs run in Amarel `main`, CPU only, with single-threaded inference and explicitly pinned 5 mm configuration, checkpoint hashes, matching Sage revision, manifest hashes, and the registered simulator binding. Run all checkpoint smokes before production; require control parity and calibrate wall limits from actual task elapsed time. Production uses bundled independent workers, up to the 6,720-CPU allocation ceiling, and reliable completion markers. Background monitoring runs at five-minute intervals and stops on completion or actionable failure.

## Run

First Amarel smoke array `61288302` completed all 27 checkpoint/control tasks at commit `3f892846`. The control-parity gate rejected HY5U rows while all sampled Random rows matched. Code inspection found that `scorer_beam.make_env` honored `NAMO_CFG=margin_5mm`, but `BeamPlanner` constructed `LiveScorer` without passing that configuration, leaving its mask renderer on the root directory's 1 mm setting. This is a mixed-margin diagnostic only and must not be aggregated. The constructor now passes the same configuration to the scorer renderer and simulator. Repeat the smoke and require parity before production.

The follow-up audit found a second path issue: the Python mask exporter resolved the primary configuration symlink before locating its sidecar, escaping `config/margin_5mm` back to the root directory. C++ preserves the caller-selected directory. Python now matches that behavior, with a regression test where the selected directory is 5 mm and the symlink destination directory is 1 mm. Array `61288360` predates this second correction and is also diagnostic-only. The independent zero-push census smoke is `61288361`.

## Results

Complete. All 21 new checkpoint arms finished both testsets; the comparison retains all 1,310 one-push and 973 two-push cached-control episodes. Mean ± sample SD across three seeds is reported in `eval/policy_group_ablations_20260907/aggregate.{json,md}` under either box's scratch root. The JSON includes per-seed results, all difficulty tiers, and excluded diagnostic keys. No training was needed and no full HY5U/Random control run was repeated.

Policy-mode findings match the main architecture conclusions: a global-only readout is much worse, while removing local sampled features alone or family ranking changes little overall. Hard episodes expose losses hidden in overall averages: HY5U versus regression-only is 41.6±1.2 versus 33.0±1.1 at one-push open@1, and 21.9±3.2 versus 15.4±3.1 at two-push open@2. Independent contacts and no edge identity also lose on those hard columns. These are policy ablations, not independent competing-method baselines. K=5/10 recovery is not depth-matched to the existing hmax=2 search experiments.

The separate 24-group pilot completed all 288 group/arm/mode tasks. HY5U policy open@2 is 76.4±2.4% versus Random's 18.1±6.4%; search solve@5 is 76.4±2.4% versus 50.0±11.0%; both reach 97.2±2.4% at 900 calls. HY5U finds two committed object-switching policy solutions per seed and 2/3/5 object-switching search solutions. This demonstrates that the fixed-group path executes pushes on different members of the same initial boundary group. It is a small, deliberately structural pilot, not a new canonical benchmark or evidence that every group requires multiple objects. The eight joint-blockage markers are geometric classifications, not minimum-push certificates. `group_aggregate.json` reports all six overlapping source leg/tier strata; those are provenance labels, not difficulty labels for the new grouped task.

## Corrected launch and census audit

At commit `ef3d8795`, corrected smoke array `61288401` passed all 27 arms and reproduced all 114 sampled cached-control episode outcomes exactly. The slowest smoke took 21.04 seconds. Controller `61288402` submitted full policy array `61288489` (168 bundles, 40 workers each, 15-minute task limit) and dependent aggregate `61288490`. All 168 tasks started, using 6,720 evaluation CPUs. The control full populations are reused, not rerun.

The zero-push full census `61288429` wrote 1,840 records from 1,807 unique rooms with zero room-processing errors, but failed while JSON-sorting its summary: unclassified exclusions had a null dictionary key alongside named boundary kinds. The summary now serializes that category as `unclassified`, with a regression test. The controller correctly stopped on the census failure; policy workers and their dependent aggregation were already submitted and continued independently.

Before any group outcomes were observed, the census showed 639 eligible multi-object groups: 625 alternative-blocker groups and 14 joint-blockage markers. Source-tier-only round-robin would have selected 24 alternatives and no joint-blockage groups. The pilot selection now stratifies by boundary kind as well as source leg/tier, so the pilot covers the structural distinction under investigation. This is a pre-outcome sampling adjustment, not selection by success or simulated cost.

Census recovery `61288700` completed in 11.34 seconds from isolated checkout `namo_policy_group_20260907_census_fix` at `9eaf1320`; the frozen pilot contains 16 alternative-blocker and 8 joint-blockage groups. The original census records and failed summary remain in `census_summary_failure_61288429/`. Unit tests passed for classification, episode identity, static eligibility, JSON serialization, deterministic pilot selection, group action switching, object-local jam pruning, and renderer clearance.

Policy array `61288489` completed 65 bundles. All remaining 103 bundles were on `halk` nodes; an in-allocation process check showed their Python children in uninterruptible `cxiWaitEventWait` filesystem waits with 0% CPU before evaluation output. These tasks were cancelled and only their indices (`9-16,72-85,87-167`) were resubmitted as `61288743`, excluding the `halk[0001-0159]` pool. The 65 completed bundles are preserved. The replacement aggregate is `61288864`; group smoke is `61288744`. The recovery uses the same evaluator and physics configuration; only the census/reporting code and placement differ. Both original and recovery controller states are retained under the campaign root for audit.

The environment-only exclusion on that retry did not appear in SLURM's `ExcNodeList`. It completed another 81 bundles, leaving 22 still on `halk`; the completed total reached 146/168. The remaining indices (`10,91-110,150`) were moved to `61288890` with an explicit `--exclude=halk[0001-0159]`, and `scontrol` verified both the exclusion and placement on `hal` nodes. The submission helper now passes `CAMPAIGN_EXCLUDE_NODES` explicitly to every child job (`ea0df67c`). Final policy aggregation is `61288918`, and replacement group smoke is `61288891`. No completed bundle was rerun.

All 168 policy bundles and four group smokes then completed. The group pilot and reducer ran as `61288935`/`61288936`. The first policy reducer `61288918` stopped because every new arm contained four extra episodes in each leg, with zero missing reference episodes. All eight extras are joint-blockage cases recovered by the additive adjacency repair `41ba1532`, which gives formerly disconnected goal regions sampled targets. They are retained in raw outputs but excluded from the precommitted comparison. A one-room no-push audit (`61289284`, 13 seconds) preceded the full 64-worker paired audit `61289289`: toggling that additive pass changed none of the fixed goal-point coordinates in all 1,772 reference-population rooms. The reducer now requires that audit, rejects missing reference episodes, and permits only census-confirmed joint-boundary extras. Final successful reduction is `61289290` at code `22066626`. The 5 mm simulator and renderer settings never change during this topology audit.
