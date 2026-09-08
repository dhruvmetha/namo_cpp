---
status: live
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

Pending.

## Corrected launch and census audit

At commit `ef3d8795`, corrected smoke array `61288401` passed all 27 arms and reproduced all 114 sampled cached-control episode outcomes exactly. The slowest smoke took 21.04 seconds. Controller `61288402` submitted full policy array `61288489` (168 bundles, 40 workers each, 15-minute task limit) and dependent aggregate `61288490`. All 168 tasks started, using 6,720 evaluation CPUs. The control full populations are reused, not rerun.

The zero-push full census `61288429` wrote 1,840 records from 1,807 unique rooms with zero room-processing errors, but failed while JSON-sorting its summary: unclassified exclusions had a null dictionary key alongside named boundary kinds. The summary now serializes that category as `unclassified`, with a regression test. The controller correctly stopped on the census failure; policy workers and their dependent aggregation were already submitted and continued independently.

Before any group outcomes were observed, the census showed 639 eligible multi-object groups: 625 alternative-blocker groups and 14 joint-blockage markers. Source-tier-only round-robin would have selected 24 alternatives and no joint-blockage groups. The pilot selection now stratifies by boundary kind as well as source leg/tier, so the pilot covers the structural distinction under investigation. This is a pre-outcome sampling adjustment, not selection by success or simulated cost.
