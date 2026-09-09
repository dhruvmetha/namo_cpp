---
status: live
type: experiment
---
# Two-object real-scene qualification at 1 mm

User request: verify the two-object two-hop scenes before physical navigation trials.

## Plan

Check donor_v2_all/med2-med2/twohop_00043 and targeted_v1/hard1-hard1/twohop_00007 without robot motion. Reuse the audited materialize_keyhole2.py with a pinned copy of the current 0.001 m car configuration and archived exhaustive algorithm. Require ordered 2→1→0 hops, blocker_0 then blocker_1, exact post-K1 input to K2, and mechanical independence within 2 mm / 1 degree. Start at depth 2 for the historical medium pair and depth 1 for the historical hard pair. Record censored or invalid pairs explicitly.

Run both ignore and penalise using the real-stack MuJoCo navigation worktrees (robot_control d869420c, namo_cpp 3df35995), fresh environment per arm, speed 0.4, 30 Hz, 90 s planner timeout, and 120 s simulated-time cap. These are simulation qualification screens; the user supplies physical verdicts later. No model ranking comparison or canonical test-set evaluation is being claimed.

## Run

Output and complete configuration/input hashes: /home/dhruv/projects_dhruv/namo/robot_control/real_exp/results/twohop_candidates/twoobject_margin_0p1cm_20260909T181512Z/run.json. Existing source configurations, environments, camera service, and reset checker remain unchanged. All long-running stages launch as background processes with separate logs and materialized outputs.

## Result

Pending.

## Preflight correction

The live Python labeler saves setup states but omits successful terminal states, forcing the audit materializer onto minimum-cost recorded fallbacks. Supersede those preliminary K1 outputs with the existing clean real-inventory-twohop checkout at 339a0dcb, whose exhaustive logs retain every successful terminal state; retain the live compiled binding and pinned 1 mm config. Navigation is repeated in navigation_current_controller/ with the live controller YAML (goal_tolerance_ratio 0.15, rotation_tolerance_deg 4.0). The simulation worktree differs from live navigation source by its clock/config injection and harness support; its path follower and executor are unchanged. The initial navigation/ outputs preserve the prior 0.2 / 2.5-degree settings for traceability.
