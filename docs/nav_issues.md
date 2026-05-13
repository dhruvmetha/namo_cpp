# Diff-Drive Nav: Outstanding Issues

Snapshot of where the navigation stack is principled, where it's hacky,
and what to fix next. Captured for the diff-drive car (350 g chassis,
0.5 floor friction, MuJoCo `<velocity kv=0.75>` wheel actuators).

The push pipeline is a separate concern; this document is **nav only**.

---

## 1. Plant context (read this first)

We are running an unrealistically light chassis on an unrealistically
slippery floor with an unrealistically aggressive velocity actuator.
Many "controller bugs" are dynamics artifacts of this combination:

- **Chassis mass**: 350 g. Yaw inertia ≈ 3×10⁻⁴ kg·m². Tiny.
- **Floor friction**: μ ≈ 0.5 (matched to `nav_env_3000e.xml`).
- **Wheel actuator**: `<velocity kv=0.75>`, force range ±0.3 Nm.
- **Solver**: `implicitfast` integrator, elliptic friction cone,
  default `solref="0.004 1"` contact regularization, 2 ms timestep.

The above defines the plant we're committing to. All controller design
should be evaluated against this plant.

---

## 2. Code map (current state)

| Component | File | Role |
|---|---|---|
| `WavefrontPlanner` | `include/wavefront/wavefront_planner.hpp` | BFS planner on inflated occupancy grid. Returns 2D waypoints. |
| `DiffDriveNavigation::execute` | `src/navigation/diff_drive_navigation.cpp` | Top-level nav state machine. Segments path, then rotate-drive per segment, final rotate. |
| `rotate_trapezoidal` (default) | same | Inner rotation controller. Closed-loop sqrt-distance profile. |
| `drive_trapezoidal` (default) | same | Inner drive controller. Closed-loop sqrt-distance profile + small heading-correction P term. |
| `rotate_pd`, `drive_p` | same | Legacy PD versions kept under `Mode::PD`. Unstable on this plant. |
| `RLEnvironment::navigate_to` | `python/namo/cpp_bindings/rl_env.cpp` | Python entry point for nav-only A→B testing. |

Default mode is `Mode::TRAPEZOIDAL`. PD path is preserved but should not be used.

---

## 3. What works (verified)

- **Rotation accuracy**: trapezoidal rotate converges to within ~0.4° of
  target heading on isolated tests. No rebound (the original bug from
  the bang-bang controller).
- **Drive accuracy**: trapezoidal drive converges to within ~5 mm of
  endpoint on clear corridors.
- **End-to-end nav from default start to a clear goal in 3000e**: ~25 mm
  position error, ~0.4° yaw error, ~22 s travel time on average.
- **`navigate_to` Python binding**: works for free-space A→B without
  invoking the push primitive.

---

## 4. Outstanding issues (ordered by leverage)

### 4.1 Magic numbers picked by guess-and-check

Every controller gain in the codebase was chosen empirically without
plant identification. None of these have a derivation backing them:

| Symbol | Current value | Where | Justification |
|---|---|---|---|
| `Params.alpha_max` | 5.0 rad/s² | `diff_drive_navigation.hpp` | "Below where I saw oscillation" — no measurement |
| `Params.accel_max` | 0.5 m/s² | same | Guessed |
| `Params.theta_converged` | 0.01 rad (0.57°) | same | "Looks tight enough" |
| `Params.xy_converged` | 0.005 m | same | "Looks tight enough" |
| `K_heading` (drive heading P term) | 2.0 | `diff_drive_navigation.cpp` | Guessed |
| Wavefront safety margin | 2.5 cm beyond half-diagonal | `rl_env.cpp` | Picked because we observed ~25 mm cross-track drift |
| `Params.angular_speed` | 1.0 rad/s | `diff_drive_navigation.hpp` | Original config; not re-derived for trapezoidal |
| `Params.linear_speed` | 0.10 m/s | same | Original config |

**Principled fix**: one plant-identification experiment.
Step input → record response → fit linear model → derive gains from
the desired closed-loop poles. Currently missing.

### 4.2 Plant identification step missing entirely

We never measured:
- Inner velocity-loop bandwidth (`kv` against actual wheel inertia + chassis coupling).
- Maximum chassis angular acceleration achievable smoothly.
- Maximum chassis linear acceleration achievable smoothly.
- Floor + caster friction coefficients in steady-state rolling vs braking.
- Contact regularization rise time under typical loads.

Without this, every gain is a guess and every "the plant can't deliver
that" diagnosis is hand-wave.

### 4.3 State machine over-rotates

`DiffDriveNavigation::execute` calls `rotate_in_place` for *every*
segment heading change, regardless of magnitude. A 5° heading error
between consecutive segments triggers a full rotate phase: stop the
chassis, spin in place, settle, then drive. This is pure overhead.

**Principled fix**: skip rotate phase when heading error is below the
threshold the in-drive heading-correction term can absorb. Threshold
derivation requires knowing the heading-correction's authority bandwidth
(see 4.1).

### 4.4 Brake-induced rebound at phase transitions

Velocity actuator with `ctrl=0` applies torque proportional to current
wheel velocity (a brake). At every phase-transition or end-of-phase,
the wheel command jumps. Residual chassis momentum interacts with the
brake torque to produce a small reverse kick. This is the same
mechanism as the original rotation rebound, just smaller and at every
seam in the state machine.

We observed it as visible jitter at rotate→drive and drive→rotate
transitions. Currently unaddressed.

**Principled fix**: at end of each phase, command
`ctrl = current_actual_wheel_ω`, then ramp it down at `α_max` until
chassis is at rest. Zero velocity error → zero brake torque. Or switch
actuator to `<motor>` (torque control) so `ctrl=0` means coast.

The motor-actuator switch was tried in commit `9e7f1c5` and reverted
due to slip at startup; would need a custom inner velocity loop.

### 4.5 Phase exit on position only, no velocity continuity

`rotate_trapezoidal` exits when `|err| < theta_converged`, regardless
of `yaw_rate`. Actual yaw rate at exit is non-zero (small but uncontrolled).
The next phase inherits this residual rate. There's no velocity-state
guarantee at phase boundaries.

We tried adding a velocity gate (`|yaw_rate| < rate_converged`); it
caused the chassis to oscillate around target without ever satisfying
both gates (the trapezoidal commanded ω is ~0 at err=0, but residual
chassis rate plus brake torque keeps the chassis moving).

**Principled fix**: same as 4.4. Active velocity matching at boundaries.

### 4.6 Hardcoded chassis dimensions in nav inflation

```cpp
double half_x = std::max(0.035, robot_info.size[0]);
double half_y = std::max(0.0525, robot_info.size[1]);
```

The numbers `0.035` and `0.0525` are chassis half-extents (with wheels)
that I read out of the XML by hand. The `robot_info.size` returned by
`NAMOEnvironment::get_robot_info()` is `[0.0175, 0.035]` — only ONE of
the two chassis collision half-boxes, not the full chassis footprint.

This is a bug in the env-side robot-info computation, papered over with
hardcoded numbers in `navigate_to`.

**Principled fix**: have `NAMOEnvironment::get_robot_info()` compute the
true chassis bounding box including wheels, casters, and any chassis
appendages, by walking the MuJoCo geom tree.

### 4.7 Heading-correction P term in drive is ad-hoc

```cpp
omega_corr = K_heading * heading_err  (clamped)
```

Sits on top of the trapezoidal sqrt-velocity profile inside drive. No
analysis of the closed-loop dynamics it creates with the inner velocity
actuator. Stability not verified; gain not derived.

**Principled fix**: treat drive as a 2-DOF tracker on cross-track + heading
error (Stanley or feedback-linearized form), with gains chosen for desired
closed-loop pole locations.

### 4.8 Waypoint path treated as gospel

Wavefront produces piecewise-linear 8-connected paths. The car cannot
actually follow these in continuous time (heading jumps of 45° are
infeasible kinematically). The state machine "fixes" this by inserting
in-place rotations — a feasibility hack, not a smooth trajectory.

**Principled fix**: smooth the path into a feasible curve before tracking.
Options: spline interpolation, hybrid A*, RRT* with Reeds-Shepp curves.
Current unsmoothed paths are why we have a state machine at all.

### 4.9 Phase budget independent of segment geometry

```cpp
const int max_phase_steps = p.max_nav_steps - steps_used_total;
```

Where `max_nav_steps = 6000`. Every phase gets whatever's left of the
6000-tick budget. A 5 cm drive and a 50 cm drive get the same budget;
short segments have huge slack, long segments can timeout on slow plants.

**Principled fix**: per-segment budget based on trapezoidal travel time:
`t_max = segment_length / v_max + 2 × ramp_time + safety`.

### 4.10 `sharp_turn_threshold` is a magic 0.35 rad

`segment_path()` decides when to start a new segment based on heading
change exceeding `sharp_turn_threshold = 0.35 rad ≈ 20°`. Wavefront
paths have heading changes in 45° increments, so 0.35 separates
"45° change" from "no change." Could be anything in (0, 45°); 0.35 is
arbitrary inside that range.

**Principled fix**: derive from grid resolution + chassis turn capability.
Or eliminate by smoothing path (4.8).

---

## 5. Issues we won't address (and why)

These showed up during analysis but are out of scope for "fix nav for
this plant":

- **Real-robot fidelity** (heavier chassis, higher friction, different
  actuator). Project uses this exact plant deliberately.
- **State-machine vs continuous controller** (regulated pure pursuit,
  MPC). Bigger structural change; deferred until we know if 4.1–4.7
  fixes are sufficient.
- **Push-side failures**. Disjoint from nav. Push primitive has its own
  controller logic.
- **Sim-to-real port**. Project goal does not currently include hardware.

---

## 6. Recommended order of work

Sorted by leverage-per-line-of-code:

1. **Plant identification** (4.1, 4.2). One Python script, ~2 hours.
   Step-response measurements for chassis-rate tracking, brake decay,
   linear acceleration. Output: numbers for every gain currently picked
   by guess.
2. **Velocity-matching at phase boundaries** (4.4, 4.5). ~30 lines C++.
   Eliminates seam jitter without changing state-machine structure.
3. **Skip rotate when heading error is small** (4.3). ~10 lines C++.
   Cuts redundant phase transitions and their associated transients.
4. **Fix `get_robot_info()`** (4.6). ~20 lines C++. Removes hardcoded
   chassis numbers from `navigate_to` and any other callers.

After (1)-(4) the trapezoidal state machine is "principled" in the sense
that every number traces back to a measurement or kinematic identity.

If results are still rough after that, then move to bigger structural
changes:

5. Smooth the wavefront path before tracking (4.8).
6. Replace state machine with regulated pure pursuit (Stanley / feedback
   linearization).
7. 2-DOF heading + cross-track tracker for drive (4.7).

---

## 7. Reference: what we tried and abandoned

- **PD on chassis yaw** (rotate_pd): limit-cycles on this low-friction
  plant. Brake-induced friction reversal at handoffs. Replaced by
  trapezoidal sqrt profile.
- **PD on drive distance** (drive_p): same family of issues at end of
  drive. Replaced by trapezoidal sqrt profile.
- **`wait_after_phase`** (zero-control settle): reintroduces brake
  rebound; replaced by no-wait + trapezoidal naturally decelerating.
- **Pure pursuit** (`pure_pursuit_along`): removed as dead code;
  end-of-segment oscillation. Worth restoring in regulated form
  (see issue 4.8).
- **`ramp_decel`** (linear command ramp at phase end): unused. Could
  serve as the velocity-matching mechanism in 4.4 with small changes.
