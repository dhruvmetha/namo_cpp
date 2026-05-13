#include "navigation/diff_drive_navigation.hpp"
#include "navigation/qpos_dump.hpp"
#include "environment/namo_environment.hpp"
#include "robot/robot_adapter.hpp"
#include "config/config_manager.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace namo {

namespace {

// Wrap angle to [-π, π]
double wrap_angle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// Segment a path into runs of consecutive waypoints whose direction of
// travel agrees within `heading_threshold`. Each segment has a target
// endpoint and a constant heading. The wavefront path has 8 possible
// inter-cell directions (0°, 45°, 90°, ...), so heading changes are
// always 45° increments.
struct PathSegment {
    std::array<double, 2> end_point;   // last waypoint of this segment
    double heading;                    // direction of travel along this segment
};

std::vector<PathSegment> segment_path(
    const std::vector<std::array<double, 2>>& path,
    double heading_threshold)
{
    std::vector<PathSegment> segments;
    if (path.size() < 2) return segments;

    // Heading of first inter-waypoint step
    double cur_heading = std::atan2(path[1][1] - path[0][1],
                                    path[1][0] - path[0][0]);
    size_t segment_end = 1;

    for (size_t i = 1; i + 1 < path.size(); i++) {
        double h = std::atan2(path[i+1][1] - path[i][1],
                               path[i+1][0] - path[i][0]);
        double diff = std::abs(wrap_angle(h - cur_heading));
        if (diff > heading_threshold) {
            // Close the current segment at path[i]; start a new one with new heading
            segments.push_back({path[i], cur_heading});
            cur_heading = h;
            segment_end = i + 1;
        } else {
            segment_end = i + 1;
        }
    }
    // Close the final segment
    segments.push_back({path[segment_end], cur_heading});
    return segments;
}

// Check if robot is in collision with any wall or any movable object,
// optionally skipping one body (the target the robot is about to push).
// Iterates all robot collision bodies (chassis + wheels for diff-drive) so
// wheel-clipping a wall is treated as a collision.
bool check_robot_collision_any(NAMOEnvironment& env, std::string& out_obj,
                                const std::string& skip_body = "") {
    const auto robot_bodies = env.get_robot_adapter()->get_collision_body_names();
    const auto& statics = env.get_static_objects();
    for (size_t i = 0; i < env.get_num_static(); i++) {
        for (const auto& rb : robot_bodies) {
            if (env.bodies_in_collision(rb, statics[i].body_name)) {
                out_obj = statics[i].body_name;
                return true;
            }
        }
    }
    const auto& movables = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const std::string& bn = movables[i].body_name;
        if (!skip_body.empty() && bn == skip_body) continue;
        for (const auto& rb : robot_bodies) {
            if (env.bodies_in_collision(rb, bn)) {
                out_obj = bn;
                if (std::getenv("NAMO_NAV_LOG")) {
                    std::cerr << "[NAV_DEBUG_COL] robot=" << rb
                              << " vs " << bn << " (skip=" << skip_body << ")" << std::endl;
                }
                return true;
            }
        }
    }
    return false;
}

// Dump full qpos to file named by NAMO_QPOS_DUMP. Thin wrapper around the
// shared helper so push_controller can write into the same stream.
inline void maybe_dump_qpos(NAMOEnvironment& env, int phase_id) {
    dump_qpos(env, phase_id);
}

// Step MuJoCo for one "control tick" (matches NAMOEnvironment::apply_control timing: 0.01s).
void step_control_tick(NAMOEnvironment& env) {
    auto* sim = env.get_mujoco_wrapper();
    double dt = 0.01;
    double timestep = sim->model()->opt.timestep;
    int n = std::max(1, static_cast<int>(dt / timestep));
    for (int i = 0; i < n; i++) sim->step();
    env.update_object_states();
}

// Get current robot (x, y, theta) via adapter.
void get_robot_pose(const NAMOEnvironment& env, double& x, double& y, double& theta) {
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    auto xy = env.get_robot_adapter()->get_xy(m, d);
    x = xy[0];
    y = xy[1];
    theta = env.get_robot_adapter()->get_theta(m, d);
}

// Smooth deceleration: ramp commanded wheel velocities to zero by
// scaling them by a factor that decreases from 1 to 0 over n_steps.
// Decomposing into (forward, rotation) and ramping each is unnecessary —
// scaling preserves the sign of every wheel command, so neither wheel ever
// reverses direction (which would cause the car to back up).
// Eliminates the jerk at phase exit while staying physically consistent.
bool ramp_decel(NAMOEnvironment& env, double final_omega_left, double final_omega_right,
                double /*unused_end_left*/, double /*unused_end_right*/, int n_steps,
                std::string& out_obj, const std::string& skip_body, int phase_id) {
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    auto adapter = env.get_robot_adapter();
    for (int i = 0; i < n_steps; i++) {
        double scale = 1.0 - (i + 1) / static_cast<double>(n_steps);  // 1 → 0
        double left  = final_omega_left  * scale;
        double right = final_omega_right * scale;
        adapter->apply_wheel_control(m, d, left, right);
        step_control_tick(env);
        maybe_dump_qpos(env, phase_id);
        if (check_robot_collision_any(env, out_obj, skip_body)) return false;
    }
    return true;
}

// Passive wait: zero control + step simulation. Lets wheel brakes and
// caster momentum dissipate naturally before the next phase begins.
// Samples qpos every step so the wait period is visible in the video.
static bool wait_after_phase(NAMOEnvironment& env, int n_steps,
                              std::string& out_obj,
                              const std::string& skip_body, int phase_id) {
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    env.get_robot_adapter()->zero_control(m, d);
    for (int i = 0; i < n_steps; i++) {
        step_control_tick(env);
        maybe_dump_qpos(env, phase_id);
        if (check_robot_collision_any(env, out_obj, skip_body)) return false;
    }
    return true;
}

// Drive (P controller variant).
static bool drive_p(
    NAMOEnvironment& env,
    const std::array<double, 2>& endpoint,
    double segment_heading,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body)
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();
    const double cos_h = std::cos(segment_heading);
    const double sin_h = std::sin(segment_heading);

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        const double dx = endpoint[0] - rx;
        const double dy = endpoint[1] - ry;
        const double along = dx * cos_h + dy * sin_h;
        const double speed = adapter->get_speed(m, d);

        if (std::abs(along) < p.xy_converged && speed < p.speed_converged) break;

        double v_cmd = p.Kp_drive * along;
        if (v_cmd < 0.0) v_cmd = 0.0;
        if (v_cmd > p.linear_speed) v_cmd = p.linear_speed;

        const double wheel_omega = v_cmd / r;
        adapter->apply_wheel_control(m, d, wheel_omega, wheel_omega);
        step_control_tick(env);
        phase_steps++;

        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, 1.0});
        maybe_dump_qpos(env, 1);

        if (check_robot_collision_any(env, result.collision_object, skip_body)) {
            result.failure_reason = "collision while driving straight";
            steps_used_total += phase_steps;
            return false;
        }
    }

    steps_used_total += phase_steps;
    if (phase_steps >= max_phase_steps) {
        result.failure_reason = "drive-straight timeout";
        return false;
    }
    return true;
}


// Drive (trapezoidal): closed-loop sqrt-distance velocity law.
//   v_cmd = clamp( sqrt(2 · a_max · max(0, along)), 0, v_max )
// Symmetric to rotate_trapezoidal. Re-evaluates every tick on actual
// along-distance, so slip and momentum are absorbed automatically.
static bool drive_trapezoidal(
    NAMOEnvironment& env,
    const std::array<double, 2>& endpoint,
    double segment_heading,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body)
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();
    const double cos_h = std::cos(segment_heading);
    const double sin_h = std::sin(segment_heading);
    const double v_max = p.linear_speed;
    const double a_max = p.accel_max;

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;

    const double b = adapter->get_wheelbase();

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        const double dx = endpoint[0] - rx;
        const double dy = endpoint[1] - ry;
        const double along = dx * cos_h + dy * sin_h;
        const double speed = adapter->get_speed(m, d);

        // Position-only exit. A brief controlled-stop after this loop
        // brings chassis speed to zero before handoff to next phase.
        if (along < p.xy_converged) break;
        (void)speed;

        // Square-root profile; clamp to forward-only.
        double v_des = (along > 0.0) ? std::sqrt(2.0 * a_max * along) : 0.0;
        if (v_des > v_max) v_des = v_max;

        // Heading correction: small differential to keep chassis on-axis.
        // P controller on heading error; corrects for drift accumulated
        // during rotation handoff or drive disturbances.
        const double heading_err = wrap_angle(segment_heading - rtheta);
        const double K_heading = 2.0;
        double omega_corr = K_heading * heading_err;
        const double omega_corr_max = 0.5 * p.angular_speed;
        if (omega_corr >  omega_corr_max) omega_corr =  omega_corr_max;
        if (omega_corr < -omega_corr_max) omega_corr = -omega_corr_max;

        // Diff-drive: blend forward velocity with heading correction.
        // v_left = (v_des - omega_corr·b/2) / r
        // v_right = (v_des + omega_corr·b/2) / r
        const double wheel_left  = (v_des - omega_corr * b / 2.0) / r;
        const double wheel_right = (v_des + omega_corr * b / 2.0) / r;
        adapter->apply_wheel_control(m, d, wheel_left, wheel_right);
        step_control_tick(env);
        phase_steps++;

        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, 1.0});
        maybe_dump_qpos(env, 1);

        if (check_robot_collision_any(env, result.collision_object, skip_body)) {
            result.failure_reason = "collision while driving straight";
            steps_used_total += phase_steps;
            return false;
        }
    }

    // Controlled stop: actively damp residual forward velocity before
    // handoff so the next phase's rotation doesn't fight residual v.
    const int max_settle = 60;
    const double K_brake_v = 3.0;
    for (int i = 0; i < max_settle; i++) {
        const double sp = adapter->get_speed(m, d);
        if (sp < p.speed_converged) break;
        // Drive both wheels backward proportional to current speed.
        // Direction we were going was +heading; brake by commanding −sp · K.
        double v_cmd = -K_brake_v * sp;
        if (v_cmd > 0.0) v_cmd = 0.0;  // forward-only segment; clamp brake
        // For low-friction plant, just commanding zero is the best we can do.
        const double wL = 0.0;
        const double wR = 0.0;
        adapter->apply_wheel_control(m, d, wL, wR);
        step_control_tick(env);
        phase_steps++;
        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, 1.0});
        maybe_dump_qpos(env, 1);
        (void)v_cmd;
    }

    steps_used_total += phase_steps;
    if (phase_steps >= max_phase_steps) {
        result.failure_reason = "drive-straight timeout";
        return false;
    }
    return true;
}


// Dispatch
static bool drive_straight_to(
    NAMOEnvironment& env,
    const std::array<double, 2>& endpoint,
    double segment_heading,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body = "")
{
    if (p.mode == DiffDriveNavigation::Mode::TRAPEZOIDAL) {
        return drive_trapezoidal(env, endpoint, segment_heading, p,
                                 steps_used_total, result, skip_body);
    }
    return drive_p(env, endpoint, segment_heading, p,
                   steps_used_total, result, skip_body);
}

// Settle: zero control, step physics until velocities approach zero.
// Dumps qpos every step for smooth video. No qvel snap — physics does the braking.
bool settle(NAMOEnvironment& env, int settle_steps, std::string& out_obj,
            const std::string& skip_body = "", int phase_id = 0) {
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    env.get_robot_adapter()->zero_control(m, d);

    for (int i = 0; i < settle_steps; i++) {
        step_control_tick(env);
        maybe_dump_qpos(env, phase_id);
        if (check_robot_collision_any(env, out_obj, skip_body)) return false;
    }
    return true;
}

} // namespace


DiffDriveNavigation::DiffDriveNavigation(const Params& params)
    : params_(params) {}

DiffDriveNavigation::DiffDriveNavigation(std::shared_ptr<ConfigManager> config) {
    // Params remain at defaults; a richer config structure can be added later.
    // For now DiffDriveNavigation uses the built-in Params defaults.
    (void)config;
}


// -----------------------------------------------------------------------------
// Trapezoidal velocity profile for in-place rotation, closed-loop variant.
//
// At each tick:
//   ω_cmd = clamp(sgn(err) · sqrt(2 · α_max · |err|), -ω_max, +ω_max)
//
// Why this shape: starting from 0 and accelerating at α_max, the velocity
// after rotating angle θ is v(θ) = sqrt(2·α_max·θ). The same curve in
// reverse is the deceleration profile that brings v→0 exactly at err=0
// while never exceeding |α_max| of deceleration. So commanding this ω at
// each tick traces a feasible velocity profile that's saturation-respectful
// on both ends, and self-corrects for slip (re-evaluated every tick from
// actual remaining error).
//
// This is sometimes called a "constant-deceleration profile" or
// "square-root-of-distance velocity law" in industrial servo control.
// Equivalent to an open-loop trapezoid in the absence of disturbance, but
// closed-loop on remaining-angle so it absorbs slip.
// -----------------------------------------------------------------------------
static bool rotate_trapezoidal(
    NAMOEnvironment& env,
    double target_theta,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body,
    int phase_id)
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();
    const double b = adapter->get_wheelbase();
    const double w_max = p.angular_speed;
    const double a_max = p.alpha_max;

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        const double err = wrap_angle(target_theta - rtheta);
        const double yaw_rate = adapter->get_yaw_rate(m, d);

        // Position-only exit (rate gate causes limit cycle). After exit,
        // a brief controlled-stop loop below brings chassis to actual rest
        // before handoff, so the next phase doesn't inherit yaw momentum.
        if (std::abs(err) < p.theta_converged) break;

        // Square-root velocity profile: gives a controlled, feasible decel.
        const double sgn = (err >= 0.0) ? 1.0 : -1.0;
        double w_des = std::sqrt(2.0 * a_max * std::abs(err));
        if (w_des > w_max) w_des = w_max;
        const double omega_cmd = sgn * w_des;
        (void)yaw_rate;

        const double wheel_omega_left  = (-omega_cmd * b / 2.0) / r;
        const double wheel_omega_right = (+omega_cmd * b / 2.0) / r;
        adapter->apply_wheel_control(m, d, wheel_omega_left, wheel_omega_right);

        step_control_tick(env);
        phase_steps++;

        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, (double)phase_id});
        maybe_dump_qpos(env, phase_id);

        if (check_robot_collision_any(env, result.collision_object, skip_body)) {
            result.failure_reason = "collision during rotation";
            steps_used_total += phase_steps;
            return false;
        }
    }

    steps_used_total += phase_steps;
    if (phase_steps >= max_phase_steps) {
        result.failure_reason = "rotation timeout";
        return false;
    }
    return true;
}


// -----------------------------------------------------------------------------
// Rotate-in-place: outer-loop PD on chassis yaw → commanded chassis ω →
// diff-drive wheel kinematics → inner velocity-mode actuator.
//
//   ω_cmd  = clamp(Kp · err - Kd · yaw_rate, ±angular_speed)
//   v_left = -ω_cmd · b/2 / r,   v_right = +ω_cmd · b/2 / r
//
// The reference shrinks continuously as err→0, so the wheels are never
// step-commanded to zero. No brake transient → no friction reversal → no
// rebound. Exits when both error and yaw rate are small (system at rest at
// target). No post-phase wait — controller brings the system to rest.
// -----------------------------------------------------------------------------
static bool rotate_pd(
    NAMOEnvironment& env,
    double target_theta,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body,
    int phase_id)
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();
    const double b = adapter->get_wheelbase();

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        const double yaw_rate = adapter->get_yaw_rate(m, d);
        const double err = wrap_angle(target_theta - rtheta);

        if (std::abs(err) < p.theta_converged && std::abs(yaw_rate) < p.rate_converged) {
            break;
        }

        double omega_cmd = p.Kp_yaw * err - p.Kd_yaw * yaw_rate;
        if (omega_cmd >  p.angular_speed) omega_cmd =  p.angular_speed;
        if (omega_cmd < -p.angular_speed) omega_cmd = -p.angular_speed;

        const double wheel_omega_left  = (-omega_cmd * b / 2.0) / r;
        const double wheel_omega_right = (+omega_cmd * b / 2.0) / r;
        adapter->apply_wheel_control(m, d, wheel_omega_left, wheel_omega_right);

        step_control_tick(env);
        phase_steps++;

        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, (double)phase_id});
        maybe_dump_qpos(env, phase_id);

        if (check_robot_collision_any(env, result.collision_object, skip_body)) {
            result.failure_reason = "collision during rotation";
            steps_used_total += phase_steps;
            return false;
        }
    }

    steps_used_total += phase_steps;
    if (phase_steps >= max_phase_steps) {
        result.failure_reason = "rotation timeout";
        return false;
    }
    return true;
}


// Dispatch: pick rotate implementation by Mode.
static bool rotate_in_place(
    NAMOEnvironment& env,
    double target_theta,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body = "",
    int phase_id = 0)
{
    if (p.mode == DiffDriveNavigation::Mode::TRAPEZOIDAL) {
        return rotate_trapezoidal(env, target_theta, p, steps_used_total,
                                  result, skip_body, phase_id);
    }
    return rotate_pd(env, target_theta, p, steps_used_total,
                     result, skip_body, phase_id);
}


// -----------------------------------------------------------------------------
// Top-level execute: three phases
// -----------------------------------------------------------------------------
NavigationResult DiffDriveNavigation::execute(
    NAMOEnvironment& env,
    const std::vector<std::array<double, 2>>& path,
    double target_theta,
    const std::string& target_object) {

    NavigationResult result;

    // The body-name of the target movable (may be empty if not provided).
    // Only Phase 3 allows contact with this body — phases 1 and 2 avoid all.
    std::string target_body;
    if (!target_object.empty()) {
        auto* info = env.get_object_info(target_object);
        if (info) target_body = info->body_name;
    }
    if (std::getenv("NAMO_NAV_LOG")) {
        std::cerr << "[NAV_DEBUG] target_object=" << target_object
                  << " target_body=" << target_body << std::endl;
    }

    // If we're already at the goal pose, skip navigation.
    // Avoids re-navigating when the push controller's MPC loop calls us
    // again after the car is already placed at the edge point.
    if (!path.empty()) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        const auto& goal = path.back();
        double dxy = std::hypot(goal[0] - rx, goal[1] - ry);
        double dth = std::abs(wrap_angle(target_theta - rtheta));
        if (dxy < params_.xy_tolerance && dth < params_.theta_tolerance) {
            if (std::getenv("NAMO_NAV_LOG")) {
                std::cerr << "[NAV_DEBUG] already at goal; skipping nav" << std::endl;
            }
            result.success = true;
            return result;
        }
    }

    if (path.size() < 2) {
        int steps = 0;
        if (!rotate_in_place(env, target_theta, params_, steps, result, target_body)) {
            result.steps_used = steps;
            return result;
        }
        result.success = true;
        result.steps_used = steps;
        return result;
    }

    int steps = 0;

    // Sample the starting pose so video begins at the actual start
    {
        double tx, ty, tt;
        get_robot_pose(env, tx, ty, tt);
        result.trajectory.push_back({tx, ty, tt, 0.0});
        maybe_dump_qpos(env, 0);
    }

    // ── State machine over path segments ─────────────────────────────────
    // Wavefront paths are 8-connected → headings are 0/45/90/135° etc.
    // segment_path() merges runs of waypoints with consistent heading; each
    // resulting segment is a STRAIGHT corridor. The car follows by
    // alternating "rotate to segment heading" and "drive straight to
    // segment endpoint" — no curvature commands → no oscillation.
    auto segments = segment_path(path, params_.sharp_turn_threshold);
    if (std::getenv("NAMO_NAV_LOG")) {
        std::cerr << "[NAV_DEBUG] " << segments.size() << " segments" << std::endl;
        for (size_t i = 0; i < segments.size(); i++) {
            std::cerr << "    seg " << i << ": end=("
                      << segments[i].end_point[0] << "," << segments[i].end_point[1]
                      << ") heading=" << segments[i].heading << " rad" << std::endl;
        }
    }

    for (size_t i = 0; i < segments.size(); i++) {
        // Rotate to align with segment heading
        if (!rotate_in_place(env, segments[i].heading, params_, steps, result, "", 0)) {
            result.steps_used = steps;
            return result;
        }
        // Drive straight to segment endpoint
        if (!drive_straight_to(env, segments[i].end_point, segments[i].heading,
                                params_, steps, result, "")) {
            result.steps_used = steps;
            return result;
        }
    }

    // Final rotation to push heading (target may touch — skip target collision)
    if (!rotate_in_place(env, target_theta, params_, steps, result, target_body, 2)) {
        result.steps_used = steps;
        return result;
    }

    result.success = true;
    result.steps_used = steps;
    return result;
}

} // namespace namo
