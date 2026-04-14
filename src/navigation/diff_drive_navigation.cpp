#include "navigation/diff_drive_navigation.hpp"
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
bool check_robot_collision_any(NAMOEnvironment& env, std::string& out_obj,
                                const std::string& skip_body = "") {
    const std::string robot_body = env.get_robot_adapter()->get_body_name();
    const auto& statics = env.get_static_objects();
    for (size_t i = 0; i < env.get_num_static(); i++) {
        if (env.bodies_in_collision(robot_body, statics[i].body_name)) {
            out_obj = statics[i].body_name;
            return true;
        }
    }
    const auto& movables = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const std::string& bn = movables[i].body_name;
        if (!skip_body.empty() && bn == skip_body) continue;
        if (env.bodies_in_collision(robot_body, bn)) {
            out_obj = bn;
            if (std::getenv("NAMO_NAV_LOG")) {
                std::cerr << "[NAV_DEBUG_COL] robot=" << robot_body
                          << " vs " << bn << " (skip=" << skip_body << ")" << std::endl;
            }
            return true;
        }
    }
    return false;
}

// Dump full qpos to a file for later video rendering. Triggered by env var.
void maybe_dump_qpos(NAMOEnvironment& env, int phase_id) {
    static FILE* dump_fp = nullptr;
    static bool init = false;
    if (!init) {
        const char* path = std::getenv("NAMO_QPOS_DUMP");
        if (path && path[0]) {
            dump_fp = std::fopen(path, "w");
        }
        init = true;
    }
    if (!dump_fp) return;

    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    std::fprintf(dump_fp, "%d %d", phase_id, m->nq);
    for (int i = 0; i < m->nq; i++) std::fprintf(dump_fp, " %.6f", d->qpos[i]);
    std::fprintf(dump_fp, "\n");
    std::fflush(dump_fp);
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

// Drive forward in a straight line (equal wheel velocities) until either:
//   - the robot is within `xy_threshold` of the endpoint, or
//   - the robot has passed the endpoint (the dot product of motion-to-go
//     onto the segment heading goes negative)
// Constant linear speed, no curvature commands. Designed for short straight
// segments produced by segment_path.
static bool drive_straight_to(
    NAMOEnvironment& env,
    const std::array<double, 2>& endpoint,
    double segment_heading,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body = "")
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();

    // Wheel angular velocity for forward driving at linear_speed
    const double wheel_omega = p.linear_speed / r;

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;
    const double cos_h = std::cos(segment_heading);
    const double sin_h = std::sin(segment_heading);

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);

        double dx = endpoint[0] - rx;
        double dy = endpoint[1] - ry;
        double dist = std::hypot(dx, dy);

        // Distance threshold OR we've overshot (projection on heading negative)
        double along = dx * cos_h + dy * sin_h;
        if (dist < p.xy_threshold || along < 0.0) break;

        // Drive both wheels forward at equal velocity (straight-line)
        adapter->apply_wheel_control(m, d, wheel_omega, wheel_omega);
        step_control_tick(env);
        phase_steps++;

        // Sample
        {
            double tx, ty, tt;
            get_robot_pose(env, tx, ty, tt);
            result.trajectory.push_back({tx, ty, tt, 1.0});
            maybe_dump_qpos(env, 1);
        }

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

    // Wait briefly to dissipate momentum (zero control, no decel ramp).
    if (!wait_after_phase(env, p.wait_steps, result.collision_object, skip_body, 1)) {
        result.failure_reason = "collision during drive wait";
        return false;
    }
    steps_used_total += p.wait_steps;
    return true;
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
// Phase 1 / Phase 3: in-place rotation to a target heading.
// Constant angular speed; sign chosen to minimize angular error.
// Exits when |error| < theta_threshold, then settles.
// -----------------------------------------------------------------------------
static bool rotate_in_place(
    NAMOEnvironment& env,
    double target_theta,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body = "",
    int phase_id = 0)
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();

    // Differential wheel speed for pure rotation (v=0, ω=angular_speed).
    // v_left = -ω·b/2, v_right = +ω·b/2. Convert to wheel ω: divide by r.
    const double b = adapter->get_wheelbase();

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;
    double last_wheel_left = 0.0, last_wheel_right = 0.0;

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);
        double err = wrap_angle(target_theta - rtheta);

        // Exit condition
        if (std::abs(err) < p.theta_threshold) break;

        // Pick direction. Turn the shorter way.
        double omega = p.angular_speed * (err > 0 ? 1.0 : -1.0);

        // Diff-drive wheel velocities for pure rotation:
        // v_left = -ω·b/2, v_right = +ω·b/2, then / r for wheel angular velocity
        double wheel_omega_left  = (-omega * b / 2.0) / r;
        double wheel_omega_right = (+omega * b / 2.0) / r;
        last_wheel_left = wheel_omega_left;
        last_wheel_right = wheel_omega_right;

        adapter->apply_wheel_control(m, d, wheel_omega_left, wheel_omega_right);
        step_control_tick(env);
        phase_steps++;

        // Sample trajectory every control step for smooth video
        {
            double tx, ty, tt;
            get_robot_pose(env, tx, ty, tt);
            result.trajectory.push_back({tx, ty, tt, (double)phase_id});
            maybe_dump_qpos(env, phase_id);
        }

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

    // Wait briefly so chassis rotation + caster momentum dissipate.
    if (!wait_after_phase(env, p.wait_steps, result.collision_object, skip_body, phase_id)) {
        result.failure_reason = "collision during rotation wait";
        return false;
    }
    steps_used_total += p.wait_steps;

    // Verify final heading within final tolerance
    double rx, ry, rtheta;
    get_robot_pose(env, rx, ry, rtheta);
    double err = wrap_angle(target_theta - rtheta);
    if (std::abs(err) > p.theta_tolerance) {
        result.failure_reason = "rotation did not settle at target heading (err=" +
            std::to_string(err) + " rad, tol=" + std::to_string(p.theta_tolerance) + ")";
        return false;
    }
    return true;
}


// -----------------------------------------------------------------------------
// Phase 2: pure pursuit along a 2D path.
// Standard pure pursuit formulation with constant linear speed.
// -----------------------------------------------------------------------------
// Returns (lookahead_x, lookahead_y) and sets reached_end=true when the
// lookahead would be beyond the final waypoint.
static std::pair<double, double> find_lookahead(
    const std::vector<std::array<double, 2>>& path,
    double rx, double ry,
    double lookahead,
    size_t& last_idx,
    bool& reached_end)
{
    reached_end = false;

    // Find the closest path segment to the robot, starting from last_idx to
    // avoid scanning the whole path every step (Nav2 pattern).
    size_t closest = last_idx;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (size_t i = last_idx; i < path.size(); i++) {
        double dx = path[i][0] - rx;
        double dy = path[i][1] - ry;
        double d2 = dx*dx + dy*dy;
        if (d2 < best_dist2) {
            best_dist2 = d2;
            closest = i;
        }
    }
    last_idx = closest;

    // Advance from the closest point along the path until we exceed lookahead.
    double remaining = lookahead;
    double prev_x = path[closest][0];
    double prev_y = path[closest][1];
    for (size_t i = closest + 1; i < path.size(); i++) {
        double dx = path[i][0] - prev_x;
        double dy = path[i][1] - prev_y;
        double seg_len = std::hypot(dx, dy);
        if (seg_len >= remaining) {
            double t = remaining / seg_len;
            return {prev_x + t * dx, prev_y + t * dy};
        }
        remaining -= seg_len;
        prev_x = path[i][0];
        prev_y = path[i][1];
    }
    // Ran off the end of the path
    reached_end = true;
    return {path.back()[0], path.back()[1]};
}

static bool pure_pursuit_along(
    NAMOEnvironment& env,
    const std::vector<std::array<double, 2>>& path,
    const DiffDriveNavigation::Params& p,
    int& steps_used_total,
    NavigationResult& result,
    const std::string& skip_body = "")
{
    const auto* adapter = env.get_robot_adapter();
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    const double r = adapter->get_wheel_radius();
    const double b = adapter->get_wheelbase();
    const double v = p.linear_speed;

    const auto& goal_xy = path.back();
    size_t last_idx = 0;

    int phase_steps = 0;
    const int max_phase_steps = p.max_nav_steps - steps_used_total;
    double last_wheel_left = 0.0, last_wheel_right = 0.0;

    while (phase_steps < max_phase_steps) {
        double rx, ry, rtheta;
        get_robot_pose(env, rx, ry, rtheta);

        // Exit if close to goal
        double dx_goal = goal_xy[0] - rx;
        double dy_goal = goal_xy[1] - ry;
        double dist_goal = std::hypot(dx_goal, dy_goal);
        if (dist_goal < p.xy_threshold) break;

        // Abort if drifting too far from path (off-track safeguard)
        double dx_closest = path[last_idx][0] - rx;
        double dy_closest = path[last_idx][1] - ry;
        if (std::hypot(dx_closest, dy_closest) > p.max_path_deviation) {
            result.failure_reason = "drifted off planned path";
            steps_used_total += phase_steps;
            return false;
        }

        // Lookahead point
        bool reached_end = false;
        auto [la_x, la_y] = find_lookahead(path, rx, ry, p.lookahead,
                                            last_idx, reached_end);

        // Exit pure pursuit when both:
        //   (a) the path is exhausted (reached_end == true), AND
        //   (b) the car is actually within `lookahead` of the goal.
        // Beyond this the algorithm oscillates around a fixed lookahead
        // point. Accept the residual gap and rely on post-settle tolerance.
        if (reached_end && dist_goal <= p.lookahead) break;

        // Angle to lookahead in robot frame
        double dx = la_x - rx;
        double dy = la_y - ry;
        double L_actual = std::max(1e-6, std::hypot(dx, dy));
        double alpha = wrap_angle(std::atan2(dy, dx) - rtheta);

        // Sharp turn recovery: if heading error is too large, stop forward
        // motion and rotate in place until we're within a smaller band.
        // Avoids ill-conditioned pure-pursuit when alpha is near ±π/2.
        if (std::abs(alpha) > p.sharp_turn_threshold) {
            // Pick rotation direction
            double rot_omega = p.angular_speed * (alpha > 0 ? 1.0 : -1.0);
            double vl_rot = (-rot_omega * b / 2.0) / r;
            double vr_rot = (+rot_omega * b / 2.0) / r;
            adapter->apply_wheel_control(m, d, vl_rot, vr_rot);
            step_control_tick(env);
            phase_steps++;
            {
                double tx, ty, tt;
                get_robot_pose(env, tx, ty, tt);
                result.trajectory.push_back({tx, ty, tt, 1.0});
                maybe_dump_qpos(env, 1);
            }
            if (check_robot_collision_any(env, result.collision_object, skip_body)) {
                result.failure_reason = "collision during sharp-turn rotation";
                steps_used_total += phase_steps;
                return false;
            }
            continue;  // re-evaluate alpha next iteration; drive resumes when |alpha| < sharp_turn_exit
        }

        // Pure pursuit curvature
        double kappa = 2.0 * std::sin(alpha) / L_actual;

        // Linear + angular velocity
        double omega = kappa * v;
        // v_left = v - ω·b/2, v_right = v + ω·b/2
        double v_left  = v - omega * b / 2.0;
        double v_right = v + omega * b / 2.0;
        last_wheel_left = v_left / r;
        last_wheel_right = v_right / r;
        // Convert to wheel angular velocity
        adapter->apply_wheel_control(m, d, last_wheel_left, last_wheel_right);

        step_control_tick(env);
        phase_steps++;

        // Sample trajectory every control step for smooth video
        {
            double tx, ty, tt;
            get_robot_pose(env, tx, ty, tt);
            result.trajectory.push_back({tx, ty, tt, 1.0});
            maybe_dump_qpos(env, 1);
        }

        if (check_robot_collision_any(env, result.collision_object, skip_body)) {
            result.failure_reason = "collision during pure pursuit";
            steps_used_total += phase_steps;
            return false;
        }
    }

    steps_used_total += phase_steps;
    if (phase_steps >= max_phase_steps) {
        result.failure_reason = "pure pursuit timeout";
        return false;
    }

    // Wait briefly for momentum to settle before completing.
    if (!wait_after_phase(env, p.wait_steps, result.collision_object, skip_body, 1)) {
        result.failure_reason = "collision during pursuit wait";
        return false;
    }
    steps_used_total += p.wait_steps;

    // Verify final position within final tolerance
    double rx, ry, rtheta;
    get_robot_pose(env, rx, ry, rtheta);
    double dx = goal_xy[0] - rx;
    double dy = goal_xy[1] - ry;
    if (std::hypot(dx, dy) > p.xy_tolerance) {
        result.failure_reason = "pure pursuit did not settle at goal";
        return false;
    }
    return true;
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
