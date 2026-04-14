#include "navigation/diff_drive_navigation.hpp"
#include "environment/namo_environment.hpp"
#include "robot/robot_adapter.hpp"
#include "config/config_manager.hpp"
#include <cmath>

namespace namo {

namespace {

// Wrap angle to [-π, π]
double wrap_angle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
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
        if (!skip_body.empty() && movables[i].body_name == skip_body) continue;
        if (env.bodies_in_collision(robot_body, movables[i].body_name)) {
            out_obj = movables[i].body_name;
            return true;
        }
    }
    return false;
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

// Settle: zero control + zero velocities (bypasses simulation momentum).
bool settle(NAMOEnvironment& env, int settle_steps, std::string& out_obj,
            const std::string& skip_body = "") {
    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    env.get_robot_adapter()->zero_control(m, d);

    for (int i = 0; i < m->nv; i++) d->qvel[i] = 0.0;
    mj_forward(m, d);
    env.update_object_states();

    for (int i = 0; i < settle_steps; i++) {
        step_control_tick(env);
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
    const std::string& skip_body = "")
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

        adapter->apply_wheel_control(m, d, wheel_omega_left, wheel_omega_right);
        step_control_tick(env);
        phase_steps++;

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

    // Settle
    if (!settle(env, p.settle_steps, result.collision_object, skip_body)) {
        result.failure_reason = "collision during rotation settle";
        return false;
    }
    steps_used_total += p.settle_steps;

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

        // Angle to lookahead in robot frame
        double dx = la_x - rx;
        double dy = la_y - ry;
        double L_actual = std::max(1e-6, std::hypot(dx, dy));
        double alpha = wrap_angle(std::atan2(dy, dx) - rtheta);

        // Pure pursuit curvature
        double kappa = 2.0 * std::sin(alpha) / L_actual;

        // Linear + angular velocity
        double omega = kappa * v;
        // v_left = v - ω·b/2, v_right = v + ω·b/2
        double v_left  = v - omega * b / 2.0;
        double v_right = v + omega * b / 2.0;
        // Convert to wheel angular velocity
        adapter->apply_wheel_control(m, d, v_left / r, v_right / r);

        step_control_tick(env);
        phase_steps++;

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

    // Settle
    if (!settle(env, p.settle_steps, result.collision_object, skip_body)) {
        result.failure_reason = "collision during pursuit settle";
        return false;
    }
    steps_used_total += p.settle_steps;

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

    // --- Phase 1: rotate to face first waypoint beyond start ---
    // Avoid ALL collisions during this phase.
    double rx, ry, rtheta;
    get_robot_pose(env, rx, ry, rtheta);
    double heading_to_path = std::atan2(
        path[1][1] - ry,
        path[1][0] - rx
    );
    if (!rotate_in_place(env, heading_to_path, params_, steps, result, "")) {
        result.steps_used = steps;
        return result;
    }

    // --- Phase 2: pure pursuit ---
    // Avoid ALL collisions; path should keep us clear.
    if (!pure_pursuit_along(env, path, params_, steps, result, "")) {
        result.steps_used = steps;
        return result;
    }

    // --- Phase 3: rotate to push heading ---
    // We're at the edge point — contact with the target is expected; skip it.
    if (!rotate_in_place(env, target_theta, params_, steps, result, target_body)) {
        result.steps_used = steps;
        return result;
    }

    result.success = true;
    result.steps_used = steps;
    return result;
}

} // namespace namo
