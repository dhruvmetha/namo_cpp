#include "planning/namo_push_controller.hpp"
#include "core/mujoco_wrapper.hpp"
#include "navigation/qpos_dump.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

constexpr double kPushLookaheadRatio = 1.0;   // Was 0.3 — short lookahead amplifies heading
                                                // noise into PP curvature corrections; 1.0
                                                // matches robot_control's PushController/Nav
                                                // and the Tier-1 PP calibration so sim PP has
                                                // the same noise sensitivity as real PP.
constexpr double kPushGoalToleranceRatio = 0.3;
constexpr double kPushPathExtendDistanceM = 0.50;

// Calibration constant: the chassis m/s the diff-drive wheels are commanded
// to produce when the follower outputs a fraction of 1.0. Mirrors the role
// of the real MicroMVP firmware's max-speed-at-PWM-1.0 mapping. With this at
// 1.0, `push_tracker_max_speed` (the follower's fraction cap, set via the
// skill config) reads directly as chassis m/s during a smooth straight push.
// Recalibrate against real-robot measurements for sim-real parity.
constexpr double kCarWheelMaxSpeedMs = 1.0;

// A push that stops on stuck detection keeps its result once the object has
// travelled at least this far from where the push started; below it the push is
// treated as never having happened and the caller restores the pre-push state.
// 1 cm is the same bar test_primitive_generation_integration.py uses for "the
// push ran", so C++ and the tests agree on what counts as motion. It also sits
// well above the stuck detector's own per-check threshold (min_position_change_,
// 1 mm over stuck_check_stride_ ticks), so settling jitter cannot reach it.
constexpr double kMinUsefulPushDisplacementM = 0.01;

}  // namespace

namespace namo {

NAMOPushController::NAMOPushController(NAMOEnvironment& env,
                                     WavefrontPlanner& planner,
                                     int push_steps,
                                     int control_steps,
                                     double scaling,
                                     int points_per_edge,
                                     bool dynamic_direction)
    : env_(env), planner_(planner),
      default_push_steps_(push_steps),
      control_steps_per_push_(control_steps),
      push_velocity_(scaling),
      points_per_edge_(points_per_edge),
      dynamic_direction_(dynamic_direction) {
    
    // Initialize robot size from wavefront planner (matches inflation)
    const auto& planner_robot_size = planner_.get_robot_size();
    robot_size_[0] = planner_robot_size.size() > 0 ? planner_robot_size[0] : 0.15;
    robot_size_[1] = planner_robot_size.size() > 1 ? planner_robot_size[1] : 0.15;
    robot_size_[2] = 0.0;  // z not used for 2D planning
    
    // Navigation is teleport-only in sim (point robot and car alike): we
    // snap the robot to the final waypoint of the wavefront path and let
    // physics settle. Kinematic diff-drive navigation lived in a separate
    // strategy and has been removed.
    auto adapter = env_.get_robot_adapter();

    if (adapter && adapter->is_diff_drive()) {
        const double full_width = 2.0 * std::abs(robot_size_[0]);
        const double full_height = 2.0 * std::abs(robot_size_[1]);
        const double car_size = std::max(
            std::max(full_width, full_height),
            1e-6);

        PushPathFollower::Params follower_params;
        follower_params.robot_width_m = (full_width > 0.0) ? full_width : 0.07;
        follower_params.robot_height_m = (full_height > 0.0) ? full_height : 0.07;
        follower_params.wheel_base_m =
            (adapter->get_wheelbase() > 0.0) ? adapter->get_wheelbase() : 0.075;
        follower_params.lookahead_distance_m = kPushLookaheadRatio * car_size;
        follower_params.goal_tolerance_m = kPushGoalToleranceRatio * car_size;
        // max_speed left at its Params default (the single source of truth
        // for the push tracking speed; see push_path_follower.hpp).
        follower_params.max_point_gap_ratio = 0.01;
        follower_params.no_skip_ratio = 0.5;
        follower_params.wheel_deadband = 0.05;
        push_path_follower_ = std::make_unique<PushPathFollower>(follower_params);
    }

    // Pre-allocate memory pools (they're already initialized as empty)
    // std::cout << "NAMO Push Controller initialized:" << std::endl;
    // std::cout << "  Push steps: " << default_push_steps_ << std::endl;
    // std::cout << "  Control steps per push: " << control_steps_per_push_ << std::endl;
    // std::cout << "  Push velocity (m/s): " << push_velocity_ << std::endl;
    // std::cout << "  Dynamic direction: " << dynamic_direction_ << std::endl;
    // std::cout << "  Robot size: [" << robot_size_[0] << ", " << robot_size_[1] << ", " << robot_size_[2] << "]" << std::endl;
}

size_t NAMOPushController::generate_edge_points(const std::string& object_name,
                                               std::array<std::array<double, 2>, MAX_EDGE_POINTS>& edge_points,
                                               std::array<std::array<double, 2>, MAX_EDGE_POINTS>& mid_points,
                                               size_t& edge_count,
                                               size_t& mid_count) {
    
    // Reset output counts
    edge_count = 0;
    mid_count = 0;
    
    // Get object state
    auto obj_info = env_.get_object_info(object_name);
    if (!obj_info) {
        std::cerr << "Object not found: " << object_name << std::endl;
        return 0;
    }
    
    auto obj_state = env_.get_object_state(object_name);
    if (!obj_state) {
        std::cerr << "Object state not available: " << object_name << std::endl;
        return 0;
    }
    
    // Generate rectangular edge points
    generate_rectangular_edge_points(obj_state->position, obj_state->size, 
                                   obj_state->quaternion, edge_points, mid_points,
                                   edge_count, mid_count);
    
    return edge_count;
}

void NAMOPushController::generate_rectangular_edge_points(const std::array<double, 3>& obj_pos,
                                                        const std::array<double, 3>& obj_size,
                                                        const std::array<double, 4>& obj_quat,
                                                        std::array<std::array<double, 2>, MAX_EDGE_POINTS>& edge_points,
                                                        std::array<std::array<double, 2>, MAX_EDGE_POINTS>& mid_points,
                                                        size_t& edge_count,
                                                        size_t& mid_count) {

    // Convert quaternion to rotation angle (yaw)
    double yaw = utils::quaternion_to_yaw(obj_quat);

    // Object dimensions - subtract margin
    double x = 0.0, y = 0.0;
    double w = obj_size[0];  // width with margin
    double d = obj_size[1];  // depth with margin

    // Robot offset for close contact pushing (rotation-safe radius + configurable margin)
    const std::vector<double> robot_half_extents = {robot_size_[0], robot_size_[1]};
    const double robot_radius = compute_rotation_safe_robot_radius_m(robot_half_extents);
    double offset = robot_radius + push_offset_margin_;
    
    int n = points_per_edge_;
    double eps_u = std::min(0.05, 0.25 * w);  // margin from corners
    double eps_v = std::min(0.05, 0.25 * d);
    
    // Helper function for linear sampling
    auto sample_lin = [](double a, double b, int n, int i) {
        if (n <= 1) return (a + b) * 0.5;
        return a + (b - a) * (double(i) / double(n - 1));
    };
    
    std::vector<std::array<double, 2>> local_edge_points;
    local_edge_points.reserve(4 * n);
    
    // Top/Bottom pairs: sample along x-direction
    for (int j = 0; j < n; ++j) {
        double u = sample_lin(-w, w, n, j);
        local_edge_points.push_back({x + u, y + d + offset});    // Top(j)
        local_edge_points.push_back({x + u, y - d - offset});    // Bottom(j)
    }
    
    // Right/Left pairs: sample along y-direction
    for (int k = 0; k < n; ++k) {
        double v = sample_lin(-d , d , n, k);
        local_edge_points.push_back({x + w + offset, y + v});    // Right(k)
        local_edge_points.push_back({x - w - offset, y + v});    // Left(k)
    }
    
    // Capacity check
    edge_count = std::min<size_t>(local_edge_points.size(), MAX_EDGE_POINTS);
    
    // Transform edge points to world coordinates
    for (size_t i = 0; i < edge_count; ++i) {
        edge_points[i] = transform_point(local_edge_points[i], obj_pos, yaw);
    }
    
    // Calculate mid points using consecutive pairing (preserves existing logic)
    std::vector<std::array<double, 2>> local_mid_points;
    local_mid_points.reserve(edge_count);
    
    for (size_t i = 0; i < edge_count; ++i) {
        size_t mate = (i % 2 == 0) ? i + 1 : i - 1;
        std::array<double, 2> mid_local = {
            0.5 * (local_edge_points[i][0] + local_edge_points[mate][0]),
            0.5 * (local_edge_points[i][1] + local_edge_points[mate][1])
        };
        local_mid_points.push_back(mid_local);
    }
    
    // Transform mid points to world coordinates
    mid_count = edge_count;
    for (size_t i = 0; i < mid_count; ++i) {
        mid_points[i] = transform_point(local_mid_points[i], obj_pos, yaw);
    }
}

double NAMOPushController::quaternion_to_yaw(const std::array<double, 4>& quaternion) {
    // Use EXACTLY the same approach as the original PRX implementation
    // From namo_utility.hpp: quaternion_to_yaw with scalar_first = false (default)
    
    // Extract w,x,y,z based on [x,y,z,w] format (scalar_first = false)
    double w = quaternion[3];  // w is at index 3 for [x,y,z,w] format
    double x = quaternion[0];  // x is at index 0 
    double y = quaternion[1];  // y is at index 1
    double z = quaternion[2];  // z is at index 2

    // Use the exact formula from original PRX namo_utility.hpp lines 24-27
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    double yaw = std::atan2(siny_cosp, cosy_cosp);
    
    // DEBUG: Show what we got vs expected
    // std::cout << "    PRX quaternion_to_yaw: [" << x << ", " << y << ", " << z << ", " << w 
              // << "] → " << yaw << " rad (" << (yaw * 180.0 / M_PI) << "°)" << std::endl;
    
    return yaw;
}

std::array<double, 2> NAMOPushController::transform_point(const std::array<double, 2>& point,
                                                        const std::array<double, 3>& translation,
                                                        double rotation_angle) {
    
    double cos_theta = std::cos(rotation_angle);
    double sin_theta = std::sin(rotation_angle);
    
    // Rotate then translate
    std::array<double, 2> result;
    result[0] = cos_theta * point[0] - sin_theta * point[1] + translation[0];
    result[1] = sin_theta * point[0] + cos_theta * point[1] + translation[1];
    
    return result;
}

std::array<double, 2> NAMOPushController::compute_push_control(const PushState& state) {
    // Control direction: from edge point toward mid point (matching older implementation)
    double dx = state.current_mid_point[0] - state.current_edge_point[0];
    double dy = state.current_mid_point[1] - state.current_edge_point[1];
    
    // Use angle-based normalization like older implementation
    double angle = std::atan2(dy, dx);
    
    // Under MuJoCo <velocity> actuators (current XML setup), this vector
    // is interpreted as the desired robot velocity in m/s. The actuator
    // tracks it inside the physics solver up to its forcerange limit.
    return {
        push_velocity_ * std::cos(angle),
        push_velocity_ * std::sin(angle)
    };
}
void NAMOPushController::print_stuck_ctrl_diag(int, int, double, double, int) {}

bool NAMOPushController::update_stuck_counter_and_check_abort(const std::array<double, 3>& prev_pos,
                                                             const std::array<double, 4>& prev_quat,
                                                             const std::array<double, 3>& curr_pos,
                                                             const std::array<double, 4>& curr_quat,
                                                             int step,
                                                             int ctrl_step) {
    double dx = curr_pos[0] - prev_pos[0];
    double dy = curr_pos[1] - prev_pos[1];
    double dist = std::sqrt(dx * dx + dy * dy);

    double yaw_prev = std::atan2(
        2.0 * (prev_quat[0] * prev_quat[3] + prev_quat[1] * prev_quat[2]),
        1.0 - 2.0 * (prev_quat[2] * prev_quat[2] + prev_quat[3] * prev_quat[3])
    );
    double yaw_curr = std::atan2(
        2.0 * (curr_quat[0] * curr_quat[3] + curr_quat[1] * curr_quat[2]),
        1.0 - 2.0 * (curr_quat[2] * curr_quat[2] + curr_quat[3] * curr_quat[3])
    );
    double dtheta = std::abs(yaw_curr - yaw_prev);
    while (dtheta > M_PI) dtheta = 2.0 * M_PI - dtheta;

    bool is_stuck_now = (dist < min_position_change_) && (dtheta < min_angle_change_);
    if (is_stuck_now) {
        last_stuck_counter_ += 1;
        print_stuck_ctrl_diag(step, ctrl_step, dist, dtheta, last_stuck_counter_);
        if (last_stuck_counter_ >= stuck_ctrl_iterations_threshold_) {
            // debug disabled
            return true;
        }
    } else {
        if (last_stuck_counter_ > 0) {
            print_stuck_ctrl_diag(step, ctrl_step, dist, dtheta, 0);
        }
        last_stuck_counter_ = 0;
    }
    return false;
}

void NAMOPushController::update_push_state(PushState& state,
                                          const std::array<double, 3>& obj_pos,
                                          const std::array<double, 3>& obj_size,
                                          const std::array<double, 4>& obj_quat) {
    
    // Regenerate edge and mid points for current object state using SE(2) transformation
    edge_point_count_ = 0;
    mid_point_count_ = 0;
    
    generate_rectangular_edge_points(obj_pos, obj_size, obj_quat, 
                                   edge_point_pool_, mid_point_pool_,
                                   edge_point_count_, mid_point_count_);
    
    // Update current points based on edge index (matching older implementation)
    if (state.edge_idx < static_cast<int>(edge_point_count_)) {
        state.current_edge_point = edge_point_pool_[state.edge_idx];
        state.current_mid_point = mid_point_pool_[state.edge_idx];
    }
}

PushPathFollower::Pose NAMOPushController::get_current_robot_pose() const {
    PushPathFollower::Pose pose;
    const auto* adapter = env_.get_robot_adapter();
    const auto* sim = env_.get_mujoco_wrapper();
    if (!adapter || !sim) {
        return pose;
    }

    const auto xy = adapter->get_xy(sim->model(), sim->data());
    pose.x_m = xy[0];
    pose.y_m = xy[1];
    pose.theta_rad = adapter->get_theta(sim->model(), sim->data());
    return pose;
}

std::vector<std::array<double, 2>> NAMOPushController::build_push_tracking_path(
    const PushState& state,
    const std::array<double, 2>& actual_pos) const {
    const double dx = state.current_mid_point[0] - state.current_edge_point[0];
    const double dy = state.current_mid_point[1] - state.current_edge_point[1];
    const double length = std::hypot(dx, dy);

    if (length <= 1e-6) {
        return {actual_pos, state.current_mid_point, state.current_mid_point};
    }

    const double dir_x = dx / length;
    const double dir_y = dy / length;
    const std::array<double, 2> shifted_mid = {
        actual_pos[0] + dir_x * length,
        actual_pos[1] + dir_y * length,
    };
    const std::array<double, 2> extended_target = {
        shifted_mid[0] + dir_x * kPushPathExtendDistanceM,
        shifted_mid[1] + dir_y * kPushPathExtendDistanceM,
    };
    return {actual_pos, shifted_mid, extended_target};
}

void NAMOPushController::log_push_path(
    int tick,
    const std::vector<std::array<double, 2>>& push_path) const {
    if (!std::getenv("NAMO_NAV_LOG")) {
        return;
    }

    std::cerr << "[PUSH_PATH] " << tick;
    for (const auto& point : push_path) {
        std::cerr << " " << point[0] << "," << point[1];
    }
    std::cerr << std::endl;
}

void NAMOPushController::log_push_control(
    int tick,
    double omega_left,
    double omega_right,
    PushPathFollower::Mode mode) const {
    if (!std::getenv("NAMO_NAV_LOG")) {
        return;
    }

    std::cerr << "[PUSH_CTRL] " << tick
              << " " << omega_left
              << " " << omega_right
              << " " << PushPathFollower::mode_name(mode)
              << std::endl;
}

void NAMOPushController::settle_after_push(bool use_diff_drive_tracking, int settle_steps) {
    if (use_diff_drive_tracking) {
        env_.apply_wheel_control(0.0, 0.0);
    } else {
        env_.apply_robot_control(0.0, 0.0);
    }
    for (int i = 0; i < settle_steps; ++i) {
        env_.step_simulation();
        env_.get_mujoco_wrapper()->notify_physics_step();
        dump_qpos(env_, /*phase=*/3);
    }
}

bool NAMOPushController::execute_push_primitive(const std::string& object_name,
                                               int edge_idx,
                                               int push_steps) {
    // Reset controller-level stuck counter at the start of every primitive execution
    last_stuck_counter_ = 0;
    last_push_stopped_early_ = false;

    // Reset collision tracking for this push
    clear_collision_tracking();

    // Generate edge points for the object
    edge_point_count_ = 0;
    mid_point_count_ = 0;
    
    if (generate_edge_points(object_name, edge_point_pool_, mid_point_pool_, 
                           edge_point_count_, mid_point_count_) == 0) {
        std::cerr << "Failed to generate edge points for object: " << object_name << std::endl;
        return false;
    }
    
    if (edge_idx >= static_cast<int>(edge_point_count_)) {
        std::cerr << "Invalid edge index: " << edge_idx << " (max: " << edge_point_count_ << ")" << std::endl;
        return false;
    }
    
    // Initialize push state
    PushState push_state;
    push_state.edge_idx = edge_idx;
    push_state.initial_edge_point = edge_point_pool_[edge_idx];
    push_state.initial_mid_point = mid_point_pool_[edge_idx];
    push_state.current_edge_point = push_state.initial_edge_point;
    push_state.current_mid_point = push_state.initial_mid_point;
    
    // Position robot at the edge point, facing the push direction
    double push_theta = std::atan2(
        push_state.initial_mid_point[1] - push_state.initial_edge_point[1],
        push_state.initial_mid_point[0] - push_state.initial_edge_point[0]
    );

    // Navigate from current robot position to the edge point.
    // For the point robot, this is a teleport. For the car, it's
    // rotate -> pure pursuit -> rotate.
    {
        // Refresh the wavefront so path extraction has the current grid.
        auto robot_state_now = env_.get_robot_state();
        std::vector<double> start_pos = robot_state_now
            ? std::vector<double>{robot_state_now->position[0], robot_state_now->position[1]}
            : std::vector<double>{0.0, 0.0};
        planner_.update_wavefront(env_, start_pos);

        auto path = planner_.extract_path(
            {start_pos[0], start_pos[1]},
            {push_state.initial_edge_point[0], push_state.initial_edge_point[1]}
        );

        if (path.empty()) {
            last_failure_reason_ = "No navigable path to edge point";
            return false;
        }

        // Teleport navigation: snap the robot to the final wavefront waypoint
        // with the target heading. Path-emptiness is already handled above.
        // The push controller does its own placement-collision check after this.
        const auto& goal_pt = path.back();
        env_.set_robot_se2(goal_pt[0], goal_pt[1], push_theta);
        (void)object_name;  // unused: collisions handled by the placement check below

        // Emit path for visualization tooling. Trajectory is empty under
        // teleport nav so we just emit the waypoint list and the phase marker.
        if (std::getenv("NAMO_NAV_LOG")) {
            std::cerr << "[NAV_PATH]";
            for (const auto& p : path) {
                std::cerr << " " << p[0] << "," << p[1];
            }
            std::cerr << std::endl;
            std::cerr << "[NAV_END]" << std::endl;
        }
    }
    
    // Check for robot collision with static objects (walls) after positioning
    const auto& static_objects = env_.get_static_objects();
    size_t num_static = env_.get_num_static();
    const auto robot_bodies = env_.get_robot_adapter()->get_collision_body_names();

    for (size_t i = 0; i < num_static; i++) {
        const auto& static_obj = static_objects[i];
        for (const auto& rb : robot_bodies) {
            if (env_.bodies_in_collision(rb, static_obj.body_name)) {
                last_failure_reason_ = "Robot placement collision with static object: " + static_obj.body_name + " (via " + rb + ")";
                last_collision_object_ = static_obj.body_name;
                return false;
            }
        }
    }

    // Check for robot collision with movable objects after positioning
    const auto& movable_objects = env_.get_movable_objects();
    size_t num_movable = env_.get_num_movable();

    for (size_t i = 0; i < num_movable; i++) {
        const auto& movable_obj = movable_objects[i];

        // Skip collision check with the object we're trying to push (expected contact)
        if (movable_obj.name == object_name) continue;
        for (const auto& rb : robot_bodies) {
            if (env_.bodies_in_collision(rb, movable_obj.body_name)) {
                last_failure_reason_ = "Robot placement collision with movable object: " + movable_obj.body_name + " (via " + rb + ")";
                last_collision_object_ = movable_obj.body_name;
                return false;
            }
        }
    }
    // ===== Unified push execution (was Path A for car, Path B for sphere) =====
    // Originally gated by is_diff_drive(): cars used one continuous loop ("Path
    // A"), spheres used a nested per-push-step loop ("Path B"). The split made
    // primitive .dat values encode different physics than the runtime sim for
    // the sphere robot, even when both went through this same function.
    // Unified to Path A semantics for all robots: zero velocities, pre-settle,
    // compute control ONCE, continuous push of push_steps × control_steps_per_push_
    // ticks (no mid-flight recomputation), post-settle. control_steps_per_push_
    // is no longer hardcoded to 250 — uses control_steps_per_push_ from config
    // so the sphere's 0.01s timestep × 75 ticks = 0.75s push step works
    // correctly. Settle phase shrunk 500 → 50 ticks because at the sphere's
    // 0.01s timestep, 50 × 0.01 = 0.5s damping is enough; the original 500
    // was tuned for the car's 0.002s timestep (= 1s damping).
    {
        // Pre/post-push settle. Default 100 ticks: at sphere's 0.01 s timestep
        // = 1 s damping; at car's 0.002 s timestep = 0.2 s. Both are enough
        // for the chassis to drop onto its wheels after the teleport nav
        // and for object velocities to bleed off. Configurable via
        // set_settle_steps() for visualization or stricter physics runs.
        const int kSettleSteps = settle_steps_override_ > 0 ? settle_steps_override_ : 100;
        const auto* robot_adapter = env_.get_robot_adapter();
        const bool use_diff_drive_tracking =
            robot_adapter && robot_adapter->is_diff_drive() && push_path_follower_;

        // 1) Zero all velocities (chassis + wheels + casters + every object) and stop wheel ctrl
        env_.set_zero_velocity();
        if (use_diff_drive_tracking) {
            env_.apply_wheel_control(0.0, 0.0);
            push_path_follower_->reset();
        } else {
            env_.apply_robot_control(0.0, 0.0);
        }

        // 2) Pre-push settle
        for (int i = 0; i < kSettleSteps; ++i) {
            env_.step_simulation();
            env_.get_mujoco_wrapper()->notify_physics_step();
            dump_qpos(env_, /*phase=*/3);
        }

        // 3) Snapshot for stuck/coll checks
        auto obj_state0 = env_.get_object_state(object_name);
        if (!obj_state0) {
            std::cerr << "Lost object state before diff-drive push" << std::endl;
            return false;
        }
        update_push_state(push_state, obj_state0->position, obj_state0->size, obj_state0->quaternion);

        // 4) For holonomic robots keep the legacy held push command.
        if (!use_diff_drive_tracking) {
            auto control = compute_push_control(push_state);
            env_.apply_robot_control(control[0], control[1]);
        }

        // 5) Continuous push for push_steps × control_steps_per_push_ sim ticks
        const int total_sim_steps = push_steps * control_steps_per_push_;
        // By value: get_object_state hands back a pointer into live simulator
        // state, so anything read through obj_state0 moves with the object.
        const std::array<double, 3> push_start_pos = obj_state0->position;
        std::array<double, 3> prev_pos_sample = obj_state0->position;
        std::array<double, 4> prev_quat_sample = obj_state0->quaternion;
        const auto robot_bodies = env_.get_robot_adapter()->get_collision_body_names();
        const double wheel_radius =
            (robot_adapter && robot_adapter->get_wheel_radius() > 0.0)
                ? robot_adapter->get_wheel_radius()
                : 0.015;

        for (int t = 0; t < total_sim_steps; ++t) {
            if (use_diff_drive_tracking) {
                if (dynamic_direction_ || t == 0) {
                    auto obj_for_path = env_.get_object_state(object_name);
                    if (!obj_for_path) {
                        return false;
                    }
                    update_push_state(
                        push_state,
                        obj_for_path->position,
                        obj_for_path->size,
                        obj_for_path->quaternion);

                    const PushPathFollower::Pose robot_pose = get_current_robot_pose();
                    const std::array<double, 2> actual_pos = {robot_pose.x_m, robot_pose.y_m};
                    const auto push_path = build_push_tracking_path(push_state, actual_pos);
                    push_path_follower_->set_path(push_path);
                    log_push_path(t, push_path);
                }

                const PushPathFollower::Pose robot_pose = get_current_robot_pose();
                const double timestamp_s = env_.get_mujoco_wrapper()->data()->time;
                const auto follower_step = push_path_follower_->step(robot_pose, timestamp_s);
                // Diff-drive: chassis m/s comes from the calibration constant,
                // NOT push_velocity_ (which is only used by the holonomic robot).
                // The follower's fraction × kCarWheelMaxSpeedMs is the chassis
                // speed contribution from that wheel; divide by wheel radius to
                // get the angular velocity the actuator wants.
                const double left_omega = (follower_step.left_speed * kCarWheelMaxSpeedMs) / wheel_radius;
                const double right_omega = (follower_step.right_speed * kCarWheelMaxSpeedMs) / wheel_radius;
                env_.apply_wheel_control(left_omega, right_omega);
                log_push_control(t, left_omega, right_omega, follower_step.mode);
            }

            env_.step_simulation();
            env_.get_mujoco_wrapper()->notify_physics_step();
            dump_qpos(env_, /*phase=*/3);

            // Periodic collision + stuck checks (don't check every tick — too expensive)
            if (t > 0 && (t % stuck_check_stride_ == 0)) {
                auto obj_now = env_.get_object_state(object_name);
                if (!obj_now) return false;

                // Collisions are scanned BEFORE the stuck decision. A push that
                // stops because the object wedged against a wall reports that
                // wall, and a robot collision still wins over any motion.
                for (size_t i = 0; i < num_static; i++) {
                    const auto& s = static_objects[i];
                    for (const auto& rb : robot_bodies) {
                        if (env_.bodies_in_collision(rb, s.body_name)) {
                            wall_collision_during_push_ = true;
                            last_failure_reason_ = "Robot collision during push with static object: " + s.body_name + " (via " + rb + ")";
                            last_collision_object_ = s.body_name;
                            return false;
                        }
                    }
                    // The pushed object touching a wall is a normal outcome, not
                    // a failed push. Record it and keep going.
                    if (env_.bodies_in_collision(object_name, s.body_name)) {
                        wall_collision_during_push_ = true;
                    }
                }
                for (size_t i = 0; i < num_movable; i++) {
                    const auto& mv = movable_objects[i];
                    if (mv.name == object_name) continue;
                    for (const auto& rb : robot_bodies) {
                        if (env_.bodies_in_collision(rb, mv.body_name)) {
                            movable_collisions_during_push_.insert(mv.name);
                            last_failure_reason_ = "Robot collision during push with movable object: " + mv.body_name + " (via " + rb + ")";
                            last_collision_object_ = mv.body_name;
                            return false;
                        }
                    }
                    // Likewise for the pushed object nudging another movable.
                    if (env_.bodies_in_collision(object_name, mv.body_name)) {
                        movable_collisions_during_push_.insert(mv.name);
                    }
                }

                const bool stuck = update_stuck_counter_and_check_abort(
                    prev_pos_sample, prev_quat_sample,
                    obj_now->position, obj_now->quaternion,
                    t / control_steps_per_push_, t % control_steps_per_push_);
                if (stuck) {
                    // Motion the object has already made is a real result. Keep
                    // it, settle, and let the caller replan from where things
                    // actually ended up. Only a push that never moved anything
                    // is worth throwing away.
                    const double dx = obj_now->position[0] - push_start_pos[0];
                    const double dy = obj_now->position[1] - push_start_pos[1];
                    if (std::sqrt(dx * dx + dy * dy) < kMinUsefulPushDisplacementM) {
                        return false;
                    }
                    last_push_stopped_early_ = true;
                    settle_after_push(use_diff_drive_tracking, kSettleSteps);
                    return true;
                }
                prev_pos_sample = obj_now->position;
                prev_quat_sample = obj_now->quaternion;
            }
        }

        // 6) Stop wheels and post-push settle
        settle_after_push(use_diff_drive_tracking, kSettleSteps);

        return true;
    }
}


bool NAMOPushController::is_push_valid(const std::string& object_name,
                                      int edge_idx,
                                      const std::array<double, 7>& goal_state) {
    // Simplified validity check - could be enhanced with trajectory prediction
    
    // Check if edge index is valid
    edge_point_count_ = 0;
    mid_point_count_ = 0;
    
    if (generate_edge_points(object_name, edge_point_pool_, mid_point_pool_,
                           edge_point_count_, mid_point_count_) == 0) {
        return false;
    }
    
    if (edge_idx >= static_cast<int>(edge_point_count_)) {
        return false;
    }
    
    // Check if robot can reach the edge point
    auto robot_state = env_.get_robot_state();
    if (!robot_state) {
        return false;
    }
    
    double dx = edge_point_pool_[edge_idx][0] - robot_state->position[0];
    double dy = edge_point_pool_[edge_idx][1] - robot_state->position[1];
    double distance = std::sqrt(dx * dx + dy * dy);
    
    // Simple distance check - could use wavefront reachability
    return distance < 5.0; // Max reach distance
}

size_t NAMOPushController::get_reachable_objects(std::array<std::string, 20>& reachable_objects,
                                               size_t& reachable_count,
                                               size_t max_objects) {
    reachable_count = 0;
    
    auto movable_objects = env_.get_movable_objects();
    auto robot_state = env_.get_robot_state();
    
    if (!robot_state) {
        return 0;
    }
    
    // Update wavefront with current robot position and environment state
    std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
    planner_.update_wavefront(env_, robot_pos);
    
    // Get the distance grid for reachability queries
    const auto& distance_grid = planner_.get_distance_grid();
    
    // Check each movable object for reachability
    for (size_t obj_idx = 0; obj_idx < env_.get_num_movable() && reachable_count < max_objects; ++obj_idx) {
        const auto& obj_info = movable_objects[obj_idx];
        
        edge_point_count_ = 0;
        mid_point_count_ = 0;
        
        if (generate_edge_points(obj_info.name, edge_point_pool_, mid_point_pool_,
                               edge_point_count_, mid_point_count_) > 0) {
            // Check if any edge point is reachable via wavefront
            bool reachable = false;
            for (size_t i = 0; i < edge_point_count_; ++i) {
                // Convert edge point to grid coordinates
                int edge_x = planner_.world_to_grid_x(edge_point_pool_[i][0]);
                int edge_y = planner_.world_to_grid_y(edge_point_pool_[i][1]);
                
                // Check if edge point is within grid bounds and reachable
                if (planner_.is_valid_grid_coord(edge_x, edge_y)) {
                    // Check for reachable (value = 1), not just non-obstacle (>= 0)
                    if (distance_grid[edge_x][edge_y] == 1) {
                        reachable = true;
                        break;
                    }
                }
            }
            
            if (reachable) {
                reachable_objects[reachable_count++] = obj_info.name;
            }
        }
    }
    
    return reachable_count;
}

std::vector<int> NAMOPushController::get_reachable_edge_indices(const std::string& object_name) {
    std::vector<int> reachable_edges;
    
    try {
        // Generate edge points for this object
        edge_point_count_ = 0;
        mid_point_count_ = 0;
        
        if (generate_edge_points(object_name, edge_point_pool_, mid_point_pool_, 
                               edge_point_count_, mid_point_count_) == 0) {
            // std::cout << "No edge points generated for object: " << object_name << std::endl;
            return reachable_edges; // Empty if no edge points
        }
        
        // std::cout << "Generated " << edge_point_count_ << " edge points for " << object_name << std::endl;
        
        // Update wavefront with current robot position
        auto robot_state = env_.get_robot_state();
        if (!robot_state) {
            // std::cout << "No robot state available for reachability check" << std::endl;
            return reachable_edges;
        }
        
        std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
        // std::cout << "Robot position: [" << robot_pos[0] << ", " << robot_pos[1] << "]" << std::endl;
        
        // Safely update wavefront
        try {
            planner_.update_wavefront(env_, robot_pos);
            // std::cout << "Wavefront updated successfully" << std::endl;
        } catch (const std::exception& e) {
            // std::cout << "Error updating wavefront: " << e.what() << std::endl;
            return reachable_edges;
        }
        
        const auto& distance_grid = planner_.get_distance_grid();
        
        // Check each edge index for reachability
        for (int edge_idx = 0; edge_idx < static_cast<int>(edge_point_count_); ++edge_idx) {
            try {
                int edge_x = planner_.world_to_grid_x(edge_point_pool_[edge_idx][0]);
                int edge_y = planner_.world_to_grid_y(edge_point_pool_[edge_idx][1]);
                
                if (planner_.is_valid_grid_coord(edge_x, edge_y)) {
                    // Check for reachable (value = 1), not just non-obstacle (>= 0)
                    if (distance_grid[edge_x][edge_y] == 1) {
                        reachable_edges.push_back(edge_idx);
                    }
                }
            } catch (const std::exception& e) {
                // std::cout << "Error checking edge " << edge_idx << ": " << e.what() << std::endl;
                continue;
            }
        }
        
        // std::cout << "Object " << object_name << ": " << reachable_edges.size() 
                  // << "/" << edge_point_count_ << " edges reachable: [";
        for (size_t i = 0; i < reachable_edges.size(); ++i) {
            // std::cout << reachable_edges[i];
            if (i < reachable_edges.size() - 1) std::cout << ", ";
        }
        // std::cout << "]" << std::endl;
        
    } catch (const std::exception& e) {
        // std::cout << "Error in get_reachable_edge_indices: " << e.what() << std::endl;
        return reachable_edges;
    }
    
    return reachable_edges;
}

void NAMOPushController::get_memory_stats(size_t& primitives_used, size_t& states_used) {
    primitives_used = primitive_count_;
    states_used = state_count_;
}

} // namespace namo
