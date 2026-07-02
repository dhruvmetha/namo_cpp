#include "planning/push_primitive_executor.hpp"
#include "planning/namo_push_controller.hpp"
#include "core/types.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <iostream>
#include <cmath>
#include <map>
#include <unordered_set>

namespace namo {

PushPrimitiveExecutor::PushPrimitiveExecutor(NAMOEnvironment& env)
    : env_(env),
      planner_(0.02, env_, {env.get_robot_info().size[0], env.get_robot_info().size[1]}, 0.005),
      // 0.05 m/s = the 1× safe value. The historic default 1.0 was tuned for
      // the old <motor> actuators where it scaled force; under the current
      // <velocity> actuators it would command ~67 rad/s wheel velocity on the
      // car and saturate. Tests that need a different speed should construct
      // with a ConfigManager and set skill.push_velocity explicitly.
      controller_(env_, planner_, 10, 250, 0.05),
      has_robot_goal_(false) {

    // Set default parameters
    set_parameters();
}

PushPrimitiveExecutor::PushPrimitiveExecutor(NAMOEnvironment& env, double resolution, const std::vector<double>& robot_size,
                         double wavefront_tier1_inflation_margin,
                         int max_push_steps, int control_steps_per_push, double push_velocity, int points_per_face,
                         bool check_object_collision,
                         bool dynamic_direction)
    : env_(env),
      planner_(resolution, env_, robot_size, wavefront_tier1_inflation_margin),
      controller_(env_, planner_, max_push_steps, control_steps_per_push, push_velocity, points_per_face, dynamic_direction),
      has_robot_goal_(false) {

    // Set default parameters
    set_parameters();

    // Configure collision checking
    controller_.set_collision_checking(check_object_collision);
}

void PushPrimitiveExecutor::set_parameters(int max_mpc_steps, 
                                double distance_threshold,
                                double angle_threshold,
                                int max_stuck_iterations) {
    max_mpc_steps_ = max_mpc_steps;
    distance_threshold_ = distance_threshold;
    angle_threshold_ = angle_threshold;
    max_stuck_iterations_ = max_stuck_iterations;
}

void PushPrimitiveExecutor::set_robot_goal(const std::array<double, 2>& robot_goal) {
    robot_goal_ = robot_goal;
    has_robot_goal_ = true;
}

ExecutionResult PushPrimitiveExecutor::execute_plan(
    const std::string& object_name,
    const std::vector<PlanStep>& plan_sequence) {
    
    ExecutionResult result;

    if (plan_sequence.empty()) {
        result.failure_reason = "Empty plan sequence";
        return result;
    }

    // std::cout << "Executing plan with " << plan_sequence.size() << " primitive steps" << std::endl;

    // Accumulate collision info across all pushes in the plan
    bool accumulated_wall_collision = false;
    std::unordered_set<std::string> accumulated_movable_collisions;

    // Execute each primitive in sequence
    for (size_t i = 0; i < plan_sequence.size(); i++) {
        const PlanStep& step = plan_sequence[i];
        
        // std::cout << "Executing step " << (i+1) << "/" << plan_sequence.size() 
                //   << " - Edge:" << step.edge_idx << " Steps:" << step.push_steps << std::endl;
        
        // Check if robot goal is reachable before executing this step
        if (has_robot_goal_ && is_robot_goal_reachable()) {
            // std::cout << "Robot goal became reachable before step " << (i+1) << std::endl;
            result.success = true;
            result.robot_goal_reached = true;
            result.steps_executed = i;
            result.final_object_state = get_object_se2_state(object_name);
            return result;
        }
        
        // Execute this primitive step
        bool step_success = execute_primitive_step(object_name, step);

        // Accumulate collision info from this push (even on failure)
        if (controller_.get_wall_collision_during_push()) {
            accumulated_wall_collision = true;
        }
        const auto& movable_set = controller_.get_movable_collisions_during_push();
        accumulated_movable_collisions.insert(movable_set.begin(), movable_set.end());

        if (!step_success) {
            result.failure_reason = "Primitive step " + std::to_string(i+1) + " failed";
            result.collision_object = controller_.get_last_collision_object();
            result.steps_executed = i;
            result.final_object_state = get_object_se2_state(object_name);
            // Copy accumulated collision info to result
            result.wall_collision_during_push = accumulated_wall_collision;
            result.movable_collisions_during_push.assign(accumulated_movable_collisions.begin(), accumulated_movable_collisions.end());
            // Propagate controller-level stuck reason if threshold was hit
            int ctrl_stuck = controller_.get_last_stuck_counter();
            if (ctrl_stuck >= controller_.get_stuck_threshold()) {
                result.failure_reason = "Controller-level stuck (counter=" + std::to_string(ctrl_stuck) + ")";
                // Surface a stuck marker via outputs channel by mapping to collision_object as empty and reason text
                // The skill will translate this into outputs["stuck"]="true" semantics if desired.
            }
            return result;
        }

        result.steps_executed = i + 1;
    }
    
    // Check final state
    if (has_robot_goal_ && is_robot_goal_reachable()) {
        // std::cout << "Robot goal reachable after plan execution" << std::endl;
        result.success = true;
        result.robot_goal_reached = true;
    } else {
        // std::cout << "Plan executed but robot goal not reachable" << std::endl;
        result.success = true;
        result.robot_goal_reached = false;
    }

    result.final_object_state = get_object_se2_state(object_name);
    // Copy accumulated collision info to result
    result.wall_collision_during_push = accumulated_wall_collision;
    result.movable_collisions_during_push.assign(accumulated_movable_collisions.begin(), accumulated_movable_collisions.end());
    return result;
}

bool PushPrimitiveExecutor::execute_primitive_step(
    const std::string& object_name,
    const PlanStep& plan_step) {
    
    // For now, we'll execute the primitive directly without explicit goal setting
    // The push controller will handle the primitive execution with physics
    // std::cout << "Executing primitive: edge=" << plan_step.edge_idx 
    //           << " steps=" << plan_step.push_steps
    //           << " target_pose=[" << plan_step.pose.x << "," << plan_step.pose.y 
    //           << "," << plan_step.pose.theta << "]" << std::endl;
    
    // Pre-push checks. Goal-reachability and edge-reachability are checked
    // once BEFORE the primitive, not between push_steps — the unified Path A
    // in NAMOPushController::execute_push_primitive does one continuous push
    // for push_steps × control_steps_per_push_ ticks with internal stuck and
    // collision detection. Wrapping it in a per-mpc_step outer loop here
    // (which is what this code used to do, calling execute_push_primitive
    // with push_steps=1 N times) reintroduced the "push-pause-push-pause"
    // fragmentation that the unified push path was meant to eliminate.
    if (has_robot_goal_ && is_robot_goal_reachable()) {
        return true;
    }

    bool edge_idx_reachable = false;
    std::vector<int> reachable_edges = get_reachable_edges_with_wavefront(object_name);
    for (int edge_idx : reachable_edges) {
        if (edge_idx == plan_step.edge_idx) {
            edge_idx_reachable = true;
            break;
        }
    }
    if (!edge_idx_reachable) {
        return false;
    }

    // Save full simulation state before attempting push (zero-allocation)
    env_.save_full_state();

    // ONE continuous primitive call. Stuck detection and collision checks
    // happen INSIDE the controller every stuck_check_stride_ ticks; no need
    // to chunk by push_step here.
    bool push_success = controller_.execute_push_primitive(
        object_name, plan_step.edge_idx, plan_step.push_steps);

    if (!push_success) {
        // Restore full simulation state on push failure (e.g., collision during robot placement)
        env_.restore_full_state();
        env_.set_zero_velocity();
        return false;
    }

    return true;
}

bool PushPrimitiveExecutor::update_wavefront_from_robot_position() {
    auto robot_state = env_.get_robot_state();
    if (!robot_state) {
        return false;
    }
    std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
    return planner_.update_wavefront(env_, robot_pos);
}

bool PushPrimitiveExecutor::is_robot_goal_reachable() {
    if (!has_robot_goal_) {
        return false;
    }

    // Use the incremental wavefront planner to check reachability
    try {
        if (!update_wavefront_from_robot_position()) {
            return false;
        }
        const double goal_tolerance = compute_goal_tolerance_m(
            planner_.get_robot_size(),
            planner_.get_tier1_inflation_margin());
        return planner_.is_goal_reachable(robot_goal_, goal_tolerance);
    } catch (const std::exception& e) {
        std::cerr << "Error checking robot goal reachability: " << e.what() << std::endl;
        return false;
    }
}

std::pair<int, int> PushPrimitiveExecutor::count_reachable_points(
    const std::vector<std::array<double, 2>>& points) {
    if (points.empty()) {
        return {0, -1};
    }
    try {
        if (!update_wavefront_from_robot_position()) {
            return {0, -1};
        }
    } catch (const std::exception& e) {
        std::cerr << "Error updating wavefront for count_reachable_points: " << e.what() << std::endl;
        return {0, -1};
    }
    const double goal_tolerance = compute_goal_tolerance_m(
        planner_.get_robot_size(),
        planner_.get_tier1_inflation_margin());
    int count = 0;
    int first_idx = -1;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (planner_.is_goal_reachable(points[i], goal_tolerance)) {
            ++count;
            if (first_idx == -1) first_idx = i;
        }
    }
    return {count, first_idx};
}


SE2State PushPrimitiveExecutor::get_object_se2_state(const std::string& object_name) {
    auto object_state = env_.get_object_state(object_name);
    if (!object_state) {
        std::cerr << "Failed to get object state for: " << object_name << std::endl;
        return SE2State();
    }
    
    // Convert quaternion to yaw angle
    double yaw = std::atan2(
        2.0 * (object_state->quaternion[0] * object_state->quaternion[3] + 
               object_state->quaternion[1] * object_state->quaternion[2]),
        1.0 - 2.0 * (object_state->quaternion[2] * object_state->quaternion[2] + 
                      object_state->quaternion[3] * object_state->quaternion[3])
    );
    
    return SE2State(object_state->position[0], object_state->position[1], yaw);
}

bool PushPrimitiveExecutor::is_object_stuck(const std::string& object_name, const SE2State& previous_state) {
    SE2State current_state = get_object_se2_state(object_name);
    
    double dx = current_state.x - previous_state.x;
    double dy = current_state.y - previous_state.y;

    // std::cout << "dx: " << dx << ", dy: " << dy << std::endl;
    double distance_moved = std::sqrt(dx*dx + dy*dy);
    
    double angle_change = std::abs(current_state.theta - previous_state.theta);
    while (angle_change > M_PI) angle_change = 2.0 * M_PI - angle_change;
    
    // Consider stuck if both position and orientation changes are very small
    const double min_position_change = 0.001;  // 1mm
    const double min_angle_change = 0.05;      // tighter yaw threshold
    
    return distance_moved < min_position_change && angle_change < min_angle_change;
}

std::vector<int> PushPrimitiveExecutor::get_reachable_edges_with_wavefront(const std::string& object_name) {
    auto detailed = get_reachable_edges_with_wavefront_detailed(object_name);
    return detailed.edge_indices;
}

PushPrimitiveExecutor::ReachableEdgesResult PushPrimitiveExecutor::get_reachable_edges_with_wavefront_detailed(
    const std::string& object_name) {
    ReachableEdgesResult result;
    if (!update_wavefront_from_robot_position()) {
        return result;
    }
    return get_reachable_edges_from_current_wavefront(object_name);
}

PushPrimitiveExecutor::ReachabilitySnapshot PushPrimitiveExecutor::compute_reachability_snapshot() {
    ReachabilitySnapshot snapshot;
    if (!update_wavefront_from_robot_position()) {
        return snapshot;
    }

    if (has_robot_goal_) {
        const double goal_tolerance = compute_goal_tolerance_m(
            planner_.get_robot_size(),
            planner_.get_tier1_inflation_margin());
        snapshot.goal_reachable = planner_.is_goal_reachable(robot_goal_, goal_tolerance);
    }

    const auto& movable_objects = env_.get_movable_objects();
    for (size_t i = 0; i < env_.get_num_movable(); ++i) {
        const auto& obj_info = movable_objects[i];
        if (!obj_info.name.empty()) {
            snapshot.object_edges[obj_info.name] =
                get_reachable_edges_from_current_wavefront(obj_info.name);
        }
    }
    return snapshot;
}

PushPrimitiveExecutor::ReachableEdgesResult PushPrimitiveExecutor::get_reachable_edges_from_current_wavefront(
    const std::string& object_name) {
    ReachableEdgesResult result;

    auto obj_pose = env_.get_object_state(object_name);
    if (!obj_pose) {
        return result;
    }

    // Use controller to generate edge points (supports dynamic n points per edge)
    std::array<std::array<double, 2>, NAMOPushController::MAX_EDGE_POINTS> edge_points;
    std::array<std::array<double, 2>, NAMOPushController::MAX_EDGE_POINTS> mid_points;   // Not used but required
    size_t edge_count = 0;
    size_t mid_count = 0;

    if (controller_.generate_edge_points(object_name, edge_points, mid_points, edge_count, mid_count) == 0) {
        return result;
    }

    result.total_edge_points = static_cast<int>(edge_count);

    // Read the current wavefront grid without mutating canonical values.
    const auto& grid = planner_.get_grid();

    // Check each transformed edge point for reachability
    for (size_t edge_idx = 0; edge_idx < edge_count; edge_idx++) {
        try {
            // Convert world edge point to grid coordinates
            int edge_x = planner_.world_to_grid_x(edge_points[edge_idx][0]);
            int edge_y = planner_.world_to_grid_y(edge_points[edge_idx][1]);

            if (planner_.is_valid_grid_coord(edge_x, edge_y)) {
                int grid_val = grid[edge_x][edge_y];
                // Reachable == 1. Non-reachable free == 0. Occupied == -1.
                if (grid_val == 1) {
                    result.edge_indices.push_back(static_cast<int>(edge_idx));
                }
            }
        } catch (const std::exception&) {
            continue;
        }
    }

    return result;
}

std::vector<int> PushPrimitiveExecutor::evaluate_primitive_priorities(
    NAMOEnvironment& env,
    const std::string& object_name,
    const std::vector<std::array<double, 3>>& target_poses,
    const std::array<double, 2>& robot_goal) {
    // Delegate to wavefront planner
    return planner_.evaluate_primitive_priorities(env, object_name, target_poses, robot_goal);
}

std::map<std::string, double> PushPrimitiveExecutor::get_last_priority_profile() const {
    return planner_.get_last_priority_profile();
}

} // namespace namo
