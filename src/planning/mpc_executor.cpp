#include "planning/mpc_executor.hpp"
#include "core/types.hpp"
#include <iostream>
#include <cmath>

namespace namo {

MPCExecutor::MPCExecutor(NAMOEnvironment& env)
    : env_(env), planner_(0.02, env_, {0.15, 0.15}), controller_(env_, planner_, 10, 250, 1.0), has_robot_goal_(false) {
    
    // Set default parameters
    set_parameters();
}

MPCExecutor::MPCExecutor(NAMOEnvironment& env, double resolution, const std::vector<double>& robot_size, 
                         int max_push_steps, int control_steps_per_push, double force_scaling, int points_per_face)
    : env_(env), 
      planner_(resolution, env_, robot_size), 
      controller_(env_, planner_, max_push_steps, control_steps_per_push, force_scaling, points_per_face), 
      has_robot_goal_(false) {
    
    // Set default parameters
    set_parameters();
}

void MPCExecutor::set_parameters(int max_mpc_steps, 
                                double distance_threshold,
                                double angle_threshold,
                                int max_stuck_iterations) {
    max_mpc_steps_ = max_mpc_steps;
    distance_threshold_ = distance_threshold;
    angle_threshold_ = angle_threshold;
    max_stuck_iterations_ = max_stuck_iterations;
}

void MPCExecutor::set_robot_goal(const std::array<double, 2>& robot_goal) {
    robot_goal_ = robot_goal;
    has_robot_goal_ = true;
}

ExecutionResult MPCExecutor::execute_plan(
    const std::string& object_name,
    const std::vector<PlanStep>& plan_sequence) {
    
    ExecutionResult result;
    
    if (plan_sequence.empty()) {
        result.failure_reason = "Empty plan sequence";
        return result;
    }
    
    // std::cout << "Executing plan with " << plan_sequence.size() << " primitive steps" << std::endl;
    
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
        
        if (!step_success) {
            result.failure_reason = "Primitive step " + std::to_string(i+1) + " failed";
            result.steps_executed = i;
            result.final_object_state = get_object_se2_state(object_name);
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
    return result;
}

bool MPCExecutor::execute_primitive_step(
    const std::string& object_name,
    const PlanStep& plan_step) {
    
    // For now, we'll execute the primitive directly without explicit goal setting
    // The push controller will handle the primitive execution with physics
    // std::cout << "Executing primitive: edge=" << plan_step.edge_idx 
    //           << " steps=" << plan_step.push_steps
    //           << " target_pose=[" << plan_step.pose.x << "," << plan_step.pose.y 
    //           << "," << plan_step.pose.theta << "]" << std::endl;
    
    // Execute MPC following old implementation approach (namo_planner.hpp:217-292)
    int stuck_counter = 0;
    SE2State previous_state = get_object_se2_state(object_name);
    
    for (int mpc_step = 0; mpc_step < plan_step.push_steps; mpc_step++) {
        // Check if robot goal became reachable during MPC
        if (has_robot_goal_ && is_robot_goal_reachable()) {
            // std::cout << "Robot goal became reachable during MPC step " << mpc_step << std::endl;
            return true;
        }
        
        // Check if object reached target
        if (is_object_at_target(object_name, plan_step.pose)) {
            // std::cout << "Object reached target in MPC step " << mpc_step << std::endl;
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

        // std::cout << "reachable_edges: " << reachable_edges.size() << std::endl;
        // for (int edge_idx : reachable_edges) {
        //     std::cout << "reachable_edge: " << edge_idx << std::endl;
        // }

        // Execute one push primitive using the controller
        // Use edge index and step count from the plan
        bool push_success = controller_.execute_push_primitive(object_name, plan_step.edge_idx, 1);
        
        // if (push_success) {
        //     // std::cout << "Push controller reached target location in MPC step " << mpc_step << std::endl;
        //     return true;
        // }
        
        // Check if object is stuck
        SE2State current_state = get_object_se2_state(object_name);
        // std::cout << "current_state: " << current_state.x << ", " << current_state.y << ", " << current_state.theta << std::endl;
        // std::cout << "previous_state: " << previous_state.x << ", " << previous_state.y << ", " << previous_state.theta << std::endl;
        if (is_object_stuck(object_name, previous_state)) {
            stuck_counter++;
            if (stuck_counter > max_stuck_iterations_) {
                // std::cout << "Object stuck for " << stuck_counter << " iterations" << std::endl;
                return false;
            }
        } else {
            stuck_counter = 0;  // Reset stuck counter if object moved
        }
        
        previous_state = current_state;
    }
    
    // std::cout << "MPC reached step limit without reaching target" << std::endl;
    return false;
}

bool MPCExecutor::is_robot_goal_reachable() {
    if (!has_robot_goal_) {
        return false;
    }
    
    // Use the incremental wavefront planner to check reachability
    try {
        // Get current robot position
        auto robot_state = env_.get_robot_state();
        std::array<double, 2> robot_pos = {robot_state->position[0], robot_state->position[1]};
        
        // Update wavefront with current robot position
        planner_.update_wavefront(env_, {robot_pos[0], robot_pos[1]});
        
        // Check if goal is reachable (using default robot size)
        return planner_.is_goal_reachable(robot_goal_, 0.15);
    } catch (const std::exception& e) {
        std::cerr << "Error checking robot goal reachability: " << e.what() << std::endl;
        return false;
    }
}

bool MPCExecutor::is_object_at_target(const std::string& object_name, const SE2State& target_state) {
    SE2State current_state = get_object_se2_state(object_name);
    
    double dx = current_state.x - target_state.x;
    double dy = current_state.y - target_state.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    
    double angle_diff = std::abs(current_state.theta - target_state.theta);
    while (angle_diff > M_PI) angle_diff = 2.0 * M_PI - angle_diff;
    
    return distance < distance_threshold_ && angle_diff < angle_threshold_;
}

SE2State MPCExecutor::get_object_se2_state(const std::string& object_name) {
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

std::vector<double> MPCExecutor::se2_to_goal_state(const SE2State& se2_state) {
    // Convert SE(2) to goal state format: [x, y, z, qw, qx, qy, qz]
    // Z is set to 0.0, quaternion represents yaw rotation
    
    double half_yaw = se2_state.theta / 2.0;
    double qw = std::cos(half_yaw);
    double qx = 0.0;
    double qy = 0.0; 
    double qz = std::sin(half_yaw);
    
    return {se2_state.x, se2_state.y, 0.0, qw, qx, qy, qz};
}

bool MPCExecutor::is_object_stuck(const std::string& object_name, const SE2State& previous_state) {
    SE2State current_state = get_object_se2_state(object_name);
    
    double dx = current_state.x - previous_state.x;
    double dy = current_state.y - previous_state.y;

    // std::cout << "dx: " << dx << ", dy: " << dy << std::endl;
    double distance_moved = std::sqrt(dx*dx + dy*dy);
    
    double angle_change = std::abs(current_state.theta - previous_state.theta);
    while (angle_change > M_PI) angle_change = 2.0 * M_PI - angle_change;
    
    // Consider stuck if both position and orientation changes are very small
    const double min_position_change = 0.01;  // 1mm
    const double min_angle_change = 0.1;      // ~0.6 degrees
    
    return distance_moved < min_position_change && angle_change < min_angle_change;
}

void MPCExecutor::save_debug_wavefront(int iteration, const std::string& base_filename) {
    planner_.save_wavefront_iteration(base_filename, iteration);
}

std::vector<int> MPCExecutor::get_reachable_edges_with_wavefront(const std::string& object_name) {
    // Get robot current position
    auto robot_state = env_.get_robot_state();
    if (!robot_state) {
        // std::cout << "Warning: Could not get robot state for wavefront update" << std::endl;
        return {}; 
    }
    
    // Update wavefront from current robot position
    std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
    bool wavefront_updated = planner_.update_wavefront(env_, robot_pos);
    
    if (!wavefront_updated) {
        // std::cout << "Warning: Wavefront update failed" << std::endl;
    }
    
    // Get object current position to generate edge points
    auto obj_pose = env_.get_object_state(object_name);
    if (!obj_pose) {
        // std::cout << "Warning: Could not get object pose for " << object_name << std::endl;
        return {};
    }
    
    std::vector<int> reachable_edges;

    // Use controller to generate edge points (supports dynamic n points per edge)
    std::array<std::array<double, 2>, 64> edge_points;  // Use MAX_EDGE_POINTS from controller
    std::array<std::array<double, 2>, 64> mid_points;   // Not used but required
    size_t edge_count, mid_count;

    if (controller_.generate_edge_points(object_name, edge_points, mid_points, edge_count, mid_count) == 0) {
        // std::cout << "Warning: Could not generate edge points for " << object_name << std::endl;
        return {};
    }
    
    // Get the mutable wavefront grid for marking edge points
    auto& grid = planner_.get_mutable_grid();
    
    // Check each transformed edge point for reachability and mark in grid
    for (size_t edge_idx = 0; edge_idx < edge_count; edge_idx++) {
        try {
            // Convert world edge point to grid coordinates  
            int edge_x = planner_.world_to_grid_x(edge_points[edge_idx][0]);  // Changed from world_edge_points
            int edge_y = planner_.world_to_grid_y(edge_points[edge_idx][1]);  // Changed from world_edge_points
            
            if (planner_.is_valid_grid_coord(edge_x, edge_y)) {
                // Check if edge point is reachable (not obstacle -2, not unreachable 0)
                // Only accept grid values > 0 (reachable positions)
                if (grid[edge_x][edge_y] > 0) {
                    reachable_edges.push_back(static_cast<int>(edge_idx));
                    // Mark reachable edge points in grid as -3
                    grid[edge_x][edge_y] = -3;
                } else {
                    // Mark unreachable edge points in grid as -4  
                    grid[edge_x][edge_y] = -4;
                }
            }
        } catch (const std::exception& e) {
            // std::cout << "Error checking edge " << edge_idx << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    // std::cout << "Wavefront analysis: " << reachable_edges.size() 
            //   << "/" << edge_count << " edges reachable for " << object_name << std::endl;
    
    return reachable_edges;
}

} // namespace namo