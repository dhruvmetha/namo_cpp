#include "planning/namo_push_controller.hpp"
#include "core/mujoco_wrapper.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace namo {

NAMOPushController::NAMOPushController(NAMOEnvironment& env, 
                                     WavefrontPlanner& planner,
                                     int push_steps,
                                     int control_steps,
                                     double scaling)
    : env_(env), planner_(planner), 
      default_push_steps_(push_steps),
      control_steps_per_push_(control_steps),
      force_scaling_(scaling) {
    
    // Initialize robot size from environment
    auto robot_info = env_.get_robot_info();
    robot_size_[0] = robot_info.size[0];
    robot_size_[1] = robot_info.size[1]; 
    robot_size_[2] = robot_info.size[2];
    
    // Pre-allocate memory pools (they're already initialized as empty)
    // std::cout << "NAMO Push Controller initialized:" << std::endl;
    // std::cout << "  Push steps: " << default_push_steps_ << std::endl;
    // std::cout << "  Control steps per push: " << control_steps_per_push_ << std::endl;
    // std::cout << "  Force scaling: " << force_scaling_ << std::endl;
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
    
    // Convert quaternion to rotation angle (yaw) - matching older implementation
    double yaw = utils::quaternion_to_yaw(obj_quat);
    
    // Object dimensions - subtract margin like older implementation
    double x = 0.0, y = 0.0;
    double w = obj_size[0] - 0.05;  // width with margin (MuJoCo size is half-extent, so this is (half_width - 0.05))
    double d = obj_size[1] - 0.05;  // depth with margin (MuJoCo size is half-extent, so this is (half_depth - 0.05))
    
    // Robot offset for close contact pushing (matching original implementation)
    double offset = robot_size_[0] + 0.05;
    
    // Generate 12 edge points exactly like older implementation
    std::array<std::array<double, 2>, 12> local_edge_points = {{
        {{x - w, y + d + offset}}, {{x - w, y - d - offset}},
        {{x, y + d + offset}}, {{x, y - d - offset}},
        {{x + w, y + d + offset}}, {{x + w, y - d - offset}},
        {{x + w + offset, y - d}}, {{x - w - offset, y - d}},
        {{x + w + offset, y}}, {{x - w - offset, y}},
        {{x + w + offset, y + d}}, {{x - w - offset, y + d}}
    }};

    // std::cout << "Local edge points: " << std::endl;
    // for (size_t i = 0; i < 12; ++i) {
    //     std::cout << "  " << local_edge_points[i][0] << ", " << local_edge_points[i][1] << std::endl;
    // }

    // Transform edge points from local to world coordinates
    edge_count = 12;
    for (size_t i = 0; i < 12; ++i) {
        edge_points[i] = transform_point(local_edge_points[i], obj_pos, yaw);
    }

    // std::cout << "Edge points: " << std::endl;
    // for (size_t i = 0; i < 12; ++i) {
    //     std::cout << "  " << edge_points[i][0] << ", " << edge_points[i][1] << std::endl;
    // }

    // Calculate mid points between consecutive edge points (matching original implementation)
    std::array<std::array<double, 2>, 12> local_mid_points;
    for (size_t i = 0; i < 12; ++i) {
        if (i % 2 == 0) {
            // Even indices: mid point between current and next edge point
            size_t next_idx = (i + 1) % 12;
            local_mid_points[i][0] = (local_edge_points[i][0] + local_edge_points[next_idx][0]) / 2.0;
            local_mid_points[i][1] = (local_edge_points[i][1] + local_edge_points[next_idx][1]) / 2.0;
        } else {
            // Odd indices: mid point between current and previous edge point
            size_t prev_idx = (i - 1);
            local_mid_points[i][0] = (local_edge_points[i][0] + local_edge_points[prev_idx][0]) / 2.0;
            local_mid_points[i][1] = (local_edge_points[i][1] + local_edge_points[prev_idx][1]) / 2.0;
        }
    }

    // std::cout << "Local mid points: " << std::endl;
    // for (size_t i = 0; i < 12; ++i) {
    //     std::cout << "  " << local_mid_points[i][0] << ", " << local_mid_points[i][1] << std::endl;
    // }

    // Transform mid points to world coordinates
    mid_count = 12;
    for (size_t i = 0; i < 12; ++i) {
        mid_points[i] = transform_point(local_mid_points[i], obj_pos, yaw);
    }

    // std::cout << "Mid points: " << std::endl;
    // for (size_t i = 0; i < 12; ++i) {
    //     std::cout << "  " << mid_points[i][0] << ", " << mid_points[i][1] << std::endl;
    // }
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
    
    // Apply scaling factor (matching older implementation's approach)
    double scaling = force_scaling_; // Use configured scaling
    return {
        scaling * std::cos(angle),
        scaling * std::sin(angle)
    };
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

bool NAMOPushController::execute_push_primitive(const std::string& object_name,
                                               int edge_idx,
                                               int push_steps) {
    
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
    
    // Position robot at the edge point
    std::array<double, 2> robot_pos = {push_state.initial_edge_point[0], push_state.initial_edge_point[1]};
    env_.set_robot_position(robot_pos);
    env_.set_zero_velocity();
    env_.step_simulation();
    
    // Check for robot collision with static objects (walls) after positioning
    const auto& static_objects = env_.get_static_objects();
    size_t num_static = env_.get_num_static();
    
    for (size_t i = 0; i < num_static; i++) {
        const auto& static_obj = static_objects[i];
        if (env_.bodies_in_collision("robot", static_obj.name)) {
            // std::cerr << "Robot collision detected with static object '" << static_obj.name 
            //           << "' at edge point [" << robot_pos[0] << ", " << robot_pos[1] 
            //           << "] for object: " << object_name << std::endl;
            return false;  // Fail the primitive execution due to collision
        }
    }
    
    // Check for robot collision with movable objects after positioning
    const auto& movable_objects = env_.get_movable_objects();
    size_t num_movable = env_.get_num_movable();
    
    for (size_t i = 0; i < num_movable; i++) {
        const auto& movable_obj = movable_objects[i];
        // Skip collision check with the object we're trying to push (expected contact)
        if (movable_obj.name != object_name && env_.bodies_in_collision("robot", movable_obj.name)) {
            // std::cerr << "Robot collision detected with movable object '" << movable_obj.name 
            //           << "' at edge point [" << robot_pos[0] << ", " << robot_pos[1] 
            //           << "] for object: " << object_name << std::endl;
            return false;  // Fail the primitive execution due to collision
        }
    }
    
    
    // auto obj_state_initial = env_.get_object_state(object_name);
    // auto robot_state_initial = env_.get_robot_state();
    // if (obj_state_initial && robot_state_initial) {
    //     double dx = robot_state_initial->position[0] - obj_state_initial->position[0];
    //     double dy = robot_state_initial->position[1] - obj_state_initial->position[1];
    //     double distance = sqrt(dx*dx + dy*dy);
    //     double robot_radius = robot_size_[0];
    //     double obj_half_size = sqrt(obj_state_initial->size[0]*obj_state_initial->size[0] + 
    //                                obj_state_initial->size[1]*obj_state_initial->size[1]);
    //     double surface_distance = distance - robot_radius - obj_half_size;
    // }
    
    // Execute push steps
    for (int step = 0; step < push_steps; ++step) {
        // Get current object state
        auto obj_state = env_.get_object_state(object_name);
        if (!obj_state) {
            std::cerr << "Lost object state during push execution" << std::endl;
            return false;
        }
        
        // Update push state based on current object position
        update_push_state(push_state, obj_state->position, obj_state->size, obj_state->quaternion);
        
        // Apply control for multiple simulation steps (matching original control loop structure)
        for (int ctrl_step = 0; ctrl_step < control_steps_per_push_; ++ctrl_step) {
            // Get current object state - CRITICAL: update every control step like original
            obj_state = env_.get_object_state(object_name);
            if (!obj_state) {
                std::cerr << "Lost object state during control step" << std::endl;
                return false;
            }
            
            // Update push state with current object position (matching original lines 80-81, 90-91)
            update_push_state(push_state, obj_state->position, obj_state->size, obj_state->quaternion);
            
            // Get current robot position
            auto robot_state = env_.get_robot_state();
            
            // Compute control forces based on updated push state
            auto control = compute_push_control(push_state);
            
            // Debug output for first control step of each push step
            if (ctrl_step == 0) {
                // std::cout << "  Step " << step << ": Robot at [" 
                          // << (robot_state ? robot_state->position[0] : 0.0) << ", "
                          // << (robot_state ? robot_state->position[1] : 0.0) 
                          // << "], Object at [" << obj_state->position[0] << ", " << obj_state->position[1]
                          // << "], Control: [" << control[0] << ", " << control[1] << "]" << std::endl;
            }
            
            // Apply control through environment dynamics system
            env_.apply_control(control[0], control[1], 0.01);  // 0.01 second timestep
        }
        
        // Reset velocities between push steps
        env_.set_zero_velocity();
        env_.step_simulation();
    }
    
    auto final_obj_state = env_.get_object_state(object_name);
    if (final_obj_state) {
        // std::cout << "Push completed. Object moved to: [" 
                  // << final_obj_state->position[0] << ", " << final_obj_state->position[1] << "]" << std::endl;
    }
    
    return true;
}

bool NAMOPushController::execute_action(const NAMOAction& action) {
    return execute_push_primitive(action.object_name, action.edge_idx, action.push_steps);
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