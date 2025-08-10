#include "planners/high_level_planner.hpp"
#include "planners/strategies/random_selection_strategy.hpp"
#include <chrono>
#include <iostream>

namespace namo {

HighLevelPlanner::HighLevelPlanner(NAMOEnvironment& env, 
                                  std::unique_ptr<SelectionStrategy> strategy,
                                  std::shared_ptr<ConfigManager> config)
    : environment_(env)
    , strategy_(std::move(strategy))
    , config_(config ? config : std::shared_ptr<ConfigManager>(ConfigManager::create_default().release()))
{
    // Create separate wavefront planner for high-level reachability analysis
    // This is independent of the skill's internal planner
    high_level_wavefront_ = std::make_unique<WavefrontPlanner>(
        0.02, environment_, config_->get_robot_size()  // Force 0.02m resolution
    );
    
    // Create push skill for object manipulation with configuration
    push_skill_ = std::make_unique<NAMOPushSkill>(environment_, config_);
    
    // Initialize wavefront with current environment state
    updateHighLevelWavefront();
}

HighLevelPlanner::~HighLevelPlanner() = default;

PlanningResult HighLevelPlanner::planToGoal(const SE2State& robot_goal, int max_iterations) {
    auto start_time = std::chrono::high_resolution_clock::now();
    PlanningResult result;
    
    robot_goal_ = robot_goal;
    execution_log_.clear();
    
    // Use configuration default if max_iterations not specified
    int actual_max_iterations = (max_iterations > 0) ? max_iterations : config_->planning().max_planning_iterations;
    
    if (config_->planning().verbose_planning) {
        std::cout << "Starting high-level planning to goal: (" 
                  << robot_goal.x << ", " << robot_goal.y << ", " << robot_goal.theta << ")\n";
        std::cout << "Max iterations: " << actual_max_iterations << std::endl;
    }
    
    // Main planning loop (based on legacy interface_namo.cpp:303-707)
    for (int iteration = 0; iteration < actual_max_iterations; ++iteration) {
        // Update high-level wavefront with current object positions
        updateHighLevelWavefront();
        
        // Check if robot goal is already reachable
        if (isRobotGoalReachable()) {
            result.success = true;
            result.failure_reason = "Goal reached";
            computeReachableObjects();
            if (config_->planning().verbose_planning) {
                std::cout << "Robot goal became reachable after " << iteration << " iterations\n";
            }
            break;
        }
        
        // Compute reachable objects using high-level wavefront
        std::vector<std::string> reachable_objects = computeReachableObjects();
        
        if (reachable_objects.empty()) {
            result.failure_reason = "No reachable objects to manipulate";
            if (config_->planning().verbose_planning) {
                std::cout << "No reachable objects found at iteration " << iteration << "\n";
            }
            break;
        }
        
        // Strategy selects object and goal
        SelectionResult selection = strategy_->selectObjectAndGoal(
            environment_, reachable_objects, robot_goal_
        );
        
        if (!selection.success) {
            result.failure_reason = "Strategy failed to select valid object and goal";
            if (config_->planning().verbose_planning) {
                std::cout << "Selection strategy failed at iteration " << iteration << "\n";
            }
            break;
        }
        
        // Execute skill to push selected object toward selected goal
        std::map<std::string, SkillParameterValue> skill_params = {
            {"object_name", selection.object_name},
            {"target_pose", selection.target_pose}
        };
        
        SkillResult skill_result = push_skill_->execute(skill_params);
        
        // Log this iteration
        logIteration(iteration, selection, skill_result);
        execution_log_.push_back(selection.object_name);
        
        // if (!skill_result.success) {
        //     result.failure_reason = "Skill execution failed: " + skill_result.failure_reason;
        //     if (config_->planning().verbose_planning) {
        //         std::cout << "Skill execution failed at iteration " << iteration 
        //                   << ": " << skill_result.failure_reason << "\n";
        //     }
        //     break;
        // }
        
        result.iterations_used = iteration + 1;
        
        // Check for early termination if robot goal becomes reachable
        // (This matches legacy behavior where MPC can terminate early)
        // updateHighLevelWavefront();
        // if (isRobotGoalReachable()) {
        //     result.success = true;
        //     result.failure_reason = "Goal reached during execution";
        //     if (config_->planning().verbose_planning) {
        //         std::cout << "Robot goal became reachable during iteration " << iteration << "\n";
        //     }
        //     break;
        // }
    }
    
    if (!result.success && result.failure_reason.empty()) {
        result.failure_reason = "Maximum iterations exceeded";
        result.iterations_used = actual_max_iterations;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(end_time - start_time).count();
    result.objects_pushed = execution_log_;
    
    if (config_->planning().verbose_planning) {
        std::cout << "Planning completed. Success: " << result.success 
                  << ", Iterations: " << result.iterations_used
                  << ", Time: " << result.total_time << "s\n";
    }
    
    return result;
}

std::vector<std::string> HighLevelPlanner::computeReachableObjects() {
    std::vector<std::string> reachable_objects;
    
    // Get robot position
    auto robot_state = environment_.get_robot_state();
    if (!robot_state) {
        return reachable_objects;
    }
    
    // Update high-level wavefront to determine reachability
    std::vector<double> robot_pos;
    robot_pos.push_back(robot_state->position[0]);
    robot_pos.push_back(robot_state->position[1]);
    bool updated = high_level_wavefront_->update_wavefront(environment_, robot_pos);
    
    if (!updated) {
        // If wavefront update failed, return empty list
        return reachable_objects;
    }
    
    // Check each movable object for reachability
    auto movable_objects = environment_.get_movable_objects();
    for (const auto& obj_info : movable_objects) {
        // Get current object state for position and rotation
        const ObjectState* obj_state = environment_.get_object_state(obj_info.name);
        if (!obj_state) continue;
        
        // Sample points around object perimeter to check reachability
        // This matches legacy reachable_points computation
        double obj_x = obj_state->position[0];
        double obj_y = obj_state->position[1];
        double obj_width = obj_state->size[0];
        double obj_height = obj_state->size[1];
        
        // Get object rotation (yaw angle) from quaternion
        double yaw = utils::quaternion_to_yaw(obj_state->quaternion);
        
        // Sample edge points around object (using configuration)
        bool object_reachable = false;
        int num_edge_points = config_->skill().num_edge_points;
        double clearance = config_->skill().object_clearance;
        
        // Generate 12 edge points using rectangular pattern (matching push controller)
        // Object dimensions (MuJoCo sizes are half-extents)
        double w = obj_width - 0.05;   // half-width with margin  
        double d = obj_height - 0.05;  // half-height with margin
        double offset = 0.15 + 0.1; // clearance;     // robot offset for reachability
        
        // 12 rectangular edge points (same pattern as push controller)
        std::array<std::array<double, 2>, 12> local_edge_points = {{
            {{-w, d + offset}}, {{-w, -d - offset}}, 
            {{0, d + offset}}, {{0, -d - offset}}, 
            {{w, d + offset}}, {{w, -d - offset}}, 
            {{w + offset, -d}}, {{-w - offset, -d}}, 
            {{w + offset, 0}}, {{-w - offset, 0}}, 
            {{w + offset, d}}, {{-w - offset, d}}
        }};
        // Check reachability for each rectangular edge point (matching NAMOPushController approach)
        const auto& distance_grid = high_level_wavefront_->get_grid();
        int reachable_edge_count = 0;
        
        for (int edge = 0; edge < 12; ++edge) {
            // Transform local edge point to world coordinates (rotation + translation)
            // Using same transformation approach as NAMOPushController
            double cos_theta = std::cos(yaw);
            double sin_theta = std::sin(yaw);
            
            double local_x = local_edge_points[edge][0];
            double local_y = local_edge_points[edge][1];
            
            // Rotate then translate
            double edge_x = cos_theta * local_x - sin_theta * local_y + obj_x;
            double edge_y = sin_theta * local_x + cos_theta * local_y + obj_y;
            
            // Convert to grid coordinates
            int grid_x = high_level_wavefront_->world_to_grid_x(edge_x);
            int grid_y = high_level_wavefront_->world_to_grid_y(edge_y);
            
            bool edge_reachable = false;
            if (grid_x >= 0 && grid_x < high_level_wavefront_->get_grid_width() &&
                grid_y >= 0 && grid_y < high_level_wavefront_->get_grid_height()) {
                
                // Check for reachable (value = 1), matching NAMOPushController logic
                if (distance_grid[grid_x][grid_y] == 1) {
                    edge_reachable = true;
                    reachable_edge_count++;
                }
                
                // DEBUG: Mark edge point in wavefront for visualization
                // Use -3 for reachable edges, -4 for unreachable edges
                high_level_wavefront_->get_mutable_grid()[grid_x][grid_y] = edge_reachable ? -3 : -4;
            }
        }
        
        // Object is reachable if at least one edge is reachable (matching original logic)
        object_reachable = (reachable_edge_count > 0);
        
        if (object_reachable) {
            reachable_objects.push_back(obj_info.name);
        }
    }
    
    return reachable_objects;
}

bool HighLevelPlanner::isRobotGoalReachable() {
    return high_level_wavefront_->is_goal_reachable({robot_goal_.x, robot_goal_.y});
}

void HighLevelPlanner::updateHighLevelWavefront() {
    // Update wavefront planner with current object positions
    // This triggers change detection and incremental updates
    auto robot_state = environment_.get_robot_state();
    if (robot_state) {
        std::vector<double> robot_pos;
        robot_pos.push_back(robot_state->position[0]);
        robot_pos.push_back(robot_state->position[1]);
        high_level_wavefront_->update_wavefront(environment_, robot_pos);
    }
}

void HighLevelPlanner::logIteration(int iteration, const SelectionResult& selection, 
                                   const SkillResult& result) {
    if (config_->planning().verbose_planning) {
        std::cout << "Iteration " << iteration << ": "
                  << "Object=" << selection.object_name 
                  << ", Goal=(" << selection.target_pose.x << "," << selection.target_pose.y << ")"
                  << ", Success=" << result.success;
        if (result.success) {
            std::cout << ", Duration=" << result.execution_time.count() << "ms";
        } else {
            std::cout << ", Failure=" << result.failure_reason;
        }
        std::cout << "\n";
    }
}

void HighLevelPlanner::setStrategy(std::unique_ptr<SelectionStrategy> strategy) {
    strategy_ = std::move(strategy);
}

void HighLevelPlanner::updateConfig(std::shared_ptr<ConfigManager> new_config) {
    config_ = new_config;
    
    // Recreate wavefront planner with new configuration
    high_level_wavefront_ = std::make_unique<WavefrontPlanner>(
        config_->get_high_level_resolution(), environment_, config_->get_robot_size()
    );
    updateHighLevelWavefront();
}

bool HighLevelPlanner::isGoalReachable(const SE2State& goal) {
    SE2State saved_goal = robot_goal_;
    robot_goal_ = goal;
    bool reachable = isRobotGoalReachable();
    robot_goal_ = saved_goal;
    return reachable;
}

void HighLevelPlanner::reset() {
    execution_log_.clear();
    updateHighLevelWavefront();
}

} // namespace namo