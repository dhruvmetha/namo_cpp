#include "planners/strategies/random_selection_strategy.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace namo {

RandomSelectionStrategy::RandomSelectionStrategy(std::shared_ptr<ConfigManager> config) 
    : gen_(rd_())
    , config_(config ? config : std::shared_ptr<ConfigManager>(ConfigManager::create_default().release()))
{
}

void RandomSelectionStrategy::configure(const std::map<std::string, double>& params) {
    // Legacy method for manual parameter override
    // Note: This creates a temporary override - for permanent changes, update the ConfigManager
    // TODO: Consider deprecating this method in favor of config-only approach
    // std::cout << "Warning: Using legacy configure method. Consider using ConfigManager for parameter management." << std::endl;
}

SelectionResult RandomSelectionStrategy::selectObjectAndGoal(
    const NAMOEnvironment& env,
    const std::vector<std::string>& reachable_objects,
    const SE2State& robot_goal
) {
    if (reachable_objects.empty()) {
        return SelectionResult::failure();
    }
    
    // Copy reachable objects for modification (legacy algorithm removes tried objects)
    std::vector<std::string> available_objects = reachable_objects;
    
    // Object selection retry loop (matching legacy interface_namo.cpp:371-410)
    int max_retries = config_->strategy().max_object_retries;
    for (int retry = 0; retry < max_retries && !available_objects.empty(); ++retry) {
        // Select random object
        std::string selected_object = selectRandomObject(available_objects);
        
        // Remove from available list (prevents infinite loops)
        available_objects.erase(
            std::remove(available_objects.begin(), available_objects.end(), selected_object),
            available_objects.end()
        );
        
        // Generate goal for this object (matching legacy set_goal_configuration)
        SE2State target_goal = generateRandomGoal(env, selected_object);
        
        // Validate goal
        if (isGoalValid(env, target_goal)) {
            return SelectionResult(selected_object, target_goal, true);
        }
    }
    
    return SelectionResult::failure();
}

std::string RandomSelectionStrategy::selectRandomObject(const std::vector<std::string>& objects) const {
    std::uniform_int_distribution<> dis(0, objects.size() - 1);
    return objects[dis(gen_)];
}

SE2State RandomSelectionStrategy::generateRandomGoal(const NAMOEnvironment& env, 
                                                    const std::string& object_name) const {
    // Try polar coordinate sampling first (legacy lines 1492-1570)
    SE2State polar_goal = tryPolarGoalGeneration(env, object_name);
    if (polar_goal.x != 0 || polar_goal.y != 0) {  // Valid goal found
        return polar_goal;
    }
    
    // Try cardinal directions as fallback (legacy lines 1571-1587)
    SE2State cardinal_goal = tryCardinalDirections(env, object_name);
    if (cardinal_goal.x != 0 || cardinal_goal.y != 0) {  // Valid goal found
        return cardinal_goal;
    }
    
    // Ultimate fallback: use object position with random orientation (legacy lines 1588-1597)
    auto object_state = env.get_object_state(object_name);
    if (object_state) {
        std::uniform_real_distribution<> angle_dis(-M_PI, M_PI);
        return SE2State(object_state->position[0], object_state->position[1], angle_dis(gen_));
    }
    
    return SE2State(0, 0, 0);  // Should never reach here
}

SE2State RandomSelectionStrategy::tryPolarGoalGeneration(const NAMOEnvironment& env,
                                                        const std::string& object_name) const {
    auto object_state = env.get_object_state(object_name);
    if (!object_state) return SE2State(0, 0, 0);
    
    double obj_x = object_state->position[0];
    double obj_y = object_state->position[1];
    
    // Polar coordinate sampling (legacy interface_namo.cpp:1520-1570)
    double min_dist = config_->strategy().min_goal_distance;
    double max_dist = config_->strategy().max_goal_distance;
    int max_attempts = config_->strategy().max_goal_attempts;
    
    std::uniform_real_distribution<> distance_dis(min_dist, max_dist);
    std::uniform_real_distribution<> angle_dis(-M_PI, M_PI);
    std::uniform_real_distribution<> orientation_dis(-M_PI, M_PI);
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        double distance = distance_dis(gen_);
        double angle = angle_dis(gen_);
        double orientation = orientation_dis(gen_);
        
        double goal_x = obj_x + distance * cos(angle);
        double goal_y = obj_y + distance * sin(angle);
        
        SE2State candidate(goal_x, goal_y, orientation);
        
        if (isGoalValid(env, candidate)) {
            return candidate;
        }
    }
    
    return SE2State(0, 0, 0);  // No valid goal found
}

SE2State RandomSelectionStrategy::tryCardinalDirections(const NAMOEnvironment& env,
                                                       const std::string& object_name) const {
    auto object_state = env.get_object_state(object_name);
    if (!object_state) return SE2State(0, 0, 0);
    
    double obj_x = object_state->position[0];
    double obj_y = object_state->position[1];
    
    // Try 4 cardinal directions at minimum distance (legacy lines 1571-1587)
    std::vector<double> angles = {0.0, M_PI/2, M_PI, 3*M_PI/2};
    std::uniform_real_distribution<> orientation_dis(-M_PI, M_PI);
    double min_dist = config_->strategy().min_goal_distance;
    
    for (double angle : angles) {
        double goal_x = obj_x + min_dist * cos(angle);
        double goal_y = obj_y + min_dist * sin(angle);
        double orientation = orientation_dis(gen_);
        
        SE2State candidate(goal_x, goal_y, orientation);
        
        if (isGoalValid(env, candidate)) {
            return candidate;
        }
    }
    
    return SE2State(0, 0, 0);  // No valid goal found
}

bool RandomSelectionStrategy::isGoalValid(const NAMOEnvironment& env, const SE2State& goal) const {
    // Check environment bounds (legacy bounds checking)
    auto bounds = env.get_environment_bounds();
    
    if (goal.x < bounds[0] || goal.x > bounds[1] ||
        goal.y < bounds[2] || goal.y > bounds[3]) {
        return false;
    }
    
    // Additional validation could be added here:
    // - Collision checking with static obstacles
    // - Reachability validation
    
    return true;
}

} // namespace namo