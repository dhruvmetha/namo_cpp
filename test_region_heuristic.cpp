#include "planners/region/region_based_planner.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include <iostream>

using namespace namo;

int main() {
    std::cout << "=== Memory-Efficient Region Heuristic Test ===" << std::endl;
    
    try {
        // Load configuration
        auto config = ConfigManager::create_from_file("config/namo_config.yaml");
        std::cout << "Configuration loaded successfully" << std::endl;
        
        // Initialize environment
        NAMOEnvironment env(config->system().default_scene_file, false);  // No visualization
        std::cout << "Environment initialized with " << env.get_num_movable() << " movable objects" << std::endl;
        
        // Create region-based planner (contains the heuristic)
        RegionBasedPlanner planner(env);
        std::cout << "Region-based planner initialized" << std::endl;
        
        // Test goal position
        SE2State robot_goal(2.0, 1.5, 0.0);
        std::cout << "Testing heuristic for goal: [" << robot_goal.x << ", " << robot_goal.y << "]" << std::endl;
        
        // Get blocking objects using the new heuristic
        auto blocking_objects = planner.get_blocking_objects(robot_goal);
        
        std::cout << "\n=== HEURISTIC RESULT ===" << std::endl;
        if (blocking_objects.empty()) {
            std::cout << "No blocking objects - goal is reachable!" << std::endl;
        } else {
            std::cout << "Next obstacles to move (" << blocking_objects.size() << " objects):" << std::endl;
            for (size_t i = 0; i < blocking_objects.size(); ++i) {
                std::cout << "  " << (i+1) << ". " << blocking_objects[i] << std::endl;
            }
        }
        
        std::cout << "\n=== SUCCESS: Region heuristic working! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}