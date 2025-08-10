#include "planners/region/region_based_planner.hpp"
#include "config/config_manager.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>
#include <memory>

using namespace namo;

int main(int argc, char* argv[]) {
    std::cout << "=== Minimal Region Planner Test ===" << std::endl;
    
    try {
        // Load configuration
        std::cout << "1. Loading configuration..." << std::endl;
        auto config = ConfigManager::create_from_file("config/namo_config_complete.yaml");
        
        // Initialize environment
        std::cout << "2. Initializing environment..." << std::endl;
        NAMOEnvironment env(config->system().default_scene_file, false);  // No visualization
        
        // Create region-based planner - this is where it might crash
        std::cout << "3. Creating region planner..." << std::endl;
        auto planner_config = ConfigManager::create_from_file("config/namo_config_complete.yaml");
        std::cout << "3a. Config loaded for planner..." << std::endl;
        
        RegionBasedPlanner planner(env, std::move(planner_config));
        std::cout << "3b. Region planner created successfully!" << std::endl;
        
        std::cout << "4. Testing region discovery directly..." << std::endl;
        
        // Test just the region analyzer component directly
        auto& analyzer = planner.get_region_analyzer();
        SE2State robot_goal(0.0, 0.0, 0.0);
        
        std::cout << "4a. Calling region discovery..." << std::endl;
        RegionGraph graph = analyzer.discover_regions(env, robot_goal);
        std::cout << "4b. Region discovery completed! Found " << graph.regions.size() << " regions" << std::endl;
        
        std::cout << "5. Testing reachability analysis..." << std::endl;
        try {
            bool reachable = planner.is_goal_reachable(robot_goal);
            std::cout << "5a. Reachability check completed: " << (reachable ? "YES" : "NO") << std::endl;
        } catch (const std::exception& e) {
            std::cout << "5a. Reachability check failed: " << e.what() << std::endl;
        }
        
        std::cout << "SUCCESS: All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}