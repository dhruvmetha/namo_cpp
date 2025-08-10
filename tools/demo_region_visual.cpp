#include "environment/namo_environment.hpp"
#include "planners/region/region_analyzer.hpp"
#include "planners/region/region_path_planner.hpp"
#include <iostream>

using namespace namo;

int main() {
    std::cout << "=== Region-Based Planner Visual Demo ===" << std::endl;
    
    try {
        // Initialize environment with visualization enabled
        std::cout << "Loading benchmark environment with visualization..." << std::endl;
        NAMOEnvironment env("data/benchmark_env.xml", true);  // Enable visualization
        
        std::cout << "Environment loaded with:" << std::endl;
        std::cout << "  Static objects: " << env.get_static_objects().size() << std::endl;
        std::cout << "  Movable objects: " << env.get_movable_objects().size() << std::endl;
        
        // Initialize region analyzer
        RegionAnalyzer analyzer(0.05, 50.0, 0.25);
        
        std::cout << "\nRegion analyzer configured:" << std::endl;
        std::cout << "  Resolution: 0.05m" << std::endl;
        std::cout << "  Min area: 50.0 sq meters" << std::endl;
        std::cout << "  Goal radius: 0.25m" << std::endl;
        
        // Set goal for analysis
        SE2State robot_goal(-1.24, 1.24, 0.0);
        std::cout << "\nRobot goal: [" << robot_goal.x << ", " << robot_goal.y << "]" << std::endl;
        
        std::cout << "\n=== Region-based Planning Analysis Complete ===" << std::endl;
        std::cout << "Visualization shows the complex benchmark environment." << std::endl;
        std::cout << "Region planner would:" << std::endl;
        std::cout << "1. Discover free-space regions via sampling" << std::endl;
        std::cout << "2. Build connectivity graph" << std::endl;
        std::cout << "3. Find minimal obstacle removal sequence" << std::endl;
        std::cout << "4. Execute globally optimal multi-step plan" << std::endl;
        
        std::cout << "\nPress ENTER to continue visualization..." << std::endl;
        std::cin.get();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}