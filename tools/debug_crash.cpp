/**
 * @file debug_crash.cpp
 * @brief Debug segfault by adding components one at a time
 */

#include "environment/namo_environment.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/mpc_executor.hpp"
#include "planning/namo_push_controller.hpp"
#include "planning/incremental_wavefront_planner.hpp"
#include <iostream>
#include <memory>

using namespace namo;

int main() {
    try {
        std::cout << "=== Debug Crash Test ===" << std::endl;
        
        std::cout << "Step 1: Creating NAMOEnvironment..." << std::endl;
        NAMOEnvironment env("data/nominal_primitive_scene.xml", false);
        std::cout << "✓ NAMOEnvironment created" << std::endl;
        
        std::cout << "Step 2: Creating GreedyPlanner..." << std::endl;
        GreedyPlanner planner;
        if (!planner.initialize("data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        std::cout << "✓ GreedyPlanner created" << std::endl;
        
        std::cout << "Step 3: Creating MPCExecutor..." << std::endl;
        MPCExecutor executor(env);
        std::cout << "✓ MPCExecutor created" << std::endl;
        
        std::cout << "Step 4: Creating IncrementalWavefrontPlanner..." << std::endl;
        std::vector<double> robot_size = {0.15, 0.15};
        auto wavefront_planner = std::make_unique<IncrementalWavefrontPlanner>(0.02, env, robot_size);
        std::cout << "✓ IncrementalWavefrontPlanner created" << std::endl;
        
        std::cout << "Step 5: Creating NAMOPushController..." << std::endl;
        auto controller_ptr = std::make_unique<NAMOPushController>(env, *wavefront_planner, 10, 250, 1.0);
        std::cout << "✓ NAMOPushController created" << std::endl;
        
        std::cout << "Step 6: Getting movable objects..." << std::endl;
        auto movable_objects = env.get_movable_objects();
        std::string object_name = movable_objects[0].name;
        std::cout << "✓ Object name: " << object_name << std::endl;
        
        std::cout << "Step 7: Testing get_reachable_edge_indices..." << std::endl;
        std::vector<int> reachable_edges = controller_ptr->get_reachable_edge_indices(object_name);
        std::cout << "✓ get_reachable_edge_indices completed: " << reachable_edges.size() << " edges" << std::endl;
        
        std::cout << "🎉 All steps completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}