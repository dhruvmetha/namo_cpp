/**
 * @file test_push_primitive_executor.cpp
 * @brief Simple test for push-primitive executor integration
 */

#include "environment/namo_environment.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/push_primitive_executor.hpp"
#include <iostream>

using namespace namo;

int main() {
    try {
        // std::cout << "=== MPC Executor Integration Test ===" << std::endl;
        
        // Test 1: Environment and Executor Creation
        // std::cout << "\n--- Test 1: Creating Environment and Executor ---" << std::endl;
        
        NAMOEnvironment env("data/nominal_primitive_scene.xml", false);
        // std::cout << "✓ Environment created" << std::endl;
        
        PushPrimitiveExecutor executor(env);
        // std::cout << "✓ MPC Executor created" << std::endl;
        
        // Test 2: Primitive Loading and Planning
        // std::cout << "\n--- Test 2: Loading Primitives and Planning ---" << std::endl;
        
        GreedyPlanner planner;
        if (!planner.initialize("data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        // std::cout << "✓ Planner initialized" << std::endl;
        
        // Test 3: Simple Planning
        SE2State start(0.0, 0.0, 0.0);
        SE2State goal(0.1, 0.1, 0.2);
        
        // std::cout << "Planning from [" << start.x << "," << start.y << "," << start.theta
                  // << "] to [" << goal.x << "," << goal.y << "," << goal.theta << "]" << std::endl;
        
        auto plan = planner.plan_push_sequence(start, goal, {}, 1000);
        
        if (plan.empty()) {
            // std::cout << "No plan found" << std::endl;
            return 1;
        }
        
        // std::cout << "✓ Plan found with " << plan.size() << " steps" << std::endl;
        
        // Test 4: MPC Execution (Basic)
        // std::cout << "\n--- Test 4: MPC Execution ---" << std::endl;
        
        // Get movable objects
        auto movable_objects = env.get_movable_objects();
        if (movable_objects.empty()) {
            // std::cout << "No movable objects found" << std::endl;
            return 1;
        }
        
        std::string object_name = movable_objects[0].name;
        // std::cout << "Using object: " << object_name << std::endl;
        
        // Execute plan (just the first step for testing)
        // std::cout << "Executing first plan step..." << std::endl;
        
        bool success = executor.execute_primitive_step(object_name, plan[0]);
        
        if (success) {
            // std::cout << "✓ MPC execution completed successfully" << std::endl;
        } else {
            // std::cout << "⚠ MPC execution completed with issues" << std::endl;
        }
        
        // std::cout << "\n🎉 All tests completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}