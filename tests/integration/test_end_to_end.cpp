/**
 * @file test_end_to_end.cpp
 * @brief Complete end-to-end test of the two-stage NAMO planning pipeline
 * 
 * Tests the complete workflow:
 * 1. Load universal primitives from binary database
 * 2. Abstract planning in empty environment (GreedyPlanner)
 * 3. MPC execution with real physics (MPCExecutor)
 * 4. Validation and performance measurement
 */

#include "environment/namo_environment.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/mpc_executor.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace namo;

struct TestCase {
    std::string name;
    SE2State start;
    SE2State goal;
    bool expect_success;
    double max_planning_time_ms;
};

int main() {
    try {
        // std::cout << "=== NAMO Complete Two-Stage Planning Pipeline Test ===" << std::endl;
        // std::cout << "Architecture: Abstract Planning → MPC Execution" << std::endl;
        // std::cout << "Universal primitives: No scaling, MPC handles dynamics\n" << std::endl;
        
        // Test cases
        std::vector<TestCase> test_cases = {
            {"Small displacement", SE2State(0.0, 0.0, 0.0), SE2State(0.1, 0.1, 0.2), true, 1000},
            {"Medium displacement", SE2State(0.0, 0.0, 0.0), SE2State(0.2, 0.15, 0.4), true, 2000},
            {"Pure rotation", SE2State(0.0, 0.0, 0.0), SE2State(0.0, 0.0, 0.8), true, 1500},
            {"Large displacement", SE2State(0.0, 0.0, 0.0), SE2State(0.3, 0.2, 0.6), true, 3000},
        };
        
        // Initialize system
        // std::cout << "--- System Initialization ---" << std::endl;
        
        NAMOEnvironment env("data/nominal_primitive_scene.xml", true);
        // std::cout << "✓ Environment initialized" << std::endl;
        
        GreedyPlanner planner;
        if (!planner.initialize("data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        // std::cout << "✓ Planner initialized with 120 universal primitives" << std::endl;
        
        MPCExecutor executor(env);
        // std::cout << "✓ MPC Executor initialized" << std::endl;
        
        // Get test object
        auto movable_objects = env.get_movable_objects();
        if (movable_objects.empty()) {
            // std::cout << "No movable objects found" << std::endl;
            return 1;
        }
        std::string object_name = movable_objects[0].name;
        // std::cout << "✓ Using test object: " << object_name << std::endl;
        
        // Performance tracking
        int passed = 0, failed = 0;
        double total_planning_time = 0.0;
        double total_execution_time = 0.0;
        
        // std::cout << "\n--- Running End-to-End Test Cases ---" << std::endl;
        
        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto& test = test_cases[i];
            // std::cout << "\nTest " << (i+1) << "/" << test_cases.size() 
                      // << ": " << test.name << std::endl;
            // std::cout << "  Goal: [" << test.start.x << "," << test.start.y << "," << test.start.theta
                      // << "] → [" << test.goal.x << "," << test.goal.y << "," << test.goal.theta << "]" << std::endl;
            
            // Reset environment to clean state
            env.reset();
            
            // Stage 1: Abstract Planning in Empty Environment
            // std::cout << "  Stage 1: Abstract planning in empty environment..." << std::endl;
            auto planning_start = std::chrono::high_resolution_clock::now();
            
            std::vector<PlanStep> plan = planner.plan_push_sequence(
                test.start, test.goal, {}, 5000);
            
            auto planning_end = std::chrono::high_resolution_clock::now();
            double planning_time = std::chrono::duration_cast<std::chrono::microseconds>(
                planning_end - planning_start).count() / 1000.0;
            
            total_planning_time += planning_time;
            
            if (plan.empty()) {
                // std::cout << "  ✗ No plan found in abstract planning stage" << std::endl;
                failed++;
                continue;
            }
            
            // std::cout << "  ✓ Generated plan with " << plan.size() 
                      // << " primitive steps in " << std::fixed << std::setprecision(2) 
                      // << planning_time << " ms" << std::endl;
            
            // Show plan summary
            // std::cout << "    Plan summary: ";
            for (size_t j = 0; j < std::min(plan.size(), size_t(3)); j++) {
                // std::cout << "E" << plan[j].edge_idx << "S" << plan[j].push_steps;
                if (j < std::min(plan.size(), size_t(3)) - 1) std::cout << " → ";
            }
            if (plan.size() > 3) std::cout << " ... (+" << (plan.size()-3) << " more)";
            // std::cout << std::endl;
            
            // Stage 2: MPC Execution with Real Physics
            // std::cout << "  Stage 2: MPC execution with real MuJoCo physics..." << std::endl;
            auto execution_start = std::chrono::high_resolution_clock::now();
            
            ExecutionResult result = executor.execute_plan(object_name, plan);
            
            auto execution_end = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                execution_end - execution_start).count();
            
            total_execution_time += execution_time;
            
            // Evaluate results
            if (result.success) {
                // std::cout << "  ✓ MPC execution successful" << std::endl;
                // std::cout << "    Executed " << result.steps_executed << "/" << plan.size() 
                          // << " primitive steps in " << std::fixed << std::setprecision(0) 
                          // << execution_time << " ms" << std::endl;
                
                if (result.robot_goal_reached) {
                    // std::cout << "    🎯 Robot goal became reachable during execution" << std::endl;
                }
                
                // Show final object state
                // std::cout << "    Final object pose: [" << std::fixed << std::setprecision(3)
                          // << result.final_object_state.x << "," 
                          // << result.final_object_state.y << ","
                          // << result.final_object_state.theta << "]" << std::endl;
                
                passed++;
            } else {
                // std::cout << "  ✗ MPC execution failed: " << result.failure_reason << std::endl;
                // std::cout << "    Executed " << result.steps_executed << "/" << plan.size() 
                          // << " steps before failure" << std::endl;
                failed++;
            }
            
            // Performance summary for this test
            // std::cout << "    Performance: " << std::fixed << std::setprecision(1)
                      // << planning_time << "ms planning + " 
                      // << execution_time << "ms execution = "
                      // << (planning_time + execution_time) << "ms total" << std::endl;
            
            // Pause for visualization
            // std::cout << "    Press Enter to continue to next test..." << std::endl;
            std::cin.get();
        }
        
        // Final Summary
        // std::cout << "\n=== Final Results ===" << std::endl;
        // std::cout << "Test Cases: " << passed << " passed, " << failed << " failed" << std::endl;
        
        if (passed > 0) {
            // std::cout << "Performance Summary:" << std::endl;
            // std::cout << "  Average planning time: " << std::fixed << std::setprecision(1)
                      // << (total_planning_time / passed) << " ms" << std::endl;
            // std::cout << "  Average execution time: " << std::fixed << std::setprecision(1)
                      // << (total_execution_time / passed) << " ms" << std::endl;
            // std::cout << "  Average total time: " << std::fixed << std::setprecision(1)
                      // << ((total_planning_time + total_execution_time) / passed) << " ms" << std::endl;
        }
        
        // std::cout << "\n=== Architecture Validation ===" << std::endl;
        // std::cout << "✓ Universal primitives used without scaling" << std::endl;
        // std::cout << "✓ Fast abstract planning in empty environment" << std::endl;
        // std::cout << "✓ MPC execution handles dynamic discrepancies" << std::endl;
        // std::cout << "✓ Zero-allocation runtime with pre-allocated structures" << std::endl;
        
        if (failed == 0) {
            // std::cout << "\n🏆 ALL TESTS PASSED! Complete pipeline is operational!" << std::endl;
        } else {
            // std::cout << "\n⚠️  Some tests failed, but core functionality is working" << std::endl;
        }
        
        return (failed == 0) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}