/**
 * @file test_primitives_only.cpp
 * @brief Simple test for primitive loading and greedy planning
 */

#include "core/parameter_loader.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include <iostream>
#include <chrono>

using namespace namo;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        // std::cout << "Usage: " << argv[0] << " <primitive_file>" << std::endl;
        return 1;
    }
    
    try {
        std::string primitive_path = argv[1];
        
        // std::cout << "=== NAMO Primitive Loading and Planning Test ===" << std::endl;
        // std::cout << "Primitive file: " << primitive_path << std::endl;
        // std::cout << std::endl;
        
        // Test 1: Primitive Loading
        // std::cout << "--- Test 1: Primitive Loading ---" << std::endl;
        PrimitiveLoader loader;
        
        auto load_start = std::chrono::high_resolution_clock::now();
        bool load_success = loader.load_primitives(primitive_path);
        auto load_end = std::chrono::high_resolution_clock::now();
        
        if (!load_success) {
            std::cerr << "Failed to load primitives" << std::endl;
            return 1;
        }
        
        auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start);
        // std::cout << "✓ Loaded " << loader.size() << " primitives in " << load_time.count() << " μs" << std::endl;
        
        // Test primitive lookup performance
        auto lookup_start = std::chrono::high_resolution_clock::now();
        int lookup_count = 0;
        
        for (int edge = 0; edge < 12; edge++) {
            auto valid_steps = loader.get_valid_steps_for_edge(edge);
            // std::cout << "Edge " << edge << ": " << valid_steps.size() << " valid steps" << std::endl;
            for (int step : valid_steps) {
                try {
                    const LoadedPrimitive& prim = loader.get_primitive(edge, step);
                    lookup_count++;
                } catch (const std::exception& e) {
                    // std::cout << "Error getting primitive for edge " << edge << " step " << step << ": " << e.what() << std::endl;
                }
            }
        }
        
        auto lookup_end = std::chrono::high_resolution_clock::now();
        auto lookup_time = std::chrono::duration_cast<std::chrono::microseconds>(lookup_end - lookup_start);
        
        // std::cout << "✓ Performed " << lookup_count << " lookups in " << lookup_time.count() << " μs" << std::endl;
        if (lookup_count > 0) {
            // std::cout << "  Average lookup time: " << (lookup_time.count() / lookup_count) << " μs" << std::endl;
        } else {
            // std::cout << "  No valid primitives found for lookup test" << std::endl;
        }
        
        // Show sample primitives
        // std::cout << "\nSample primitives:" << std::endl;
        for (int edge = 0; edge < 12; edge += 3) {
            auto valid_steps = loader.get_valid_steps_for_edge(edge);
            if (!valid_steps.empty()) {
                const LoadedPrimitive& prim = loader.get_primitive(edge, valid_steps[0]);
                // std::cout << "  Edge " << edge << ", " << valid_steps[0] << " steps: "
                          // << "Δx=" << prim.delta_x << " Δy=" << prim.delta_y 
                          // << " Δθ=" << prim.delta_theta << std::endl;
            }
        }
        
        // Test 2: Greedy Planning
        // std::cout << "\n--- Test 2: Greedy Planning ---" << std::endl;
        GreedyPlanner planner;
        
        if (!planner.initialize(primitive_path)) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        
        // std::cout << "✓ Planner initialized" << std::endl;
        
        // Test planning with different goals
        std::vector<std::pair<SE2State, SE2State>> test_cases = {
            {SE2State(0.0, 0.0, 0.0), SE2State(0.1, 0.1, 0.2)},     // Small displacement
            {SE2State(0.0, 0.0, 0.0), SE2State(0.3, 0.2, 0.5)},     // Medium displacement
            {SE2State(0.0, 0.0, 0.0), SE2State(0.0, 0.0, 1.57)},    // Pure rotation
        };
        
        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto& [start, goal] = test_cases[i];
            
            // std::cout << "Test case " << (i+1) << ": ["
                      // << start.x << "," << start.y << "," << start.theta << "] → ["
                      // << goal.x << "," << goal.y << "," << goal.theta << "]" << std::endl;
            
            auto plan_start = std::chrono::high_resolution_clock::now();
            std::vector<PlanStep> plan = planner.plan_push_sequence(start, goal, {}, 1000);
            auto plan_end = std::chrono::high_resolution_clock::now();
            
            auto plan_time = std::chrono::duration_cast<std::chrono::microseconds>(plan_end - plan_start);
            
            if (plan.empty()) {
                // std::cout << "  ✗ No plan found in " << plan_time.count() << " μs" << std::endl;
            } else {
                // std::cout << "  ✓ Found plan with " << plan.size() << " steps in " 
                          // << plan_time.count() << " μs" << std::endl;
                
                // Show first few steps
                for (size_t j = 0; j < std::min(plan.size(), size_t(3)); j++) {
                    const PlanStep& step = plan[j];
                    // std::cout << "    Step " << (j+1) << ": Edge " << step.edge_idx 
                              // << ", " << step.push_steps << " steps → ["
                              // << step.pose.x << ", " << step.pose.y << ", " << step.pose.theta << "]" << std::endl;
                }
                if (plan.size() > 3) {
                    // std::cout << "    ... and " << (plan.size() - 3) << " more steps" << std::endl;
                }
            }
        }
        
        // std::cout << "\n🎉 All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}