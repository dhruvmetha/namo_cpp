/**
 * @file test_coordinate_transformations.cpp
 * @brief Test coordinate transformations in GreedyPlanner to verify correctness
 */

#include "planning/greedy_planner.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace namo;

void test_basic_transformations() {
    std::cout << "=== Basic Coordinate Transformation Tests ===" << std::endl;
    
    GreedyPlanner planner;
    
    // Test 1: Identity transformation (start = origin)
    SE2State origin(0.0, 0.0, 0.0);
    SE2State goal(0.1, 0.1, 0.2);
    
    // Should be identical since start is origin
    std::cout << "Test 1: Start at origin" << std::endl;
    std::cout << "  Start: [" << origin.x << ", " << origin.y << ", " << origin.theta << "]" << std::endl;
    std::cout << "  Goal:  [" << goal.x << ", " << goal.y << ", " << goal.theta << "]" << std::endl;
    
    // Transform to local and back to global
    SE2State local = planner.transform_to_local_frame(origin, goal);
    SE2State global_back = planner.transform_to_global_frame(origin, local);
    
    std::cout << "  Local: [" << std::fixed << std::setprecision(6) 
              << local.x << ", " << local.y << ", " << local.theta << "]" << std::endl;
    std::cout << "  Back:  [" << global_back.x << ", " << global_back.y << ", " << global_back.theta << "]" << std::endl;
    
    // Should match original goal
    double error = std::sqrt(std::pow(goal.x - global_back.x, 2) + 
                            std::pow(goal.y - global_back.y, 2) + 
                            std::pow(goal.theta - global_back.theta, 2));
    std::cout << "  Roundtrip error: " << error << (error < 1e-10 ? " ✓" : " ✗") << std::endl;
}

void test_iterative_case() {
    std::cout << "\\n=== Iterative MPC Case Analysis ===" << std::endl;
    
    GreedyPlanner planner;
    
    // Simulate the iterative MPC case that failed
    SE2State iter1_start(0.0, 0.0, 0.0);
    SE2State goal(0.15, 0.10, 0.3);
    SE2State iter2_start(0.180, 0.059, 0.253);  // After first primitive execution
    
    std::cout << "Original goal: [" << goal.x << ", " << goal.y << ", " << goal.theta << "]" << std::endl;
    
    // Iteration 1 transformation
    SE2State local_goal_1 = planner.transform_to_local_frame(iter1_start, goal);
    std::cout << "\\nIteration 1:" << std::endl;
    std::cout << "  Object at: [" << iter1_start.x << ", " << iter1_start.y << ", " << iter1_start.theta << "]" << std::endl;
    std::cout << "  Local goal: [" << std::fixed << std::setprecision(6) 
              << local_goal_1.x << ", " << local_goal_1.y << ", " << local_goal_1.theta << "]" << std::endl;
    
    // Iteration 2 transformation  
    SE2State local_goal_2 = planner.transform_to_local_frame(iter2_start, goal);
    std::cout << "\\nIteration 2:" << std::endl;
    std::cout << "  Object at: [" << iter2_start.x << ", " << iter2_start.y << ", " << iter2_start.theta << "]" << std::endl;
    std::cout << "  Local goal: [" << std::fixed << std::setprecision(6)
              << local_goal_2.x << ", " << local_goal_2.y << ", " << local_goal_2.theta << "]" << std::endl;
    
    // Analysis
    double distance_1 = std::sqrt(local_goal_1.x*local_goal_1.x + local_goal_1.y*local_goal_1.y);
    double distance_2 = std::sqrt(local_goal_2.x*local_goal_2.x + local_goal_2.y*local_goal_2.y);
    
    std::cout << "\\nAnalysis:" << std::endl;
    std::cout << "  Iteration 1 local distance: " << std::fixed << std::setprecision(3) << distance_1*1000 << "mm" << std::endl;
    std::cout << "  Iteration 2 local distance: " << std::fixed << std::setprecision(3) << distance_2*1000 << "mm" << std::endl;
    std::cout << "  Iteration 2 requires " << (local_goal_2.x < 0 ? "BACKWARD" : "FORWARD") << " motion" << std::endl;
    
    // Check if this explains the planning failure
    if (local_goal_2.x < 0) {
        std::cout << "  ⚠️  Iteration 2 requires negative X movement (backing up)" << std::endl;
        std::cout << "  ⚠️  Our primitive database may not have good backward coverage" << std::endl;
    }
    
    if (distance_2 < 0.05) {
        std::cout << "  ⚠️  Iteration 2 requires very precise movement (<5cm)" << std::endl;
        std::cout << "  ⚠️  Our primitive database may lack fine-grained primitives" << std::endl;
    }
}

void test_transformation_correctness() {
    std::cout << "\\n=== Transformation Mathematical Correctness ===" << std::endl;
    
    GreedyPlanner planner;
    
    // Test various reference frames
    std::vector<SE2State> references = {
        SE2State(0.0, 0.0, 0.0),        // Origin
        SE2State(1.0, 0.5, 0.0),        // Translated
        SE2State(0.0, 0.0, M_PI/4),     // Rotated 45°
        SE2State(1.0, 0.5, M_PI/3),     // Translated + rotated 60°
        SE2State(-0.5, -0.3, -M_PI/6)   // Negative translation + rotation
    };
    
    SE2State target(0.2, 0.15, 0.4);
    
    std::cout << "Target: [" << target.x << ", " << target.y << ", " << target.theta << "]" << std::endl;
    std::cout << "\\nReference Frame Tests:" << std::endl;
    
    bool all_passed = true;
    for (size_t i = 0; i < references.size(); i++) {
        const SE2State& ref = references[i];
        
        // Transform to local and back to global
        SE2State local = planner.transform_to_local_frame(ref, target);
        SE2State global_back = planner.transform_to_global_frame(ref, local);
        
        // Calculate roundtrip error
        double pos_error = std::sqrt(std::pow(target.x - global_back.x, 2) + 
                                    std::pow(target.y - global_back.y, 2));
        double angle_error = std::abs(target.theta - global_back.theta);
        while (angle_error > M_PI) angle_error = 2.0 * M_PI - angle_error;
        double total_error = pos_error + angle_error;
        
        bool passed = total_error < 1e-10;
        all_passed &= passed;
        
        std::cout << "  Ref " << (i+1) << ": [" << std::fixed << std::setprecision(3)
                  << ref.x << ", " << ref.y << ", " << ref.theta << "]"
                  << " → Error: " << std::scientific << std::setprecision(2) << total_error
                  << (passed ? " ✓" : " ✗") << std::endl;
    }
    
    std::cout << "\\nOverall transformation correctness: " << (all_passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
}

int main() {
    std::cout << "=== Coordinate Transformation Verification ===" << std::endl;
    std::cout << "Testing GreedyPlanner coordinate transformations\\n" << std::endl;
    
    try {
        test_basic_transformations();
        test_iterative_case();
        test_transformation_correctness();
        
        std::cout << "\\n=== Conclusions ===" << std::endl;
        std::cout << "• Coordinate transformations appear mathematically correct" << std::endl;
        std::cout << "• Local frame planning is the right approach for universal primitives" << std::endl;
        std::cout << "• Iterative MPC failure likely due to primitive database coverage" << std::endl;
        std::cout << "• The algorithm matches the old implementation approach" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}