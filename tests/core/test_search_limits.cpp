/**
 * @file test_search_limits.cpp
 * @brief Test different search limits and parameters for the problematic case
 */

#include "planning/greedy_planner.hpp"
#include <iostream>
#include <iomanip>

using namespace namo;

int main() {
    // std::cout << "=== Testing Search Limits for Problematic Case ===" << std::endl;
    
    GreedyPlanner planner;
    if (!planner.initialize("data/motion_primitives.dat")) {
        std::cerr << "Failed to initialize planner" << std::endl;
        return 1;
    }
    
    // The problematic case from iterative MPC
    SE2State origin(0.0, 0.0, 0.0);
    SE2State problematic_goal(-0.019, 0.047, 0.047);  // Tiny goal requiring precision
    
    // std::cout << "Testing goal: [" << std::fixed << std::setprecision(6)
              // << problematic_goal.x << ", " << problematic_goal.y << ", " << problematic_goal.theta << "]" << std::endl;
    
    // Test with increasing expansion limits
    std::vector<int> limits = {1000, 2000, 5000, 10000, 20000, 50000};
    
    for (int limit : limits) {
        // std::cout << "\\nTesting with expansion limit: " << limit << std::endl;
        
        std::vector<PlanStep> plan = planner.plan_push_sequence(origin, problematic_goal, {}, limit);
        
        if (plan.empty()) {
            // std::cout << "  ✗ Still no plan found" << std::endl;
        } else {
            // std::cout << "  ✓ Plan found with " << plan.size() << " steps!" << std::endl;
            
            // Show the plan
            for (size_t i = 0; i < plan.size(); i++) {
                // std::cout << "    Step " << (i+1) << ": Edge=" << plan[i].edge_idx 
                          // << " Steps=" << plan[i].push_steps 
                          // << " → [" << std::fixed << std::setprecision(3)
                          // << plan[i].pose.x << "," << plan[i].pose.y << "," << plan[i].pose.theta << "]" << std::endl;
            }
            
            // Calculate final error
            const SE2State& final = plan.back().pose;
            double dx = problematic_goal.x - final.x;
            double dy = problematic_goal.y - final.y;
            double dtheta = problematic_goal.theta - final.theta;
            double pos_error = std::sqrt(dx*dx + dy*dy);
            
            // std::cout << "    Final error: " << std::fixed << std::setprecision(1)
                      // << pos_error*1000 << "mm position, " 
                      // << std::abs(dtheta)*180/M_PI << "° rotation" << std::endl;
            break;
        }
    }
    
    // Also test some other small goals
    // std::cout << "\\n=== Testing Other Small Goals ===" << std::endl;
    
    std::vector<SE2State> small_goals = {
        SE2State(0.01, 0.01, 0.01),     // Very small forward
        SE2State(-0.01, 0.01, 0.01),    // Very small backward
        SE2State(0.05, 0.02, 0.05),     // Small mixed
        SE2State(-0.05, -0.02, -0.05),  // Small backward mixed
    };
    
    for (size_t i = 0; i < small_goals.size(); i++) {
        const SE2State& goal = small_goals[i];
        // std::cout << "\\nSmall goal " << (i+1) << ": [" << std::fixed << std::setprecision(3)
                  // << goal.x << ", " << goal.y << ", " << goal.theta << "]" << std::endl;
        
        std::vector<PlanStep> plan = planner.plan_push_sequence(origin, goal, {}, 10000);
        
        if (plan.empty()) {
            // std::cout << "  ✗ No plan found" << std::endl;
        } else {
            // std::cout << "  ✓ Plan found with " << plan.size() << " steps" << std::endl;
        }
    }
    
    // std::cout << "\\n=== Analysis ===" << std::endl;
    // std::cout << "If higher expansion limits help, the issue is search depth" << std::endl;
    // std::cout << "If no limits help, the issue is primitive granularity" << std::endl;
    // std::cout << "The smallest primitive distance is 183mm, but we need 19mm precision" << std::endl;
    
    return 0;
}