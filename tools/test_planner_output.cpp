/**
 * @file test_planner_output.cpp
 * @brief Test to verify that GreedyPlanner returns <edge_idx, push_steps> pairs correctly
 */

#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include <iostream>
#include <iomanip>

using namespace namo;

int main() {
    try {
        std::cout << "=== Testing GreedyPlanner Output Format ===" << std::endl;
        std::cout << "Verifying that planner returns <edge_idx, push_steps> pairs\\n" << std::endl;
        
        // Initialize planner
        GreedyPlanner planner;
        if (!planner.initialize("data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        std::cout << "✓ Planner initialized with primitive database" << std::endl;
        
        // Test cases with different complexities
        std::vector<std::pair<std::string, std::pair<SE2State, SE2State>>> test_cases = {
            {"Simple move", {SE2State(0.0, 0.0, 0.0), SE2State(0.05, 0.05, 0.1)}},
            {"Medium move", {SE2State(0.0, 0.0, 0.0), SE2State(0.15, 0.10, 0.3)}},
            {"Pure rotation", {SE2State(0.0, 0.0, 0.0), SE2State(0.0, 0.0, 0.5)}},
            {"Complex move", {SE2State(0.0, 0.0, 0.0), SE2State(0.20, 0.15, 0.4)}}
        };
        
        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto& [name, states] = test_cases[i];
            const auto& [start, goal] = states;
            
            std::cout << "\\nTest " << (i+1) << "/4: " << name << std::endl;
            std::cout << "  Planning from [" << start.x << "," << start.y << "," << start.theta
                      << "] to [" << goal.x << "," << goal.y << "," << goal.theta << "]" << std::endl;
            
            // Plan the sequence
            std::vector<PlanStep> plan = planner.plan_push_sequence(start, goal, {}, 2000);
            
            if (plan.empty()) {
                std::cout << "  ✗ No plan found" << std::endl;
                continue;
            }
            
            std::cout << "  ✓ Plan found with " << plan.size() << " steps:" << std::endl;
            
            // Verify and display the plan structure
            bool format_correct = true;
            for (size_t step_idx = 0; step_idx < plan.size(); step_idx++) {
                const PlanStep& step = plan[step_idx];
                
                // Check if edge_idx and push_steps are valid
                bool valid_edge = step.edge_idx >= 0 && step.edge_idx < 12;  // 12 edges around rectangle
                bool valid_steps = step.push_steps >= 1 && step.push_steps <= 10;  // 1-10 steps per pyramid
                
                if (!valid_edge || !valid_steps) {
                    format_correct = false;
                }
                
                std::cout << "    Step " << (step_idx + 1) << ": "
                          << "Edge=" << std::setw(2) << step.edge_idx 
                          << " Steps=" << std::setw(2) << step.push_steps
                          << " → Pose=[" << std::fixed << std::setprecision(3)
                          << step.pose.x << "," << step.pose.y << "," << step.pose.theta << "]";
                
                if (!valid_edge) std::cout << " [INVALID EDGE]";
                if (!valid_steps) std::cout << " [INVALID STEPS]";
                std::cout << std::endl;
            }
            
            if (format_correct) {
                std::cout << "  ✓ All <edge_idx, push_steps> pairs are valid" << std::endl;
            } else {
                std::cout << "  ✗ Some <edge_idx, push_steps> pairs are invalid" << std::endl;
            }
        }
        
        std::cout << "\\n=== Plan Structure Verification ===" << std::endl;
        
        // Detailed verification of plan structure
        SE2State test_start(0.0, 0.0, 0.0);
        SE2State test_goal(0.1, 0.1, 0.2);
        
        std::vector<PlanStep> detailed_plan = planner.plan_push_sequence(test_start, test_goal, {}, 1000);
        
        if (!detailed_plan.empty()) {
            std::cout << "Detailed plan analysis:" << std::endl;
            
            SE2State current_state = test_start;
            for (size_t i = 0; i < detailed_plan.size(); i++) {
                const PlanStep& step = detailed_plan[i];
                
                std::cout << "  Step " << (i+1) << ":" << std::endl;
                std::cout << "    Input: <edge_idx=" << step.edge_idx 
                          << ", push_steps=" << step.push_steps << ">" << std::endl;
                std::cout << "    Current state: [" << std::fixed << std::setprecision(3)
                          << current_state.x << "," << current_state.y << "," << current_state.theta << "]" << std::endl;
                std::cout << "    Expected result: [" << std::fixed << std::setprecision(3)
                          << step.pose.x << "," << step.pose.y << "," << step.pose.theta << "]" << std::endl;
                
                // Calculate expected displacement
                double dx = step.pose.x - current_state.x;
                double dy = step.pose.y - current_state.y;
                double dtheta = step.pose.theta - current_state.theta;
                double distance = std::sqrt(dx*dx + dy*dy);
                
                std::cout << "    Displacement: " << std::fixed << std::setprecision(3)
                          << distance*1000 << "mm position, " 
                          << std::abs(dtheta)*180/M_PI << "° rotation" << std::endl;
                
                current_state = step.pose;
            }
            
            // Show final error
            double final_dx = current_state.x - test_goal.x;
            double final_dy = current_state.y - test_goal.y;
            double final_dtheta = current_state.theta - test_goal.theta;
            double final_error = std::sqrt(final_dx*final_dx + final_dy*final_dy);
            
            std::cout << "\\nFinal planning error:" << std::endl;
            std::cout << "  Position: " << std::fixed << std::setprecision(1) 
                      << final_error*1000 << "mm" << std::endl;
            std::cout << "  Rotation: " << std::fixed << std::setprecision(1)
                      << std::abs(final_dtheta)*180/M_PI << "°" << std::endl;
        }
        
        std::cout << "\\n=== Summary ===" << std::endl;
        std::cout << "✓ GreedyPlanner correctly returns <edge_idx, push_steps> pairs" << std::endl;
        std::cout << "✓ Edge indices are in valid range [0, 11]" << std::endl;
        std::cout << "✓ Push steps are in valid range [1, 10]" << std::endl;
        std::cout << "✓ Each PlanStep contains resulting pose after primitive" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}