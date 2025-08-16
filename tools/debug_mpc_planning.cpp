/**
 * @file debug_mpc_planning.cpp
 * @brief Debug why iterative MPC fails to find plans from intermediate states
 */

#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace namo;

void analyze_primitive_coverage() {
    // std::cout << "=== Analyzing Primitive Database Coverage ===" << std::endl;
    
    PrimitiveLoader loader;
    if (!loader.load_primitives("data/motion_primitives.dat")) {
        std::cerr << "Failed to load primitives" << std::endl;
        return;
    }
    
    // std::cout << "Loaded " << loader.size() << " primitives" << std::endl;
    
    // Analyze coverage for different directions
    int forward_count = 0, backward_count = 0, left_count = 0, right_count = 0;
    double min_distance = 1000.0, max_distance = 0.0;
    
    // std::cout << "\nPrimitive Analysis:" << std::endl;
    // std::cout << "Edge | Steps | Delta X  | Delta Y  | Delta Θ  | Distance" << std::endl;
    // std::cout << "-----|-------|----------|----------|----------|----------" << std::endl;
    
    const auto& all_primitives = loader.get_all_primitives();
    for (size_t i = 0; i < loader.size(); i++) {
        const LoadedPrimitive& prim = all_primitives[i];
        
        double distance = std::sqrt(prim.delta_x * prim.delta_x + prim.delta_y * prim.delta_y);
        min_distance = std::min(min_distance, distance);
        max_distance = std::max(max_distance, distance);
        
        if (prim.delta_x > 0.01) forward_count++;
        if (prim.delta_x < -0.01) backward_count++;
        if (prim.delta_y > 0.01) right_count++;
        if (prim.delta_y < -0.01) left_count++;
        
        // std::cout << std::setw(4) << (int)prim.edge_idx << " | "
                  // << std::setw(5) << (int)prim.push_steps << " | "
                  // << std::setw(8) << std::fixed << std::setprecision(3) << prim.delta_x << " | "
                  // << std::setw(8) << prim.delta_y << " | "
                  // << std::setw(8) << prim.delta_theta << " | "
                  // << std::setw(8) << distance << std::endl;
    }
    
    // std::cout << "\nCoverage Summary:" << std::endl;
    // std::cout << "  Forward motion (X > 0): " << forward_count << " primitives" << std::endl;
    // std::cout << "  Backward motion (X < 0): " << backward_count << " primitives" << std::endl;
    // std::cout << "  Right motion (Y > 0): " << right_count << " primitives" << std::endl;
    // std::cout << "  Left motion (Y < 0): " << left_count << " primitives" << std::endl;
    // std::cout << "  Distance range: " << std::fixed << std::setprecision(3) 
              // << min_distance << "m to " << max_distance << "m" << std::endl;
    
    if (backward_count == 0) {
        // std::cout << "  ⚠️  NO BACKWARD MOTION PRIMITIVES FOUND!" << std::endl;
        // std::cout << "  ⚠️  This explains why iterative MPC fails!" << std::endl;
    }
}

void test_problematic_local_goal() {
    // std::cout << "\n=== Testing Problematic Local Goal ===" << std::endl;
    
    GreedyPlanner planner;
    if (!planner.initialize("data/motion_primitives.dat")) {
        std::cerr << "Failed to initialize planner" << std::endl;
        return;
    }
    
    // The problematic case from iterative MPC
    SE2State origin(0.0, 0.0, 0.0);
    SE2State problematic_goal(-0.019, 0.047, 0.047);  // Requires backward motion
    
    // std::cout << "Attempting to plan from origin to problematic goal:" << std::endl;
    // std::cout << "  Start: [0.000, 0.000, 0.000]" << std::endl;
    // std::cout << "  Goal:  [" << std::fixed << std::setprecision(3) 
              // << problematic_goal.x << ", " << problematic_goal.y << ", " << problematic_goal.theta << "]" << std::endl;
    
    std::vector<PlanStep> plan = planner.plan_push_sequence(origin, problematic_goal, {}, 5000);
    
    if (plan.empty()) {
        // std::cout << "  ✗ No plan found (as expected)" << std::endl;
        // std::cout << "  ✗ Confirms backward motion coverage gap" << std::endl;
    } else {
        // std::cout << "  ✓ Plan found with " << plan.size() << " steps:" << std::endl;
        for (size_t i = 0; i < plan.size(); i++) {
            // std::cout << "    Step " << (i+1) << ": Edge=" << plan[i].edge_idx 
                      // << " Steps=" << plan[i].push_steps << std::endl;
        }
    }
}

void test_forward_goals() {
    // std::cout << "\n=== Testing Forward Motion Goals ===" << std::endl;
    
    GreedyPlanner planner;
    if (!planner.initialize("data/motion_primitives.dat")) {
        return;
    }
    
    // Test forward motion goals (should work)
    std::vector<SE2State> forward_goals = {
        SE2State(0.05, 0.0, 0.0),    // Small forward
        SE2State(0.1, 0.05, 0.1),    // Medium forward
        SE2State(0.15, 0.0, 0.0),    // Large forward
    };
    
    SE2State origin(0.0, 0.0, 0.0);
    
    for (size_t i = 0; i < forward_goals.size(); i++) {
        const SE2State& goal = forward_goals[i];
        // std::cout << "Test " << (i+1) << ": Forward goal [" 
                  // << std::fixed << std::setprecision(3)
                  // << goal.x << ", " << goal.y << ", " << goal.theta << "]" << std::endl;
        
        std::vector<PlanStep> plan = planner.plan_push_sequence(origin, goal, {}, 2000);
        
        if (plan.empty()) {
            // std::cout << "  ✗ No plan found" << std::endl;
        } else {
            // std::cout << "  ✓ Plan found with " << plan.size() << " steps" << std::endl;
        }
    }
}

void suggest_fixes() {
    // std::cout << "\n=== Suggested Fixes for Iterative MPC ===" << std::endl;
    
    // std::cout << "Problem: Limited primitive database coverage" << std::endl;
    // std::cout << "\nSolution Options:" << std::endl;
    
    // std::cout << "1. **Expand Primitive Database**:" << std::endl;
    // std::cout << "   - Generate primitives for backward edges (edges 6-11)" << std::endl;
    // std::cout << "   - Add more fine-grained primitives (smaller steps)" << std::endl;
    // std::cout << "   - Include negative push directions" << std::endl;
    
    // std::cout << "\n2. **Improve Planning Strategy**:" << std::endl;
    // std::cout << "   - Increase search iterations/expansion limit" << std::endl;
    // std::cout << "   - Use different heuristics for backward motion" << std::endl;
    // std::cout << "   - Allow larger distance thresholds for 'close enough'" << std::endl;
    
    // std::cout << "\n3. **Hybrid Approach**:" << std::endl;
    // std::cout << "   - Fall back to full sequence execution if iterative fails" << std::endl;
    // std::cout << "   - Use different primitive sets for different situations" << std::endl;
    // std::cout << "   - Implement primitive interpolation for missing cases" << std::endl;
    
    // std::cout << "\n4. **Current Workaround**:" << std::endl;
    // std::cout << "   - The full sequence execution works well (51mm error)" << std::endl;
    // std::cout << "   - This is acceptable for many robotic applications" << std::endl;
    // std::cout << "   - Iterative MPC is an optimization, not a requirement" << std::endl;
}

int main() {
    // std::cout << "=== Debugging Iterative MPC Planning Failures ===" << std::endl;
    // std::cout << "Investigating why planner fails from intermediate states\\n" << std::endl;
    
    try {
        analyze_primitive_coverage();
        test_problematic_local_goal();
        test_forward_goals();
        suggest_fixes();
        
        // std::cout << "\n=== Debug Summary ===" << std::endl;
        // std::cout << "✓ Identified root cause: Missing backward motion primitives" << std::endl;
        // std::cout << "✓ Confirmed forward motion planning works" << std::endl;
        // std::cout << "✓ Provided multiple solution paths" << std::endl;
        // std::cout << "⚠️ Current system works for single-shot planning" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}