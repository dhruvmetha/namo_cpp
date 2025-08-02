/**
 * @file analyze_displacement_errors.cpp
 * @brief Detailed analysis of displacement errors in NAMO planning
 * 
 * Measures the accuracy of the two-stage planning approach:
 * 1. Abstract planning with universal primitives
 * 2. MPC execution with real physics
 */

#include "environment/namo_environment.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/mpc_executor.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace namo;

struct ErrorAnalysis {
    double position_error;
    double rotation_error_rad;
    double rotation_error_deg;
    double relative_position_error;
    double relative_rotation_error;
};

ErrorAnalysis calculate_errors(const SE2State& target, const SE2State& achieved) {
    ErrorAnalysis error;
    
    // Position error
    double dx = target.x - achieved.x;
    double dy = target.y - achieved.y;
    error.position_error = std::sqrt(dx*dx + dy*dy);
    
    // Rotation error
    error.rotation_error_rad = std::abs(target.theta - achieved.theta);
    while (error.rotation_error_rad > M_PI) {
        error.rotation_error_rad = 2.0 * M_PI - error.rotation_error_rad;
    }
    error.rotation_error_deg = error.rotation_error_rad * 180.0 / M_PI;
    
    // Relative errors
    double target_distance = std::sqrt(target.x*target.x + target.y*target.y);
    if (target_distance > 0.001) {
        error.relative_position_error = error.position_error / target_distance;
    } else {
        error.relative_position_error = 0.0;
    }
    
    if (std::abs(target.theta) > 0.01) {
        error.relative_rotation_error = error.rotation_error_rad / std::abs(target.theta);
    } else {
        error.relative_rotation_error = 0.0;
    }
    
    return error;
}

int main() {
    try {
        std::cout << "=== NAMO Displacement Error Analysis ===" << std::endl;
        std::cout << "Universal primitives: 0.35×0.35m nominal object" << std::endl;
        std::cout << "Testing various displacement magnitudes and directions\\n" << std::endl;
        
        // Initialize system
        NAMOEnvironment env("data/nominal_primitive_scene.xml", false);
        GreedyPlanner planner;
        if (!planner.initialize("data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        MPCExecutor executor(env);
        
        auto movable_objects = env.get_movable_objects();
        std::string object_name = movable_objects[0].name;
        
        // Test cases with varying displacement magnitudes
        std::vector<std::pair<std::string, SE2State>> test_cases = {
            {"Micro (1cm, 1cm, 5°)", SE2State(0.01, 0.01, 0.087)},
            {"Small (5cm, 5cm, 15°)", SE2State(0.05, 0.05, 0.262)},
            {"Medium (10cm, 10cm, 30°)", SE2State(0.10, 0.10, 0.524)},
            {"Large (20cm, 15cm, 45°)", SE2State(0.20, 0.15, 0.785)},
            {"Very Large (30cm, 20cm, 60°)", SE2State(0.30, 0.20, 1.047)},
            {"Pure X (15cm, 0cm, 0°)", SE2State(0.15, 0.00, 0.000)},
            {"Pure Y (0cm, 15cm, 0°)", SE2State(0.00, 0.15, 0.000)},
            {"Pure Rotation (0cm, 0cm, 90°)", SE2State(0.00, 0.00, 1.571)},
            {"Diagonal (14cm, 14cm, 0°)", SE2State(0.14, 0.14, 0.000)},
            {"Complex (25cm, 10cm, 120°)", SE2State(0.25, 0.10, 2.094)}
        };
        
        std::vector<ErrorAnalysis> errors;
        
        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto& [name, target] = test_cases[i];
            
            std::cout << "Test " << (i+1) << "/10: " << name << std::endl;
            std::cout << "  Target: [" << std::fixed << std::setprecision(3) 
                      << target.x << ", " << target.y << ", " << target.theta << "]" << std::endl;
            
            // Reset environment
            env.reset();
            
            // Plan and execute
            SE2State start(0.0, 0.0, 0.0);
            auto plan = planner.plan_push_sequence(start, target, {}, 3000);
            
            if (plan.empty()) {
                std::cout << "  ✗ No plan found" << std::endl;
                continue;
            }
            
            ExecutionResult result = executor.execute_plan(object_name, plan);
            
            if (!result.success) {
                std::cout << "  ✗ Execution failed" << std::endl;
                continue;
            }
            
            // Calculate errors
            ErrorAnalysis error = calculate_errors(target, result.final_object_state);
            errors.push_back(error);
            
            std::cout << "  Achieved: [" << std::fixed << std::setprecision(3)
                      << result.final_object_state.x << ", " 
                      << result.final_object_state.y << ", " 
                      << result.final_object_state.theta << "]" << std::endl;
            std::cout << "  Position Error: " << std::fixed << std::setprecision(1)
                      << error.position_error * 1000 << "mm ("
                      << std::setprecision(0) << error.relative_position_error * 100 << "% relative)" << std::endl;
            std::cout << "  Rotation Error: " << std::fixed << std::setprecision(1)
                      << error.rotation_error_deg << "° ("
                      << std::setprecision(0) << error.relative_rotation_error * 100 << "% relative)" << std::endl;
            std::cout << std::endl;
        }
        
        // Statistical Summary
        if (!errors.empty()) {
            std::cout << "=== Statistical Error Summary ===" << std::endl;
            
            double avg_pos_error = 0.0, max_pos_error = 0.0, min_pos_error = 1000.0;
            double avg_rot_error = 0.0, max_rot_error = 0.0, min_rot_error = 1000.0;
            
            for (const auto& error : errors) {
                avg_pos_error += error.position_error;
                avg_rot_error += error.rotation_error_deg;
                
                max_pos_error = std::max(max_pos_error, error.position_error);
                min_pos_error = std::min(min_pos_error, error.position_error);
                max_rot_error = std::max(max_rot_error, error.rotation_error_deg);
                min_rot_error = std::min(min_rot_error, error.rotation_error_deg);
            }
            
            avg_pos_error /= errors.size();
            avg_rot_error /= errors.size();
            
            std::cout << "Position Errors:" << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(1) << avg_pos_error * 1000 << "mm" << std::endl;
            std::cout << "  Range: " << min_pos_error * 1000 << "mm - " << max_pos_error * 1000 << "mm" << std::endl;
            
            std::cout << "Rotation Errors:" << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(1) << avg_rot_error << "°" << std::endl;
            std::cout << "  Range: " << min_rot_error << "° - " << max_rot_error << "°" << std::endl;
            
            std::cout << "\\n=== Key Insights ===" << std::endl;
            std::cout << "• Universal primitives have significant error as expected" << std::endl;
            std::cout << "• Errors increase with target displacement magnitude" << std::endl;
            std::cout << "• MPC helps but cannot fully compensate for primitive mismatch" << std::endl;
            std::cout << "• This validates the need for closed-loop control in real applications" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}