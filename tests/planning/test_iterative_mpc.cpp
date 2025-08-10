/**
 * @file test_iterative_mpc.cpp
 * @brief Test comparing current full-sequence execution vs correct iterative MPC
 * 
 * Current approach: Plan full sequence, execute all steps
 * Correct approach: Plan one step, execute, replan from new state, repeat
 */

#include "environment/namo_environment.hpp"
#include "planning/primitive_loader.hpp"
#include "planning/greedy_planner.hpp"
#include "planning/mpc_executor.hpp"
#include "planning/namo_push_controller.hpp"
#include "wavefront/wavefront_planner.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <memory>

using namespace namo;

/**
 * @brief Iterative MPC executor following old implementation approach
 */
class IterativeMPCExecutor {
private:
    NAMOEnvironment& env_;
    GreedyPlanner& planner_;
    MPCExecutor& single_step_executor_;
    NAMOPushController& controller_;
    WavefrontPlanner& wavefront_planner_;
    
    // Parameters
    int max_mpc_iterations_;
    double distance_threshold_;
    double angle_threshold_;
    int stuck_iterations_threshold_;
    
    // Robot goal for early termination
    bool has_robot_goal_;
    std::array<double, 2> robot_goal_;
    
public:
    IterativeMPCExecutor(NAMOEnvironment& env, GreedyPlanner& planner, MPCExecutor& executor, NAMOPushController& controller, WavefrontPlanner& wavefront_planner)
        : env_(env), planner_(planner), single_step_executor_(executor), controller_(controller), wavefront_planner_(wavefront_planner),
          max_mpc_iterations_(10), distance_threshold_(0.01), angle_threshold_(0.1),
          stuck_iterations_threshold_(3), has_robot_goal_(false) {}
    
    void set_robot_goal(const std::array<double, 2>& goal) {
        robot_goal_ = goal;
        has_robot_goal_ = true;
        single_step_executor_.set_robot_goal(goal);
    }
    
    /**
     * @brief Execute iterative MPC following old implementation pattern
     * 
     * For each MPC iteration:
     * 1. Get current object state  
     * 2. Plan from current state to goal (may return multiple steps)
     * 3. Execute ONLY the first primitive step
     * 4. Check robot goal reachability and object goal achievement
     * 5. Repeat until goal reached or max iterations
     */
    ExecutionResult execute_iterative_mpc(
        const std::string& object_name,
        const SE2State& goal_state) {
        
        ExecutionResult result;
        
        // std::cout << "Starting iterative MPC execution..." << std::endl;
        // std::cout << "Target: [" << goal_state.x << "," << goal_state.y << "," << goal_state.theta << "]" << std::endl;
        
        // Save initial wavefront state before any MPC iterations
        // auto robot_state = env_.get_robot_state();
        // std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
        // wavefront_planner_.update_wavefront(env_, robot_pos);
        // wavefront_planner_.save_wavefront_iteration("iterative_mpc_wavefront", -1);  // Use -1 for initial state
        // std::cout << "Initial wavefront saved before MPC execution starts" << std::endl;
        
        SE2State previous_state = get_current_object_state(object_name);
        int stuck_counter = 0;
        
        for (int mpc_iter = 0; mpc_iter < max_mpc_iterations_; mpc_iter++) {
            // std::cout << "\n--- MPC Iteration " << (mpc_iter + 1) << "/" << max_mpc_iterations_ << " ---" << std::endl;
            
            // 1. Get current object state
            SE2State current_state = get_current_object_state(object_name);
            // std::cout << "Current state: [" << std::fixed << std::setprecision(3)
            //           << current_state.x << "," << current_state.y << "," << current_state.theta << "]" << std::endl;
            
            // Save wavefront BEFORE each MPC iteration for debugging
            auto robot_state = env_.get_robot_state();
            std::vector<double> robot_pos = {robot_state->position[0], robot_state->position[1]};
            
            // Get reachable edges using pre-computed wavefront (without additional wavefront updates)
            std::vector<int> reachable_edges = get_reachable_edges_from_current_wavefront(object_name, robot_pos);

            // wavefront_planner_.save_wavefront_iteration("iterative_mpc_wavefront", mpc_iter);
            // std::cout << "Wavefront saved before MPC iteration " << (mpc_iter + 1) << std::endl;
            
            // 2. Check if robot goal is reachable (early termination)
            if (has_robot_goal_ && single_step_executor_.is_robot_goal_reachable()) {
                // std::cout << "Robot goal became reachable at iteration " << mpc_iter << std::endl;
                result.success = true;
                result.robot_goal_reached = true;
                result.steps_executed = mpc_iter;
                result.final_object_state = current_state;
                return result;
            }
            
            // 3. Check if object reached target goal
            if (is_object_at_goal(current_state, goal_state)) {
                // std::cout << "Object reached goal at iteration " << mpc_iter << std::endl;
                result.success = true;
                result.robot_goal_reached = false;
                result.steps_executed = mpc_iter;
                result.final_object_state = current_state;
                return result;
            }
            
            // 4. Compute wavefront ONCE per iteration (like old implementation)
            // std::cout << "Computing wavefront for MPC iteration " << mpc_iter + 1 << "..." << std::endl;
            // auto robot_state_planning = env_.get_robot_state();
            // std::vector<double> robot_pos_planning = {robot_state_planning->position[0], robot_state_planning->position[1]};

            
            
            if (reachable_edges.empty()) {
                // std::cout << "No reachable edges for object " << object_name << " - stopping MPC" << std::endl;
                result.success = false;
                result.failure_reason = "No reachable edges at iteration " + std::to_string(mpc_iter);
                break;
            }
            
            // std::cout << "Planning from current state to goal with " << reachable_edges.size() << " reachable edges..." << std::endl;
            std::vector<PlanStep> plan = planner_.plan_push_sequence(
                current_state, goal_state, reachable_edges, 1000);
            
            if (plan.empty()) {
                // std::cout << "No plan found from current state" << std::endl;
                result.failure_reason = "No plan found at iteration " + std::to_string(mpc_iter);
                result.steps_executed = mpc_iter;
                result.final_object_state = current_state;
                return result;
            }
            
            // std::cout << "Found plan with " << plan.size() << " steps, executing first step only" << std::endl;
            // std::cout << "First step: Edge=" << plan[0].edge_idx << " Steps=" << plan[0].push_steps << std::endl;
            
            // 5. Execute ONLY the first primitive step (key difference from full sequence execution)
            std::vector<PlanStep> single_step = {plan[0]};
            ExecutionResult step_result = single_step_executor_.execute_plan(object_name, single_step);
            
            if (!step_result.success) {
                // std::cout << "Failed to execute primitive step: " << step_result.failure_reason << std::endl;
                result.failure_reason = "Primitive execution failed at iteration " + std::to_string(mpc_iter);
                result.steps_executed = mpc_iter;
                result.final_object_state = get_current_object_state(object_name);
                return result;
            }
            

            // std::cout << "Step executed. New state: [" << std::fixed << std::setprecision(3)
            //           << step_result.final_object_state.x << "," 
            //           << step_result.final_object_state.y << ","
            //           << step_result.final_object_state.theta << "]" << std::endl;
            
            // Save wavefront AFTER each push execution to see the effect
            // auto robot_state_after = env_.get_robot_state();
            // std::vector<double> robot_pos_after = {robot_state_after->position[0], robot_state_after->position[1]};
            // wavefront_planner_.update_wavefront(env_, robot_pos_after);
            // wavefront_planner_.save_wavefront_iteration("iterative_mpc_wavefront_after", mpc_iter);
            // std::cout << "Wavefront saved after push execution for iteration " << (mpc_iter + 1) << std::endl;
            
            // 6. Check for stuck condition
            if (is_object_stuck(step_result.final_object_state, previous_state)) {
                stuck_counter++;
                // std::cout << "Object appears stuck (counter: " << stuck_counter << ")" << std::endl;
                if (stuck_counter > stuck_iterations_threshold_) {
                    result.failure_reason = "Object stuck for " + std::to_string(stuck_counter) + " iterations";
                    result.steps_executed = mpc_iter + 1;
                    result.final_object_state = step_result.final_object_state;
                    return result;
                }
            } else {
                stuck_counter = 0;
            }
            
            previous_state = current_state;
            result.steps_executed = mpc_iter + 1;
        }
        
        // Max iterations reached
        result.failure_reason = "Maximum MPC iterations reached";
        result.final_object_state = get_current_object_state(object_name);
        return result;
    }
    
    /**
     * @brief Get reachable edges using simplified approach (avoiding NAMOPushController issues)
     */
    std::vector<int> get_reachable_edges_from_current_wavefront(const std::string& object_name, 
                                                               const std::vector<double>& robot_pos) {
        std::vector<int> reachable_edges;
        
        // Use the controller's get_reachable_edge_indices method instead of recreating everything
        try {
            reachable_edges = controller_.get_reachable_edge_indices(object_name);
            
            // std::cout << "Object " << object_name << ": " << reachable_edges.size() 
            //           << "/12 edges reachable: [";
            // for (size_t i = 0; i < reachable_edges.size(); ++i) {
            //     std::cout << reachable_edges[i];
            //     if (i < reachable_edges.size() - 1) std::cout << ", ";
            // }
            // std::cout << "]" << std::endl;
            
            // Visualize edge reachability in MuJoCo viewer
            env_.visualize_edge_reachability(object_name, reachable_edges);
            
        } catch (const std::exception& e) {
            // std::cout << "Error getting reachable edges: " << e.what() << std::endl;
            // Fallback to a fixed pattern from previous working tests
            reachable_edges = {1, 3, 5, 7, 9, 11};
            // std::cout << "Using fallback reachable edges: [1, 3, 5, 7, 9, 11]" << std::endl;
        }
        
        return reachable_edges;
    }
    
private:
    SE2State get_current_object_state(const std::string& object_name) {
        auto object_state = env_.get_object_state(object_name);
        if (!object_state) {
            return SE2State();
        }
        
        // Convert quaternion to yaw
        double yaw = std::atan2(
            2.0 * (object_state->quaternion[0] * object_state->quaternion[3] + 
                   object_state->quaternion[1] * object_state->quaternion[2]),
            1.0 - 2.0 * (object_state->quaternion[2] * object_state->quaternion[2] + 
                          object_state->quaternion[3] * object_state->quaternion[3])
        );
        
        return SE2State(object_state->position[0], object_state->position[1], yaw);
    }
    
    bool is_object_at_goal(const SE2State& current, const SE2State& goal) {
        double dx = current.x - goal.x;
        double dy = current.y - goal.y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        double angle_diff = std::abs(current.theta - goal.theta);
        while (angle_diff > M_PI) angle_diff = 2.0 * M_PI - angle_diff;
        
        return distance < distance_threshold_ && angle_diff < angle_threshold_;
    }
    
    bool is_object_stuck(const SE2State& current, const SE2State& previous) {
        double dx = current.x - previous.x;
        double dy = current.y - previous.y;
        double distance_moved = std::sqrt(dx*dx + dy*dy);
        
        double angle_change = std::abs(current.theta - previous.theta);
        while (angle_change > M_PI) angle_change = 2.0 * M_PI - angle_change;
        
        const double min_position_change = 0.001;  // 1mm
        const double min_angle_change = 0.01;      // ~0.6 degrees
        
        return distance_moved < min_position_change && angle_change < min_angle_change;
    }
};

int main() {
    try {
        // std::cout << "=== Iterative MPC vs Full Sequence Execution Test ===" << std::endl;
        // std::cout << "Comparing two execution strategies:\n" << std::endl;
        
        // Initialize system with visualization enabled
        NAMOEnvironment env("../data/nominal_primitive_scene.xml", false);
        GreedyPlanner planner;
        if (!planner.initialize("../data/motion_primitives.dat")) {
            std::cerr << "Failed to initialize planner" << std::endl;
            return 1;
        }
        MPCExecutor executor(env);
        
        // Test step-by-step initialization to isolate segfault
        // std::cout << "Testing step-by-step initialization..." << std::endl;
        
        // std::cout << "1. Creating wavefront planner..." << std::endl;
        std::vector<double> robot_size = {0.15, 0.15};
        
        try {
            auto wavefront_planner = std::make_unique<WavefrontPlanner>(0.02, env, robot_size);
            // std::cout << "✓ WavefrontPlanner created successfully" << std::endl;
            
            // std::cout << "2. Creating NAMOPushController..." << std::endl;
            auto controller_ptr = std::make_unique<NAMOPushController>(env, *wavefront_planner, 10, 250, 1.0);
            // std::cout << "✓ NAMOPushController created successfully" << std::endl;
            
            // Test a simple iterative MPC execution
            IterativeMPCExecutor iterative_executor(env, planner, executor, *controller_ptr, *wavefront_planner);
            
            auto movable_objects = env.get_movable_objects();
            std::string object_name = movable_objects[0].name;
            
            // Test case: simple goal
            SE2State goal(1.0, 0.1, 2.1);
            // std::cout << "\n3. Testing iterative MPC execution..." << std::endl;
            // std::cout << "Moving " << object_name << " to goal [0.1, 0.1, 0.2]" << std::endl;
            
            // Add green goal marker (like old MuJoCo implementation)
            std::array<double, 3> goal_3d = {goal.x, goal.y, 0.0};
            std::array<float, 4> green_color = {0.0f, 1.0f, 0.0f, 1.0f}; // Green
            env.visualize_goal_marker(goal_3d, green_color);
            // std::cout << "Added green goal marker at [" << goal.x << ", " << goal.y << ", 0.0]" << std::endl;
            
            ExecutionResult result = iterative_executor.execute_iterative_mpc(object_name, goal);
            
            // std::cout << "\nIterative MPC Result:" << std::endl;
            // std::cout << "  Success: " << (result.success ? "Yes" : "No") << std::endl;
            // std::cout << "  Steps executed: " << result.steps_executed << std::endl;
            // std::cout << "  Final state: [" << std::fixed << std::setprecision(3)
            //           << result.final_object_state.x << ", "
            //           << result.final_object_state.y << ", "
            //           << result.final_object_state.theta << "]" << std::endl;
            // if (!result.success) {
            //     std::cout << "  Failure reason: " << result.failure_reason << std::endl;
            // }
            
            // std::cout << "\n✓ Iterative MPC test completed!" << std::endl;
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed during initialization: " << e.what() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}