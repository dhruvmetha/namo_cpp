#pragma once

#include "planning/greedy_planner.hpp"
#include "environment/namo_environment.hpp"
#include "planning/namo_push_controller.hpp"
#include "planning/incremental_wavefront_planner.hpp"
#include <memory>

namespace namo {

/**
 * @brief Execution result for a primitive sequence
 */
struct ExecutionResult {
    bool success;                    // True if execution completed successfully
    bool robot_goal_reached;         // True if robot goal became reachable during execution
    int steps_executed;              // Number of primitive steps executed
    SE2State final_object_state;     // Final object pose after execution
    std::string failure_reason;     // Description of failure if success=false
    
    ExecutionResult() : success(false), robot_goal_reached(false), steps_executed(0) {}
};

/**
 * @brief MPC execution layer for primitive plans
 * 
 * Takes abstract primitive sequences from GreedyPlanner and executes them
 * with real MuJoCo physics using existing NAMO infrastructure.
 * Handles discrepancies between universal primitives and actual dynamics.
 */
class MPCExecutor {
private:
    NAMOEnvironment& env_;
    IncrementalWavefrontPlanner planner_;
    NAMOPushController controller_;
    
    // Execution parameters
    int max_mpc_steps_;              // Maximum MPC steps per primitive
    double distance_threshold_;      // Distance threshold for goal checking
    double angle_threshold_;         // Angle threshold for goal checking  
    int max_stuck_iterations_;       // Max iterations before considering object stuck
    
    // Robot goal for early termination
    bool has_robot_goal_;
    std::array<double, 2> robot_goal_;
    
public:
    /**
     * @brief Constructor
     * 
     * @param env Reference to NAMO environment
     */
    MPCExecutor(NAMOEnvironment& env);
    
    /**
     * @brief Set execution parameters
     * 
     * @param max_mpc_steps Maximum MPC steps per primitive (default: 10)
     * @param distance_threshold Distance threshold for goal reaching (default: 0.01)
     * @param angle_threshold Angle threshold for goal reaching (default: 0.1)
     * @param max_stuck_iterations Max stuck iterations before failure (default: 3)
     */
    void set_parameters(int max_mpc_steps = 10, 
                       double distance_threshold = 0.01,
                       double angle_threshold = 0.1,
                       int max_stuck_iterations = 3);
    
    /**
     * @brief Set robot goal for early termination checking
     * 
     * @param robot_goal Robot goal position [x, y]
     */
    void set_robot_goal(const std::array<double, 2>& robot_goal);
    
    /**
     * @brief Clear robot goal (disable early termination)
     */
    void clear_robot_goal() { has_robot_goal_ = false; }
    
    /**
     * @brief Execute a sequence of primitive plans with MPC
     * 
     * Follows old implementation approach:
     * 1. For each primitive in sequence, set goal state in environment
     * 2. Use existing push controller to execute primitive with real physics
     * 3. Check if robot goal becomes reachable (early termination)
     * 4. Handle stuck situations and dynamic discrepancies
     * 
     * @param object_name Name of object to manipulate
     * @param plan_sequence Sequence of primitive actions from GreedyPlanner
     * @return ExecutionResult Result of execution with success/failure info
     */
    ExecutionResult execute_plan(
        const std::string& object_name,
        const std::vector<PlanStep>& plan_sequence
    );
    
    /**
     * @brief Execute a single primitive step with MPC
     * 
     * @param object_name Name of object to manipulate
     * @param plan_step Single primitive action to execute
     * @return bool True if execution succeeded
     */
    bool execute_primitive_step(
        const std::string& object_name,
        const PlanStep& plan_step
    );
    
    /**
     * @brief Check if robot goal is reachable (public access for iterative MPC)
     * 
     * @return bool True if robot goal is reachable
     */
    bool is_robot_goal_reachable();
    
private:
    
    /**
     * @brief Check if object has reached the target state
     * 
     * @param object_name Name of object
     * @param target_state Target SE(2) state
     * @return bool True if object is close enough to target
     */
    bool is_object_at_target(const std::string& object_name, const SE2State& target_state);
    
    /**
     * @brief Get current object state as SE(2)
     * 
     * @param object_name Name of object
     * @return SE2State Current object pose
     */
    SE2State get_object_se2_state(const std::string& object_name);
    
    /**
     * @brief Convert SE2State to goal state vector format
     * 
     * @param se2_state SE(2) state
     * @return std::vector<double> Goal state in [x, y, z, qw, qx, qy, qz] format
     */
    std::vector<double> se2_to_goal_state(const SE2State& se2_state);
    
    /**
     * @brief Check if object is stuck (not moving)
     * 
     * @param object_name Name of object
     * @param previous_state Previous object state
     * @return bool True if object appears stuck
     */
    bool is_object_stuck(const std::string& object_name, const SE2State& previous_state);
};

} // namespace namo