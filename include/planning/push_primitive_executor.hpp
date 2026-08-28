#pragma once

#include "planning/greedy_planner.hpp"
#include "environment/namo_environment.hpp"
#include "planning/namo_push_controller.hpp"
#include "wavefront/wavefront_planner.hpp"
#include <memory>
#include <map>
#include <unordered_map>

namespace namo {

/**
 * @brief Execution result for a primitive sequence
 */
struct ExecutionResult {
    bool success;                    // True if execution completed successfully
    bool robot_goal_reached;         // True if robot goal became reachable during execution
    int steps_executed;              // Number of primitive steps executed
    SE2State final_object_state;     // Final object pose after execution
    std::string failure_reason;      // Description of failure if success=false
    std::string collision_object;    // Name of object that caused collision (if any)

    // Collision tracking for hardness metrics (accumulated during push)
    bool wall_collision_during_push;                      // Did object hit any wall during push?
    std::vector<std::string> movable_collisions_during_push;  // Unique movable objects hit during push

    // A step stopped on stuck detection after moving the object at least
    // kMinUsefulPushDisplacementM. The motion is kept and the plan stops here,
    // so the caller replans from the pose the object actually reached instead of
    // running the rest of a chain the world no longer matches.
    bool stopped_early;

    ExecutionResult() : success(false), robot_goal_reached(false), steps_executed(0), wall_collision_during_push(false), stopped_early(false) {}
};

/**
 * @brief Executes a single push primitive (edge_idx, push_steps) with real MuJoCo physics.
 *
 * Wraps NAMOPushController + WavefrontPlanner. Given an object name and a
 * PlanStep (edge_idx, push_steps), it checks reachability against the
 * current wavefront, executes one continuous push primitive through the
 * controller, and reports collisions / stuck conditions.
 *
 * Historical note: this class used to drive an outer MPC retry loop (hence
 * the original name "MPCExecutor"). The retry loop was removed — the class
 * now just executes one primitive.
 */
class PushPrimitiveExecutor {
private:
    NAMOEnvironment& env_;
    WavefrontPlanner planner_;
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
    struct ReachableEdgesResult {
        std::vector<int> edge_indices;
        int total_edge_points = 0;
    };

    struct ReachabilitySnapshot {
        bool goal_reachable = false;
        std::map<std::string, ReachableEdgesResult> object_edges;
    };

    /**
     * @brief Constructor with default hardcoded values
     *
     * @param env Reference to NAMO environment
     */
    PushPrimitiveExecutor(NAMOEnvironment& env);

    /**
     * @brief Constructor with ConfigManager parameters
     *
     * @param env Reference to NAMO environment
     * @param resolution Wavefront planning resolution
     * @param robot_size Robot size for wavefront inflation
     * @param wavefront_tier1_inflation_margin Wavefront tier-1 inflation margin
     * @param max_push_steps Max push steps per primitive
     * @param control_steps_per_push Control steps per push
     * @param push_velocity Velocity-command magnitude (m/s) for the push actuator
     * @param points_per_face Number of edge points per object face
     */
    PushPrimitiveExecutor(NAMOEnvironment& env, double resolution, const std::vector<double>& robot_size,
                double wavefront_tier1_inflation_margin,
                int max_push_steps, int control_steps_per_push, double push_velocity, int points_per_face = 3,
                bool dynamic_direction = true);

    /**
     * @brief Set execution parameters
     *
     * @param max_mpc_steps Maximum MPC steps per primitive (default: 10)
     * @param distance_threshold Distance threshold for goal reaching (default: 0.01)
     * @param angle_threshold Angle threshold for goal reaching (default: 0.1)
     * @param max_stuck_iterations Max stuck iterations before failure (default: 2)
     */
    void set_parameters(int max_mpc_steps = 10,
                       double distance_threshold = 0.01,
                       double angle_threshold = 0.1,
                       int max_stuck_iterations = 2);

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
     * @brief Set robot-trajectory collision checking on the underlying controller
     */
    void set_robot_trajectory_collision_checking(bool enabled) {
        controller_.set_robot_trajectory_collision_checking(enabled);
    }

    /**
     * @brief Execute a sequence of primitive plans
     *
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
     * @brief Execute a single primitive step
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
     * @brief Check if robot goal is reachable
     *
     * @return bool True if robot goal is reachable
     */
    bool is_robot_goal_reachable();

    /**
     * @brief Count how many of the given points are reachable from the robot's
     * current position, using the cached wavefront. Does NOT mutate the robot
     * goal. Updates the wavefront once, then performs cheap grid lookups.
     *
     * @param points List of (x, y) world coordinates to test.
     * @return std::pair<int, int> (count, first_reachable_index). The second
     *         element is -1 if no point is reachable.
     */
    std::pair<int, int> count_reachable_points(
        const std::vector<std::array<double, 2>>& points);


    /**
     * @brief Get controller for configuration access
     *
     * @return Reference to push controller
     */
    NAMOPushController& get_controller() { return controller_; }

    /**
     * @brief Update wavefront and get reachable edges for object
     *
     * @param object_name Name of object to check reachability for
     * @return std::vector<int> List of reachable edge indices
     */
    std::vector<int> get_reachable_edges_with_wavefront(const std::string& object_name);

    /**
     * @brief Update wavefront and get detailed reachable-edge stats for one object.
     */
    ReachableEdgesResult get_reachable_edges_with_wavefront_detailed(const std::string& object_name);

    /**
     * @brief Compute one unified reachability snapshot from a single wavefront update.
     */
    ReachabilitySnapshot compute_reachability_snapshot();

    /**
     * @brief Evaluate geometric transport priorities for primitive target poses
     * @param env Environment reference
     * @param object_name Object to evaluate
     * @param target_poses Vector of target poses [x, y, theta]
     * @param robot_goal Robot goal position [x, y]
     * @return Vector of priorities (1=best, 4=worst) for each target pose
     */
    std::vector<int> evaluate_primitive_priorities(
        NAMOEnvironment& env,
        const std::string& object_name,
        const std::vector<std::array<double, 3>>& target_poses,
        const std::array<double, 2>& robot_goal);

    /**
     * @brief Score virtual primitive endpoints by reachable target-region fraction.
     */
    std::vector<double> evaluate_primitive_region_scores(
        NAMOEnvironment& env,
        const std::string& object_name,
        const std::vector<std::array<double, 3>>& target_poses,
        const std::vector<std::array<double, 2>>& region_samples);

    /**
     * @brief Get timing breakdown for the last evaluate_primitive_priorities() call.
     */
    std::map<std::string, double> get_last_priority_profile() const;

private:
    /**
     * @brief Update internal wavefront using current robot state.
     */
    bool update_wavefront_from_robot_position();

    /**
     * @brief Read reachable-edge stats from current cached wavefront.
     * @note Assumes wavefront already updated for current state.
     */
    ReachableEdgesResult get_reachable_edges_from_current_wavefront(const std::string& object_name);

    /**
     * @brief Get current object state as SE(2)
     *
     * @param object_name Name of object
     * @return SE2State Current object pose
     */
    SE2State get_object_se2_state(const std::string& object_name);


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
