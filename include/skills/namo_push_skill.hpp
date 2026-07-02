#pragma once

#include "skills/manipulation_skill.hpp"
#include "planning/push_primitive_executor.hpp"
#include "planning/greedy_planner.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include <optional>
#include <filesystem>
#include <map>

namespace namo {

/**
 * @brief Proper NAMO push skill implementation
 *
 * Eliminates hacks and properly integrates with existing systems
 */
class NAMOPushSkill : public ManipulationSkill {
private:
    NAMOEnvironment& env_;

    std::unique_ptr<PushPrimitiveExecutor> executor_;
    std::shared_ptr<ConfigManager> config_;

    // Robot goal state for MCTS
    std::array<double, 3> robot_goal_{0.0, 0.0, 0.0};
    bool has_robot_goal_{false};

public:
    /**
     * @brief Constructor with proper dependency injection
     */
    explicit NAMOPushSkill(NAMOEnvironment& env, std::shared_ptr<ConfigManager> config);

private:
    void initialize_skill();

public:

    // ManipulationSkill interface
    std::string get_name() const override {
        return "namo_push";
    }

    std::string get_description() const override {
        return "Push a rectangular object to a target SE(2) pose using NAMO planning with physics simulation";
    }

    std::map<std::string, ParameterSchema> get_parameter_schema() const override;
    bool is_applicable(const std::map<std::string, SkillParameterValue>& parameters) const override;
    std::chrono::milliseconds estimate_duration(const std::map<std::string, SkillParameterValue>& parameters) const override;
    SkillResult execute(const std::map<std::string, SkillParameterValue>& parameters) override;
    std::map<std::string, SkillParameterValue> get_world_state() const override;
    std::vector<std::string> check_preconditions(const std::map<std::string, SkillParameterValue>& parameters) const override;

    /**
     * @brief Query methods for RL environment
     */
    std::vector<std::string> get_reachable_objects() const;
    bool is_object_reachable(const std::string& object_name) const;
    std::vector<int> get_reachable_edges(const std::string& object_name) const;
    PushPrimitiveExecutor::ReachabilitySnapshot get_reachability_snapshot() const;

    /**
     * @brief Robot goal management for MCTS (leverages cached wavefront)
     */
    void set_robot_goal(double x, double y, double theta = 0.0);
    bool is_robot_goal_reachable() const;
    std::array<double, 3> get_robot_goal() const;
    void clear_robot_goal();

    /**
     * @brief Count how many of the given (x, y) points are reachable from the
     * robot's current position. Does NOT mutate the robot goal — use this for
     * subgoal probes (e.g., region opening) instead of overwriting the task goal.
     *
     * @param points List of (x, y) world coordinates to test.
     * @return std::pair<int, int> (count, first_reachable_index). The second
     *         element is -1 if none are reachable.
     */
    std::pair<int, int> count_reachable_points(
        const std::vector<std::array<double, 2>>& points) const;

    /**
     * @brief Evaluate geometric transport priorities for primitive target poses
     * @param object_name Object to evaluate
     * @param target_poses Vector of target poses [x, y, theta]
     * @param robot_goal Robot goal position [x, y]
     * @return Vector of priorities (1=best, 4=worst) for each target pose
     */
    std::vector<int> evaluate_primitive_priorities(
        const std::string& object_name,
        const std::vector<std::array<double, 3>>& target_poses,
        const std::array<double, 2>& robot_goal);

    /**
     * @brief Get timing breakdown for the last evaluate_primitive_priorities() call.
     */
    std::map<std::string, double> get_last_priority_profile() const;

    /**
     * @brief Runtime configuration for collision checking
     */
    void set_collision_checking(bool enabled);
    void set_robot_trajectory_collision_checking(bool enabled);

private:
    /**
     * @brief Helper methods for skill implementation
     */
    bool is_object_movable(const std::string& object_name) const;
    std::optional<SE2State> get_object_current_pose(const std::string& object_name) const;
};

} // namespace namo
