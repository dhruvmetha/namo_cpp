#pragma once

#include "skills/manipulation_skill.hpp"
#include "planning/mpc_executor.hpp"
#include "planning/greedy_planner.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include <optional>

namespace namo {

/**
 * @brief Proper NAMO push skill implementation
 * 
 * Eliminates hacks and properly integrates with existing systems
 */
class NAMOPushSkill : public ManipulationSkill {
private:
    NAMOEnvironment& env_;
    std::unique_ptr<GreedyPlanner> planner_;
    std::unique_ptr<MPCExecutor> executor_;
    std::shared_ptr<ConfigManager> config_;
    
    // Deprecated - kept for backward compatibility
    struct Config {
        double tolerance = 0.01;
        int max_planning_attempts = 3;
        int max_mpc_iterations = 5;
        std::chrono::milliseconds planning_timeout{5000};
        std::string primitive_database_path = "data/motion_primitives.dat";
    };
    Config legacy_config_;
    
    // Robot goal state for MCTS
    std::array<double, 3> robot_goal_{0.0, 0.0, 0.0};
    bool has_robot_goal_{false};
    
public:
    /**
     * @brief Constructor with proper dependency injection
     */
    explicit NAMOPushSkill(NAMOEnvironment& env);
    explicit NAMOPushSkill(NAMOEnvironment& env, const Config& config);  // Legacy
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
    
    /**
     * @brief Robot goal management for MCTS (leverages cached wavefront)
     */
    void set_robot_goal(double x, double y, double theta = 0.0);
    bool is_robot_goal_reachable() const;
    std::array<double, 3> get_robot_goal() const;
    void clear_robot_goal();
    
private:
    /**
     * @brief Helper methods for skill implementation
     */
    bool is_object_movable(const std::string& object_name) const;
    std::optional<SE2State> get_object_current_pose(const std::string& object_name) const;
    bool is_target_within_bounds(const SE2State& target_pose) const;
    
    /**
     * @brief Helper methods for iterative MPC
     */
    bool is_object_at_goal(const SE2State& current, const SE2State& goal, double tolerance) const;
    std::vector<int> get_reachable_edges(const std::string& object_name) const;
};

} // namespace namo