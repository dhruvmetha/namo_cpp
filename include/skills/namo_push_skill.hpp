#pragma once

#include "skills/manipulation_skill.hpp"
#include "planning/mpc_executor.hpp"
#include "planning/greedy_planner.hpp"
#include "environment/namo_environment.hpp"
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
    
public:
    // Configuration structure
    struct Config {
        double tolerance = 0.01;
        int max_planning_attempts = 3;
        std::chrono::milliseconds planning_timeout{5000};
        std::string primitive_database_path = "data/motion_primitives.dat";
    };

private:
    Config config_;
    
public:
    /**
     * @brief Constructor with proper dependency injection
     */
    explicit NAMOPushSkill(NAMOEnvironment& env);
    explicit NAMOPushSkill(NAMOEnvironment& env, const Config& config);

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
    
private:
    /**
     * @brief Helper methods for skill implementation
     */
    bool is_object_movable(const std::string& object_name) const;
    std::optional<SE2State> get_object_current_pose(const std::string& object_name) const;
    bool is_target_within_bounds(const SE2State& target_pose) const;
};

} // namespace namo