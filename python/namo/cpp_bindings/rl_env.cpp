#include "python/namo/cpp_bindings/rl_env.hpp"
#include "core/types.hpp"
#include <iostream>
#include <mujoco/mujoco.h>

namespace namo {

RLEnvironment::RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize) {
    // std::cout << "Initializing RLEnvironment..." << std::endl;
    try {
        config_ = std::shared_ptr<ConfigManager>(ConfigManager::create_from_file(config_path).release());
        env_ = std::make_unique<NAMOEnvironment>(xml_path, visualize); // Use provided visualization parameter
        skill_ = std::make_unique<NAMOPushSkill>(*env_, config_);
        
        // Cache immutable object info once during initialization
        cached_object_info_ = env_->get_all_object_info();
        
        // std::cout << "RLEnvironment initialized successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during RLEnvironment initialization: " << e.what() << std::endl;
        throw;
    }
}

RLEnvironment::~RLEnvironment() = default;

void RLEnvironment::reset() {
    env_->reset(); // reset_to_initial_state();
}

RLEnvironment::StepResult RLEnvironment::step(const Action& action) {
    std::map<std::string, SkillParameterValue> params = {
        {"object_name", action.object_id},
        {"target_pose", SE2State(action.x, action.y, action.theta)}
    };

    if (!skill_->is_applicable(params)) {
        return {false, -10.0, {{"failure_reason", "Action not applicable"}}};
    }

    auto result = skill_->execute(params);
    
    StepResult rl_result;
    rl_result.done = result.success;
    
    // MCTS sparse reward: +1 if robot goal reachable, -1 otherwise
    bool goal_reached = false;
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        goal_reached = std::get<bool>(it->second);
    }
    rl_result.reward = goal_reached ? 1.0 : -1.0;
    
    rl_result.info["failure_reason"] = result.failure_reason;
    rl_result.info["failure_type"] = std::to_string(static_cast<int>(result.failure_type));

    if (auto it = result.outputs.find("steps_executed"); it != result.outputs.end()) {
        rl_result.info["steps_executed"] = std::to_string(std::get<int>(it->second));
    }
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        rl_result.info["robot_goal_reached"] = std::get<bool>(it->second) ? "true" : "false";
    }
    if (auto it = result.outputs.find("collision_object"); it != result.outputs.end()) {
        rl_result.info["collision_object"] = std::get<std::string>(it->second);
    }

    return rl_result;
}

std::map<std::string, std::vector<double>> RLEnvironment::get_observation() const {
    auto world_state = skill_->get_world_state();
    std::map<std::string, std::vector<double>> state_map;

    for (const auto& [key, value] : world_state) {
        if (std::holds_alternative<SE2State>(value)) {
            const auto& pose = std::get<SE2State>(value);
            state_map[key] = {pose.x, pose.y, pose.theta};
        }
    }
    return state_map;
}

RLState RLEnvironment::get_full_state() const {
    // Use NAMOEnvironment's zero-allocation method, then convert to RLState
    auto full_state = env_->get_full_state();
    
    RLState rl_state;
    
    // Copy qpos
    rl_state.qpos.resize(full_state.nq);
    for (int i = 0; i < full_state.nq; i++) {
        rl_state.qpos[i] = full_state.qpos[i];
    }
    
    // Copy qvel
    rl_state.qvel.resize(full_state.nv);
    for (int i = 0; i < full_state.nv; i++) {
        rl_state.qvel[i] = full_state.qvel[i];
    }
    
    return rl_state;
}

void RLEnvironment::set_full_state(const RLState& state) {
    // Convert RLState to NAMOEnvironment::FullSimState, then use zero-allocation method
    NAMOEnvironment::FullSimState full_state;
    
    // Copy qpos
    full_state.nq = std::min(static_cast<int>(state.qpos.size()), 
                            static_cast<int>(NAMOEnvironment::FullSimState::MAX_QPOS));
    for (int i = 0; i < full_state.nq; i++) {
        full_state.qpos[i] = state.qpos[i];
    }
    
    // Always zero qvel for consistent physics simulation (matching original RL behavior)
    full_state.nv = std::min(static_cast<int>(state.qvel.size()), 
                            static_cast<int>(NAMOEnvironment::FullSimState::MAX_QVEL));
    for (int i = 0; i < full_state.nv; i++) {
        full_state.qvel[i] = 0.0;  // Zero qvel for consistent physics
    }
    
    // Use NAMOEnvironment's centralized state setting
    env_->set_full_state(full_state);
}

void RLEnvironment::render() {
    auto* sim = env_->get_mujoco_wrapper();
    sim->render();
}

std::vector<std::string> RLEnvironment::get_reachable_objects() const {
    return skill_->get_reachable_objects();
}

bool RLEnvironment::is_object_reachable(const std::string& object_name) const {
    return skill_->is_object_reachable(object_name);
}

const std::map<std::string, std::map<std::string, double>>& RLEnvironment::get_object_info() const {
    // Return cached reference - zero cost operation!
    return cached_object_info_;
}

void RLEnvironment::set_robot_goal(double x, double y, double theta) {
    skill_->set_robot_goal(x, y, theta);
}

bool RLEnvironment::is_robot_goal_reachable() const {
    return skill_->is_robot_goal_reachable();
}

std::array<double, 3> RLEnvironment::get_robot_goal() const {
    return skill_->get_robot_goal();
}

std::vector<double> RLEnvironment::get_world_bounds() const {
    return env_->get_environment_bounds();
}

RLEnvironment::ActionConstraints RLEnvironment::get_action_constraints() const {
    return ActionConstraints{}; // Use default values: distance [0.3, 1.0], theta [-π, π]
}

} // namespace namo
