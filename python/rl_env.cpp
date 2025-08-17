#include "python/rl_env.hpp"
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
        // std::cout << "RLEnvironment initialized successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during RLEnvironment initialization: " << e.what() << std::endl;
        throw;
    }
}

RLEnvironment::~RLEnvironment() = default;

void RLEnvironment::reset() {
    env_->reset_to_initial_state();
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

    if (auto it = result.outputs.find("steps_executed"); it != result.outputs.end()) {
        rl_result.info["steps_executed"] = std::to_string(std::get<int>(it->second));
    }
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        rl_result.info["robot_goal_reached"] = std::get<bool>(it->second) ? "true" : "false";
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
    auto* sim = env_->get_mujoco_wrapper();
    const mjData* d = sim->data();
    const mjModel* m = sim->model();
    
    RLState state;
    
    // Copy qpos
    state.qpos.resize(m->nq);
    for (int i = 0; i < m->nq; i++) {
        state.qpos[i] = d->qpos[i];
    }
    
    // Copy qvel
    state.qvel.resize(m->nv);
    for (int i = 0; i < m->nv; i++) {
        state.qvel[i] = d->qvel[i];
    }
    
    return state;
}

void RLEnvironment::set_full_state(const RLState& state) {
    auto* sim = env_->get_mujoco_wrapper();
    mjData* d = sim->data();
    mjModel* m = sim->model();
    
    // Set qpos
    for (int i = 0; i < m->nq && i < static_cast<int>(state.qpos.size()); i++) {
        d->qpos[i] = state.qpos[i];
    }
    
    // Always zero qvel for consistent physics simulation
    for (int i = 0; i < m->nv; i++) {
        d->qvel[i] = 0.0;
    }
    
    // Apply the new state to the simulation
    mj_forward(m, d);
    
    // Update the environment's internal state tracking
    env_->update_object_states();
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

void RLEnvironment::set_robot_goal(double x, double y, double theta) {
    skill_->set_robot_goal(x, y, theta);
}

bool RLEnvironment::is_robot_goal_reachable() const {
    return skill_->is_robot_goal_reachable();
}

std::array<double, 3> RLEnvironment::get_robot_goal() const {
    return skill_->get_robot_goal();
}

RLEnvironment::ActionConstraints RLEnvironment::get_action_constraints() const {
    return ActionConstraints{}; // Use default values: distance [0.3, 1.0], theta [-π, π]
}

} // namespace namo
