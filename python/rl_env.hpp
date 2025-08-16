#pragma once

#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include <vector>

namespace namo {

// Represents a complete snapshot of the simulation state
struct RLState {
    std::vector<double> qpos;
    std::vector<double> qvel;
};

class RLEnvironment {
public:
    struct Action {
        std::string object_id;
        double x, y, theta;
    };

    struct StepResult {
        bool done;
        double reward;
        std::map<std::string, std::string> info;
    };

    RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize = false);
    ~RLEnvironment();

    // Standard RL methods
    void reset();
    StepResult step(const Action& action);
    std::map<std::string, std::vector<double>> get_observation() const;

    // State management for MCTS
    RLState get_full_state() const;
    void set_full_state(const RLState& state);

    // Visualization methods
    void render();
    
    // Reachability queries
    std::vector<std::string> get_reachable_objects() const;
    bool is_object_reachable(const std::string& object_name) const;


private:
    std::unique_ptr<NAMOEnvironment> env_;
    std::unique_ptr<NAMOPushSkill> skill_;
    std::shared_ptr<ConfigManager> config_;
};

} // namespace namo
