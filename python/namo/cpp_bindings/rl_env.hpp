#pragma once

#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"
#include "config/config_manager.hpp"
#include "wavefront/wavefront_grid.hpp"
#include <vector>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <string>

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
    
    // Object geometry information (returns cached reference)
    const std::map<std::string, std::map<std::string, double>>& get_object_info() const;
    
    // World bounds information
    std::vector<double> get_world_bounds() const;

    // Robot goal management for MCTS
    void set_robot_goal(double x, double y, double theta = 0.0);
    bool is_robot_goal_reachable() const;
    std::array<double, 3> get_robot_goal() const;
    
    // Action space constraints for MCTS progressive widening
    struct ActionConstraints {
        double min_distance = 0.3;  // Minimum distance from object
        double max_distance = 1.0;  // Maximum distance from object  
        double theta_min = -M_PI;   // Minimum theta
        double theta_max = M_PI;    // Maximum theta
    };
    ActionConstraints get_action_constraints() const;

    using RegionAdjacency = std::unordered_map<std::string, std::unordered_set<std::string>>;
    using RegionEdgeObjects = std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>>;
    using RegionLabels = std::unordered_map<int, std::string>;
    using RegionGoalSamples = std::unordered_map<std::string, RegionGoalBundle>;

    std::tuple<RegionAdjacency, RegionEdgeObjects, RegionLabels> get_region_connectivity() const;
    RegionGoalSamples sample_region_goals(int goals_per_region) const;

    const std::string& get_xml_path() const { return xml_path_; }
    const std::string& get_config_path() const { return config_path_; }

private:
    std::unique_ptr<NAMOEnvironment> env_;
    std::unique_ptr<NAMOPushSkill> skill_;
    std::shared_ptr<ConfigManager> config_;

    std::string xml_path_;
    std::string config_path_;
    
    // Cached immutable object info (built once during initialization)
    std::map<std::string, std::map<std::string, double>> cached_object_info_;
};

} // namespace namo
