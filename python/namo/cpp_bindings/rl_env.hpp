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
        int edge_idx = -1;  // -1 means use MPC+search, >=0 means execute directly
        int depth = -1;     // 0-indexed (depth=0 means push_steps=1)
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
    void set_camera_position(double distance, double azimuth, double elevation);
    void set_camera_lookat(double x, double y, double z);
    
    // Reachability queries
    std::vector<std::string> get_reachable_objects() const;
    bool is_object_reachable(const std::string& object_name) const;
    std::vector<int> get_reachable_edges(const std::string& object_name) const;

    // Edge point queries (for visualization)
    std::vector<std::array<double, 2>> get_edge_points(const std::string& object_name) const;

    // Object geometry information (returns cached reference)
    const std::map<std::string, std::map<std::string, double>>& get_object_info() const;
    
    // World bounds information
    std::vector<double> get_world_bounds() const;

    // Robot goal management for MCTS
    void set_robot_goal(double x, double y, double theta = 0.0);
    bool is_robot_goal_reachable() const;
    std::array<double, 3> get_robot_goal() const;
    void clear_robot_goal();

    // Collision checking control (for region opening planner)
    void set_collision_checking(bool enable);
    bool get_collision_checking() const;

    // Robot goal termination control (defaults to false)
    void set_robot_goal_termination(bool enable);
    bool get_robot_goal_termination() const;

    // Video recording interface
    void start_recording(int width = 640, int height = 480,
                        int capture_frequency = 100, size_t max_frames = 10000);
    void stop_recording();
    bool is_recording() const;
    size_t get_frame_count() const;
    std::vector<std::vector<unsigned char>> get_frames() const;
    void clear_frames();
    std::tuple<int, int> get_recording_dimensions() const;

    // Geometric transport heuristic for primitive prioritization
    std::vector<int> evaluate_primitive_priorities(
        const std::string& object_name,
        const std::vector<std::array<double, 3>>& target_poses,
        const std::array<double, 2>& robot_goal);

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
