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
#include <map>

namespace namo {

// Represents a complete snapshot of the simulation state
struct RLState {
    std::vector<double> qpos;
    std::vector<double> qvel;
};

class RLEnvironment {
public:
    struct ObjectReachabilitySummary {
        bool reachable = false;
        int reachable_edges = 0;
        int total_edges = 0;
        int reachable_primitives = 0;
        int total_primitives = 0;
        std::vector<int> reachable_edge_indices;
    };

    struct ReachabilitySummary {
        bool goal_reachable = false;
        std::map<std::string, ObjectReachabilitySummary> objects;
    };

    struct RegionSnapshot {
        std::unordered_map<std::string, std::unordered_set<std::string>> adjacency;
        std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>> edge_objects;
        std::unordered_map<int, std::string> region_labels;
        std::unordered_map<std::string, RegionGoalBundle> region_goals;
        std::string robot_label;
        std::string goal_label;
        bool goal_reachable = false;
        bool goal_in_free_space = false;
    };

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

    // NavigateResult struct and navigate_to() declaration removed: the impl
    // was deleted in commit 254e5c7 (2026-04-14, "Unify wavefront semantics
    // and C++ region snapshot") and no consumer exists — all Python
    // navigation goes through robot_control's NavigationController.


    RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize = false,
                  bool skip_warmup = false);

    /// Run the env's post-load physics warm-up explicitly. Only needed when
    /// the env was constructed with skip_warmup=true (e.g. car planning,
    /// where the caller teleports the robot to a safe pose before allowing
    /// physics to integrate).
    void warm_up();
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
    ReachabilitySummary get_reachability_summary(bool analysis_mode = false) const;

    // Edge point queries (for visualization)
    std::vector<std::array<double, 2>> get_edge_points(const std::string& object_name) const;

    // Object geometry information (returns cached reference)
    const std::map<std::string, std::map<std::string, double>>& get_object_info() const;
    
    // World bounds information
    std::vector<double> get_world_bounds() const;

	    // Override the robot's pose loaded from the XML. Needed for car
	    // (diff-drive) planning where the freejoint spawn position lives
	    // inside the included little_car.xml and can't be parameterized
	    // through a top-level <include>. The bridge calls this once
	    // right after the env is constructed, with the live observation
	    // pose, so the planner searches from the correct starting state.
	    // Sphere XMLs bake the pose into the geom directly and don't
	    // need this call.
	    void set_robot_pose(double x, double y, double theta);

	    // Robot goal management for MCTS
	    void set_robot_goal(double x, double y, double theta = 0.0);
	    // Set robot goal without updating the visualization goal marker (useful for
	    // reachability checks over many sampled goals without flickering the marker).
	    void set_robot_goal_silent(double x, double y, double theta = 0.0);
	    bool is_robot_goal_reachable() const;
	    std::array<double, 3> get_robot_goal() const;
	    void clear_robot_goal();
	    void set_goal_site_visible(bool visible);

    // Collision checking control (for region opening planner)
    void set_collision_checking(bool enable);
    void set_robot_trajectory_collision_checking(bool enable);
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
    // Get one captured frame by index (no copy). Intended for streaming encoders.
    const std::vector<unsigned char>& get_frame_ref(size_t idx) const;
    void clear_frames();
    std::tuple<int, int> get_recording_dimensions() const;

    // Geometric transport heuristic for primitive prioritization
    std::vector<int> evaluate_primitive_priorities(
        const std::string& object_name,
        const std::vector<std::array<double, 3>>& target_poses,
        const std::array<double, 2>& robot_goal);
    std::map<std::string, double> get_last_priority_profile() const;

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
    RegionSnapshot get_region_snapshot(
        int goals_per_region = 0,
        double goal_radius = -1.0,
        bool local_info_only = false,
        unsigned int seed = 42,
        bool use_xml_goal = true) const;

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
