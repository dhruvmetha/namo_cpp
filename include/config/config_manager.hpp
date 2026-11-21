#pragma once

#include "core/parameter_loader.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

namespace namo {

class ConfigSchemaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Centralized configuration management for NAMO system
 * 
 * Provides type-safe access to all configuration parameters with sensible defaults.
 * Eliminates hardcoded values throughout the codebase.
 */
class ConfigManager {
public:
    struct PlanningConfig {
        // High-level planning
        double high_level_resolution = 0.05;     // 5cm grid for reachability
        int max_planning_iterations = 10;
        bool verbose_planning = false;

        // Wavefront planning
        double skill_level_resolution = 0.02;    // 2cm grid for detailed planning
        std::vector<double> robot_size = {0.15, 0.15};  // [half_extent_x, half_extent_y] in meters

        // Robot type: "holonomic" (default point robot) or "diff_drive" (car)
        std::string robot_type = "holonomic";

        double wavefront_tier1_inflation_margin = kDefaultWavefrontTier1MarginM;  // meters
        double wavefront_edge_offset_margin = 0.020;      // meters beyond robot radius for edge spawn points

        std::vector<double> robot_goal = {0.5455398969960719, -0.8430872280407762};  // [x, y] for goal
        
        // Grid limits
        size_t max_bfs_queue = 10000000;
        size_t max_grid_width = 2000;
        size_t max_grid_height = 2000;
        size_t max_changes = 10000;
        
        // Performance thresholds
        double position_threshold = 1e-4;        // 0.1mm
        double rotation_threshold = 1e-3;        // ~0.06 degrees
        double small_rotation_limit = 0.05;      // ~3 degrees
    };
    
    struct StrategyConfig {
        // Random strategy parameters
        double min_goal_distance = 0.3;          // meters
        double max_goal_distance = 1.0;          // meters
        int max_goal_attempts = 100;             // polar sampling attempts
        int max_object_retries = 100;            // object selection retries
        
        // ML strategy parameters
        std::string zmq_endpoint = "tcp://localhost:5555";
        int zmq_timeout_ms = 1000;
        bool fallback_to_random = true;
    };
    
    struct SkillConfig {
        // NAMO push skill parameters
        int max_push_steps = 10;
        int control_steps_per_push = 250;

        // Push velocity (m/s) when the actuator is MuJoCo <velocity>.
        // compute_push_control() returns a vector of this magnitude in
        // the push direction; the velocity actuator tracks it.
        // Default 0.10 m/s ≈ 10 cm/s sim ≈ 1.7 cm/s real-equivalent at
        // scale 6. See namo/velocity_actuator_experiment_log.md.
        double push_velocity = 0.10;

        // Max-speed fraction (∈ [0, 1]) for the diff-drive push path
        // tracker. The follower outputs left/right wheel speeds as a
        // fraction of full-speed; the controller scales by push_velocity
        // to get actual rad/s for the wheel actuators. Default 0.3 keeps
        // the car gentle during the push (mirrors robot_control's
        // FollowPathController behavior).
        double push_tracker_max_speed = 0.3;

        // When true, the controller re-derives push direction from the
        // current object pose every control tick (robot traces an arc
        // as object yaws). When false, direction is fixed at primitive
        // start (matches real-side push.py default).
        bool dynamic_direction = true;

        // Skill execution
        double goal_tolerance = 0.1;             // meters
        bool check_robot_trajectory_collision = true;  // Abort push if robot body collides with walls/other objects during push trajectory (set false to disable wall-collision rejection)
        // Controller-level stuck detection tuning
        int stuck_check_stride = 20;            // control steps between checks
        int controller_stuck_threshold = 3;     // number of stuck detections before abort
        double controller_min_position_change = 0.001; // meters
        double controller_min_angle_change = 0.05;     // radians
        
        // Object interaction
        int points_per_face = 3;                // points per object face (4 faces total)
        int num_edge_points = 12;               // points around object perimeter (backward compatibility)

        // Car v2 controller (diff-drive only). Every default reproduces car v1.
        // Controller update rate in Hz on sim time; 0 updates on every physics tick (v1). With
        // N > 0 the command is recomputed when sim time reaches the next update and held between.
        double car_controller_hz = 0.0;
        // "teleport" (v1) snaps the robot to the push start. "drive" runs the port of
        // robot_control's NavigationController along the wavefront path to it.
        std::string car_nav_mode = "teleport";
        // "wheel_velocity" (v1): ctrl = command * car_wheel_max_speed_ms / wheel_radius, for a
        // velocity actuator. "normalized": ctrl = the controller's [-1, 1] command, for a PWM motor.
        std::string car_actuation = "wheel_velocity";
        // Chassis m/s per unit command under wheel_velocity actuation. 1.0 is v1's
        // kCarWheelMaxSpeedMs.
        double car_wheel_max_speed_ms = 1.0;
        // Normalized actuation only: |command| below this goes to 0 (drivetrain dead zone).
        double car_command_deadzone = 0.0;
        // "v1" = PushPathFollower as before. "follow_path" = the port of robot_control's
        // FollowPathController, which adds its ALIGN rotate-in-place mode.
        std::string car_push_follower = "v1";
        // Drive mode: a drive still going after this much sim time fails.
        double car_nav_max_time_s = 120.0;
        // robot_control config/controller.yaml navigation: block, same field names
        // (skill.navigation.<field>). Read only in drive mode.
        double nav_max_speed = 0.4;
        double nav_pre_rotation_skip_angle = 45.0;
        double nav_goal_tolerance_ratio = 0.15;
        double nav_rotation_tolerance_deg = 4.0;
        double nav_rotation_stable_time = 0.5;
        double nav_rotation_speed_max = 0.25;
        double nav_rotation_speed_min = 0.15;
        double nav_wheel_deadband = 0.05;
        // controller.yaml push: block (skill.push.<field>). The push cap push.max_speed is the
        // existing skill.push_tracker_max_speed. lookahead_ratio 1.0 is v1's kPushLookaheadRatio.
        double push_lookahead_ratio = 1.0;
        // Drive mode skips navigation when the robot already sits this close to the push start
        // (push.py:381-388). controller.yaml gives the distance in cm; this is metres.
        double push_approach_skip_distance_m = 0.01;
        double push_approach_skip_angle = 30.0;
    };
    
    struct EnvironmentConfig {
        // Environment bounds - now calculated dynamically from MuJoCo XML via NAMOEnvironment::get_environment_bounds()
        
        // Memory limits
        size_t max_static_objects = 20;
        size_t max_movable_objects = 10;
        size_t max_actions = 100;
        size_t max_motion_primitives = 1000;
        size_t max_planning_nodes = 10000;
        
        // Logging
        size_t log_buffer_size = 100000;         // bytes
        bool enable_state_logging = false;
        std::string log_directory = "logs/";
    };
    
    struct SystemConfig {
        // Performance optimization
        bool enable_visualization = true;
        bool enable_profiling = false;
        int num_threads = 1;                     // for parallel processing
        
        // File paths
        std::string motion_primitives_file = "data/motion_primitives.dat";
        std::string default_scene_file = "data/test_scene.xml";
        
        // Data collection
        bool collect_training_data = false;
        std::string training_data_directory = "training_data/";
    };
    
    struct OptimizationConfig {
        // Sequence optimization
        bool enable_sequence_optimization = false;   // Disabled by default (expensive)
        int default_method = 1;                      // 0=Exhaustive, 1=ReverseOrder, 2=GreedyRemoval
        double timeout_seconds = 30.0;               // Maximum time for optimization
        int max_sequence_length = 15;                // Don't optimize sequences longer than this
        
        // Performance settings
        int max_sequences_tested = 10000;           // Limit for exhaustive search
        bool enable_optimization_logging = true;     // Log optimization results
        bool fallback_on_timeout = true;            // Use original sequence if optimization times out
    };

private:
    std::unique_ptr<FastParameterLoader> loader_;
    
    // Configuration sections
    PlanningConfig planning_;
    StrategyConfig strategy_;
    SkillConfig skill_;
    EnvironmentConfig environment_;
    SystemConfig system_;
    OptimizationConfig optimization_;
    
    // Internal helpers
    void load_planning_config();
    void load_strategy_config();
    void load_skill_config();
    void load_environment_config();
    void load_system_config();
    void load_optimization_config();
    void load_wavefront_inflation_config(const std::string& primary_config_file);
    void validate_config_schema() const;
    void validate_configuration() const;

public:
    /**
     * @brief Constructor with configuration file
     * @param config_file Path to YAML configuration file
     */
    explicit ConfigManager(const std::string& config_file);
    
    /**
     * @brief Default constructor using default configuration
     */
    ConfigManager();
    
    /**
     * @brief Destructor
     */
    ~ConfigManager();
    
    // Configuration access
    const PlanningConfig& planning() const { return planning_; }
    const StrategyConfig& strategy() const { return strategy_; }
    const SkillConfig& skill() const { return skill_; }
    const EnvironmentConfig& environment() const { return environment_; }
    const SystemConfig& system() const { return system_; }
    const OptimizationConfig& optimization() const { return optimization_; }
    
    // Convenience methods for commonly used values
    double get_high_level_resolution() const { return planning_.high_level_resolution; }
    double get_skill_level_resolution() const { return planning_.skill_level_resolution; }
    const std::vector<double>& get_robot_size() const { return planning_.robot_size; }
    const std::string& get_robot_type() const { return planning_.robot_type; }
    // get_environment_bounds() removed - use NAMOEnvironment::get_environment_bounds() instead
    
    // Runtime configuration updates
    void set_verbose_planning(bool enabled) { planning_.verbose_planning = enabled; }
    void set_max_iterations(int max_iter) { planning_.max_planning_iterations = max_iter; }
    void set_visualization(bool enabled) { system_.enable_visualization = enabled; }

    // Static factory methods
    static std::unique_ptr<ConfigManager> create_from_file(const std::string& config_file);
};

} // namespace namo
