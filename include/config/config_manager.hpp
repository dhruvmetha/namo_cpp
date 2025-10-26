#pragma once

#include "core/parameter_loader.hpp"
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace namo {

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
        std::vector<double> robot_size = {0.15, 0.15};  // [width, height] for inflation
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
        int max_mpc_iterations = 10;
        int control_steps_per_push = 250;
        double force_scaling = 1.0;
        
        // Skill execution
        double goal_tolerance = 0.1;             // meters
        double stuck_threshold = 0.01;           // meters - minimum movement to avoid stuck detection
        int max_stuck_iterations = 3;
        bool check_object_collision = true;      // Check pushed object collisions during execution
        // Controller-level stuck detection tuning
        int stuck_check_stride = 20;            // control steps between checks
        int controller_stuck_threshold = 3;     // number of stuck detections before abort
        double controller_min_position_change = 0.001; // meters
        double controller_min_angle_change = 0.05;     // radians
        
        // Object interaction
        double object_clearance = 0.1;           // meters around objects for edge point sampling
        int points_per_face = 3;                // points per object face (4 faces total)
        int num_edge_points = 12;               // points around object perimeter (backward compatibility)
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
    // get_environment_bounds() removed - use NAMOEnvironment::get_environment_bounds() instead
    
    // Runtime configuration updates
    void set_verbose_planning(bool enabled) { planning_.verbose_planning = enabled; }
    void set_max_iterations(int max_iter) { planning_.max_planning_iterations = max_iter; }
    void set_visualization(bool enabled) { system_.enable_visualization = enabled; }
    void set_collision_checking(bool enabled) { skill_.check_object_collision = enabled; }

    // Configuration validation and diagnostics
    void print_configuration() const;
    bool validate_paths() const;
    
    // Static factory methods
    static std::unique_ptr<ConfigManager> create_default();
    static std::unique_ptr<ConfigManager> create_from_file(const std::string& config_file);
};

} // namespace namo