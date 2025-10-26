#include "config/config_manager.hpp"
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

namespace namo {

ConfigManager::ConfigManager(const std::string& config_file) {
    try {
        loader_ = std::make_unique<FastParameterLoader>(config_file);
        // std::cout << "ConfigManager: Loading configuration from " << config_file << std::endl;
        
        // Load all configuration sections
        load_planning_config();
        load_strategy_config();
        load_skill_config();
        load_environment_config();
        load_system_config();
        load_optimization_config();
        
        validate_configuration();
        // std::cout << "ConfigManager: Configuration loaded successfully" << std::endl;
        
    } catch (const std::exception& e) {
        // std::cerr << "ConfigManager: Failed to load config file '" << config_file 
                  // << "': " << e.what() << std::endl;
        // std::cout << "ConfigManager: Using default configuration" << std::endl;
        validate_configuration();
    }
}

ConfigManager::ConfigManager() {
    // Use default values (already set in struct definitions)
    validate_configuration();
}

ConfigManager::~ConfigManager() = default;

void ConfigManager::load_planning_config() {
    if (!loader_) return;
    
    if (loader_->has_key("planning.high_level_resolution")) {
        planning_.high_level_resolution = loader_->get_double("planning.high_level_resolution");
    }
    if (loader_->has_key("planning.skill_level_resolution")) {
        planning_.skill_level_resolution = loader_->get_double("planning.skill_level_resolution");
    }
    if (loader_->has_key("planning.max_iterations")) {
        planning_.max_planning_iterations = loader_->get_int("planning.max_iterations");
    }
    if (loader_->has_key("planning.verbose")) {
        planning_.verbose_planning = loader_->get_bool("planning.verbose");
    }
    
    // Robot size (array)
    if (loader_->has_key("planning.robot_size")) {
        planning_.robot_size = loader_->get_vector("planning.robot_size");
    }
    
    // Grid limits
    if (loader_->has_key("planning.max_bfs_queue")) {
        planning_.max_bfs_queue = loader_->get_int("planning.max_bfs_queue");
    }
    if (loader_->has_key("planning.max_grid_width")) {
        planning_.max_grid_width = loader_->get_int("planning.max_grid_width");
    }
    if (loader_->has_key("planning.max_grid_height")) {
        planning_.max_grid_height = loader_->get_int("planning.max_grid_height");
    }
    if (loader_->has_key("planning.max_changes")) {
        planning_.max_changes = loader_->get_int("planning.max_changes");
    }
    
    // Thresholds
    if (loader_->has_key("planning.position_threshold")) {
        planning_.position_threshold = loader_->get_double("planning.position_threshold");
    }
    if (loader_->has_key("planning.rotation_threshold")) {
        planning_.rotation_threshold = loader_->get_double("planning.rotation_threshold");
    }
    if (loader_->has_key("planning.small_rotation_limit")) {
        planning_.small_rotation_limit = loader_->get_double("planning.small_rotation_limit");
    }
}

void ConfigManager::load_strategy_config() {
    if (!loader_) return;
    
    // Random strategy parameters
    if (loader_->has_key("strategy.random.min_goal_distance")) {
        strategy_.min_goal_distance = loader_->get_double("strategy.random.min_goal_distance");
    }
    if (loader_->has_key("strategy.random.max_goal_distance")) {
        strategy_.max_goal_distance = loader_->get_double("strategy.random.max_goal_distance");
    }
    if (loader_->has_key("strategy.random.max_goal_attempts")) {
        strategy_.max_goal_attempts = loader_->get_int("strategy.random.max_goal_attempts");
    }
    if (loader_->has_key("strategy.random.max_object_retries")) {
        strategy_.max_object_retries = loader_->get_int("strategy.random.max_object_retries");
    }
    
    // ML strategy parameters
    if (loader_->has_key("strategy.ml.zmq_endpoint")) {
        strategy_.zmq_endpoint = loader_->get_string("strategy.ml.zmq_endpoint");
    }
    if (loader_->has_key("strategy.ml.timeout_ms")) {
        strategy_.zmq_timeout_ms = loader_->get_int("strategy.ml.timeout_ms");
    }
    if (loader_->has_key("strategy.ml.fallback_to_random")) {
        strategy_.fallback_to_random = loader_->get_bool("strategy.ml.fallback_to_random");
    }
}

void ConfigManager::load_skill_config() {
    if (!loader_) return;
    
    if (loader_->has_key("skill.max_push_steps")) {
        skill_.max_push_steps = loader_->get_int("skill.max_push_steps");
    }
    // std::cout << "ConfigManager: Checking for key 'skill.max_mpc_iterations'" << std::endl;
    if (loader_->has_key("skill.max_mpc_iterations")) {
        skill_.max_mpc_iterations = loader_->get_int("skill.max_mpc_iterations");
        // std::cout << "ConfigManager: Loaded skill.max_mpc_iterations = " << skill_.max_mpc_iterations << std::endl;
    } else {
        // std::cout << "ConfigManager: skill.max_mpc_iterations key not found, using default = " << skill_.max_mpc_iterations << std::endl;
        // Debug: Let's see what keys are actually available
        // std::cout << "ConfigManager: Available keys with 'skill' prefix:" << std::endl;
        // This is a debug hack - we'll check some known keys to see the pattern
        if (loader_->has_key("skill.max_push_steps")) {
            // std::cout << "  - skill.max_push_steps: FOUND" << std::endl;
        } else {
            // std::cout << "  - skill.max_push_steps: NOT FOUND" << std::endl;
        }
    }
    if (loader_->has_key("skill.control_steps_per_push")) {
        skill_.control_steps_per_push = loader_->get_int("skill.control_steps_per_push");
    }
    if (loader_->has_key("skill.force_scaling")) {
        skill_.force_scaling = loader_->get_double("skill.force_scaling");
    }
    
    // Execution parameters
    if (loader_->has_key("skill.goal_tolerance")) {
        skill_.goal_tolerance = loader_->get_double("skill.goal_tolerance");
    }
    if (loader_->has_key("skill.stuck_threshold")) {
        skill_.stuck_threshold = loader_->get_double("skill.stuck_threshold");
    }
    if (loader_->has_key("skill.max_stuck_iterations")) {
        skill_.max_stuck_iterations = loader_->get_int("skill.max_stuck_iterations");
    }
    if (loader_->has_key("skill.check_object_collision")) {
        skill_.check_object_collision = loader_->get_bool("skill.check_object_collision");
    }

    // Controller-level stuck detection tuning
    if (loader_->has_key("skill.stuck_check_stride")) {
        skill_.stuck_check_stride = loader_->get_int("skill.stuck_check_stride");
    }
    if (loader_->has_key("skill.controller_stuck_threshold")) {
        skill_.controller_stuck_threshold = loader_->get_int("skill.controller_stuck_threshold");
    }
    if (loader_->has_key("skill.controller_min_position_change")) {
        skill_.controller_min_position_change = loader_->get_double("skill.controller_min_position_change");
    }
    if (loader_->has_key("skill.controller_min_angle_change")) {
        skill_.controller_min_angle_change = loader_->get_double("skill.controller_min_angle_change");
    }

    // Object interaction
    if (loader_->has_key("skill.object_clearance")) {
        skill_.object_clearance = loader_->get_double("skill.object_clearance");
    }
    
    // Edge point sampling - prefer points_per_face, fallback to num_edge_points
    if (loader_->has_key("skill.points_per_face")) {
        skill_.points_per_face = loader_->get_int("skill.points_per_face");
        skill_.points_per_face = std::clamp(skill_.points_per_face, 1, 16); // Respect capacity limits
        skill_.num_edge_points = 4 * skill_.points_per_face; // Derive total for backward compatibility
    } else if (loader_->has_key("skill.num_edge_points")) {
        skill_.num_edge_points = loader_->get_int("skill.num_edge_points");
        skill_.points_per_face = std::max(1, skill_.num_edge_points / 4); // Derive per-face
    }
    // push_force_magnitude parameter removed - unused (force_scaling used instead)
}

void ConfigManager::load_environment_config() {
    if (!loader_) return;
    
    // Environment bounds - now calculated dynamically from MuJoCo XML via NAMOEnvironment::get_environment_bounds()
    // Bounds loading removed - use environment.get_environment_bounds() instead
    
    // Memory limits
    if (loader_->has_key("environment.max_static_objects")) {
        environment_.max_static_objects = loader_->get_int("environment.max_static_objects");
    }
    if (loader_->has_key("environment.max_movable_objects")) {
        environment_.max_movable_objects = loader_->get_int("environment.max_movable_objects");
    }
    if (loader_->has_key("environment.max_actions")) {
        environment_.max_actions = loader_->get_int("environment.max_actions");
    }
    if (loader_->has_key("environment.max_motion_primitives")) {
        environment_.max_motion_primitives = loader_->get_int("environment.max_motion_primitives");
    }
    if (loader_->has_key("environment.max_planning_nodes")) {
        environment_.max_planning_nodes = loader_->get_int("environment.max_planning_nodes");
    }
    
    // Logging
    if (loader_->has_key("environment.log_buffer_size")) {
        environment_.log_buffer_size = loader_->get_int("environment.log_buffer_size");
    }
    if (loader_->has_key("environment.enable_logging")) {
        environment_.enable_state_logging = loader_->get_bool("environment.enable_logging");
    }
    if (loader_->has_key("environment.log_directory")) {
        environment_.log_directory = loader_->get_string("environment.log_directory");
    }
}

void ConfigManager::load_system_config() {
    if (!loader_) return;
    
    // Performance options
    if (loader_->has_key("system.enable_visualization")) {
        system_.enable_visualization = loader_->get_bool("system.enable_visualization");
    }
    if (loader_->has_key("system.enable_profiling")) {
        system_.enable_profiling = loader_->get_bool("system.enable_profiling");
    }
    if (loader_->has_key("system.num_threads")) {
        system_.num_threads = loader_->get_int("system.num_threads");
    }
    
    // File paths
    if (loader_->has_key("system.motion_primitives_file")) {
        system_.motion_primitives_file = loader_->get_string("system.motion_primitives_file");
    }
    if (loader_->has_key("system.default_scene_file")) {
        system_.default_scene_file = loader_->get_string("system.default_scene_file");
    }
    
    // Data collection
    if (loader_->has_key("system.collect_training_data")) {
        system_.collect_training_data = loader_->get_bool("system.collect_training_data");
    }
    if (loader_->has_key("system.training_data_directory")) {
        system_.training_data_directory = loader_->get_string("system.training_data_directory");
    }
}

void ConfigManager::load_optimization_config() {
    if (!loader_) return;
    
    // Sequence optimization settings
    if (loader_->has_key("optimization.enable_sequence_optimization")) {
        optimization_.enable_sequence_optimization = loader_->get_bool("optimization.enable_sequence_optimization");
    }
    if (loader_->has_key("optimization.default_method")) {
        optimization_.default_method = loader_->get_int("optimization.default_method");
    }
    if (loader_->has_key("optimization.timeout_seconds")) {
        optimization_.timeout_seconds = loader_->get_double("optimization.timeout_seconds");
    }
    if (loader_->has_key("optimization.max_sequence_length")) {
        optimization_.max_sequence_length = loader_->get_int("optimization.max_sequence_length");
    }
    
    // Performance settings
    if (loader_->has_key("optimization.max_sequences_tested")) {
        optimization_.max_sequences_tested = loader_->get_int("optimization.max_sequences_tested");
    }
    if (loader_->has_key("optimization.enable_optimization_logging")) {
        optimization_.enable_optimization_logging = loader_->get_bool("optimization.enable_optimization_logging");
    }
    if (loader_->has_key("optimization.fallback_on_timeout")) {
        optimization_.fallback_on_timeout = loader_->get_bool("optimization.fallback_on_timeout");
    }
}

void ConfigManager::validate_configuration() const {
    // Validate critical parameters
    if (planning_.high_level_resolution <= 0 || planning_.skill_level_resolution <= 0) {
        throw std::invalid_argument("Planning resolution must be positive");
    }
    
    if (planning_.max_planning_iterations <= 0) {
        throw std::invalid_argument("Max planning iterations must be positive");
    }
    
    if (strategy_.min_goal_distance >= strategy_.max_goal_distance) {
        throw std::invalid_argument("Min goal distance must be less than max goal distance");
    }
    
    if (skill_.max_push_steps <= 0 || skill_.control_steps_per_push <= 0) {
        throw std::invalid_argument("Push and control steps must be positive");
    }
    
    if (planning_.robot_size.size() != 2) {
        throw std::invalid_argument("Planning robot size vector must have 2 dimensions");
    }
    
    // Environment bounds validation removed - bounds now calculated dynamically
}

void ConfigManager::print_configuration() const {
    // std::cout << "=== NAMO Configuration (Using Defaults) ===" << std::endl;
    
    // std::cout << "\nPlanning:" << std::endl;
    // std::cout << "  High-level resolution: " << planning_.high_level_resolution << "m" << std::endl;
    // std::cout << "  Skill-level resolution: " << planning_.skill_level_resolution << "m" << std::endl;
    // std::cout << "  Max iterations: " << planning_.max_planning_iterations << std::endl;
    // std::cout << "  Robot size: [" << planning_.robot_size[0] << ", " << planning_.robot_size[1] << "]" << std::endl;
    // std::cout << "  Verbose: " << (planning_.verbose_planning ? "enabled" : "disabled") << std::endl;
    
    // std::cout << "\nStrategy (Random):" << std::endl;
    // std::cout << "  Goal distance: [" << strategy_.min_goal_distance << ", " << strategy_.max_goal_distance << "]m" << std::endl;
    // std::cout << "  Max goal attempts: " << strategy_.max_goal_attempts << std::endl;
    // std::cout << "  Max object retries: " << strategy_.max_object_retries << std::endl;
    
    // std::cout << "\nSkill:" << std::endl;
    // std::cout << "  Max push steps: " << skill_.max_push_steps << std::endl;
    // std::cout << "  Control steps per push: " << skill_.control_steps_per_push << std::endl;
    // std::cout << "  Goal tolerance: " << skill_.goal_tolerance << "m" << std::endl;
    // std::cout << "  Execution timeout: " << skill_.execution_timeout_seconds << "s" << std::endl;
    
    // std::cout << "\nEnvironment:" << std::endl;
    // std::cout << "  Bounds: calculated dynamically from MuJoCo XML" << std::endl;
    // std::cout << "  Max objects: " << environment_.max_static_objects << " static, " 
              // << environment_.max_movable_objects << " movable" << std::endl;
    
    // std::cout << "\nSystem:" << std::endl;
    // std::cout << "  Visualization: " << (system_.enable_visualization ? "enabled" : "disabled") << std::endl;
    // std::cout << "  Motion primitives: " << system_.motion_primitives_file << std::endl;
    // std::cout << "  Default scene: " << system_.default_scene_file << std::endl;
}

bool ConfigManager::validate_paths() const {
    bool all_valid = true;
    
    // Check motion primitives file
    if (!std::filesystem::exists(system_.motion_primitives_file)) {
        std::cerr << "Warning: Motion primitives file not found: " << system_.motion_primitives_file << std::endl;
        all_valid = false;
    }
    
    // Check default scene file
    if (!std::filesystem::exists(system_.default_scene_file)) {
        std::cerr << "Warning: Default scene file not found: " << system_.default_scene_file << std::endl;
        all_valid = false;
    }
    
    // Check log directory
    if (environment_.enable_state_logging && !std::filesystem::exists(environment_.log_directory)) {
        // std::cout << "Creating log directory: " << environment_.log_directory << std::endl;
        std::filesystem::create_directories(environment_.log_directory);
    }
    
    return all_valid;
}

std::unique_ptr<ConfigManager> ConfigManager::create_default() {
    return std::make_unique<ConfigManager>();
}

std::unique_ptr<ConfigManager> ConfigManager::create_from_file(const std::string& config_file) {
    return std::make_unique<ConfigManager>(config_file);
}

} // namespace namo