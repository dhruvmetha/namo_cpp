#pragma once

#include "core/types.hpp"
#include "core/mujoco_wrapper.hpp"
#include "robot/robot_adapter.hpp"
#include "config/config_manager.hpp"
#include <memory>
#include <fstream>
#include <map>
#include <vector>

namespace namo {

/**
 * @brief NAMO Environment with high-performance object state management
 * 
 * Manages robot, static obstacles, and movable objects using pre-allocated
 * fixed-size containers for zero-allocation runtime performance.
 */
class NAMOEnvironment {
public:
    /**
     * @brief Constructor
     * @param xml_path Path to MuJoCo XML model file
     * @param visualize Enable visualization
     * @param enable_logging Enable state logging
     */
    NAMOEnvironment(const std::string& xml_path, bool visualize = false, bool enable_logging = false);

    /// Config-aware constructor: creates the correct RobotAdapter from config.robot_type
    NAMOEnvironment(const std::string& xml_path, std::shared_ptr<ConfigManager> config,
                    bool visualize = false, bool enable_logging = false);

    /**
     * @brief Destructor
     */
    ~NAMOEnvironment();
    
    // Simulation control
    void step(const Control& control, double dt);
    void step_simulation();
    void reset();
    void update_object_states();
    
    // State management for optimization
    void save_current_state();
    void restore_saved_state();
    void reset_to_initial_state();
    
    // Full simulation state for zero-allocation save/restore
    struct FullSimState {
        static constexpr size_t MAX_QPOS = 500;  // Adjust based on model complexity
        static constexpr size_t MAX_QVEL = 500;
        
        std::array<double, MAX_QPOS> qpos;
        std::array<double, MAX_QVEL> qvel;
        int nq = 0;  // Actual size used
        int nv = 0;
        
        FullSimState() { qpos.fill(0.0); qvel.fill(0.0); }
    };
    
    // Full state management (zero-allocation)
    FullSimState get_full_state() const;
    void set_full_state(const FullSimState& state);
    void save_full_state();
    void restore_full_state();
    
    // State management
    void set_robot_position(const std::array<double, 2>& pos);
    void set_robot_position(const std::array<double, 3>& pos);
    /// Teleport robot to (x, y, theta) in world frame. Uses adapter for robot-specific behavior.
    void set_robot_se2(double x, double y, double theta);
    void set_zero_velocity();
    void enable_logging();
    void disable_logging();

    // Robot control
    void apply_robot_control(double control_x, double control_y);
    void set_robot_control(double control_x, double control_y);
    void apply_control(double control_x, double control_y, double dt);

    /// Access the robot adapter (for push controller, skill, etc.)
    const RobotAdapter* get_robot_adapter() const { return robot_adapter_.get(); }
    
    // Environment bounds
    std::vector<double> get_environment_bounds() const;
    std::vector<double> get_random_state() const;
    
    // Object accessors
    const std::array<ObjectInfo, 100>& get_static_objects() const { return static_objects_; }
    const std::array<ObjectInfo, 100>& get_movable_objects() const { return movable_objects_; }
    size_t get_num_static() const { return num_static_; }
    size_t get_num_movable() const { return num_movable_; }
    
    const ObjectInfo& get_robot_info() const { return robot_info_; }
    std::array<double, 2> get_robot_planning_half_extents() const {
        return {robot_info_.size[0], robot_info_.size[1]};
    }
    const ObjectState* get_robot_state() const { return &robot_state_; }
    
    // Object state queries
    const ObjectInfo* get_object_info(const std::string& name) const;
    const ObjectState* get_object_state(const std::string& name) const;
    const std::unordered_map<std::string, ObjectState>& get_all_object_states() const { return object_states_; }
    
    // Batch object info for efficient access (returns all immutable object data)
    std::map<std::string, std::map<std::string, double>> get_all_object_info() const;
    
	    // Goal management
	    void set_robot_goal(const std::array<double, 2>& goal) { robot_goal_ = goal; }
	    std::array<double, 2> get_robot_goal() const { return robot_goal_; }

	    // Visualization: some XMLs include a fixed `<site name="goal"...>` marker.
	    // For region-opening visual debugging, it can be confusing (region-opening cares
	    // about sampled neighbour-region goals, not the original goal site).
	    void set_goal_site_visible(bool visible) {
	        if (!sim_) return;
	        if (visible) {
	            sim_->set_site_rgba("goal", {0.0f, 1.0f, 0.0f, 0.5f});
	        } else {
	            sim_->set_site_rgba("goal", {0.0f, 1.0f, 0.0f, 0.0f});
	        }
	    }
    
    // Visualization
    void visualize_edge_reachability(
        const std::string& object_name,
        const std::vector<int>& reachable_edges,
        double edge_offset_margin_m = 0.020);
                                   
    // Visualization for goal marker (like old MuJoCo implementation)
    void visualize_goal_marker(
        const std::array<double, 3>& goal_position,
        const std::array<float, 4>& color = {0.0f, 1.0f, 0.0f, 1.0f},
        double goal_radius_m = -1.0);
                              
    // Visualization for object goal marker with object-specific size
    void visualize_object_goal_marker(const std::array<double, 3>& goal_position,
                                     const std::array<double, 3>& object_size,
                                     double theta = 0.0,
                                     const std::array<float, 4>& color = {0.0f, 0.8f, 1.0f, 0.5f});

    // Clear the object target marker
    void clear_object_target_marker() { sim_->clear_object_target_marker(); }

    // Collision detection
    bool is_in_collision() const { return sim_->in_collision(); }
    bool bodies_in_collision(const std::string& body1, const std::string& body2) const {
        return sim_->bodies_in_collision(body1, body2);
    }
    
    // Visualization and rendering
    void render() { sim_->render(); }
    bool should_close() const { return sim_->should_close(); }
    void set_camera_position(double distance, double azimuth, double elevation) {
        sim_->set_camera_position(distance, azimuth, elevation);
    }
    void set_camera_lookat(const std::array<double, 3>& lookat) {
        sim_->set_camera_lookat(lookat);
    }

    // Video recording interface
    void start_recording(int width = 640, int height = 480,
                        int capture_frequency = 100, size_t max_frames = 10000) {
        sim_->start_recording(width, height, capture_frequency, max_frames);
    }
    void stop_recording() { sim_->stop_recording(); }
    bool is_recording() const { return sim_->is_recording(); }
    size_t get_frame_count() const { return sim_->get_frame_count(); }
    const std::vector<std::vector<unsigned char>>& get_captured_frames() const {
        return sim_->get_captured_frames();
    }
    const std::vector<unsigned char>& get_captured_frame(size_t idx) const {
        return sim_->get_captured_frame(idx);
    }
    void clear_captured_frames() { sim_->clear_captured_frames(); }
    int get_frame_width() const { return sim_->get_frame_width(); }
    int get_frame_height() const { return sim_->get_frame_height(); }

    // Direct MuJoCo access for advanced usage
    OptimizedMujocoWrapper* get_mujoco_wrapper() { return sim_.get(); }
    const OptimizedMujocoWrapper* get_mujoco_wrapper() const { return sim_.get(); }
    
    // Configuration
    const std::string& get_config_name() const { return config_name_; }
    bool get_logging_enabled() const { return logging_enabled_; }
    
    // Data collection
    void save_objects_to_file(const std::string& filename) const;
    void increment_wavefront_id() { wavefront_id_++; }
    
private:
    // MuJoCo simulation
    std::unique_ptr<OptimizedMujocoWrapper> sim_;
    std::shared_ptr<ConfigManager> config_;

    // Robot adapter (abstracts robot-specific position/control/identity)
    std::unique_ptr<RobotAdapter> robot_adapter_;
    
    // Fixed-size object storage
    static constexpr size_t MAX_STATIC_OBJECTS = 100;
    static constexpr size_t MAX_MOVABLE_OBJECTS = 100;
    
    std::array<ObjectInfo, MAX_STATIC_OBJECTS> static_objects_;
    std::array<ObjectInfo, MAX_MOVABLE_OBJECTS> movable_objects_;
    size_t num_static_ = 0;
    size_t num_movable_ = 0;
    
    // Object state tracking
    std::unordered_map<std::string, ObjectState> object_states_;
    ObjectInfo robot_info_;
    ObjectState robot_state_;
    
    // Robot properties
    int robot_id_ = -1;
    std::array<double, 3> init_robot_pos_ = {0.0, 0.0, 0.0};
    std::array<double, 2> robot_goal_ = {0.0, 0.0};
    
    // Configuration
    std::string config_name_;
    
    // High-performance logging
    bool logging_enabled_ = false;
    std::ofstream state_log_file_;
    bool header_written_ = false;
    unsigned long frame_count_ = 0;
    int wavefront_id_ = -1;
    int state_log_idx_ = 0;
    
    // State management for optimization
    bool has_saved_state_ = false;
    std::vector<double> saved_qpos_;
    std::vector<double> saved_qvel_;
    std::vector<double> initial_qpos_;
    std::vector<double> initial_qvel_;
    
    // Full state management (zero-allocation)
    FullSimState saved_full_state_;
    bool has_saved_full_state_ = false;
    
    static constexpr size_t LOG_BUFFER_SIZE = 100000;
    std::array<char, LOG_BUFFER_SIZE> log_buffer_;
    size_t log_position_ = 0;
    
    // Initialization helpers
    void init_robot_from_adapter();   // Read robot info using adapter
    bool derive_robot_footprint_from_collision_geometry(std::array<double, 3>& derived_size) const;
    void process_environment_objects();
    void warm_up();
    
    // Logging helpers
    void log_state();
    void flush_log_buffer();
    
    // Object management helpers
    void add_static_object(const ObjectInfo& obj);
    void add_movable_object(const ObjectInfo& obj);
};

} // namespace namo
