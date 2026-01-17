#pragma once

#include "core/types.hpp"
#include <string>
#include <memory>
#include <vector>

extern "C" {
#include "mujoco/mujoco.h"
}

#ifdef HAVE_GLFW
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#endif

namespace namo {

/**
 * @brief High-performance MuJoCo wrapper with minimal overhead
 * 
 * Direct MuJoCo API integration without abstraction layers
 * Pre-allocated buffers to avoid runtime allocations
 */
class OptimizedMujocoWrapper {
private:
    mjModel* m_ = nullptr;
    mjData* d_ = nullptr;
    bool visualize_;
    bool owns_model_ = true;
    
    // Pre-allocated buffers for state/control operations
    static constexpr size_t MAX_STATE_SIZE = 100;
    static constexpr size_t MAX_CONTROL_SIZE = 50;
    std::array<double, MAX_STATE_SIZE> state_buffer_;
    std::array<double, MAX_CONTROL_SIZE> control_buffer_;
    
#ifdef HAVE_GLFW
    // Visualization components
    GLFWwindow* window_ = nullptr;
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;
    
    // Mouse interaction state
    bool button_left_ = false;
    bool button_middle_ = false;
    bool button_right_ = false;
    double lastx_ = 0.0;
    double lasty_ = 0.0;
#endif
    
public:
    /**
     * @brief Construct wrapper from XML file path
     */
    OptimizedMujocoWrapper(const std::string& model_path, bool visualize = false);
    
    /**
     * @brief Construct wrapper from existing mjModel (doesn't take ownership)
     */
    OptimizedMujocoWrapper(mjModel* model, bool visualize = false);
    
    /**
     * @brief Destructor
     */
    ~OptimizedMujocoWrapper();
    
    // Disable copy/move to avoid accidental double-free
    OptimizedMujocoWrapper(const OptimizedMujocoWrapper&) = delete;
    OptimizedMujocoWrapper& operator=(const OptimizedMujocoWrapper&) = delete;
    OptimizedMujocoWrapper(OptimizedMujocoWrapper&&) = delete;
    OptimizedMujocoWrapper& operator=(OptimizedMujocoWrapper&&) = delete;
    
    /**
     * @brief Initialize the simulation after construction
     */
    void initialize();
    
    /**
     * @brief Step simulation forward
     */
    void step();
    
    /**
     * @brief Add visual markers to the scene
     * @param positions Array of marker positions [x, y, z]
     * @param colors Array of marker colors [r, g, b, a] (0-1 range)
     * @param sizes Array of marker sizes (radius)
     * @param count Number of markers to add
     */
    void add_visual_markers(const std::array<double, 3>* positions, 
                           const std::array<float, 4>* colors,
                           const double* sizes,
                           size_t count);
                           
    /**
     * @brief Set a goal marker (like old MuJoCo implementation)
     * @param position Goal position [x, y, z]
     * @param orientation Goal orientation [w, x, y, z] quaternion
     * @param size Goal size [x, y, z]
     * @param geom_type MuJoCo geometry type
     */
    void set_goal_marker(const std::array<double, 3>& position,
                        const std::array<double, 4>& orientation,
                        const std::array<double, 3>& size,
                        int geom_type);
                        
    /**
     * @brief Clear the goal marker
     */
    void clear_goal_marker();

    /**
     * @brief Set object target marker (separate from robot goal, with custom color)
     * @param position Target position [x, y, z]
     * @param orientation Target orientation [w, x, y, z] quaternion
     * @param size Target size [x, y, z]
     * @param color RGBA color [r, g, b, a]
     * @param geom_type MuJoCo geometry type
     */
    void set_object_target_marker(const std::array<double, 3>& position,
                                  const std::array<double, 4>& orientation,
                                  const std::array<double, 3>& size,
                                  const std::array<float, 4>& color,
                                  int geom_type);

    /**
     * @brief Clear the object target marker
     */
    void clear_object_target_marker();

    /**
     * @brief Reset simulation to initial state
     */
    void reset();
    
    /**
     * @brief Set robot position (assumes first 2 DOFs are x,y)
     */
    void set_robot_position(const std::array<double, 2>& pos);
    void set_robot_position(const std::array<double, 3>& pos);
    
    /**
     * @brief Set robot velocity (assumes first DOFs are translational)
     */
    void set_robot_velocity(const std::array<double, 2>& vel);
    void set_zero_velocity();
    
    /**
     * @brief Set control inputs
     */
    void set_control(const Control& control);
    void set_control(const double* ctrl, int nctrl);
    void set_robot_control(double control_x, double control_y);
    void set_zero_control();
    
    /**
     * @brief Get robot state
     */
    void get_robot_position(std::array<double, 2>& pos) const;
    void get_robot_position(std::array<double, 3>& pos) const;
    void get_robot_velocity(std::array<double, 2>& vel) const;
    
    /**
     * @brief Get full system state
     */
    void get_state(State& state) const;
    void set_state(const State& state);
    
    /**
     * @brief Get body position and orientation by name
     */
    bool get_body_position(const std::string& name, std::array<double, 3>& pos) const;
    bool get_body_quaternion(const std::string& name, std::array<double, 4>& quat) const;
    bool get_body_pose(const std::string& name, std::array<double, 7>& pose) const;
    
    /**
     * @brief Get geom position and orientation by name
     */
    bool get_geom_position(const std::string& name, std::array<double, 3>& pos) const;
    bool get_geom_quaternion(const std::string& name, std::array<double, 4>& quat) const;
    bool get_geom_pose(const std::string& name, std::array<double, 7>& pose) const;
    
    /**
     * @brief Collision detection
     */
    bool in_collision() const;
    bool bodies_in_collision(const std::string& body1, const std::string& body2) const;
    
    /**
     * @brief Rendering and visualization
     */
    void render();
    bool should_close() const;
    void set_camera_position(double distance, double azimuth, double elevation);
    void set_camera_lookat(const std::array<double, 3>& lookat);
    
    /**
     * @brief Direct access to MuJoCo objects (for advanced usage)
     */
    mjModel* model() { return m_; }
    mjData* data() { return d_; }
    const mjModel* model() const { return m_; }
    const mjData* data() const { return d_; }
    
    /**
     * @brief Utility functions
     */
    int get_body_id(const std::string& name) const;
    int get_geom_id(const std::string& name) const;
    int get_joint_id(const std::string& name) const;
    int get_actuator_id(const std::string& name) const;
    
    std::string get_body_name(int id) const;
    std::string get_geom_name(int id) const;
    
    /**
     * @brief Performance monitoring
     */
    double get_simulation_time() const { return d_ ? d_->time : 0.0; }
    double get_timestep() const { return m_ ? m_->opt.timestep : 0.0; }
    
private:
    void init_visualization();
    void cleanup_visualization();
    
    // Goal marker storage (like old MuJoCo implementation)
    struct GoalMarker {
        bool active = false;
        std::array<double, 3> position = {0.0, 0.0, 0.0};
        std::array<double, 4> orientation = {1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> size = {0.1, 0.1, 0.1};
        int geom_type = 6; // mjGEOM_BOX
    } goal_marker_;

    // Object target marker (separate from robot goal, with custom color)
    struct ObjectTargetMarker {
        bool active = false;
        std::array<double, 3> position = {0.0, 0.0, 0.0};
        std::array<double, 4> orientation = {1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> size = {0.1, 0.1, 0.1};
        std::array<float, 4> color = {0.0f, 0.8f, 1.0f, 0.5f}; // Cyan by default
        int geom_type = 6; // mjGEOM_BOX
    } object_target_marker_;
    
#ifdef HAVE_GLFW
    // Mouse interaction callbacks
    static void mouse_button_callback(GLFWwindow* window, int button, int act, int mods);
    static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    void mouse_button(int button, int act, int mods);
    void mouse_move(double xpos, double ypos);
    void scroll(double yoffset);
#endif

    // ========== Video Recording / Frame Capture ==========
private:
    // Offscreen rendering state (works without GLFW window)
    bool offscreen_initialized_ = false;
    mjvCamera offscreen_cam_;
    mjvOption offscreen_opt_;
    mjvScene offscreen_scn_;
    mjrContext offscreen_con_;
    int offscreen_width_ = 640;
    int offscreen_height_ = 480;

    // Frame capture state
    bool recording_active_ = false;
    int capture_frequency_ = 100;  // Capture every N physics steps
    int physics_step_counter_ = 0;
    std::vector<std::vector<unsigned char>> captured_frames_;
    size_t max_frames_ = 10000;

public:
    /**
     * @brief Initialize offscreen rendering for frame capture
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @return true if initialization succeeded
     */
    bool init_offscreen_rendering(int width = 640, int height = 480);

    /**
     * @brief Clean up offscreen rendering resources
     */
    void cleanup_offscreen_rendering();

    /**
     * @brief Start recording frames during physics execution
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @param capture_frequency Capture frame every N physics steps
     * @param max_frames Maximum frames to capture (memory limit)
     */
    void start_recording(int width = 640, int height = 480,
                        int capture_frequency = 100, size_t max_frames = 10000);

    /**
     * @brief Stop recording frames
     */
    void stop_recording();

    /**
     * @brief Check if recording is active
     */
    bool is_recording() const { return recording_active_; }

    /**
     * @brief Notify that a physics step has occurred (call after each mj_step)
     * This will capture a frame if recording is active and frequency threshold is met
     */
    void notify_physics_step();

    /**
     * @brief Get captured frames
     * @return Vector of frames, each frame is RGB data (width * height * 3 bytes)
     */
    const std::vector<std::vector<unsigned char>>& get_captured_frames() const {
        return captured_frames_;
    }

    /**
     * @brief Clear captured frames to free memory
     */
    void clear_captured_frames();

    /**
     * @brief Get number of captured frames
     */
    size_t get_frame_count() const { return captured_frames_.size(); }

    /**
     * @brief Get frame dimensions
     */
    int get_frame_width() const { return offscreen_width_; }
    int get_frame_height() const { return offscreen_height_; }

private:
    /**
     * @brief Render current scene to offscreen buffer
     */
    void render_offscreen();

    /**
     * @brief Capture current frame from offscreen buffer
     * @return true if frame was captured successfully
     */
    bool capture_frame();
};

} // namespace namo