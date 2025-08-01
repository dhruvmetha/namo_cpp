#pragma once

#include "core/types.hpp"
#include <string>
#include <memory>

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
    
#ifdef HAVE_GLFW
    // Mouse interaction callbacks
    static void mouse_button_callback(GLFWwindow* window, int button, int act, int mods);
    static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    
    void mouse_button(int button, int act, int mods);
    void mouse_move(double xpos, double ypos);
    void scroll(double yoffset);
#endif
};

} // namespace namo