#include "core/mujoco_wrapper.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace namo {

OptimizedMujocoWrapper::OptimizedMujocoWrapper(const std::string& model_path, bool visualize) 
    : visualize_(visualize), owns_model_(true) {
    
    // Load model from file
    char error[1000] = "Could not load binary model";
    m_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
    
    if (!m_) {
        throw std::runtime_error("Failed to load MuJoCo model: " + std::string(error));
    }
    
    // Create data structure
    d_ = mj_makeData(m_);
    if (!d_) {
        mj_deleteModel(m_);
        throw std::runtime_error("Failed to create MuJoCo data structure");
    }
}

OptimizedMujocoWrapper::OptimizedMujocoWrapper(mjModel* model, bool visualize) 
    : m_(model), visualize_(visualize), owns_model_(false) {
    
    if (!m_) {
        throw std::runtime_error("Invalid MuJoCo model provided");
    }
    
    // Create data structure
    d_ = mj_makeData(m_);
    if (!d_) {
        throw std::runtime_error("Failed to create MuJoCo data structure");
    }
}

OptimizedMujocoWrapper::~OptimizedMujocoWrapper() {
    cleanup_visualization();
    
    if (d_) {
        mj_deleteData(d_);
    }
    
    if (m_ && owns_model_) {
        mj_deleteModel(m_);
    }
}

void OptimizedMujocoWrapper::initialize() {
    // Forward kinematics and dynamics
    mj_kinematics(m_, d_);
    mj_forward(m_, d_);
    
    // Initialize visualization if requested
    if (visualize_) {
        init_visualization();
    }
}

void OptimizedMujocoWrapper::step() {
    mj_step(m_, d_);
    
    if (visualize_) {
        render();
    }
}

void OptimizedMujocoWrapper::reset() {
    mj_resetData(m_, d_);
    mj_forward(m_, d_);
}

void OptimizedMujocoWrapper::set_robot_position(const std::array<double, 2>& pos) {
    if (m_->nq >= 2) {
        d_->qpos[0] = pos[0];
        d_->qpos[1] = pos[1];
        mj_forward(m_, d_);
    }
}

void OptimizedMujocoWrapper::set_robot_position(const std::array<double, 3>& pos) {
    if (m_->nq >= 3) {
        d_->qpos[0] = pos[0];
        d_->qpos[1] = pos[1];
        d_->qpos[2] = pos[2];
        mj_forward(m_, d_);
    }
}

void OptimizedMujocoWrapper::set_robot_velocity(const std::array<double, 2>& vel) {
    if (m_->nv >= 2) {
        d_->qvel[0] = vel[0];
        d_->qvel[1] = vel[1];
        mj_forward(m_, d_);
    }
}

void OptimizedMujocoWrapper::set_zero_velocity() {
    for (int i = 0; i < m_->nv; i++) {
        d_->qvel[i] = 0.0;
    }
    mj_forward(m_, d_);
}

void OptimizedMujocoWrapper::set_control(const Control& control) {
    int nctrl = std::min(static_cast<int>(control.size()), m_->nu);
    for (int i = 0; i < nctrl; i++) {
        d_->ctrl[i] = control[i];
    }
}

void OptimizedMujocoWrapper::set_control(const double* ctrl, int nctrl) {
    int n = std::min(nctrl, m_->nu);
    for (int i = 0; i < n; i++) {
        d_->ctrl[i] = ctrl[i];
    }
}

void OptimizedMujocoWrapper::set_robot_control(double control_x, double control_y) {
    // Assume first two actuators control robot's x,y movement
    if (m_->nu >= 2) {
        d_->ctrl[0] = control_x;
        d_->ctrl[1] = control_y;
    }
}

void OptimizedMujocoWrapper::set_zero_control() {
    for (int i = 0; i < m_->nu; i++) {
        d_->ctrl[i] = 0.0;
    }
}

void OptimizedMujocoWrapper::get_robot_position(std::array<double, 2>& pos) const {
    pos[0] = (m_->nq >= 1) ? d_->qpos[0] : 0.0;
    pos[1] = (m_->nq >= 2) ? d_->qpos[1] : 0.0;
}

void OptimizedMujocoWrapper::get_robot_position(std::array<double, 3>& pos) const {
    pos[0] = (m_->nq >= 1) ? d_->qpos[0] : 0.0;
    pos[1] = (m_->nq >= 2) ? d_->qpos[1] : 0.0;
    pos[2] = (m_->nq >= 3) ? d_->qpos[2] : 0.0;
}

void OptimizedMujocoWrapper::get_robot_velocity(std::array<double, 2>& vel) const {
    vel[0] = (m_->nv >= 1) ? d_->qvel[0] : 0.0;
    vel[1] = (m_->nv >= 2) ? d_->qvel[1] : 0.0;
}

void OptimizedMujocoWrapper::get_state(State& state) const {
    state.clear();
    
    // Add positions
    for (int i = 0; i < m_->nq && state.size() < 10; i++) {
        state.push_back(d_->qpos[i]);
    }
    
    // Add velocities
    for (int i = 0; i < m_->nv && state.size() < 10; i++) {
        state.push_back(d_->qvel[i]);
    }
}

void OptimizedMujocoWrapper::set_state(const State& state) {
    size_t idx = 0;
    
    // Set positions
    for (int i = 0; i < m_->nq && idx < state.size(); i++, idx++) {
        d_->qpos[i] = state[idx];
    }
    
    // Set velocities
    for (int i = 0; i < m_->nv && idx < state.size(); i++, idx++) {
        d_->qvel[i] = state[idx];
    }
    
    mj_forward(m_, d_);
}

bool OptimizedMujocoWrapper::get_body_position(const std::string& name, std::array<double, 3>& pos) const {
    int id = mj_name2id(m_, mjOBJ_BODY, name.c_str());
    if (id < 0) return false;
    
    pos[0] = d_->xpos[3 * id];
    pos[1] = d_->xpos[3 * id + 1];
    pos[2] = d_->xpos[3 * id + 2];
    return true;
}

bool OptimizedMujocoWrapper::get_body_quaternion(const std::string& name, std::array<double, 4>& quat) const {
    int id = mj_name2id(m_, mjOBJ_BODY, name.c_str());
    if (id < 0) return false;
    
    quat[0] = d_->xquat[4 * id];     // w
    quat[1] = d_->xquat[4 * id + 1]; // x
    quat[2] = d_->xquat[4 * id + 2]; // y
    quat[3] = d_->xquat[4 * id + 3]; // z
    return true;
}

bool OptimizedMujocoWrapper::get_body_pose(const std::string& name, std::array<double, 7>& pose) const {
    int id = mj_name2id(m_, mjOBJ_BODY, name.c_str());
    if (id < 0) return false;
    
    // Position
    pose[0] = d_->xpos[3 * id];
    pose[1] = d_->xpos[3 * id + 1];
    pose[2] = d_->xpos[3 * id + 2];
    
    // Quaternion
    pose[3] = d_->xquat[4 * id];     // w
    pose[4] = d_->xquat[4 * id + 1]; // x
    pose[5] = d_->xquat[4 * id + 2]; // y
    pose[6] = d_->xquat[4 * id + 3]; // z
    
    return true;
}

bool OptimizedMujocoWrapper::get_geom_position(const std::string& name, std::array<double, 3>& pos) const {
    int id = mj_name2id(m_, mjOBJ_GEOM, name.c_str());
    if (id < 0) return false;
    
    pos[0] = d_->geom_xpos[3 * id];
    pos[1] = d_->geom_xpos[3 * id + 1];
    pos[2] = d_->geom_xpos[3 * id + 2];
    return true;
}

bool OptimizedMujocoWrapper::get_geom_quaternion(const std::string& name, std::array<double, 4>& quat) const {
    int id = mj_name2id(m_, mjOBJ_GEOM, name.c_str());
    if (id < 0) return false;
    
    // Convert rotation matrix to quaternion
    mjtNum* mat = d_->geom_xmat + 9 * id;
    mju_mat2Quat(quat.data(), mat);
    return true;
}

bool OptimizedMujocoWrapper::get_geom_pose(const std::string& name, std::array<double, 7>& pose) const {
    int id = mj_name2id(m_, mjOBJ_GEOM, name.c_str());
    if (id < 0) return false;
    
    // Position
    pose[0] = d_->geom_xpos[3 * id];
    pose[1] = d_->geom_xpos[3 * id + 1];
    pose[2] = d_->geom_xpos[3 * id + 2];
    
    // Convert rotation matrix to quaternion
    mjtNum* mat = d_->geom_xmat + 9 * id;
    mju_mat2Quat(&pose[3], mat);
    
    return true;
}

bool OptimizedMujocoWrapper::in_collision() const {
    return d_->ncon > 0;
}

bool OptimizedMujocoWrapper::bodies_in_collision(const std::string& body1, const std::string& body2) const {
    int id1 = mj_name2id(m_, mjOBJ_BODY, body1.c_str());
    int id2 = mj_name2id(m_, mjOBJ_BODY, body2.c_str());
    
    if (id1 < 0 || id2 < 0) return false;
    
    // Check all contacts
    for (int i = 0; i < d_->ncon; i++) {
        mjContact* contact = &d_->contact[i];
        int geom1 = contact->geom1;
        int geom2 = contact->geom2;
        
        int body1_id = m_->geom_bodyid[geom1];
        int body2_id = m_->geom_bodyid[geom2];
        
        if ((body1_id == id1 && body2_id == id2) || 
            (body1_id == id2 && body2_id == id1)) {
            return true;
        }
    }
    
    return false;
}

int OptimizedMujocoWrapper::get_body_id(const std::string& name) const {
    return mj_name2id(m_, mjOBJ_BODY, name.c_str());
}

int OptimizedMujocoWrapper::get_geom_id(const std::string& name) const {
    return mj_name2id(m_, mjOBJ_GEOM, name.c_str());
}

int OptimizedMujocoWrapper::get_joint_id(const std::string& name) const {
    return mj_name2id(m_, mjOBJ_JOINT, name.c_str());
}

int OptimizedMujocoWrapper::get_actuator_id(const std::string& name) const {
    return mj_name2id(m_, mjOBJ_ACTUATOR, name.c_str());
}

std::string OptimizedMujocoWrapper::get_body_name(int id) const {
    if (id < 0 || id >= m_->nbody) return "";
    const char* name = mj_id2name(m_, mjOBJ_BODY, id);
    return name ? std::string(name) : "";
}

std::string OptimizedMujocoWrapper::get_geom_name(int id) const {
    if (id < 0 || id >= m_->ngeom) return "";
    const char* name = mj_id2name(m_, mjOBJ_GEOM, id);
    return name ? std::string(name) : "";
}

#ifdef HAVE_GLFW

void OptimizedMujocoWrapper::init_visualization() {
    if (!glfwInit()) {
        std::cerr << "Warning: Failed to initialize GLFW. Visualization disabled." << std::endl;
        visualize_ = false;
        return;
    }
    
    window_ = glfwCreateWindow(1200, 900, "NAMO Simulation", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Warning: Failed to create GLFW window. Visualization disabled." << std::endl;
        glfwTerminate();
        visualize_ = false;
        return;
    }
    
    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);
    
    // Set up mouse callbacks
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, mouse_move_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    
    // Initialize MuJoCo visualization
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);
    
    // Disable contact force visualization
    opt_.flags[mjVIS_CONTACTFORCE] = 0;
    
    mjv_makeScene(m_, &scn_, 2000);
    mjr_makeContext(m_, &con_, mjFONTSCALE_150);
    
    // Set reasonable default camera position
    cam_.distance = 5.0;
    cam_.azimuth = 90.0;
    cam_.elevation = -20.0;
    cam_.lookat[0] = 0.0;
    cam_.lookat[1] = 0.0;
    cam_.lookat[2] = 0.0;
}

void OptimizedMujocoWrapper::cleanup_visualization() {
    if (window_) {
        mjv_freeScene(&scn_);
        mjr_freeContext(&con_);
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

void OptimizedMujocoWrapper::render() {
    if (!visualize_ || !window_) return;
    
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
    
    mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);
    
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

bool OptimizedMujocoWrapper::should_close() const {
    return window_ ? glfwWindowShouldClose(window_) : false;
}

void OptimizedMujocoWrapper::set_camera_position(double distance, double azimuth, double elevation) {
    cam_.distance = distance;
    cam_.azimuth = azimuth;
    cam_.elevation = elevation;
}

void OptimizedMujocoWrapper::set_camera_lookat(const std::array<double, 3>& lookat) {
    cam_.lookat[0] = lookat[0];
    cam_.lookat[1] = lookat[1];
    cam_.lookat[2] = lookat[2];
}

// Static callback functions
void OptimizedMujocoWrapper::mouse_button_callback(GLFWwindow* window, int button, int act, int mods) {
    OptimizedMujocoWrapper* wrapper = static_cast<OptimizedMujocoWrapper*>(glfwGetWindowUserPointer(window));
    if (wrapper) {
        wrapper->mouse_button(button, act, mods);
    }
}

void OptimizedMujocoWrapper::mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
    OptimizedMujocoWrapper* wrapper = static_cast<OptimizedMujocoWrapper*>(glfwGetWindowUserPointer(window));
    if (wrapper) {
        wrapper->mouse_move(xpos, ypos);
    }
}

void OptimizedMujocoWrapper::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    OptimizedMujocoWrapper* wrapper = static_cast<OptimizedMujocoWrapper*>(glfwGetWindowUserPointer(window));
    if (wrapper) {
        wrapper->scroll(yoffset);
    }
}

// Mouse interaction implementation
void OptimizedMujocoWrapper::mouse_button(int button, int act, int mods) {
    button_left_ = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle_ = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right_ = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    
    glfwGetCursorPos(window_, &lastx_, &lasty_);
}

void OptimizedMujocoWrapper::mouse_move(double xpos, double ypos) {
    if (!button_left_ && !button_middle_ && !button_right_) {
        return;
    }
    
    double dx = xpos - lastx_;
    double dy = ypos - lasty_;
    lastx_ = xpos;
    lasty_ = ypos;
    
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    
    bool mod_shift = (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
                     glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    
    mjtMouse action;
    if (button_right_) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left_) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }
    
    mjv_moveCamera(m_, action, dx/height, dy/height, &scn_, &cam_);
}

void OptimizedMujocoWrapper::scroll(double yoffset) {
    mjv_moveCamera(m_, mjMOUSE_ZOOM, 0, 0.05 * yoffset, &scn_, &cam_);
}

#else

void OptimizedMujocoWrapper::init_visualization() {
    std::cout << "Visualization not available (GLFW not found)" << std::endl;
    visualize_ = false;
}

void OptimizedMujocoWrapper::cleanup_visualization() {
    // No-op
}

void OptimizedMujocoWrapper::render() {
    // No-op
}

bool OptimizedMujocoWrapper::should_close() const {
    return false;
}

void OptimizedMujocoWrapper::set_camera_position(double distance, double azimuth, double elevation) {
    // No-op
}

void OptimizedMujocoWrapper::set_camera_lookat(const std::array<double, 3>& lookat) {
    // No-op
}

#endif

} // namespace namo