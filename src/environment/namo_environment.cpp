#include "environment/namo_environment.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <iomanip>

extern "C" {
#include "mujoco/mujoco.h"
}

namespace namo {

NAMOEnvironment::NAMOEnvironment(const std::string& xml_path, bool visualize, bool enable_logging) 
    : logging_enabled_(enable_logging) {
    
    // Create MuJoCo wrapper
    sim_ = std::make_unique<OptimizedMujocoWrapper>(xml_path, visualize);
    sim_->initialize();
    
    // Set reasonable camera defaults
    sim_->set_camera_lookat({0.0, 0.0, 0.0});
    sim_->set_camera_position(5.0, 90.0, -20.0);
    
    // Warm up simulation
    warm_up();
    
    // Extract config name from XML path
    std::filesystem::path xml_file_path(xml_path);
    config_name_ = xml_file_path.stem().string();
    
    // Get robot information first
    robot_id_ = sim_->get_geom_id("robot");
    if (robot_id_ >= 0) {
        robot_info_.body_id = sim_->get_body_id("robot");
        robot_info_.geom_id = robot_id_;
        robot_info_.name = "robot";
        robot_info_.is_static = false;
        
        // Get robot pose
        std::array<double, 7> robot_pose;
        if (sim_->get_geom_pose("robot", robot_pose)) {
            for (int i = 0; i < 3; i++) robot_info_.position[i] = robot_pose[i];
            for (int i = 0; i < 4; i++) robot_info_.quaternion[i] = robot_pose[i + 3];
        }
        
        // Get robot size from MuJoCo model
        mjModel* model = sim_->model();
        if (robot_id_ < model->ngeom) {
            for (int i = 0; i < 3; i++) {
                robot_info_.size[i] = model->geom_size[robot_id_ * 3 + i];
            }
        }
        
        init_robot_pos_ = robot_info_.position;
    }
    
    // Process environment objects
    process_environment_objects();
    
    // Initialize logging if enabled
    if (logging_enabled_) {
        state_log_file_.open("namo_state_log_" + std::to_string(state_log_idx_) + ".csv");
        if (!state_log_file_.is_open()) {
            std::cerr << "Warning: Could not open state log file. Logging disabled." << std::endl;
            logging_enabled_ = false;
        }
    }
    
    // Initial state update
    update_object_states();
    
    // std::cout << "NAMO Environment initialized:" << std::endl;
    // std::cout << "  Config: " << config_name_ << std::endl;
    // std::cout << "  Static objects: " << num_static_ << std::endl;
    // std::cout << "  Movable objects: " << num_movable_ << std::endl;
    // std::cout << "  Robot ID: " << robot_id_ << std::endl;
    // std::cout << "  Visualization: " << (visualize ? "enabled" : "disabled") << std::endl;
    // std::cout << "  Logging: " << (logging_enabled_ ? "enabled" : "disabled") << std::endl;
}

NAMOEnvironment::~NAMOEnvironment() {
    if (logging_enabled_ && state_log_file_.is_open()) {
        flush_log_buffer();
        state_log_file_.close();
    }
}

void NAMOEnvironment::warm_up() {
    // Step simulation a few times to stabilize physics
    for (int i = 0; i < 3; i++) {
        sim_->step();
    }
    
    // // Save initial state for optimization reset
    // State initial_state;
    // sim_->get_state(initial_state);
    // initial_qpos_.resize(initial_state.size());
    // for (size_t i = 0; i < initial_state.size(); ++i) {
    //     initial_qpos_[i] = initial_state[i];
    // }
    // initial_qvel_.clear();  // Not using separate velocity storage for now
}

void NAMOEnvironment::process_environment_objects() {
    num_static_ = 0;
    num_movable_ = 0;
    
    mjModel* model = sim_->model();
    
    // Iterate through all bodies
    for (int i = 0; i < model->nbody; i++) {
        const char* body_name_ptr = mj_id2name(model, mjOBJ_BODY, i);
        if (!body_name_ptr) continue;
        
        std::string body_name(body_name_ptr);
        
        // Skip robot and world bodies
        if (body_name == "robot" || body_name == "world") {
            continue;
        }
        
        // Find geoms associated with this body
        for (int j = 0; j < model->ngeom; j++) {
            if (model->geom_bodyid[j] == i) {
                ObjectInfo obj;
                obj.body_id = i;
                obj.geom_id = j;
                
                const char* geom_name_ptr = mj_id2name(model, mjOBJ_GEOM, j);
                obj.name = geom_name_ptr ? std::string(geom_name_ptr) : ("geom_" + std::to_string(j));
                
                // Get size
                for (int k = 0; k < 3; k++) {
                    obj.size[k] = model->geom_size[j * 3 + k];
                }
                
                // Get position and quaternion from model (rest pose) - exactly like original PRX
                for (int k = 0; k < 3; k++) {
                    obj.position[k] = model->geom_pos[j * 3 + k];
                }
                for (int k = 0; k < 4; k++) {
                    obj.quaternion[k] = model->geom_quat[j * 4 + k];
                }

                
                // Determine symmetry based on size
                double size_ratio = std::max(obj.size[0], obj.size[1]) / std::min(obj.size[0], obj.size[1]);
                obj.symmetry_rotations = (size_ratio < 1.05) ? 4 : 2;
                
                // Categorize as static or movable based on name
                obj.is_static = (body_name.find("static") != std::string::npos || 
                               body_name.find("wall") != std::string::npos);
                
                if (obj.is_static) {
                    add_static_object(obj);
                } else if (body_name.find("movable") != std::string::npos) {
                    // std::cout << "Adding movable object: " << obj.name << " " << obj.geom_id << " " << obj.body_id << std::endl;
                    add_movable_object(obj);
                }
            }
        }
    }
}

void NAMOEnvironment::add_static_object(const ObjectInfo& obj) {
    if (num_static_ < MAX_STATIC_OBJECTS) {
        static_objects_[num_static_++] = obj;
    } else {
        std::cerr << "Warning: Maximum static objects exceeded. Increase MAX_STATIC_OBJECTS." << std::endl;
        std::cout << config_name_ << std::endl;
    }
}

void NAMOEnvironment::add_movable_object(const ObjectInfo& obj) {
    if (num_movable_ < MAX_MOVABLE_OBJECTS) {
        movable_objects_[num_movable_++] = obj;
        
        // Initialize object state
        ObjectState state;
        state.name = obj.name;
        state.position = obj.position;
        state.quaternion = obj.quaternion;
        state.size = obj.size;
        object_states_[obj.name] = state;

    } else {
        std::cerr << "Warning: Maximum movable objects exceeded. Increase MAX_MOVABLE_OBJECTS." << std::endl;
    }
}

void NAMOEnvironment::step(const Control& control, double dt) {
    sim_->set_control(control);
    sim_->step();
    update_object_states();
}

void NAMOEnvironment::step_simulation() {
    sim_->step();
    update_object_states();
}

void NAMOEnvironment::reset() {
    sim_->reset();
    warm_up();
    
    if (logging_enabled_) {
        flush_log_buffer();
        state_log_idx_++;
        header_written_ = false;
        wavefront_id_ = -1;
        frame_count_ = 0;
        
        state_log_file_.close();
        state_log_file_.open("namo_state_log_" + std::to_string(state_log_idx_) + ".csv");
    }
    
    update_object_states();
}

void NAMOEnvironment::set_robot_position(const std::array<double, 2>& pos) {
    // Calculate offset from initial position
    std::array<double, 2> robot_pos = {
        pos[0] - init_robot_pos_[0], 
        pos[1] - init_robot_pos_[1], 
    };
    sim_->set_robot_position(robot_pos);
    update_object_states();
}

void NAMOEnvironment::set_robot_position(const std::array<double, 3>& pos) {
    sim_->set_robot_position(pos);
    update_object_states();
}

void NAMOEnvironment::set_zero_velocity() {
    sim_->set_zero_velocity();
    update_object_states();
}

void NAMOEnvironment::apply_robot_control(double control_x, double control_y) {
    sim_->set_robot_control(control_x, control_y);
}

void NAMOEnvironment::set_robot_control(double control_x, double control_y) {
    sim_->set_robot_control(control_x, control_y);
}

void NAMOEnvironment::apply_control(double control_x, double control_y, double dt) {
    // Apply control for the specified duration (matching original PRX implementation)
    sim_->set_robot_control(control_x, control_y);
    
    // Calculate number of simulation steps for the given time duration
    // MuJoCo default timestep is typically 0.002, but we should check the model
    mjModel* model = sim_->model();
    double timestep = model->opt.timestep;
    int num_steps = static_cast<int>(dt / timestep);
    
    // Ensure we take at least one step
    num_steps = std::max(1, num_steps);
    
    // Step simulation for the calculated duration
    for (int i = 0; i < num_steps; i++) {
        sim_->step();
    }
    
    // Update object states after control application
    update_object_states();
}

void NAMOEnvironment::update_object_states() {
    // Update robot state
    robot_state_.name = "robot";
    std::array<double, 7> robot_pose;
    if (sim_->get_geom_pose("robot", robot_pose)) {
        for (int i = 0; i < 3; i++) robot_state_.position[i] = robot_pose[i];
        for (int i = 0; i < 4; i++) robot_state_.quaternion[i] = robot_pose[i + 3];
    }
    robot_state_.size = robot_info_.size;
    
    // Update movable objects
    for (size_t i = 0; i < num_movable_; i++) {
        const auto& obj = movable_objects_[i];
        ObjectState& state = object_states_[obj.name];
        
        state.name = obj.name;
        state.size = obj.size;

        
        // Get current pose - exactly like original PRX implementation
        mjModel* model = sim_->model();
        mjData* data = sim_->data();
        

        // Get position from geom_xpos (like original)
        for (int j = 0; j < 3; j++) {
            state.position[j] = data->geom_xpos[obj.geom_id * 3 + j];
        }
        
        // Get quaternion from rotation matrix using mju_mat2Quat (exactly like original line 570)
        mjtNum* obj_quat = data->geom_xmat + obj.geom_id * 9;
        mju_mat2Quat(state.quaternion.data(), obj_quat);
      
        // Get velocities if body exists
        if (obj.body_id >= 0 && obj.body_id < model->nbody) {
            // Get body velocities from cvel (Cartesian velocity)
            int vel_adr = 6 * obj.body_id;
            if (vel_adr + 5 < model->nbody * 6) {
                for (int j = 0; j < 3; j++) {
                    state.linear_vel[j] = data->cvel[vel_adr + j];
                    state.angular_vel[j] = data->cvel[vel_adr + 3 + j];
                }
            }
        }
    }
    
    // Log state if enabled
    if (logging_enabled_) {
        log_state();
    }
}

void NAMOEnvironment::log_state() {
    if (!header_written_) {
        // Write header
        int written = snprintf(log_buffer_.data() + log_position_, 
                              LOG_BUFFER_SIZE - log_position_,
                              "frame,robot_x,robot_y,");
        log_position_ += written;
        
        // Add headers for movable objects
        for (size_t i = 0; i < num_movable_; i++) {
            const auto& obj = movable_objects_[i];
            written = snprintf(log_buffer_.data() + log_position_, 
                              LOG_BUFFER_SIZE - log_position_,
                              "%s_x,%s_y,%s_qw,%s_qx,%s_qy,%s_qz,",
                              obj.name.c_str(), obj.name.c_str(),
                              obj.name.c_str(), obj.name.c_str(),
                              obj.name.c_str(), obj.name.c_str());
            log_position_ += written;
        }
        
        written = snprintf(log_buffer_.data() + log_position_, 
                          LOG_BUFFER_SIZE - log_position_,
                          "wavefront_id\n");
        log_position_ += written;
        
        header_written_ = true;
    }
    
    // Write frame data
    int written = snprintf(log_buffer_.data() + log_position_, 
                          LOG_BUFFER_SIZE - log_position_,
                          "%lu,%.6f,%.6f,",
                          frame_count_, robot_state_.position[0], robot_state_.position[1]);
    log_position_ += written;
    
    // Write movable object data
    for (size_t i = 0; i < num_movable_; i++) {
        const auto& obj = movable_objects_[i];
        const auto& state = object_states_.at(obj.name);
        
        written = snprintf(log_buffer_.data() + log_position_, 
                          LOG_BUFFER_SIZE - log_position_,
                          "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                          state.position[0], state.position[1],
                          state.quaternion[0], state.quaternion[1],
                          state.quaternion[2], state.quaternion[3]);
        log_position_ += written;
    }
    
    written = snprintf(log_buffer_.data() + log_position_, 
                      LOG_BUFFER_SIZE - log_position_,
                      "%d\n", wavefront_id_);
    log_position_ += written;
    
    frame_count_++;
    
    // Flush buffer when nearly full
    if (log_position_ > LOG_BUFFER_SIZE - 1000) {
        flush_log_buffer();
    }
}

void NAMOEnvironment::flush_log_buffer() {
    if (state_log_file_.is_open() && log_position_ > 0) {
        state_log_file_.write(log_buffer_.data(), log_position_);
        state_log_file_.flush();
        log_position_ = 0;
    }
}

void NAMOEnvironment::enable_logging() {
    if (!logging_enabled_) {
        logging_enabled_ = true;
        state_log_file_.open("namo_state_log_" + std::to_string(state_log_idx_) + ".csv");
        if (!state_log_file_.is_open()) {
            std::cerr << "Warning: Could not open state log file. Logging disabled." << std::endl;
            logging_enabled_ = false;
        }
    }
}

void NAMOEnvironment::disable_logging() {
    if (logging_enabled_) {
        flush_log_buffer();
        state_log_file_.close();
        logging_enabled_ = false;
    }
}

std::vector<double> NAMOEnvironment::get_environment_bounds() const {
    // Start with minimum bounds of [-2,-2] to [2,2]
    std::vector<double> bounds = {
        -2.0,  // x_min
         2.0,  // x_max
        -2.0,  // y_min
         2.0   // y_max
    };
    
    // Include static objects
    for (size_t i = 0; i < num_static_; i++) {
        const auto& obj = static_objects_[i];
        
        double half_width = obj.size[0] * 0.5;
        double half_height = obj.size[1] * 0.5;
        double yaw = utils::quaternion_to_yaw(obj.quaternion);
        
        // Calculate rotated corners
        std::array<std::pair<double, double>, 4> corners = {{
            {obj.position[0] + (-half_width * std::cos(yaw) - -half_height * std::sin(yaw)),
             obj.position[1] + (-half_width * std::sin(yaw) + -half_height * std::cos(yaw))},
            {obj.position[0] + ( half_width * std::cos(yaw) - -half_height * std::sin(yaw)),
             obj.position[1] + ( half_width * std::sin(yaw) + -half_height * std::cos(yaw))},
            {obj.position[0] + ( half_width * std::cos(yaw) -  half_height * std::sin(yaw)),
             obj.position[1] + ( half_width * std::sin(yaw) +  half_height * std::cos(yaw))},
            {obj.position[0] + (-half_width * std::cos(yaw) -  half_height * std::sin(yaw)),
             obj.position[1] + (-half_width * std::sin(yaw) +  half_height * std::cos(yaw))}
        }};
        
        for (const auto& corner : corners) {
            bounds[0] = std::min(bounds[0], corner.first);   // Expand x_min if needed
            bounds[1] = std::max(bounds[1], corner.first);   // Expand x_max if needed
            bounds[2] = std::min(bounds[2], corner.second);  // Expand y_min if needed
            bounds[3] = std::max(bounds[3], corner.second);  // Expand y_max if needed
        }
    }
    
    // Include movable objects
    for (size_t i = 0; i < num_movable_; i++) {
        const auto& obj = movable_objects_[i];
        
        double half_width = obj.size[0] * 0.5;
        double half_height = obj.size[1] * 0.5;
        double yaw = utils::quaternion_to_yaw(obj.quaternion);
        
        // Calculate rotated corners
        std::array<std::pair<double, double>, 4> corners = {{
            {obj.position[0] + (-half_width * std::cos(yaw) - -half_height * std::sin(yaw)),
             obj.position[1] + (-half_width * std::sin(yaw) + -half_height * std::cos(yaw))},
            {obj.position[0] + ( half_width * std::cos(yaw) - -half_height * std::sin(yaw)),
             obj.position[1] + ( half_width * std::sin(yaw) + -half_height * std::cos(yaw))},
            {obj.position[0] + ( half_width * std::cos(yaw) -  half_height * std::sin(yaw)),
             obj.position[1] + ( half_width * std::sin(yaw) +  half_height * std::cos(yaw))},
            {obj.position[0] + (-half_width * std::cos(yaw) -  half_height * std::sin(yaw)),
             obj.position[1] + (-half_width * std::sin(yaw) +  half_height * std::cos(yaw))}
        }};
        
        for (const auto& corner : corners) {
            bounds[0] = std::min(bounds[0], corner.first);   // Expand x_min if needed
            bounds[1] = std::max(bounds[1], corner.first);   // Expand x_max if needed
            bounds[2] = std::min(bounds[2], corner.second);  // Expand y_min if needed
            bounds[3] = std::max(bounds[3], corner.second);  // Expand y_max if needed
        }
    }
    
    // Include robot position (CRITICAL FIX for wavefront)
    double robot_x = robot_state_.position[0];
    double robot_y = robot_state_.position[1];
    double robot_radius = robot_info_.size[0]; // Use robot radius for bounds
    
    bounds[0] = std::min(bounds[0], robot_x - robot_radius);  // Expand x_min if needed
    bounds[1] = std::max(bounds[1], robot_x + robot_radius);  // Expand x_max if needed
    bounds[2] = std::min(bounds[2], robot_y - robot_radius);  // Expand y_min if needed
    bounds[3] = std::max(bounds[3], robot_y + robot_radius);  // Expand y_max if needed
    
    // Add padding
    const double PADDING = 0.5;
    bounds[0] -= PADDING;
    bounds[1] += PADDING;
    bounds[2] -= PADDING;
    bounds[3] += PADDING;
    
    return bounds;
}

void NAMOEnvironment::visualize_edge_reachability(const std::string& object_name, 
                                                const std::vector<int>& reachable_edges) {
    // Get object state
    const ObjectState* obj_state = get_object_state(object_name);
    if (!obj_state) {
        // std::cout << "Object not found for visualization: " << object_name << std::endl;
        return;
    }
    
    // Generate all 12 edge points around the object
    std::array<std::array<double, 2>, 12> edge_points_2d;
    std::array<std::array<double, 2>, 12> mid_points_2d;  // Not used but required
    
    // Object dimensions with margin (same as push controller)
    double yaw = 0.0; // Simplified for now - could extract from quaternion
    double x = obj_state->position[0], y = obj_state->position[1];
    double w = obj_state->size[0] - 0.05;  // width with margin
    double d = obj_state->size[1] - 0.05;  // depth with margin
    double offset = 0.15 + 0.05; // robot radius + margin
    
    // Generate 12 edge points (same pattern as push controller)
    std::array<std::array<double, 2>, 12> local_edge_points = {{
        {{x - w, y + d + offset}}, {{x - w, y - d - offset}}, 
        {{x, y + d + offset}}, {{x, y - d - offset}}, 
        {{x + w, y + d + offset}}, {{x + w, y - d - offset}}, 
        {{x + w + offset, y - d}}, {{x - w - offset, y - d}}, 
        {{x + w + offset, y}}, {{x - w - offset, y}}, 
        {{x + w + offset, y + d}}, {{x - w - offset, y + d}}
    }};
    
    // Prepare arrays for visualization
    std::array<std::array<double, 3>, 12> positions_3d;
    std::array<std::array<float, 4>, 12> colors;
    std::array<double, 12> sizes;
    
    // Set up colors and positions for all 12 edges
    for (int i = 0; i < 12; i++) {
        // Position (add z=0.3 to place markers above ground)
        positions_3d[i][0] = local_edge_points[i][0];
        positions_3d[i][1] = local_edge_points[i][1];
        positions_3d[i][2] = 0.3;  // Height above ground
        
        // Size
        sizes[i] = 0.05;  // 5cm radius spheres
        
        // Color: green if reachable, red if not
        bool is_reachable = std::find(reachable_edges.begin(), reachable_edges.end(), i) 
                           != reachable_edges.end();
        
        if (is_reachable) {
            // Green for reachable
            colors[i][0] = 0.0f;  // R
            colors[i][1] = 1.0f;  // G
            colors[i][2] = 0.0f;  // B
            colors[i][3] = 0.8f;  // A (semi-transparent)
        } else {
            // Red for unreachable
            colors[i][0] = 1.0f;  // R
            colors[i][1] = 0.0f;  // G
            colors[i][2] = 0.0f;  // B
            colors[i][3] = 0.8f;  // A (semi-transparent)
        }
    }
    
    // Add markers to MuJoCo scene (only if simulation is available)
    if (sim_) {
        sim_->add_visual_markers(positions_3d.data(), colors.data(), sizes.data(), 12);
    } else {
        // std::cout << "Simulation not available - visual markers not displayed" << std::endl;
    }
    
    // std::cout << "Added visual markers: " << reachable_edges.size() << " green (reachable), " 
            //   << (12 - reachable_edges.size()) << " red (unreachable)" << std::endl;
}

std::vector<double> NAMOEnvironment::get_random_state() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::vector<double> bounds = get_environment_bounds();
    std::uniform_real_distribution<double> x_dist(bounds[0], bounds[1]);
    std::uniform_real_distribution<double> y_dist(bounds[2], bounds[3]);
    std::uniform_real_distribution<double> yaw_dist(-M_PI, M_PI);
    
    return {x_dist(gen), y_dist(gen), yaw_dist(gen)};
}

const ObjectInfo* NAMOEnvironment::get_object_info(const std::string& name) const {
    // Check robot
    if (name == "robot") {
        return &robot_info_;
    }
    
    // Check static objects
    for (size_t i = 0; i < num_static_; i++) {
        if (static_objects_[i].name == name) {
            return &static_objects_[i];
        }
    }
    
    // Check movable objects
    for (size_t i = 0; i < num_movable_; i++) {
        if (movable_objects_[i].name == name) {
            return &movable_objects_[i];
        }
    }
    
    return nullptr;
}

std::map<std::string, std::map<std::string, double>> NAMOEnvironment::get_all_object_info() const {
    std::map<std::string, std::map<std::string, double>> all_object_info;
    
    // Add robot info (only size is immutable)
    all_object_info[robot_info_.name] = {
        {"size_x", robot_info_.size[0]},
        {"size_y", robot_info_.size[1]},
        {"size_z", robot_info_.size[2]}
    };
    
    // Add static objects (position, orientation, AND size are all immutable)
    for (size_t i = 0; i < num_static_; i++) {
        const auto& obj = static_objects_[i];
        all_object_info[obj.name] = {
            {"size_x", obj.size[0]},
            {"size_y", obj.size[1]},
            {"size_z", obj.size[2]},
            {"pos_x", obj.position[0]},
            {"pos_y", obj.position[1]},
            {"pos_z", obj.position[2]},
            {"quat_w", obj.quaternion[0]},
            {"quat_x", obj.quaternion[1]},
            {"quat_y", obj.quaternion[2]},
            {"quat_z", obj.quaternion[3]}
        };
    }
    
    // Add movable objects (only size is immutable)
    for (size_t i = 0; i < num_movable_; i++) {
        const auto& obj = movable_objects_[i];
        all_object_info[obj.name] = {
            {"size_x", obj.size[0]},
            {"size_y", obj.size[1]},
            {"size_z", obj.size[2]}
        };
    }
    
    return all_object_info;
}

const ObjectState* NAMOEnvironment::get_object_state(const std::string& name) const {
    if (name == "robot") {
        return &robot_state_;
    }
    
    auto it = object_states_.find(name);
    return (it != object_states_.end()) ? &(it->second) : nullptr;
}

void NAMOEnvironment::save_objects_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open object data file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "object_type,object_name,size_x,size_y\n";
    
    // Write robot information (type 2)
    file << "2," << robot_info_.name << "," 
         << std::fixed << std::setprecision(6)
         << robot_info_.size[0] << "," << robot_info_.size[1] << "\n";
    
    // Write static objects (type 0)
    for (size_t i = 0; i < num_static_; i++) {
        const auto& obj = static_objects_[i];
        file << "0," << obj.name << "," 
             << obj.size[0] << "," << obj.size[1] << "\n";
    }
    
    // Write movable objects (type 1)
    for (size_t i = 0; i < num_movable_; i++) {
        const auto& obj = movable_objects_[i];
        file << "1," << obj.name << "," 
             << obj.size[0] << "," << obj.size[1] << "\n";
    }
    
    file.close();
}

void NAMOEnvironment::visualize_goal_marker(const std::array<double, 3>& goal_position, 
                                           const std::array<float, 4>& color) {
    if (!sim_) return;
    
    // Make goal marker more visible - match typical object size
    std::array<double, 4> orientation = {1.0, 0.0, 0.0, 0.0}; // Identity quaternion
    std::array<double, 3> size = {0.35, 0.35, 0.05}; // Match movable object width/height, thin but visible
    int geom_type = 6; // mjGEOM_BOX = 6 - use thin box to show goal footprint
    
    sim_->set_goal_marker(goal_position, orientation, size, geom_type);
}

void NAMOEnvironment::visualize_object_goal_marker(const std::array<double, 3>& goal_position,
                                                  const std::array<double, 3>& object_size,
                                                  double theta,
                                                  const std::array<float, 4>& color) {
    if (!sim_) return;
    
    // Convert theta (yaw angle) to quaternion
    double half_theta = theta * 0.5;
    std::array<double, 4> orientation = {
        std::cos(half_theta),  // w
        0.0,                   // x
        0.0,                   // y
        std::sin(half_theta)   // z
    };
    
    std::array<double, 3> marker_size = {object_size[0], object_size[1], 0.05}; // Match object footprint, thin but visible
    int geom_type = 6; // mjGEOM_BOX = 6 - use thin box to show goal footprint
    
    sim_->set_goal_marker(goal_position, orientation, marker_size, geom_type);
}

//=============================================================================
// State management for optimization
//=============================================================================

void NAMOEnvironment::save_current_state() {
    if (!sim_) return;
    
    // Get current state from MuJoCo
    State current_state;
    sim_->get_state(current_state);
    saved_qpos_.resize(current_state.size());
    for (size_t i = 0; i < current_state.size(); ++i) {
        saved_qpos_[i] = current_state[i];
    }
    saved_qvel_.clear();  // Not using separate velocity storage for now
    has_saved_state_ = true;
}

void NAMOEnvironment::restore_saved_state() {
    if (!sim_ || !has_saved_state_) return;
    
    // Restore state to MuJoCo
    State state;
    state.resize(saved_qpos_.size());
    for (size_t i = 0; i < saved_qpos_.size(); ++i) {
        state[i] = saved_qpos_[i];
    }
    sim_->set_state(state);
    
    // Update our object state tracking
    update_object_states();
}

void NAMOEnvironment::reset_to_initial_state() {
    if (!sim_) return;
    reset();
}

} // namespace namo