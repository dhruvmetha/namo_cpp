#include "environment/namo_environment.hpp"
#include "robot/holonomic_adapter.hpp"
#include "robot/diff_drive_adapter.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <iomanip>
#include <cmath>
#include <sstream>

extern "C" {
#include "mujoco/mujoco.h"
}

namespace namo {

namespace {

constexpr double kRobotFootprintMismatchEpsilon = 1e-6;
constexpr std::array<double, 2> kDefaultRobotPlanningHalfExtents = {0.15, 0.15};

bool is_descendant_body(const mjModel* model, int body_id, int ancestor_id) {
    while (body_id >= 0) {
        if (body_id == ancestor_id) {
            return true;
        }
        const int parent_id = model->body_parentid[body_id];
        if (parent_id == body_id) {
            break;
        }
        body_id = parent_id;
    }
    return false;
}

bool is_collidable_geom(const mjModel* model, int geom_id) {
    return model->geom_contype[geom_id] != 0 || model->geom_conaffinity[geom_id] != 0;
}

std::array<double, 3> geom_local_half_extents(const mjModel* model, int geom_id) {
    const mjtNum* size = model->geom_size + 3 * geom_id;
    switch (model->geom_type[geom_id]) {
        case mjGEOM_BOX:
        case mjGEOM_ELLIPSOID:
            return {size[0], size[1], size[2]};
        case mjGEOM_SPHERE:
            return {size[0], size[0], size[0]};
        case mjGEOM_CYLINDER:
            return {size[0], size[0], size[1]};
        case mjGEOM_CAPSULE:
            return {size[0], size[0], size[0] + size[1]};
        default: {
            const double radius = (model->geom_rbound[geom_id] > 0.0)
                ? model->geom_rbound[geom_id]
                : std::max({double(size[0]), double(size[1]), double(size[2])});
            return {radius, radius, radius};
        }
    }
}

std::array<double, 3> rotate_world_to_local(const mjtNum* local_to_world_rot, const std::array<double, 3>& world_vec) {
    return {
        local_to_world_rot[0] * world_vec[0] + local_to_world_rot[3] * world_vec[1] + local_to_world_rot[6] * world_vec[2],
        local_to_world_rot[1] * world_vec[0] + local_to_world_rot[4] * world_vec[1] + local_to_world_rot[7] * world_vec[2],
        local_to_world_rot[2] * world_vec[0] + local_to_world_rot[5] * world_vec[1] + local_to_world_rot[8] * world_vec[2],
    };
}

void multiply_transpose_a_b(const mjtNum* a, const mjtNum* b, double out[9]) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += static_cast<double>(a[k * 3 + row]) * static_cast<double>(b[k * 3 + col]);
            }
            out[row * 3 + col] = sum;
        }
    }
}

std::array<double, 3> oriented_box_aabb_half_extents(const double rot[9], const std::array<double, 3>& half_extents) {
    std::array<double, 3> out = {0.0, 0.0, 0.0};
    for (int row = 0; row < 3; ++row) {
        out[row] =
            std::abs(rot[row * 3 + 0]) * half_extents[0] +
            std::abs(rot[row * 3 + 1]) * half_extents[1] +
            std::abs(rot[row * 3 + 2]) * half_extents[2];
    }
    return out;
}

bool materially_differs(const std::array<double, 2>& lhs, const std::vector<double>& rhs) {
    return rhs.size() < 2 ||
           std::abs(lhs[0] - rhs[0]) > kRobotFootprintMismatchEpsilon ||
           std::abs(lhs[1] - rhs[1]) > kRobotFootprintMismatchEpsilon;
}

std::string format_half_extents(const std::array<double, 2>& extents) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "[" << extents[0] << ", " << extents[1] << "]";
    return oss.str();
}

std::string format_half_extents(const std::vector<double>& extents) {
    if (extents.size() < 2) {
        return "[]";
    }
    return format_half_extents(std::array<double, 2>{extents[0], extents[1]});
}

}  // namespace

NAMOEnvironment::NAMOEnvironment(const std::string& xml_path, bool visualize, bool enable_logging) 
    : logging_enabled_(enable_logging) {
    
    // Create MuJoCo wrapper
    sim_ = std::make_unique<OptimizedMujocoWrapper>(xml_path, visualize);
    sim_->initialize();
    
    // Set reasonable camera defaults (top-down view)
    sim_->set_camera_lookat({0.0, 0.0, 0.0});
    sim_->set_camera_position(15.0, 0.0, -90.0);
    
    // Warm up simulation
    warm_up();
    
    // Extract config name from XML path
    std::filesystem::path xml_file_path(xml_path);
    config_name_ = xml_file_path.stem().string();
    
    // Create default holonomic adapter (backward compatible constructor)
    {
        std::array<double, 3> init_pos = {0.0, 0.0, 0.0};
        std::array<double, 7> pose;
        if (sim_->get_geom_pose("robot", pose)) {
            init_pos = {pose[0], pose[1], pose[2]};
        }
        robot_adapter_ = std::make_unique<HolonomicAdapter>(init_pos);
    }
    init_robot_from_adapter();
    
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

NAMOEnvironment::NAMOEnvironment(const std::string& xml_path, std::shared_ptr<ConfigManager> config,
                                 bool visualize, bool enable_logging)
    : config_(std::move(config)), logging_enabled_(enable_logging) {

    sim_ = std::make_unique<OptimizedMujocoWrapper>(xml_path, visualize);
    sim_->initialize();
    sim_->set_camera_lookat({0.0, 0.0, 0.0});
    sim_->set_camera_position(15.0, 0.0, -90.0);
    warm_up();

    std::filesystem::path xml_file_path(xml_path);
    config_name_ = xml_file_path.stem().string();

    // Create adapter based on config robot_type
    const std::string& robot_type = config_ ? config_->get_robot_type() : "holonomic";
    if (robot_type == "holonomic") {
        std::array<double, 3> init_pos = {0.0, 0.0, 0.0};
        std::array<double, 7> pose;
        if (sim_->get_geom_pose("robot", pose)) {
            init_pos = {pose[0], pose[1], pose[2]};
        }
        robot_adapter_ = std::make_unique<HolonomicAdapter>(init_pos);
    } else if (robot_type == "diff_drive") {
        // Get initial body position from MuJoCo (after warm_up/mj_forward)
        std::array<double, 3> car_init_pos = {0.0, 0.0, 0.0};
        std::array<double, 7> car_pose;
        if (sim_->get_body_pose("car", car_pose)) {
            car_init_pos = {car_pose[0], car_pose[1], car_pose[2]};
        }
        robot_adapter_ = std::make_unique<DiffDriveAdapter>(sim_->model(), car_init_pos);
    } else {
        throw std::runtime_error("Unknown robot_type: " + robot_type +
                                 " (supported: 'holonomic', 'diff_drive')");
    }
    init_robot_from_adapter();

    process_environment_objects();

    if (logging_enabled_) {
        state_log_file_.open("namo_state_log_" + std::to_string(state_log_idx_) + ".csv");
        if (!state_log_file_.is_open()) {
            std::cerr << "Warning: Could not open state log file. Logging disabled." << std::endl;
            logging_enabled_ = false;
        }
    }

    update_object_states();
}

void NAMOEnvironment::init_robot_from_adapter() {
    // Use adapter to identify robot and read initial pose
    std::string pose_name = robot_adapter_->get_pose_source_name();
    std::string body_name = robot_adapter_->get_body_name();

    robot_info_.body_id = sim_->get_body_id(body_name);
    robot_info_.name = "robot";  // Internal name stays "robot" for Python compatibility
    robot_info_.is_static = false;

    // Find a geom for size queries: try pose_name first, then first geom on body
    robot_id_ = sim_->get_geom_id(pose_name);
    if (robot_id_ < 0 && robot_info_.body_id >= 0) {
        // pose_name is a body, not a geom. Find the first geom on this body.
        mjModel* model = sim_->model();
        for (int j = 0; j < model->ngeom; j++) {
            if (model->geom_bodyid[j] == robot_info_.body_id) {
                robot_id_ = j;
                break;
            }
        }
    }
    robot_info_.geom_id = robot_id_;

    if (robot_info_.body_id >= 0) {
        // Read pose: body pose for car (no single center geom), geom pose for point robot
        std::array<double, 7> robot_pose;
        bool got_pose = robot_adapter_->use_body_pose()
            ? sim_->get_body_pose(body_name, robot_pose)
            : sim_->get_geom_pose(pose_name, robot_pose);
        if (got_pose) {
            for (int i = 0; i < 3; i++) robot_info_.position[i] = robot_pose[i];
            for (int i = 0; i < 4; i++) robot_info_.quaternion[i] = robot_pose[i + 3];
        }

        // Read one geom size as a last-resort fallback. The canonical planning/export
        // footprint is derived from the full collidable robot subtree below.
        mjModel* model = sim_->model();
        if (robot_id_ >= 0 && robot_id_ < model->ngeom) {
            for (int i = 0; i < 3; i++) {
                robot_info_.size[i] = model->geom_size[robot_id_ * 3 + i];
            }
        }

        std::array<double, 3> derived_size = robot_info_.size;
        if (derive_robot_footprint_from_collision_geometry(derived_size)) {
            robot_info_.size = derived_size;
            if (config_ && config_->is_robot_size_explicitly_configured()) {
                const auto derived_xy = get_robot_planning_half_extents();
                const auto& cfg_xy = config_->planning().robot_size;
                if (materially_differs(derived_xy, cfg_xy)) {
                    std::cerr
                        << "Warning: planning.robot_size is legacy fallback only and differs from "
                        << "geometry-derived robot footprint; using geometry-derived values. "
                        << "yaml=" << format_half_extents(cfg_xy)
                        << ", geometry=" << format_half_extents(derived_xy) << std::endl;
                } else {
                    std::cerr
                        << "Info: planning.robot_size is legacy fallback only and is ignored at runtime; "
                        << "using geometry-derived robot footprint "
                        << format_half_extents(derived_xy) << std::endl;
                }
            }
        } else {
            if (config_ && config_->planning().robot_size.size() >= 2) {
                robot_info_.size[0] = config_->planning().robot_size[0];
                robot_info_.size[1] = config_->planning().robot_size[1];
                if (robot_info_.size[2] <= 0.0) {
                    robot_info_.size[2] = std::max(robot_info_.size[0], robot_info_.size[1]);
                }
                std::cerr
                    << "Warning: failed to derive robot planning footprint from collision geometry; "
                    << "falling back to planning.robot_size "
                    << format_half_extents(config_->planning().robot_size) << std::endl;
            } else {
                robot_info_.size[0] = (robot_info_.size[0] > 0.0) ? robot_info_.size[0] : kDefaultRobotPlanningHalfExtents[0];
                robot_info_.size[1] = (robot_info_.size[1] > 0.0) ? robot_info_.size[1] : kDefaultRobotPlanningHalfExtents[1];
                if (robot_info_.size[2] <= 0.0) {
                    robot_info_.size[2] = std::max(robot_info_.size[0], robot_info_.size[1]);
                }
                std::cerr
                    << "Warning: failed to derive robot planning footprint from collision geometry; "
                    << "falling back to default half-extents "
                    << format_half_extents(std::array<double, 2>{robot_info_.size[0], robot_info_.size[1]}) << std::endl;
            }
        }

        init_robot_pos_ = robot_info_.position;
    }
}

bool NAMOEnvironment::derive_robot_footprint_from_collision_geometry(std::array<double, 3>& derived_size) const {
    const mjModel* model = sim_ ? sim_->model() : nullptr;
    const mjData* data = sim_ ? sim_->data() : nullptr;
    if (!model || !data || robot_info_.body_id < 0 || robot_info_.body_id >= model->nbody) {
        return false;
    }

    const mjtNum* root_pos = data->xpos + 3 * robot_info_.body_id;
    const mjtNum* root_rot = data->xmat + 9 * robot_info_.body_id;
    std::array<double, 3> max_abs_extent = {0.0, 0.0, 0.0};
    bool found_collidable_geom = false;

    for (int geom_id = 0; geom_id < model->ngeom; ++geom_id) {
        const int body_id = model->geom_bodyid[geom_id];
        if (!is_descendant_body(model, body_id, robot_info_.body_id) || !is_collidable_geom(model, geom_id)) {
            continue;
        }

        found_collidable_geom = true;
        const auto local_half_extents = geom_local_half_extents(model, geom_id);
        const mjtNum* geom_pos = data->geom_xpos + 3 * geom_id;
        const mjtNum* geom_rot = data->geom_xmat + 9 * geom_id;
        const std::array<double, 3> delta_world = {
            geom_pos[0] - root_pos[0],
            geom_pos[1] - root_pos[1],
            geom_pos[2] - root_pos[2],
        };
        const auto center_in_root = rotate_world_to_local(root_rot, delta_world);

        double geom_in_root_rot[9];
        multiply_transpose_a_b(root_rot, geom_rot, geom_in_root_rot);
        const auto geom_root_half_extents = oriented_box_aabb_half_extents(geom_in_root_rot, local_half_extents);

        for (int axis = 0; axis < 3; ++axis) {
            max_abs_extent[axis] = std::max(
                max_abs_extent[axis],
                std::abs(center_in_root[axis]) + geom_root_half_extents[axis]);
        }
    }

    if (!found_collidable_geom || max_abs_extent[0] <= 0.0 || max_abs_extent[1] <= 0.0) {
        return false;
    }

    derived_size = max_abs_extent;
    return true;
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
        
        // Skip robot-related and world bodies (adapter provides the list)
        auto skip = robot_adapter_->get_skip_body_names();
        if (std::find(skip.begin(), skip.end(), body_name) != skip.end()) {
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


                const char* body_name_ptr = mj_id2name(model, mjOBJ_BODY, i);
                obj.body_name = body_name_ptr ? std::string(body_name_ptr) : ("body_" + std::to_string(i));

                // std::cout << "obj.name: " << obj.name << std::endl;
                // std::cout << "obj.body_name: " << obj.body_name << std::endl;
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
    robot_adapter_->set_xy(sim_->model(), sim_->data(), pos[0], pos[1]);
    update_object_states();
}

void NAMOEnvironment::set_robot_position(const std::array<double, 3>& pos) {
    robot_adapter_->set_se2(sim_->model(), sim_->data(), pos[0], pos[1], pos[2]);
    update_object_states();
}

void NAMOEnvironment::set_robot_se2(double x, double y, double theta) {
    robot_adapter_->set_se2(sim_->model(), sim_->data(), x, y, theta);
    update_object_states();
}

void NAMOEnvironment::set_zero_velocity() {
    sim_->set_zero_velocity();
    update_object_states();
}

void NAMOEnvironment::apply_robot_control(double control_x, double control_y) {
    robot_adapter_->apply_control(sim_->model(), sim_->data(), control_x, control_y);
}

void NAMOEnvironment::set_robot_control(double control_x, double control_y) {
    robot_adapter_->apply_control(sim_->model(), sim_->data(), control_x, control_y);
}

void NAMOEnvironment::apply_control(double control_x, double control_y, double dt) {
    // Apply control for the specified duration (matching original PRX implementation)
    robot_adapter_->apply_control(sim_->model(), sim_->data(), control_x, control_y);
    
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
    // Update robot state: use body pose for car, geom pose for point robot
    robot_state_.name = "robot";
    std::array<double, 7> robot_pose;
    std::string pose_src = robot_adapter_->get_pose_source_name();
    bool got_pose = robot_adapter_->use_body_pose()
        ? sim_->get_body_pose(pose_src, robot_pose)
        : sim_->get_geom_pose(pose_src, robot_pose);
    if (got_pose) {
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
    const auto robot_half_extents = get_robot_planning_half_extents();
    const double robot_radius =
        compute_rotation_safe_robot_radius_m({robot_half_extents[0], robot_half_extents[1]});
    
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

void NAMOEnvironment::visualize_edge_reachability(
    const std::string& object_name,
    const std::vector<int>& reachable_edges,
    double edge_offset_margin_m) {
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
    std::vector<double> robot_half_extents = {
        robot_info_.size[0],
        robot_info_.size[1],
    };
    const double rotation_safe_robot_radius =
        compute_rotation_safe_robot_radius_m(robot_half_extents);
    const double margin = (edge_offset_margin_m > 0.0) ? edge_offset_margin_m : 0.020;
    const double offset = rotation_safe_robot_radius + margin;
    
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

void NAMOEnvironment::visualize_goal_marker(
    const std::array<double, 3>& goal_position,
    const std::array<float, 4>& color,
    double goal_radius_m) {
    if (!sim_) return;
    
    // Visualization-only marker for the current robot goal (also used by region-opening
    // scripts to show the sampled neighbour-region goal). Use a sphere so it matches the
    // XML `<site name="goal" type="sphere" ...>` style that users are familiar with.
    //
    std::vector<double> robot_half_extents = {
        robot_info_.size[0],
        robot_info_.size[1],
    };
    const double auto_goal_radius =
        compute_goal_tolerance_m(robot_half_extents, kDefaultWavefrontTier1MarginM);
    const double goal_viz_radius = (goal_radius_m > 0.0) ? goal_radius_m : auto_goal_radius;

    std::array<double, 4> orientation = {1.0, 0.0, 0.0, 0.0}; // Identity quaternion
    std::array<double, 3> size = {goal_viz_radius, goal_viz_radius, goal_viz_radius};
    int geom_type = 2; // mjGEOM_SPHERE = 2
    
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

    // Use separate object target marker (doesn't overwrite robot goal marker)
    sim_->set_object_target_marker(goal_position, orientation, marker_size, color, geom_type);
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

//=============================================================================
// Full state management (zero-allocation)
//=============================================================================

NAMOEnvironment::FullSimState NAMOEnvironment::get_full_state() const {
    FullSimState state;
    if (!sim_) return state;
    
    auto* sim = get_mujoco_wrapper();
    const mjData* d = sim->data();
    const mjModel* m = sim->model();
    
    // Store actual sizes
    state.nq = m->nq;
    state.nv = m->nv;
    
    // Safety check against MAX sizes
    if (state.nq > static_cast<int>(FullSimState::MAX_QPOS)) {
        std::cerr << "Warning: Model nq (" << state.nq << ") exceeds MAX_QPOS (" 
                  << FullSimState::MAX_QPOS << "). Truncating." << std::endl;
        state.nq = FullSimState::MAX_QPOS;
    }
    if (state.nv > static_cast<int>(FullSimState::MAX_QVEL)) {
        std::cerr << "Warning: Model nv (" << state.nv << ") exceeds MAX_QVEL (" 
                  << FullSimState::MAX_QVEL << "). Truncating." << std::endl;
        state.nv = FullSimState::MAX_QVEL;
    }
    
    // Copy qpos
    for (int i = 0; i < state.nq; i++) {
        state.qpos[i] = d->qpos[i];
    }
    
    // Copy qvel
    for (int i = 0; i < state.nv; i++) {
        state.qvel[i] = d->qvel[i];
    }
    
    return state;
}

void NAMOEnvironment::set_full_state(const FullSimState& state) {
    if (!sim_) return;
    
    auto* sim = get_mujoco_wrapper();
    mjData* d = sim->data();
    mjModel* m = sim->model();
    
    // Apply qpos (up to model limits)
    int qpos_to_copy = std::min(state.nq, m->nq);
    for (int i = 0; i < qpos_to_copy; i++) {
        d->qpos[i] = state.qpos[i];
    }
    
    // Apply qvel (up to model limits)
    int qvel_to_copy = std::min(state.nv, m->nv);
    for (int i = 0; i < qvel_to_copy; i++) {
        d->qvel[i] = state.qvel[i];
    }
    
    // Forward kinematics and update tracking
    mj_forward(m, d);
    update_object_states();
}

void NAMOEnvironment::save_full_state() {
    saved_full_state_ = get_full_state();
    has_saved_full_state_ = true;
}

void NAMOEnvironment::restore_full_state() {
    if (!has_saved_full_state_) return;
    set_full_state(saved_full_state_);
}

} // namespace namo
