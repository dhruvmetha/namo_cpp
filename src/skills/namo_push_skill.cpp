#include "skills/namo_push_skill.hpp"
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace namo {

namespace {

std::string add_shape_suffix(const std::string& base, const std::string& shape) {
    auto dot = base.find_last_of('.');
    return dot == std::string::npos
        ? base + "_" + shape
        : base.substr(0, dot) + "_" + shape + base.substr(dot);
}

std::string infer_primitive_shape_suffix(const ObjectInfo& obj_info) {
    const double min_extent = std::min(obj_info.size[0], obj_info.size[1]);
    const double max_extent = std::max(obj_info.size[0], obj_info.size[1]);
    if (min_extent <= 0.0) {
        return "square";
    }

    const double size_ratio = max_extent / min_extent;
    if (size_ratio < 1.05) {
        return "square";
    }
    return obj_info.size[0] >= obj_info.size[1] ? "wide" : "tall";
}

}  // namespace


NAMOPushSkill::NAMOPushSkill(NAMOEnvironment& env)
    : env_(env), config_(nullptr), legacy_config_() {
    initialize_skill();
}

NAMOPushSkill::NAMOPushSkill(NAMOEnvironment& env, const Config& config)
    : env_(env), config_(nullptr), legacy_config_(config) {
    initialize_skill();
}

NAMOPushSkill::NAMOPushSkill(NAMOEnvironment& env, std::shared_ptr<ConfigManager> config)
    : env_(env), config_(config), legacy_config_() {
    initialize_skill();
}

void NAMOPushSkill::initialize_skill() {
    // Initialize executor with configuration parameters
    if (config_) {
        // Use ConfigManager parameters
        executor_ = std::make_unique<PushPrimitiveExecutor>(
            env_,
            config_->planning().skill_level_resolution,
            config_->planning().robot_size,
            config_->planning().wavefront_tier1_inflation_margin,
            config_->skill().max_push_steps,
            config_->skill().control_steps_per_push,
            // Velocity actuators interpret this as m/s command magnitude.
            config_->skill().push_velocity,
            config_->skill().points_per_face,
            config_->skill().check_object_collision,
            config_->skill().dynamic_direction
        );

        // Configure controller-level stuck parameters from config
        auto& controller = executor_->get_controller();
        controller.set_stuck_check_stride(config_->skill().stuck_check_stride);
        controller.set_stuck_threshold(config_->skill().controller_stuck_threshold);
        controller.set_min_position_change(config_->skill().controller_min_position_change);
        controller.set_min_angle_change(config_->skill().controller_min_angle_change);
        controller.set_push_offset_margin(config_->planning().wavefront_edge_offset_margin);
        controller.set_robot_trajectory_collision_checking(config_->skill().check_robot_trajectory_collision);
        controller.set_push_tracker_max_speed(config_->skill().push_tracker_max_speed);
    } else {
        // Use legacy hardcoded values
        executor_ = std::make_unique<PushPrimitiveExecutor>(env_);
    }
}

std::map<std::string, ParameterSchema> NAMOPushSkill::get_parameter_schema() const {
    return {
        {"object_name", {ParameterSchema::STRING, "Name of movable object to push"}},
        {"target_pose", {ParameterSchema::POSE_2D, "Target SE(2) pose (x, y, theta)"}},
        {"robot_goal", {ParameterSchema::POSE_2D, "Optional robot goal for early termination",
                       SkillParameterValue(SE2State())}},  // Optional with default
        {"tolerance", {ParameterSchema::DOUBLE, "Goal tolerance in meters",
                      SkillParameterValue(config_ ? config_->skill().goal_tolerance : legacy_config_.tolerance)}},
        {"edge_idx", {ParameterSchema::INT, "Primitive edge index for direct execution",
                     SkillParameterValue(-1)}},  // Required for execution; -1 = invalid
        {"depth", {ParameterSchema::INT, "Primitive depth for direct execution",
                  SkillParameterValue(-1)}}  // Required for execution; -1 = invalid
    };
}

bool NAMOPushSkill::is_applicable(const std::map<std::string, SkillParameterValue>& parameters) const {
    std::string error;
    if (!validate_parameters(parameters, error)) {
        return false;
    }

    // Extract and validate object
    auto object_name = std::get<std::string>(parameters.at("object_name"));
    if (!is_object_movable(object_name)) {
        return false;
    }

    // (No target-bounds check: the planner picks (edge_idx, depth) over the
    // primitive table; the target_pose is a derived prediction that may be
    // mis-calibrated. Rejecting on it caused false negatives. Out-of-bounds
    // outcomes are harmless — the obstacle just exits the wavefront grid.)
    return true;
}

std::chrono::milliseconds NAMOPushSkill::estimate_duration(const std::map<std::string, SkillParameterValue>& parameters) const {
    if (!is_applicable(parameters)) {
        return std::chrono::milliseconds::max();
    }

    auto object_name = std::get<std::string>(parameters.at("object_name"));
    auto target_pose = std::get<SE2State>(parameters.at("target_pose"));

    auto current_pose = get_object_current_pose(object_name);
    if (!current_pose) {
        return std::chrono::milliseconds::max();
    }

    // Distance-based cost estimation
    double distance = std::sqrt(
        std::pow(target_pose.x - current_pose->x, 2) +
        std::pow(target_pose.y - current_pose->y, 2)
    );

    // Empirical formula: 500ms base + 1000ms per meter
    return std::chrono::milliseconds(static_cast<long>(500 + distance * 1000));
}

SkillResult NAMOPushSkill::execute(const std::map<std::string, SkillParameterValue>& parameters) {
    auto start_time = std::chrono::high_resolution_clock::now();

    SkillResult result;
    result.skill_name = get_name();

    // Validate parameters
    std::string validation_error;
    if (!validate_parameters(parameters, validation_error)) {
        result.failure_reason = "Parameter validation failed: " + validation_error;
        return result;
    }

    // Extract parameters with proper type safety
    auto object_name = std::get<std::string>(parameters.at("object_name"));
    auto target_pose = std::get<SE2State>(parameters.at("target_pose"));

    // Visualize the target object goal in MuJoCo using the actual object size (cyan color)
    const ObjectInfo* obj_info = env_.get_object_info(object_name);
    if (obj_info) {
        std::array<double, 3> target_3d = {target_pose.x, target_pose.y, 0.1}; // Slightly above ground
        std::array<float, 4> cyan_color = {0.0f, 0.8f, 1.0f, 1.0f}; // Cyan for object target goals
        env_.visualize_object_goal_marker(target_3d, obj_info->size, target_pose.theta, cyan_color);
    }

    // Python must provide explicit (edge_idx, depth). This skill is a thin
    // wrapper around a single primitive execution — the MPC retry loop that
    // used to back the (edge_idx=-1, depth=-1) sentinel was removed.
    int provided_edge_idx = -1;
    int provided_depth = -1;
    if (auto it = parameters.find("edge_idx"); it != parameters.end()) {
        provided_edge_idx = std::get<int>(it->second);
    }
    if (auto it = parameters.find("depth"); it != parameters.end()) {
        provided_depth = std::get<int>(it->second);
    }

    if (provided_edge_idx < 0 || provided_depth < 0) {
        result.failure_reason = "edge_idx and depth must both be >= 0; "
                                "this skill no longer supports the MPC search fallback.";
        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return result;
    }

    // Get reachable edges to verify the requested edge is accessible
    std::vector<int> reachable_edges = executor_->get_reachable_edges_with_wavefront(object_name);

    // Check if requested edge is reachable
    bool edge_reachable = std::find(reachable_edges.begin(), reachable_edges.end(),
                                     provided_edge_idx) != reachable_edges.end();

    if (!edge_reachable) {
        result.failure_reason = "Requested edge " + std::to_string(provided_edge_idx) + " not reachable";
        result.failure_type = FailureType::NO_REACHABLE_EDGES;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return result;
    }

    // Execute the specific primitive directly
    int push_steps = provided_depth + 1;  // Convert 0-indexed depth to 1-indexed push_steps
    std::vector<PlanStep> single_step = {PlanStep(provided_edge_idx, push_steps, target_pose)};

    auto step_result = executor_->execute_plan(object_name, single_step);

    // Populate result
    auto final_pose = get_object_current_pose(object_name);
    result.success = step_result.success;
    result.outputs["steps_executed"] = 1;
    result.outputs["final_pose"] = final_pose ? *final_pose : SE2State();
    result.outputs["object_name"] = object_name;
    result.outputs["direct_execution"] = true;  // Flag indicating direct primitive execution

    // Report robot-goal reachability after the push.
    result.outputs["robot_goal_reached"] = (has_robot_goal_ && executor_->is_robot_goal_reachable());

    if (!step_result.success) {
        result.failure_reason = step_result.failure_reason;
        // Copy collision info if present
        if (!step_result.collision_object.empty()) {
            result.outputs["collision_object"] = step_result.collision_object;
            result.failure_type = FailureType::OBJECT_COLLISION_DURING_PUSH;
        }
        // Check for stuck condition
        if (step_result.failure_reason.find("Controller-level stuck") != std::string::npos) {
            result.outputs["stuck"] = std::string("true");  // explicit string to avoid implicit const char* → bool
            result.failure_type = FailureType::OBJECT_STUCK;
        }
    }

    // Collision tracking outputs
    result.outputs["wall_collision"] = step_result.wall_collision_during_push;
    {
        std::string movable_str;
        for (auto it = step_result.movable_collisions_during_push.begin();
             it != step_result.movable_collisions_during_push.end(); ++it) {
            if (it != step_result.movable_collisions_during_push.begin()) movable_str += ",";
            movable_str += *it;
        }
        result.outputs["movable_collisions"] = movable_str;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return result;
}

std::map<std::string, SkillParameterValue> NAMOPushSkill::get_world_state() const {
    std::map<std::string, SkillParameterValue> state;

    // Robot state
    if (auto robot_state = env_.get_robot_state()) {
        SE2State robot_pose;
        robot_pose.x = robot_state->position[0];
        robot_pose.y = robot_state->position[1];
        auto adapter = env_.get_robot_adapter();
        robot_pose.theta = adapter ? adapter->get_theta(env_.get_mujoco_wrapper()->model(),
                                                         env_.get_mujoco_wrapper()->data()) : 0.0;
        state["robot_pose"] = robot_pose;
    }

    // All movable objects
    const auto& movable_objects = env_.get_movable_objects();
    for (size_t i = 0; i < env_.get_num_movable(); i++) {
        const auto& obj_info = movable_objects[i];
        if (!obj_info.name.empty()) {
            if (auto pose = get_object_current_pose(obj_info.name)) {
                state[obj_info.name + "_pose"] = *pose;
            }
        }
    }

    return state;
}

std::vector<std::string> NAMOPushSkill::check_preconditions(const std::map<std::string, SkillParameterValue>& parameters) const {
    std::vector<std::string> unmet;

    // Basic parameter validation
    std::string validation_error;
    if (!validate_parameters(parameters, validation_error)) {
        unmet.push_back(validation_error);
        return unmet;
    }

    auto object_name = std::get<std::string>(parameters.at("object_name"));

    // Check object exists and is movable
    if (!is_object_movable(object_name)) {
        unmet.push_back("Object '" + object_name + "' does not exist or is not movable");
    }

    // (Target-bounds check removed — target is a derived prediction. The
    // planner enumerates (edge_idx, depth) directly and trusts the C++
    // primitive table for actual motion.)

    return unmet;
}

bool NAMOPushSkill::is_object_movable(const std::string& object_name) const {
    if (auto obj_info = env_.get_object_info(object_name)) {
        return !obj_info->is_static;  // movable = not static
    }
    return false;
}

std::optional<SE2State> NAMOPushSkill::get_object_current_pose(const std::string& object_name) const {
    auto obj_state = env_.get_object_state(object_name);
    if (!obj_state) {
        return std::nullopt;
    }

    SE2State pose;
    pose.x = obj_state->position[0];
    pose.y = obj_state->position[1];

    // Proper quaternion to yaw conversion (NO HACK!)
    const auto& q = obj_state->quaternion;
    pose.theta = std::atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),  // w*z + x*y
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])  // 1 - 2*(y^2 + z^2)
    );

    return pose;
}

bool NAMOPushSkill::is_target_within_bounds(const SE2State& target_pose) const {
    auto bounds = env_.get_environment_bounds();
    if (bounds.size() < 4) return false;

    return target_pose.x >= bounds[0] && target_pose.x <= bounds[1] &&
           target_pose.y >= bounds[2] && target_pose.y <= bounds[3];
}

std::string NAMOPushSkill::resolve_motion_primitives_file_for_object(const std::string& object_name) const {
    const ObjectInfo* obj_info = env_.get_object_info(object_name);
    if (!obj_info) {
        return {};
    }

    const std::string base_path = config_
        ? config_->system().motion_primitives_file
        : legacy_config_.primitive_database_path;
    if (base_path.empty()) {
        return {};
    }

    return add_shape_suffix(base_path, infer_primitive_shape_suffix(*obj_info));
}

GreedyPlanner* NAMOPushSkill::get_planner_for_object(const std::string& object_name) const {
    if (auto it = primitive_library_planners_.find(object_name); it != primitive_library_planners_.end()) {
        return it->second.get();
    }

    const std::string primitive_path = resolve_motion_primitives_file_for_object(object_name);
    if (primitive_path.empty() || !std::filesystem::exists(primitive_path)) {
        return nullptr;
    }

    auto planner = std::make_unique<GreedyPlanner>();
    if (!planner->initialize(primitive_path)) {
        return nullptr;
    }

    if (const ObjectInfo* obj_info = env_.get_object_info(object_name)) {
        planner->set_object_symmetry(obj_info->symmetry_rotations);
    }
    planner->set_name(object_name);

    GreedyPlanner* raw = planner.get();
    primitive_library_planners_[object_name] = std::move(planner);
    return raw;
}

std::vector<int> NAMOPushSkill::get_reachable_edges(const std::string& object_name) const {
    // Use executor's wavefront-based reachability analysis
    if (!executor_) {
        return {};
    }

    // Note: executor_->get_reachable_edges_with_wavefront() is non-const, so we need to cast
    return const_cast<PushPrimitiveExecutor*>(executor_.get())->get_reachable_edges_with_wavefront(object_name);
}

PushPrimitiveExecutor::ReachabilitySnapshot NAMOPushSkill::get_reachability_snapshot() const {
    if (!executor_) {
        return PushPrimitiveExecutor::ReachabilitySnapshot{};
    }
    // executor methods are non-const because they refresh internal wavefront caches.
    return const_cast<PushPrimitiveExecutor*>(executor_.get())->compute_reachability_snapshot();
}

bool NAMOPushSkill::get_primitive_library_target_pose(
    const std::string& object_name,
    int edge_idx,
    int depth_idx,
    SE2State& out_target_pose,
    std::string* error_message) const {
    if (depth_idx < 0) {
        if (error_message) {
            *error_message = "depth_idx must be >= 0";
        }
        return false;
    }

    auto current_pose_opt = get_object_current_pose(object_name);
    if (!current_pose_opt) {
        if (error_message) {
            *error_message = "object pose unavailable for " + object_name;
        }
        return false;
    }

    GreedyPlanner* planner = get_planner_for_object(object_name);
    if (!planner) {
        if (error_message) {
            *error_message = "planner unavailable for " + object_name;
        }
        return false;
    }

    const int push_steps = depth_idx + 1;
    try {
        const LoadedPrimitive& primitive =
            planner->get_primitive_loader().get_primitive(edge_idx, push_steps);

        const SE2State& current_pose = *current_pose_opt;
        const double cos_theta = std::cos(current_pose.theta);
        const double sin_theta = std::sin(current_pose.theta);

        const double dx = primitive.delta_x;
        const double dy = primitive.delta_y;
        double next_theta = current_pose.theta + primitive.delta_theta;
        while (next_theta > M_PI) next_theta -= 2.0 * M_PI;
        while (next_theta < -M_PI) next_theta += 2.0 * M_PI;

        out_target_pose = SE2State(
            current_pose.x + dx * cos_theta - dy * sin_theta,
            current_pose.y + dx * sin_theta + dy * cos_theta,
            next_theta
        );
        return true;
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = e.what();
        }
        return false;
    }
}

std::vector<int> NAMOPushSkill::get_valid_primitive_depth_indices(
    const std::string& object_name,
    int edge_idx) const {
    std::vector<int> depth_indices;
    GreedyPlanner* planner = get_planner_for_object(object_name);
    if (!planner) {
        return depth_indices;
    }

    try {
        const std::vector<int> push_steps =
            planner->get_primitive_loader().get_valid_steps_for_edge(edge_idx);
        depth_indices.reserve(push_steps.size());
        for (int steps : push_steps) {
            if (steps <= 0) {
                continue;
            }
            depth_indices.push_back(steps - 1);
        }
        std::sort(depth_indices.begin(), depth_indices.end());
        depth_indices.erase(std::unique(depth_indices.begin(), depth_indices.end()), depth_indices.end());
    } catch (const std::exception&) {
        depth_indices.clear();
    }
    return depth_indices;
}

std::vector<std::string> NAMOPushSkill::get_reachable_objects() const {
    std::vector<std::string> reachable_objects;

    // Get all movable objects
    const auto& movable_objects = env_.get_movable_objects();

    for (size_t i = 0; i < env_.get_num_movable(); i++) {
        const auto& obj_info = movable_objects[i];
        if (!obj_info.name.empty() && is_object_reachable(obj_info.name)) {
            reachable_objects.push_back(obj_info.name);
        }
    }

    return reachable_objects;
}

bool NAMOPushSkill::is_object_reachable(const std::string& object_name) const {
    // Check if object exists and is movable
    if (!is_object_movable(object_name)) {
        return false;
    }

    // Get robot current state
    auto robot_state = env_.get_robot_state();
    if (!robot_state) {
        return false;
    }

    // Use the executor to check reachability via wavefront
    try {
        std::vector<int> reachable_edges = executor_->get_reachable_edges_with_wavefront(object_name);
        return !reachable_edges.empty();
    } catch (const std::exception& e) {
        // If wavefront computation fails, object is not reachable
        return false;
    }
}

void NAMOPushSkill::set_robot_goal(double x, double y, double theta) {
    robot_goal_ = {x, y, theta};
    has_robot_goal_ = true;
    // Also set in executor for immediate use
    executor_->set_robot_goal({x, y});
}

bool NAMOPushSkill::is_robot_goal_reachable() const {
    if (!has_robot_goal_) {
        return false;
    }
    // Leverage the executor's cached wavefront computation
    return executor_->is_robot_goal_reachable();
}

std::array<double, 3> NAMOPushSkill::get_robot_goal() const {
    return robot_goal_;
}

void NAMOPushSkill::clear_robot_goal() {
    has_robot_goal_ = false;
    executor_->clear_robot_goal();
}

std::pair<int, int> NAMOPushSkill::count_reachable_points(
    const std::vector<std::array<double, 2>>& points) const {
    return executor_->count_reachable_points(points);
}

void NAMOPushSkill::set_collision_checking(bool enabled) {
    // Propagate collision checking setting to the executor's controller
    if (executor_) {
        executor_->set_collision_checking(enabled);
    }
}

void NAMOPushSkill::set_robot_trajectory_collision_checking(bool enabled) {
    if (executor_) {
        executor_->set_robot_trajectory_collision_checking(enabled);
    }
}

std::vector<int> NAMOPushSkill::evaluate_primitive_priorities(
    const std::string& object_name,
    const std::vector<std::array<double, 3>>& target_poses,
    const std::array<double, 2>& robot_goal) {
    // Delegate to executor's wavefront planner
    return executor_->evaluate_primitive_priorities(env_, object_name, target_poses, robot_goal);
}

std::map<std::string, double> NAMOPushSkill::get_last_priority_profile() const {
    if (!executor_) {
        return {};
    }
    return executor_->get_last_priority_profile();
}

} // namespace namo
