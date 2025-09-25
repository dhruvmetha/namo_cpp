#include "skills/namo_push_skill.hpp"
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace namo {

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
    // Get base primitive database path
    std::string base_db_path = config_ ? config_->system().motion_primitives_file 
                                       : legacy_config_.primitive_database_path;
    
    // Helper function to add suffix to filename
    auto add_suffix_to_filename = [](const std::string& base_path, const std::string& suffix) {
        auto dot_pos = base_path.find_last_of('.');
        if (dot_pos == std::string::npos) {
            return base_path + "_" + suffix;
        }
        return base_path.substr(0, dot_pos) + "_" + suffix + base_path.substr(dot_pos);
    };
    
    // Helper function to try loading a planner with fallback
    auto try_load_planner = [&](const std::string& preferred_path, const std::string& shape_name) -> std::unique_ptr<GreedyPlanner> {
        auto planner = std::make_unique<GreedyPlanner>();
        
        // Try preferred path first
        if (std::filesystem::exists(preferred_path)) {
            if (planner->initialize(preferred_path)) {
                return planner;
            }
        }
        
        // Fallback to base path
        if (std::filesystem::exists(base_db_path)) {
            if (planner->initialize(base_db_path)) {
                return planner;
            }
        }
        
        // Last resort: try default path
        std::string default_path = "data/motion_primitives.dat";
        if (std::filesystem::exists(default_path)) {
            if (planner->initialize(default_path)) {
                return planner;
            }
        }
        
        throw std::runtime_error("Failed to initialize " + shape_name + " planner. Tried: " + 
                                preferred_path + ", " + base_db_path + ", " + default_path);
    };
    
    // Initialize all three planners with their respective databases
    std::string square_path = add_suffix_to_filename(base_db_path, "square");
    std::string wide_path = add_suffix_to_filename(base_db_path, "wide");
    std::string tall_path = add_suffix_to_filename(base_db_path, "tall");
    
    planner_square_ = try_load_planner(square_path, "square");
    planner_square_->set_name("square");
    // std::cout << "Initialized square planner" << std::endl;
    
    planner_wide_ = try_load_planner(wide_path, "wide");
    planner_wide_->set_name("wide");
    // std::cout << "Initialized wide planner" << std::endl;
    
    planner_tall_ = try_load_planner(tall_path, "tall");
    planner_tall_->set_name("tall");
    // std::cout << "Initialized tall planner" << std::endl;
    
    // Initialize executor with configuration parameters
    if (config_) {
        // Use ConfigManager parameters
        executor_ = std::make_unique<MPCExecutor>(
            env_,
            config_->planning().skill_level_resolution,
            config_->planning().robot_size,
            config_->skill().max_push_steps,
            config_->skill().control_steps_per_push,
            config_->skill().force_scaling,
            config_->skill().points_per_face
        );
    } else {
        // Use legacy hardcoded values
        executor_ = std::make_unique<MPCExecutor>(env_);
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
        {"max_attempts", {ParameterSchema::INT, "Maximum MPC iterations", 
                         SkillParameterValue(config_ ? config_->skill().max_mpc_iterations : legacy_config_.max_mpc_iterations)}}
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
    
    // Validate target pose is reachable
    auto target_pose = std::get<SE2State>(parameters.at("target_pose"));
    if (!is_target_within_bounds(target_pose)) {
        return false;
    }
    
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
    
    // Get optional parameters with defaults
    double tolerance = config_ ? config_->skill().goal_tolerance : legacy_config_.tolerance;
    if (auto it = parameters.find("tolerance"); it != parameters.end()) {
        tolerance = std::get<double>(it->second);
    }
    
    int max_mpc_iterations = config_ ? config_->skill().max_mpc_iterations : legacy_config_.max_mpc_iterations;
    if (auto it = parameters.find("max_attempts"); it != parameters.end()) {
        max_mpc_iterations = std::get<int>(it->second);
    }
    
    // Debug output to verify parameter loading
    // std::cout << "NAMOPushSkill: config_ = " << (config_ ? "valid" : "null") << std::endl;
    // if (config_) {
    //     std::cout << "NAMOPushSkill: config_->skill().max_mpc_iterations = " << config_->skill().max_mpc_iterations << std::endl;
    // } else {
    //     std::cout << "NAMOPushSkill: legacy_config_.max_mpc_iterations = " << legacy_config_.max_mpc_iterations << std::endl;
    // }
    // std::cout << "NAMOPushSkill: Using max_mpc_iterations = " << max_mpc_iterations << std::endl;
    
    // Set robot goal if provided
    // bool has_robot_goal = false;
    // if (auto it = parameters.find("robot_goal"); it != parameters.end()) {
    //     auto robot_goal = std::get<SE2State>(it->second);
    //     executor_->set_robot_goal({robot_goal.x, robot_goal.y});
    //     has_robot_goal = true;
    // } else {
    //     executor_->clear_robot_goal();
    // }
    
    // std::cout << "Starting iterative MPC execution for object: " << object_name << std::endl;
    // std::cout << "Target: [" << target_pose.x << "," << target_pose.y << "," << target_pose.theta << "]" << std::endl;
    
    // Visualize the target object goal in MuJoCo using the actual object size (cyan color)
    const ObjectInfo* obj_info = env_.get_object_info(object_name);
    if (obj_info) {
        std::array<double, 3> target_3d = {target_pose.x, target_pose.y, 0.1}; // Slightly above ground
        std::array<float, 4> cyan_color = {0.0f, 0.8f, 1.0f, 1.0f}; // Cyan for object target goals
        env_.visualize_object_goal_marker(target_3d, obj_info->size, target_pose.theta, cyan_color);
    }
    
    // **ITERATIVE MPC LOOP**
    SE2State previous_state = *get_object_current_pose(object_name); // Initialize for stuck detection
    int stuck_counter = 0;
    int previous_edge_idx = -1;
    const int max_stuck_iterations = config_ ? config_->skill().max_stuck_iterations : 2;
    std::unordered_set<int> stuck_edges;  // edges that caused a stuck outcome recently
    
    for (int mpc_iter = 0; mpc_iter < max_mpc_iterations; mpc_iter++) {
        // std::cout << "\n--- MPC Iteration " << (mpc_iter + 1) << "/" << max_mpc_iterations << " ---" << std::endl;
        
        // 1. Get current object state
        auto current_pose = get_object_current_pose(object_name);
        if (!current_pose) {
            std::cout << "Could not get current pose for object: " << object_name << " at iteration " << mpc_iter << std::endl;
            result.failure_reason = "Could not get current pose for object: " + object_name + " at iteration " + std::to_string(mpc_iter);
            return result;
        }
        
        SE2State current_state = *current_pose;
        
        // 2. Check if object is stuck (after first iteration)
        if (mpc_iter > 0) {
            if (is_object_stuck(previous_state, current_state)) {
                stuck_counter++;
                // Add the previously executed edge to stuck edges list
                if (previous_edge_idx >= 0) {
                    stuck_edges.insert(previous_edge_idx);
                    // std::cout << "Edge " << previous_edge_idx << " led to stuck state, adding to blacklist" << std::endl;
                }
                // std::cout << "Object stuck detection: " << stuck_counter << "/" << max_stuck_iterations << std::endl;
                
                if (stuck_counter >= max_stuck_iterations) {
                    // std::cout << "Object " << object_name << " stuck for " << stuck_counter << " iterations - stopping MPC" << std::endl;
                    // result.failure_reason = "Object stuck for " + std::to_string(stuck_counter) + " iterations at MPC iteration " + std::to_string(mpc_iter);
                    result.outputs["steps_executed"] = mpc_iter;
                    result.outputs["final_pose"] = current_state;
                    result.outputs["object_name"] = object_name;
                    
                    auto end_time = std::chrono::high_resolution_clock::now();
                    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    return result;
                }
            } else {
                stuck_counter = 0; // Reset stuck counter if object moved
                stuck_edges.clear(); // Object moved → forgive all previously stuck edges
                // std::cout << "Object moved, clearing stuck edges blacklist" << std::endl;
            }
        }
        // std::cout << "Current state: [" << std::fixed << std::setprecision(3)
        //           << current_state.x << "," << current_state.y << "," << current_state.theta << "]" << std::endl;
        
        // 3. Check if robot goal is reachable (early termination)
        if (has_robot_goal_ && executor_->is_robot_goal_reachable()) {
            // std::cout << "Robot goal became reachable at iteration " << mpc_iter << std::endl;
            result.success = true;
            result.outputs["robot_goal_reached"] = true;
            result.outputs["steps_executed"] = mpc_iter;
            result.outputs["final_pose"] = current_state;
            result.outputs["object_name"] = object_name;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }
        
        // 4. Check if object reached target goal
        if (is_object_at_goal(current_state, target_pose, tolerance)) {
            // std::cout << "Object reached goal at iteration " << mpc_iter << std::endl;
            result.success = true;
            result.outputs["robot_goal_reached"] = false;
            result.outputs["steps_executed"] = mpc_iter;
            result.outputs["final_pose"] = current_state;  
            result.outputs["object_name"] = object_name;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }
        
        // 5. Update reachability using wavefront planner and save for debugging
        // std::cout << "Updating wavefront and checking reachability..." << std::endl;
        // executor_->save_debug_wavefront(mpc_iter, "mpc_wavefront_reachability");
        
        std::vector<int> reachable_edges = executor_->get_reachable_edges_with_wavefront(object_name);
        // std::cout << "reachable_edges: " << reachable_edges.size() << std::endl;
        // for (int edge : reachable_edges) {
        //     std::cout << "reachable edge idx: " << edge << std::endl;
        // }
        
        // Filter out edges that previously led to a stuck outcome
        std::vector<int> filtered_edges;
        filtered_edges.reserve(reachable_edges.size());
        for (int edge : reachable_edges) {
            if (stuck_edges.count(edge) == 0) {
                filtered_edges.push_back(edge);
            }
            // else {
            //     std::cout << "edge idx: " << edge << " is a stuck edge" << std::endl;
            // }
        }
        
        // std::cout << "Reachable edges: " << reachable_edges.size() << ", after filtering stuck edges: " << filtered_edges.size() << std::endl;
        
        // Save wavefront for debugging BEFORE checking reachability
        
        if (filtered_edges.empty()) {
            // std::cout << "No reachable edges for object " << object_name << " after filtering stuck edges - stopping MPC" << std::endl;
            // executor_->save_debug_wavefront(mpc_iter, "mpc_wavefront_no_reachable_edges_after_filter");
            result.failure_reason = "No reachable edges after filtering stuck edges at iteration " + std::to_string(mpc_iter);
            result.outputs["steps_executed"] = mpc_iter;
            result.outputs["final_pose"] = current_state;
            result.outputs["object_name"] = object_name;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }
        // std::cout << "Found " << filtered_edges.size() << " filtered reachable edges" << std::endl;
        
        // 6. Plan from current state to goal
        // std::cout << "Planning from current state to goal..." << std::endl;
        std::vector<PlanStep> plan;
        try {
            GreedyPlanner* planner = get_planner_for_object(object_name);
            // std::cout << "Selected planner: " << planner->get_name() << " for object: " << object_name << std::endl;
            
            plan = planner->plan_push_sequence(current_state, target_pose, filtered_edges, 25000);
            // std::cout << "filtered_edges: " << filtered_edges.size() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Planning failed: " << e.what() << std::endl;
            result.failure_reason = "Planning failed at iteration " + std::to_string(mpc_iter) + ": " + e.what();
            result.outputs["steps_executed"] = mpc_iter;
            result.outputs["final_pose"] = current_state;
            result.outputs["object_name"] = object_name;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }
        
        if (plan.empty()) {
            // std::cout << "No plan found from current state" << std::endl;
            result.failure_reason = "No plan found at iteration " + std::to_string(mpc_iter);
            result.outputs["steps_executed"] = mpc_iter;
            result.outputs["final_pose"] = current_state;
            result.outputs["object_name"] = object_name;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            return result;
        }
        
        // std::cout << "Found plan with " << plan.size() << " steps, executing first step only" << std::endl;
        // std::cout << "First step: Edge=" << plan[0].edge_idx << " Steps=" << plan[0].push_steps << std::endl;
        
        // 7. Execute ONLY the first primitive step (key difference from full sequence execution)
        std::vector<PlanStep> single_step = {plan[0]};
        
        previous_edge_idx = single_step[0].edge_idx;  // Remember which edge we're executing for next iteration's stuck check
        // std::cout << "pushing on edge idx: " << previous_edge_idx << std::endl;
        auto step_result = executor_->execute_plan(object_name, single_step);

        if (!step_result.success) {
            stuck_edges.insert(previous_edge_idx);
        }
        previous_state = current_state;
    }
    
    // If we reach here, we hit the iteration limit
    auto final_pose = get_object_current_pose(object_name);
    // std::cout << "MPC reached iteration limit without reaching goal" << std::endl;
    result.failure_reason = "MPC reached iteration limit (" + std::to_string(max_mpc_iterations) + ") without reaching goal";
    result.outputs["steps_executed"] = max_mpc_iterations;
    result.outputs["final_pose"] = final_pose ? *final_pose : SE2State();
    result.outputs["object_name"] = object_name;
    result.outputs["robot_goal_reached"] = false;
    
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
        robot_pose.theta = 0.0;  // Robot has no orientation in this system
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
    auto target_pose = std::get<SE2State>(parameters.at("target_pose"));
    
    // Check object exists and is movable
    if (!is_object_movable(object_name)) {
        unmet.push_back("Object '" + object_name + "' does not exist or is not movable");
    }
    
    // Check target is within bounds
    if (!is_target_within_bounds(target_pose)) {
        unmet.push_back("Target pose is outside environment bounds");
    }
    
    // Reachability is determined dynamically by wavefront planning during execution
    
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

bool NAMOPushSkill::is_object_at_goal(const SE2State& current, const SE2State& goal, double tolerance) const {
    double position_error = std::sqrt(
        std::pow(current.x - goal.x, 2) + 
        std::pow(current.y - goal.y, 2)
    );
    
    double rotation_error = std::abs(current.theta - goal.theta);
    // Handle angle wrapping
    while (rotation_error > M_PI) rotation_error -= 2 * M_PI;
    while (rotation_error < -M_PI) rotation_error += 2 * M_PI;
    rotation_error = std::abs(rotation_error);
    
    return position_error <= tolerance && rotation_error <= (tolerance * 2); // More lenient on rotation
}

bool NAMOPushSkill::is_object_stuck(const SE2State& previous_state, const SE2State& current_state) const {
    double dx = current_state.x - previous_state.x;
    double dy = current_state.y - previous_state.y;
    double distance_moved = std::sqrt(dx*dx + dy*dy);
    
    double angle_change = std::abs(current_state.theta - previous_state.theta);
    while (angle_change > M_PI) angle_change = 2.0 * M_PI - angle_change;
    
    // Consider stuck if both position and orientation changes are very small
    const double min_position_change = config_ ? config_->skill().stuck_threshold : 0.01;  // From config or 1cm default
    const double min_angle_change = 0.1;      // ~0.6 degrees (no config parameter for this)
    
    return distance_moved < min_position_change && angle_change < min_angle_change;
}

std::vector<int> NAMOPushSkill::get_reachable_edges(const std::string& object_name) const {
    // Get robot current position
    auto robot_state = env_.get_robot_state();
    if (!robot_state) {
        return {}; // No reachable edges if can't get robot state
    }
    
    // Get object current position
    auto obj_pose = get_object_current_pose(object_name);
    if (!obj_pose) {
        return {}; // No reachable edges if can't get object pose
    }
    
    // This method is deprecated - the skill system now uses wavefront-based reachability
    // All edges are considered potentially reachable; actual reachability is determined
    // by the wavefront planner during execution
    std::vector<int> reachable_edges;
    for (int i = 0; i < 12; i++) {
        reachable_edges.push_back(i);
    }
    return reachable_edges;
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
    
    // Use the MPC executor to check reachability via wavefront
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

GreedyPlanner* NAMOPushSkill::get_planner_for_object(const std::string& object_name) const {
    const ObjectInfo* info = env_.get_object_info(object_name);
    GreedyPlanner* selected_planner = nullptr;
    int symmetry_rotations = 1;  // Default: no symmetry

    if (!info) {
        // std::cout << "Object info not available for " << object_name << ", defaulting to square planner" << std::endl;
        selected_planner = planner_square_.get();
        symmetry_rotations = 4;  // Assume square symmetry for unknown objects
    } else {
        double x = info->size[0];
        double y = info->size[1];

        if (x <= 0.0 || y <= 0.0) {
            // std::cout << "Invalid dimensions for " << object_name << " [" << x << "x" << y << "], defaulting to square planner" << std::endl;
            selected_planner = planner_square_.get();
            symmetry_rotations = 4;
        } else {
            // Use same 5% tolerance as ObjectInfo symmetry detection
            double ratio = std::max(x, y) / std::min(x, y);
            // std::cout << "Object " << object_name << " dimensions: [" << x << "x" << y << "], ratio: " << ratio << std::endl;

            if (ratio < 1.05) {
                // Square object: 4-way rotational symmetry
                // std::cout << "Ratio < 1.05, selecting square planner with 4-way symmetry" << std::endl;
                selected_planner = planner_square_.get();
                symmetry_rotations = 4;
            } else {
                // Rectangle object: 2-way rotational symmetry
                symmetry_rotations = 2;

                if (x > y) {
                    // std::cout << "x > y, selecting wide planner with 2-way symmetry" << std::endl;
                    selected_planner = planner_wide_.get();
                } else {
                    // std::cout << "y > x, selecting tall planner with 2-way symmetry" << std::endl;
                    selected_planner = planner_tall_.get();
                }
            }
        }

        // Use the ObjectInfo's computed symmetry if available (should match our calculation)
        if (info->symmetry_rotations > 0) {
            symmetry_rotations = info->symmetry_rotations;
        }
    }

    // Set symmetry information on the selected planner
    selected_planner->set_object_symmetry(symmetry_rotations);

    return selected_planner;
}

} // namespace namo