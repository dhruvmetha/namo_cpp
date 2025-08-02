#include "skills/namo_push_skill.hpp"
#include <chrono>

namespace namo {

NAMOPushSkill::NAMOPushSkill(NAMOEnvironment& env) 
    : env_(env), config_() {
    initialize_skill();
}

NAMOPushSkill::NAMOPushSkill(NAMOEnvironment& env, const Config& config) 
    : env_(env), config_(config) {
    initialize_skill();
}

void NAMOPushSkill::initialize_skill() {
    // Initialize planner with proper error checking
    planner_ = std::make_unique<GreedyPlanner>();
    if (!planner_->initialize(config_.primitive_database_path)) {
        throw std::runtime_error("Failed to initialize motion primitive database from: " + 
                               config_.primitive_database_path);
    }
    
    // Initialize executor
    executor_ = std::make_unique<MPCExecutor>(env_);
}

std::map<std::string, ParameterSchema> NAMOPushSkill::get_parameter_schema() const {
    return {
        {"object_name", {ParameterSchema::STRING, "Name of movable object to push"}},
        {"target_pose", {ParameterSchema::POSE_2D, "Target SE(2) pose (x, y, theta)"}},
        {"robot_goal", {ParameterSchema::POSE_2D, "Optional robot goal for early termination", 
                       SkillParameterValue(SE2State())}},  // Optional with default
        {"tolerance", {ParameterSchema::DOUBLE, "Goal tolerance in meters", 
                      SkillParameterValue(config_.tolerance)}},  // Optional with default
        {"max_attempts", {ParameterSchema::INT, "Maximum planning attempts", 
                         SkillParameterValue(config_.max_planning_attempts)}}  // Optional with default
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
    double tolerance = config_.tolerance;
    if (auto it = parameters.find("tolerance"); it != parameters.end()) {
        tolerance = std::get<double>(it->second);
    }
    
    int max_attempts = config_.max_planning_attempts;
    if (auto it = parameters.find("max_attempts"); it != parameters.end()) {
        max_attempts = std::get<int>(it->second);
    }
    
    // Set robot goal if provided
    if (auto it = parameters.find("robot_goal"); it != parameters.end()) {
        auto robot_goal = std::get<SE2State>(it->second);
        executor_->set_robot_goal({robot_goal.x, robot_goal.y});
    } else {
        executor_->clear_robot_goal();
    }
    
    // Get current object pose with proper orientation
    auto current_pose = get_object_current_pose(object_name);
    if (!current_pose) {
        result.failure_reason = "Could not get current pose for object: " + object_name;
        return result;
    }
    
    // Planning with retry logic
    std::vector<PlanStep> plan;
    bool planning_success = false;
    
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        try {
            plan = planner_->plan_push_sequence(*current_pose, target_pose);
            if (!plan.empty()) {
                planning_success = true;
                break;
            }
        } catch (const std::exception& e) {
            if (attempt == max_attempts) {
                result.failure_reason = "Planning failed after " + std::to_string(max_attempts) + 
                                      " attempts. Last error: " + e.what();
                return result;
            }
        }
    }
    
    if (!planning_success) {
        result.failure_reason = "Could not find valid plan after " + std::to_string(max_attempts) + " attempts";
        return result;
    }
    
    // Execute the plan
    auto execution_result = executor_->execute_plan(object_name, plan);
    
    // Convert to skill result
    result.success = execution_result.success;
    result.failure_reason = execution_result.failure_reason;
    
    // Add outputs
    result.outputs["final_pose"] = execution_result.final_object_state;
    result.outputs["steps_executed"] = execution_result.steps_executed;
    result.outputs["robot_goal_reached"] = execution_result.robot_goal_reached;
    result.outputs["object_name"] = object_name;
    
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
    
    // Check robot can potentially reach object
    if (auto robot_state = env_.get_robot_state()) {
        if (auto obj_pose = get_object_current_pose(object_name)) {
            double distance = std::sqrt(
                std::pow(robot_state->position[0] - obj_pose->x, 2) +
                std::pow(robot_state->position[1] - obj_pose->y, 2)
            );
            
            constexpr double MAX_REACH_DISTANCE = 5.0;  // meters
            if (distance > MAX_REACH_DISTANCE) {
                unmet.push_back("Object is too far from robot (distance: " + 
                               std::to_string(distance) + "m, max: " + 
                               std::to_string(MAX_REACH_DISTANCE) + "m)");
            }
        }
    }
    
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

} // namespace namo