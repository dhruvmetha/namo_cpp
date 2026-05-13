#include "python/namo/cpp_bindings/rl_env.hpp"
#include "core/types.hpp"
#include "wavefront/wavefront_grid.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "navigation/diff_drive_navigation.hpp"
#include "navigation/holonomic_navigation.hpp"
#include "robot/robot_adapter.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

namespace namo {

RLEnvironment::RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize)
    : xml_path_(xml_path), config_path_(config_path) {
    // std::cout << "Initializing RLEnvironment..." << std::endl;
    try {
        config_ = std::shared_ptr<ConfigManager>(ConfigManager::create_from_file(config_path).release());
        // Effective visualization: explicit arg OR config (system.enable_visualization).
        // This makes the YAML the source of truth while letting callers force-enable.
        bool effective_visualize = visualize || config_->system().enable_visualization;
        env_ = std::make_unique<NAMOEnvironment>(xml_path, config_, effective_visualize);
        skill_ = std::make_unique<NAMOPushSkill>(*env_, config_);
        
        // Cache immutable object info once during initialization
        cached_object_info_ = env_->get_all_object_info();
        
        // std::cout << "RLEnvironment initialized successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during RLEnvironment initialization: " << e.what() << std::endl;
        throw;
    }
}

RLEnvironment::~RLEnvironment() = default;

void RLEnvironment::reset() {
    env_->reset(); // reset_to_initial_state();
}

RLEnvironment::StepResult RLEnvironment::step(const Action& action) {
    std::map<std::string, SkillParameterValue> params = {
        {"object_name", action.object_id},
        {"target_pose", SE2State(action.x, action.y, action.theta)},
        {"edge_idx", action.edge_idx},  // Pass to skill for direct primitive execution
        {"depth", action.depth}         // Pass to skill for direct primitive execution
    };

    if (!skill_->is_applicable(params)) {
        return {false, -10.0, {{"failure_reason", "Action not applicable"}}};
    }

    auto result = skill_->execute(params);
    
    StepResult rl_result;
    rl_result.done = result.success;
    
    // MCTS sparse reward: +1 if robot goal reachable, -1 otherwise
    bool goal_reached = false;
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        goal_reached = std::get<bool>(it->second);
    }
    rl_result.reward = goal_reached ? 1.0 : -1.0;
    
    rl_result.info["failure_reason"] = result.failure_reason;
    rl_result.info["failure_type"] = std::to_string(static_cast<int>(result.failure_type));

    if (auto it = result.outputs.find("steps_executed"); it != result.outputs.end()) {
        rl_result.info["steps_executed"] = std::to_string(std::get<int>(it->second));
    }
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        rl_result.info["robot_goal_reached"] = std::get<bool>(it->second) ? "true" : "false";
    }
    if (auto it = result.outputs.find("collision_object"); it != result.outputs.end()) {
        rl_result.info["collision_object"] = std::get<std::string>(it->second);
    }
    if (auto it = result.outputs.find("stuck"); it != result.outputs.end()) {
        rl_result.info["stuck"] = std::get<std::string>(it->second);
    }
    // Collision tracking outputs for hardness metrics
    if (auto it = result.outputs.find("wall_collision"); it != result.outputs.end()) {
        rl_result.info["wall_collision"] = std::get<bool>(it->second) ? "true" : "false";
    }
    if (auto it = result.outputs.find("movable_collisions"); it != result.outputs.end()) {
        rl_result.info["movable_collisions"] = std::get<std::string>(it->second);
    }

    return rl_result;
}

RLEnvironment::NavigateResult RLEnvironment::navigate_to(double x, double y, double theta) {
    NavigateResult res;
    auto wrap_pi = [](double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    };

    if (!env_) { res.failure_reason = "no_env"; return res; }

    const auto* adapter = env_->get_robot_adapter();
    auto* mjw = env_->get_mujoco_wrapper();
    auto* model = mjw ? mjw->model() : nullptr;
    auto* data  = mjw ? mjw->data()  : nullptr;
    auto rs = env_->get_robot_state();
    if (!rs) { res.failure_reason = "no_robot_state"; return res; }

    // Heap-allocate the WavefrontPlanner: it carries multi-MB inline
    // grid+queue buffers that overflow the stack if declared as a local.
    const double resolution = config_ ? config_->get_skill_level_resolution() : 0.005;
    const auto& robot_info = env_->get_robot_info();
    // Inflate by chassis half-diagonal + safety margin so the wavefront path
    // keeps the chassis's rotational sweep clear AND tolerates nav drift
    // (~25 mm cross-track on average). Half-diagonal alone gives 0 margin
    // beyond the chassis sweep — adding 2.5 cm covers realistic drift.
    //
    // robot_info.size = [0.0175, 0.035] only covers ONE chassis half; full
    // chassis with wheels has half-extents [0.035, 0.0525].
    double half_x = std::max(0.035, robot_info.size[0]);
    double half_y = std::max(0.0525, robot_info.size[1]);
    const double half_diag = std::hypot(half_x, half_y);
    const double safety_margin = 0.025;
    const double inflation = half_diag + safety_margin;
    std::vector<double> robot_size = {inflation, inflation};
    auto planner = std::make_unique<WavefrontPlanner>(resolution, *env_, robot_size);

    std::vector<double> start_pos = {rs->position[0], rs->position[1]};
    if (!planner->update_wavefront(*env_, start_pos)) {
        res.failure_reason = "wavefront_failed";
        return res;
    }
    auto path = planner->extract_path({start_pos[0], start_pos[1]}, {x, y});
    if (path.empty()) { res.failure_reason = "no_path"; return res; }

    // Emit [NAV_PATH] line so render_nav_video.py --path-file can overlay
    // the wavefront waypoints. Mirrors what NAMOPushController does.
    if (std::getenv("NAMO_NAV_LOG")) {
        std::cerr << "[NAV_PATH]";
        for (const auto& p : path) std::cerr << " " << p[0] << "," << p[1];
        std::cerr << std::endl;
    }

    std::unique_ptr<NavigationStrategy> nav;
    if (adapter && adapter->is_diff_drive()) {
        nav = std::make_unique<DiffDriveNavigation>(DiffDriveNavigation::Params{});
    } else {
        nav = std::make_unique<HolonomicNavigation>();
    }
    auto nav_result = nav->execute(*env_, path, theta, /*target_object=*/"");

    res.success          = nav_result.success;
    res.failure_reason   = nav_result.failure_reason;
    res.collision_object = nav_result.collision_object;
    res.steps_used       = nav_result.steps_used;

    auto rs2 = env_->get_robot_state();
    res.final_x = rs2 ? rs2->position[0] : 0.0;
    res.final_y = rs2 ? rs2->position[1] : 0.0;
    res.final_theta = (adapter && model && data) ? adapter->get_theta(model, data) : 0.0;
    res.pos_error_m = std::hypot(x - res.final_x, y - res.final_y);
    res.yaw_error_rad = std::abs(wrap_pi(theta - res.final_theta));
    return res;
}

std::map<std::string, std::vector<double>> RLEnvironment::get_observation() const {
    auto world_state = skill_->get_world_state();
    std::map<std::string, std::vector<double>> state_map;

    for (const auto& [key, value] : world_state) {
        if (std::holds_alternative<SE2State>(value)) {
            const auto& pose = std::get<SE2State>(value);
            state_map[key] = {pose.x, pose.y, pose.theta};
        }
    }
    return state_map;
}

RLState RLEnvironment::get_full_state() const {
    // Use NAMOEnvironment's zero-allocation method, then convert to RLState
    auto full_state = env_->get_full_state();
    
    RLState rl_state;
    
    // Copy qpos
    rl_state.qpos.resize(full_state.nq);
    for (int i = 0; i < full_state.nq; i++) {
        rl_state.qpos[i] = full_state.qpos[i];
    }
    
    // Copy qvel
    rl_state.qvel.resize(full_state.nv);
    for (int i = 0; i < full_state.nv; i++) {
        rl_state.qvel[i] = full_state.qvel[i];
    }
    
    return rl_state;
}

void RLEnvironment::set_full_state(const RLState& state) {
    // Convert RLState to NAMOEnvironment::FullSimState, then use zero-allocation method
    NAMOEnvironment::FullSimState full_state;
    
    // Copy qpos
    full_state.nq = std::min(static_cast<int>(state.qpos.size()), 
                            static_cast<int>(NAMOEnvironment::FullSimState::MAX_QPOS));
    for (int i = 0; i < full_state.nq; i++) {
        full_state.qpos[i] = state.qpos[i];
    }
    
    // Always zero qvel for consistent physics simulation (matching original RL behavior)
    full_state.nv = std::min(static_cast<int>(state.qvel.size()), 
                            static_cast<int>(NAMOEnvironment::FullSimState::MAX_QVEL));
    for (int i = 0; i < full_state.nv; i++) {
        full_state.qvel[i] = 0.0;  // Zero qvel for consistent physics
    }
    
    // Use NAMOEnvironment's centralized state setting
    env_->set_full_state(full_state);
}

void RLEnvironment::render() {
    auto* sim = env_->get_mujoco_wrapper();
    sim->render();
}

void RLEnvironment::set_camera_position(double distance, double azimuth, double elevation) {
    env_->set_camera_position(distance, azimuth, elevation);
}

void RLEnvironment::set_camera_lookat(double x, double y, double z) {
    env_->set_camera_lookat({x, y, z});
}

std::vector<std::string> RLEnvironment::get_reachable_objects() const {
    return skill_->get_reachable_objects();
}

bool RLEnvironment::is_object_reachable(const std::string& object_name) const {
    return skill_->is_object_reachable(object_name);
}

std::vector<int> RLEnvironment::get_reachable_edges(const std::string& object_name) const {
    return skill_->get_reachable_edges(object_name);
}

const std::map<std::string, std::map<std::string, double>>& RLEnvironment::get_object_info() const {
    // Return cached reference - zero cost operation!
    return cached_object_info_;
}

void RLEnvironment::set_robot_goal(double x, double y, double theta) {
    // Keep the C++ environment and visualization marker in sync with the goal used
    // by the skill/executor (especially important for region-opening validation loops).
    if (env_) {
        env_->set_robot_goal({x, y});
        // Visualization-only goal marker: keep it on the ground plane so it matches
        // the XML goal site conventions used in most scenes.
        env_->visualize_goal_marker({x, y, 0.0});
    }
    skill_->set_robot_goal(x, y, theta);
}

void RLEnvironment::set_robot_goal_silent(double x, double y, double theta) {
    // Same as set_robot_goal, but do not update the visualization marker.
    // This avoids flickering the goal marker while iterating over many sampled goals.
    if (env_) {
        env_->set_robot_goal({x, y});
    }
    skill_->set_robot_goal(x, y, theta);
}

void RLEnvironment::set_goal_site_visible(bool visible) {
    if (env_) {
        env_->set_goal_site_visible(visible);
    }
}

bool RLEnvironment::is_robot_goal_reachable() const {
    return skill_->is_robot_goal_reachable();
}

std::array<double, 3> RLEnvironment::get_robot_goal() const {
    return skill_->get_robot_goal();
}

void RLEnvironment::clear_robot_goal() {
    skill_->clear_robot_goal();
}

void RLEnvironment::set_collision_checking(bool enable) {
    if (config_) {
        config_->set_collision_checking(enable);
    }
    // Propagate to the skill's controller
    if (skill_) {
        skill_->set_collision_checking(enable);
    }
}

bool RLEnvironment::get_collision_checking() const {
    return config_ ? config_->skill().check_object_collision : true;
}

// Video recording interface
void RLEnvironment::start_recording(int width, int height, int capture_frequency, size_t max_frames) {
    env_->start_recording(width, height, capture_frequency, max_frames);
}

void RLEnvironment::stop_recording() {
    env_->stop_recording();
}

bool RLEnvironment::is_recording() const {
    return env_->is_recording();
}

size_t RLEnvironment::get_frame_count() const {
    return env_->get_frame_count();
}

std::vector<std::vector<unsigned char>> RLEnvironment::get_frames() const {
    return env_->get_captured_frames();  // Returns copy
}

const std::vector<unsigned char>& RLEnvironment::get_frame_ref(size_t idx) const {
    return env_->get_captured_frame(idx);
}

void RLEnvironment::clear_frames() {
    env_->clear_captured_frames();
}

std::tuple<int, int> RLEnvironment::get_recording_dimensions() const {
    return {env_->get_frame_width(), env_->get_frame_height()};
}

std::vector<double> RLEnvironment::get_world_bounds() const {
    return env_->get_environment_bounds();
}

RLEnvironment::ActionConstraints RLEnvironment::get_action_constraints() const {
    return ActionConstraints{}; // Use default values: distance [0.3, 1.0], theta [-π, π]
}

std::tuple<RLEnvironment::RegionAdjacency, RLEnvironment::RegionEdgeObjects, RLEnvironment::RegionLabels>
RLEnvironment::get_region_connectivity() const {
    std::vector<double> robot_size = {0.15, 0.15};
    if (config_) {
        const auto& cfg_size = config_->planning().robot_size;
        if (cfg_size.size() >= 2) {
            robot_size[0] = cfg_size[0];
            robot_size[1] = cfg_size[1];
        }
    }

    WavefrontGrid grid(*env_, robot_size);
    grid.update_dynamic_grid(*env_);

    struct CoutSilencer {
        std::streambuf* original_buf;
        std::ostringstream null_stream;

        CoutSilencer() : original_buf(std::cout.rdbuf(null_stream.rdbuf())) {}
        ~CoutSilencer() { std::cout.rdbuf(original_buf); }
    } silencer;

    auto adjacency = grid.build_region_connectivity_graph(*env_);
    auto edge_objects = grid.get_region_edge_objects();
    auto region_labels = grid.get_region_labels();

    return {std::move(adjacency), std::move(edge_objects), std::move(region_labels)};
}

RLEnvironment::RegionGoalSamples RLEnvironment::sample_region_goals(int goals_per_region) const {
    if (goals_per_region <= 0) {
        return {};
    }

    std::vector<double> robot_size = {0.15, 0.15};
    if (config_) {
        const auto& cfg_size = config_->planning().robot_size;
        if (cfg_size.size() >= 2) {
            robot_size[0] = cfg_size[0];
            robot_size[1] = cfg_size[1];
        }
    }

    WavefrontGrid grid(*env_, robot_size);
    grid.update_dynamic_grid(*env_);

    struct CoutSilencer {
        std::streambuf* original_buf;
        std::ostringstream null_stream;

        CoutSilencer() : original_buf(std::cout.rdbuf(null_stream.rdbuf())) {}
        ~CoutSilencer() { std::cout.rdbuf(original_buf); }
    } silencer;

    grid.build_region_connectivity_graph(*env_);
    return grid.sample_region_goals(goals_per_region);
}

void RLEnvironment::set_robot_goal_termination(bool enable) {
    if (skill_) {
        skill_->set_robot_goal_termination(enable);
    }
}

bool RLEnvironment::get_robot_goal_termination() const {
    if (skill_) {
        return skill_->get_robot_goal_termination();
    }
    return false;
}

std::vector<int> RLEnvironment::evaluate_primitive_priorities(
    const std::string& object_name,
    const std::vector<std::array<double, 3>>& target_poses,
    const std::array<double, 2>& robot_goal) {
    if (skill_) {
        return skill_->evaluate_primitive_priorities(object_name, target_poses, robot_goal);
    }
    return std::vector<int>(target_poses.size(), 3);  // Default to priority 3
}

std::map<std::string, double> RLEnvironment::get_last_priority_profile() const {
    if (skill_) {
        return skill_->get_last_priority_profile();
    }
    return {};
}

} // namespace namo
