#include "python/namo/cpp_bindings/rl_env.hpp"
#include "core/types.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include "wavefront/wavefront_grid.hpp"
#include "skills/manipulation_skill.hpp"  // For SkillResult
#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <queue>
#include <algorithm>
#include <variant>

namespace namo {


namespace {

bool parse_goal_position(const std::string& pos_str, std::array<double, 3>& goal_pose) {
    std::istringstream ss(pos_str);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(ss >> x >> y)) {
        return false;
    }
    if (!(ss >> z)) {
        z = 0.0;
    }
    goal_pose = {x, y, z};
    return true;
}

bool extract_xml_goal_pose(const std::string& xml_path, std::array<double, 3>& goal_pose) {
    std::ifstream file(xml_path);
    if (!file.is_open()) {
        return false;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string xml = buffer.str();

    const std::regex site_regex(R"(<site\b([^>]*)>)");
    const std::regex name_regex(R"re(name\s*=\s*"([^\"]+)")re");
    const std::regex pos_regex(R"re(pos\s*=\s*"([^\"]+)")re");

    struct Candidate {
        std::string name;
        std::array<double, 3> pose;
    };

    std::vector<Candidate> candidates;
    for (std::sregex_iterator it(xml.begin(), xml.end(), site_regex), end; it != end; ++it) {
        const std::string attrs = (*it)[1].str();

        std::smatch name_match;
        if (!std::regex_search(attrs, name_match, name_regex)) {
            continue;
        }
        const std::string name = name_match[1].str();
        if (name.find("goal") == std::string::npos) {
            continue;
        }

        std::smatch pos_match;
        if (!std::regex_search(attrs, pos_match, pos_regex)) {
            continue;
        }

        std::array<double, 3> pose;
        if (!parse_goal_position(pos_match[1].str(), pose)) {
            continue;
        }
        candidates.push_back({name, pose});
    }

    if (candidates.empty()) {
        return false;
    }

    for (const auto& candidate : candidates) {
        if (candidate.name == "goal") {
            goal_pose = candidate.pose;
            return true;
        }
    }

    goal_pose = candidates.front().pose;
    return true;
}

std::string find_robot_label(const std::unordered_map<int, std::string>& region_labels) {
    for (const auto& [region_id, label] : region_labels) {
        (void)region_id;
        if (label.find("robot") != std::string::npos) {
            return label;
        }
    }
    return "";
}

std::string find_goal_label(const std::unordered_map<int, std::string>& region_labels) {
    for (const auto& [region_id, label] : region_labels) {
        (void)region_id;
        if (label == "goal") {
            return label;
        }
    }
    for (const auto& [region_id, label] : region_labels) {
        (void)region_id;
        if (label.find("goal") != std::string::npos) {
            return label;
        }
    }
    return "";
}

void restrict_to_local_regions(
    std::unordered_map<std::string, std::unordered_set<std::string>>& adjacency,
    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>>& edge_objects,
    std::unordered_map<int, std::string>& region_labels,
    const std::string& robot_label) {
    if (robot_label.empty()) {
        return;
    }

    const auto neighbors_it = adjacency.find(robot_label);
    if (neighbors_it == adjacency.end()) {
        return;
    }

    const std::unordered_set<std::string> neighbours = neighbors_it->second;

    std::unordered_map<std::string, std::unordered_set<std::string>> filtered_adjacency;
    filtered_adjacency[robot_label] = neighbours;
    for (const auto& neighbour : neighbours) {
        filtered_adjacency[neighbour] = {robot_label};
    }
    adjacency = std::move(filtered_adjacency);

    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>> filtered_edge_objects;
    const auto robot_edges_it = edge_objects.find(robot_label);
    for (const auto& neighbour : neighbours) {
        if (robot_edges_it != edge_objects.end()) {
            auto edge_it = robot_edges_it->second.find(neighbour);
            if (edge_it != robot_edges_it->second.end()) {
                filtered_edge_objects[robot_label][neighbour] = edge_it->second;
            }
        }

        auto neighbour_edges_it = edge_objects.find(neighbour);
        if (neighbour_edges_it != edge_objects.end()) {
            auto back_edge_it = neighbour_edges_it->second.find(robot_label);
            if (back_edge_it != neighbour_edges_it->second.end()) {
                filtered_edge_objects[neighbour][robot_label] = back_edge_it->second;
            }
        }
    }
    edge_objects = std::move(filtered_edge_objects);

    std::unordered_map<int, std::string> filtered_labels;
    for (const auto& [region_id, label] : region_labels) {
        if (label == robot_label || neighbours.find(label) != neighbours.end()) {
            filtered_labels[region_id] = label;
        }
    }
    region_labels = std::move(filtered_labels);
}

std::vector<std::array<double, 2>> build_goal_cells(
    const WavefrontGrid& grid,
    const std::array<double, 2>& goal_xy,
    double goal_radius) {
    std::vector<std::array<double, 2>> goal_cells;
    const double resolution = grid.get_resolution();
    const int center_x = grid.world_to_grid_x(goal_xy[0]);
    const int center_y = grid.world_to_grid_y(goal_xy[1]);
    if (!grid.is_valid_grid_coord(center_x, center_y)) {
        return goal_cells;
    }

    const int radius_cells = std::max(0, static_cast<int>(std::ceil(goal_radius / resolution)));
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
        for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
            const double dist_sq = static_cast<double>(dx * dx + dy * dy) * resolution * resolution;
            if (dist_sq > goal_radius * goal_radius) {
                continue;
            }
            const int gx = center_x + dx;
            const int gy = center_y + dy;
            if (!grid.is_valid_grid_coord(gx, gy)) {
                continue;
            }
            goal_cells.push_back({
                grid.grid_to_world_x(gx) + 0.5 * resolution,
                grid.grid_to_world_y(gy) + 0.5 * resolution,
            });
        }
    }

    if (goal_cells.empty()) {
        goal_cells.push_back({
            grid.grid_to_world_x(center_x) + 0.5 * resolution,
            grid.grid_to_world_y(center_y) + 0.5 * resolution,
        });
    }
    return goal_cells;
}

}  // namespace

RLEnvironment::RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize,
                             bool skip_warmup)
    : xml_path_(xml_path), config_path_(config_path) {
    // std::cout << "Initializing RLEnvironment..." << std::endl;
    try {
        config_ = std::shared_ptr<ConfigManager>(ConfigManager::create_from_file(config_path).release());
        env_ = std::make_unique<NAMOEnvironment>(xml_path, config_, visualize, /*enable_logging=*/false, skip_warmup);
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

    try {
        if (!skill_->is_applicable(params)) {
            return {false, -10.0, {{"failure_reason", "Action not applicable"}}};
        }
    } catch (const std::bad_variant_access& e) {
        std::cerr << "[rl_env::step] bad_variant_access in is_applicable: " << e.what()
                  << " | action: obj=" << action.object_id
                  << " edge=" << action.edge_idx << " depth=" << action.depth
                  << " pose=(" << action.x << "," << action.y << "," << action.theta << ")"
                  << std::endl;
        return {false, -10.0, {{"failure_reason", std::string("bad_variant_access in is_applicable: ") + e.what()}}};
    }

    SkillResult result;
    try {
        result = skill_->execute(params);
    } catch (const std::bad_variant_access& e) {
        std::cerr << "[rl_env::step] bad_variant_access in execute: " << e.what()
                  << " | action: obj=" << action.object_id
                  << " edge=" << action.edge_idx << " depth=" << action.depth
                  << " pose=(" << action.x << "," << action.y << "," << action.theta << ")"
                  << std::endl;
        return {false, -10.0, {{"failure_reason", std::string("bad_variant_access in execute: ") + e.what()}}};
    } catch (const std::exception& e) {
        std::cerr << "[rl_env::step] std::exception in execute: " << e.what()
                  << " | action: obj=" << action.object_id
                  << " edge=" << action.edge_idx << " depth=" << action.depth
                  << std::endl;
        return {false, -10.0, {{"failure_reason", std::string("exception in execute: ") + e.what()}}};
    }

    StepResult rl_result;
    rl_result.done = result.success;

    // Diagnostic helper: extract a variant alternative safely. If the variant
    // currently holds a different type than expected, log the mismatch and
    // return a sentinel — never throw bad_variant_access.
    auto get_or_warn_bool = [&](const char* key, const SkillParameterValue& v) -> bool {
        if (auto* p = std::get_if<bool>(&v)) return *p;
        std::cerr << "[rl_env::step WARN] outputs['" << key << "'] expected bool, got variant index "
                  << v.index() << " — returning false" << std::endl;
        return false;
    };
    auto get_or_warn_int = [&](const char* key, const SkillParameterValue& v) -> int {
        if (auto* p = std::get_if<int>(&v)) return *p;
        std::cerr << "[rl_env::step WARN] outputs['" << key << "'] expected int, got variant index "
                  << v.index() << " — returning 0" << std::endl;
        return 0;
    };
    auto get_or_warn_string = [&](const char* key, const SkillParameterValue& v) -> std::string {
        if (auto* p = std::get_if<std::string>(&v)) return *p;
        std::cerr << "[rl_env::step WARN] outputs['" << key << "'] expected string, got variant index "
                  << v.index() << " — returning empty" << std::endl;
        return std::string{};
    };

    // MCTS sparse reward: +1 if robot goal reachable, -1 otherwise
    bool goal_reached = false;
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        goal_reached = get_or_warn_bool("robot_goal_reached", it->second);
    }
    rl_result.reward = goal_reached ? 1.0 : -1.0;

    rl_result.info["failure_reason"] = result.failure_reason;
    rl_result.info["failure_type"] = std::to_string(static_cast<int>(result.failure_type));

    if (auto it = result.outputs.find("steps_executed"); it != result.outputs.end()) {
        rl_result.info["steps_executed"] = std::to_string(get_or_warn_int("steps_executed", it->second));
    }
    if (auto it = result.outputs.find("robot_goal_reached"); it != result.outputs.end()) {
        rl_result.info["robot_goal_reached"] = get_or_warn_bool("robot_goal_reached", it->second) ? "true" : "false";
    }
    if (auto it = result.outputs.find("collision_object"); it != result.outputs.end()) {
        rl_result.info["collision_object"] = get_or_warn_string("collision_object", it->second);
    }
    if (auto it = result.outputs.find("stuck"); it != result.outputs.end()) {
        rl_result.info["stuck"] = get_or_warn_string("stuck", it->second);
    }
    if (auto it = result.outputs.find("wall_collision"); it != result.outputs.end()) {
        rl_result.info["wall_collision"] = get_or_warn_bool("wall_collision", it->second) ? "true" : "false";
    }
    if (auto it = result.outputs.find("movable_collisions"); it != result.outputs.end()) {
        rl_result.info["movable_collisions"] = get_or_warn_string("movable_collisions", it->second);
    }

    return rl_result;
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

RLEnvironment::ReachabilitySummary RLEnvironment::get_reachability_summary(bool analysis_mode) const {
    ReachabilitySummary summary;
    if (!skill_) {
        return summary;
    }

    const int push_depth_count = config_ ? config_->skill().max_push_steps : 10;
    PushPrimitiveExecutor::ReachabilitySnapshot snapshot = skill_->get_reachability_snapshot();
    summary.goal_reachable = snapshot.goal_reachable;

    const auto& movable_objects = env_->get_movable_objects();
    for (size_t i = 0; i < env_->get_num_movable(); ++i) {
        const auto& obj = movable_objects[i];
        if (obj.name.empty()) {
            continue;
        }

        ObjectReachabilitySummary obj_summary;
        auto it = snapshot.object_edges.find(obj.name);
        if (it != snapshot.object_edges.end()) {
            obj_summary.reachable_edges = static_cast<int>(it->second.edge_indices.size());
            obj_summary.total_edges = it->second.total_edge_points;
            obj_summary.reachable = obj_summary.reachable_edges > 0;
            if (analysis_mode) {
                obj_summary.reachable_edge_indices = it->second.edge_indices;
            }
        }

        obj_summary.total_primitives = obj_summary.total_edges * push_depth_count;
        obj_summary.reachable_primitives = obj_summary.reachable_edges * push_depth_count;
        summary.objects[obj.name] = std::move(obj_summary);
    }

    return summary;
}

const std::map<std::string, std::map<std::string, double>>& RLEnvironment::get_object_info() const {
    // Return cached reference - zero cost operation!
    return cached_object_info_;
}

void RLEnvironment::warm_up() {
    if (!env_) {
        throw std::runtime_error("RLEnvironment::warm_up called before env init");
    }
    env_->warm_up();
}

void RLEnvironment::set_robot_pose(double x, double y, double theta) {
    if (!env_) {
        throw std::runtime_error("RLEnvironment::set_robot_pose called before env init");
    }
    // Delegate to the robot adapter's set_se2 via NAMOEnvironment::
    // set_robot_position. The 3-arg overload takes {x, y, theta} (theta in
    // radians, x/y in meters, world frame). Zero qvel is handled inside the
    // adapter so we don't get a spurious push from leftover wheel velocity
    // after the teleport.
    env_->set_robot_position(std::array<double, 3>{x, y, theta});
}

void RLEnvironment::set_robot_goal(double x, double y, double theta) {
    // Keep the C++ environment and visualization marker in sync with the goal used
    // by the skill/executor (especially important for region-opening validation loops).
    if (env_) {
        std::vector<double> robot_size = {kDefaultWavefrontRobotRadiusM, kDefaultWavefrontRobotRadiusM};
        double tier1_margin = kDefaultWavefrontTier1MarginM;
        if (config_) {
            const auto& cfg_size = config_->planning().robot_size;
            if (cfg_size.size() >= 2) {
                robot_size[0] = cfg_size[0];
                robot_size[1] = cfg_size[1];
            }
            tier1_margin = config_->planning().wavefront_tier1_inflation_margin;
        }
        const double goal_radius = compute_goal_tolerance_m(robot_size, tier1_margin);

        env_->set_robot_goal({x, y, theta});
        // Visualization-only goal marker: keep it on the ground plane so it matches
        // the XML goal site conventions used in most scenes.
        env_->visualize_goal_marker({x, y, 0.0}, {0.0f, 1.0f, 0.0f, 1.0f}, goal_radius);
    }
    skill_->set_robot_goal(x, y, theta);
}

void RLEnvironment::set_robot_goal_silent(double x, double y, double theta) {
    // Same as set_robot_goal, but do not update the visualization marker.
    // This avoids flickering the goal marker while iterating over many sampled goals.
    if (env_) {
        env_->set_robot_goal({x, y, theta});
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

std::pair<int, int> RLEnvironment::count_reachable_points(
    const std::vector<std::array<double, 2>>& points) const {
    return skill_->count_reachable_points(points);
}

std::array<double, 3> RLEnvironment::get_robot_goal() const {
    return skill_->get_robot_goal();
}

void RLEnvironment::clear_robot_goal() {
    skill_->clear_robot_goal();
    if (env_) {
        // Park the runtime sphere far underground so it stops drawing at the
        // last set position. The visualize_goal_marker API has no explicit hide;
        // a far-Z relocation is the least invasive way to suppress the stale
        // marker that set_robot_goal left behind.
        env_->visualize_goal_marker({0.0, 0.0, -1000.0},
                                    {0.0f, 1.0f, 0.0f, 0.0f},
                                    0.001);
    }
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

void RLEnvironment::set_robot_trajectory_collision_checking(bool enable) {
    if (skill_) {
        skill_->set_robot_trajectory_collision_checking(enable);
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
    std::vector<double> robot_size = {kDefaultWavefrontRobotRadiusM, kDefaultWavefrontRobotRadiusM};
    double tier1_margin = kDefaultWavefrontTier1MarginM;
    if (config_) {
        const auto& cfg_size = config_->planning().robot_size;
        if (cfg_size.size() >= 2) {
            robot_size[0] = cfg_size[0];
            robot_size[1] = cfg_size[1];
        }
        tier1_margin = config_->planning().wavefront_tier1_inflation_margin;
    }
    const double goal_radius = compute_goal_tolerance_m(robot_size, tier1_margin);

    auto snapshot = get_region_snapshot(
        /*goals_per_region=*/0,
        /*goal_radius=*/goal_radius,
        /*local_info_only=*/false,
        /*seed=*/42,
        /*use_xml_goal=*/true
    );
    return {
        std::move(snapshot.adjacency),
        std::move(snapshot.edge_objects),
        std::move(snapshot.region_labels),
    };
}

RLEnvironment::RegionGoalSamples RLEnvironment::sample_region_goals(int goals_per_region) const {
    std::vector<double> robot_size = {kDefaultWavefrontRobotRadiusM, kDefaultWavefrontRobotRadiusM};
    double tier1_margin = kDefaultWavefrontTier1MarginM;
    if (config_) {
        const auto& cfg_size = config_->planning().robot_size;
        if (cfg_size.size() >= 2) {
            robot_size[0] = cfg_size[0];
            robot_size[1] = cfg_size[1];
        }
        tier1_margin = config_->planning().wavefront_tier1_inflation_margin;
    }
    const double goal_radius = compute_goal_tolerance_m(robot_size, tier1_margin);

    return get_region_snapshot(
        /*goals_per_region=*/goals_per_region,
        /*goal_radius=*/goal_radius,
        /*local_info_only=*/false,
        /*seed=*/42,
        /*use_xml_goal=*/true
    ).region_goals;
}

RLEnvironment::RegionSnapshot RLEnvironment::get_region_snapshot(
    int goals_per_region,
    double goal_radius,
    bool local_info_only,
    unsigned int seed,
    bool use_xml_goal) const {
    RegionSnapshot snapshot;

    std::vector<double> robot_size = {kDefaultWavefrontRobotRadiusM, kDefaultWavefrontRobotRadiusM};
    double tier1_margin = kDefaultWavefrontTier1MarginM;
    if (config_) {
        const auto& cfg_size = config_->planning().robot_size;
        if (cfg_size.size() >= 2) {
            robot_size[0] = cfg_size[0];
            robot_size[1] = cfg_size[1];
        }
        tier1_margin = config_->planning().wavefront_tier1_inflation_margin;
    }

    WavefrontGrid grid(*env_, robot_size, tier1_margin);
    // NOTE: the WavefrontGrid ctor already calls rebuild_grids(*env_); update_dynamic_grid()
    // is just rebuild_grids() again on the same (unchanged) env, so it was a redundant second
    // full-grid rebuild per snapshot. Dropped — behavior-identical, ~half the snapshot rebuild cost.

    struct CoutSilencer {
        std::streambuf* original_buf;
        std::ostringstream null_stream;

        CoutSilencer() : original_buf(std::cout.rdbuf(null_stream.rdbuf())) {}
        ~CoutSilencer() { std::cout.rdbuf(original_buf); }
    } silencer;

    auto goal_pose_se2 = env_->get_robot_goal();
    std::array<double, 2> goal_xy = {goal_pose_se2[0], goal_pose_se2[1]};
    if (use_xml_goal) {
        std::array<double, 3> xml_goal_pose{};
        if (extract_xml_goal_pose(xml_path_, xml_goal_pose)) {
            goal_xy = {xml_goal_pose[0], xml_goal_pose[1]};
        }
    }

    const double effective_goal_radius =
        (goal_radius > 0.0) ? goal_radius : compute_goal_tolerance_m(robot_size, tier1_margin);

    const ObjectState* robot_state = env_->get_robot_state();
    if (!robot_state) {
        return snapshot;
    }
    const std::array<double, 2> robot_xy = {robot_state->position[0], robot_state->position[1]};
    const auto goal_cells = build_goal_cells(grid, goal_xy, effective_goal_radius);
    grid.find_connected_components(robot_xy, goal_cells);

    snapshot.adjacency = grid.build_region_connectivity_graph(*env_);
    snapshot.edge_objects = grid.get_region_edge_objects();
    snapshot.region_labels = grid.get_region_labels();

    snapshot.robot_label = find_robot_label(snapshot.region_labels);
    snapshot.goal_label = find_goal_label(snapshot.region_labels);
    snapshot.goal_reachable =
        snapshot.robot_label.find("goal") != std::string::npos;
    snapshot.goal_in_free_space =
        !snapshot.goal_label.empty() || snapshot.goal_reachable;

    if (goals_per_region > 0) {
        snapshot.region_goals = grid.sample_region_goals(goals_per_region, seed);
    }

    if (local_info_only) {
        restrict_to_local_regions(
            snapshot.adjacency,
            snapshot.edge_objects,
            snapshot.region_labels,
            snapshot.robot_label
        );
    }

    return snapshot;
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
