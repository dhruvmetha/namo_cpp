#include "wavefront/wavefront_planner.hpp"
#include "environment/namo_environment.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace namo {

WavefrontPlanner::WavefrontPlanner(double resolution, NAMOEnvironment& env,
                                   const std::vector<double>& robot_size,
                                   double tier1_inflation_margin)
    : resolution_(resolution), robot_size_(robot_size),
      tier1_inflation_margin_(tier1_inflation_margin),
      bfs_queue_(std::make_unique<std::pair<int, int>[]>(MAX_BFS_QUEUE)) {
    
    // Get environment bounds
    bounds_ = env.get_environment_bounds();
    grid_width_ = static_cast<int>((bounds_[1] - bounds_[0]) / resolution_);
    grid_height_ = static_cast<int>((bounds_[3] - bounds_[2]) / resolution_);
    
    // Allocate grids
    static_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    dynamic_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    reachability_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    
    // Initialize static obstacles
    initialize_static_grid(env);

    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);

    // Add movable objects to initial dynamic grid
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        
        // Get current object state
        const ObjectState* obj_state = env.get_object_state(obj.name);
        if (!obj_state) continue;
        
        // Create inflated object for robot size
        ObjectInfo inflated_obj = obj;

        inflated_obj.size[0] += inflate_r;
        inflated_obj.size[1] += inflate_r;

        
        // Add object footprint to dynamic grid
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        for (size_t j = 0; j < footprint.num_cells; j++) {
            int x = footprint.cells[j].first;
            int y = footprint.cells[j].second;
            if (is_valid_grid_coord(x, y)) {
                dynamic_grid_[x][y] = -1;  // Obstacle
            }
        }
        // No change detection needed - simple rebuild approach
    }

    
    // std::cout << "Initialized wavefront planner:" << std::endl;
    // std::cout << "  Grid size: " << grid_width_ << "x" << grid_height_ << std::endl;
    // std::cout << "  Resolution: " << resolution_ << "m" << std::endl;
    // std::cout << "  Bounds: [" << bounds_[0] << ", " << bounds_[1] << "] x ["
    //           << bounds_[2] << ", " << bounds_[3] << "]" << std::endl;
}

void WavefrontPlanner::initialize_static_grid(NAMOEnvironment& env) {
    auto start_time = std::chrono::high_resolution_clock::now();
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);

    // Process static obstacles once using cell CENTER (matching Python's wavefront_snapshot.py)
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x) + 0.5 * resolution_;  // Cell center
            double world_y = grid_to_world_y(y) + 0.5 * resolution_;  // Cell center

            // Check against all static objects
            const auto& static_objects = env.get_static_objects();
            for (size_t i = 0; i < env.get_num_static(); i++) {
                const auto& obj = static_objects[i];
                
                // Create inflated object for robot size
                ObjectInfo inflated_obj = obj;
                inflated_obj.size[1] += inflate_r;
                inflated_obj.size[0] += inflate_r;
                
                // Use object info directly for static objects (no state)
                ObjectState static_state;
                static_state.position = obj.position;
                static_state.quaternion = obj.quaternion;
                
                if (is_point_in_rotated_rectangle(world_x, world_y, static_state, inflated_obj)) {
                    static_grid_[x][y] = -1;  // Obstacle
                    break;
                }
            }
        }
    }
    
    // Copy static grid to dynamic grid initially
    dynamic_grid_ = static_grid_;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "Static grid initialization took " << duration.count() << " ms" << std::endl;
}

bool WavefrontPlanner::update_wavefront(NAMOEnvironment& env,
                                                  const std::vector<double>& start_pos) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Build a state fingerprint = BFS start + every movable object's pose. reachability_grid_
    // is a pure function of exactly these (see header note), so an unchanged fingerprint means
    // the cached grid is still bit-identical and we can skip the full rebuild + BFS.
    std::vector<double> fp;
    const size_t nmov = env.get_num_movable();
    fp.reserve(2 + nmov * 7);
    fp.push_back(start_pos.size() > 0 ? start_pos[0] : 0.0);
    fp.push_back(start_pos.size() > 1 ? start_pos[1] : 0.0);
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < nmov; i++) {
        const ObjectState* s = env.get_object_state(movable_objects[i].name);
        if (s) {
            fp.push_back(s->position[0]); fp.push_back(s->position[1]); fp.push_back(s->position[2]);
            fp.push_back(s->quaternion[0]); fp.push_back(s->quaternion[1]);
            fp.push_back(s->quaternion[2]); fp.push_back(s->quaternion[3]);
        } else {
            for (int k = 0; k < 7; ++k) fp.push_back(0.0);
        }
    }

    if (wf_cache_valid_ && fp == wf_cache_state_) {
        stats_.wavefront_updates++;           // logical update served from cache
        return true;                          // reachability_grid_ is still valid
    }

    // State changed (or first call): rebuild and recompute wavefront from scratch.
    recompute_wavefront(env, start_pos);
    wf_cache_state_ = std::move(fp);
    wf_cache_valid_ = true;

    // Update basic statistics
    stats_.wavefront_updates++;

    update_performance_stats(start_time, std::chrono::high_resolution_clock::now());
    return true;
}

// All change detection methods removed - no longer needed for simple rebuild approach

GridFootprint WavefrontPlanner::calculate_rotated_footprint(const ObjectInfo& obj, 
                                                                      const ObjectState& state) {
    GridFootprint footprint;
    footprint.clear();
    
    // Safety checks
    if (obj.size[0] <= 0 || obj.size[1] <= 0) {
        // std::cout << "Warning: Invalid object size [" << obj.size[0] << ", " << obj.size[1] << "]" << std::endl;
        return footprint;
    }
    
    // Check quaternion validity
    double quat_norm = std::sqrt(state.quaternion[0]*state.quaternion[0] + 
                                state.quaternion[1]*state.quaternion[1] + 
                                state.quaternion[2]*state.quaternion[2] + 
                                state.quaternion[3]*state.quaternion[3]);
    if (std::abs(quat_norm - 1.0) > 0.01) {
        // std::cout << "Warning: Invalid quaternion norm " << quat_norm << ", using identity" << std::endl;
        // Use zero rotation as fallback
    }
    
    // Calculate rotated corners
    double half_w = obj.size[0];  // Use half-width
    double half_h = obj.size[1];  // Use half-height
    double yaw = utils::quaternion_to_yaw(state.quaternion);
    double cos_a = std::cos(yaw);
    double sin_a = std::sin(yaw);
    
    // Calculate axis-aligned bounding box of rotated rectangle
    std::array<std::pair<double, double>, 4> corners = {{
        {state.position[0] + (-half_w * cos_a - -half_h * sin_a),
         state.position[1] + (-half_w * sin_a + -half_h * cos_a)},
        {state.position[0] + ( half_w * cos_a - -half_h * sin_a),
         state.position[1] + ( half_w * sin_a + -half_h * cos_a)},
        {state.position[0] + ( half_w * cos_a -  half_h * sin_a),
         state.position[1] + ( half_w * sin_a +  half_h * cos_a)},
        {state.position[0] + (-half_w * cos_a -  half_h * sin_a),
         state.position[1] + (-half_w * sin_a +  half_h * cos_a)}
    }};
    
    double min_x = corners[0].first, max_x = corners[0].first;
    double min_y = corners[0].second, max_y = corners[0].second;
    
    for (int i = 1; i < 4; i++) {
        min_x = std::min(min_x, corners[i].first);
        max_x = std::max(max_x, corners[i].first);
        min_y = std::min(min_y, corners[i].second);
        max_y = std::max(max_y, corners[i].second);
    }
    
    // Convert to grid coordinates
    int grid_min_x = std::max(0, world_to_grid_x(min_x));
    int grid_max_x = std::min(grid_width_ - 1, world_to_grid_x(max_x));
    int grid_min_y = std::max(0, world_to_grid_y(min_y));
    int grid_max_y = std::min(grid_height_ - 1, world_to_grid_y(max_y));
    
    // Test each cell in bounding box using cell CENTER (matching Python's wavefront_snapshot.py)
    for (int x = grid_min_x; x <= grid_max_x; x++) {
        for (int y = grid_min_y; y <= grid_max_y; y++) {
            double world_x = grid_to_world_x(x) + 0.5 * resolution_;  // Cell center
            double world_y = grid_to_world_y(y) + 0.5 * resolution_;  // Cell center

            if (is_point_in_rotated_rectangle(world_x, world_y, state, obj)) {
                footprint.add_cell(x, y);
            }
        }
    }
    
    return footprint;
}

bool WavefrontPlanner::is_point_in_rotated_rectangle(double px, double py, 
                                                               const ObjectState& state, 
                                                               const ObjectInfo& obj) const {
    // Transform point to object's local coordinate system
    double dx = px - state.position[0];
    double dy = py - state.position[1];
    
    // Get rotation angle
    double yaw = utils::quaternion_to_yaw(state.quaternion);
    double cos_a = std::cos(yaw);
    double sin_a = std::sin(yaw);
    
    // Rotate point to object's local frame (inverse rotation)
    double local_x = dx * cos_a + dy * sin_a;
    double local_y = -dx * sin_a + dy * cos_a;
    
    // Check if point is inside axis-aligned rectangle in local frame
    double half_w = obj.size[0];
    double half_h = obj.size[1];
    
    return (std::abs(local_x) <= half_w) && (std::abs(local_y) <= half_h);
}


std::tuple<
    std::vector<std::vector<int>>, 
    std::unordered_map<std::string, std::vector<std::array<double, 2>>>,
    std::unordered_map<std::string, std::vector<int>>
> WavefrontPlanner::compute_wavefront(
    NAMOEnvironment& env,
    const std::vector<double>& start_pos,
    const std::unordered_map<std::string, std::vector<std::array<double, 2>>>& goal_positions) {
    
    // Update wavefront first
    update_wavefront(env, start_pos);
    
    // Initialize result structures
    std::unordered_map<std::string, std::vector<std::array<double, 2>>> reachable_points;
    std::unordered_map<std::string, std::vector<int>> reachability_flags;
    
    // Check goal reachability
    for (const auto& [obj_name, edge_points] : goal_positions) {
        reachability_flags[obj_name] = std::vector<int>(edge_points.size(), 0);
        
        for (size_t i = 0; i < edge_points.size(); i++) {
            if (is_goal_reachable(edge_points[i])) {
                reachable_points[obj_name].push_back(edge_points[i]);
                reachability_flags[obj_name][i] = 1;
            }
        }
    }
    
    return {reachability_grid_, reachable_points, reachability_flags};
}

bool WavefrontPlanner::is_goal_reachable(const std::array<double, 2>& goal_pos, 
                                                   double goal_size) const {
    // Calculate grid bounds for goal region
    int min_x = std::max(0, world_to_grid_x(goal_pos[0] - goal_size));
    int max_x = std::min(grid_width_ - 1, world_to_grid_x(goal_pos[0] + goal_size));
    int min_y = std::max(0, world_to_grid_y(goal_pos[1] - goal_size));
    int max_y = std::min(grid_height_ - 1, world_to_grid_y(goal_pos[1] + goal_size));
    
    // Check if any cell in goal region is reachable (value = 1)
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (reachability_grid_[x][y] == 1) {
                return true;
            }
        }
    }
    
    return false;
}

void WavefrontPlanner::save_wavefront(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);
            file << world_x << " " << world_y << " " << reachability_grid_[x][y] << "\n";
        }
    }
    
    file.close();
}

void WavefrontPlanner::save_wavefront_iteration(const std::string& base_filename, int iteration) const {
    std::string filename = base_filename + "_iter_" + std::to_string(iteration) + ".txt";
    save_wavefront(filename);
    // std::cout << "Wavefront saved for iteration " << iteration << ": " << filename << std::endl;
}

void WavefrontPlanner::recompute_wavefront(NAMOEnvironment& env, const std::vector<double>& start_pos) {
    // 1. Rebuild dynamic grid from current object positions
    rebuild_dynamic_grid_from_current_objects(env);

    // 2. Reset all reachability values: -1=obstacle, 0=unreachable, 1=reachable
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            if (dynamic_grid_[x][y] == -1) {
                reachability_grid_[x][y] = -1;  // Obstacle
            } else {
                reachability_grid_[x][y] = 0;   // Unreachable (until proven otherwise)
            }
        }
    }

    // 3. Simple BFS for reachability from start position
    //
    // The trapped-start handling below (dilation + force-enqueue) is
    // mirrored in Python at
    // `robot_control/src/robot_control/utils/wavefront.py`
    // (`WavefrontPlanner.apply_trapped_start_recovery`). Keep both
    // sides in sync — divergence here is what produced the 2026-05-19
    // "Goal REACHABLE / path FAILED" incident.
    int start_x = world_to_grid_x(start_pos[0]);
    int start_y = world_to_grid_y(start_pos[1]);

    // ADAPTIVE CLEARING: Check if robot is trapped and clear area accordingly
    bool is_trapped = true;
    
    // Check if robot has any free neighbors (not trapped)
    if (is_valid_grid_coord(start_x, start_y)) {
        for (const auto& [dx, dy] : DIRECTIONS) {
            int nx = start_x + dx; 
            int ny = start_y + dy;
            if (is_valid_grid_coord(nx, ny) && dynamic_grid_[nx][ny] != -1) {
                is_trapped = false;
                break;
            }
        }
    }
    
    // Determine clearing radius: 1 if trapped, 0 if not trapped
    int clear_radius = is_trapped ? 2 : 0;
    
    // Clear robot position and adaptive radius around it
    for (int dx = -clear_radius; dx <= clear_radius; dx++) {
        for (int dy = -clear_radius; dy <= clear_radius; dy++) {
            int nx = start_x + dx;
            int ny = start_y + dy;
            
            if (is_valid_grid_coord(nx, ny)) {
                dynamic_grid_[nx][ny] = 0;            // Mark as free space
                reachability_grid_[nx][ny] = 0;       // Reset to unreachable (will become reachable in BFS)
            }
        }
    }
    
    reset_bfs_queue();
    if (is_valid_grid_coord(start_x, start_y)) {
        bfs_enqueue(start_x, start_y);
        reachability_grid_[start_x][start_y] = 1;  // Mark start as reachable
    }
    
    // Fast reachability BFS with closed list (using reachability_grid_ values as closed set)
    while (!bfs_empty()) {
        auto [x, y] = bfs_dequeue();
        if (x < 0) break;

        for (const auto& [dx, dy] : DIRECTIONS) {
            int nx = x + dx;
            int ny = y + dy;

            if (is_valid_grid_coord(nx, ny) &&
                reachability_grid_[nx][ny] != -1 &&
                dynamic_grid_[nx][ny] != -1 &&    // Not an obstacle
                reachability_grid_[nx][ny] == 0) { // Not already visited (closed list check)

                reachability_grid_[nx][ny] = 1;  // Mark as reachable AND visited
                bfs_enqueue(nx, ny);

            }
        }
    }

}

void WavefrontPlanner::rebuild_dynamic_grid_from_current_objects(NAMOEnvironment& env) {
    // Start fresh with static obstacles only
    dynamic_grid_ = static_grid_;
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);
    
    // Add all current movable objects directly (no incremental tracking)
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        const ObjectState* obj_state = env.get_object_state(obj.name);
        
        if (obj_state) {
            // Create inflated object for robot size
            ObjectInfo inflated_obj = obj;
            inflated_obj.size[0] += inflate_r;
            inflated_obj.size[1] += inflate_r;

            // std::cout << "inflated_obj.size: " << inflated_obj.size[0] << ", " << inflated_obj.size[1] << " " << robot_size_[0] << " " << robot_size_[1] << std::endl;
            
            // Calculate current footprint and add to grid
            GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
            add_footprint_to_dynamic_grid(footprint);
        }
    }
}

void WavefrontPlanner::add_footprint_to_dynamic_grid(const GridFootprint& footprint) {
    // Simple direct addition - no complicated change tracking
    for (size_t i = 0; i < footprint.num_cells; i++) {
        int x = footprint.cells[i].first;
        int y = footprint.cells[i].second;
        if (is_valid_grid_coord(x, y)) {
            dynamic_grid_[x][y] = -1;  // Mark as obstacle
        }
    }
}

// Removed unused full_reachability_recompute method

void WavefrontPlanner::update_performance_stats(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) const {

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats_.wavefront_time += duration.count() / 1000.0; // Convert to ms
    stats_.total_planning_time += duration.count() / 1000.0;
}

// ========== Geometric Transport Heuristic Implementation ==========

void WavefrontPlanner::compute_grid_without_object(
    NAMOEnvironment& env,
    const std::string& object_name,
    std::vector<std::vector<int>>& output_grid) {

    // Resize and copy static grid
    output_grid = static_grid_;
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);

    // Add all movable objects EXCEPT the specified one
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];

        // Skip the object we want to remove
        if (obj.name == object_name) continue;

        const ObjectState* obj_state = env.get_object_state(obj.name);
        if (!obj_state) continue;

        // Create inflated object for robot size
        ObjectInfo inflated_obj = obj;
        inflated_obj.size[0] += inflate_r;
        inflated_obj.size[1] += inflate_r;

        // Calculate footprint and add to grid
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        for (size_t j = 0; j < footprint.num_cells; j++) {
            int x = footprint.cells[j].first;
            int y = footprint.cells[j].second;
            if (is_valid_grid_coord(x, y)) {
                output_grid[x][y] = -1;  // Obstacle
            }
        }
    }
}

std::vector<std::pair<int, int>> WavefrontPlanner::get_path_cells(
    const std::vector<std::vector<int>>& grid,
    const std::array<double, 2>& start_pos,
    const std::array<double, 2>& goal_pos) {

    std::vector<std::pair<int, int>> path;

    // Convert to grid coordinates
    int start_x = world_to_grid_x(start_pos[0]);
    int start_y = world_to_grid_y(start_pos[1]);
    int goal_x = world_to_grid_x(goal_pos[0]);
    int goal_y = world_to_grid_y(goal_pos[1]);

    if (!is_valid_grid_coord(start_x, start_y) || !is_valid_grid_coord(goal_x, goal_y)) {
        return path;  // Empty path
    }

    // Check if start or goal is in obstacle
    if (grid[start_x][start_y] == -1 || grid[goal_x][goal_y] == -1) {
        return path;  // Empty path
    }

    // BFS with parent tracking.
    // Internal parent-grid sentinels (not wavefront cell semantics):
    // -1 = unvisited, -2 = start, otherwise encodes parent direction.
    constexpr int kParentUnvisited = -1;
    constexpr int kParentStart = -2;
    std::vector<std::vector<int>> parent(
        grid_width_, std::vector<int>(grid_height_, kParentUnvisited));

    // Direction encoding: 0-7 maps to DIRECTIONS indices
    reset_bfs_queue();
    bfs_enqueue(start_x, start_y);
    parent[start_x][start_y] = kParentStart;  // Mark start

    bool found = false;

    while (!bfs_empty() && !found) {
        auto [x, y] = bfs_dequeue();
        if (x < 0) break;

        // Check if we reached the goal
        if (x == goal_x && y == goal_y) {
            found = true;
            break;
        }

        for (int dir = 0; dir < 8; dir++) {
            int nx = x + DIRECTIONS[dir].first;
            int ny = y + DIRECTIONS[dir].second;

            if (is_valid_grid_coord(nx, ny) &&
                grid[nx][ny] != -1 &&      // Not obstacle
                parent[nx][ny] == kParentUnvisited) {    // Not visited

                parent[nx][ny] = dir;  // Store direction we came from
                bfs_enqueue(nx, ny);

                if (nx == goal_x && ny == goal_y) {
                    found = true;
                    break;
                }
            }
        }
    }

    if (!found) {
        return path;  // Empty path - goal unreachable
    }

    // Backtrack from goal to start
    int cx = goal_x, cy = goal_y;
    while (parent[cx][cy] != kParentStart) {  // Until we reach start
        path.push_back({cx, cy});

        int dir = parent[cx][cy];
        // Reverse the direction to go back
        cx -= DIRECTIONS[dir].first;
        cy -= DIRECTIONS[dir].second;
    }
    path.push_back({start_x, start_y});  // Add start

    // Reverse to get start->goal order
    std::reverse(path.begin(), path.end());

    return path;
}

std::vector<std::array<double, 2>> WavefrontPlanner::extract_path(
    const std::array<double, 2>& start_pos,
    const std::array<double, 2>& goal_pos) {

    // Reuse the existing BFS path finder on the current dynamic grid
    auto cells = get_path_cells(dynamic_grid_, start_pos, goal_pos);
    std::vector<std::array<double, 2>> waypoints;
    waypoints.reserve(cells.size());
    for (const auto& [gx, gy] : cells) {
        waypoints.push_back({grid_to_world_x(gx), grid_to_world_y(gy)});
    }
    return waypoints;
}

bool WavefrontPlanner::footprint_blocks_path(
    const GridFootprint& footprint,
    const std::vector<std::pair<int, int>>& path_cells) const {

    if (path_cells.empty() || footprint.num_cells == 0) {
        return false;
    }

    // Build set of path cells for O(1) lookup
    std::unordered_set<int64_t> path_set;
    for (const auto& [px, py] : path_cells) {
        // Encode (x, y) as single int64 for set lookup
        int64_t key = static_cast<int64_t>(px) * grid_height_ + py;
        path_set.insert(key);
    }

    // Check if any footprint cell is in the path
    for (size_t i = 0; i < footprint.num_cells; i++) {
        int x = footprint.cells[i].first;
        int y = footprint.cells[i].second;
        int64_t key = static_cast<int64_t>(x) * grid_height_ + y;

        if (path_set.count(key) > 0) {
            return true;  // Footprint blocks the path
        }
    }

    return false;
}

bool WavefrontPlanner::check_static_collision(
    const std::array<double, 3>& target_pose,
    const std::array<double, 3>& object_size) {

    // Create temporary object state and info for footprint calculation
    ObjectState target_state;
    target_state.position = {target_pose[0], target_pose[1], 0.0};
    target_state.quaternion = utils::yaw_to_quaternion(target_pose[2]);

    ObjectInfo obj_info;
    obj_info.size = object_size;

    // Inflate for robot size
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);
    ObjectInfo inflated_obj = obj_info;
    inflated_obj.size[0] += inflate_r;
    inflated_obj.size[1] += inflate_r;

    // Calculate footprint at target pose
    GridFootprint footprint = calculate_rotated_footprint(inflated_obj, target_state);

    // Check if any footprint cell overlaps with static obstacles
    for (size_t i = 0; i < footprint.num_cells; i++) {
        int x = footprint.cells[i].first;
        int y = footprint.cells[i].second;

        if (is_valid_grid_coord(x, y) && static_grid_[x][y] == -1) {
            return true;  // Collision with static obstacle
        }
    }

    return false;
}

std::vector<std::string> WavefrontPlanner::check_movable_collision(
    const std::string& object_name,
    const std::array<double, 3>& target_pose,
    const std::array<double, 3>& object_size,
    NAMOEnvironment& env) {

    std::vector<std::string> colliding_objects;

    // Create target object state
    ObjectState target_state;
    target_state.position = {target_pose[0], target_pose[1], 0.0};
    target_state.quaternion = utils::yaw_to_quaternion(target_pose[2]);

    ObjectInfo target_info;
    target_info.size = object_size;

    // Calculate footprint at target pose (no inflation needed for movable-movable check)
    GridFootprint target_footprint = calculate_rotated_footprint(target_info, target_state);

    // Build set of target footprint cells
    std::unordered_set<int64_t> target_cells;
    for (size_t i = 0; i < target_footprint.num_cells; i++) {
        int x = target_footprint.cells[i].first;
        int y = target_footprint.cells[i].second;
        int64_t key = static_cast<int64_t>(x) * grid_height_ + y;
        target_cells.insert(key);
    }

    // Check against all other movable objects
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];

        // Skip self
        if (obj.name == object_name) continue;

        const ObjectState* obj_state = env.get_object_state(obj.name);
        if (!obj_state) continue;

        // Calculate footprint of other object (no inflation)
        GridFootprint other_footprint = calculate_rotated_footprint(obj, *obj_state);

        // Check for overlap
        bool overlaps = false;
        for (size_t j = 0; j < other_footprint.num_cells && !overlaps; j++) {
            int x = other_footprint.cells[j].first;
            int y = other_footprint.cells[j].second;
            int64_t key = static_cast<int64_t>(x) * grid_height_ + y;

            if (target_cells.count(key) > 0) {
                overlaps = true;
            }
        }

        if (overlaps) {
            colliding_objects.push_back(obj.name);
        }
    }

    return colliding_objects;
}

std::vector<int> WavefrontPlanner::evaluate_primitive_priorities(
    NAMOEnvironment& env,
    const std::string& object_name,
    const std::vector<std::array<double, 3>>& target_poses,
    const std::array<double, 2>& robot_goal) {

    using Clock = std::chrono::steady_clock;
    auto to_ms = [](const Clock::time_point& a, const Clock::time_point& b) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
    };

    std::map<std::string, double> profile;
    const auto t_total0 = Clock::now();
    profile["target_poses_count"] = static_cast<double>(target_poses.size());

    std::vector<int> priorities(target_poses.size(), 3);  // Default to priority 3

    if (target_poses.empty()) {
        profile["total_ms"] = to_ms(t_total0, Clock::now());
        last_priority_profile_ = std::move(profile);
        return priorities;
    }

    // 1. Get object info
    const ObjectState* obj_state = env.get_object_state(object_name);
    if (!obj_state) {
        profile["total_ms"] = to_ms(t_total0, Clock::now());
        last_priority_profile_ = std::move(profile);
        return priorities;
    }

    // Get object size from movable objects
    std::array<double, 3> object_size = {0.1, 0.1, 0.1};  // Default
    const auto& movable_objects = env.get_movable_objects();
    profile["movable_objects_count"] = static_cast<double>(env.get_num_movable());
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        if (movable_objects[i].name == object_name) {
            object_size = movable_objects[i].size;
            break;
        }
    }

    // 2. Compute grid without this object (ONE grid computation)
    std::vector<std::vector<int>> base_grid;
    const auto t_grid0 = Clock::now();
    compute_grid_without_object(env, object_name, base_grid);
    profile["compute_grid_without_object_ms"] = to_ms(t_grid0, Clock::now());

    // 3. Get robot position
    const auto* robot_state = env.get_robot_state();
    std::array<double, 2> robot_pos = {
        robot_state->position[0],
        robot_state->position[1]
    };

    // 4. Get path cells from robot to goal (ONE BFS)
    const auto t_path0 = Clock::now();
    auto path_cells = get_path_cells(base_grid, robot_pos, robot_goal);
    profile["get_path_cells_ms"] = to_ms(t_path0, Clock::now());
    profile["path_cells_count"] = static_cast<double>(path_cells.size());

    // 5. If goal not reachable without object, all priorities = 3 (object isn't only blocker)
    if (path_cells.empty()) {
        std::fill(priorities.begin(), priorities.end(), 3);
        profile["total_ms"] = to_ms(t_total0, Clock::now());
        last_priority_profile_ = std::move(profile);
        return priorities;
    }

    // 6. Evaluate each target pose
    double footprint_ms = 0.0;
    double blocks_ms = 0.0;
    double static_collision_ms = 0.0;
    double movable_collision_ms = 0.0;
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);
    const auto t_loop0 = Clock::now();
    for (size_t i = 0; i < target_poses.size(); i++) {
        const auto& pose = target_poses[i];

        // Get footprint at target pose (with robot inflation)
        ObjectState target_state;
        target_state.position = {pose[0], pose[1], 0.0};
        target_state.quaternion = utils::yaw_to_quaternion(pose[2]);

        ObjectInfo inflated_info;
        inflated_info.size = object_size;
        inflated_info.size[0] += inflate_r;
        inflated_info.size[1] += inflate_r;

        auto t0 = Clock::now();
        GridFootprint footprint = calculate_rotated_footprint(inflated_info, target_state);
        footprint_ms += to_ms(t0, Clock::now());

        // Check if blocks path
        t0 = Clock::now();
        bool blocks = footprint_blocks_path(footprint, path_cells);
        blocks_ms += to_ms(t0, Clock::now());

        // Check static collision
        t0 = Clock::now();
        bool static_collision = check_static_collision(pose, object_size);
        static_collision_ms += to_ms(t0, Clock::now());

        // Check movable collision
        t0 = Clock::now();
        auto movable_hits = check_movable_collision(object_name, pose, object_size, env);
        movable_collision_ms += to_ms(t0, Clock::now());

        // Assign priority based on heuristic:
        // OPENINGS FIRST (clean -> movable -> static):
        // Priority 1: No collision, creates opening
        // Priority 2: Movable collision, creates opening
        // Priority 3: Static collision, creates opening
        // NO OPENINGS (clean -> movable -> static):
        // Priority 4: No collision, no opening
        // Priority 5: Movable collision, no opening
        // Priority 6: Static collision, no opening
        if (!blocks) {
            // Creates opening
            if (static_collision) {
                priorities[i] = 3;
            } else if (movable_hits.empty()) {
                priorities[i] = 1;
            } else {
                priorities[i] = 2;
            }
        } else {
            // No opening
            if (static_collision) {
                priorities[i] = 6;
            } else if (movable_hits.empty()) {
                priorities[i] = 4;
            } else {
                priorities[i] = 5;
            }
        }
    }

    profile["per_pose_loop_ms"] = to_ms(t_loop0, Clock::now());
    profile["per_pose_calculate_footprint_ms"] = footprint_ms;
    profile["per_pose_blocks_path_ms"] = blocks_ms;
    profile["per_pose_check_static_collision_ms"] = static_collision_ms;
    profile["per_pose_check_movable_collision_ms"] = movable_collision_ms;
    profile["total_ms"] = to_ms(t_total0, Clock::now());
    last_priority_profile_ = std::move(profile);
    return priorities;
}

} // namespace namo
