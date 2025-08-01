#include "planning/incremental_wavefront_planner.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace namo {

IncrementalWavefrontPlanner::IncrementalWavefrontPlanner(double resolution, NAMOEnvironment& env, 
                                                       const std::vector<double>& robot_size)
    : resolution_(resolution), robot_size_(robot_size) {
    
    // Get environment bounds
    bounds_ = env.get_environment_bounds();
    grid_width_ = static_cast<int>((bounds_[1] - bounds_[0]) / resolution_);
    grid_height_ = static_cast<int>((bounds_[3] - bounds_[2]) / resolution_);
    
    // Allocate grids
    static_grid_.resize(grid_width_, std::vector<int>(grid_height_, -1));
    dynamic_grid_.resize(grid_width_, std::vector<int>(grid_height_, -1));
    reachability_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    
    // Initialize static obstacles
    initialize_static_grid(env);

    // Add movable objects to initial dynamic grid
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        
        // Get current object state
        const ObjectState* obj_state = env.get_object_state(obj.name);
        if (!obj_state) continue;
        
        // Create inflated object for robot size
        ObjectInfo inflated_obj = obj;
        inflated_obj.size[0] += robot_size_[0];
        inflated_obj.size[1] += robot_size_[1];
        
        // Add object footprint to dynamic grid
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        for (size_t j = 0; j < footprint.num_cells; j++) {
            int x = footprint.cells[j].first;
            int y = footprint.cells[j].second;
            if (is_valid_grid_coord(x, y)) {
                dynamic_grid_[x][y] = -2;  // Obstacle
            }
        }
        
        // Initialize object snapshot for future change detection
        object_snapshots_[obj.name] = RotatingObjectSnapshot{
            .position = {obj_state->position[0], obj_state->position[1], obj_state->position[2]},
            .quaternion = obj_state->quaternion,
            .position_changed = false,
            .rotation_changed = false
        };
    }

    
    std::cout << "Initialized incremental wavefront planner:" << std::endl;
    std::cout << "  Grid size: " << grid_width_ << "x" << grid_height_ << std::endl;
    std::cout << "  Resolution: " << resolution_ << "m" << std::endl;
    std::cout << "  Bounds: [" << bounds_[0] << ", " << bounds_[1] << "] x ["
              << bounds_[2] << ", " << bounds_[3] << "]" << std::endl;
}

void IncrementalWavefrontPlanner::initialize_static_grid(NAMOEnvironment& env) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process static obstacles once
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);
            
            // Check against all static objects
            const auto& static_objects = env.get_static_objects();
            for (size_t i = 0; i < env.get_num_static(); i++) {
                const auto& obj = static_objects[i];
                
                // Create inflated object for robot size
                ObjectInfo inflated_obj = obj;
                inflated_obj.size[0] += robot_size_[0];
                inflated_obj.size[1] += robot_size_[1];
                
                // Use object info directly for static objects (no state)
                ObjectState static_state;
                static_state.position = obj.position;
                static_state.quaternion = obj.quaternion;
                
                if (is_point_in_rotated_rectangle(world_x, world_y, static_state, inflated_obj)) {
                    static_grid_[x][y] = -2;  // Obstacle
                    break;
                }
            }
        }
    }
    
    // Copy static grid to dynamic grid initially
    dynamic_grid_ = static_grid_;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Static grid initialization took " << duration.count() << " ms" << std::endl;
}

bool IncrementalWavefrontPlanner::update_wavefront(NAMOEnvironment& env, 
                                                  const std::vector<double>& start_pos) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Detect changed objects (for statistics only)
    detect_object_changes(env);
    
    // Update affected grid cells 
    update_affected_cells(env);
    
    // Always do full reachability recompute - fast and correct
    full_reachability_recompute(start_pos);
    
    // Update statistics
    stats_.wavefront_updates++;
    stats_.object_movements_detected += num_changed_objects_;
    stats_.grid_changes_processed += num_pending_changes_;
    
    update_performance_stats(start_time, std::chrono::high_resolution_clock::now());
    return num_changed_objects_ > 0; // Return true if objects moved
}

void IncrementalWavefrontPlanner::detect_object_changes(NAMOEnvironment& env) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    num_changed_objects_ = 0;
    
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        
        if (has_object_moved(obj.name, env)) {
            if (num_changed_objects_ < changed_objects_.size()) {
                changed_objects_[num_changed_objects_++] = obj.name;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    stats_.change_detection_time += duration.count() / 1000.0; // Convert to ms
}

bool IncrementalWavefrontPlanner::has_object_moved(const std::string& obj_name, NAMOEnvironment& env) {
    const ObjectState* current_state = env.get_object_state(obj_name);
    if (!current_state) return false;
    
    auto& snapshot = object_snapshots_[obj_name];
    
    // Check position change
    const double POS_THRESHOLD = 1e-4;  // 0.1mm
    bool position_changed = false;
    for (int i = 0; i < 3; i++) {
        if (std::abs(current_state->position[i] - snapshot.position[i]) > POS_THRESHOLD) {
            position_changed = true;
            break;
        }
    }
    
    // Check rotation change
    const double ROT_THRESHOLD = 1e-3;  // ~0.06 degrees
    bool rotation_changed = false;
    for (int i = 0; i < 4; i++) {
        if (std::abs(current_state->quaternion[i] - snapshot.quaternion[i]) > ROT_THRESHOLD) {
            rotation_changed = true;
            break;
        }
    }
    
    if (position_changed || rotation_changed) {
        snapshot.position_changed = position_changed;
        snapshot.rotation_changed = rotation_changed;
        snapshot.needs_update = true;
        return true;
    }
    
    return false;
}

void IncrementalWavefrontPlanner::update_affected_cells(NAMOEnvironment& env) {
    num_pending_changes_ = 0;
    
    // For each changed object, find affected cells
    for (size_t i = 0; i < num_changed_objects_; i++) {
        const std::string& obj_name = changed_objects_[i];
        const ObjectInfo* obj_info = env.get_object_info(obj_name);
        const ObjectState* obj_state = env.get_object_state(obj_name);
        
        if (obj_info && obj_state) {
            auto& snapshot = object_snapshots_[obj_name];
            handle_combined_motion(*obj_info, *obj_state, snapshot, env);
        }
    }
    
    // Apply all changes at once
    apply_pending_changes();
}

void IncrementalWavefrontPlanner::handle_combined_motion(const ObjectInfo& obj, 
                                                        const ObjectState& current_state,
                                                        RotatingObjectSnapshot& snapshot,
                                                        NAMOEnvironment& env) {
    
    // Calculate old and new footprints
    ObjectState old_state;
    old_state.position = snapshot.position;
    old_state.quaternion = snapshot.quaternion;
    
    ObjectInfo inflated_obj = obj;
    inflated_obj.size[0] += robot_size_[0];
    inflated_obj.size[1] += robot_size_[1];
    
    GridFootprint old_footprint = calculate_rotated_footprint(inflated_obj, old_state);
    GridFootprint new_footprint = calculate_rotated_footprint(inflated_obj, current_state);
    
    // Find differences
    find_footprint_differences(old_footprint, new_footprint);
    
    // Update snapshot
    snapshot.update_from_state(current_state);
    snapshot.cached_footprint = new_footprint;
    snapshot.needs_update = false;
}

GridFootprint IncrementalWavefrontPlanner::calculate_rotated_footprint(const ObjectInfo& obj, 
                                                                      const ObjectState& state) {
    GridFootprint footprint;
    footprint.clear();
    
    // Calculate rotated corners
    double half_w = obj.size[0];
    double half_h = obj.size[1];
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
    
    // Test each cell in bounding box
    for (int x = grid_min_x; x <= grid_max_x; x++) {
        for (int y = grid_min_y; y <= grid_max_y; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);
            
            if (is_point_in_rotated_rectangle(world_x, world_y, state, obj)) {
                footprint.add_cell(x, y);
            }
        }
    }
    
    return footprint;
}

bool IncrementalWavefrontPlanner::is_point_in_rotated_rectangle(double px, double py, 
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

void IncrementalWavefrontPlanner::find_footprint_differences(const GridFootprint& old_footprint, 
                                                            const GridFootprint& new_footprint) {
    // Use sets for fast lookup
    std::unordered_set<uint64_t> old_cells, new_cells;
    
    // Convert old footprint to set
    for (size_t i = 0; i < old_footprint.num_cells; i++) {
        uint64_t key = utils::pack_coords(old_footprint.cells[i].first, old_footprint.cells[i].second);
        old_cells.insert(key);
    }
    
    // Convert new footprint to set  
    for (size_t i = 0; i < new_footprint.num_cells; i++) {
        uint64_t key = utils::pack_coords(new_footprint.cells[i].first, new_footprint.cells[i].second);
        new_cells.insert(key);
    }
    
    // Find cells that became obstacles (in new, not in old)
    for (const auto& key : new_cells) {
        if (old_cells.find(key) == old_cells.end()) {
            auto [x, y] = utils::unpack_coords(key);
            queue_cell_change(x, y, true);  // became obstacle
        }
    }
    
    // Find cells that became free (in old, not in new)
    for (const auto& key : old_cells) {
        if (new_cells.find(key) == new_cells.end()) {
            auto [x, y] = utils::unpack_coords(key);
            queue_cell_change(x, y, false);  // became free
        }
    }
}

void IncrementalWavefrontPlanner::queue_cell_change(int x, int y, bool became_obstacle) {
    if (num_pending_changes_ < MAX_CHANGES) {
        pending_changes_[num_pending_changes_++] = GridChange(x, y, became_obstacle);
    }
}

void IncrementalWavefrontPlanner::apply_pending_changes() {
    for (size_t i = 0; i < num_pending_changes_; i++) {
        const auto& change = pending_changes_[i];
        if (is_valid_grid_coord(change.x, change.y)) {
            dynamic_grid_[change.x][change.y] = change.became_obstacle ? -2 : -1;
        }
    }
}


std::tuple<
    std::vector<std::vector<int>>, 
    std::unordered_map<std::string, std::vector<std::array<double, 2>>>,
    std::unordered_map<std::string, std::vector<int>>
> IncrementalWavefrontPlanner::compute_wavefront(
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

bool IncrementalWavefrontPlanner::is_goal_reachable(const std::array<double, 2>& goal_pos, 
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

bool IncrementalWavefrontPlanner::is_point_in_goal_region(double px, double py, 
                                                         const std::array<double, 2>& goal_pos,
                                                         double goal_size) const {
    return std::abs(px - goal_pos[0]) <= goal_size && std::abs(py - goal_pos[1]) <= goal_size;
}

void IncrementalWavefrontPlanner::save_wavefront(const std::string& filename) const {
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

void IncrementalWavefrontPlanner::full_reachability_recompute(const std::vector<double>& start_pos) {
    // Apply all pending changes to dynamic grid first
    apply_pending_changes();
    
    // Reset all reachability values: -2=obstacle, 0=unreachable, 1=reachable
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            if (dynamic_grid_[x][y] == -2) {
                reachability_grid_[x][y] = -2;  // Obstacle
            } else {
                reachability_grid_[x][y] = 0;   // Unreachable (until proven otherwise)
            }
        }
    }
    
    // Simple BFS for reachability from start position
    int start_x = world_to_grid_x(start_pos[0]);
    int start_y = world_to_grid_y(start_pos[1]);
    
    // CRITICAL FIX: Ensure robot's current position is always free
    // The robot cannot be inside an obstacle at its current location
    if (is_valid_grid_coord(start_x, start_y)) {
        dynamic_grid_[start_x][start_y] = -1;      // Mark as free space
        reachability_grid_[start_x][start_y] = 0;  // Reset to unreachable (will become reachable in BFS)
    }
    
    reset_bfs_queue();
    if (is_valid_grid_coord(start_x, start_y)) {
        bfs_enqueue(start_x, start_y);
        reachability_grid_[start_x][start_y] = 1;  // Mark start as reachable
    }
    
    // Fast reachability BFS - no distance tracking
    while (!bfs_empty()) {
        auto [x, y] = bfs_dequeue();
        if (x < 0) break;
        
        for (const auto& [dx, dy] : DIRECTIONS) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (is_valid_grid_coord(nx, ny) && 
                reachability_grid_[nx][ny] == 0 && 
                dynamic_grid_[nx][ny] != -2) {
                
                reachability_grid_[nx][ny] = 1;  // Mark as reachable
                bfs_enqueue(nx, ny);
            }
        }
    }
}

void IncrementalWavefrontPlanner::update_performance_stats(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) const {
        
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats_.wavefront_time += duration.count() / 1000.0; // Convert to ms
    stats_.total_planning_time += duration.count() / 1000.0;
}

} // namespace namo