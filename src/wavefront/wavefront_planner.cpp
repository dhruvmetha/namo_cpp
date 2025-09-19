#include "wavefront/wavefront_planner.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace namo {

WavefrontPlanner::WavefrontPlanner(double resolution, NAMOEnvironment& env, 
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

    // std::cout << "robot_size: " << robot_size_[0] << ", " << robot_size_[1] << std::endl;

    // Add movable objects to initial dynamic grid
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        
        // Get current object state
        const ObjectState* obj_state = env.get_object_state(obj.name);
        if (!obj_state) continue;
        
        // Create inflated object for robot size
        ObjectInfo inflated_obj = obj;

        inflated_obj.size[0] += robot_size_[0] + 0.005;
        inflated_obj.size[1] += robot_size_[1] + 0.005;

        
        // Add object footprint to dynamic grid
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        for (size_t j = 0; j < footprint.num_cells; j++) {
            int x = footprint.cells[j].first;
            int y = footprint.cells[j].second;
            if (is_valid_grid_coord(x, y)) {
                dynamic_grid_[x][y] = -2;  // Obstacle
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
                inflated_obj.size[1] += robot_size_[1] + 0.005;
                inflated_obj.size[0] += robot_size_[0] + 0.005;
                
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
    // std::cout << "Static grid initialization took " << duration.count() << " ms" << std::endl;
}

bool WavefrontPlanner::update_wavefront(NAMOEnvironment& env, 
                                                  const std::vector<double>& start_pos) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Rebuild and recompute wavefront from scratch
    recompute_wavefront(env, start_pos);
    
    // Update basic statistics
    stats_.wavefront_updates++;
    
    update_performance_stats(start_time, std::chrono::high_resolution_clock::now());
    return true; // Always return true since we always rebuild
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

bool WavefrontPlanner::is_point_in_goal_region(double px, double py, 
                                                         const std::array<double, 2>& goal_pos,
                                                         double goal_size) const {
    return std::abs(px - goal_pos[0]) <= goal_size && std::abs(py - goal_pos[1]) <= goal_size;
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
    
    // 2. Reset all reachability values: -2=obstacle, 0=unreachable, 1=reachable
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            if (dynamic_grid_[x][y] == -2) {
                reachability_grid_[x][y] = -2;  // Obstacle
            } else {
                reachability_grid_[x][y] = 0;   // Unreachable (until proven otherwise)
            }
        }
    }
    
    // 3. Simple BFS for reachability from start position
    int start_x = world_to_grid_x(start_pos[0]);
    int start_y = world_to_grid_y(start_pos[1]);
    
    // ADAPTIVE CLEARING: Check if robot is trapped and clear area accordingly
    bool is_trapped = true;
    
    // Check if robot has any free neighbors (not trapped)
    if (is_valid_grid_coord(start_x, start_y)) {
        for (const auto& [dx, dy] : DIRECTIONS) {
            int nx = start_x + dx;
            int ny = start_y + dy;
            if (is_valid_grid_coord(nx, ny) && dynamic_grid_[nx][ny] != -2) {
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
                dynamic_grid_[nx][ny] = -1;           // Mark as free space
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
                reachability_grid_[nx][ny] != -2 &&
                dynamic_grid_[nx][ny] != -2 &&    // Not an obstacle
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
    
    // Add all current movable objects directly (no incremental tracking)
    const auto& movable_objects = env.get_movable_objects();
    for (size_t i = 0; i < env.get_num_movable(); i++) {
        const auto& obj = movable_objects[i];
        const ObjectState* obj_state = env.get_object_state(obj.name);
        
        if (obj_state) {
            // Create inflated object for robot size
            ObjectInfo inflated_obj = obj;
            inflated_obj.size[0] += robot_size_[0] + 0.005;
            inflated_obj.size[1] += robot_size_[1] + 0.005;

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
            dynamic_grid_[x][y] = -2;  // Mark as obstacle
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

} // namespace namo