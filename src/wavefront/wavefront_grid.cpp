#include "wavefront/wavefront_grid.hpp"
#include "environment/namo_environment.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <queue>

namespace namo {

WavefrontGrid::WavefrontGrid(NAMOEnvironment& env,
                             const std::vector<double>& robot_size)
    : resolution_(kResolution), robot_size_(robot_size) {
    
    // Get environment bounds
    bounds_ = env.get_environment_bounds();
    grid_width_ = static_cast<int>((bounds_[1] - bounds_[0]) / resolution_);
    grid_height_ = static_cast<int>((bounds_[3] - bounds_[2]) / resolution_);
    
    // Allocate grids
    uninflated_grid_.resize(grid_width_, std::vector<int>(grid_height_, -1));
    static_grid_.resize(grid_width_, std::vector<int>(grid_height_, -1));
    dynamic_grid_.resize(grid_width_, std::vector<int>(grid_height_, -1));
    
    // Build initial occupancy grids (uninflated + inflated)
    rebuild_grids(env);
    
    // Initialize region caching
    regions_valid_ = false;
    
    std::cout << "Initialized wavefront grid:" << std::endl;
    std::cout << "  Grid size: " << grid_width_ << "x" << grid_height_ << std::endl;
    std::cout << "  Resolution: " << resolution_ << "m" << std::endl;
    std::cout << "  Bounds: [" << bounds_[0] << ", " << bounds_[1] << "] x ["
              << bounds_[2] << ", " << bounds_[3] << "]" << std::endl;
    std::cout << "  Robot size: [" << robot_size_[0] << ", " << robot_size_[1] << "]" << std::endl;
}

void WavefrontGrid::rebuild_grids(NAMOEnvironment& env) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const auto& static_objects = env.get_static_objects();
    const auto& movable_objects = env.get_movable_objects();
    const size_t num_static = env.get_num_static();
    const size_t num_movable = env.get_num_movable();
    const double inflate_x = robot_size_[0] + 0.005;
    const double inflate_y = robot_size_[1] + 0.005;

    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);

            bool occupied_uninflated = false;
            bool occupied_inflated = false;

            // Check static objects
            for (size_t i = 0; i < num_static; i++) {
                const auto& obj = static_objects[i];

                ObjectState static_state;
                static_state.position = obj.position;
                static_state.quaternion = obj.quaternion;

                if (!occupied_uninflated &&
                    is_point_in_rotated_rectangle(world_x, world_y, static_state, obj)) {
                    occupied_uninflated = true;
                }

                if (!occupied_inflated) {
                    ObjectInfo inflated_obj = obj;
                    inflated_obj.size[0] += inflate_x;
                    inflated_obj.size[1] += inflate_y;

                    if (is_point_in_rotated_rectangle(world_x, world_y, static_state, inflated_obj)) {
                        occupied_inflated = true;
                    }
                }

                if (occupied_uninflated && occupied_inflated) {
                    break;
                }
            }

            // Check movable objects (using their current state)
            if (!occupied_uninflated || !occupied_inflated) {
                for (size_t i = 0; i < num_movable; i++) {
                    const auto& obj = movable_objects[i];
                    const ObjectState* obj_state = env.get_object_state(obj.name);
                    if (!obj_state) {
                        continue;
                    }

                    if (!occupied_uninflated &&
                        is_point_in_rotated_rectangle(world_x, world_y, *obj_state, obj)) {
                        occupied_uninflated = true;
                    }

                    if (!occupied_inflated) {
                        ObjectInfo inflated_obj = obj;
                        inflated_obj.size[0] += inflate_x;
                        inflated_obj.size[1] += inflate_y;

                        if (is_point_in_rotated_rectangle(world_x, world_y, *obj_state, inflated_obj)) {
                            occupied_inflated = true;
                        }
                    }

                    if (occupied_uninflated && occupied_inflated) {
                        break;
                    }
                }
            }

            uninflated_grid_[x][y] = occupied_uninflated ? -2 : -1;
            static_grid_[x][y] = occupied_inflated ? -2 : -1;
        }
    }

    dynamic_grid_ = static_grid_;
    regions_valid_ = false;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Grid rebuild took " << duration.count() << " ms" << std::endl;
}

void WavefrontGrid::update_dynamic_grid(NAMOEnvironment& env) {
    rebuild_grids(env);
}

GridFootprint WavefrontGrid::calculate_rotated_footprint(const ObjectInfo& obj, 
                                                        const ObjectState& state) {
    GridFootprint footprint;
    footprint.clear();
    
    // Safety checks
    if (obj.size[0] <= 0 || obj.size[1] <= 0) {
        std::cout << "Warning: Invalid object size [" << obj.size[0] << ", " << obj.size[1] << "]" << std::endl;
        return footprint;
    }
    
    // Check quaternion validity
    double quat_norm = std::sqrt(state.quaternion[0]*state.quaternion[0] + 
                                state.quaternion[1]*state.quaternion[1] + 
                                state.quaternion[2]*state.quaternion[2] + 
                                state.quaternion[3]*state.quaternion[3]);
    if (std::abs(quat_norm - 1.0) > 0.01) {
        std::cout << "Warning: Invalid quaternion norm " << quat_norm << ", using identity" << std::endl;
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

bool WavefrontGrid::is_point_in_rotated_rectangle(double px, double py, 
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

bool WavefrontGrid::is_cell_free(int x, int y) const {
    if (!is_valid_grid_coord(x, y)) {
        return false;
    }
    return dynamic_grid_[x][y] != -2;  // Free if not obstacle
}

bool WavefrontGrid::is_position_free(double world_x, double world_y) const {
    int grid_x = world_to_grid_x(world_x);
    int grid_y = world_to_grid_y(world_y);
    return is_cell_free(grid_x, grid_y);
}

void WavefrontGrid::clear_region(double world_x, double world_y, int clear_radius) {
    int center_x = world_to_grid_x(world_x);
    int center_y = world_to_grid_y(world_y);
    
    // Clear specified radius around center position
    for (int dx = -clear_radius; dx <= clear_radius; dx++) {
        for (int dy = -clear_radius; dy <= clear_radius; dy++) {
            int nx = center_x + dx;
            int ny = center_y + dy;
            
            if (is_valid_grid_coord(nx, ny)) {
                dynamic_grid_[nx][ny] = -1;  // Mark as free space
            }
        }
    }
    
    // Invalidate cached regions since grid has changed
    regions_valid_ = false;
}

void WavefrontGrid::save_grid(const std::string& filename, bool use_static_grid) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // use_static_grid=true → static_grid_ (inflated), false → dynamic_grid_ (inflated + movable)
    const auto& grid = use_static_grid ? static_grid_ : dynamic_grid_;
    
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);
            file << world_x << " " << world_y << " " << grid[x][y] << "\n";
        }
    }
    
    file.close();
    std::cout << "Grid saved to: " << filename << std::endl;
}

void WavefrontGrid::save_uninflated_grid(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            double world_x = grid_to_world_x(x);
            double world_y = grid_to_world_y(y);
            file << world_x << " " << world_y << " " << uninflated_grid_[x][y] << "\n";
        }
    }
    
    file.close();
    std::cout << "Uninflated grid saved to: " << filename << std::endl;
}

// ========================
// Connected Components Analysis
// ========================

std::pair<int, int> WavefrontGrid::select_random_point(
    const std::unordered_set<std::pair<int, int>, CoordinateHash>& points) const {
    
    if (points.empty()) {
        throw std::runtime_error("Cannot select random point from empty set");
    }
    
    // Create random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // Select random index
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
    size_t random_index = dist(gen);
    
    // Advance iterator to random position
    auto it = points.begin();
    std::advance(it, random_index);
    
    return *it;
}

std::unordered_set<std::pair<int, int>, CoordinateHash> WavefrontGrid::explore_connected_region(
    const std::pair<int, int>& start_point,
    std::unordered_set<std::pair<int, int>, CoordinateHash>& unvisited_points) const {
    
    std::unordered_set<std::pair<int, int>, CoordinateHash> region;
    std::queue<std::pair<int, int>> bfs_queue;
    
    // Start BFS from the given point
    bfs_queue.push(start_point);
    region.insert(start_point);
    unvisited_points.erase(start_point);  // Remove from unvisited
    
    // 8-connected neighbors (up, down, left, right, diagonals)
    const std::array<std::pair<int, int>, 8> directions = {{
        {0, 1}, {0, -1}, {1, 0}, {-1, 0},        // Cardinal directions
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}       // Diagonal directions
    }};
    
    while (!bfs_queue.empty()) {
        auto [x, y] = bfs_queue.front();
        bfs_queue.pop();
        
        // Explore all 8-connected neighbors
        for (const auto& [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;
            std::pair<int, int> neighbor = {nx, ny};
            
            // Check if neighbor is valid and unvisited
            if (is_valid_grid_coord(nx, ny) && 
                is_cell_free(nx, ny) &&
                unvisited_points.find(neighbor) != unvisited_points.end()) {
                
                // Add to region and queue for exploration
                region.insert(neighbor);
                unvisited_points.erase(neighbor);
                bfs_queue.push(neighbor);
            }
        }
    }
    
    return region;
}

std::unordered_map<int, std::unordered_set<std::pair<int, int>, CoordinateHash>> 
WavefrontGrid::find_connected_components(const std::array<double, 2>& robot_pos, 
                                        const std::vector<std::array<double, 2>>& goal_cells) const {
    
    // Clear previous results
    cached_regions_.clear();
    cached_region_labels_.clear();
    region_grid_.assign(grid_width_, std::vector<int>(grid_height_, 0));
    
    // Initialize set of all free (unvisited) points
    std::unordered_set<std::pair<int, int>, CoordinateHash> unvisited_points;
    
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            if (is_cell_free(x, y)) {
                unvisited_points.insert({x, y});
            }
        }
    }
    
    // === ROBOT REGION (ID = 1) ===
    int robot_grid_x = world_to_grid_x(robot_pos[0]);
    int robot_grid_y = world_to_grid_y(robot_pos[1]);
    std::pair<int, int> robot_grid_pos = {robot_grid_x, robot_grid_y};
    
    // If robot cell is marked as obstacle due to discretization, clear the
    // robot cell and its immediate 8-neighborhood (inflated grid correction).
    if (!is_cell_free(robot_grid_x, robot_grid_y)) {
        if (!is_valid_grid_coord(robot_grid_x, robot_grid_y)) {
            throw std::runtime_error("Robot position is outside valid grid bounds");
        }
        const std::array<std::pair<int, int>, 9> robot_clear_offsets = {{
            {0, 0}, {0, 1}, {0, -1}, {1, 0}, {-1, 0},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        }};
        for (const auto& [dx, dy] : robot_clear_offsets) {
            int nx = robot_grid_x + dx;
            int ny = robot_grid_y + dy;
            if (!is_valid_grid_coord(nx, ny)) {
                continue;
            }
            if (dynamic_grid_[nx][ny] == -2) {
                dynamic_grid_[nx][ny] = -1;
            }
            unvisited_points.insert({nx, ny});
        }
    }
    
    // Add robot cell to unvisited points if not already there
    if (unvisited_points.find(robot_grid_pos) == unvisited_points.end()) {
        unvisited_points.insert(robot_grid_pos);
    }
    
    // Prepare goal cells in grid coordinates
    std::vector<std::pair<int, int>> valid_goal_cells;
    valid_goal_cells.reserve(goal_cells.size());
    std::unordered_set<std::pair<int, int>, CoordinateHash> free_goal_cells;
    std::unordered_set<std::pair<int, int>, CoordinateHash> blocked_goal_cells;

    for (const auto& goal_world : goal_cells) {
        int goal_x = world_to_grid_x(goal_world[0]);
        int goal_y = world_to_grid_y(goal_world[1]);
        std::pair<int, int> goal_cell = {goal_x, goal_y};

        if (!is_valid_grid_coord(goal_x, goal_y)) {
            blocked_goal_cells.insert(goal_cell);
            continue;
        }

        valid_goal_cells.push_back(goal_cell);
        if (is_cell_free(goal_x, goal_y)) {
            free_goal_cells.insert(goal_cell);
        } else {
            blocked_goal_cells.insert(goal_cell);
        }
    }

    bool robot_reaches_goal = false;
    
    // Explore robot region
    std::unordered_set<std::pair<int, int>, CoordinateHash> robot_region = 
        explore_connected_region(robot_grid_pos, unvisited_points);
    
    // Check if robot region contains any goal cells
    for (const auto& goal_cell : valid_goal_cells) {
        if (robot_region.find(goal_cell) != robot_region.end()) {
            robot_reaches_goal = true;
            break;
        }
    }
    
    if (valid_goal_cells.empty()) {
        cached_regions_[1] = robot_region;
        cached_region_labels_[1] = "robot";

        for (const auto& [x, y] : robot_region) {
            region_grid_[x][y] = 1;
        }

        if (goal_cells.empty()) {
            std::cout << "No goal cells provided; treating robot region as standalone" << std::endl;
        } else {
            std::cout << "Goal cells provided but all are out of bounds (" << blocked_goal_cells.size()
                      << " entries); treating robot region as standalone" << std::endl;
        }

    } else if (robot_reaches_goal) {
        cached_regions_[1] = robot_region;
        cached_region_labels_[1] = "robot_goal";

        for (const auto& [x, y] : robot_region) {
            region_grid_[x][y] = 1;
        }

        std::cout << "Robot and goal are in the same connected region" << std::endl;
    } else {
        cached_regions_[1] = robot_region;
        cached_region_labels_[1] = "robot";

        for (const auto& [x, y] : robot_region) {
            region_grid_[x][y] = 1;
        }

        if (!free_goal_cells.empty()) {
            // === GOAL REGION (ID = 2) ===
            std::unordered_set<std::pair<int, int>, CoordinateHash> goal_region;
            std::queue<std::pair<int, int>> bfs_queue;

            const std::array<std::pair<int, int>, 8> directions = {{
                {0, 1}, {0, -1}, {1, 0}, {-1, 0},
                {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
            }};

            for (const auto& goal_cell : free_goal_cells) {
                auto it = unvisited_points.find(goal_cell);
                if (it != unvisited_points.end()) {
                    goal_region.insert(goal_cell);
                    unvisited_points.erase(it);
                    bfs_queue.push(goal_cell);
                }
            }

            while (!bfs_queue.empty()) {
                auto [x, y] = bfs_queue.front();
                bfs_queue.pop();

                for (const auto& [dx, dy] : directions) {
                    int nx = x + dx;
                    int ny = y + dy;
                    std::pair<int, int> neighbor = {nx, ny};

                    if (is_valid_grid_coord(nx, ny) &&
                        is_cell_free(nx, ny) &&
                        unvisited_points.find(neighbor) != unvisited_points.end()) {

                        goal_region.insert(neighbor);
                        unvisited_points.erase(neighbor);
                        bfs_queue.push(neighbor);
                    }
                }
            }

            if (!goal_region.empty()) {
                cached_regions_[2] = goal_region;
                cached_region_labels_[2] = "goal";

                for (const auto& [x, y] : goal_region) {
                    region_grid_[x][y] = 2;
                }

                std::cout << "Goal region identified with " << goal_region.size()
                          << " free cells" << std::endl;
            } else {
                std::cout << "Goal cells exist but none are reachable as free space" << std::endl;
            }
        } else {
            std::cout << "All goal cells are blocked (" << blocked_goal_cells.size()
                      << ") or otherwise unavailable; no goal region created" << std::endl;
        }
    }
    
    // === OTHER REGIONS (ID = 3+) ===
    auto touches_grid_border = [this](const std::unordered_set<std::pair<int, int>, CoordinateHash>& region) {
        for (const auto& [x, y] : region) {
            if (x == 0 || y == 0 || x == grid_width_ - 1 || y == grid_height_ - 1) {
                return true;
            }
        }
        return false;
    };

    int region_id = 3;

    while (!unvisited_points.empty()) {
        std::pair<int, int> seed_point = select_random_point(unvisited_points);

        std::unordered_set<std::pair<int, int>, CoordinateHash> region =
            explore_connected_region(seed_point, unvisited_points);

        if (region.empty()) {
            continue;
        }

        if (touches_grid_border(region)) {
            // Treat border-touching regions as outside the playable area; skip labeling them.
            continue;
        }

        cached_regions_[region_id] = region;
        cached_region_labels_[region_id] = "region_" + std::to_string(region_id);

        for (const auto& [x, y] : region) {
            region_grid_[x][y] = region_id;
        }

        region_id++;
    }

    regions_valid_ = true;
    
    std::cout << "Found " << cached_regions_.size() << " connected components:" << std::endl;
    for (const auto& [id, region] : cached_regions_) {
        const std::string& label = cached_region_labels_[id];
        std::cout << "  Region " << id << " (" << label << "): " << region.size() << " cells" << std::endl;
    }
    
    return cached_regions_;
}

std::unordered_map<int, std::unordered_set<std::pair<int, int>, CoordinateHash>> 
WavefrontGrid::find_connected_components() const {
    
    // Check if cached result is still valid
    if (regions_valid_) {
        return cached_regions_;
    }
    
    // Clear previous results
    cached_regions_.clear();
    cached_region_labels_.clear();
    region_grid_.assign(grid_width_, std::vector<int>(grid_height_, 0));
    
    // Initialize set of all free (unvisited) points
    std::unordered_set<std::pair<int, int>, CoordinateHash> unvisited_points;
    
    for (int x = 0; x < grid_width_; x++) {
        for (int y = 0; y < grid_height_; y++) {
            if (is_cell_free(x, y)) {
                unvisited_points.insert({x, y});
            }
        }
    }
    
    int region_id = 1;  // Start region IDs at 1
    
    // Continue until all free points are assigned to regions
    while (!unvisited_points.empty()) {
        // Select random unvisited point as seed for new region
        std::pair<int, int> seed_point = select_random_point(unvisited_points);
        
        // Explore connected region using BFS
        std::unordered_set<std::pair<int, int>, CoordinateHash> region = 
            explore_connected_region(seed_point, unvisited_points);
        
        // Store the region
        cached_regions_[region_id] = region;
        cached_region_labels_[region_id] = "region_" + std::to_string(region_id);
        
        // Update region grid for fast lookup
        for (const auto& [x, y] : region) {
            region_grid_[x][y] = region_id;
        }
        
        region_id++;
    }
    
    regions_valid_ = true;
    
    std::cout << "Found " << cached_regions_.size() << " connected components:" << std::endl;
    for (const auto& [id, region] : cached_regions_) {
        std::cout << "  Region " << id << ": " << region.size() << " cells" << std::endl;
    }
    
    return cached_regions_;
}

int WavefrontGrid::get_cell_region_id(int x, int y) const {
    if (!is_valid_grid_coord(x, y)) {
        return 0;  // Invalid coordinates
    }
    
    if (!is_cell_free(x, y)) {
        return 0;  // Obstacle cell
    }
    
    // Ensure regions are computed
    if (!regions_valid_) {
        find_connected_components();
    }
    
    return region_grid_[x][y];
}

std::unordered_map<int, std::string> WavefrontGrid::get_region_labels() const {
    // Ensure regions are computed
    if (!regions_valid_) {
        find_connected_components();
    }
    
    return cached_region_labels_;
}

std::unordered_map<std::string, std::unordered_set<std::string>> 
WavefrontGrid::build_region_connectivity_graph(NAMOEnvironment& env) {
    
    // First, ensure we have baseline regions with all objects present
    find_connected_components();
    auto region_labels = get_region_labels();
    
    std::cout << "Building region connectivity graph with " << cached_regions_.size() << " regions" << std::endl;
    
    // Initialize adjacency list
    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency_list;
    adjacency_object_map_.clear();
    for (const auto& [region_id, label] : region_labels) {
        adjacency_list[label] = std::unordered_set<std::string>();
        adjacency_object_map_[label] = std::unordered_map<std::string, std::unordered_set<std::string>>();
    }
    
    // Get movable objects
    const auto& movable_objects = env.get_movable_objects();
    
    // Process each movable object
    for (size_t obj_idx = 0; obj_idx < env.get_num_movable(); obj_idx++) {
        const auto& obj = movable_objects[obj_idx];
        const ObjectState* obj_state = env.get_object_state(obj.name);
        
        if (!obj_state) {
            std::cout << "Warning: No state found for object " << obj.name << std::endl;
            continue;
        }
        
        std::cout << "Processing object " << obj.name << " (" << (obj_idx + 1) << "/" 
                  << env.get_num_movable() << ")" << std::endl;
        
        // === STEP 1: Calculate object footprint ===
        ObjectInfo inflated_obj = obj;
        inflated_obj.size[0] += robot_size_[0] + 0.005;
        inflated_obj.size[1] += robot_size_[1] + 0.005;
        
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        
        if (footprint.num_cells == 0) {
            std::cout << "  Object has no footprint - skipping" << std::endl;
            continue;
        }
        
        // === STEP 2: Temporarily remove object from grid ===
        std::vector<std::pair<int, int>> removed_cells;
        
        for (size_t i = 0; i < footprint.num_cells; i++) {
            int x = footprint.cells[i].first;
            int y = footprint.cells[i].second;
            if (is_valid_grid_coord(x, y) && dynamic_grid_[x][y] == -2) {
                dynamic_grid_[x][y] = -1;  // Mark as free
                removed_cells.push_back({x, y});
            }
        }
        
        if (removed_cells.empty()) {
            std::cout << "  No cells to remove - skipping" << std::endl;
            continue;
        }
        
        // === STEP 3: Run optimized BFS from one removed cell ===
        std::pair<int, int> seed_cell = removed_cells[0];  // Pick first removed cell
        std::unordered_set<int> connected_region_ids;
        
        // BFS with optimization: stop expanding at already-free cells
        std::queue<std::pair<int, int>> bfs_queue;
        std::unordered_set<std::pair<int, int>, CoordinateHash> visited;
        
        bfs_queue.push(seed_cell);
        visited.insert(seed_cell);
        
        // 8-connected neighbors
        const std::array<std::pair<int, int>, 8> directions = {{
            {0, 1}, {0, -1}, {1, 0}, {-1, 0},        // Cardinal directions
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}       // Diagonal directions
        }};
        
        while (!bfs_queue.empty()) {
            auto [x, y] = bfs_queue.front();
            bfs_queue.pop();
            
            // Explore all 8-connected neighbors
            for (const auto& [dx, dy] : directions) {
                int nx = x + dx;
                int ny = y + dy;
                std::pair<int, int> neighbor = {nx, ny};
                
                if (!is_valid_grid_coord(nx, ny) || 
                    visited.find(neighbor) != visited.end() ||
                    dynamic_grid_[nx][ny] == -2) {  // Skip obstacles
                    continue;
                }
                
                visited.insert(neighbor);
                
                // Check if this cell was originally free (belongs to existing region)
                bool was_originally_free = true;
                for (const auto& removed_cell : removed_cells) {
                    if (removed_cell.first == nx && removed_cell.second == ny) {
                        was_originally_free = false;
                        break;
                    }
                }
                
                if (was_originally_free) {
                    // This cell belongs to an existing region - record it but don't expand further
                    int region_id = region_grid_[nx][ny];
                    if (region_id > 0) {
                        connected_region_ids.insert(region_id);
                    }
                } else {
                    // This cell was also blocked by the object - continue exploring
                    bfs_queue.push(neighbor);
                }
            }
        }
        
        // === STEP 4: Create edges if multiple regions connected ===
        if (connected_region_ids.size() >= 2) {
            std::cout << "  Object connects " << connected_region_ids.size() << " regions: ";
            
            // Convert region IDs to labels and create complete subgraph
            std::vector<std::string> connected_labels;
            for (int region_id : connected_region_ids) {
                auto label_it = region_labels.find(region_id);
                if (label_it != region_labels.end()) {
                    connected_labels.push_back(label_it->second);
                    std::cout << label_it->second << " ";
                }
            }
            std::cout << std::endl;
            
            // Add edges between all pairs of connected regions
            for (size_t i = 0; i < connected_labels.size(); i++) {
                for (size_t j = i + 1; j < connected_labels.size(); j++) {
                    const std::string& label1 = connected_labels[i];
                    const std::string& label2 = connected_labels[j];

                    adjacency_list[label1].insert(label2);
                    adjacency_list[label2].insert(label1);

                    auto& edge_set_1 = adjacency_object_map_[label1][label2];
                    edge_set_1.insert(obj.name);
                    auto& edge_set_2 = adjacency_object_map_[label2][label1];
                    edge_set_2.insert(obj.name);
                }
            }
        } else {
            std::cout << "  Object connects " << connected_region_ids.size() 
                      << " regions - no edges added" << std::endl;
        }
        
        // === STEP 5: Restore object to grid ===
        for (const auto& cell : removed_cells) {
            dynamic_grid_[cell.first][cell.second] = -2;  // Mark as obstacle again
        }
    }
    
    // Print summary
    std::cout << "\nRegion Connectivity Graph Summary:" << std::endl;
    for (const auto& [region_label, neighbors] : adjacency_list) {
        std::cout << "  " << region_label << " -> {";
        bool first = true;
        for (const auto& neighbor : neighbors) {
            if (!first) std::cout << ", ";
            std::cout << neighbor;
            first = false;
        }
        std::cout << "}" << std::endl;
    }
    
    return adjacency_list;
}

} // namespace namo
