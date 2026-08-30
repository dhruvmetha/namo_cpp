#include "wavefront/wavefront_grid.hpp"
#include "environment/namo_environment.hpp"
#include "wavefront/goal_tolerance_utils.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <queue>
#include <random>
#include <array>
#include <cstdlib>

namespace namo {

namespace {
// WavefrontGrid diagnostics run once per region-opening search NODE. They used to spam stdout
// (and rl_env.cpp installed a CoutSilencer hack to suppress them in the snapshot path). Route
// them through a compile-time-gated sink instead: off by default, flip kWavefrontGridVerbose to
// re-enable. std::cerr warnings (invalid geometry) are intentionally left ungated.
constexpr bool kWavefrontGridVerbose = false;
inline std::ostream& wg_dbg() {
    static std::ostream null_sink(nullptr);  // no streambuf -> discards without formatting work
    return kWavefrontGridVerbose ? std::cout : null_sink;
}
}  // namespace

WavefrontGrid::WavefrontGrid(NAMOEnvironment& env,
                             const std::vector<double>& robot_size,
                             double tier1_inflation_margin)
    : resolution_(kResolution),
      robot_size_(robot_size),
      tier1_inflation_margin_(tier1_inflation_margin) {
    
    // Get environment bounds
    bounds_ = env.get_environment_bounds();
    grid_width_ = static_cast<int>((bounds_[1] - bounds_[0]) / resolution_);
    grid_height_ = static_cast<int>((bounds_[3] - bounds_[2]) / resolution_);
    
    // Allocate grids
    uninflated_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    static_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    dynamic_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    occupancy_count_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    static_count_grid_.resize(grid_width_, std::vector<int>(grid_height_, 0));
    
    // Build initial occupancy grids (uninflated + inflated)
    rebuild_grids(env);
    
    // Initialize region caching
    regions_valid_ = false;
    
    wg_dbg() << "Initialized wavefront grid:" << std::endl;
    wg_dbg() << "  Grid size: " << grid_width_ << "x" << grid_height_ << std::endl;
    wg_dbg() << "  Resolution: " << resolution_ << "m" << std::endl;
    wg_dbg() << "  Bounds: [" << bounds_[0] << ", " << bounds_[1] << "] x ["
              << bounds_[2] << ", " << bounds_[3] << "]" << std::endl;
    wg_dbg() << "  Robot size: [" << robot_size_[0] << ", " << robot_size_[1] << "]" << std::endl;
}

void WavefrontGrid::rebuild_grids(NAMOEnvironment& env) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const auto& static_objects = env.get_static_objects();
    const auto& movable_objects = env.get_movable_objects();
    const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);

    for (int x = 0; x < grid_width_; ++x) {
        std::fill(uninflated_grid_[x].begin(), uninflated_grid_[x].end(), 0);
        std::fill(static_grid_[x].begin(), static_grid_[x].end(), 0);
        std::fill(occupancy_count_grid_[x].begin(), occupancy_count_grid_[x].end(), 0);
        std::fill(static_count_grid_[x].begin(), static_count_grid_[x].end(), 0);
    }

    // `is_static` comes from the caller's loop rather than ObjectInfo::is_static: the two loops below
    // are already authoritative about which list an object came from, and that needs no trust in a
    // field being populated.
    auto add_object = [&](const ObjectInfo& obj, const ObjectState& state, bool is_static) {
        {
            const GridFootprint footprint = calculate_rotated_footprint(obj, state);
            for (size_t i = 0; i < footprint.num_cells; ++i) {
                const auto [x, y] = footprint.cells[i];
                if (is_valid_grid_coord(x, y)) {
                    uninflated_grid_[x][y] = -1;
                }
            }
        }

        ObjectInfo inflated_obj = obj;
        inflated_obj.size[0] += inflate_r;
        inflated_obj.size[1] += inflate_r;
        {
            const GridFootprint footprint = calculate_rotated_footprint(inflated_obj, state);
            for (size_t i = 0; i < footprint.num_cells; ++i) {
                const auto [x, y] = footprint.cells[i];
                if (is_valid_grid_coord(x, y)) {
                    ++occupancy_count_grid_[x][y];
                    if (is_static) {
                        ++static_count_grid_[x][y];
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < env.get_num_static(); ++i) {
        const auto& obj = static_objects[i];
        ObjectState state;
        state.position = obj.position;
        state.quaternion = obj.quaternion;
        add_object(obj, state, true);
    }

    for (size_t i = 0; i < env.get_num_movable(); ++i) {
        const auto& obj = movable_objects[i];
        const ObjectState* state = env.get_object_state(obj.name);
        if (state) {
            add_object(obj, *state, false);
        }
    }

    for (int x = 0; x < grid_width_; ++x) {
        for (int y = 0; y < grid_height_; ++y) {
            static_grid_[x][y] = occupancy_count_grid_[x][y] > 0 ? -1 : 0;
        }
    }

    dynamic_grid_ = static_grid_;
    regions_valid_ = false;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    wg_dbg() << "Grid rebuild took " << duration.count() << " ms" << std::endl;
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
        wg_dbg() << "Warning: Invalid object size [" << obj.size[0] << ", " << obj.size[1] << "]" << std::endl;
        return footprint;
    }
    
    // Check quaternion validity
    double quat_norm = std::sqrt(state.quaternion[0]*state.quaternion[0] + 
                                state.quaternion[1]*state.quaternion[1] + 
                                state.quaternion[2]*state.quaternion[2] + 
                                state.quaternion[3]*state.quaternion[3]);
    if (std::abs(quat_norm - 1.0) > 0.01) {
        wg_dbg() << "Warning: Invalid quaternion norm " << quat_norm << ", using identity" << std::endl;
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
            const double world_x = grid_to_world_x(x) + 0.5 * resolution_;
            const double world_y = grid_to_world_y(y) + 0.5 * resolution_;
            
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
    return dynamic_grid_[x][y] != -1;  // Free if not obstacle
}

// ========================
// Connected Components Analysis
// ========================

std::pair<int, int> WavefrontGrid::select_random_point(
    const std::unordered_set<std::pair<int, int>, CoordinateHash>& points) const {
    
    if (points.empty()) {
        throw std::runtime_error("Cannot select seed point from empty set");
    }

    // Deterministic seed selection so region IDs are stable across runs.
    auto best_it = points.begin();
    for (auto it = std::next(points.begin()); it != points.end(); ++it) {
        if (it->first < best_it->first ||
            (it->first == best_it->first && it->second < best_it->second)) {
            best_it = it;
        }
    }
    return *best_it;
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
            if (dynamic_grid_[nx][ny] == -1) {
                dynamic_grid_[nx][ny] = 0;
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
            wg_dbg() << "No goal cells provided; treating robot region as standalone" << std::endl;
        } else {
            wg_dbg() << "Goal cells provided but all are out of bounds (" << blocked_goal_cells.size()
                      << " entries); treating robot region as standalone" << std::endl;
        }

    } else if (robot_reaches_goal) {
        cached_regions_[1] = robot_region;
        cached_region_labels_[1] = "robot_goal";

        for (const auto& [x, y] : robot_region) {
            region_grid_[x][y] = 1;
        }

        wg_dbg() << "Robot and goal are in the same connected region" << std::endl;
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

                wg_dbg() << "Goal region identified with " << goal_region.size()
                          << " free cells" << std::endl;
            } else {
                wg_dbg() << "Goal cells exist but none are reachable as free space" << std::endl;
            }
        } else {
            wg_dbg() << "All goal cells are blocked (" << blocked_goal_cells.size()
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
    
    wg_dbg() << "Found " << cached_regions_.size() << " connected components:" << std::endl;
    for (const auto& [id, region] : cached_regions_) {
        const std::string& label = cached_region_labels_[id];
        wg_dbg() << "  Region " << id << " (" << label << "): " << region.size() << " cells" << std::endl;
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
    
    wg_dbg() << "Found " << cached_regions_.size() << " connected components:" << std::endl;
    for (const auto& [id, region] : cached_regions_) {
        wg_dbg() << "  Region " << id << ": " << region.size() << " cells" << std::endl;
    }
    
    return cached_regions_;
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
    
    wg_dbg() << "Building region connectivity graph with " << cached_regions_.size() << " regions" << std::endl;
    
    // Initialize adjacency list
    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency_list;
    adjacency_object_map_.clear();
    multi_object_edges_.clear();
    for (const auto& [region_id, label] : region_labels) {
        adjacency_list[label] = std::unordered_set<std::string>();
        adjacency_object_map_[label] = std::unordered_map<std::string, std::unordered_set<std::string>>();
    }
    
    // Get movable objects
    const auto& movable_objects = env.get_movable_objects();

    // === MOVABLE-BLOB PASS, PHASE A: label connected clumps of movable-only cells ===
    //
    // The per-object loop below asks "would removing THIS ONE object join two regions". A doorway
    // held shut by two touching blocks answers no for each block alone, so it records nothing and the
    // regions read as unconnected. Measured on the real-table two-movable pool: adjacency came back
    // {'robot': {}, 'goal': {}} on 67 of 300 scenes, which then sampled zero goal poses and got
    // silently dropped by the labeller.
    //
    // This pass floods clumps of cells blocked only by movables and treats each clump as one plug.
    // It runs BEFORE the per-object loop because that loop temporarily mutates dynamic_grid_.
    //
    // ⛔ The passable test reads dynamic_grid_, NOT the occupancy counts. find_connected_components
    // force-clears cells around the robot without touching either count, so a count-based test would
    // call those free cells blocked and swallow them into a blob.
    const bool blob_pass_enabled = std::getenv("NAMO_DISABLE_MOVABLE_BLOB_EDGES") == nullptr;
    std::vector<std::vector<int>> blob_id(grid_width_, std::vector<int>(grid_height_, -1));
    std::vector<std::unordered_set<int>> blob_regions;
    std::vector<std::unordered_set<std::string>> blob_objects;

    // 8-connected, matching every other flood in this file. A 4-connected version would split clumps
    // that touch only at a corner, and then disagree with the per-object pass on the same geometry.
    const std::array<std::pair<int, int>, 8> blob_dirs = {{
        {0, 1}, {0, -1}, {1, 0}, {-1, 0},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    }};

    auto is_movable_only = [&](int x, int y) {
        return dynamic_grid_[x][y] == -1 && static_count_grid_[x][y] == 0;
    };

    if (blob_pass_enabled) {
        // x-major scan so blob ids are deterministic across runs and boxes.
        for (int sx = 0; sx < grid_width_; ++sx) {
            for (int sy = 0; sy < grid_height_; ++sy) {
                if (blob_id[sx][sy] != -1 || !is_movable_only(sx, sy)) {
                    continue;
                }
                const int b = static_cast<int>(blob_regions.size());
                blob_regions.emplace_back();
                blob_objects.emplace_back();
                blob_id[sx][sy] = b;

                std::queue<std::pair<int, int>> q;
                q.push({sx, sy});
                while (!q.empty()) {
                    auto [x, y] = q.front();
                    q.pop();
                    for (const auto& [dx, dy] : blob_dirs) {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (!is_valid_grid_coord(nx, ny)) {
                            continue;
                        }
                        if (dynamic_grid_[nx][ny] != -1) {
                            // Free cell: record which region it belongs to and stop. Same
                            // record-but-do-not-enter rule the per-object flood uses. region_grid_ is
                            // 0 both for obstacles and for free cells of border-touching regions,
                            // which are deliberately left unlabeled, so > 0 skips those on purpose.
                            const int rid = region_grid_[nx][ny];
                            if (rid > 0) {
                                blob_regions[b].insert(rid);
                            }
                            continue;
                        }
                        if (static_count_grid_[nx][ny] > 0) {
                            continue;   // static wall: the plug ends here
                        }
                        if (blob_id[nx][ny] == -1) {
                            blob_id[nx][ny] = b;
                            q.push({nx, ny});
                        }
                    }
                }
            }
        }
        wg_dbg() << "Movable-blob pass: " << blob_regions.size() << " blob(s)" << std::endl;
    }

    // Process each movable object
    for (size_t obj_idx = 0; obj_idx < env.get_num_movable(); obj_idx++) {
        const auto& obj = movable_objects[obj_idx];
        const ObjectState* obj_state = env.get_object_state(obj.name);
        
        if (!obj_state) {
            wg_dbg() << "Warning: No state found for object " << obj.name << std::endl;
            continue;
        }
        
        wg_dbg() << "Processing object " << obj.name << " (" << (obj_idx + 1) << "/" 
                  << env.get_num_movable() << ")" << std::endl;
        
        // === STEP 1: Calculate object footprint ===
        ObjectInfo inflated_obj = obj;
        const double inflate_r = compute_wavefront_inflation_radius_m(robot_size_, tier1_inflation_margin_);
        inflated_obj.size[0] += inflate_r;
        inflated_obj.size[1] += inflate_r;
        
        GridFootprint footprint = calculate_rotated_footprint(inflated_obj, *obj_state);
        
        if (footprint.num_cells == 0) {
            wg_dbg() << "  Object has no footprint - skipping" << std::endl;
            continue;
        }

        // Blob attribution rides on the footprint STEP 1 just computed, which is the same
        // rasterization rebuild_grids used to fill the counts, so every movable-only cell is claimed
        // by at least one object. Done here rather than in Phase A because Phase A works off the grid
        // and has no object identity.
        if (blob_pass_enabled) {
            for (size_t i = 0; i < footprint.num_cells; i++) {
                const int fx = footprint.cells[i].first;
                const int fy = footprint.cells[i].second;
                if (is_valid_grid_coord(fx, fy) && blob_id[fx][fy] >= 0) {
                    blob_objects[blob_id[fx][fy]].insert(obj.name);
                }
            }
        }
        
        // === STEP 2: Temporarily remove object from grid ===
        std::vector<std::pair<int, int>> removed_cells;
        
        for (size_t i = 0; i < footprint.num_cells; i++) {
            int x = footprint.cells[i].first;
            int y = footprint.cells[i].second;
            if (is_valid_grid_coord(x, y) &&
                dynamic_grid_[x][y] == -1 &&
                occupancy_count_grid_[x][y] == 1) {
                dynamic_grid_[x][y] = 0;  // Mark as free
                removed_cells.push_back({x, y});
            }
        }
        
        if (removed_cells.empty()) {
            wg_dbg() << "  No cells to remove - skipping" << std::endl;
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
                    dynamic_grid_[nx][ny] == -1) {  // Skip obstacles
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
            wg_dbg() << "  Object connects " << connected_region_ids.size() << " regions: ";
            
            // Convert region IDs to labels and create complete subgraph
            std::vector<std::string> connected_labels;
            for (int region_id : connected_region_ids) {
                auto label_it = region_labels.find(region_id);
                if (label_it != region_labels.end()) {
                    connected_labels.push_back(label_it->second);
                    wg_dbg() << label_it->second << " ";
                }
            }
            wg_dbg() << std::endl;
            
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
            wg_dbg() << "  Object connects " << connected_region_ids.size() 
                      << " regions - no edges added" << std::endl;
        }
        
        // === STEP 5: Restore object to grid ===
        for (const auto& cell : removed_cells) {
            dynamic_grid_[cell.first][cell.second] = -1;  // Mark as obstacle again
        }
    }

    // === MOVABLE-BLOB PASS, PHASE B: write edges for clumps of two or more objects ===
    //
    // Only multi-object blobs write here. A single-object blob is already the per-object loop's job,
    // and letting this pass touch those would change single-movable scenes, which is exactly the
    // regression we need to be able to rule out. That restriction is also what lets us leave the
    // per-object loop's split-footprint bug alone, documented in
    // docs/known_limitations_region_graph.md #1.
    //
    // The per-object result as it stands right now, before this pass adds anything. A blob member
    // appearing here for a pair means that object ALONE opens that boundary, so the blob is not a
    // required group and must not be recorded as one. Without this check a block that merely touches
    // the opener's inflated footprint, off to one side and needed by nobody, produced exactly the
    // same output as a door neither block opens alone, which defeats the point of the marker.
    const auto per_object_edges = adjacency_object_map_;
    if (blob_pass_enabled) {
        for (size_t b = 0; b < blob_regions.size(); ++b) {
            if (blob_objects[b].size() < 2 || blob_regions[b].size() < 2) {
                continue;
            }
            std::vector<std::string> connected_labels;
            for (int region_id : blob_regions[b]) {
                auto label_it = region_labels.find(region_id);
                if (label_it != region_labels.end()) {
                    connected_labels.push_back(label_it->second);
                }
            }
            if (connected_labels.size() < 2) {
                continue;
            }
            // The whole clump is the plug, so every object in it is a candidate on every pair.
            std::vector<std::string> objs(blob_objects[b].begin(), blob_objects[b].end());
            std::sort(objs.begin(), objs.end());
            for (size_t i = 0; i < connected_labels.size(); i++) {
                for (size_t j = i + 1; j < connected_labels.size(); j++) {
                    const std::string& label1 = connected_labels[i];
                    const std::string& label2 = connected_labels[j];

                    // Did the per-object pass already open this SAME region pair, with any object at
                    // all? Checking only members of this blob is not enough. Two regions can share
                    // more than one doorway: object C opens one on its own while blob {A,B} plugs
                    // another. Suppressing on blob members only would mark that pair as needing a
                    // group when C alone opens it, and the marker is per pair, so it would be false.
                    // Looking up one direction is sound because the per-object writer always records
                    // both.
                    bool opened_by_one_object = false;
                    auto pe = per_object_edges.find(label1);
                    if (pe != per_object_edges.end()) {
                        auto pair_it = pe->second.find(label2);
                        opened_by_one_object = (pair_it != pe->second.end() && !pair_it->second.empty());
                    }

                    if (opened_by_one_object) {
                        // Leave the pair entirely alone. Naming the blob's other members here looked
                        // defensible (a wider candidate set, verified downstream anyway) and is not:
                        // a blob spans every doorway its objects touch, so on a chained scene it puts
                        // the far block's name on a door it is nowhere near. Measured on the 300-scene
                        // pool, writing names here changed 109 more scenes than leaving them out, all
                        // of it that kind of smearing. The per-object pass already listed the right
                        // candidates for a boundary one object opens.
                        continue;
                    }

                    adjacency_list[label1].insert(label2);
                    adjacency_list[label2].insert(label1);
                    for (const auto& name : objs) {
                        adjacency_object_map_[label1][label2].insert(name);
                        adjacency_object_map_[label2][label1].insert(name);
                    }
                    multi_object_edges_[label1].insert(label2);
                    multi_object_edges_[label2].insert(label1);
                }
            }
            wg_dbg() << "  Blob " << b << " (" << objs.size() << " objects) connects "
                      << connected_labels.size() << " regions" << std::endl;
        }
    }

    // Print summary
    wg_dbg() << "\nRegion Connectivity Graph Summary:" << std::endl;
    for (const auto& [region_label, neighbors] : adjacency_list) {
        wg_dbg() << "  " << region_label << " -> {";
        bool first = true;
        for (const auto& neighbor : neighbors) {
            if (!first) wg_dbg() << ", ";
            wg_dbg() << neighbor;
            first = false;
        }
        wg_dbg() << "}" << std::endl;
    }
    
    return adjacency_list;
}

std::unordered_map<std::string, RegionGoalBundle> WavefrontGrid::sample_region_goals(int goals_per_region) const {
    static thread_local std::mt19937 rng(std::random_device{}());
    return sample_region_goals(goals_per_region, rng());
}

std::unordered_map<std::string, RegionGoalBundle> WavefrontGrid::sample_region_goals(
    int goals_per_region, uint32_t seed) const {
    std::unordered_map<std::string, RegionGoalBundle> result;
    if (goals_per_region <= 0) {
        return result;
    }

    if (!regions_valid_) {
        find_connected_components();
    }

    if (cached_regions_.empty()) {
        return result;
    }

    std::string robot_label = "robot";
    for (const auto& entry : cached_region_labels_) {
        if (entry.second.find("robot") != std::string::npos) {
            robot_label = entry.second;
            break;
        }
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency;
    for (const auto& [from_label, neighbor_map] : adjacency_object_map_) {
        auto& neighbors = adjacency[from_label];
        for (const auto& [to_label, _objects] : neighbor_map) {
            neighbors.insert(to_label);
            adjacency[to_label].insert(from_label);
        }
    }

    std::unordered_set<std::string> reachable_labels;
    if (adjacency.find(robot_label) != adjacency.end()) {
        std::queue<std::string> bfs_queue;
        reachable_labels.insert(robot_label);
        bfs_queue.push(robot_label);
        while (!bfs_queue.empty()) {
            const std::string current = bfs_queue.front();
            bfs_queue.pop();
            auto it = adjacency.find(current);
            if (it == adjacency.end()) {
                continue;
            }
            for (const auto& neighbor : it->second) {
                if (reachable_labels.insert(neighbor).second) {
                    bfs_queue.push(neighbor);
                }
            }
        }
    }

    // Fallback to all labels when adjacency is unavailable.
    if (reachable_labels.empty()) {
        for (const auto& [region_id, label] : cached_region_labels_) {
            (void)region_id;
            reachable_labels.insert(label);
        }
    }

    std::vector<std::string> sorted_labels(reachable_labels.begin(), reachable_labels.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    std::mt19937 rng(seed);

    std::unordered_map<std::string, int> label_to_region_id;
    for (const auto& [region_id, label] : cached_region_labels_) {
        label_to_region_id[label] = region_id;
    }

    for (const auto& region_label : sorted_labels) {
        auto id_it = label_to_region_id.find(region_label);
        if (id_it == label_to_region_id.end()) {
            continue;
        }
        const int region_id = id_it->second;
        auto region_it = cached_regions_.find(region_id);
        if (region_it == cached_regions_.end()) {
            continue;
        }
        std::string key = region_label;
        if (region_label.find("robot") != std::string::npos) {
            key = "robot_region";
        }

        const auto& cell_set = region_it->second;
        std::vector<std::pair<int, int>> cells(cell_set.begin(), cell_set.end());
        std::sort(cells.begin(), cells.end());
        if (cells.empty()) {
            result.emplace(key, RegionGoalBundle{});
            continue;
        }

        std::shuffle(cells.begin(), cells.end(), rng);
        const size_t sample_count = std::min(static_cast<size_t>(goals_per_region), cells.size());

        RegionGoalBundle bundle;
        bundle.goals.reserve(sample_count);

        for (size_t idx = 0; idx < sample_count; ++idx) {
            const auto& cell = cells[idx];
            const double wx = grid_to_world_x(cell.first) + 0.5 * resolution_;
            const double wy = grid_to_world_y(cell.second) + 0.5 * resolution_;
            bundle.goals.push_back({wx, wy, 0.0});
        }

        if (region_label.find("robot") == std::string::npos) {
            using NeighborMap = std::unordered_map<std::string, std::unordered_set<std::string>>;
            const NeighborMap* robot_neighbors = [&]() -> const NeighborMap* {
                auto primary = adjacency_object_map_.find(robot_label);
                if (primary != adjacency_object_map_.end()) {
                    return &primary->second;
                }
                if (robot_label != "robot") {
                    auto fallback = adjacency_object_map_.find("robot");
                    if (fallback != adjacency_object_map_.end()) {
                        return &fallback->second;
                    }
                }
                return nullptr;
            }();

            if (robot_neighbors) {
                auto neighbor_it = robot_neighbors->find(region_label);
                if (neighbor_it != robot_neighbors->end()) {
                    bundle.blocking_objects.insert(neighbor_it->second.begin(), neighbor_it->second.end());
                }
            }
        }

        result[key] = std::move(bundle);
    }

    return result;
}

} // namespace namo
