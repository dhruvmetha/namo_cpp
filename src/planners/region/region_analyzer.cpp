#include "planners/region/region_analyzer.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace namo {

RegionAnalyzer::RegionAnalyzer(double resolution, double sampling_density, double goal_region_radius)
    : resolution_(resolution)
    , sampling_density_(sampling_density)
    , goal_region_radius_(goal_region_radius)
    , region_merge_threshold_(0.1)
    , grid_width_(0)
    , grid_height_(0)
    , queue_front_(0)
    , queue_back_(0)
    , rng_(std::random_device{}()) {
    
    // Initialize region usage tracking
    region_used_.fill(false);
}

RegionGraph RegionAnalyzer::discover_regions(NAMOEnvironment& env, const SE2State& robot_goal) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear previous analysis state
    clear_workspace();
    last_stats_ = AnalysisStats{};
    
    // Initialize grids based on environment
    initialize_grids(env);
    
    // Generate sample points using PRM-style sampling
    generate_sample_points(bounds_);
    
    // Discover regions from valid samples using flood-fill
    discover_regions_from_samples();
    
    // Build initial region graph
    RegionGraph graph;
    
    // Extract lightweight regions (no grid cell storage)
    for (int region_id = 0; region_id < MAX_REGIONS; ++region_id) {
        if (region_used_[region_id]) {
            Region region(region_id);
            graph.add_region(region);
            last_stats_.regions_discovered++;
        }
    }
    
    auto flood_fill_end_time = std::chrono::high_resolution_clock::now();
    last_stats_.flood_fill_time_ms = std::chrono::duration<double, std::milli>(
        flood_fill_end_time - start_time).count();
    
    // Merge similar regions
    merge_similar_regions(graph);
    
    // Build connectivity graph between regions
    build_connectivity_graph(graph, env);
    
    // Handle special goal region
    handle_goal_region(graph, robot_goal);
    
    // Identify which region contains the robot
    identify_robot_region(graph, env);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    last_stats_.graph_construction_time_ms = std::chrono::duration<double, std::milli>(
        end_time - flood_fill_end_time).count();
    
    return graph;
}

RegionGraph RegionAnalyzer::update_regions(NAMOEnvironment& env, 
                                          const RegionGraph& existing_graph,
                                          const SE2State& robot_goal) {
    // For now, implement as full recomputation
    // TODO: Implement incremental updates for better performance
    return discover_regions(env, robot_goal);
}

int RegionAnalyzer::find_region_containing_point(const RegionGraph& graph, 
                                               double world_x, double world_y) const {
    int grid_x = world_to_grid_x(world_x);
    int grid_y = world_to_grid_y(world_y);
    
    if (!is_valid_grid_coord(grid_x, grid_y)) {
        return -1;
    }
    
    // Look up directly from region labels grid
    if (region_labels_[grid_x][grid_y] >= 0) {
        return region_labels_[grid_x][grid_y];
    }
    
    return -1;  // No region contains this point
}

void RegionAnalyzer::initialize_grids(NAMOEnvironment& env) {
    // Get environment bounds
    bounds_ = env.get_environment_bounds();
    
    // Calculate grid dimensions
    grid_width_ = static_cast<int>(std::ceil((bounds_[1] - bounds_[0]) / resolution_));
    grid_height_ = static_cast<int>(std::ceil((bounds_[3] - bounds_[2]) / resolution_));
    
    // Initialize grids
    occupancy_grid_.assign(grid_width_, std::vector<int>(grid_height_, 0));
    region_labels_.assign(grid_width_, std::vector<int>(grid_height_, -1));
    
    // Mark occupied cells
    const auto& static_objects = env.get_static_objects();
    size_t num_static = env.get_num_static();
    
    for (size_t i = 0; i < num_static; ++i) {
        const ObjectInfo& obj = static_objects[i];
        mark_object_in_grid(obj, occupancy_grid_, 1);  // 1 = occupied
    }
    
    // Mark movable objects as occupied too
    const auto& movable_objects = env.get_movable_objects();
    size_t num_movable = env.get_num_movable();
    
    for (size_t i = 0; i < num_movable; ++i) {
        const ObjectInfo& obj = movable_objects[i];
        mark_object_in_grid(obj, occupancy_grid_, 1);  // 1 = occupied
    }
}

void RegionAnalyzer::generate_sample_points(const std::vector<double>& env_bounds) {
    sample_points_.clear();
    
    // Use reasonable fixed sample count for connected component discovery
    // Not area-based which creates excessive samples
    int max_samples = 500;  // Sufficient to find all connected components
    int max_attempts = 2000; // Allow some retries for valid samples
    
    // Generate random samples
    std::uniform_real_distribution<double> x_dist(env_bounds[0], env_bounds[1]);
    std::uniform_real_distribution<double> y_dist(env_bounds[2], env_bounds[3]);
    
    last_stats_.total_samples_generated = 0;
    
    for (int attempt = 0; attempt < max_attempts && sample_points_.size() < max_samples; ++attempt) {
        double x = x_dist(rng_);
        double y = y_dist(rng_);
        last_stats_.total_samples_generated++;
        
        // Check if sample is in free space
        int grid_x = world_to_grid_x(x);
        int grid_y = world_to_grid_y(y);
        
        if (is_valid_grid_coord(grid_x, grid_y) && is_cell_free(grid_x, grid_y)) {
            sample_points_.push_back({x, y});
            last_stats_.valid_samples_found++;
        }
    }
    
    // std::cout << "Generated " << sample_points_.size() << " valid samples from " 
              // << last_stats_.total_samples_generated << " attempts" << std::endl;
}

void RegionAnalyzer::discover_regions_from_samples() {
    int current_region_id = 0;
    int samples_since_last_region = 0;
    int max_idle_samples = 100;  // Stop if no new regions found in 100 consecutive samples
    int min_region_size = 10;    // Ignore tiny regions (noise)
    
    // std::cout << "Discovering regions from " << sample_points_.size() << " samples..." << std::endl;
    
    for (size_t i = 0; i < sample_points_.size() && current_region_id < MAX_REGIONS; ++i) {
        double x = sample_points_[i].first;
        double y = sample_points_[i].second;
        
        int grid_x = world_to_grid_x(x);
        int grid_y = world_to_grid_y(y);
        
        // Skip if this cell is already assigned to a region
        if (region_labels_[grid_x][grid_y] >= 0) {
            samples_since_last_region++;
            continue;
        }
        
        // Perform flood-fill to discover the connected component
        int cells_filled = flood_fill_from_point(grid_x, grid_y, current_region_id);
        
        if (cells_filled >= min_region_size) {
            region_used_[current_region_id] = true;
            current_region_id++;
            samples_since_last_region = 0;
            // std::cout << "Found region " << (current_region_id-1) << " with " << cells_filled << " cells" << std::endl;
        } else if (cells_filled > 0) {
            // Revert tiny region labeling
            for (int x = 0; x < grid_width_; ++x) {
                for (int y = 0; y < grid_height_; ++y) {
                    if (region_labels_[x][y] == current_region_id) {
                        region_labels_[x][y] = -1;
                    }
                }
            }
            samples_since_last_region++;
        } else {
            samples_since_last_region++;
        }
        
        // Early termination: if no new regions found recently, likely discovered all components
        if (samples_since_last_region >= max_idle_samples) {
            // std::cout << "Early termination: no new regions in " << max_idle_samples << " samples" << std::endl;
            break;
        }
    }
    
    // std::cout << "Region discovery completed: found " << current_region_id << " regions" << std::endl;
}

int RegionAnalyzer::flood_fill_from_point(int start_x, int start_y, int region_id) {
    if (!is_valid_grid_coord(start_x, start_y) || !is_cell_free(start_x, start_y) ||
        region_labels_[start_x][start_y] >= 0) {
        return 0;
    }
    
    // Initialize flood-fill queue
    reset_flood_fill_queue();
    enqueue_cell(start_x, start_y);
    region_labels_[start_x][start_y] = region_id;
    
    int cells_filled = 0;
    
    while (!is_queue_empty()) {
        auto [x, y] = dequeue_cell();
        cells_filled++;
        
        // Check all 8-connected neighbors
        for (const auto& [dx, dy] : DIRECTIONS) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (is_valid_grid_coord(nx, ny) && is_cell_free(nx, ny) && 
                region_labels_[nx][ny] < 0) {
                
                region_labels_[nx][ny] = region_id;
                enqueue_cell(nx, ny);
            }
        }
    }
    
    return cells_filled;
}

Region RegionAnalyzer::extract_region_from_grid(int region_id) const {
    Region region(region_id);
    // Note: No grid cells stored in lightweight approach
    // The region is identified only by its ID
    return region;
}

void RegionAnalyzer::merge_similar_regions(RegionGraph& graph) {
    // Simple region merging based on proximity of centroids
    bool merged_any = true;
    
    while (merged_any) {
        merged_any = false;
        
        for (size_t i = 0; i < graph.regions.size() && !merged_any; ++i) {
            for (size_t j = i + 1; j < graph.regions.size(); ++j) {
                if (should_merge_regions(graph.regions[i], graph.regions[j])) {
                    merge_regions(graph, i, j);
                    merged_any = true;
                    last_stats_.regions_merged++;
                    break;
                }
            }
        }
    }
}

bool RegionAnalyzer::should_merge_regions(const Region& region_a, const Region& region_b) const {
    // Simplified: never merge regions in the lightweight approach
    // Region merging requires computing centroids which we no longer store
    return false;
}

void RegionAnalyzer::merge_regions(RegionGraph& graph, int region_a_id, int region_b_id) {
    // Skip merging in lightweight approach - it's complex and not needed for the heuristic
    // Region merging would require updating all region labels in the grid
    return;
}

void RegionAnalyzer::build_connectivity_graph(RegionGraph& graph, NAMOEnvironment& env) {
    // std::cout << "Building connectivity graph for " << graph.regions.size() << " regions..." << std::endl;
    
    // Find spatially adjacent regions and check if they're blocked by movable objects
    for (size_t i = 0; i < graph.regions.size(); ++i) {
        const Region& region_a = graph.regions[i];
        
        // Find all regions that are spatially adjacent to region_a
        std::vector<int> adjacent_regions = find_spatially_adjacent_regions(region_a, graph);
        
        for (int j : adjacent_regions) {
            if (j > static_cast<int>(i)) {  // Avoid duplicate edges
                const Region& region_b = graph.regions[j];
                
                // Find movable objects that block connection between these regions
                std::vector<std::string> blocking_objects = find_blocking_objects_between_regions(region_a, region_b, env);
                
                // Only add edge if there are blocking objects
                if (!blocking_objects.empty()) {
                    graph.add_blocked_edge(i, j, blocking_objects);
                    // std::cout << "Added blocked edge: regions " << i << " and " << j 
                             // << " (blocked by " << blocking_objects.size() << " objects)" << std::endl;
                }
            }
        }
    }
    
    // std::cout << "Created " << graph.num_edges() << " blocked edges between regions" << std::endl;
}

std::vector<int> RegionAnalyzer::find_spatially_adjacent_regions(const Region& region, const RegionGraph& graph) const {
    std::vector<int> adjacent_regions;
    std::set<int> found_regions;  // Avoid duplicates
    
    // Scan the entire grid to find cells belonging to this region and check their neighbors
    for (int x = 0; x < grid_width_; ++x) {
        for (int y = 0; y < grid_height_; ++y) {
            if (region_labels_[x][y] == region.id) {
                // Check all 8 neighbors of this cell
                for (const auto& [dx, dy] : DIRECTIONS) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (is_valid_grid_coord(nx, ny) && region_labels_[nx][ny] >= 0) {
                        int neighbor_region_id = region_labels_[nx][ny];
                        
                        // Skip if it's the same region or already found
                        if (neighbor_region_id != region.id && found_regions.find(neighbor_region_id) == found_regions.end()) {
                            found_regions.insert(neighbor_region_id);
                            adjacent_regions.push_back(neighbor_region_id);
                        }
                    }
                }
            }
        }
    }
    
    return adjacent_regions;
}

std::vector<std::string> RegionAnalyzer::find_blocking_objects_between_regions(
    const Region& region_a, const Region& region_b, NAMOEnvironment& env) const {
    
    std::vector<std::string> blocking_objects;
    
    // Get all movable objects from environment
    const auto& movable_objects = env.get_movable_objects();
    size_t num_movable = env.get_num_movable();
    
    // For each movable object, check if it blocks the connection between regions
    for (size_t i = 0; i < num_movable; ++i) {
        const ObjectInfo& obj = movable_objects[i];
        
        // Check if object separates the two regions by being between them
        if (is_object_blocking_region_connection(region_a, region_b, obj)) {
            blocking_objects.push_back(obj.name);
        }
    }
    
    return blocking_objects;
}

bool RegionAnalyzer::is_object_blocking_region_connection(const Region& region_a, const Region& region_b, const ObjectInfo& obj) const {
    // Simple heuristic: check if object footprint overlaps with the boundary between regions
    
    // Get object bounds
    double obj_min_x = obj.position[0] - obj.size[0] * 0.5;
    double obj_max_x = obj.position[0] + obj.size[0] * 0.5;
    double obj_min_y = obj.position[1] - obj.size[1] * 0.5;
    double obj_max_y = obj.position[1] + obj.size[1] * 0.5;
    
    // Convert to grid coordinates
    int grid_min_x = world_to_grid_x(obj_min_x);
    int grid_max_x = world_to_grid_x(obj_max_x);
    int grid_min_y = world_to_grid_y(obj_min_y);
    int grid_max_y = world_to_grid_y(obj_max_y);
    
    // Check if object footprint contains cells from both regions or separates them
    bool touches_region_a = false;
    bool touches_region_b = false;
    
    // Check all cells in object's footprint and their immediate neighbors
    for (int x = grid_min_x - 1; x <= grid_max_x + 1; ++x) {
        for (int y = grid_min_y - 1; y <= grid_max_y + 1; ++y) {
            if (is_valid_grid_coord(x, y) && region_labels_[x][y] >= 0) {
                if (region_labels_[x][y] == region_a.id) {
                    touches_region_a = true;
                }
                if (region_labels_[x][y] == region_b.id) {
                    touches_region_b = true;
                }
            }
        }
    }
    
    // Object blocks connection if it's adjacent to both regions
    return touches_region_a && touches_region_b;
}

bool RegionAnalyzer::are_regions_adjacent(const Region& region_a, const Region& region_b) const {
    // Check if regions are spatially adjacent by scanning the region labels grid
    for (int x = 0; x < grid_width_; ++x) {
        for (int y = 0; y < grid_height_; ++y) {
            if (region_labels_[x][y] == region_a.id) {
                // Check all 8 neighbors of this cell
                for (const auto& [dx, dy] : DIRECTIONS) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (is_valid_grid_coord(nx, ny) && region_labels_[nx][ny] == region_b.id) {
                        return true;  // Found adjacent cell
                    }
                }
            }
        }
    }
    
    return false;  // No adjacent cells found
}

double RegionAnalyzer::calculate_connection_strength(const Region& region_a, const Region& region_b) const {
    // Simplified: uniform connection strength since we don't store centroids
    return 1.0;
}

void RegionAnalyzer::handle_goal_region(RegionGraph& graph, const SE2State& robot_goal) {
    // Check if goal position is already in a region
    int containing_region = find_region_containing_point(graph, robot_goal.x, robot_goal.y);
    
    if (containing_region >= 0) {
        // Goal is within an existing region - use that region as goal
        graph.goal_region_id = containing_region;
    } else {
        // Goal is isolated - create new goal region
        Region goal_region = create_goal_region(robot_goal);
        graph.goal_region_id = graph.add_region(goal_region);
    }
    
    // Mark the goal region
    if (graph.goal_region_id >= 0 && graph.goal_region_id < static_cast<int>(graph.regions.size())) {
        graph.regions[graph.goal_region_id].is_goal_region = true;
    }
}

Region RegionAnalyzer::create_goal_region(const SE2State& robot_goal) const {
    Region goal_region;
    goal_region.is_goal_region = true;
    // Note: No grid cells or centroid stored in lightweight approach
    // The goal region will be identified by its region ID in the grid labels
    return goal_region;
}

int RegionAnalyzer::split_region_for_goal(RegionGraph& graph, int containing_region_id, 
                                        const SE2State& robot_goal) {
    // Simplified: just use the containing region as the goal region
    // No actual splitting in the lightweight approach
    return containing_region_id;
}

void RegionAnalyzer::identify_robot_region(RegionGraph& graph, NAMOEnvironment& env) {
    // Get robot position
    const ObjectState* robot_state = env.get_robot_state();
    if (!robot_state) {
        graph.robot_region_id = -1;
        return;
    }
    
    // Find which region contains the robot
    graph.robot_region_id = find_region_containing_point(
        graph, robot_state->position[0], robot_state->position[1]);
    
    // Mark the robot region
    if (graph.robot_region_id >= 0 && graph.robot_region_id < static_cast<int>(graph.regions.size())) {
        graph.regions[graph.robot_region_id].contains_robot = true;
    }
}

// Utility methods
void RegionAnalyzer::mark_object_in_grid(const ObjectInfo& obj, 
                                       std::vector<std::vector<int>>& grid, int value) {
    // Simple rectangular footprint marking
    double half_width = obj.size[0] * 0.5;
    double half_height = obj.size[1] * 0.5;
    
    int min_x = world_to_grid_x(obj.position[0] - half_width);
    int max_x = world_to_grid_x(obj.position[0] + half_width);
    int min_y = world_to_grid_y(obj.position[1] - half_height);
    int max_y = world_to_grid_y(obj.position[1] + half_height);
    
    for (int x = min_x; x <= max_x; ++x) {
        for (int y = min_y; y <= max_y; ++y) {
            if (is_valid_grid_coord(x, y)) {
                grid[x][y] = value;
            }
        }
    }
}

bool RegionAnalyzer::is_valid_grid_coord(int x, int y) const {
    return x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_;
}

bool RegionAnalyzer::is_cell_free(int x, int y) const {
    return is_valid_grid_coord(x, y) && occupancy_grid_[x][y] == 0;
}

int RegionAnalyzer::world_to_grid_x(double world_x) const {
    return static_cast<int>(std::floor((world_x - bounds_[0]) / resolution_));
}

int RegionAnalyzer::world_to_grid_y(double world_y) const {
    return static_cast<int>(std::floor((world_y - bounds_[2]) / resolution_));
}

double RegionAnalyzer::grid_to_world_x(int grid_x) const {
    return bounds_[0] + grid_x * resolution_;
}

double RegionAnalyzer::grid_to_world_y(int grid_y) const {
    return bounds_[2] + grid_y * resolution_;
}

double RegionAnalyzer::distance_between_points(double x1, double y1, double x2, double y2) const {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

void RegionAnalyzer::save_region_grid(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for region grid output: " << filename << std::endl;
        return;
    }
    
    // Write header with grid dimensions
    file << "# Region Grid Visualization" << std::endl;
    file << "# Grid dimensions: " << grid_width_ << "x" << grid_height_ << std::endl;
    file << "# Resolution: " << resolution_ << " meters per cell" << std::endl;
    file << "# Bounds: [" << bounds_[0] << ", " << bounds_[1] << "] x [" << bounds_[2] << ", " << bounds_[3] << "]" << std::endl;
    file << "# Cell values: -1=unassigned, >=0=region_id" << std::endl;
    file << std::endl;
    
    // Write grid data (row by row, top to bottom)
    for (int y = grid_height_ - 1; y >= 0; --y) {  // Start from top
        for (int x = 0; x < grid_width_; ++x) {
            if (x > 0) file << " ";
            if (region_labels_.empty()) {
                file << "-1";  // No regions computed
            } else {
                file << region_labels_[x][y];
            }
        }
        file << std::endl;
    }
    
    file.close();
    // std::cout << "Region grid saved to: " << filename << std::endl;
}

void RegionAnalyzer::clear_workspace() {
    sample_points_.clear();
    region_used_.fill(false);
    queue_front_ = queue_back_ = 0;
    
    occupancy_grid_.clear();
    region_labels_.clear();
}

} // namespace namo