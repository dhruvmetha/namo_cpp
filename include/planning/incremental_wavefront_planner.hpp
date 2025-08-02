#pragma once

#include "core/types.hpp"
#include "core/memory_manager.hpp"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace namo {

// Forward declaration
class NAMOEnvironment;

/**
 * @brief High-performance incremental wavefront planner
 * 
 * Uses change detection and differential updates to achieve 10-100x speedup
 * over naive wavefront recomputation for moving/rotating objects.
 */
class IncrementalWavefrontPlanner {
public:
    /**
     * @brief Constructor
     * @param resolution Grid resolution in meters
     * @param env Environment (used for initial setup)
     * @param robot_size Robot size for inflation [width, height]
     */
    IncrementalWavefrontPlanner(double resolution, NAMOEnvironment& env, 
                               const std::vector<double>& robot_size);
    
    /**
     * @brief Update wavefront incrementally
     * @param env Current environment state
     * @param start_pos Start position [x, y]
     * @return True if wavefront was updated
     */
    bool update_wavefront(NAMOEnvironment& env, const std::vector<double>& start_pos);
    
    /**
     * @brief Compute wavefront with goal reachability
     * @param env Environment
     * @param start_pos Start position
     * @param goal_positions Map of object names to goal positions
     * @return Tuple of (grid, reachable_points, reachability_flags)
     */
    std::tuple<
        std::vector<std::vector<int>>, 
        std::unordered_map<std::string, std::vector<std::array<double, 2>>>,
        std::unordered_map<std::string, std::vector<int>>
    > compute_wavefront(
        NAMOEnvironment& env,
        const std::vector<double>& start_pos,
        const std::unordered_map<std::string, std::vector<std::array<double, 2>>>& goal_positions
    );
    
    /**
     * @brief Check if goal is reachable
     * @param goal_pos Goal position [x, y]
     * @param goal_size Goal region size (default 0.05)
     * @return True if reachable
     */
    bool is_goal_reachable(const std::array<double, 2>& goal_pos, double goal_size = 0.05) const;
    
    /**
     * @brief Save wavefront to file for debugging
     * @param filename Output filename
     */
    void save_wavefront(const std::string& filename) const;
    
    /**
     * @brief Save wavefront with automatic iteration numbering for MPC debugging
     * @param base_filename Base filename (e.g., "debug_wavefront")
     * @param iteration MPC iteration number
     */
    void save_wavefront_iteration(const std::string& base_filename, int iteration) const;
    
    /**
     * @brief Get current reachability grid for external use
     */
    const std::vector<std::vector<int>>& get_grid() const { return reachability_grid_; }
    
    /**
     * @brief Get grid dimensions
     */
    int get_grid_width() const { return grid_width_; }
    int get_grid_height() const { return grid_height_; }
    double get_resolution() const { return resolution_; }
    const std::vector<double>& get_bounds() const { return bounds_; }
    
    /**
     * @brief Get performance statistics
     */
    const PlanningStats& get_statistics() const { return stats_; }
    void reset_statistics() { stats_.reset(); }
    
    /**
     * @brief Get reachability grid (same as get_grid for compatibility)
     */
    const std::vector<std::vector<int>>& get_distance_grid() const { return reachability_grid_; }
    
    /**
     * @brief Convert world coordinates to grid coordinates
     */
    int world_to_grid_x(double world_x) const {
        return static_cast<int>(std::floor((world_x - bounds_[0]) / resolution_));
    }
    
    int world_to_grid_y(double world_y) const {
        return static_cast<int>(std::floor((world_y - bounds_[2]) / resolution_));
    }
    
    /**
     * @brief Check if grid coordinates are valid
     */
    bool is_valid_grid_coord(int x, int y) const {
        return x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_;
    }
    
private:
    // Grid parameters
    double resolution_;
    std::vector<double> bounds_;
    int grid_width_;
    int grid_height_;
    std::vector<double> robot_size_;
    
    // Persistent grids - never reallocated
    std::vector<std::vector<int>> static_grid_;      // Static obstacles only
    std::vector<std::vector<int>> dynamic_grid_;     // Current full state
    std::vector<std::vector<int>> reachability_grid_; // Reachability from start: -2=obstacle, 0=unreachable, 1=reachable
    
    // Change tracking (simplified for reachability-only)
    static constexpr size_t MAX_CHANGES = 10000;
    std::array<GridChange, MAX_CHANGES> pending_changes_;
    size_t num_pending_changes_ = 0;
    
    // Object state tracking for change detection (kept for statistics)
    std::unordered_map<std::string, RotatingObjectSnapshot> object_snapshots_;
    std::array<std::string, 20> changed_objects_;  // Max movable objects
    size_t num_changed_objects_ = 0;
    
    // BFS workspace - reused across calls (simplified)
    static constexpr size_t MAX_BFS_QUEUE = 100000;
    std::array<std::pair<int, int>, MAX_BFS_QUEUE> bfs_queue_;
    size_t queue_front_ = 0, queue_back_ = 0;
    
    // 8-connected grid directions
    static constexpr std::array<std::pair<int, int>, 8> DIRECTIONS = {{
        {1,0}, {-1,0}, {0,1}, {0,-1},
        {1,1}, {1,-1}, {-1,1}, {-1,-1}
    }};
    
    // Performance tracking
    mutable PlanningStats stats_;
    
    // Grid utility functions
    double grid_to_world_x(int grid_x) const {
        return bounds_[0] + grid_x * resolution_;
    }
    
    double grid_to_world_y(int grid_y) const {
        return bounds_[2] + grid_y * resolution_;
    }
    
    // Core algorithm methods
    void initialize_static_grid(NAMOEnvironment& env);
    void detect_object_changes(NAMOEnvironment& env);
    void update_affected_cells(NAMOEnvironment& env);
    void full_reachability_recompute(const std::vector<double>& start_pos);
    
    // Simplified algorithm methods (bypass expensive incremental tracking)
    void simple_reachability_recompute(NAMOEnvironment& env, const std::vector<double>& start_pos);
    void rebuild_dynamic_grid_from_current_objects(NAMOEnvironment& env);
    void add_footprint_to_dynamic_grid(const GridFootprint& footprint);
    
    // Change detection helpers
    bool has_object_moved(const std::string& obj_name, NAMOEnvironment& env);
    void handle_combined_motion(const ObjectInfo& obj, const ObjectState& current_state,
                               RotatingObjectSnapshot& snapshot, NAMOEnvironment& env);
    GridFootprint calculate_rotated_footprint(const ObjectInfo& obj, const ObjectState& state);
    void find_footprint_differences(const GridFootprint& old_footprint, 
                                   const GridFootprint& new_footprint);
    
    // Geometric queries
    bool is_point_in_rotated_rectangle(double px, double py, const ObjectState& state, 
                                      const ObjectInfo& obj) const;
    bool is_point_in_goal_region(double px, double py, const std::array<double, 2>& goal_pos,
                                 double goal_size = 0.05) const;
    
    // BFS utilities
    void reset_bfs_queue() { queue_front_ = queue_back_ = 0; }
    void bfs_enqueue(int x, int y) {
        if (queue_back_ >= MAX_BFS_QUEUE) {
            throw std::runtime_error("BFS queue overflow - increase MAX_BFS_QUEUE or reduce grid size");
        }
        bfs_queue_[queue_back_++] = {x, y};
    }
    std::pair<int, int> bfs_dequeue() {
        return (queue_front_ < queue_back_) ? bfs_queue_[queue_front_++] : std::make_pair(-1, -1);
    }
    bool bfs_empty() const { return queue_front_ >= queue_back_; }
    
    // Change tracking utilities (simplified)
    void queue_cell_change(int x, int y, bool became_obstacle);
    void apply_pending_changes();
    
    // Performance monitoring
    void update_performance_stats(const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& end) const;
};

} // namespace namo