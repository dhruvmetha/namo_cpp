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
 * @brief High-performance wavefront planner
 * 
 * Computes reachability grid using BFS from robot start position.
 * Rebuilds the wavefront from scratch on each update for simplicity and correctness.
 */
class WavefrontPlanner {
public:
    /**
     * @brief Constructor
     * @param resolution Grid resolution in meters
     * @param env Environment (used for initial setup)
     * @param robot_size Robot size for inflation [width, height]
     */
    WavefrontPlanner(double resolution, NAMOEnvironment& env, 
                    const std::vector<double>& robot_size);
    
    /**
     * @brief Update wavefront by rebuilding from scratch
     * @param env Current environment state
     * @param start_pos Start position [x, y]
     * @return Always true (wavefront is always rebuilt)
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
     * @brief Get mutable access to grid for debugging (use carefully!)
     */
    std::vector<std::vector<int>>& get_mutable_grid() { return reachability_grid_; }
    
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
    
    // No change tracking needed - always rebuild from scratch
    
    // BFS workspace - reused across calls (simplified)
    static constexpr size_t MAX_BFS_QUEUE = 4000000;  // Increased to handle 1410x2210 grid (3.1M cells)
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
    void recompute_wavefront(NAMOEnvironment& env, const std::vector<double>& start_pos);
    void rebuild_dynamic_grid_from_current_objects(NAMOEnvironment& env);
    void add_footprint_to_dynamic_grid(const GridFootprint& footprint);
    
    // Geometry helpers
    GridFootprint calculate_rotated_footprint(const ObjectInfo& obj, const ObjectState& state);
    
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
    
    // No change tracking utilities needed
    
    // Performance monitoring
    void update_performance_stats(const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& end) const;
};

} // namespace namo