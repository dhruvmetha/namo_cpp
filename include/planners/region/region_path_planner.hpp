#pragma once

#include "planners/region/region_graph.hpp"
#include <vector>
#include <string>
#include <array>
#include <climits>
#include <chrono>

namespace namo {

/**
 * @brief Result of path planning on region graph
 * 
 * Contains the shortest path from robot region to goal region and
 * the objects that need to be moved to achieve this path.
 */
struct PathSolution {
    // Path through regions (sequence of region IDs)
    GenericFixedVector<int, MAX_REGIONS> region_path;
    
    // Objects that must be moved to achieve this path (in order)
    GenericFixedVector<std::string, MAX_BLOCKING_OBJECTS_PER_EDGE * MAX_REGION_EDGES> blocking_objects;
    
    // Total number of obstacles that need to be removed
    int total_obstacles_to_remove = 0;
    
    // Path cost (for future extensions)
    double path_cost = 0.0;
    
    // Success flag
    bool path_found = false;
    
    // Additional information
    std::string failure_reason;
    
    PathSolution() = default;
    
    // Check if path is valid
    bool is_valid() const {
        return path_found && region_path.size() >= 2 && total_obstacles_to_remove >= 0;
    }
    
    // Get path length (number of regions in path)
    size_t path_length() const {
        return region_path.size();
    }
    
    // Check if path requires object movements
    bool requires_object_movement() const {
        return total_obstacles_to_remove > 0;
    }
    
    // Get start and goal regions
    int start_region() const {
        return region_path.empty() ? -1 : region_path[0];
    }
    
    int goal_region() const {
        return region_path.empty() ? -1 : region_path.back();
    }
};

/**
 * @brief Plans shortest paths on region connectivity graphs
 * 
 * Uses BFS to find paths that minimize the number of objects that need
 * to be moved. This provides the object selection heuristic for the
 * high-level planner.
 */
class RegionPathPlanner {
public:
    /**
     * @brief Constructor
     */
    RegionPathPlanner() = default;
    
    /**
     * @brief Find shortest path from robot region to goal region
     * @param graph Region connectivity graph
     * @return Path solution with minimal obstacle removal
     */
    PathSolution find_shortest_path(const RegionGraph& graph);
    
    /**
     * @brief Find shortest path between specific regions
     * @param graph Region connectivity graph
     * @param start_region_id Start region ID
     * @param goal_region_id Goal region ID
     * @return Path solution
     */
    PathSolution find_path_between_regions(const RegionGraph& graph, 
                                         int start_region_id, 
                                         int goal_region_id);
    
    /**
     * @brief Check if goal is reachable from robot position
     * @param graph Region connectivity graph
     * @return True if path exists (possibly requiring object movement)
     */
    bool is_goal_reachable(const RegionGraph& graph);
    
    /**
     * @brief Find all objects that block the optimal path
     * @param graph Region connectivity graph
     * @return List of blocking objects in priority order
     */
    std::vector<std::string> find_critical_blocking_objects(const RegionGraph& graph);
    
    /**
     * @brief Statistics and debugging
     */
    struct PlanningStats {
        int nodes_expanded = 0;
        int total_regions_explored = 0;
        int total_edges_examined = 0;
        double planning_time_ms = 0.0;
        bool found_optimal_path = false;
    };
    
    const PlanningStats& get_last_planning_stats() const { return last_stats_; }
    void reset_statistics() { last_stats_ = PlanningStats{}; }

private:
    // Pre-allocated BFS workspace
    static constexpr size_t MAX_BFS_QUEUE = 1000;
    std::array<int, MAX_BFS_QUEUE> bfs_queue_;
    size_t queue_front_ = 0, queue_back_ = 0;
    
    // Pre-allocated search state
    std::array<bool, MAX_REGIONS> visited_;
    std::array<int, MAX_REGIONS> parent_;
    std::array<int, MAX_REGIONS> distance_;
    std::array<int, MAX_REGIONS> obstacles_count_;  // Number of obstacles to reach each region
    
    // Statistics tracking
    mutable PlanningStats last_stats_;
    
    // Core BFS implementation
    PathSolution run_bfs(const RegionGraph& graph, int start_region, int goal_region);
    
    // Path reconstruction
    PathSolution reconstruct_path(const RegionGraph& graph, int start_region, int goal_region);
    
    // Extract blocking objects from path
    void extract_blocking_objects_from_path(const RegionGraph& graph, 
                                           PathSolution& solution);
    
    // BFS queue management
    void reset_bfs_queue() { queue_front_ = queue_back_ = 0; }
    void enqueue_region(int region_id) {
        if (queue_back_ < MAX_BFS_QUEUE) {
            bfs_queue_[queue_back_++] = region_id;
        }
    }
    int dequeue_region() {
        return (queue_front_ < queue_back_) ? bfs_queue_[queue_front_++] : -1;
    }
    bool is_queue_empty() const { return queue_front_ >= queue_back_; }
    
    // Search state management
    void reset_search_state() {
        visited_.fill(false);
        parent_.fill(-1);
        distance_.fill(INT_MAX);
        obstacles_count_.fill(INT_MAX);
    }
    
    // Utilities
    int calculate_edge_cost(const RegionEdge& edge) const {
        // Cost is number of blocking objects on this edge
        return static_cast<int>(edge.blocking_objects.size());
    }
    
    bool is_valid_region_id(int region_id, const RegionGraph& graph) const {
        return region_id >= 0 && region_id < static_cast<int>(graph.regions.size());
    }
    
    void update_statistics(const std::chrono::high_resolution_clock::time_point& start,
                          const std::chrono::high_resolution_clock::time_point& end,
                          bool found_path) const;
};

} // namespace namo