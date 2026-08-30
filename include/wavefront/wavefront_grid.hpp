#pragma once

#include "core/types.hpp"
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <cstdint>
#include <cstddef>

namespace namo {

// Forward declaration
class NAMOEnvironment;

/**
 * @brief Hash function for coordinate pairs (used in unordered_set/map)
 *
 * Grid coordinates are small non-negative ints, so packing (x,y) into a 64-bit key gives a
 * COLLISION-FREE hash. The previous `hash(x) ^ (hash(y)<<1)` (= `x ^ (y<<1)`) collided
 * systematically on a grid — e.g. (0,0) and (2,1) both hash to 0 — so the connected-components
 * flood-fill's unordered_set find/insert/erase degraded to super-linear, making
 * find_connected_components take ~20-30 s on large grids (5 mm car scenes were unaffected at
 * ~1 ms). This is PURELY a performance fix: the region labeling is hash-independent (BFS is
 * FIFO-queue order; region seeds are the lexicographically-smallest cell — see
 * WavefrontGrid::select_random_point), so outputs are bit-identical.
 */
struct CoordinateHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32)
               | static_cast<std::uint32_t>(p.second);
    }
};

struct RegionGoalSample {
    double x;
    double y;
    double theta;
};

struct RegionGoalBundle {
    std::vector<RegionGoalSample> goals;
    std::unordered_set<std::string> blocking_objects;
};


/**
 * @brief Grid discretization utility for spatial planning
 * 
 * Handles discretization of continuous space into grid cells and inflation
 * of objects to account for robot size. This class is focused solely on
 * grid management and geometric operations - no path planning algorithms.
 */
class WavefrontGrid {
public:
    static constexpr double kResolution = 0.01;

    /**
     * @brief Constructor
    * @note Uses a fixed grid resolution of 0.01 meters
     * @param env Environment (used to determine bounds)
     * @param robot_size Robot size for inflation [width, height]
     * @param tier1_inflation_margin Tier-1 additive inflation margin in meters
     */
    WavefrontGrid(NAMOEnvironment& env, 
                  const std::vector<double>& robot_size,
                  double tier1_inflation_margin = 0.005);
    
    /**
     * @brief Update dynamic grid with current object positions
     * @param env Environment containing current object states
     */
    void update_dynamic_grid(NAMOEnvironment& env);
    
    /**
     * @brief Check if a grid cell is free (not occupied by obstacles)
     * @param x Grid x coordinate
     * @param y Grid y coordinate
     * @return True if cell is free
     */
    bool is_cell_free(int x, int y) const;

    // Grid dimension accessors
    int get_grid_width() const { return grid_width_; }
    int get_grid_height() const { return grid_height_; }
    double get_resolution() const { return resolution_; }
    
    // Coordinate conversion utilities
    int world_to_grid_x(double world_x) const {
        return static_cast<int>(std::floor((world_x - bounds_[0]) / resolution_));
    }
    
    int world_to_grid_y(double world_y) const {
        return static_cast<int>(std::floor((world_y - bounds_[2]) / resolution_));
    }
    
    double grid_to_world_x(int grid_x) const {
        return bounds_[0] + grid_x * resolution_;
    }
    
    double grid_to_world_y(int grid_y) const {
        return bounds_[2] + grid_y * resolution_;
    }
    
    bool is_valid_grid_coord(int x, int y) const {
        return x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_;
    }
    

    // ========================
    // Connected Components Analysis
    // ========================
    
    /**
     * @brief Find all disconnected regions of free space with robot and goal priorities
     * @param robot_pos Robot position in world coordinates [x, y]
     * @param goal_cells Set of goal cell coordinates in world coordinates (assumed to be valid and in free space)
     * @return Map from region_id to set of grid coordinates (1=robot region, 2=goal region, 3+=other regions)
     */
    std::unordered_map<int, std::unordered_set<std::pair<int, int>, CoordinateHash>> 
    find_connected_components(const std::array<double, 2>& robot_pos, 
                             const std::vector<std::array<double, 2>>& goal_cells) const;
                             
    /**
     * @brief Find all disconnected regions (original method for backward compatibility)
     * @return Map from region_id (starting at 1) to set of grid coordinates in that region
     */
    std::unordered_map<int, std::unordered_set<std::pair<int, int>, CoordinateHash>> 
    find_connected_components() const;
    
    /**
     * @brief Get region labels (robot=1, goal=2, other regions=3+)
     * @return Map from region_id to human-readable label
     */
    std::unordered_map<int, std::string> get_region_labels() const;
    
    /**
     * @brief Build region connectivity graph by analyzing movable object removal
     * @param env Environment containing movable objects
     * @return Adjacency list mapping region labels to connected region labels
     */
    std::unordered_map<std::string, std::unordered_set<std::string>> 
    build_region_connectivity_graph(NAMOEnvironment& env);

    /**
     * Movable objects that plug the boundary between two regions. An entry written by the per-object
     * pass carries the stronger guarantee that the object ALONE joins the regions. An entry written
     * by the movable-blob pass means the named objects form the plug together, and clearing some
     * subset of them joins the regions. Both are candidate sets that the simulator then verifies;
     * neither is a proof that any particular push succeeds.
     *
     * To tell the two apart, read get_multi_object_edges().
     */
    const std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>>&
    get_region_edge_objects() const { return adjacency_object_map_; }

    /**
     * Region pairs that no single object opens on its own, so the whole plug named in
     * get_region_edge_objects() is needed. This is the one-object versus needs-a-group distinction, and it
     * is a bit per edge rather than a list, because for such an edge get_region_edge_objects() already
     * names exactly the objects that form the plug.
     *
     * An edge absent from here was written by the per-object pass, so some single object suffices.
     */
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
    get_multi_object_edges() const { return multi_object_edges_; }

    std::unordered_map<std::string, RegionGoalBundle> sample_region_goals(int goals_per_region) const;
    std::unordered_map<std::string, RegionGoalBundle> sample_region_goals(int goals_per_region, uint32_t seed) const;

private:
    // Grid parameters
    double resolution_;
    std::vector<double> bounds_;
    int grid_width_;
    int grid_height_;
    std::vector<double> robot_size_;
    double tier1_inflation_margin_;
    
    // Grid storage
    std::vector<std::vector<int>> uninflated_grid_;    // Original obstacles without inflation (for visualization)
    std::vector<std::vector<int>> static_grid_;        // MISNOMER: holds ALL obstacles (static AND
                                                      // movable) with inflation, built from the total
                                                      // occupancy count and then copied to dynamic_grid_.
                                                      // The name is load-bearing in call sites; do not
                                                      // "fix" it by changing what it stores.
    mutable std::vector<std::vector<int>> dynamic_grid_;  // All obstacles (static + movable) with inflation
                                                          // Mutable to allow fixing discretization artifacts
    std::vector<std::vector<int>> occupancy_count_grid_;  // Number of inflated objects occupying each cell,
                                                         // statics AND movables together
    std::vector<std::vector<int>> static_count_grid_;     // Number of inflated STATIC objects per cell.
                                                         // Exists so "blocked only by movables" is askable:
                                                         // dynamic_grid_ == -1 && static_count_grid_ == 0.
                                                         // ⛔ Do NOT substitute occupancy_count_grid_ > 0 for
                                                         // the dynamic_grid_ test. find_connected_components
                                                         // force-clears cells around the robot without
                                                         // touching either count, so counts alone would call
                                                         // those cells blocked when they are free.
    
    // Core initialization and update methods
    void rebuild_grids(NAMOEnvironment& env);
    
    // Geometric calculation methods
    GridFootprint calculate_rotated_footprint(const ObjectInfo& obj, const ObjectState& state);
    bool is_point_in_rotated_rectangle(double px, double py, 
                                      const ObjectState& state, 
                                      const ObjectInfo& obj) const;
    
    // Connected components analysis helper methods
    std::pair<int, int> select_random_point(const std::unordered_set<std::pair<int, int>, CoordinateHash>& points) const;
    std::unordered_set<std::pair<int, int>, CoordinateHash> explore_connected_region(
        const std::pair<int, int>& start_point, 
        std::unordered_set<std::pair<int, int>, CoordinateHash>& unvisited_points) const;
        
    // Cached region analysis (computed on demand)
    mutable std::unordered_map<int, std::unordered_set<std::pair<int, int>, CoordinateHash>> cached_regions_;
    mutable std::unordered_map<int, std::string> cached_region_labels_;  // Maps region ID to label
    mutable std::vector<std::vector<int>> region_grid_;  // Maps each cell to its region ID
    mutable bool regions_valid_ = false;
    mutable std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<std::string>>> adjacency_object_map_;
    mutable std::unordered_map<std::string, std::unordered_set<std::string>> multi_object_edges_;
};

} // namespace namo
