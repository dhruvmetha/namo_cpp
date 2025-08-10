#pragma once

#include "planners/region/region_graph.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "environment/namo_environment.hpp"
#include "core/types.hpp"
#include <random>
#include <unordered_set>
#include <set>

namespace namo {

/**
 * @brief Analyzes environment to discover free-space regions using PRM + flood-fill
 * 
 * This class implements the core region discovery algorithm:
 * 1. Sample random points in environment (PRM-style)
 * 2. For each valid sample, perform flood-fill to discover connected region
 * 3. Merge overlapping regions and build connectivity graph
 * 4. Handle special goal region (always separate vertex)
 */
class RegionAnalyzer {
public:
    /**
     * @brief Constructor
     * @param resolution Grid resolution for flood-fill (should match wavefront planner)
     * @param sampling_density Number of samples per square meter
     * @param goal_region_radius Radius around goal position (default 0.25m)
     */
    RegionAnalyzer(double resolution = 0.05, 
                  double sampling_density = 100.0,
                  double goal_region_radius = 0.25);
    
    /**
     * @brief Discover all free-space regions in environment
     * @param env Environment to analyze
     * @param robot_goal Goal position for robot navigation
     * @return Complete region graph with connectivity information
     */
    RegionGraph discover_regions(NAMOEnvironment& env, const SE2State& robot_goal);
    
    /**
     * @brief Update existing region graph after object movements
     * @param env Current environment state
     * @param existing_graph Graph to update
     * @param robot_goal Current robot goal
     * @return Updated region graph
     */
    RegionGraph update_regions(NAMOEnvironment& env, 
                              const RegionGraph& existing_graph,
                              const SE2State& robot_goal);
    
    /**
     * @brief Find which region contains a given point
     * @param graph Region graph to search
     * @param world_x X coordinate in world frame
     * @param world_y Y coordinate in world frame
     * @return Region ID containing the point, or -1 if no region found
     */
    int find_region_containing_point(const RegionGraph& graph, 
                                   double world_x, double world_y) const;
    
    /**
     * @brief Configuration setters
     */
    void set_sampling_density(double density) { sampling_density_ = density; }
    void set_goal_region_radius(double radius) { goal_region_radius_ = radius; }
    void set_region_merge_threshold(double threshold) { region_merge_threshold_ = threshold; }
    
    /**
     * @brief Get current configuration
     */
    double get_sampling_density() const { return sampling_density_; }
    double get_goal_region_radius() const { return goal_region_radius_; }
    double get_region_merge_threshold() const { return region_merge_threshold_; }
    
    /**
     * @brief Save region labels to file (similar to wavefront visualization)
     * @param filename Output filename for region grid
     */
    void save_region_grid(const std::string& filename) const;
    
    /**
     * @brief Statistics and debugging
     */
    struct AnalysisStats {
        int total_samples_generated = 0;
        int valid_samples_found = 0;
        int regions_discovered = 0;
        int regions_merged = 0;
        double analysis_time_ms = 0.0;
        double flood_fill_time_ms = 0.0;
        double graph_construction_time_ms = 0.0;
    };
    
    const AnalysisStats& get_last_analysis_stats() const { return last_stats_; }

private:
    // Configuration
    double resolution_;
    double sampling_density_;
    double goal_region_radius_;
    double region_merge_threshold_;
    
    // Grid and environment bounds
    std::vector<double> bounds_;
    int grid_width_, grid_height_;
    
    // Pre-allocated workspace for flood-fill
    static constexpr size_t MAX_FLOOD_FILL_QUEUE = 50000;
    std::array<std::pair<int, int>, MAX_FLOOD_FILL_QUEUE> flood_fill_queue_;
    size_t queue_front_, queue_back_;
    
    // Pre-allocated sample storage
    static constexpr size_t MAX_SAMPLES = 1000;  // Reduced from 10,000 to avoid stack overflow
    GenericFixedVector<std::pair<double, double>, MAX_SAMPLES> sample_points_;
    
    // Region discovery workspace
    std::vector<std::vector<int>> occupancy_grid_;     // 0=free, 1=occupied, 2=visited
    std::vector<std::vector<int>> region_labels_;      // Region ID for each cell
    std::array<bool, MAX_REGIONS> region_used_;        // Track which region IDs are used
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Statistics tracking
    mutable AnalysisStats last_stats_;
    
    // 8-connected grid directions for flood-fill
    static constexpr std::array<std::pair<int, int>, 8> DIRECTIONS = {{
        {1,0}, {-1,0}, {0,1}, {0,-1},
        {1,1}, {1,-1}, {-1,1}, {-1,-1}
    }};
    
    // Core algorithm methods
    void initialize_grids(NAMOEnvironment& env);
    void generate_sample_points(const std::vector<double>& env_bounds);
    void discover_regions_from_samples();
    void merge_similar_regions(RegionGraph& graph);
    void build_connectivity_graph(RegionGraph& graph, NAMOEnvironment& env);
    void handle_goal_region(RegionGraph& graph, const SE2State& robot_goal);
    void identify_robot_region(RegionGraph& graph, NAMOEnvironment& env);
    
    // Flood-fill implementation
    int flood_fill_from_point(int start_x, int start_y, int region_id);
    bool is_valid_grid_coord(int x, int y) const;
    bool is_cell_free(int x, int y) const;
    
    // Grid coordinate conversion
    int world_to_grid_x(double world_x) const;
    int world_to_grid_y(double world_y) const;
    double grid_to_world_x(int grid_x) const;
    double grid_to_world_y(int grid_y) const;
    
    // Sample validation
    bool is_sample_valid(double x, double y, NAMOEnvironment& env) const;
    
    // Region processing
    Region extract_region_from_grid(int region_id) const;
    bool should_merge_regions(const Region& region_a, const Region& region_b) const;
    void merge_regions(RegionGraph& graph, int region_a_id, int region_b_id);
    
    // Connectivity analysis
    std::vector<int> find_spatially_adjacent_regions(const Region& region, const RegionGraph& graph) const;
    bool are_regions_adjacent(const Region& region_a, const Region& region_b) const;
    double calculate_connection_strength(const Region& region_a, const Region& region_b) const;
    std::vector<std::string> find_blocking_objects_between_regions(
        const Region& region_a, const Region& region_b, NAMOEnvironment& env) const;
    bool is_object_blocking_region_connection(const Region& region_a, const Region& region_b, const ObjectInfo& obj) const;
    
    // Goal region handling
    Region create_goal_region(const SE2State& robot_goal) const;
    int split_region_for_goal(RegionGraph& graph, int containing_region_id, 
                             const SE2State& robot_goal);
    
    // Flood-fill queue management
    void reset_flood_fill_queue() { queue_front_ = queue_back_ = 0; }
    void enqueue_cell(int x, int y) {
        assert(queue_back_ < MAX_FLOOD_FILL_QUEUE);
        flood_fill_queue_[queue_back_++] = {x, y};
    }
    std::pair<int, int> dequeue_cell() {
        return (queue_front_ < queue_back_) ? flood_fill_queue_[queue_front_++] : std::make_pair(-1, -1);
    }
    bool is_queue_empty() const { return queue_front_ >= queue_back_; }
    
    // Utilities
    double distance_between_points(double x1, double y1, double x2, double y2) const;
    void clear_workspace();
    void update_statistics(const std::chrono::high_resolution_clock::time_point& start_time,
                          const std::chrono::high_resolution_clock::time_point& end_time) const;
    void mark_object_in_grid(const ObjectInfo& obj, std::vector<std::vector<int>>& grid, int value);
};

} // namespace namo