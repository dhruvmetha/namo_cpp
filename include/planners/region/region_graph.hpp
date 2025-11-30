#pragma once

#include "core/types.hpp"
#include <array>
#include <vector>
#include <string>
#include <utility>
#include <cassert>

namespace namo {

// Forward declarations
class NAMOEnvironment;

// Pre-allocation constants for zero-allocation runtime
static constexpr size_t MAX_REGIONS = 50;
static constexpr size_t MAX_REGION_EDGES = 200;  // Much lower now with proper adjacency
static constexpr size_t MAX_SAMPLES_PER_REGION = 1000;
static constexpr size_t MAX_GRID_CELLS_PER_REGION = 60000;  // Increased for large regions
static constexpr size_t MAX_BLOCKING_OBJECTS_PER_EDGE = 10;

/**
 * @brief Generic fixed-size vector for any type T
 * 
 * Extends the concept from FixedVector<double> to work with any type,
 * maintaining zero-allocation guarantees during runtime.
 */
template<typename T, size_t MAX_SIZE>
class GenericFixedVector {
private:
    std::array<T, MAX_SIZE> data_;
    size_t size_ = 0;
    
public:
    GenericFixedVector() = default;
    
    void push_back(const T& val) { 
        assert(size_ < MAX_SIZE); 
        data_[size_++] = val; 
    }
    
    void emplace_back(T&& val) {
        assert(size_ < MAX_SIZE);
        data_[size_++] = std::move(val);
    }
    
    void clear() { size_ = 0; }
    
    void resize(size_t new_size) {
        assert(new_size <= MAX_SIZE);
        size_ = new_size;
    }
    
    T& operator[](size_t i) { 
        assert(i < size_);
        return data_[i]; 
    }
    
    const T& operator[](size_t i) const { 
        assert(i < size_);
        return data_[i]; 
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t capacity() const { return MAX_SIZE; }
    
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // STL-like iterators
    T* begin() { return data_.data(); }
    T* end() { return data_.data() + size_; }
    const T* begin() const { return data_.data(); }
    const T* end() const { return data_.data() + size_; }
    
    // Back access
    T& back() { 
        assert(size_ > 0); 
        return data_[size_ - 1]; 
    }
    const T& back() const { 
        assert(size_ > 0); 
        return data_[size_ - 1]; 
    }
    
    // Find element
    T* find(const T& val) {
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] == val) return &data_[i];
        }
        return nullptr;
    }
    
    const T* find(const T& val) const {
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] == val) return &data_[i];
        }
        return nullptr;
    }
};

/**
 * @brief Lightweight region representation for memory-efficient heuristic
 * 
 * Only stores essential information needed for obstacle selection heuristic.
 * Grid cells are NOT stored to save memory - only used during discovery.
 */
struct Region {
    int id = -1;
    bool contains_robot = false; // Is robot currently in this region?
    bool is_goal_region = false; // Is this the special goal region?
    
    Region() = default;
    Region(int region_id) : id(region_id) {}
};

/**
 * @brief Represents a connection between two regions
 * 
 * An edge exists between regions that would be connected if certain
 * obstacles were removed. The edge tracks which objects block the connection.
 */
struct RegionEdge {
    int region_a_id = -1;
    int region_b_id = -1;
    
    // Objects that block the connection between these regions
    GenericFixedVector<std::string, MAX_BLOCKING_OBJECTS_PER_EDGE> blocking_objects;
    
    // Connection properties
    double connection_strength = 0.0;  // How "close" the regions are (for prioritization)
    bool is_currently_blocked = false; // Are there objects currently blocking this edge?
    
    RegionEdge() = default;
    RegionEdge(int a_id, int b_id) : region_a_id(a_id), region_b_id(b_id) {}
    
    // Add a blocking object to this edge
    void add_blocking_object(const std::string& object_name) {
        // Avoid duplicates
        if (blocking_objects.find(object_name) == nullptr) {
            blocking_objects.push_back(object_name);
        }
        update_blocked_status();
    }
    
    // Remove a blocking object (when object is moved away)
    void remove_blocking_object(const std::string& object_name) {
        // Find and remove the object
        for (size_t i = 0; i < blocking_objects.size(); ++i) {
            if (blocking_objects[i] == object_name) {
                // Move last element to this position and decrease size
                blocking_objects[i] = blocking_objects.back();
                blocking_objects.resize(blocking_objects.size() - 1);
                break;
            }
        }
        update_blocked_status();
    }
    
    // Update blocked status based on current blocking objects
    void update_blocked_status() {
        is_currently_blocked = !blocking_objects.empty();
    }
    
    // Get the other region ID given one region ID
    int get_other_region(int region_id) const {
        if (region_id == region_a_id) return region_b_id;
        if (region_id == region_b_id) return region_a_id;
        return -1;  // Invalid
    }
    
    // Check if this edge connects the given regions
    bool connects(int region_1, int region_2) const {
        return (region_a_id == region_1 && region_b_id == region_2) ||
               (region_a_id == region_2 && region_b_id == region_1);
    }
};

/**
 * @brief Memory-efficient region connectivity graph for obstacle selection heuristic
 * 
 * Focuses only on adjacency relationships and blocking objects between regions.
 * Optimized for the specific use case: BFS shortest path to find next obstacles.
 */
struct RegionGraph {
    // Lightweight regions (no heavy data storage)
    GenericFixedVector<Region, MAX_REGIONS> regions;
    
    // Adjacency list: region_id -> list of adjacent region_ids
    std::array<GenericFixedVector<int, MAX_REGIONS>, MAX_REGIONS> adjacency_list;
    
    // Blocking objects per edge: stores movable objects that separate adjacent regions
    std::array<std::array<GenericFixedVector<std::string, MAX_BLOCKING_OBJECTS_PER_EDGE>, MAX_REGIONS>, MAX_REGIONS> blocking_objects;
    
    // Essential region tracking
    int robot_region_id = -1;   // Which region contains the robot
    int goal_region_id = -1;    // Which region is the goal region
    
    RegionGraph() = default;
    
    // Add a new region
    int add_region(const Region& region) {
        assert(regions.size() < MAX_REGIONS);
        int region_id = static_cast<int>(regions.size());
        Region new_region = region;
        new_region.id = region_id;
        regions.push_back(new_region);
        return region_id;
    }
    
    // Add an edge between adjacent regions (only if blocked by movable objects)
    void add_blocked_edge(int region_a, int region_b, const std::vector<std::string>& movable_objects) {
        assert(region_a < static_cast<int>(regions.size()));
        assert(region_b < static_cast<int>(regions.size()));
        assert(region_a < MAX_REGIONS && region_b < MAX_REGIONS);
        
        // Only add edge if there are blocking objects
        if (movable_objects.empty()) {
            return;
        }
        
        // Check if edge already exists
        if (has_edge(region_a, region_b)) {
            return;
        }
        
        // Add to adjacency lists
        adjacency_list[region_a].push_back(region_b);
        adjacency_list[region_b].push_back(region_a);
        
        // Store blocking objects (both directions for fast lookup)
        for (const auto& obj : movable_objects) {
            blocking_objects[region_a][region_b].push_back(obj);
            blocking_objects[region_b][region_a].push_back(obj);
        }
    }
    
    // Check if edge exists between two regions
    bool has_edge(int region_a, int region_b) const {
        if (region_a >= MAX_REGIONS || region_b >= MAX_REGIONS) return false;
        
        for (size_t i = 0; i < adjacency_list[region_a].size(); ++i) {
            if (adjacency_list[region_a][i] == region_b) {
                return true;
            }
        }
        return false;
    }
    
    // Get blocking objects between two adjacent regions
    const GenericFixedVector<std::string, MAX_BLOCKING_OBJECTS_PER_EDGE>& get_blocking_objects(int region_a, int region_b) const {
        assert(region_a < MAX_REGIONS && region_b < MAX_REGIONS);
        return blocking_objects[region_a][region_b];
    }
    
    // Core heuristic: BFS to find next obstacles to move
    std::vector<std::string> get_next_obstacles_to_move() const {
        if (robot_region_id < 0 || goal_region_id < 0) {
            return {};  // Invalid state
        }
        
        if (robot_region_id == goal_region_id) {
            return {};  // Already in goal region
        }
        
        // BFS to find shortest path from robot region to goal region
        std::array<bool, MAX_REGIONS> visited{};
        std::array<int, MAX_REGIONS> parent{};
        std::array<int, MAX_REGIONS> queue{};
        
        // Initialize BFS
        int front = 0, back = 0;
        queue[back++] = robot_region_id;
        visited[robot_region_id] = true;
        parent[robot_region_id] = -1;
        
        int goal_parent = -1;
        
        // BFS search
        while (front < back) {
            int current = queue[front++];
            
            if (current == goal_region_id) {
                goal_parent = parent[current];
                break;
            }
            
            // Explore adjacent regions
            for (size_t i = 0; i < adjacency_list[current].size(); ++i) {
                int neighbor = adjacency_list[current][i];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    queue[back++] = neighbor;
                }
            }
        }
        
        // If path found, return blocking objects on first edge
        if (goal_parent >= 0) {
            int first_step = goal_region_id;
            while (parent[first_step] != robot_region_id) {
                first_step = parent[first_step];
            }
            
            // Return blocking objects on edge: robot_region -> first_step
            const auto& blocking = get_blocking_objects(robot_region_id, first_step);
            std::vector<std::string> result;
            for (size_t i = 0; i < blocking.size(); ++i) {
                result.push_back(blocking[i]);
            }
            return result;
        }
        
        return {};  // No path found
    }
    
    // Get all neighbor regions of a given region
    const GenericFixedVector<int, MAX_REGIONS>& get_neighbors(int region_id) const {
        assert(region_id < MAX_REGIONS);
        return adjacency_list[region_id];
    }
    
    // Clear all data (for recomputation)
    void clear() {
        regions.clear();
        for (auto& adj : adjacency_list) {
            adj.clear();
        }
        for (int i = 0; i < MAX_REGIONS; ++i) {
            for (int j = 0; j < MAX_REGIONS; ++j) {
                blocking_objects[i][j].clear();
            }
        }
        robot_region_id = -1;
        goal_region_id = -1;
    }
    
    // Simple statistics
    size_t num_regions() const { return regions.size(); }
    size_t num_edges() const { 
        size_t count = 0;
        for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
            count += adjacency_list[i].size();
        }
        return count / 2;  // Each edge counted twice
    }
    
    // Validation
    bool is_valid() const {
        return robot_region_id >= 0 && robot_region_id < static_cast<int>(regions.size()) &&
               goal_region_id >= 0 && goal_region_id < static_cast<int>(regions.size());
    }
};

/**
 * @brief Lightweight state representation for tree search
 * 
 * Fast state representation that avoids full MuJoCo simulation state copying.
 * Used for rapid tree exploration in the region-based planner.
 */
struct LightweightState {
    // Robot pose in SE(2)
    SE2State robot_pose;
    
    // Movable object poses (indices match environment's movable object array)
    std::array<SE2State, 100> movable_object_poses;  // MAX_MOVABLE_OBJECTS = 100

    // Object names for mapping (must match environment)
    GenericFixedVector<std::string, 100> object_names;
    
    LightweightState() = default;
    
    // Create from environment state
    void initialize_from_environment(NAMOEnvironment& env);
    
    // Fast copy operation (no dynamic allocation)
    LightweightState copy() const {
        LightweightState result = *this;
        return result;
    }
    
    // Apply object movement (used during tree search simulation)
    void apply_object_movement(const std::string& object_name, const SE2State& new_pose) {
        // Find object index
        for (size_t i = 0; i < object_names.size(); ++i) {
            if (object_names[i] == object_name) {
                movable_object_poses[i] = new_pose;
                break;
            }
        }
    }
    
    // Apply object movement by index (faster)
    void apply_object_movement(int object_index, const SE2State& new_pose) {
        assert(object_index >= 0 && object_index < static_cast<int>(object_names.size()));
        movable_object_poses[object_index] = new_pose;
    }
    
    // Get object pose by name
    const SE2State* get_object_pose(const std::string& object_name) const {
        for (size_t i = 0; i < object_names.size(); ++i) {
            if (object_names[i] == object_name) {
                return &movable_object_poses[i];
            }
        }
        return nullptr;
    }
    
    // Get object pose by index
    const SE2State& get_object_pose(int object_index) const {
        assert(object_index >= 0 && object_index < static_cast<int>(object_names.size()));
        return movable_object_poses[object_index];
    }
    
    // Set robot pose
    void set_robot_pose(const SE2State& pose) {
        robot_pose = pose;
    }
    
    // Validation
    bool is_valid() const {
        return !object_names.empty();
    }
    
    // Create a hash for state comparison (useful for cycle detection)
    std::size_t compute_hash() const {
        std::size_t h1 = std::hash<double>{}(robot_pose.x);
        std::size_t h2 = std::hash<double>{}(robot_pose.y);
        std::size_t h3 = std::hash<double>{}(robot_pose.theta);
        
        std::size_t result = h1 ^ (h2 << 1) ^ (h3 << 2);
        
        for (size_t i = 0; i < object_names.size(); ++i) {
            std::size_t h_obj = std::hash<double>{}(movable_object_poses[i].x) ^
                               (std::hash<double>{}(movable_object_poses[i].y) << 1) ^
                               (std::hash<double>{}(movable_object_poses[i].theta) << 2);
            result ^= h_obj << (i + 3);
        }
        
        return result;
    }
};

} // namespace namo