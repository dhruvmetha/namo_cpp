#pragma once

#include <array>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cassert>
#include <cmath>

namespace namo {

// Forward declarations
class NAMOEnvironment;
class WavefrontPlanner;
class NAMOPlanner;
class PushController;

// Fixed-size containers to avoid runtime allocations
template<size_t MAX_SIZE>
class FixedVector {
private:
    std::array<double, MAX_SIZE> data_;
    size_t size_ = 0;
    
public:
    FixedVector() = default;
    
    void push_back(double val) { 
        assert(size_ < MAX_SIZE); 
        data_[size_++] = val; 
    }
    
    void clear() { size_ = 0; }
    
    void resize(size_t new_size) {
        assert(new_size <= MAX_SIZE);
        size_ = new_size;
    }
    
    double& operator[](size_t i) { 
        assert(i < size_);
        return data_[i]; 
    }
    
    const double& operator[](size_t i) const { 
        assert(i < size_);
        return data_[i]; 
    }
    
    size_t size() const { return size_; }
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    
    // STL-like iterators
    double* begin() { return data_.data(); }
    double* end() { return data_.data() + size_; }
    const double* begin() const { return data_.data(); }
    const double* end() const { return data_.data() + size_; }
};

// Commonly used fixed-size types
using State = FixedVector<10>;        // Robot state (x, y, theta, velocities, etc.)
using Control = FixedVector<6>;       // Control inputs
using Pose2D = std::array<double, 3>; // x, y, theta
using Pose3D = std::array<double, 7>; // x, y, z, qw, qx, qy, qz

// SE2 state representation for 2D planning
struct SE2State {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
    
    SE2State() = default;
    SE2State(double x_, double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
    
    // Normalize angle to [-π, π]
    void normalize_angle() {
        while (theta > M_PI) theta -= 2.0 * M_PI;
        while (theta < -M_PI) theta += 2.0 * M_PI;
    }
    
    // Convert to Pose2D array
    Pose2D to_pose2d() const { return {x, y, theta}; }
    
    // Create from Pose2D array
    static SE2State from_pose2d(const Pose2D& pose) {
        return SE2State(pose[0], pose[1], pose[2]);
    }
};

// Object information structure
struct ObjectInfo {
    std::string name;
    std::string body_name;
    int body_id = -1;
    int geom_id = -1;
    bool is_static = false;
    
    // Geometric properties
    std::array<double, 3> position = {0.0, 0.0, 0.0};
    std::array<double, 3> size = {1.0, 1.0, 1.0};
    std::array<double, 4> quaternion = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
    
    // Symmetry information for planning
    int symmetry_rotations = 4;
    
    // Default constructor
    ObjectInfo() = default;
    
    // Constructor with size (auto-detects symmetry)
    ObjectInfo(const std::array<double, 3>& obj_size) : size(obj_size) {
        // If x and y dimensions are within 5% of each other, assume 4-way symmetry
        double size_ratio = std::max(size[0], size[1]) / std::min(size[0], size[1]);
        symmetry_rotations = (size_ratio < 1.05) ? 4 : 2;
    }
};

// Object state structure (dynamic information)
struct ObjectState {
    std::string name;
    std::array<double, 3> position = {0.0, 0.0, 0.0};
    std::array<double, 4> quaternion = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> size = {1.0, 1.0, 1.0};
    std::array<double, 3> linear_vel = {0.0, 0.0, 0.0};
    std::array<double, 3> angular_vel = {0.0, 0.0, 0.0};
    
    ObjectState() = default;
    ObjectState(const std::string& obj_name) : name(obj_name) {}
};

// Grid footprint for change detection
struct GridFootprint {
    int min_x = 0, max_x = -1;
    int min_y = 0, max_y = -1;
    
    // Pre-allocated storage for occupied cells
    static constexpr size_t MAX_CELLS = 200000;
    std::array<std::pair<int, int>, MAX_CELLS> cells;
    size_t num_cells = 0;
    
    void clear() {
        min_x = 0; max_x = -1;
        min_y = 0; max_y = -1;
        num_cells = 0;
    }
    
    void add_cell(int x, int y) {
        assert(num_cells < MAX_CELLS);
        cells[num_cells++] = {x, y};
        
        // Update bounding box
        if (num_cells == 1) {
            min_x = max_x = x;
            min_y = max_y = y;
        } else {
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
    }
    
    bool is_empty() const { return num_cells == 0; }
};

// Object snapshot for change detection
struct RotatingObjectSnapshot {
    std::array<double, 3> position = {0.0, 0.0, 0.0};
    std::array<double, 4> quaternion = {1.0, 0.0, 0.0, 0.0};
    double yaw_angle = 0.0;
    
    GridFootprint cached_footprint;
    bool needs_update = true;
    
    // Motion analysis
    bool position_changed = false;
    bool rotation_changed = false;
    double rotation_delta = 0.0;
    
    void update_from_state(const ObjectState& state) {
        for (int i = 0; i < 3; i++) position[i] = state.position[i];
        for (int i = 0; i < 4; i++) quaternion[i] = state.quaternion[i];
        
        // Update yaw angle (assuming planar motion)
        yaw_angle = std::atan2(2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                              1.0 - 2.0 * (quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]));
    }
};

// Action step for NAMO planning
struct ActionStepMPC {
    std::string object_name;
    State goal_state;
    State final_state;
    
    // Pre-allocated trajectory storage
    static constexpr size_t MAX_TRAJECTORY_POINTS = 100;
    std::array<State, MAX_TRAJECTORY_POINTS> trajectory;
    size_t trajectory_length = 0;
    
    void clear() {
        object_name.clear();
        goal_state.clear();
        final_state.clear();
        trajectory_length = 0;
    }
    
    void add_trajectory_point(const State& state) {
        assert(trajectory_length < MAX_TRAJECTORY_POINTS);
        trajectory[trajectory_length++] = state;
    }
};

// Grid change tracking
struct GridChange {
    int x, y;
    bool became_obstacle;
    
    GridChange() = default;
    GridChange(int grid_x, int grid_y, bool is_obstacle) 
        : x(grid_x), y(grid_y), became_obstacle(is_obstacle) {}
};

// Motion primitive structure
struct MotionPrimitive {
    Control control_input;
    double duration = 0.0;
    State end_state;
    
    // Cached properties for fast lookup
    double cost = 0.0;
    bool is_valid = true;
    
    void clear() {
        control_input.clear();
        duration = 0.0;
        end_state.clear();
        cost = 0.0;
        is_valid = true;
    }
};

// Configuration limits for memory pre-allocation
struct MemoryLimits {
    size_t max_static_objects = 20;
    size_t max_movable_objects = 10;
    size_t max_actions = 100;
    size_t max_motion_primitives = 1000;
    size_t max_planning_nodes = 10000;
    size_t grid_max_width = 2000;
    size_t grid_max_height = 2000;
    
    // Performance thresholds
    double position_threshold = 1e-4;  // 0.1mm
    double rotation_threshold = 1e-3;  // ~0.06 degrees
    double small_rotation_limit = 0.05; // ~3 degrees
};

// Planning statistics
struct PlanningStats {
    int total_iterations = 0;
    int wavefront_updates = 0;
    int object_movements_detected = 0;
    int grid_changes_processed = 0;
    
    double total_planning_time = 0.0;
    double wavefront_time = 0.0;
    double change_detection_time = 0.0;
    
    void reset() {
        total_iterations = 0;
        wavefront_updates = 0;
        object_movements_detected = 0;
        grid_changes_processed = 0;
        total_planning_time = 0.0;
        wavefront_time = 0.0;
        change_detection_time = 0.0;
    }
};

// Utility functions for common operations
namespace utils {

// Normalize angle to [-π, π]
inline double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// Convert quaternion to yaw angle
inline double quaternion_to_yaw(const std::array<double, 4>& quat) {
    return std::atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]));
}

// Convert yaw angle to quaternion
inline std::array<double, 4> yaw_to_quaternion(double yaw) {
    double half_yaw = yaw * 0.5;
    return {std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw)};
}

// Rotate a 2D point around another point
inline std::array<double, 2> rotate_point(double px, double py, 
                                         double cx, double cy, double angle) {
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    double dx = px - cx;
    double dy = py - cy;
    
    return {cx + dx * cos_a - dy * sin_a,
            cy + dx * sin_a + dy * cos_a};
}

// Fast coordinate packing for hash maps
inline uint64_t pack_coords(int x, int y) {
    return (static_cast<uint64_t>(x) << 32) | static_cast<uint64_t>(y);
}

inline std::pair<int, int> unpack_coords(uint64_t key) {
    return {static_cast<int>(key >> 32), static_cast<int>(key & 0xFFFFFFFF)};
}

} // namespace utils

} // namespace namo