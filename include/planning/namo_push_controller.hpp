#pragma once

#include "core/types.hpp"
#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"
#include <array>
#include <memory>

namespace namo {

/**
 * @brief Push motion primitive for NAMO planning
 * 
 * Optimized for zero-allocation with fixed-size containers
 */
struct PushPrimitive {
    std::array<double, 2> position;
    std::array<double, 4> quaternion;
    int edge_idx;
    std::array<double, 2> edge_point;
    std::array<double, 2> mid_point;
    int push_steps;
    double scaling;
    
    PushPrimitive() : edge_idx(0), push_steps(0), scaling(0.5) {
        position.fill(0.0);
        quaternion = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
        edge_point.fill(0.0);
        mid_point.fill(0.0);
    }
};

/**
 * @brief Push state for tracking during execution
 */
struct PushState {
    int edge_idx;
    std::array<double, 2> initial_edge_point;
    std::array<double, 2> initial_mid_point;
    std::array<double, 2> current_edge_point;
    std::array<double, 2> current_mid_point;
    
    PushState() : edge_idx(0) {
        initial_edge_point.fill(0.0);
        initial_mid_point.fill(0.0);
        current_edge_point.fill(0.0);
        current_mid_point.fill(0.0);
    }
};

/**
 * @brief Action step for NAMO planning
 */
struct NAMOAction {
    std::string object_name;
    int edge_idx;
    int push_steps;
    std::array<double, 7> goal_state; // x, y, z, qw, qx, qy, qz
    
    NAMOAction() : edge_idx(0), push_steps(0) {
        goal_state.fill(0.0);
        goal_state[3] = 1.0; // w component of quaternion
    }
};

/**
 * @brief High-performance NAMO push controller
 * 
 * Features:
 * - Zero-allocation runtime performance
 * - Pre-allocated memory pools for primitives
 * - Direct MuJoCo integration
 * - Incremental wavefront integration
 */
class NAMOPushController {
public:
    static constexpr size_t MAX_EDGE_POINTS = 64;  // Public for external use
    
private:
    static constexpr size_t MAX_PRIMITIVES = 1000;
    static constexpr size_t MAX_TRAJECTORY_POINTS = 100;
    
    // Environment and planner references
    NAMOEnvironment& env_;
    WavefrontPlanner& planner_;
    
    // Pre-allocated memory pools using standard containers for complex types
    std::array<PushPrimitive, MAX_PRIMITIVES> primitive_pool_;
    std::array<PushState, MAX_TRAJECTORY_POINTS> state_pool_;
    std::array<std::array<double, 2>, MAX_EDGE_POINTS> edge_point_pool_;
    std::array<std::array<double, 2>, MAX_EDGE_POINTS> mid_point_pool_;
    
    // Size tracking for pools
    size_t primitive_count_ = 0;
    size_t state_count_ = 0;
    size_t edge_point_count_ = 0;
    size_t mid_point_count_ = 0;
    
    // Configuration parameters
    int default_push_steps_;
    int control_steps_per_push_;
    double force_scaling_;
    int points_per_edge_;
    std::array<double, 3> robot_size_;
    
public:
    /**
     * @brief Constructor
     * 
     * @param env NAMO environment reference
     * @param planner Incremental wavefront planner reference
     * @param push_steps Default number of push steps
     * @param control_steps Control steps per push step
     * @param scaling Force scaling factor
     * @param points_per_edge Number of approach points per edge (default 3)
     */
    NAMOPushController(NAMOEnvironment& env, 
                      WavefrontPlanner& planner,
                      int push_steps = 20,
                      int control_steps = 500,
                      double scaling = 0.5,
                      int points_per_edge = 3);
    
    /**
     * @brief Generate edge points for pushing an object
     * 
     * @param object_name Name of the object to push
     * @param edge_points Output container for edge points (must have space for at least 8 points)
     * @param mid_points Output container for mid points (must have space for at least 8 points)
     * @return Number of edge points generated
     */
    size_t generate_edge_points(const std::string& object_name,
                               std::array<std::array<double, 2>, MAX_EDGE_POINTS>& edge_points,
                               std::array<std::array<double, 2>, MAX_EDGE_POINTS>& mid_points,
                               size_t& edge_count,
                               size_t& mid_count);
    
    /**
     * @brief Execute a push primitive
     * 
     * @param object_name Object to push
     * @param edge_idx Edge index to push from
     * @param push_steps Number of push steps
     * @return true if push executed successfully
     */
    bool execute_push_primitive(const std::string& object_name,
                               int edge_idx,
                               int push_steps);
    
    /**
     * @brief Execute a complete NAMO action
     * 
     * @param action Action to execute
     * @return true if action completed successfully
     */
    bool execute_action(const NAMOAction& action);
    
    /**
     * @brief Check if a push action is valid (can reach goal)
     * 
     * @param object_name Object to push
     * @param edge_idx Edge to push from
     * @param goal_state Desired final state
     * @return true if action is valid
     */
    bool is_push_valid(const std::string& object_name,
                      int edge_idx,
                      const std::array<double, 7>& goal_state);
    
    /**
     * @brief Get reachable objects from current robot position
     * 
     * @param reachable_objects Output container for reachable object names
     * @param max_objects Maximum number of objects to check
     * @return Number of reachable objects
     */
    size_t get_reachable_objects(std::array<std::string, 20>& reachable_objects, 
                               size_t& reachable_count,
                               size_t max_objects = 20);
    
    /**
     * @brief Get reachable edge indices for a specific object
     * 
     * @param object_name Name of the object to check
     * @return Vector of reachable edge indices (0-63)
     */
    std::vector<int> get_reachable_edge_indices(const std::string& object_name);
    
    /**
     * @brief Update push state based on current object position
     */
    void update_push_state(PushState& state,
                          const std::array<double, 3>& obj_pos,
                          const std::array<double, 3>& obj_size,
                          const std::array<double, 4>& obj_quat);
    
    /**
     * @brief Compute control forces for pushing
     */
    std::array<double, 2> compute_push_control(const PushState& state);
    
    /**
     * @brief Get statistics about memory usage
     */
    void get_memory_stats(size_t& primitives_used, size_t& states_used);
    
private:
    /**
     * @brief Generate edge points around a rectangular object
     */
    void generate_rectangular_edge_points(const std::array<double, 3>& obj_pos,
                                         const std::array<double, 3>& obj_size,
                                         const std::array<double, 4>& obj_quat,
                                         std::array<std::array<double, 2>, MAX_EDGE_POINTS>& edge_points,
                                         std::array<std::array<double, 2>, MAX_EDGE_POINTS>& mid_points,
                                         size_t& edge_count,
                                         size_t& mid_count);
    
    /**
     * @brief Convert quaternion to yaw angle (rotation around Z axis)
     */
    double quaternion_to_yaw(const std::array<double, 4>& quaternion);
    
    /**
     * @brief Transform a point by rotation and translation
     */
    std::array<double, 2> transform_point(const std::array<double, 2>& point,
                                         const std::array<double, 3>& translation,
                                         double rotation_angle);
};

} // namespace namo