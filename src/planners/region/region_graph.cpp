#include "planners/region/region_graph.hpp"
#include "environment/namo_environment.hpp"
#include <cmath>

namespace namo {

// Note: All old Region methods have been removed as the new Region is lightweight
// The heavy Region calculations are handled directly in RegionAnalyzer during discovery

void LightweightState::initialize_from_environment(NAMOEnvironment& env) {
    // Clear existing data
    object_names.clear();
    
    // Get robot state
    const ObjectState* robot_state = env.get_robot_state();
    if (robot_state) {
        robot_pose.x = robot_state->position[0];
        robot_pose.y = robot_state->position[1];
        // Convert quaternion to 2D rotation (assuming rotation around Z-axis)
        double qw = robot_state->quaternion[0];
        double qz = robot_state->quaternion[3];
        robot_pose.theta = 2.0 * std::atan2(qz, qw);
    }
    
    // Get movable object states
    const auto& movable_objects = env.get_movable_objects();
    size_t num_movable = env.get_num_movable();
    
    for (size_t i = 0; i < num_movable; ++i) {
        const ObjectInfo& obj_info = movable_objects[i];
        const ObjectState* obj_state = env.get_object_state(obj_info.name);
        
        if (obj_state) {
            object_names.push_back(obj_info.name);
            
            // Convert object state to SE2
            SE2State obj_pose;
            obj_pose.x = obj_state->position[0];
            obj_pose.y = obj_state->position[1];
            // Convert quaternion to 2D rotation (assuming rotation around Z-axis)
            double qw = obj_state->quaternion[0];
            double qz = obj_state->quaternion[3];
            obj_pose.theta = 2.0 * std::atan2(qz, qw);
            
            movable_object_poses[i] = obj_pose;
        }
    }
}

} // namespace namo