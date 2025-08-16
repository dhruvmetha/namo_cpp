#include "planners/strategies/ml_diffusion_strategy.hpp"
#include "planners/strategies/random_selection_strategy.hpp"
#include <iostream>

namespace namo {

MLDiffusionStrategy::MLDiffusionStrategy(const std::string& zmq_endpoint,
                                        int timeout_ms,
                                        bool fallback_to_random)
    : zmq_endpoint_(zmq_endpoint)
    , timeout_ms_(timeout_ms)
    , fallback_to_random_(fallback_to_random)
{
    // TODO: Initialize ZMQ context and socket
    // std::cout << "MLDiffusionStrategy created with endpoint: " << zmq_endpoint_ << std::endl;
    // std::cout << "WARNING: ZMQ communication not yet implemented, will fallback to random" << std::endl;
}

void MLDiffusionStrategy::configure(const std::map<std::string, double>& params) {
    auto it = params.find("timeout_ms");
    if (it != params.end()) timeout_ms_ = static_cast<int>(it->second);
    
    it = params.find("fallback_to_random");
    if (it != params.end()) fallback_to_random_ = (it->second > 0.5);
}

SelectionResult MLDiffusionStrategy::selectObjectAndGoal(
    const NAMOEnvironment& env,
    const std::vector<std::string>& reachable_objects,
    const SE2State& robot_goal
) {
    // TODO: Implement full ZMQ communication following legacy interface_namo.cpp:471-660
    
    // Placeholder implementation that shows the intended structure:
    
    // 1. Send state to ML server
    bool ml_communication_success = sendStateToML(env, reachable_objects, robot_goal);
    
    if (ml_communication_success) {
        // 2. Receive ML decision
        SelectionResult ml_result = receiveMLDecision();
        
        if (ml_result.success) {
            return ml_result;
        }
    }
    
    // 3. Fallback to random strategy if ML fails
    if (fallback_to_random_) {
        // std::cout << "ML strategy failed, falling back to random selection" << std::endl;
        return fallbackToRandom(env, reachable_objects, robot_goal);
    }
    
    return SelectionResult::failure();
}

bool MLDiffusionStrategy::sendStateToML(const NAMOEnvironment& env,
                                       const std::vector<std::string>& reachable_objects,
                                       const SE2State& robot_goal) {
    // TODO: Implement ZMQ state encoding and transmission
    // Based on legacy interface_namo.cpp:498-534 createStateJson()
    
    /*
    Intended implementation:
    
    1. Create JSON state message:
       - msg_type: "decision_req"
       - robot_goal: [x, y, theta]
       - reachable_objects: [object_names...]
       - object_states: {name: {position: [x,y,z], quaternion: [w,x,y,z], size: [w,h,d]}}
       - environment_bounds: {min_x, max_x, min_y, max_y}
    
    2. Send via ZMQ REQ socket with timeout
    
    3. Return success/failure
    */
    
    // std::cout << "TODO: Sending state to ML server at " << zmq_endpoint_ << std::endl;
    return false;  // Not implemented yet
}

SelectionResult MLDiffusionStrategy::receiveMLDecision() {
    // TODO: Implement ZMQ response parsing
    // Based on legacy interface_namo.cpp:535-600
    
    /*
    Intended implementation:
    
    1. Receive JSON response from ML server
    2. Parse response:
       - selected_object: object name
       - goal_cluster: array of [x, y, theta] poses
    3. Validate goals (distance < 1.0m from object, bounds checking)
    4. Return first valid goal or failure
    */
    
    // std::cout << "TODO: Receiving ML decision from server" << std::endl;
    return SelectionResult::failure();  // Not implemented yet
}

SelectionResult MLDiffusionStrategy::fallbackToRandom(const NAMOEnvironment& env,
                                                     const std::vector<std::string>& reachable_objects,
                                                     const SE2State& robot_goal) {
    // Use random strategy as fallback
    RandomSelectionStrategy random_strategy;
    return random_strategy.selectObjectAndGoal(env, reachable_objects, robot_goal);
}

} // namespace namo