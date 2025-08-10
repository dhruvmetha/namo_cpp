#pragma once

#include "core/types.hpp"
#include "environment/namo_environment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace namo {

// Selection result containing object name and target pose
struct SelectionResult {
    std::string object_name;
    SE2State target_pose;
    bool success;
    
    SelectionResult(const std::string& obj, const SE2State& pose, bool s = true)
        : object_name(obj), target_pose(pose), success(s) {}
        
    static SelectionResult failure() {
        return SelectionResult("", SE2State(0, 0, 0), false);
    }
};

// Abstract strategy interface for object and goal selection
class SelectionStrategy {
public:
    virtual ~SelectionStrategy() = default;
    
    // Select object and goal given current environment state
    virtual SelectionResult selectObjectAndGoal(
        const NAMOEnvironment& env,
        const std::vector<std::string>& reachable_objects,
        const SE2State& robot_goal
    ) = 0;
    
    // Optional: Get strategy name for logging/debugging
    virtual std::string getStrategyName() const = 0;
    
    // Optional: Strategy-specific configuration
    virtual void configure(const std::map<std::string, double>& params) {}
};

// Factory for creating selection strategies
class SelectionStrategyFactory {
public:
    enum class StrategyType {
        RANDOM,
        ML_DIFFUSION,
        REGION_WAVEFRONT
    };
    
    static std::unique_ptr<SelectionStrategy> create(StrategyType type);
    static std::unique_ptr<SelectionStrategy> create(const std::string& strategy_name);
};

} // namespace namo