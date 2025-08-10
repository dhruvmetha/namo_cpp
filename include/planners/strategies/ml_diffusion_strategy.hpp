#pragma once

#include "planners/strategies/selection_strategy.hpp"
#include <string>
#include <map>

namespace namo {

// Placeholder for ML-based diffusion strategy using ZMQ communication
// Based on legacy interface_namo.cpp Strategy 1 (lines 471-660)
class MLDiffusionStrategy : public SelectionStrategy {
private:
    std::string zmq_endpoint_;
    int timeout_ms_;
    bool fallback_to_random_;
    
    // ZMQ communication placeholders
    bool sendStateToML(const NAMOEnvironment& env, 
                      const std::vector<std::string>& reachable_objects,
                      const SE2State& robot_goal);
    SelectionResult receiveMLDecision();
    SelectionResult fallbackToRandom(const NAMOEnvironment& env,
                                   const std::vector<std::string>& reachable_objects,
                                   const SE2State& robot_goal);

public:
    MLDiffusionStrategy(const std::string& zmq_endpoint = "tcp://localhost:5555",
                       int timeout_ms = 1000,
                       bool fallback_to_random = true);
    
    SelectionResult selectObjectAndGoal(
        const NAMOEnvironment& env,
        const std::vector<std::string>& reachable_objects,
        const SE2State& robot_goal
    ) override;
    
    std::string getStrategyName() const override {
        return "MLDiffusion";
    }
    
    void configure(const std::map<std::string, double>& params) override;
    
    // ML-specific configuration
    void setZMQEndpoint(const std::string& endpoint) { zmq_endpoint_ = endpoint; }
    void setTimeout(int timeout_ms) { timeout_ms_ = timeout_ms; }
    void setFallbackEnabled(bool enabled) { fallback_to_random_ = enabled; }
};

} // namespace namo