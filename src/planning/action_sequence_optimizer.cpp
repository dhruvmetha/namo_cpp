#include "planning/action_sequence_optimizer.hpp"
#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "skills/namo_push_skill.hpp"
#include "config/config_manager.hpp"
#include "planners/high_level_planner.hpp"  // For PlanningResult
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace namo {

//=============================================================================
// ActionSequenceOptimizer Implementation
//=============================================================================

ActionSequenceOptimizer::ActionSequenceOptimizer(std::shared_ptr<ConfigManager> config)
    : config_(config), has_saved_state_(false) {
    
    // Set default strategy based on config
    if (config_ && config_->optimization().enable_sequence_optimization) {
        auto method = static_cast<OptimizationMethod>(config_->optimization().default_method);
        set_optimization_method(method);
    } else {
        // Default to reverse-order strategy
        set_optimization_method(OptimizationMethod::REVERSE_ORDER);
    }
}

void ActionSequenceOptimizer::set_optimization_method(OptimizationMethod method) {
    strategy_ = create_strategy(method, config_);
}

void ActionSequenceOptimizer::set_custom_strategy(std::unique_ptr<OptimizationStrategy> strategy) {
    strategy_ = std::move(strategy);
}

OptimizationMethod ActionSequenceOptimizer::get_current_method() const {
    return strategy_ ? strategy_->get_method_type() : OptimizationMethod::NONE;
}

std::string ActionSequenceOptimizer::get_current_method_name() const {
    return strategy_ ? strategy_->get_method_name() : "None";
}

bool ActionSequenceOptimizer::is_optimization_enabled() const {
    return config_ && config_->optimization().enable_sequence_optimization && strategy_;
}

OptimizationResult ActionSequenceOptimizer::optimize_sequence(
    const std::vector<std::string>& object_sequence,
    const std::map<std::string, SE2State>& target_states,
    const std::map<std::string, SE2State>& final_states,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    OptimizationResult result;
    result.original_sequence_length = object_sequence.size();
    
    if (!is_optimization_enabled()) {
        result.optimization_successful = false;
        result.failure_reason = "Optimization disabled in configuration";
        return result;
    }
    
    if (object_sequence.empty()) {
        result.optimization_successful = true;
        result.minimal_sequence_length = 0;
        return result;
    }
    
    // Convert to ActionStep format
    std::vector<ActionStep> action_steps;
    action_steps.reserve(object_sequence.size());
    
    for (const auto& object_name : object_sequence) {
        auto target_it = target_states.find(object_name);
        auto final_it = final_states.find(object_name);
        
        if (target_it == target_states.end() || final_it == final_states.end()) {
            result.optimization_successful = false;
            result.failure_reason = "Missing state information for object: " + object_name;
            return result;
        }
        
        action_steps.emplace_back(
            object_name,
            target_it->second,
            final_it->second,
            true  // Assume execution was successful if we got here
        );
    }
    
    // Delegate to strategy
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        result = strategy_->optimize(action_steps, robot_goal, env, wavefront_planner, skill);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time);
        result.optimization_time_seconds = duration.count();
        
    } catch (const std::exception& e) {
        result.optimization_successful = false;
        result.failure_reason = "Optimization strategy failed: " + std::string(e.what());
    }
    
    return result;
}

OptimizationResult ActionSequenceOptimizer::optimize_planning_result(
    const PlanningResult& planning_result,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    OptimizationResult result;
    
    if (!planning_result.success || planning_result.objects_pushed.empty()) {
        result.optimization_successful = false;
        result.failure_reason = "Cannot optimize failed or empty planning result";
        return result;
    }
    
    // For now, we need to extract target and final states from the planning result
    // This is a simplified version - in a full implementation, we'd need the planner
    // to store more detailed action information
    std::map<std::string, SE2State> target_states;
    std::map<std::string, SE2State> final_states;
    
    for (const auto& object_name : planning_result.objects_pushed) {
        // Get current object state as final state
        auto objects = env.get_movable_objects();
        bool found = false;
        
        for (size_t i = 0; i < env.get_num_movable(); ++i) {
            if (objects[i].name == object_name) {
                final_states[object_name] = SE2State(
                    objects[i].position[0],
                    objects[i].position[1],
                    0.0  // Simplified - would need actual orientation
                );
                
                // For target state, use final state as approximation
                // In a full implementation, this would come from the planner
                target_states[object_name] = final_states[object_name];
                found = true;
                break;
            }
        }
        
        if (!found) {
            result.optimization_successful = false;
            result.failure_reason = "Could not find object state for: " + object_name;
            return result;
        }
    }
    
    return optimize_sequence(
        planning_result.objects_pushed,
        target_states,
        final_states,
        robot_goal,
        env,
        wavefront_planner,
        skill
    );
}

void ActionSequenceOptimizer::print_optimization_summary(const OptimizationResult& result) const {
    // std::cout << "\n=== Action Sequence Optimization Results ===" << std::endl;
    // std::cout << "Method: " << get_current_method_name() << std::endl;
    // std::cout << "Success: " << (result.optimization_successful ? "YES" : "NO") << std::endl;
    
    if (!result.optimization_successful) {
        // std::cout << "Failure reason: " << result.failure_reason << std::endl;
        return;
    }
    
    // std::cout << "Original sequence length: " << result.original_sequence_length << std::endl;
    // std::cout << "Minimal sequence length: " << result.minimal_sequence_length << std::endl;
    // std::cout << "Reduction: " << (result.original_sequence_length - result.minimal_sequence_length) 
              // << " actions (" << std::fixed << std::setprecision(1)
              // << (100.0 * (result.original_sequence_length - result.minimal_sequence_length) / 
                //   std::max(1, result.original_sequence_length)) << "%)" << std::endl;
    // std::cout << "Sequences tested: " << result.sequences_tested << std::endl;
    // std::cout << "Optimization time: " << std::fixed << std::setprecision(3) 
              // << result.optimization_time_seconds << " seconds" << std::endl;
    // std::cout << "Minimal sequences found: " << result.minimal_sequences.size() << std::endl;
    
    if (!result.minimal_sequences.empty()) {
        // std::cout << "First minimal sequence: ";
        for (size_t i = 0; i < result.minimal_sequences[0].size(); ++i) {
            // std::cout << result.minimal_sequences[0][i];
            if (i < result.minimal_sequences[0].size() - 1) std::cout << " -> ";
        }
        // std::cout << std::endl;
    }
}

bool ActionSequenceOptimizer::test_action_sequence(
    const std::vector<int>& sequence_indices,
    const std::vector<ActionStep>& action_steps,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    // Reset environment to initial state
    env.reset_to_initial_state();
    
    // Execute each action in the sequence
    for (int idx : sequence_indices) {
        if (idx < 0 || idx >= static_cast<int>(action_steps.size())) {
            return false;  // Invalid index
        }
        
        const ActionStep& action = action_steps[idx];
        if (!execute_single_action(action, env, skill)) {
            return false;  // Action execution failed
        }
    }
    
    // Check if robot can reach goal after executing the sequence
    std::array<double, 2> goal_array = {robot_goal.x, robot_goal.y};
    return wavefront_planner.is_goal_reachable(goal_array, 0.3);  // 0.3m tolerance
}

bool ActionSequenceOptimizer::execute_single_action(
    const ActionStep& action,
    NAMOEnvironment& env,
    NAMOPushSkill& skill) {
    
    // Create parameters for skill execution
    std::map<std::string, SkillParameterValue> params;
    params["object_name"] = action.object_name;
    params["target_pose"] = action.final_state;  // Use final_state, not target_state
    
    // Check if skill is applicable
    if (!skill.is_applicable(params)) {
        return false;
    }
    
    // Execute the skill
    auto result = skill.execute(params);
    return result.success;
}

std::unique_ptr<OptimizationStrategy> ActionSequenceOptimizer::create_strategy(
    OptimizationMethod method,
    const std::shared_ptr<ConfigManager>& config) {
    
    double timeout = 30.0;
    int max_length = 20;
    
    if (config) {
        timeout = config->optimization().timeout_seconds;
        max_length = config->optimization().max_sequence_length;
    }
    
    switch (method) {
        case OptimizationMethod::EXHAUSTIVE:
            return std::make_unique<ExhaustiveStrategy>(timeout, max_length);
        case OptimizationMethod::REVERSE_ORDER:
            return std::make_unique<ReverseOrderStrategy>(timeout, max_length);
        case OptimizationMethod::GREEDY_REMOVAL:
            return std::make_unique<GreedyRemovalStrategy>(timeout);
        case OptimizationMethod::NONE:
        default:
            return nullptr;
    }
}

//=============================================================================
// ReverseOrderStrategy Implementation (optimizeActionSequence2)
//=============================================================================

OptimizationResult ReverseOrderStrategy::optimize(
    const std::vector<ActionStep>& action_steps,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    OptimizationResult result;
    result.original_sequence_length = action_steps.size();
    result.sequences_tested = 0;
    
    if (action_steps.empty()) {
        result.optimization_successful = true;
        result.minimal_sequence_length = 0;
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int n = static_cast<int>(action_steps.size());
    
    // Helper lambda for testing sequences
    auto test_sequence = [&](const std::vector<int>& sequence_indices) -> bool {
        // Reset environment to initial state
        env.reset_to_initial_state();
        
        // Execute each action in the sequence
        for (int idx : sequence_indices) {
            if (idx < 0 || idx >= static_cast<int>(action_steps.size())) {
                return false;  // Invalid index
            }
            
            const ActionStep& action = action_steps[idx];
            
            // Create parameters for skill execution
            std::map<std::string, SkillParameterValue> params;
            params["object_name"] = action.object_name;
            params["target_pose"] = action.final_state;  // Use final_state, not target_state
            
            // Check if skill is applicable
            if (!skill.is_applicable(params)) {
                return false;
            }
            
            // Execute the skill
            auto skill_result = skill.execute(params);
            if (!skill_result.success) {
                return false;
            }
        }
        
        // Check if robot can reach goal after executing the sequence
        std::array<double, 2> goal_array = {robot_goal.x, robot_goal.y};
        return wavefront_planner.is_goal_reachable(goal_array, 0.3);  // 0.3m tolerance
    };
    
    // Test sequences of increasing length (1, 2, 3, ..., n)
    for (int seq_length = 1; seq_length <= n && seq_length <= max_sequence_length_; seq_length++) {
        bool found_working_sequence = false;
        
        // Check timeout
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
        if (elapsed > timeout_seconds_) {
            result.optimization_successful = false;
            result.failure_reason = "Optimization timeout exceeded";
            return result;
        }
        
        if (seq_length == 1) {
            // Test just the last action alone
            std::vector<int> sequence = {n - 1};
            result.sequences_tested++;
            
            if (test_sequence(sequence)) {
                // Convert indices to object names
                std::vector<std::string> object_sequence;
                object_sequence.push_back(action_steps[n - 1].object_name);
                
                result.minimal_sequences.push_back(object_sequence);
                result.minimal_indices.push_back(sequence);
                found_working_sequence = true;
            }
        } else {
            // Generate all combinations of (seq_length - 1) actions from first (n-1) actions
            // Always append the last action
            
            std::vector<bool> selector(n - 1, false);
            std::fill(selector.begin(), selector.begin() + (seq_length - 1), true);
            
            do {
                // Check timeout periodically
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                if (elapsed > timeout_seconds_) {
                    result.optimization_successful = false;
                    result.failure_reason = "Optimization timeout exceeded";
                    return result;
                }
                
                // Create sequence from selector + last action
                std::vector<int> sequence;
                for (int i = 0; i < n - 1; i++) {
                    if (selector[i]) {
                        sequence.push_back(i);
                    }
                }
                sequence.push_back(n - 1);  // Always add the last action
                
                result.sequences_tested++;
                
                // Test this sequence
                if (test_sequence(sequence)) {
                    // Convert indices to object names
                    std::vector<std::string> object_sequence;
                    for (int idx : sequence) {
                        object_sequence.push_back(action_steps[idx].object_name);
                    }
                    
                    result.minimal_sequences.push_back(object_sequence);
                    result.minimal_indices.push_back(sequence);
                    found_working_sequence = true;
                }
                
            } while (std::prev_permutation(selector.begin(), selector.end()));
        }
        
        // Stop at first working length (minimal sequences)
        if (found_working_sequence) {
            result.optimization_successful = true;
            result.minimal_sequence_length = seq_length;
            break;
        }
    }
    
    // If no minimal sequence found, optimization failed
    if (result.minimal_sequences.empty()) {
        result.optimization_successful = false;
        result.failure_reason = "No working subsequence found";
        result.minimal_sequence_length = result.original_sequence_length;
    }
    
    return result;
}

//=============================================================================
// Placeholder implementations for other strategies
//=============================================================================

OptimizationResult ExhaustiveStrategy::optimize(
    const std::vector<ActionStep>& action_steps,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    // Placeholder - will implement exhaustive search later
    OptimizationResult result;
    result.optimization_successful = false;
    result.failure_reason = "ExhaustiveStrategy not implemented yet";
    return result;
}

OptimizationResult GreedyRemovalStrategy::optimize(
    const std::vector<ActionStep>& action_steps,
    const SE2State& robot_goal,
    NAMOEnvironment& env,
    WavefrontPlanner& wavefront_planner,
    NAMOPushSkill& skill) {
    
    // Placeholder - will implement greedy removal later
    OptimizationResult result;
    result.optimization_successful = false;
    result.failure_reason = "GreedyRemovalStrategy not implemented yet";
    return result;
}

} // namespace namo