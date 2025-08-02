# NAMO Skill Usage Guide for High-Level Planners

## Overview

The NAMO Push Skill provides a standardized interface for high-level planners to control object manipulation without knowing internal NAMO system details. This guide shows how any planner (PDDL, behavior trees, RL policies, etc.) can use the skill.

## Quick Start

```cpp
#include "skills/namo_push_skill.hpp"
#include "environment/namo_environment.hpp"

// 1. Setup environment and skill
NAMOEnvironment env("data/scene.xml", false);
NAMOPushSkill skill(env);

// 2. Create skill parameters
std::map<std::string, SkillParameterValue> params = {
    {"object_name", std::string("box_1")},
    {"target_pose", SE2State(2.0, 1.5, 0.0)}  // x, y, theta
};

// 3. Check if skill is applicable
if (skill.is_applicable(params)) {
    // 4. Execute skill
    auto result = skill.execute(params);
    
    // 5. Check result
    if (result.success) {
        std::cout << "Object moved successfully!" << std::endl;
    } else {
        std::cout << "Failed: " << result.failure_reason << std::endl;
    }
}
```

## Skill Interface API

### Core Methods

#### 1. Parameter Schema Discovery
```cpp
auto schema = skill.get_parameter_schema();
// Returns: std::map<std::string, ParameterSchema>
// Use this to understand what parameters the skill accepts
```

#### 2. Applicability Checking
```cpp
bool applicable = skill.is_applicable(parameters);
// Returns: true if skill can potentially succeed with these parameters
// Use this for action selection in planning
```

#### 3. Precondition Validation
```cpp
auto unmet = skill.check_preconditions(parameters);
// Returns: std::vector<std::string> of unmet preconditions
// Use this for detailed failure diagnosis
```

#### 4. Duration Estimation
```cpp
auto duration = skill.estimate_duration(parameters);
// Returns: std::chrono::milliseconds estimated execution time
// Use this for temporal planning and scheduling
```

#### 5. World State Observation
```cpp
auto state = skill.get_world_state();
// Returns: std::map<std::string, SkillParameterValue> of current world state
// Use this for state monitoring and goal checking
```

#### 6. Skill Execution
```cpp
auto result = skill.execute(parameters);
// Returns: SkillResult with success/failure and detailed outputs
// Use this to actually perform the manipulation
```

## Parameter Reference

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `object_name` | `std::string` | Name of movable object to push | `"box_1"` |
| `target_pose` | `SE2State` | Target pose (x, y, theta) in meters/radians | `SE2State(2.0, 1.5, 0.0)` |

### Optional Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `robot_goal` | `SE2State` | Robot goal for early termination | None |
| `tolerance` | `double` | Goal tolerance in meters | `0.01` |
| `max_attempts` | `int` | Maximum planning attempts | `3` |

### Parameter Creation Examples

```cpp
// Basic parameters
std::map<std::string, SkillParameterValue> params = {
    {"object_name", std::string("obstacle_1_movable")},
    {"target_pose", SE2State(1.0, 2.0, M_PI/4)}
};

// With optional parameters
params["tolerance"] = 0.02;
params["max_attempts"] = 5;
params["robot_goal"] = SE2State(3.0, 3.0, 0.0);
```

## Integration Patterns

### 1. PDDL Planner Integration

```cpp
class PDDLExecutor {
    NAMOPushSkill& skill_;
    
public:
    bool execute_action(const PDDLAction& action) {
        if (action.name == "push") {
            std::map<std::string, SkillParameterValue> params = {
                {"object_name", action.object},
                {"target_pose", SE2State(action.x, action.y, action.theta)}
            };
            
            auto result = skill_.execute(params);
            return result.success;
        }
        return false;
    }
};
```

### 2. Behavior Tree Integration

```cpp
class PushObjectNode : public BT::ActionNodeBase {
    NAMOPushSkill& skill_;
    
public:
    BT::NodeStatus tick() override {
        // Get parameters from blackboard
        auto object = getInput<std::string>("object_name");
        auto target = getInput<SE2State>("target_pose");
        
        std::map<std::string, SkillParameterValue> params = {
            {"object_name", object.value()},
            {"target_pose", target.value()}
        };
        
        if (!skill_.is_applicable(params)) {
            return BT::NodeStatus::FAILURE;
        }
        
        auto result = skill_.execute(params);
        return result.success ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }
};
```

### 3. RL Policy Integration

```cpp
class RLEnvironment {
    NAMOPushSkill& skill_;
    
public:
    bool step(const Action& action) {
        std::map<std::string, SkillParameterValue> params = {
            {"object_name", action.object_id},
            {"target_pose", SE2State(action.x, action.y, action.theta)}
        };
        
        // Check applicability for reward shaping
        if (!skill_.is_applicable(params)) {
            return false;  // Invalid action
        }
        
        // Execute and get reward based on result
        auto result = skill_.execute(params);
        compute_reward(result);
        
        return result.success;
    }
};
```

### 4. Task Planning Integration

```cpp
class TaskPlanner {
    NAMOPushSkill& skill_;
    
public:
    std::vector<SkillResult> execute_plan(const Plan& plan) {
        std::vector<SkillResult> results;
        
        for (const auto& step : plan.steps) {
            if (step.skill_name == "namo_push") {
                // Check preconditions before execution
                auto unmet = skill_.check_preconditions(step.parameters);
                if (!unmet.empty()) {
                    // Handle precondition failures
                    continue;
                }
                
                // Execute with timing
                auto start = std::chrono::high_resolution_clock::now();
                auto result = skill_.execute(step.parameters);
                auto elapsed = std::chrono::high_resolution_clock::now() - start;
                
                result.actual_duration = elapsed;
                results.push_back(result);
                
                if (!result.success) {
                    break;  // Stop execution on failure
                }
            }
        }
        
        return results;
    }
};
```

## Error Handling Patterns

### 1. Graceful Degradation

```cpp
auto result = skill.execute(params);
if (!result.success) {
    if (result.failure_reason.find("planning failed") != std::string::npos) {
        // Try with relaxed parameters
        params["tolerance"] = 0.05;  // Increase tolerance
        params["max_attempts"] = 10;  // More attempts
        result = skill.execute(params);
    }
}
```

### 2. Precondition Monitoring

```cpp
auto unmet = skill.check_preconditions(params);
for (const auto& condition : unmet) {
    if (condition.find("too far from robot") != std::string::npos) {
        // Navigate robot closer first
        navigate_robot_to_object(object_name);
    }
}
```

### 3. State Validation

```cpp
auto initial_state = skill.get_world_state();
auto result = skill.execute(params);

if (result.success) {
    auto final_state = skill.get_world_state();
    // Validate state change
    if (poses_approximately_equal(
        std::get<SE2State>(final_state[object_name + "_pose"]),
        std::get<SE2State>(params.at("target_pose"))
    )) {
        // Success confirmed
    }
}
```

## Performance Considerations

### 1. Batch Precondition Checking

```cpp
// Check multiple actions before committing to one
std::vector<std::map<std::string, SkillParameterValue>> candidates;
// ... populate candidates

for (const auto& params : candidates) {
    if (skill.is_applicable(params)) {
        auto duration = skill.estimate_duration(params);
        // Select based on estimated cost
    }
}
```

### 2. State Caching

```cpp
class StateCachingPlanner {
    std::map<std::string, SkillParameterValue> cached_state_;
    std::chrono::time_point<std::chrono::high_resolution_clock> cache_time_;
    
public:
    auto get_current_state() {
        auto now = std::chrono::high_resolution_clock::now();
        if (now - cache_time_ > std::chrono::milliseconds(100)) {
            cached_state_ = skill_.get_world_state();
            cache_time_ = now;
        }
        return cached_state_;
    }
};
```

### 3. Timeout Management

```cpp
NAMOPushSkill::Config config;
config.planning_timeout = std::chrono::seconds(10);  // Set reasonable timeout
config.max_planning_attempts = 5;
NAMOPushSkill skill(env, config);
```

## Advanced Usage

### 1. Multi-Object Manipulation

```cpp
// Get all movable objects
auto world_state = skill.get_world_state();
for (const auto& [key, value] : world_state) {
    if (key.ends_with("_pose")) {
        std::string object_name = key.substr(0, key.find("_pose"));
        // Plan manipulation for each object
    }
}
```

### 2. Goal Decomposition

```cpp
bool move_object_through_waypoints(const std::vector<SE2State>& waypoints) {
    for (size_t i = 0; i < waypoints.size(); i++) {
        std::map<std::string, SkillParameterValue> params = {
            {"object_name", object_name},
            {"target_pose", waypoints[i]}
        };
        
        auto result = skill.execute(params);
        if (!result.success) {
            return false;
        }
    }
    return true;
}
```

### 3. Reactive Execution

```cpp
class ReactiveController {
public:
    void execute_with_monitoring() {
        auto result = skill.execute(params);
        
        // Monitor during execution
        while (!result.success && should_retry()) {
            // Adapt parameters based on current state
            auto current_state = skill.get_world_state();
            params = adapt_parameters(params, current_state);
            result = skill.execute(params);
        }
    }
};
```

## Best Practices

1. **Always check applicability** before execution to avoid unnecessary failures
2. **Use precondition checking** for detailed failure diagnosis
3. **Monitor world state** before and after execution for validation
4. **Handle timeouts gracefully** with reasonable configuration
5. **Cache state observations** when making multiple queries
6. **Use duration estimates** for temporal planning and resource allocation
7. **Implement fallback strategies** for common failure modes
8. **Validate goal achievement** after successful execution

## Common Pitfalls

1. **Not checking applicability first** - leads to predictable failures
2. **Ignoring precondition failures** - miss opportunities for recovery
3. **Using absolute coordinates** - remember poses are in world frame
4. **Ignoring robot constraints** - skill respects robot reachability limits
5. **Not handling timeouts** - planning can take significant time for complex scenes
6. **Assuming deterministic execution** - physics simulation introduces variability

## Integration Checklist

- [ ] Include required headers (`skills/namo_push_skill.hpp`)
- [ ] Initialize NAMOEnvironment with appropriate scene
- [ ] Create skill instance with proper configuration
- [ ] Implement parameter creation for your action representation
- [ ] Add applicability checking to action selection
- [ ] Implement error handling for skill failures
- [ ] Add state monitoring for goal validation
- [ ] Configure appropriate timeouts and tolerances
- [ ] Test with realistic multi-object scenarios
- [ ] Profile performance for your planning frequency

This skill interface provides a clean, powerful abstraction that allows any high-level planner to leverage the sophisticated NAMO MPC system while maintaining complete separation of concerns.