#pragma once

#include "planning/greedy_planner.hpp"  // For SE2State
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <variant>
#include <optional>

namespace namo {

/**
 * @brief Type-safe skill parameter using std::variant
 * 
 * Modern C++ approach - no manual type tracking
 */
using SkillParameterValue = std::variant<
    std::string,
    double, 
    int, 
    bool,
    SE2State,
    std::array<double, 7>,  // 3D pose
    std::vector<double>
>;

/**
 * @brief Skill execution result with proper error handling
 */
struct SkillResult {
    bool success = false;
    std::string skill_name;
    std::map<std::string, SkillParameterValue> outputs;
    std::string failure_reason;
    std::chrono::milliseconds execution_time{0};
    
    // Type-safe output access
    template<typename T>
    std::optional<T> get_output(const std::string& key) const {
        auto it = outputs.find(key);
        if (it != outputs.end()) {
            if (auto* value = std::get_if<T>(&it->second)) {
                return *value;
            }
        }
        return std::nullopt;
    }
    
    bool has_output(const std::string& key) const {
        return outputs.find(key) != outputs.end();
    }
};

/**
 * @brief Parameter schema definition
 */
struct ParameterSchema {
    enum Type { STRING, DOUBLE, INT, BOOL, POSE_2D, POSE_3D, ARRAY_DOUBLE };
    Type type;
    std::string description;
    bool required = true;
    SkillParameterValue default_value;
    
    ParameterSchema(Type t, const std::string& desc, bool req = true) 
        : type(t), description(desc), required(req) {}
    
    ParameterSchema(Type t, const std::string& desc, const SkillParameterValue& def_val)
        : type(t), description(desc), required(false), default_value(def_val) {}
    
    ParameterSchema(Type t, const std::string& desc, bool req, const SkillParameterValue& def_val)
        : type(t), description(desc), required(req), default_value(def_val) {}
};

/**
 * @brief Clean skill interface without inheritance complexity
 */
class ManipulationSkill {
public:
    virtual ~ManipulationSkill() = default;
    
    // Core interface
    virtual std::string get_name() const = 0;
    virtual std::string get_description() const = 0;
    virtual std::map<std::string, ParameterSchema> get_parameter_schema() const = 0;
    
    // Execution interface  
    virtual bool is_applicable(const std::map<std::string, SkillParameterValue>& parameters) const = 0;
    virtual std::chrono::milliseconds estimate_duration(const std::map<std::string, SkillParameterValue>& parameters) const = 0;
    virtual SkillResult execute(const std::map<std::string, SkillParameterValue>& parameters) = 0;
    
    // State interface
    virtual std::map<std::string, SkillParameterValue> get_world_state() const = 0;
    virtual std::vector<std::string> check_preconditions(const std::map<std::string, SkillParameterValue>& parameters) const = 0;
    
protected:
    // Helper for parameter validation
    bool validate_parameters(const std::map<std::string, SkillParameterValue>& params, 
                           std::string& error) const {
        auto schema = get_parameter_schema();
        
        // Check required parameters
        for (const auto& [key, spec] : schema) {
            if (spec.required && params.find(key) == params.end()) {
                error = "Missing required parameter: " + key;
                return false;
            }
        }
        
        // Check parameter types
        for (const auto& [key, value] : params) {
            auto schema_it = schema.find(key);
            if (schema_it == schema.end()) {
                error = "Unknown parameter: " + key;
                return false;
            }
            
            // Type checking would go here
            // (simplified for brevity)
        }
        
        return true;
    }
};

} // namespace namo