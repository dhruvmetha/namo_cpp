#pragma once

#include <string>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace namo {

/**
 * @brief Fast parameter loader with fallback to simple parser
 * 
 * Uses yaml-cpp if available, otherwise falls back to simple key=value parser
 */
class FastParameterLoader {
private:
#ifdef HAVE_YAML_CPP
    YAML::Node root_;
    mutable std::unordered_map<std::string, YAML::Node> cache_;
#else
    std::unordered_map<std::string, std::string> params_;
#endif
    
public:
    FastParameterLoader(const std::string& filename);
    
    // Generic get method with type conversion
    template<typename T>
    T get(const std::string& key) const;
    
    // Specialized get methods for common types
    bool get_bool(const std::string& key) const;
    int get_int(const std::string& key) const;
    double get_double(const std::string& key) const;
    std::string get_string(const std::string& key) const;
    
    std::vector<double> get_vector(const std::string& key) const;

    // Check if key exists
    bool has_key(const std::string& key) const;
    
private:
#ifdef HAVE_YAML_CPP
    YAML::Node get_node(const std::string& key) const;
#else
    void parse_simple_config(const std::string& filename);
    std::string trim(const std::string& str) const;
    template<typename T>
    T convert_string(const std::string& value) const;
#endif
};

// Template specializations and implementations

#ifdef HAVE_YAML_CPP

template<typename T>
T FastParameterLoader::get(const std::string& key) const {
    YAML::Node node = get_node(key);
    if (!node) {
        throw std::runtime_error("Parameter not found: " + key);
    }
    return node.as<T>();
}

#else // Simple parser fallback

template<typename T>
T FastParameterLoader::get(const std::string& key) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
        throw std::runtime_error("Parameter not found: " + key);
    }
    return convert_string<T>(it->second);
}

template<>
inline bool FastParameterLoader::convert_string<bool>(const std::string& value) const {
    std::string lower_value = value;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
    return lower_value == "true" || lower_value == "1" || lower_value == "yes";
}

template<>
inline int FastParameterLoader::convert_string<int>(const std::string& value) const {
    return std::stoi(value);
}

template<>
inline double FastParameterLoader::convert_string<double>(const std::string& value) const {
    return std::stod(value);
}

template<>
inline std::string FastParameterLoader::convert_string<std::string>(const std::string& value) const {
    return value;
}

#endif

} // namespace namo