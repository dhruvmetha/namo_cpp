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
    
    // Array getters
    template<size_t N>
    std::array<double, N> get_array(const std::string& key) const;
    
    std::vector<double> get_vector(const std::string& key) const;
    std::vector<std::string> get_string_vector(const std::string& key) const;
    
    // Check if key exists
    bool has_key(const std::string& key) const;
    
    // Batch parameter loading for performance
    void preload_keys(const std::vector<std::string>& keys);
    
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

template<size_t N>
std::array<double, N> FastParameterLoader::get_array(const std::string& key) const {
    YAML::Node node = get_node(key);
    if (!node) {
        throw std::runtime_error("Parameter not found: " + key);
    }
    if (!node.IsSequence()) {
        throw std::runtime_error("Parameter is not an array: " + key + " (type: " + 
                                std::to_string(static_cast<int>(node.Type())) + ")");
    }
    
    if (node.size() != N) {
        throw std::runtime_error("Array size mismatch for parameter: " + key + 
                                " (expected: " + std::to_string(N) + 
                                ", got: " + std::to_string(node.size()) + ")");
    }
    
    std::array<double, N> result;
    for (size_t i = 0; i < N; i++) {
        try {
            result[i] = node[i].as<double>();
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Failed to convert array element " + std::to_string(i) + 
                                    " of parameter " + key + " to double: " + e.what());
        }
    }
    return result;
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

template<size_t N>
std::array<double, N> FastParameterLoader::get_array(const std::string& key) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
        throw std::runtime_error("Parameter not found: " + key);
    }
    
    // Parse comma-separated values
    std::array<double, N> result;
    std::stringstream ss(it->second);
    std::string item;
    size_t index = 0;
    
    // Remove brackets if present
    std::string values = it->second;
    if (values.front() == '[') values = values.substr(1);
    if (values.back() == ']') values = values.substr(0, values.length() - 1);
    
    ss.str(values);
    
    while (std::getline(ss, item, ',') && index < N) {
        result[index++] = std::stod(trim(item));
    }
    
    if (index != N) {
        throw std::runtime_error("Array size mismatch for parameter: " + key);
    }
    
    return result;
}

#endif

} // namespace namo