#include "core/parameter_loader.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>

namespace namo {

FastParameterLoader::FastParameterLoader(const std::string& filename) {
#ifdef HAVE_YAML_CPP
    try {
        root_ = YAML::LoadFile(filename);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load YAML file: " + filename + " - " + e.what());
    }
#else
    parse_simple_config(filename);
#endif
}

#ifdef HAVE_YAML_CPP

YAML::Node FastParameterLoader::get_node(const std::string& key) const {
    // Check cache first
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    // Navigate through nested keys (e.g., "motion_primitives.control_steps").
    //
    // yaml-cpp 0.6 quirk: declaring `YAML::Node node = root_;` and then
    // reassigning via `node = node[token]` inside the loop does not behave
    // like one would expect — operator= on Node has reference-semantics that
    // can leave the rebound `node` disconnected from the parsed document.
    // The robust pattern is to track the *path* with std::vector<std::string>
    // and resolve via chained operator[] from root_ each iteration.
    std::vector<std::string> tokens;
    {
        std::istringstream ss(key);
        std::string token;
        while (std::getline(ss, token, '.')) {
            tokens.push_back(token);
        }
    }

    YAML::Node cur = YAML::Clone(root_);  // Clone to avoid mutating root_
    for (const auto& tok : tokens) {
        if (!cur.IsMap()) {
            YAML::Node null_node;
            cache_[key] = null_node;
            return null_node;
        }
        YAML::Node child = cur[tok];
        if (!child.IsDefined()) {
            YAML::Node null_node;
            cache_[key] = null_node;
            return null_node;
        }
        cur.reset(child);  // rebind cur to point at child, preserving semantics
    }

    cache_[key] = cur;
    return cur;
}

bool FastParameterLoader::get_bool(const std::string& key) const {
    return get<bool>(key);
}

int FastParameterLoader::get_int(const std::string& key) const {
    return get<int>(key);
}

double FastParameterLoader::get_double(const std::string& key) const {
    return get<double>(key);
}

std::string FastParameterLoader::get_string(const std::string& key) const {
    return get<std::string>(key);
}

std::vector<double> FastParameterLoader::get_vector(const std::string& key) const {
    YAML::Node node = get_node(key);
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("Parameter is not a sequence: " + key);
    }
    
    std::vector<double> result;
    result.reserve(node.size());
    for (const auto& item : node) {
        result.push_back(item.as<double>());
    }
    return result;
}

bool FastParameterLoader::has_key(const std::string& key) const {
    YAML::Node node = get_node(key);
    return node.IsDefined() && !node.IsNull();
}

#else // Simple parser implementation

void FastParameterLoader::parse_simple_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Handle sections [section_name]
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Handle YAML sections (key:)
        if (line.back() == ':' && line.find(' ') == std::string::npos) {
            current_section = line.substr(0, line.length() - 1);
            continue;
        }
        
        // Handle key=value pairs (INI format)
        size_t eq_pos = line.find('=');
        size_t colon_pos = line.find(':');
        
        size_t sep_pos = std::string::npos;
        if (eq_pos != std::string::npos && colon_pos != std::string::npos) {
            sep_pos = std::min(eq_pos, colon_pos);
        } else if (eq_pos != std::string::npos) {
            sep_pos = eq_pos;
        } else if (colon_pos != std::string::npos) {
            sep_pos = colon_pos;
        }
        
        if (sep_pos != std::string::npos) {
            std::string key = trim(line.substr(0, sep_pos));
            std::string value = trim(line.substr(sep_pos + 1));
            
            // Remove YAML comments *before* quote stripping
            size_t comment_pos = value.find('#');
            if (comment_pos != std::string::npos) {
                value = trim(value.substr(0, comment_pos));
            }

            // Remove quotes if present
            if (value.length() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\''))) {
                value = value.substr(1, value.length() - 2);
            }
            
            // Add section prefix if we're in a section
            if (!current_section.empty()) {
                key = current_section + "." + key;
            }
            
            params_[key] = value;
        }
    }
}

std::string FastParameterLoader::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

bool FastParameterLoader::get_bool(const std::string& key) const {
    return get<bool>(key);
}

int FastParameterLoader::get_int(const std::string& key) const {
    return get<int>(key);
}

double FastParameterLoader::get_double(const std::string& key) const {
    return get<double>(key);
}

std::string FastParameterLoader::get_string(const std::string& key) const {
    return get<std::string>(key);
}

std::vector<double> FastParameterLoader::get_vector(const std::string& key) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
        throw std::runtime_error("Parameter not found: " + key);
    }
    
    std::vector<double> result;
    std::stringstream ss(it->second);
    std::string item;
    
    // Remove brackets if present
    std::string values = it->second;
    if (values.front() == '[') values = values.substr(1);
    if (values.back() == ']') values = values.substr(0, values.length() - 1);
    
    ss.str(values);
    
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(trim(item)));
    }
    
    return result;
}

bool FastParameterLoader::has_key(const std::string& key) const {
    return params_.find(key) != params_.end();
}

#endif

} // namespace namo