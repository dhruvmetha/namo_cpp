#include "planning/primitive_loader.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>

namespace namo {

PrimitiveLoader::PrimitiveLoader() : loaded_count_(0), is_loaded_(false) {
    // Initialize lookup table with invalid indices
    for (size_t edge = 0; edge < MAX_EDGES; edge++) {
        for (size_t step = 0; step < MAX_STEPS; step++) {
            lookup_table_[edge][step] = -1;
        }
    }
}

bool PrimitiveLoader::load_primitives(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open primitive file: " << filepath << std::endl;
        return false;
    }
    
    // Read primitive data directly into our array
    // The binary format matches our NominalPrimitive struct from the generator
    struct __attribute__((packed)) BinaryPrimitive {
        float delta_x;        // Position change in x
        float delta_y;        // Position change in y  
        float delta_theta;    // Rotation change (yaw)
        uint8_t edge_idx;     // Push direction (0-11)
        uint8_t push_steps;   // Push step number (1-10)
    };
    
    // Verify struct size matches expected
    constexpr size_t expected_size = 4 + 4 + 4 + 1 + 1; // 14 bytes
    static_assert(sizeof(BinaryPrimitive) == expected_size, 
                  "BinaryPrimitive struct has unexpected size due to padding");
    
    // Read header (primitive count)
    uint32_t primitive_count;
    if (!file.read(reinterpret_cast<char*>(&primitive_count), sizeof(primitive_count))) {
        std::cerr << "Failed to read primitive count header" << std::endl;
        return false;
    }
    
    std::cout << "Binary file contains " << primitive_count << " primitives" << std::endl;
    
    loaded_count_ = 0;
    BinaryPrimitive binary_primitive;
    
    while (file.read(reinterpret_cast<char*>(&binary_primitive), sizeof(BinaryPrimitive)) && 
           loaded_count_ < MAX_PRIMITIVES && loaded_count_ < primitive_count) {
        
        // Copy into our primitive array
        primitives_[loaded_count_] = LoadedPrimitive(
            binary_primitive.delta_x,
            binary_primitive.delta_y,
            binary_primitive.delta_theta,
            binary_primitive.edge_idx,
            binary_primitive.push_steps
        );
        
        // Update lookup table for O(1) access
        int edge_idx = binary_primitive.edge_idx;
        int push_steps = binary_primitive.push_steps;
        
        if (edge_idx >= 0 && edge_idx < MAX_EDGES && 
            push_steps >= 1 && push_steps <= MAX_STEPS) {
            lookup_table_[edge_idx][push_steps - 1] = loaded_count_;
        }
        
        loaded_count_++;
    }
    
    file.close();
    
    if (loaded_count_ > 0) {
        is_loaded_ = true;
        std::cout << "Loaded " << loaded_count_ << " motion primitives from " << filepath << std::endl;
        return true;
    } else {
        std::cerr << "No primitives loaded from " << filepath << std::endl;
        return false;
    }
}

const LoadedPrimitive& PrimitiveLoader::get_primitive(int edge_idx, int push_steps) const {
    if (!is_loaded_) {
        throw std::runtime_error("Primitives not loaded");
    }
    
    if (edge_idx < 0 || edge_idx >= MAX_EDGES || 
        push_steps < 1 || push_steps > MAX_STEPS) {
        throw std::runtime_error("Invalid edge_idx or push_steps");
    }
    
    int primitive_index = lookup_table_[edge_idx][push_steps - 1];
    if (primitive_index < 0) {
        throw std::runtime_error("Primitive not found for edge " + std::to_string(edge_idx) + 
                                " steps " + std::to_string(push_steps));
    }
    
    return primitives_[primitive_index];
}

std::vector<int> PrimitiveLoader::get_valid_steps_for_edge(int edge_idx) const {
    std::vector<int> valid_steps;
    if (edge_idx < 0 || edge_idx >= MAX_EDGES) {
        return valid_steps;
    }
    
    for (int step = 1; step <= MAX_STEPS; step++) {
        if (lookup_table_[edge_idx][step - 1] >= 0) {
            valid_steps.push_back(step);
        }
    }
    
    return valid_steps;
}

} // namespace namo