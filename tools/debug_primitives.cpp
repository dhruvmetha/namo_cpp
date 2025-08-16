/**
 * @file debug_primitives.cpp
 * @brief Debug primitive binary format
 */

#include <iostream>
#include <fstream>
#include <cstdint>

struct __attribute__((packed)) BinaryPrimitive {
    float delta_x;        // Position change in x
    float delta_y;        // Position change in y  
    float delta_theta;    // Rotation change (yaw)
    uint8_t edge_idx;     // Push direction (0-11)
    uint8_t push_steps;   // Push step number (1-10)
};

int main() {
    std::ifstream file("data/motion_primitives.dat", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    
    // Read header (primitive count)
    uint32_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    // std::cout << "Struct size: " << sizeof(BinaryPrimitive) << " bytes" << std::endl;
    // std::cout << "Primitives in file: " << count << std::endl;
    
    BinaryPrimitive prim;
    for (int i = 0; i < 10 && i < (int)count && file.read(reinterpret_cast<char*>(&prim), sizeof(prim)); i++) {
        // std::cout << "Primitive " << i << ": "
                  // << "dx=" << prim.delta_x << " "
                  // << "dy=" << prim.delta_y << " "
                  // << "dtheta=" << prim.delta_theta << " "
                  // << "edge=" << (int)prim.edge_idx << " "
                  // << "steps=" << (int)prim.push_steps << std::endl;
        
        // Show raw bytes
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&prim);
        // std::cout << "  Raw bytes: ";
        for (size_t j = 0; j < sizeof(prim); j++) {
            printf("%02x ", bytes[j]);
        }
        // std::cout << std::endl;
    }
    
    return 0;
}