#include <iostream>
#include <fstream>
#include <cstdint>
#include <iomanip>

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
    
    std::cout << "Struct size: " << sizeof(BinaryPrimitive) << " bytes" << std::endl;
    std::cout << "Primitives in file: " << count << std::endl;
    std::cout << "\nShowing edge alignment across different edges:" << std::endl;
    
    BinaryPrimitive prim;
    for (int i = 0; i < 50 && i < (int)count && file.read(reinterpret_cast<char*>(&prim), sizeof(prim)); i++) {
        std::cout << "Primitive " << std::setw(2) << i << ": "
                  << std::fixed << std::setprecision(4)
                  << "dx=" << std::setw(8) << prim.delta_x << " "
                  << "dy=" << std::setw(8) << prim.delta_y << " "
                  << "dtheta=" << std::setw(8) << prim.delta_theta << " "
                  << "edge=" << std::setw(2) << (int)prim.edge_idx << " "
                  << "steps=" << std::setw(2) << (int)prim.push_steps << std::endl;
    }
    
    return 0;
}