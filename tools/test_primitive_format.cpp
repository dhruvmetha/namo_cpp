#include <iostream>
#include <fstream>
#include <cstdint>
#include <iomanip>

struct __attribute__((packed)) TestPrimitive {
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
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // std::cout << "File size: " << file_size << " bytes" << std::endl;
    // std::cout << "Expected struct size: " << sizeof(TestPrimitive) << " bytes" << std::endl;
    // std::cout << "Expected number of primitives: " << file_size / sizeof(TestPrimitive) << std::endl;
    
    // Read first few bytes to see the pattern
    char buffer[64];
    file.read(buffer, 64);
    file.seekg(0, std::ios::beg);
    
    // std::cout << "\nFirst 64 bytes in hex:" << std::endl;
    for (int i = 0; i < 64 && i < (int)file_size; i++) {
        if (i % 16 == 0) std::cout << std::hex << std::setw(4) << std::setfill('0') << i << ": ";
        // std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned char)buffer[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }
    // std::cout << std::endl;
    
    // Try to read as expected format
    TestPrimitive prim;
    // std::cout << "\nReading as TestPrimitive format:" << std::endl;
    for (int i = 0; i < 10 && file.read(reinterpret_cast<char*>(&prim), sizeof(prim)); i++) {
        // std::cout << "Primitive " << i << ": "
                  // << std::fixed << std::setprecision(6)
                  // << "dx=" << prim.delta_x << " "
                  // << "dy=" << prim.delta_y << " "
                  // << "dtheta=" << prim.delta_theta << " "
                  // << "edge=" << (int)prim.edge_idx << " "
                  // << "steps=" << (int)prim.push_steps << std::endl;
    }
    
    return 0;
}