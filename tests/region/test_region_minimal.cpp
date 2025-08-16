#include "planners/region/region_graph.hpp"
#include <iostream>

using namespace namo;

int main() {
    // std::cout << "=== Minimal Region Test ===" << std::endl;
    
    try {
        // Test 1: GenericFixedVector only
        // std::cout << "\n1. Testing GenericFixedVector..." << std::endl;
        GenericFixedVector<int, 10> small_vec;
        small_vec.push_back(1);
        small_vec.push_back(2);
        // std::cout << "   Size: " << small_vec.size() << std::endl;
        
        // Test 2: Region with minimal data
        // std::cout << "\n2. Testing Region..." << std::endl;
        Region test_region(0);
        test_region.centroid = {1.0, 2.0};
        // std::cout << "   Region ID: " << test_region.id << std::endl;
        
        // std::cout << "\n=== Minimal Test Passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}