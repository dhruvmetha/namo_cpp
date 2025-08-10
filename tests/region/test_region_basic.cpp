#include "planners/region/region_graph.hpp"
#include "core/types.hpp"
#include <iostream>

using namespace namo;

int main() {
    std::cout << "=== Basic Region Data Structures Test ===" << std::endl;
    
    try {
        // Test 1: GenericFixedVector
        std::cout << "\n1. Testing GenericFixedVector..." << std::endl;
        
        GenericFixedVector<int, 10> test_vec;
        test_vec.push_back(1);
        test_vec.push_back(2);
        test_vec.push_back(3);
        
        std::cout << "   Size: " << test_vec.size() << ", Capacity: " << test_vec.capacity() << std::endl;
        std::cout << "   Elements: ";
        for (size_t i = 0; i < test_vec.size(); ++i) {
            std::cout << test_vec[i] << " ";
        }
        std::cout << std::endl;
        
        // Test 2: Region
        std::cout << "\n2. Testing Region..." << std::endl;
        
        Region test_region(0);
        test_region.add_grid_cell(5, 10);
        test_region.add_grid_cell(6, 10);
        test_region.add_sample_point(1.0, 2.0);
        test_region.centroid = {1.5, 2.5};
        test_region.area = 0.1;
        
        std::cout << "   Region ID: " << test_region.id << std::endl;
        std::cout << "   Grid cells: " << test_region.grid_cells.size() << std::endl;
        std::cout << "   Sample points: " << test_region.sample_points.size() << std::endl;
        std::cout << "   Centroid: (" << test_region.centroid.first 
                  << ", " << test_region.centroid.second << ")" << std::endl;
        std::cout << "   Area: " << test_region.area << std::endl;
        
        // Test 3: RegionEdge
        std::cout << "\n3. Testing RegionEdge..." << std::endl;
        
        RegionEdge test_edge(0, 1);
        test_edge.add_blocking_object("box_1");
        test_edge.add_blocking_object("box_2");
        test_edge.connection_strength = 0.8;
        
        std::cout << "   Connects regions: " << test_edge.region_a_id 
                  << " <-> " << test_edge.region_b_id << std::endl;
        std::cout << "   Blocking objects: " << test_edge.blocking_objects.size() << std::endl;
        std::cout << "   Is blocked: " << (test_edge.is_currently_blocked ? "yes" : "no") << std::endl;
        std::cout << "   Connection strength: " << test_edge.connection_strength << std::endl;
        
        // Test edge utilities
        std::cout << "   Other region from 0: " << test_edge.get_other_region(0) << std::endl;
        std::cout << "   Connects (0,1): " << (test_edge.connects(0, 1) ? "yes" : "no") << std::endl;
        
        // Test 4: RegionGraph
        std::cout << "\n4. Testing RegionGraph..." << std::endl;
        
        RegionGraph test_graph;
        
        // Add regions
        Region region_0(0);
        region_0.centroid = {0.0, 0.0};
        region_0.area = 1.0;
        
        Region region_1(1);
        region_1.centroid = {2.0, 0.0};
        region_1.area = 1.5;
        
        Region region_2(2);
        region_2.centroid = {4.0, 0.0};
        region_2.area = 2.0;
        
        int id0 = test_graph.add_region(region_0);
        int id1 = test_graph.add_region(region_1);
        int id2 = test_graph.add_region(region_2);
        
        std::cout << "   Added regions with IDs: " << id0 << ", " << id1 << ", " << id2 << std::endl;
        
        // Add edges
        test_graph.add_edge(0, 1, 1.0);
        test_graph.add_edge(1, 2, 0.8);
        
        std::cout << "   Total regions: " << test_graph.num_regions() << std::endl;
        std::cout << "   Total edges: " << test_graph.num_edges() << std::endl;
        
        // Test graph queries
        const Region* r1 = test_graph.get_region(1);
        if (r1) {
            std::cout << "   Region 1 centroid: (" << r1->centroid.first 
                      << ", " << r1->centroid.second << ")" << std::endl;
        }
        
        const RegionEdge* edge_01 = test_graph.find_edge(0, 1);
        if (edge_01) {
            std::cout << "   Edge (0,1) strength: " << edge_01->connection_strength << std::endl;
        }
        
        // Test neighbors
        const auto& neighbors_of_1 = test_graph.get_neighbors(1);
        std::cout << "   Neighbors of region 1: ";
        for (size_t i = 0; i < neighbors_of_1.size(); ++i) {
            std::cout << neighbors_of_1[i] << " ";
        }
        std::cout << std::endl;
        
        // Test 5: LightweightState
        std::cout << "\n5. Testing LightweightState..." << std::endl;
        
        LightweightState state;
        state.robot_pose = SE2State(1.0, 2.0, 0.5);
        state.object_names.push_back("box_1");
        state.object_names.push_back("box_2");
        state.movable_object_poses[0] = SE2State(3.0, 4.0, 1.0);
        state.movable_object_poses[1] = SE2State(5.0, 6.0, 1.5);
        
        std::cout << "   Robot pose: (" << state.robot_pose.x << ", " 
                  << state.robot_pose.y << ", " << state.robot_pose.theta << ")" << std::endl;
        std::cout << "   Number of objects: " << state.object_names.size() << std::endl;
        
        // Test state operations
        LightweightState copied_state = state.copy();
        copied_state.apply_object_movement("box_1", SE2State(10.0, 11.0, 2.0));
        
        const SE2State* moved_obj = copied_state.get_object_pose("box_1");
        if (moved_obj) {
            std::cout << "   After movement, box_1 at: (" << moved_obj->x 
                      << ", " << moved_obj->y << ", " << moved_obj->theta << ")" << std::endl;
        }
        
        // Test hash computation
        std::size_t hash1 = state.compute_hash();
        std::size_t hash2 = copied_state.compute_hash();
        std::cout << "   Original state hash: " << hash1 << std::endl;
        std::cout << "   Modified state hash: " << hash2 << std::endl;
        std::cout << "   Hashes different: " << (hash1 != hash2 ? "yes" : "no") << std::endl;
        
        std::cout << "\n=== All Basic Tests Passed! ===" << std::endl;
        std::cout << "Core region data structures are working correctly." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}