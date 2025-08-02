#include "environment/namo_environment.hpp"
#include <iostream>

int main() {
    NAMOEnvironment env("data/nominal_primitive_scene.xml", false);
    
    auto robot_state = env.get_robot_state();
    std::cout << "Robot position: [" << robot_state->position[0] << ", " 
              << robot_state->position[1] << ", " << robot_state->position[2] << "]" << std::endl;
    
    return 0;
}
EOF < /dev/null
