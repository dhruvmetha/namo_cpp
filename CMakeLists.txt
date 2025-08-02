cmake_minimum_required(VERSION 3.16)
project(namo_standalone VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -pg -fno-omit-frame-pointer -DNDEBUG")

# Add PROFILE build type
set(CMAKE_BUILD_TYPE_OPTIONS "Debug" "Release" "Profile")
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    message(STATUS "Building with profiling enabled")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find MuJoCo
if(DEFINED ENV{MJ_PATH})
    set(MUJOCO_DIR $ENV{MJ_PATH})
else()
    message(FATAL_ERROR "MJ_PATH environment variable not set. Please set it to your MuJoCo installation directory.")
endif()

message(STATUS "Using MuJoCo from: ${MUJOCO_DIR}")

# Find MuJoCo library in different possible locations
find_library(MUJOCO_LIB 
    NAMES mujoco libmujoco.so libmujoco.a
    PATHS 
        ${MUJOCO_DIR}/lib
        ${MUJOCO_DIR}/build/lib
        ${MUJOCO_DIR}/dist/lib
        ${MUJOCO_DIR}
    NO_DEFAULT_PATH
)

if(NOT MUJOCO_LIB)
    # Try to build MuJoCo if source is available
    if(EXISTS ${MUJOCO_DIR}/CMakeLists.txt)
        message(STATUS "MuJoCo library not found, attempting to build...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E make_directory ${MUJOCO_DIR}/build
            WORKING_DIRECTORY ${MUJOCO_DIR}
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} .. -DCMAKE_BUILD_TYPE=Release
            WORKING_DIRECTORY ${MUJOCO_DIR}/build
            RESULT_VARIABLE BUILD_RESULT
        )
        if(BUILD_RESULT EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} --build . --target mujoco -j4
                WORKING_DIRECTORY ${MUJOCO_DIR}/build
                RESULT_VARIABLE BUILD_RESULT
            )
        endif()
        
        # Try to find the library again
        find_library(MUJOCO_LIB 
            NAMES mujoco libmujoco.so libmujoco.a
            PATHS ${MUJOCO_DIR}/build/lib ${MUJOCO_DIR}/lib
            NO_DEFAULT_PATH
        )
    endif()
    
    if(NOT MUJOCO_LIB)
        message(FATAL_ERROR "MuJoCo library not found. Please build MuJoCo first or check MJ_PATH.")
    endif()
endif()

# Find system dependencies
find_package(PkgConfig REQUIRED)

# OpenCV (required for visualization and data collection)
find_package(OpenCV REQUIRED)

# GLFW and OpenGL (for visualization)
find_package(glfw3 QUIET)
find_package(OpenGL QUIET)

# ZMQ (for distributed inference - optional) - TEMPORARILY DISABLED
# pkg_check_modules(ZMQ QUIET libzmq)

# JSON library (header-only, will download if needed)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp PATHS external/nlohmann_json/include)

# YAML-CPP (will use external if system version not found)
# find_package(yaml-cpp QUIET)

# Download external dependencies if not found
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "Downloading nlohmann/json...")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann_json/include/nlohmann/json.hpp"
        SHOW_PROGRESS
    )
    set(NLOHMANN_JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann_json/include")
endif()

# Include directories
include_directories(
    include
    ${MUJOCO_DIR}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE PLANNING_SOURCES "src/planning/*.cpp")
file(GLOB_RECURSE ENVIRONMENT_SOURCES "src/environment/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")
file(GLOB_RECURSE SKILLS_SOURCES "src/skills/*.cpp")

set(ALL_SOURCES 
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
    ${SKILLS_SOURCES}
    src/main.cpp
)

# Create main executable
add_executable(namo_standalone ${ALL_SOURCES})

# Create primitive generation tool (exclude problematic sources)
file(GLOB_RECURSE PRIMITIVE_GEN_PLANNING_SOURCES "src/planning/*.cpp")
list(REMOVE_ITEM PRIMITIVE_GEN_PLANNING_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/planning/mpc_executor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/planning/greedy_planner.cpp")

add_executable(generate_motion_primitives_db 
    tools/generate_motion_primitives_db.cpp 
    ${CORE_SOURCES}
    ${PRIMITIVE_GEN_PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Create complete planning test (DISABLED - needs API fixes)
# add_executable(test_complete_planning
#     tools/test_complete_planning.cpp
#     ${CORE_SOURCES}
#     ${PLANNING_SOURCES}
#     ${ENVIRONMENT_SOURCES}
#     ${UTILS_SOURCES}
# )

# Create simple primitive test
add_executable(test_primitives_only
    tools/test_primitives_only.cpp
    src/core/parameter_loader.cpp
    src/planning/primitive_loader.cpp
    src/planning/greedy_planner.cpp
)

# Create debug primitive test
add_executable(debug_primitives
    tools/debug_primitives.cpp
)

# Create MPC executor test
add_executable(test_mpc_executor
    tools/test_mpc_executor.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Create end-to-end test
add_executable(test_end_to_end
    tools/test_end_to_end.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Create displacement error analysis tool
add_executable(analyze_displacement_errors
    tools/analyze_displacement_errors.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Create planner output test
add_executable(test_planner_output
    tools/test_planner_output.cpp
    src/core/parameter_loader.cpp
    src/planning/primitive_loader.cpp
    src/planning/greedy_planner.cpp
)

# Create iterative MPC test
add_executable(test_iterative_mpc
    tools/test_iterative_mpc.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Create coordinate transformation test
add_executable(test_coordinate_transformations
    tools/test_coordinate_transformations.cpp
    src/core/parameter_loader.cpp
    src/planning/primitive_loader.cpp
    src/planning/greedy_planner.cpp
)

# Create MPC planning debug tool
add_executable(debug_mpc_planning
    tools/debug_mpc_planning.cpp
    src/core/parameter_loader.cpp
    src/planning/primitive_loader.cpp
    src/planning/greedy_planner.cpp
)

# Create search limits test
add_executable(test_search_limits
    tools/test_search_limits.cpp
    src/core/parameter_loader.cpp
    src/planning/primitive_loader.cpp
    src/planning/greedy_planner.cpp
)

# Link required libraries
target_link_libraries(namo_standalone
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link primitive generation tool
target_link_libraries(generate_motion_primitives_db
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link complete planning test (DISABLED)
# target_link_libraries(test_complete_planning
#     ${MUJOCO_LIB}
#     ${OpenCV_LIBS}
#     pthread
#     m  # math library
# )

# Link simple primitive test
target_link_libraries(test_primitives_only
    pthread
    m  # math library
)

# Link debug primitive test
target_link_libraries(debug_primitives
    pthread
    m  # math library
)

# Link MPC executor test
target_link_libraries(test_mpc_executor
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link end-to-end test
target_link_libraries(test_end_to_end
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link displacement error analysis tool
target_link_libraries(analyze_displacement_errors
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link planner output test
target_link_libraries(test_planner_output
    pthread
    m  # math library
)

# Link iterative MPC test
target_link_libraries(test_iterative_mpc
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Add GLFW and OpenGL support if available
if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(test_iterative_mpc glfw OpenGL::GL)
    target_compile_definitions(test_iterative_mpc PRIVATE HAVE_GLFW)
endif()

# Add profiling support for test_iterative_mpc
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_compile_options(test_iterative_mpc PRIVATE -pg -fno-omit-frame-pointer)
    target_link_options(test_iterative_mpc PRIVATE -pg)
endif()

# Visual markers test executable
add_executable(test_visual_markers
    tools/test_visual_markers.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
)

# Link visual markers test
target_link_libraries(test_visual_markers
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Debug crash test executable
add_executable(debug_crash
    tools/debug_crash.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
)

# Link debug crash test
target_link_libraries(debug_crash
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link coordinate transformation test
target_link_libraries(test_coordinate_transformations
    pthread
    m  # math library
)

# Link MPC planning debug tool
target_link_libraries(debug_mpc_planning
    pthread
    m  # math library
)

# Link search limits test
target_link_libraries(test_search_limits
    pthread
    m  # math library
)

# NAMO Skill test
add_executable(test_namo_skill
    tests/test_namo_skill.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
)

target_link_libraries(test_namo_skill
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Simple skill test
add_executable(test_simple_skill
    tests/test_simple_skill.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
)

target_link_libraries(test_simple_skill
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Skill demonstration executable
add_executable(skill_demo
    examples/skill_demo.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
)

target_link_libraries(skill_demo
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link optional libraries if available
if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(namo_standalone glfw OpenGL::GL)
    target_compile_definitions(namo_standalone PRIVATE HAVE_GLFW)
    
    # Also link GLFW to primitive generation tool
    target_link_libraries(generate_motion_primitives_db glfw OpenGL::GL)
    target_compile_definitions(generate_motion_primitives_db PRIVATE HAVE_GLFW)
    
    # Also link GLFW to planning test (DISABLED)
    # target_link_libraries(test_complete_planning glfw OpenGL::GL)
    # target_compile_definitions(test_complete_planning PRIVATE HAVE_GLFW)
    
    # Also link GLFW to MPC executor test
    target_link_libraries(test_mpc_executor glfw OpenGL::GL)
    target_compile_definitions(test_mpc_executor PRIVATE HAVE_GLFW)
    
    # Also link GLFW to end-to-end test
    target_link_libraries(test_end_to_end glfw OpenGL::GL)
    target_compile_definitions(test_end_to_end PRIVATE HAVE_GLFW)
    
    # Also link GLFW to error analysis tool
    target_link_libraries(analyze_displacement_errors glfw OpenGL::GL)
    target_compile_definitions(analyze_displacement_errors PRIVATE HAVE_GLFW)
    
    # Also link GLFW to skill tests
    target_link_libraries(test_namo_skill glfw OpenGL::GL)
    target_compile_definitions(test_namo_skill PRIVATE HAVE_GLFW)
    
    # Also link GLFW to simple skill test
    target_link_libraries(test_simple_skill glfw OpenGL::GL)
    target_compile_definitions(test_simple_skill PRIVATE HAVE_GLFW)
    
    # Also link GLFW to skill demo
    target_link_libraries(skill_demo glfw OpenGL::GL)
    target_compile_definitions(skill_demo PRIVATE HAVE_GLFW)
    
    message(STATUS "GLFW and OpenGL found - visualization enabled")
else()
    message(STATUS "GLFW or OpenGL not found - visualization disabled")
endif()

# if(yaml-cpp_FOUND)
#     target_link_libraries(namo_standalone yaml-cpp)
#     target_compile_definitions(namo_standalone PRIVATE HAVE_YAML_CPP)
#     message(STATUS "yaml-cpp found - YAML config support enabled")
# else()
message(STATUS "yaml-cpp not found - using simple config parser")
# endif()

if(ZMQ_FOUND)
    target_link_libraries(namo_standalone ${ZMQ_LIBRARIES})
    target_include_directories(namo_standalone PRIVATE ${ZMQ_INCLUDE_DIRS})
    target_compile_definitions(namo_standalone PRIVATE HAVE_ZMQ)
    message(STATUS "ZMQ found - distributed inference enabled")
else()
    message(STATUS "ZMQ not found - distributed inference disabled") 
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(namo_standalone PRIVATE -fopenmp -ffast-math)
    target_link_libraries(namo_standalone gomp)
endif()

# Create test executable
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
if(TEST_SOURCES)
    add_executable(namo_test ${TEST_SOURCES} ${CORE_SOURCES} ${PLANNING_SOURCES} ${ENVIRONMENT_SOURCES} ${UTILS_SOURCES})
    target_link_libraries(namo_test ${MUJOCO_LIB} ${OpenCV_LIBS} pthread m)
    
    if(glfw3_FOUND AND OpenGL_FOUND)
        target_link_libraries(namo_test glfw OpenGL::GL)
        target_compile_definitions(namo_test PRIVATE HAVE_GLFW)
    endif()
    
    if(yaml-cpp_FOUND)
        target_link_libraries(namo_test yaml-cpp)
        target_compile_definitions(namo_test PRIVATE HAVE_YAML_CPP)
    endif()
endif()

# Install targets
install(TARGETS namo_standalone DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/namo_standalone/config)
install(DIRECTORY data/ DESTINATION share/namo_standalone/data OPTIONAL)