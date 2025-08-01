cmake_minimum_required(VERSION 3.16)
project(namo_standalone VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

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

set(ALL_SOURCES 
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
    src/main.cpp
)

# Create main executable
add_executable(namo_standalone ${ALL_SOURCES})

# Link required libraries
target_link_libraries(namo_standalone
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link optional libraries if available
if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(namo_standalone glfw OpenGL::GL)
    target_compile_definitions(namo_standalone PRIVATE HAVE_GLFW)
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