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

# Collect source files with new structure
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE PLANNING_SOURCES "src/planning/*.cpp")
file(GLOB_RECURSE WAVEFRONT_SOURCES "src/wavefront/*.cpp")
file(GLOB_RECURSE PLANNER_SOURCES "src/planners/*.cpp")
file(GLOB_RECURSE STRATEGY_SOURCES "src/planners/strategies/*.cpp")
file(GLOB_RECURSE ENVIRONMENT_SOURCES "src/environment/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")
file(GLOB_RECURSE SKILLS_SOURCES "src/skills/*.cpp")
file(GLOB_RECURSE CONFIG_SOURCES "src/config/*.cpp")

set(ALL_SOURCES 
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${WAVEFRONT_SOURCES}
    ${PLANNER_SOURCES}
    ${STRATEGY_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
    ${SKILLS_SOURCES}
    ${CONFIG_SOURCES}
    src/main.cpp
)

# High-level planning demonstration executable (MAIN TARGET)
add_executable(high_level_planning_demo
    examples/high_level_planning_demo.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${WAVEFRONT_SOURCES}
    ${PLANNER_SOURCES}
    ${STRATEGY_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
    ${CONFIG_SOURCES}
)

# Link high-level planning demo
target_link_libraries(high_level_planning_demo
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link optional libraries if available
if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(high_level_planning_demo glfw OpenGL::GL)
    target_compile_definitions(high_level_planning_demo PRIVATE HAVE_GLFW)
    message(STATUS "GLFW and OpenGL found - visualization enabled")
else()
    message(STATUS "GLFW or OpenGL not found - visualization disabled")
endif()

message(STATUS "yaml-cpp not found - using simple config parser")

if(ZMQ_FOUND)
    target_link_libraries(high_level_planning_demo ${ZMQ_LIBRARIES})
    target_include_directories(high_level_planning_demo PRIVATE ${ZMQ_INCLUDE_DIRS})
    target_compile_definitions(high_level_planning_demo PRIVATE HAVE_ZMQ)
    message(STATUS "ZMQ found - distributed inference enabled")
else()
    message(STATUS "ZMQ not found - distributed inference disabled") 
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(high_level_planning_demo PRIVATE -fopenmp -ffast-math)
    target_link_libraries(high_level_planning_demo gomp)
endif()

# Optional Python bindings target
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Building Python bindings...")
    
    # Find Python3 first
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 version: ${Python3_VERSION}")
    message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
    message(STATUS "Python3 SOABI: ${Python3_SOABI}")
    
    # Set variables that pybind11 uses for Python detection
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(PYTHON_LIBRARY ${Python3_LIBRARIES})
    set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
    set(PYTHON_VERSION_STRING ${Python3_VERSION})
    set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR})
    set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR})
    
    # Always download latest compatible pybind11 for Python 3.11 support
    message(STATUS "Downloading compatible pybind11 for Python ${Python3_VERSION}...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.12.0  # Latest version with Python 3.11+ support
    )
    FetchContent_MakeAvailable(pybind11)
    
    # Python RL environment sources
    set(PYTHON_RL_SOURCES
        python/namo/cpp_bindings/rl_env.cpp
        python/namo/cpp_bindings/bindings.cpp
        ${CORE_SOURCES}
        ${PLANNING_SOURCES}
        ${WAVEFRONT_SOURCES}
        ${PLANNER_SOURCES}
        ${STRATEGY_SOURCES}
        ${ENVIRONMENT_SOURCES}
        ${SKILLS_SOURCES}
        ${CONFIG_SOURCES}
    )
    
    # Create Python module with explicit Python3 linking
    pybind11_add_module(namo_rl ${PYTHON_RL_SOURCES})
    
    # Set include directories for Python module
    target_include_directories(namo_rl PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # Add project root for python/rl_env.hpp
        include
        ${MUJOCO_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}  # Explicitly use Python3 includes
    )
    
    # Link libraries including explicit Python3 libraries
    target_link_libraries(namo_rl PRIVATE
        ${MUJOCO_LIB}
        ${OpenCV_LIBS}
        Python3::Module  # Explicitly link Python3 module target
        pthread
        m  # math library
    )
    
    # Link optional libraries if available
    if(glfw3_FOUND AND OpenGL_FOUND)
        target_link_libraries(namo_rl PRIVATE glfw OpenGL::GL)
        target_compile_definitions(namo_rl PRIVATE HAVE_GLFW)
    endif()
    
    if(ZMQ_FOUND)
        target_link_libraries(namo_rl PRIVATE ${ZMQ_LIBRARIES})
        target_include_directories(namo_rl PRIVATE ${ZMQ_INCLUDE_DIRS})
        target_compile_definitions(namo_rl PRIVATE HAVE_ZMQ)
    endif()
    
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(namo_rl PRIVATE -fopenmp -ffast-math)
        target_link_libraries(namo_rl PRIVATE gomp)
    endif()
    
    # Set properties for Python module to use the correct Python3 version
    target_compile_definitions(namo_rl PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    
    # Force the module suffix to match the target Python version
    set_target_properties(namo_rl PROPERTIES
        OUTPUT_NAME "namo_rl"
        SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    )
    
    message(STATUS "Python bindings will be built as 'namo_rl' module")
    message(STATUS "Target Python3 SOABI: ${Python3_SOABI}")
    message(STATUS "Module suffix will be: ${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Motion primitives generator tool
add_executable(generate_motion_primitives_db
    tools/generate_motion_primitives_db.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${WAVEFRONT_SOURCES}
    ${PLANNER_SOURCES}
    ${STRATEGY_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
    ${CONFIG_SOURCES}
)

# Link motion primitives generator
target_link_libraries(generate_motion_primitives_db
    ${MUJOCO_LIB}
    ${OpenCV_LIBS}
    pthread
    m  # math library
)

# Link optional libraries if available
if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(generate_motion_primitives_db glfw OpenGL::GL)
    target_compile_definitions(generate_motion_primitives_db PRIVATE HAVE_GLFW)
endif()

if(ZMQ_FOUND)
    target_link_libraries(generate_motion_primitives_db ${ZMQ_LIBRARIES})
    target_include_directories(generate_motion_primitives_db PRIVATE ${ZMQ_INCLUDE_DIRS})
    target_compile_definitions(generate_motion_primitives_db PRIVATE HAVE_ZMQ)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(generate_motion_primitives_db PRIVATE -fopenmp -ffast-math)
    target_link_libraries(generate_motion_primitives_db gomp)
endif()

# Test executable for planner output with visualization
add_executable(test_planner_output
    tests/planning/test_planner_output.cpp
    ${CORE_SOURCES}
    ${PLANNING_SOURCES}
    ${WAVEFRONT_SOURCES}
    ${PLANNER_SOURCES}
    ${STRATEGY_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${SKILLS_SOURCES}
    ${CONFIG_SOURCES}
)

target_link_libraries(test_planner_output
    ${MUJOCO_LIB}
    pthread
    m  # math library
    yaml-cpp
)

target_include_directories(test_planner_output PRIVATE
    include
    ${MUJOCO_DIR}/include
)

if(glfw3_FOUND AND OpenGL_FOUND)
    target_link_libraries(test_planner_output glfw OpenGL::GL)
    target_compile_definitions(test_planner_output PRIVATE HAVE_GLFW)
endif()

if(ZMQ_FOUND)
    target_link_libraries(test_planner_output ${ZMQ_LIBRARIES})
    target_include_directories(test_planner_output PRIVATE ${ZMQ_INCLUDE_DIRS})
    target_compile_definitions(test_planner_output PRIVATE HAVE_ZMQ)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_planner_output PRIVATE -fopenmp -ffast-math)
    target_link_libraries(test_planner_output gomp)
endif()

# Install targets
install(TARGETS high_level_planning_demo generate_motion_primitives_db test_planner_output DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/namo_standalone/config)
install(DIRECTORY data/ DESTINATION share/namo_standalone/data OPTIONAL)