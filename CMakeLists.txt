# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.18)

project(magnetron LANGUAGES C)

message("Configuring magnetron SDK for ${CMAKE_SYSTEM_PROCESSOR}...")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

set(CMAKE_C_STANDARD 99) # Use C99 standard
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")      # Place all built libraries in the build directory
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")   # Place all built executables in the build directory
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# General build options
option(MAGNETRON_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)  # Build Python bindings (requires Python development environment and C++ toolchain)
option(MAGNETRON_BUILD_TESTS "Build tests" ON)                      # Build C++ unit tests (requires C++ toolchain)
option(MAGNETRON_BUILD_BENCHMARKS "Build benchmaryks" ON)           # Build C++ benchmarks (requires C++ toolchain)
option(MAGNETRON_DEBUG "Enable debug mode" OFF)                     # Enable debug assertions, bound checks and other debug features. (Always enabled in Debug builds)

# Backend build options
option(MAGNETRON_ENABLE_BACKEND_CPU "Enable CPU backend" ON)                # Enable CPU backend
option(MAGNETRON_ENABLE_BACKEND_CUDA "Enable CUDA backend" ON)              # Enable CUDA backend (requires CUDA toolkit)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable C++ for required targets (Python bindings, tests, benchmarks)
# Magnetron's core and CPU backend are C 99, but Python bindings and tests/benchmarks require C++17
if (${MAGNETRON_BUILD_PYTHON_BINDINGS} OR ${MAGNETRON_BUILD_TESTS} OR ${MAGNETRON_BUILD_BENCHMARKS}})
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/arch.cmake)
include(cmake/target_config.cmake)

# TODO: This should go into extern/CMakeLists.txt but it needs to be loaded before the magnetron targets and extern is currently loaded after...
if (${MAGNETRON_BUILD_PYTHON_BINDINGS})
    #======================= nanobind + Python dev modules =======================
    add_subdirectory(extern/nanobind)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
endif()

if (${MAGNETRON_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${MAGNETRON_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

add_subdirectory(magnetron)
add_subdirectory(extern)

message(STATUS "magnetron configuration complete!")
