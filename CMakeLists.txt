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
option(MAGNETRON_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)     # Build Python bindings (requires C++ toolchain)
option(MAGNETRON_BUILD_TESTS "Build tests" ON)                      # Build C++ unit tests (requires C++ toolchain)
option(MAGNETRON_BUILD_BENCHMARKS "Build benchmaryks" ON)           # Build C++ benchmarks (requires C++ toolchain)
option(MAGNETRON_DEBUG "Enable debug mode" OFF)                     # Enable debug assertions, bound checks and other debug features. (Always enabled in Debug builds)

# Backend build options
option(MAGNETRON_ENABLE_BACKEND_CPU "Enable CPU backend" ON)                # Enable CPU backend
option(MAGNETRON_ENABLE_BACKEND_CUDA "Enable CUDA backend" OFF)              # Enable CUDA backend (requires CUDA toolkit)

# Magnetron's core and CPU backend is C99 but Python bindings, unit tests or benchmarks require C++ 20.
if (${MAGNETRON_BUILD_PYTHON_BINDINGS} OR ${MAGNETRON_BUILD_TESTS} OR ${MAGNETRON_BUILD_BENCHMARKS})
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/arch.cmake)
include(cmake/target_config.cmake)

if (${MAGNETRON_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${MAGNETRON_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

if (${MAGNETRON_BUILD_PYTHON_BINDINGS})
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()

    message("Building Magnetron Python bindings - Python interpreter module required...")
    find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

    add_subdirectory(extern/nanobind)
endif()

add_subdirectory(magnetron)
add_subdirectory(extern)

message(STATUS "magnetron configuration complete!")
