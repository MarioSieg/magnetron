# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.18)

project(magnetron LANGUAGES C)

message("Configuring magnetron for ${CMAKE_SYSTEM_PROCESSOR}...")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

set(CMAKE_C_STANDARD 99) # Use C99 standard
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

option(MAGNETRON_BUILD_TESTS "Build tests" ON)                           # Build C++ unit tests
option(MAGNETRON_BUILD_BENCHMARKS "Build benchmarks" ON)                 # Build C++ benchmarks
option(MAGNETRON_BUILD_FUZZERS "Build fuzzers" OFF)                      # (Experimental) Build C++ data format fuzzer
option(MAGNETRON_DEBUG "Enable debug mode" OFF)                          # Enable debug assertions, bound checks and other debug features. (Always enabled in Debug builds)
option(MAGNETRON_ENABLE_CUDA "Enable CUDA support" OFF)                  # Enable CUDA backend support (requires CUDA toolkit)
option(MAGNETRON_ENABLE_MIMALLOC "Use mimalloc as memory allocator" ON)  # Use mimalloc as memory allocator for faster memory allocation

set(MAGNETRON_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc" CACHE STRING "Path to the CUDA compiler") # Set to your CUDA compiler path, only used if CMAKE can't find the CUDA compiler on its own
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (${MAGNETRON_BUILD_TESTS} OR ${MAGNETRON_BUILD_BENCHMARKS} OR ${MAGNETRON_BUILD_FUZZERS})
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/arch.cmake)
include(cmake/lib.cmake)
include(cmake/compiler_config.cmake)

if (${MAGNETRON_ENABLE_CUDA})
    include(cmake/cuda.cmake)
endif()

if (${MAGNETRON_ENABLE_MIMALLOC})
    include(cmake/allocator.cmake)
endif()

if (${MAGNETRON_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${MAGNETRON_BUILD_FUZZERS})
    add_subdirectory(fuzzer)
endif()

if (${MAGNETRON_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

set_target_properties(magnetron PROPERTIES OUTPUT_NAME "magnetron")

set_target_properties(magnetron PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS magnetron
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/magnetron
    RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/magnetron
    ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/magnetron
)

message(STATUS "magnetron configuration complete!")
