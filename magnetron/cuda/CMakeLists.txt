# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CUDA_SOURCES "*.cuh" "*.cu")

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # ┌────────────────────────────┬──────────────────────────────────────────────────────────────┬──────────────────────────────┐
    # │ Compute Capability (SM)    │ GPU Architecture / Example                                   │ Key Features / Requirements   │
    # ├────────────────────────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────┤
    # │ 50                         │ Maxwell (e.g., GTX 900 series)                               │ Lowest supported by CUDA 12   │
    # │ 60                         │ Pascal (Tesla P100)                                          │ FP16 CUDA intrinsics          │
    # │ 61                         │ Pascal (GTX 10x0 series)                                     │ __dp4a per-byte int dot prod. │
    # │ 70                         │ Volta (Tesla V100)                                           │ FP16 tensor cores             │
    # │ 75                         │ Turing (RTX 20x0 / T4)                                       │ INT8 tensor cores             │
    # │ 80                         │ Ampere (A100, RTX 30x0)                                      │ Async copy, faster TCs        │
    # │ 86                         │ Ampere (RTX 30x0 consumer)                                   │ Needs CUDA ≥ 11.1             │
    # │ 89                         │ Ada Lovelace (RTX 40x0)                                      │ Needs CUDA ≥ 11.8             │
    # │ 90                         │ Blackwell (RTX 5000 / B100)                                  │ Needs CUDA ≥ 13.0             │
    # └────────────────────────────┴──────────────────────────────────────────────────────────────┴──────────────────────────────┘
    #
    # Suffix meanings:
    #   • “-virtual” → compile CUDA code as PTX only (JIT-compiled to binary at runtime)
    #   • “-real”    → compile as native device binary for that specific architecture
    #   • no suffix  → build both PTX and device code
    #
    # Default (non-native) behavior:
    #   • Build virtual architectures to cover all features for best performance
    #   • Build real architectures for the most common GPUs

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if (CUDAToolkit_VERSION VERSION_LESS "13")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
        endif ()
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
        endif()
    endif()

    add_library(magnetron_cuda SHARED ${MAGNETRON_CUDA_SOURCES})
    apply_common_config_to_target(magnetron_cuda FALSE)
    target_compile_options(magnetron_cuda PRIVATE "-Wall -Wextra -Werror -fvisibility=hidden -Wno-unused-parameter")
    target_link_libraries(magnetron_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_libraries(magnetron_cuda PRIVATE magnetron_core)
    target_include_directories(magnetron_cuda PRIVATE ../)
else()
    message(WARNING "CUDA Toolkit not found, magnetron CUDA backend is disabled. To build the CUDA backend, install CUDA and ensure it is available in your PATH.")
    set(${MAGNETRON_ENABLE_BACKEND_CUDA} OFF)
endif()
