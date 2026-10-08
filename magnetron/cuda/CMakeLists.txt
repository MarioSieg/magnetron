# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CUDA_SOURCES "*.cuh" "*.cu")

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND AND NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND CUDAToolkit_BIN_DIR)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
endif()

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MAGNETRON_CUDA_ARCHITECTURES "auto" CACHE STRING
        "CUDA architectures for the magnetron CUDA backend; 'auto' covers every supported GPU")

    if (MAGNETRON_CUDA_ARCHITECTURES STREQUAL "auto")
        set(MAG_CUDA_ARCHS 90-real)                                   # Hopper: H100, H200
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND MAG_CUDA_ARCHS 100-real 100-virtual)          # Blackwell datacenter + PTX for 10.x
            list(APPEND MAG_CUDA_ARCHS 120-real 120-virtual)          # Blackwell consumer + PTX for 12.x
        else()
            list(APPEND MAG_CUDA_ARCHS 90-virtual)                    # PTX for 9.x only
        endif()
    else()
        set(MAG_CUDA_ARCHS ${MAGNETRON_CUDA_ARCHITECTURES})
    endif()

    foreach (arch IN LISTS MAG_CUDA_ARCHS)
        string(REGEX MATCH "^[0-9]+" arch_num "${arch}")
        if (arch_num AND arch_num LESS 90)
            message(FATAL_ERROR
                "magnetron CUDA backend requires sm_90 or newer, got '${arch}'. The TMA matmul "
                "kernel emits cp.async.bulk.tensor, which ptxas rejects below sm_90.")
        endif()
    endforeach()

    message(STATUS "magnetron CUDA architectures: ${MAG_CUDA_ARCHS}")

    include(cmake/arch_specialization.cmake)
    set(MAG_CUDA_SPECIALIZED_SOURCES "")

    mag_register_cuda_arch(90)      # Hopper: H100, H200
    mag_register_cuda_arch(100      # Blackwell datacenter: B100, B200, GB200 (B300 via the sm_100f family target)
        arch/mag_cuda_gemm_sm100.cu # tcgen05 + TMEM GEMM behind matmul and implicit-GEMM convolution
    )
    mag_register_cuda_arch(120)     # Blackwell consumer: RTX 50x0

    add_library(magnetron_cuda SHARED ${MAGNETRON_CUDA_SOURCES} ${MAG_CUDA_ARCH_OBJECTS})
    set_property(TARGET magnetron_cuda PROPERTY CUDA_ARCHITECTURES ${MAG_CUDA_ARCHS})
    apply_common_config_to_target(magnetron_cuda FALSE)
    target_compile_options(magnetron_cuda PRIVATE -Xcompiler=-Wall,-Wextra,-fvisibility=hidden,-Wno-unused-parameter,-Wno-unused-function)
    target_link_libraries(magnetron_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_libraries(magnetron_cuda PRIVATE magnetron_core)
    target_include_directories(magnetron_cuda PRIVATE ../)
    if (MAG_CUDA_ARCH_MACROS)
        target_compile_definitions(magnetron_cuda PRIVATE ${MAG_CUDA_ARCH_MACROS})
    endif()

    mag_print_cuda_arch_summary()
else()
    message(WARNING "CUDA Toolkit not found, magnetron CUDA backend is disabled. To build the CUDA backend, install CUDA and ensure it is available in your PATH.")
    set(${MAGNETRON_ENABLE_BACKEND_CUDA} OFF)
endif()
