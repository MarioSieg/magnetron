# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CUDA_SOURCES "*.cuh" "*.cu")

find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc) # Set to your CUDA compiler path if not found automatically
endif()

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # ┌────────────────────────────┬──────────────────────────────────────────────────────────────┬──────────────────────────────┐
    # │ Compute Capability (SM)    │ GPU Architecture / Example                                   │ Key Features / Requirements   │
    # ├────────────────────────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────┤
    # │ 75                         │ Turing (RTX 20x0 / T4)                                       │ INT8 tensor cores             │
    # │ 80                         │ Ampere (A100, RTX 30x0)                                      │ Async copy, bf16 tensor cores │
    # │ 86                         │ Ampere (RTX 30x0 consumer)                                   │ Needs CUDA ≥ 11.1             │
    # │ 89                         │ Ada Lovelace (RTX 40x0)                                      │ Needs CUDA ≥ 11.8             │
    # │ 90                         │ Hopper (H100, H200)                                          │ TMA, mbarrier; CUDA ≥ 11.8    │
    # │ 100                        │ Blackwell datacenter (B100, B200, GB200)                     │ Needs CUDA ≥ 12.8             │
    # │ 103                        │ Blackwell datacenter (B300, GB300)                           │ Needs CUDA ≥ 12.9             │
    # │ 120                        │ Blackwell consumer (RTX 50x0)                                │ Needs CUDA ≥ 12.8             │
    # └────────────────────────────┴──────────────────────────────────────────────────────────────┴──────────────────────────────┘
    #
    # Suffix meanings:
    #   • “-virtual” → compile CUDA code as PTX only (JIT-compiled to binary at runtime)
    #   • “-real”    → compile as native device binary for that specific architecture
    #   • no suffix  → build both PTX and device code
    #
    # sm_90 is a hard floor for this backend, not a tuning choice: the TMA/WMMA matmul kernel
    # in mag_cuda_matmul.cu emits cp.async.bulk.tensor, which ptxas rejects below sm_90.
    #
    # Blackwell is split across two incompatible families - datacenter (10.x: B100/B200/B300)
    # and consumer (12.x: RTX 50x0). Neither cubins nor PTX cross that boundary: compute_120
    # PTX cannot JIT onto an sm_100 B200, and compute_100 PTX cannot JIT onto an sm_120 RTX
    # 50x0. Both families therefore need their own real and virtual entries. Getting this
    # wrong is silent at build time and only shows up as cudaErrorNoKernelImageForDevice on
    # the machine that was left out.
    #
    # This list applies to the generic sources, which are compiled once as fat binaries.
    # Kernels that need genuinely different code per architecture go through
    # mag_register_cuda_arch() below instead.
    #
    # Set MAGNETRON_CUDA_ARCHITECTURES to cut build time for a known target, e.g.
    # -DMAGNETRON_CUDA_ARCHITECTURES="100-real" for a B200-only build. The default covers
    # every GPU this backend can run on.
    #
    # Note this deliberately does not key off CMAKE_CUDA_ARCHITECTURES: enable_language(CUDA)
    # initializes and caches that variable itself (CMP0104), so after the first configure it
    # is always set and "did the user ask for this?" becomes undecidable. The architectures
    # are applied as a target property below rather than through the global variable.

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

    # Sources compiled once per registered architecture, each in its own namespace, and
    # selected at runtime by compute capability. Keep these under arch/ so the non-recursive
    # file(GLOB "*.cu") above does not also compile them into the generic target.
    #
    #   set(MAG_CUDA_SPECIALIZED_SOURCES arch/mag_cuda_matmul_arch.cu)
    #
    # See cmake/arch_specialization.cmake for the namespace and dispatch contract.
    set(MAG_CUDA_SPECIALIZED_SOURCES "")

    mag_register_cuda_arch(90)      # Hopper: H100, H200
    mag_register_cuda_arch(100)     # Blackwell datacenter: B100, B200, GB200
    mag_register_cuda_arch(120)     # Blackwell consumer: RTX 50x0

    add_library(magnetron_cuda SHARED ${MAGNETRON_CUDA_SOURCES} ${MAG_CUDA_ARCH_OBJECTS})
    set_property(TARGET magnetron_cuda PROPERTY CUDA_ARCHITECTURES ${MAG_CUDA_ARCHS})
    apply_common_config_to_target(magnetron_cuda FALSE)
    target_compile_options(magnetron_cuda PRIVATE "-Wall -Wextra -Werror -fvisibility=hidden -Wno-unused-parameter")
    target_link_libraries(magnetron_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_libraries(magnetron_cuda PRIVATE magnetron_core)
    target_include_directories(magnetron_cuda PRIVATE ../)
    if (MAG_CUDA_ARCH_MACROS) # Tell the runtime dispatcher which specializations exist
        target_compile_definitions(magnetron_cuda PRIVATE ${MAG_CUDA_ARCH_MACROS})
    endif()

    mag_print_cuda_arch_summary()
else()
    message(WARNING "CUDA Toolkit not found, magnetron CUDA backend is disabled. To build the CUDA backend, install CUDA and ensure it is available in your PATH.")
    set(${MAGNETRON_ENABLE_BACKEND_CUDA} OFF)
endif()
