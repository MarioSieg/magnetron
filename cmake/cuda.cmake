# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75")
    endif()

    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${MAGNETRON_CUDA_COMPILER}")
    endif()

    set(CMAKE_CUDA_STANDARD 17)
    enable_language(CUDA)
    file(GLOB_RECURSE MAGNETRON_CUDA_SOURCES magnetron/*.cu magnetron/*.cuh)
    add_library(magnetron_cuda SHARED ${MAGNETRON_CUDA_SOURCES})
    target_include_directories(magnetron_cuda PRIVATE include)
    target_include_directories(magnetron_cuda PRIVATE extern)
    target_link_libraries(magnetron_cuda CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(magnetron_cuda PRIVATE MAG_ENABLE_CUDA)
    target_compile_definitions(magnetron PRIVATE MAG_ENABLE_CUDA)
    target_compile_definitions(magnetron-static PRIVATE MAG_ENABLE_CUDA)
    target_link_libraries(magnetron magnetron_cuda)
    target_link_libraries(magnetron-static magnetron_cuda)
else()
    message(WARNING "CUDA not found, disabling CUDA support")
    set(MAGNETRON_ENABLE_CUDA OFF)
endif()