# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

set(MAG_CUDA_ARCH_OBJECTS "")
set(MAG_CUDA_ARCH_MACROS "")
set(MAG_CUDA_ARCH_ROWS "")

function(_mag_cuda_min_toolkit sm out)
    if (sm GREATER_EQUAL 110)
        set(ver "13.0")
    elseif (sm GREATER_EQUAL 103)
        set(ver "12.9")
    elseif (sm GREATER_EQUAL 100)
        set(ver "12.8")
    else()
        set(ver "11.8")
    endif()
    set(${out} "${ver}" PARENT_SCOPE)
endfunction()

function(mag_register_cuda_arch sm)
    set(srcs ${ARGN})
    if (NOT srcs)
        set(srcs ${MAG_CUDA_SPECIALIZED_SOURCES})
    endif()
    set(arch_suffix "")
    set(family_macro "")
    if (sm EQUAL 100)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
            set(arch_suffix "f")
            set(family_macro "MAG_CUDA_SM100_FAMILY=1")
        else()
            set(arch_suffix "a")
            set(family_macro "MAG_CUDA_SM100_FAMILY=0")
        endif()
    endif()

    set(status "Skipped")
    set(note "unknown")

    _mag_cuda_min_toolkit(${sm} min_ver)

    if (sm LESS 90)
        set(note "below the sm_90 floor of this backend")
    elseif (CUDAToolkit_VERSION VERSION_LESS "${min_ver}")
        set(note "needs CUDA >= ${min_ver}, have ${CUDAToolkit_VERSION}")
    elseif (NOT srcs)
        set(note "no specialized sources registered yet")
    else()
        set(tgt magnetron_cuda_sm${sm})

        add_library(${tgt} OBJECT ${srcs})
        set_target_properties(${tgt} PROPERTIES
            CUDA_ARCHITECTURES OFF
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_compile_options(${tgt} PRIVATE
            "--generate-code=arch=compute_${sm}${arch_suffix},code=[sm_${sm}${arch_suffix}]"
            -Xcompiler=-Wall,-Wextra,-fvisibility=hidden,-Wno-unused-parameter,-Wno-unused-function
            "SHELL:-diag-suppress 20012,3288"
        )
        target_compile_definitions(${tgt} PRIVATE
            MAG_CUDA_SM=${sm}
            MAG_CUDA_ARCH_NS=sm_${sm}
            ${family_macro}
        )
        target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
        target_link_libraries(${tgt} PRIVATE magnetron_core CUDA::cudart CUDA::cuda_driver)
        apply_common_config_to_target(${tgt} FALSE)

        list(APPEND MAG_CUDA_ARCH_OBJECTS "$<TARGET_OBJECTS:${tgt}>")
        list(APPEND MAG_CUDA_ARCH_MACROS "MAG_HAVE_CUDA_SM_${sm}")
        set(MAG_CUDA_ARCH_OBJECTS "${MAG_CUDA_ARCH_OBJECTS}" PARENT_SCOPE)
        set(MAG_CUDA_ARCH_MACROS "${MAG_CUDA_ARCH_MACROS}" PARENT_SCOPE)

        list(LENGTH srcs nsrc)
        set(status "Built")
        set(note "${nsrc} source(s) -> ${tgt} (sm_${sm}${arch_suffix})")
    endif()

    list(APPEND MAG_CUDA_ARCH_ROWS "${sm}::${status}::${note}")
    set(MAG_CUDA_ARCH_ROWS "${MAG_CUDA_ARCH_ROWS}" PARENT_SCOPE)
endfunction()

function(mag_print_cuda_arch_summary)
    if (NOT MAG_CUDA_ARCH_ROWS)
        return()
    endif()
    message(STATUS "magnetron CUDA arch specializations:")
    foreach (row IN LISTS MAG_CUDA_ARCH_ROWS)
        string(REPLACE "::" ";" fields "${row}")
        list(GET fields 0 sm)
        list(GET fields 1 status)
        list(GET fields 2 note)
        message(STATUS "  sm_${sm}\t${status}\t${note}")
    endforeach()
endfunction()
