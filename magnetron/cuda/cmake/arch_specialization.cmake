# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

# Per-architecture specialization for the CUDA backend.
#
# The CPU backend attaches -march= to individual source files, because that is a source-level
# flag. CUDA architectures are not: CUDA_ARCHITECTURES is a *target* property and applies to
# every source in the target. Compiling one .cu against several architectures therefore needs
# one OBJECT library per architecture, all linked into magnetron_cuda.
#
# Each specialization compiles the registered sources with:
#
#   MAG_CUDA_SM      = <sm>       e.g. 100
#   MAG_CUDA_ARCH_NS = sm_<sm>    e.g. sm_100
#
# Specialized sources must wrap their kernels in `namespace MAG_CUDA_ARCH_NS { ... }`, so each
# per-arch copy gets distinct mangled names rather than colliding at link time:
#
#   namespace mag::MAG_CUDA_ARCH_NS {
#     mag_status_t misc_op_matmul(mag_error_t *err, const mag_command_t &cmd) { ... }
#   }
#
# The dispatcher in mag_cuda.cu then picks a variant at runtime from
# physical_device::compute_capability(), guarded by the MAG_HAVE_CUDA_SM_<sm> macros that get
# defined on the magnetron_cuda target:
#
#   #ifdef MAG_HAVE_CUDA_SM_100
#     if (cc >= 1000 && cc < 1200) return mag::sm_100::misc_op_matmul(err, cmd);
#   #endif
#
# Specialized sources belong in a subdirectory (arch/), NOT next to the generic ones: the
# file(GLOB "*.cu") in CMakeLists.txt is non-recursive, so arch/*.cu stays out of the generic
# target and does not get compiled twice.

set(MAG_CUDA_ARCH_OBJECTS "")   # $<TARGET_OBJECTS:...> to fold into magnetron_cuda
set(MAG_CUDA_ARCH_MACROS "")    # MAG_HAVE_CUDA_SM_<sm> for the runtime dispatcher
set(MAG_CUDA_ARCH_ROWS "")      # records: SM::Status::Note

# Oldest CUDA toolkit that knows a given compute capability.
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

# mag_register_cuda_arch(<sm> [sources...])
#
# Registers an architecture-specialized object library. Sources default to
# MAG_CUDA_SPECIALIZED_SOURCES. Unsupported or unbuildable architectures are recorded and
# skipped rather than failing the configure, so a toolkit that predates a GPU still builds.
function(mag_register_cuda_arch sm)
    set(srcs ${ARGN})
    if (NOT srcs)
        set(srcs ${MAG_CUDA_SPECIALIZED_SOURCES})
    endif()

    set(status "Skipped")
    set(note "unknown")

    _mag_cuda_min_toolkit(${sm} min_ver)

    if (sm LESS 90)
        # Not a tuning choice: the TMA matmul kernel emits cp.async.bulk.tensor, which ptxas
        # rejects below sm_90.
        set(note "below the sm_90 floor of this backend")
    elseif (CUDAToolkit_VERSION VERSION_LESS "${min_ver}")
        set(note "needs CUDA >= ${min_ver}, have ${CUDAToolkit_VERSION}")
    elseif (NOT srcs)
        set(note "no specialized sources registered yet")
    else()
        set(tgt magnetron_cuda_sm${sm})

        add_library(${tgt} OBJECT ${srcs})
        set_target_properties(${tgt} PROPERTIES
            CUDA_ARCHITECTURES "${sm}-real"
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_compile_definitions(${tgt} PRIVATE
            MAG_CUDA_SM=${sm}
            MAG_CUDA_ARCH_NS=sm_${sm}
        )
        target_compile_options(${tgt} PRIVATE "-Wall -Wextra -Werror -fvisibility=hidden -Wno-unused-parameter")
        target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
        target_link_libraries(${tgt} PRIVATE magnetron_core CUDA::cudart CUDA::cuda_driver)

        list(APPEND MAG_CUDA_ARCH_OBJECTS "$<TARGET_OBJECTS:${tgt}>")
        list(APPEND MAG_CUDA_ARCH_MACROS "MAG_HAVE_CUDA_SM_${sm}")
        set(MAG_CUDA_ARCH_OBJECTS "${MAG_CUDA_ARCH_OBJECTS}" PARENT_SCOPE)
        set(MAG_CUDA_ARCH_MACROS "${MAG_CUDA_ARCH_MACROS}" PARENT_SCOPE)

        list(LENGTH srcs nsrc)
        set(status "Built")
        set(note "${nsrc} source(s) -> ${tgt}")
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
