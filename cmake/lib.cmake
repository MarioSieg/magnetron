# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

set(MAGNETRON_SOURCES
    include/magnetron/magnetron.h
    magnetron/magnetron.c
    magnetron/magnetron_aux.c
    magnetron/magnetron_cpu.c
    magnetron/magnetron_cpu_blas.inl
    magnetron/magnetron_cpu_blas_fallback.c
    magnetron/magnetron_internal.h
    magnetron/magnetron_device_registry.c
    magnetron/magnetron_ops.c
)

include(cmake/blas_tune.cmake)

add_library(magnetron SHARED ${MAGNETRON_SOURCES})
add_library(magnetron-static STATIC ${MAGNETRON_SOURCES})

target_include_directories(magnetron PUBLIC include)
target_include_directories(magnetron PRIVATE extern)
target_include_directories(magnetron-static PUBLIC include)
target_include_directories(magnetron-static PRIVATE extern)
