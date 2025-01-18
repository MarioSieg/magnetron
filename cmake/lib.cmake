# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

set(MAGNETRON_SOURCES
    magnetron/magnetron.h
    magnetron/magnetron.c
    magnetron/magnetron_cpu.cpp
    magnetron/magnetron_cpu_blas.inl
    magnetron/magnetron_cpu_blas_fallback.c
    magnetron/magnetron_internal.h
    magnetron/magnetron_device_registry.c
)

include(cmake/blas_tune.cmake)

if (${MAGNETRON_BUILD_SHARED})
    add_library(magnetron SHARED ${MAGNETRON_SOURCES})
else()
    add_library(magnetron STATIC ${MAGNETRON_SOURCES})
endif()

target_include_directories(magnetron PRIVATE extern)
