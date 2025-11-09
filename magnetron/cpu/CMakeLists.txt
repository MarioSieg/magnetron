# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CPU_SOURCES "*.h" "*.c" "*.inl")

include(cmake/backend_specialization.cmake)
include(cmake/backend_summary.cmake)

if(${IS_AMD64})  # x86-64 specific compilation options
    include(amd64/amd64.cmake)
elseif(${IS_ARM64}) # arm64 specific compilation options
    include(arm64/arm64.cmake)
endif()

set(MAGNETRON_CPU_SOURCES ${MAGNETRON_CPU_SOURCES} ${MAG_CPU_SOURCES_ENABLED}) # Add them to the list of sources to compile

add_library(magnetron_cpu SHARED ${MAGNETRON_CPU_SOURCES})
apply_common_config_to_target(magnetron_cpu TRUE)
if (MAG_ENABLED_CPU_MACROS)
    target_compile_definitions(magnetron_cpu PRIVATE ${MAG_ENABLED_CPU_MACROS}) # Add macro definitions for compiled backends e.g. MAG_CPU_ZEN4, ...
endif()
target_link_libraries(magnetron_cpu PRIVATE magnetron_core)
target_include_directories(magnetron_cpu PRIVATE ../)

mag_print_cpu_backend_summary()
