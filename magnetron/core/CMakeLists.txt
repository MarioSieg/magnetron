# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CORE_SOURCES "*.c" "*.h" "../include/magnetron/*.h")

add_library(magnetron_core SHARED ${MAGNETRON_CORE_SOURCES})
target_include_directories(magnetron_core PRIVATE "../") # Required to make #include <core/...> work from within core lol
apply_common_config_to_target(magnetron_core TRUE)
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(mag_op_stubs.c PROPERTIES COMPILE_OPTIONS "-Wno-error=format-truncation")
endif()

install(TARGETS magnetron_core
    LIBRARY DESTINATION magnetron
    COMPONENT python
)
