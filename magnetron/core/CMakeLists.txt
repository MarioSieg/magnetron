# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CORE_SOURCES "*.c" "*.h" "../include/magnetron/*.h")

add_library(magnetron_core SHARED ${MAGNETRON_CORE_SOURCES})
target_include_directories(magnetron_core PRIVATE "../") # Required to make #include <core/...> work from within core lol
apply_common_config_to_target(magnetron_core TRUE)
