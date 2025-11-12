# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_CORE_SOURCES "*.c" "*.h" "../include/magnetron/*.h")

add_library(magnetron_core SHARED ${MAGNETRON_CORE_SOURCES})
apply_common_config_to_target(magnetron_core TRUE)
