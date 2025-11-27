# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDINGS_SOURCES "*.cpp" "*.hpp")

nanobind_add_module(magnetron_bindings ${MAGNETRON_BINDINGS_SOURCES})
add_dependencies(magnetron_bindings magnetron_runtime)
target_link_libraries(magnetron_bindings PRIVATE magnetron_core)
target_include_directories(magnetron_bindings PRIVATE ../)
