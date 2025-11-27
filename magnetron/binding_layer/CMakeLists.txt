# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDING_LAYER_SOURCES "*.cpp" "*.hpp")

nanobind_add_module(magnetron_binding_layer ${MAGNETRON_BINDING_LAYER_SOURCES})
add_dependencies(magnetron_binding_layer magnetron_runtime)
target_link_libraries(magnetron_binding_layer PRIVATE magnetron_core)
target_include_directories(magnetron_binding_layer PRIVATE ../)
