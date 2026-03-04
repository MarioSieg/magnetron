# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDINGS_SOURCES "*.cpp" "*.hpp")

# The module is called magnetron as this is the Python package name.
nanobind_add_module(magnetron NOMINSIZE ${MAGNETRON_BINDINGS_SOURCES})
target_include_directories(magnetron PRIVATE "../") # Required to make #include <core/...> work from within core lol
target_include_directories(magnetron PRIVATE "../../include")
