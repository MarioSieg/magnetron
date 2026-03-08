# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDINGS_SOURCES "*.cpp" "*.hpp")

# The module is called magnetron as this is the Python package name.
nanobind_add_module(_magnetron_bindings NOMINSIZE ${MAGNETRON_BINDINGS_SOURCES})
target_include_directories(_magnetron_bindings PRIVATE "../") # Required to make #include <core/...> work from within core lol
target_include_directories(_magnetron_bindings PRIVATE "../../include")

set_target_properties(_magnetron_bindings PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS _magnetron_bindings
    LIBRARY DESTINATION magnetron
    COMPONENT python
)
