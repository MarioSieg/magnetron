# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDINGS_SOURCES "*.cpp" "*.hpp")

# The module is called magnetron as this is the Python package name.
nanobind_add_module(_magnetron NOMINSIZE ${MAGNETRON_BINDINGS_SOURCES})
target_include_directories(_magnetron PRIVATE "../") # Required to make #include <core/...> work from within core lol
target_include_directories(_magnetron PRIVATE "../../include")

set_target_properties(_magnetron PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS _magnetron
    LIBRARY DESTINATION magnetron
    COMPONENT python
)
