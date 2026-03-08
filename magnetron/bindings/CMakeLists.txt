# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

file(GLOB MAGNETRON_BINDINGS_SOURCES "*.cpp" "*.hpp")

# The module is called magnetron as this is the Python package name.
message(STATUS "Adding binding module...")
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

# Generate .pyi stubs for IDE/type checker support (nanobind stubgen).
# INSTALL_TIME: run stubgen after the extension is installed so it can be imported.
message(STATUS "Generating .pyi stubs for _magnetron_bindings...")
nanobind_add_stub(magnetron_stubs
    INSTALL_TIME
    MODULE _magnetron_bindings
    OUTPUT magnetron/_magnetron_bindings.pyi
    PYTHON_PATH magnetron
    MARKER_FILE py.typed
    COMPONENT python
)
