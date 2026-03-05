# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

add_subdirectory(core)

# Magnetron runtime umbrella target, which depends on the core and all backends
add_custom_target(magnetron_runtime ALL
    DEPENDS magnetron_core
)

function(add_backend name)
    message(STATUS "Configuring magnetron backend module: ${name}")
    add_subdirectory("${name}")
    if (TARGET "magnetron_${name}")
        add_dependencies(magnetron_runtime "magnetron_${name}")
        install(TARGETS "magnetron_${name}"
            LIBRARY DESTINATION magnetron
            COMPONENT python
        )
    endif()
endfunction()

if (${MAGNETRON_ENABLE_BACKEND_CPU})
    add_backend(cpu)
endif()
if (${MAGNETRON_ENABLE_BACKEND_CUDA})
    add_backend(cuda)
endif()

# Optional Python bindings
if (${MAGNETRON_BUILD_PYTHON_BINDINGS})
    add_subdirectory(bindings)
    add_dependencies(_magnetron magnetron_runtime) # _magnetron is name of the Python bindings target
    target_link_libraries(_magnetron PRIVATE magnetron_core)
    set_target_properties(_magnetron PROPERTIES
    MACOSX_RPATH ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@loader_path"
    )
endif()
