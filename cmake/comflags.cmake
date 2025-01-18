# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

set(MAG_MSVC_COMPILE_FLAGS
    /W3
    /Oi
    /arch:SSE2
)
set(MAG_MSVC_RELEASE_COMPILE_FLAGS
    /O2
    /Oy
    /Ot
    /Ob3
)
set(MAG_MSVC_LINK_OPTIONS "")
set(MAG_MSVC_RELEASE_LINK_OPTIONS "")

set(MAG_CLANG_COMPILE_FLAGS
    -fvisibility=hidden
    -Wall
    -Werror
    -Wno-gnu-zero-variadic-macro-arguments
    -Wno-error=overflow
    -Wno-error=unused-function
)
set(MAG_CLANG_RELEASE_COMPILE_FLAGS
    -O3
    -flto
)
set(MAG_CLANG_LINK_OPTIONS "")
set(MAG_CLANG_RELEASE_LINK_OPTIONS -flto)

set(MAG_GCC_COMPILE_FLAGS
    -fvisibility=hidden
    -Wall
    -Werror
    -Wno-gnu-zero-variadic-macro-arguments
    -Wno-error=overflow
    -Wno-error=unused-function
    -Wno-error=format-truncation
)
set(MAG_GCC_RELEASE_COMPILE_FLAGS
    -O3
    -flto
    -fomit-frame-pointer
)
set(MAG_GCC_LINK_OPTIONS "")
set(MAG_GCC_RELEASE_LINK_OPTIONS -flto)

if (WIN32) # Windows (MSVC) specific config
    target_compile_options(magnetron PRIVATE ${MAG_MSVC_COMPILE_FLAGS})
    target_link_options(magnetron PRIVATE ${MAG_MSVC_LINK_OPTIONS})
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
        target_compile_options(magnetron PRIVATE ${MAG_MSVC_RELEASE_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_MSVC_RELEASE_LINK_OPTIONS})
    endif()
else() # GCC/Clang specific config
    target_link_libraries(magnetron m) # link math library

    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(magnetron PRIVATE ${MAG_GCC_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_GCC_LINK_OPTIONS})
    else()
        target_compile_options(magnetron PRIVATE ${MAG_CLANG_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_CLANG_LINK_OPTIONS})
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(magnetron PRIVATE ${MAG_GCC_RELEASE_COMPILE_FLAGS})
            target_link_options(magnetron PRIVATE ${MAG_GCC_RELEASE_LINK_OPTIONS})
        else()
            target_compile_options(magnetron PRIVATE ${MAG_CLANG_RELEASE_COMPILE_FLAGS})
            target_link_options(magnetron PRIVATE ${MAG_CLANG_RELEASE_LINK_OPTIONS})
        endif()
    endif()

    if (${MAGNETRON_BUILD_SHARED})
        target_compile_definitions(magnetron PRIVATE MAG_SHARED)
    endif()
    if (${MAGNETRON_CPU_APPROX_MATH})
        target_compile_definitions(magnetron PRIVATE MAG_APPROXMATH)
    endif()
    if (${MAGNETRON_DEBUG})
        target_compile_definitions(magnetron PRIVATE MAG_DEBUG)
    endif()
endif()