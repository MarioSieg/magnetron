# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#[===[

    -moutline-atomics
        Outline-atomics is a gcc compilation flag that adds runtime detection on if the cpu supports atomic instructions.
        Some older ARM CPU's such as the chip on the Raspberry PI 4 don't support atomic instructions. Using them will result in a SIGILL.
        When the outline-atomics flag is used, the compiler will generate code that checks if the CPU supports atomic instructions at runtime.
        CPUs that don't support atomic instructions will use the old load-exclusive/store-exclusive instructions.
        If a different compilation flag defined an architecture that unconditionally supports atomic instructions (e.g. -march=armv8.2), the outline-atomic flag will have no effect.
]===]

message("Configuring magnetron project for ${CMAKE_SYSTEM_PROCESSOR}...")
message("C compiler: ${CMAKE_C_COMPILER_ID}")

set(MAG_MSVC_COMPILE_FLAGS /W3 /Oi /arch:SSE2)
set(MAG_MSVC_RELEASE_COMPILE_FLAGS /O2 /Oy /Ot /Ob3 /RTC-)
set(MAG_MSVC_LINK_OPTIONS "")
set(MAG_MSVC_RELEASE_LINK_OPTIONS "")

set(MAG_CLANG_COMPILE_FLAGS -std=c99 -std=gnu99 -fvisibility=hidden -fno-math-errno -Wall -Werror -Wno-error=overflow -Wno-error=unused-function -Wno-unused-parameter -Wno-unused-function)
set(MAG_CLANG_RELEASE_COMPILE_FLAGS -O3 -flto=thin -fomit-frame-pointer)
set(MAG_CLANG_LINK_OPTIONS "")
set(MAG_CLANG_RELEASE_LINK_OPTIONS -flto=thin)

set(MAG_GCC_COMPILE_FLAGS -std=c99 -std=gnu99 -fvisibility=hidden -fno-math-errno -Wall -Werror -Wno-error=overflow -Wno-error=unused-function -Wno-error=format-truncation -Wno-unused-parameter -Wno-unused-function)
set(MAG_GCC_RELEASE_COMPILE_FLAGS -O3 -flto=auto -fomit-frame-pointer
)
set(MAG_GCC_LINK_OPTIONS "")
set(MAG_GCC_RELEASE_LINK_OPTIONS -flto=auto)

function(configure_mag_lib target_name)
   if(${IS_ARM64})
        if (NOT WIN32)
            add_compile_options(-moutline-atomics)
        endif()
        set(MAG_CLANG_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -march=armv8-a -moutline-atomics) # See beginning for file for info of -moutline-atomics
        set(MAG_GCC_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -march=armv8-a -moutline-atomics)
    elseif (${IS_AMD64})
        set(MAG_CLANG_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -msse -msse2)
        set(MAG_GCC_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -msse -msse2)
    endif()

    if(WIN32) # Windows (MSVC) specific config
        target_compile_options(${target_name} PRIVATE ${MAG_MSVC_COMPILE_FLAGS})
        target_link_options(${target_name} PRIVATE ${MAG_MSVC_LINK_OPTIONS})
        if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
            message(STATUS "! Generating optimized MAGNETRON release build")
            target_compile_options(${target_name} PRIVATE ${MAG_MSVC_RELEASE_COMPILE_FLAGS})
            target_link_options(${target_name} PRIVATE ${MAG_MSVC_RELEASE_LINK_OPTIONS})
        endif()
    else() # GCC/Clang specific config
        target_link_libraries(${target_name} m) # link math library
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target_name} PRIVATE ${MAG_GCC_COMPILE_FLAGS})
            target_link_options(${target_name} PRIVATE ${MAG_GCC_LINK_OPTIONS})
        else()
            target_compile_options(${target_name} PRIVATE ${MAG_CLANG_COMPILE_FLAGS})
            target_link_options(${target_name} PRIVATE ${MAG_CLANG_LINK_OPTIONS})
        endif()

        if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
            message(STATUS "Enabling release build optimizations for ${target_name}")
            if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
                target_compile_options(${target_name} PRIVATE ${MAG_GCC_RELEASE_COMPILE_FLAGS})
                target_link_options(${target_name} PRIVATE ${MAG_GCC_RELEASE_LINK_OPTIONS})
            else()
                target_compile_options(${target_name} PRIVATE ${MAG_CLANG_RELEASE_COMPILE_FLAGS})
                target_link_options(${target_name} PRIVATE ${MAG_CLANG_RELEASE_LINK_OPTIONS})
            endif()
        endif()

        # For some reasons, no symbols are exported with LDD, will investigate later
        # if(NOT APPLE) # Use LLD linker on non-Apple platforms for faster linking
        #    find_program(LLD_PATH NAMES ld.lld lld)
        #    if(LLD_PATH)
        #        message(STATUS "LLD found: ${LLD_PATH}")
        #        target_link_options(${target_name} PRIVATE -fuse-ld=lld)
        #        target_link_options(${target_name} PRIVATE "-Wl,--thinlto-cache-dir=${PROJECT_BINARY_DIR}/LTO.cache")
        #        target_link_options(${target_name} PRIVATE "-Wl,--thinlto-jobs=64")
        #    else()
        #        message(STATUS "LLD not found, using default linker")
        #    endif()
        #endif()

        if (${MAGNETRON_CPU_APPROX_MATH})
            target_compile_definitions(${target_name} PRIVATE MAG_APPROXMATH)
        endif()
        if (${MAGNETRON_DEBUG})
            target_compile_definitions(${target_name} PRIVATE MAG_DEBUG)
        endif()
    endif()

    if (WIN32) # Link sync lib on Win32
        target_link_libraries(${target_name} Synchronization.lib)
    endif()

    get_target_property(MAIN_CFLAGS ${target_name} COMPILE_OPTIONS)
    message(STATUS "${target_name} target flags: ${MAIN_CFLAGS}")
endfunction()

configure_mag_lib(magnetron)
configure_mag_lib(magnetron-static)
