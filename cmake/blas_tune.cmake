# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

include(CheckCCompilerFlag)

function(add_source_flags_if_supported SRC) # Adds source optimization flags, only if the current compiler supports it
    if (ARGC LESS 2)
        message(FATAL_ERROR "add_source_flags_if_supported(<src> flag [flag …])")
    endif()
    set(src "${ARGV0}")
    list(REMOVE_AT ARGV 0)
    set(flags "${ARGV}")
    set(accepted)
    foreach(flag IN LISTS flags)
        string(MAKE_C_IDENTIFIER "HAVE_${flag}" flag_var)
        check_c_compiler_flag("${flag}" ${flag_var})
        if (${flag_var})
            list(APPEND accepted "${flag}")
        else()
            message(WARNING
              "!! [${src}] compiler does NOT understand \"${flag}\" – "
              "source will be built without it. "
              "Upgrade GCC/Clang to enable better optimizations!")
        endif()
    endforeach()
    if (accepted)
        string(REPLACE ";" " " accepted_str "${accepted}")
        set_property(SOURCE "${src}" APPEND PROPERTY COMPILE_FLAGS "${accepted_str}")
    endif()
endfunction()

function(set_blas_spec_arch filename posix_arch msvc_arch)
    message(STATUS "BLAS CPU permutation ${filename} ${posix_arch} / ${msvc_arch}")
    if (WIN32)
        separate_arguments(flag_list WINDOWS_COMMAND "${msvc_arch}")
    else()
        separate_arguments(flag_list UNIX_COMMAND "${posix_arch}")
    endif()
    add_source_flags_if_supported("${CMAKE_SOURCE_DIR}/magnetron/${filename}" ${flag_list})
endfunction()

set(MAG_BLAS_SPEC_AMD64_SOURCES
    magnetron/magnetron_cpu_blas_amd64_v2.c
    magnetron/magnetron_cpu_blas_amd64_v2_5.c
    magnetron/magnetron_cpu_blas_amd64_v3.c
    magnetron/magnetron_cpu_blas_amd64_v4.c
    magnetron/magnetron_cpu_blas_amd64_v4_5.c
)

set(MAG_BLAS_SPEC_ARM64_SOURCES
    magnetron/magnetron_cpu_blas_arm64_v8_2.c
    magnetron/magnetron_cpu_blas_arm64_v9.c
)

if(${IS_AMD64})  # x86-64 specific compilation options
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAG_BLAS_SPEC_AMD64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v2.c"
        "-mtune=nehalem -mcx16 -mpopcnt -msse3 -mssse3 -msse4.1 -msse4.2"
        "/arch:SSE4.2")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v2_5.c"
        "-mtune=ivybridge -mavx -mf16c -mno-avx2 -mcx16 -mpopcnt -msse3 -mssse3 -msse4.1 -msse4.2"
        "/arch:AVX")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v3.c"
        "-mtune=haswell -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe"
        "/arch:AVX2 /D__BMI__=1 /D__BMI2__=1 /D__F16C__=1 /D__FMA__=1") # MSVC is just annoying
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v4.c"
        "-mtune=cannonlake -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe"
        "/arch:AVX512")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v4_5.c"
        "-mtune=generic -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mavx512fp16 -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe"
        "/arch:AVX512")
elseif(${IS_ARM64})
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAG_BLAS_SPEC_ARM64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_arm64_v8_2.c" "-march=armv8.2-a+dotprod+fp16" "")
    if (NOT APPLE)
        # On Apple arm64, enabling SVE crashes LLVM with some error: LLVM ERROR: Cannot select: 0x10f4f5200: nxv4i32 = AArch64ISD::SUNPKHI 0x10f4f4a20
        # So we disable SVE for now, as no Apple silicon Mac supports SVE yet.
        set_blas_spec_arch("magnetron_cpu_blas_arm64_v9.c" "-march=armv9-a+sve+sve2" "")
    endif()
endif()