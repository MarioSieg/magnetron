# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

include(CheckCCompilerFlag)

set(MAG_BACKEND_ROWS "")          # records: Name::Source::POSIX::MSVC::Status::Note
set(MAG_ENABLED_CPU_MACROS "")    # macros to define on target
set(MAG_CPU_SOURCES_ENABLED "")   # sources that will actually be compiled

function(mag_register_cpu_backend src posix_flags msvc_flags)
    set(CPU_NAME "unknown")
    set(DET_FLAG "")

    # Derive backend name + detection flag.
    # x86/ARM-style:
    #   -march=znver5
    # LoongArch-style:
    #   -msimd=lsx
    #   -msimd=lasx
    if (posix_flags MATCHES "-march=([^ ]+)")
        string(REGEX REPLACE ".*-march=([^ ]+).*" "\\1" CPU_NAME "${posix_flags}")
        set(DET_FLAG "-march=${CPU_NAME}")

    elseif (posix_flags MATCHES "-msimd=([^ ]+)")
        string(REGEX REPLACE ".*-msimd=([^ ]+).*" "\\1" CPU_NAME "${posix_flags}")
        set(DET_FLAG "-msimd=${CPU_NAME}")

    else()
        separate_arguments(_tmp_flags UNIX_COMMAND "${posix_flags}")
        if (_tmp_flags)
            list(GET _tmp_flags 0 DET_FLAG)
            set(CPU_NAME "${DET_FLAG}")
        endif()
    endif()

    if (ARGC GREATER 3 AND NOT "${ARGV3}" STREQUAL "")
        set(CPU_NAME "${ARGV3}")
    endif()

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" CPU_NAME_CLEAN "${CPU_NAME}")
    string(TOUPPER "${CPU_NAME_CLEAN}" CPU_NAME_UPPER)
    set(MACRO_NAME "MAG_HAVE_CPU_${CPU_NAME_UPPER}")

    if (WIN32)
        set(DET_FLAG "${msvc_flags}")
    endif()

    string(MAKE_C_IDENTIFIER "HAVE_${DET_FLAG}" HAVE_VAR)

    message(STATUS "Configuring CPU backend '${CPU_NAME}', Macro: ${MACRO_NAME}, Flags: POSIX='${posix_flags}' MSVC='${msvc_flags}'")

    check_c_compiler_flag("${DET_FLAG}" ${HAVE_VAR})
    set(SUPPORTED ${${HAVE_VAR}})

    set(status "Skipped")
    set(note "compiler doesn't support ${DET_FLAG}")

    if (SUPPORTED)
        if (WIN32)
            separate_arguments(_flags WINDOWS_COMMAND "${msvc_flags}")
        else()
            separate_arguments(_flags UNIX_COMMAND "${posix_flags}")
        endif()

        set(_accepted "")

        foreach(f IN LISTS _flags)
            if (f STREQUAL "")
                continue()
            endif()

            string(MAKE_C_IDENTIFIER "HAVE_${f}" HAVE_F)
            check_c_compiler_flag("${f}" ${HAVE_F})

            if (${HAVE_F})
                list(APPEND _accepted "${f}")
            endif()
        endforeach()

        if (_accepted)
            string(REPLACE ";" " " _accepted_str "${_accepted}")
            set_property(
                SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
                APPEND PROPERTY COMPILE_FLAGS "${_accepted_str}"
            )
        endif()

        set(status "Built")

        if (_accepted)
            set(note "enabled: ${_accepted_str}")
        else()
            set(note "no extra flags accepted")
        endif()

        list(APPEND MAG_CPU_SOURCES_ENABLED "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        list(APPEND MAG_ENABLED_CPU_MACROS "${MACRO_NAME}")

        set(MAG_CPU_SOURCES_ENABLED "${MAG_CPU_SOURCES_ENABLED}" PARENT_SCOPE)
        set(MAG_ENABLED_CPU_MACROS "${MAG_ENABLED_CPU_MACROS}" PARENT_SCOPE)
    endif()

    list(APPEND MAG_BACKEND_ROWS "${CPU_NAME}::${src}::${posix_flags}::${msvc_flags}::${status}::${note}")
    set(MAG_BACKEND_ROWS "${MAG_BACKEND_ROWS}" PARENT_SCOPE)
endfunction()

