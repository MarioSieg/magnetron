# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else()
    set(IS_AMD64 FALSE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)|(ARM64)")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()
