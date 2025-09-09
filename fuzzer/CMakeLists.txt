# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fuzzer requires LLVM as compiler
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Fuzzer requires Clang as compiler, current compiler is ${CMAKE_CXX_COMPILER_ID}")
endif()

add_executable(magnetron_fuzzer fuzzer.cpp)
target_compile_options(magnetron_fuzzer PRIVATE -fsanitize=fuzzer -O3 -march=native)
target_link_options(magnetron_fuzzer PRIVATE -fsanitize=fuzzer)
target_link_libraries(magnetron_fuzzer magnetron)
target_include_directories(magnetron_fuzzer PRIVATE ../include)
target_include_directories(magnetron_fuzzer PRIVATE ../test/cpp)
