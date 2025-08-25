# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fuzzer requires LLVM as compiler
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Fuzzer requires Clang as compiler")
endif()

add_executable(msml_fuzzer fuzzer.cpp)
target_compile_options(msml_fuzzer PRIVATE -fsanitize=fuzzer -O3 -march=native)
target_link_options(msml_fuzzer PRIVATE -fsanitize=fuzzer)
target_link_libraries(msml_fuzzer msml)
target_include_directories(msml_fuzzer PRIVATE ../msml)
