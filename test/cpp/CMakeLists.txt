# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

add_subdirectory(extern/googletest)

file(GLOB_RECURSE TEST_SOURCES *.hpp *.cpp)
add_executable(magnetron_test ${TEST_SOURCES})
add_dependencies(magnetron_test magnetron_runtime)
target_link_libraries(magnetron_test magnetron_core)
target_include_directories(magnetron_test PRIVATE ../../magnetron)
target_include_directories(magnetron_test PRIVATE ../../include)
target_include_directories(magnetron_test PRIVATE extern/half/include)
target_include_directories(magnetron_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    magnetron_test
    GTest::gtest_main
    GTest::gmock
)
add_test(NAME magnetron_test COMMAND magnetron_test)

# The fusion tests assert that a generated kernel matches an op-by-op reference bit for bit.
# Generated kernels are built with -ffp-contract=off, so the reference has to be too: left to
# contract, the compiler folds the reference's multiply-add into an FMA and it lands one ULP away.
if (NOT MSVC)
    set_source_files_properties(core/fusion.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_precompile_headers(magnetron_test PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test/cpp/prelude.hpp>"
)

if (${MAGNETRON_ENABLE_BACKEND_CUDA})
    target_compile_definitions(magnetron_test PRIVATE MAG_ENABLE_CUDA)
endif()
