# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

add_executable(magnetron_benchmark benchmarks.cpp)
add_dependencies(magnetron_benchmark magnetron_runtime)
target_link_libraries(magnetron_benchmark magnetron_core)
target_include_directories(magnetron_benchmark PRIVATE ../include)
target_include_directories(magnetron_benchmark PRIVATE ../magnetron)
target_include_directories(magnetron_benchmark PRIVATE extern/nanobench/src/include)
