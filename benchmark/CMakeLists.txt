# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

add_executable(magnetron_benchmark benchmarks.cpp)
target_link_libraries(magnetron_benchmark magnetron)
target_include_directories(magnetron_benchmark PRIVATE ../include)
target_include_directories(magnetron_benchmark PRIVATE extern/nanobench/src/include)
