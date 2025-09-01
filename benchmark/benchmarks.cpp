// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.
// To supress sample stability warnings, add to environ: NANOBENCH_SUPPRESS_WARNINGS=1

#include <magnetron/magnetron.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

using namespace magnetron;

auto main() -> int {
    ankerl::nanobench::Bench bench {};
    auto type = dtype::e8m23;
    bench.title("matmul " + std::string{dtype_name(type)})
        .unit("matmul " + std::string{dtype_name(type)})
        .warmup(100)
        .performanceCounters(true);
        context ctx {compute_device::cpu};
        tensor x {ctx, type, 7, 768, 3072};
        x.fill_float(1.0f);
        tensor y {ctx, type, 7, 3072, 768};
        y.fill_float(3.0f);

        bench.run("matmul", [&] {
            tensor r {x % y};
            ankerl::nanobench::doNotOptimizeAway(r);
        });
    return 0;
}
