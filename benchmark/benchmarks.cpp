// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.
// To supress sample stability warnings, add to environ: NANOBENCH_SUPPRESS_WARNINGS=1

#include <magnetron/magnetron.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

using namespace magnetron;

static auto bench_op(dtype type) -> void {
    ankerl::nanobench::Bench bench {};
    bench.title("matmul " + std::string{dtype_name(type)})
        .unit("matmul " + std::string{dtype_name(type)})
        .warmup(100)
        .performanceCounters(true);
    auto run_cycle {[&](std::int64_t numel) {
        context ctx {compute_device::cpu};
        tensor x {ctx, type, numel, numel};
        x.fill_float(1.0f);
        tensor y {ctx, type, numel, numel};
        y.fill_float(3.0f);

        bench.run(std::to_string(numel) + " elements", [&] {
            tensor r {x % y};
            ankerl::nanobench::doNotOptimizeAway(r);
        });
    }};
    run_cycle(10000);
    run_cycle(1000);
    run_cycle(750);
    run_cycle(500);
    run_cycle(250);
    run_cycle(100);
    run_cycle(10);
    run_cycle(4);
}

auto main() -> int {
    bench_op(dtype::e8m23);
    return 0;
}
