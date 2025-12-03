// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.
// To supress sample stability warnings, add to environ: NANOBENCH_SUPPRESS_WARNINGS=1

#include <../test/cpp/magnetron.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

using namespace magnetron;

auto main() -> int {
    ankerl::nanobench::Bench bench {};
    auto type = dtype::float32;
    bench.title("add " + std::string{dtype_name(type)})
        .unit("add " + std::string{dtype_name(type)})
        .warmup(100)
        .performanceCounters(true);
        context ctx {};
        tensor x {ctx, type, 2048, 2048};
        x.fill_(1.0f);
        tensor y {ctx, type, 2048, 2048};
        y.fill_(3.0f);

        tensor yT = y.permute({3, 2, 1});
        bench.run("add (non-cont)", [&] {
            tensor r {x + yT};
            ankerl::nanobench::doNotOptimizeAway(r);
        });
    return 0;
}
