// (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.
// To supress sample stability warnings, add to environ: NANOBENCH_SUPPRESS_WARNINGS=1

#include <../test/cpp/magnetron.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

using namespace magnetron;

int main() {
    ankerl::nanobench::Bench bench {};
    bench.title("BMM_VGEM " + std::string{dtype_name(dtype::bfloat16)})
        .unit("BMM_VGEM " + std::string{dtype_name(dtype::bfloat16)})
        .warmup(200)
        .minEpochIterations(200)
        .performanceCounters(true);
        context ctx {};
        tensor x {ctx, dtype::bfloat16, 1, 1, 2560};
        x.fill_(1.0f);
        tensor scale {ctx, dtype::float32, 1};
        scale.fill_(1.0f);
        tensor y {ctx, dtype::float8_e4m3fn, 9728, 2560};
        y.fill_(3.0f);
        tensor yT = y.transpose();
        bench.run("fp8_scaled_gemm BMM VGEM S", [&] {
            tensor r {x.scaled_mm(yT, scale)};
            ankerl::nanobench::doNotOptimizeAway(r);
        });
    return 0;
}
