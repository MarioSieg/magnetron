// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.
// To supress sample stability warnings, add to environ: NANOBENCH_SUPPRESS_WARNINGS=1

#include <../test/cpp/magnetron.hpp>

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
        tensor x {ctx, type, 1, 1, 768};
        x.fill(1.0f);
        tensor yy {ctx, type, 50257, 768};
        yy.fill(3.0f);
        tensor y = yy.transpose();
        std::cout << "x shape: ";
        for (auto d : x.shape()) std::cout << d << " ";
        std::cout << "\ny shape: ";
        for (auto d : y.shape()) std::cout << d << " ";
        std::cout << "\n";
        std::cout << "x contigous: " << x.is_contiguous() << "\n";
        std::cout << "y contigous: " << y.is_contiguous() << "\n";
        bench.run("matmul (AB^T)", [&] {
            tensor r {x % y};
            ankerl::nanobench::doNotOptimizeAway(r);
        });
    return 0;
}
