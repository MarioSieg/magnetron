// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

// ON LINUX: Before running the benchmark, execute: prepare_system.sh to setup the system for performance measurements.

#include <magnetron.h>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include <thread>

#include "magnetron_internal.h"

static auto bench_cpu_compute(std::int64_t numel_per_dim) -> void {
    ankerl::nanobench::Bench bench {};
    bench.title("Parallel MM Big Tensor | Numel per Dim: " + std::to_string(numel_per_dim))
        .unit("MM")
        .warmup(100)
        .relative(true)
        .performanceCounters(true);

    std::cout << "Benchmarking Parallel MM on CPU with Numel per Dim: " << numel_per_dim << std::endl;

    auto exec_bench = [&](std::uint32_t threads) {
        mag_device_descriptor_t desc {};
        desc.type = MAG_COMPUTE_DEVICE_TYPE_CPU;
        desc.thread_count = threads;
        mag_ctx_t* ctx = mag_ctx_create2(&desc);
        mag_tensor_t* A = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, numel_per_dim, numel_per_dim);
        mag_tensor_fill_random_normal(A, 0.0f, 1.0f);
        mag_tensor_t* B = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, numel_per_dim, numel_per_dim);
        mag_tensor_fill_random_normal(B, 0.0f, 1.0f);
        bench.run("Parallel MM on " + std::to_string(threads) + " threads, Elems = " + std::to_string(A->numel), [&] {
            mag_tensor_t* R = mag_add(A, B);
            ankerl::nanobench::doNotOptimizeAway(R);
            mag_tensor_decref(R);
        });

        ankerl::nanobench::doNotOptimizeAway(ctx);
        mag_tensor_decref(B);
        mag_tensor_decref(A);
        mag_ctx_destroy(ctx);
    };

    std::uint32_t num_threads = std::max(1u, std::thread::hardware_concurrency());

    for (std::uint32_t i=1; i <= num_threads;) {
        exec_bench(i);
        if (i == 1) ++i;
        else i += 2;
    }
}

auto main() -> int {
    //bench_cpu_compute(10000);
    bench_cpu_compute(1000);
    bench_cpu_compute(750);
    bench_cpu_compute(500);
    bench_cpu_compute(250);
    bench_cpu_compute(100);
    bench_cpu_compute(10);
    bench_cpu_compute(2);
    //bench_cpu_compute(250);
    return 0;
}
