/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "prelude.hpp"

namespace mag::bindings {
    static std::once_flag g_ctx_once;
    static std::atomic<mag_context_t*> g_ctx{nullptr};
    static std::mutex g_mutex;

    mag_context_t *get_ctx() {
        std::call_once(g_ctx_once, [] {
            mag_context_t *ctx = mag_ctx_create("cpu:0");
            g_ctx.store(ctx, std::memory_order_release);
        });
        return g_ctx.load(std::memory_order_acquire);
    }

    std::mutex &get_global_mutex() { return g_mutex; }

    static void destroy_ctx(void *) noexcept {
        if (mag_context_t* ctx = g_ctx.exchange(nullptr, std::memory_order_acq_rel))
            mag_ctx_destroy(ctx, false);
    }

    void init_bindings_context(nb::module_ &m) {
        // Keep the context guard alive for the lifetime of the module
        // When the module unloads, the capsule destructor runs
        m.attr("_ctx_guard") = nb::capsule {reinterpret_cast<const void *>(1), &destroy_ctx};

        auto context = m.def_submodule(
            "context",
            "Global runtime controls (errors, RNG, CPU info, backend, etc.)."
        );
        context.def("start_grad_recorder", []() -> void {
            std::lock_guard lock {get_global_mutex()};
            mag_ctx_grad_recorder_start(get_ctx());
        });
        context.def("stop_grad_recorder", []() -> void {
            std::lock_guard lock {get_global_mutex()};
            mag_ctx_grad_recorder_stop(get_ctx());
        });
        context.def("is_grad_recording", []() -> bool {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_grad_recorder_is_running(get_ctx());
        });
        context.def("manual_seed", [](uint64_t seed) -> void {
            std::lock_guard lock {get_global_mutex()};
            mag_ctx_manual_seed(get_ctx(), seed);
        }, "seed"_a);
        context.def("os_name", []() -> std::string {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_os_name(get_ctx());
        });
        context.def("cpu_name", []() -> std::string {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_cpu_name(get_ctx());
        });
        context.def("cpu_virtual_cores", []() -> uint32_t {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_cpu_virtual_cores(get_ctx());
        });
        context.def("cpu_physical_cores", []() -> uint32_t {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_cpu_physical_cores(get_ctx());
        });
        context.def("cpu_sockets", []() -> uint32_t {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_cpu_sockets(get_ctx());
        });
        context.def("physical_memory_total", []() -> uint64_t {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_get_physical_memory_total(get_ctx());
        });
        context.def("physical_memory_free", []() -> uint64_t {
            std::lock_guard lock {get_global_mutex()};
           return mag_ctx_get_physical_memory_free(get_ctx());
        });
        context.def("is_numa_system", []() -> bool {
            std::lock_guard lock {get_global_mutex()};
            return mag_ctx_is_numa_system(get_ctx());
        });
    }
}
