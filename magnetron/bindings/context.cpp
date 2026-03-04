/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "prelude.hpp"

static std::once_flag g_ctx_once;
static std::atomic<mag_context_t*> g_ctx{nullptr};

mag_context_t *get_ctx() {
    std::call_once(g_ctx_once, [] {
        mag_context_t *ctx = mag_ctx_create("cpu:0");
        g_ctx.store(ctx, std::memory_order_release);
    });
    return g_ctx.load(std::memory_order_acquire);
}

static void destroy_ctx(void *) noexcept {
    if (mag_context_t* ctx = g_ctx.exchange(nullptr, std::memory_order_acq_rel))
        mag_ctx_destroy(ctx, false);
}

void mag_init_bindings_context(nb::module_ &m) {
    // Keep the context guard alive for the lifetime of the module
    // When the module unloads, the capsule destructor runs
    m.attr("_ctx_guard") = nb::capsule {reinterpret_cast<const void *>(1), &destroy_ctx};

    auto context = m.def_submodule(
        "context",
        "Global runtime controls (errors, RNG, CPU info, backend, etc.)."
    );
    context.def("last_error", []() -> int {
        return mag_ctx_get_error_code(get_ctx());
    });
    context.def("has_error", []() -> bool {
        return mag_ctx_has_error(get_ctx());
    });
    context.def("clear_error", []() -> void {
        mag_ctx_clear_error(get_ctx());
    });
    context.def("last_error_name", []() -> std::string {
        return mag_status_get_name(mag_ctx_get_error_code(get_ctx()));
    });
    context.def("start_grad_recorder", []() -> void {
        mag_ctx_grad_recorder_start(get_ctx());
    });
    context.def("stop_grad_recorder", []() -> void {
        mag_ctx_grad_recorder_stop(get_ctx());
    });
    context.def("is_grad_recording", []() -> bool {
        return mag_ctx_grad_recorder_is_running(get_ctx());
    });
    context.def("manual_seed", [](uint64_t seed) -> void {
        mag_ctx_manual_seed(get_ctx(), seed);
    }, "seed"_a);
    context.def("os_name", []() -> std::string {
       return mag_ctx_get_os_name(get_ctx());
    });
    context.def("cpu_name", []() -> std::string {
        return mag_ctx_get_cpu_name(get_ctx());
    });
    context.def("cpu_virtual_cores", []() -> uint32_t {
        return mag_ctx_get_cpu_virtual_cores(get_ctx());
    });
    context.def("cpu_physical_cores", []() -> uint32_t {
        return mag_ctx_get_cpu_physical_cores(get_ctx());
    });
    context.def("cpu_sockets", []() -> uint32_t {
        return mag_ctx_get_cpu_sockets(get_ctx());
    });
    context.def("physical_memory_total", []() -> uint64_t {
        return mag_ctx_get_physical_memory_total(get_ctx());
    });
    context.def("physical_memory_free", []() -> uint64_t {
       return mag_ctx_get_physical_memory_free(get_ctx());
    });
    context.def("is_numa_system", []() -> bool {
        return mag_ctx_is_numa_system(get_ctx());
    });
}
