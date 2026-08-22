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

#include <atomic>

#include "core/mag_context.h"

namespace mag::bindings {
  static std::once_flag g_ctx_once;
  static std::atomic<mag_context_t*> g_ctx{nullptr};
  static std::mutex g_mutex;

  mag_context_t *get_ctx() {
    std::call_once(g_ctx_once, [] {
      mag_context_t *ctx = nullptr;
      mag_error_t err {};
      throw_if_error(mag_ctx_create(&err, &ctx), err);
      g_ctx.store(ctx, std::memory_order_release);
    });
    return g_ctx.load(std::memory_order_acquire);
  }

  std::mutex &get_global_mutex() { return g_mutex; }
  /* The default lives in the C context, not in a binding-local string, so the C and Python APIs cannot
     disagree about where an unqualified tensor lands. It is always stored fully resolved. */
  std::string get_default_device_unlocked() {
    char buf[32];
    mag_device_id_to_str(mag_ctx_default_device(get_ctx()), &buf);
    return std::string {buf};
  }
  std::string get_default_device() {
    std::lock_guard lock {get_global_mutex()};
    return get_default_device_unlocked();
  }

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
    }, "Start recording ops for autodiff.");
    context.def("stop_grad_recorder", []() -> void {
      std::lock_guard lock {get_global_mutex()};
      mag_ctx_grad_recorder_stop(get_ctx());
    }, "Stop recording ops for autodiff.");
    context.def("is_grad_recording", []() -> bool {
      std::lock_guard lock {get_global_mutex()};
      return mag_ctx_grad_recorder_is_running(get_ctx());
    }, "True if gradient recording is active.");
    context.def("manual_seed", [](uint64_t seed) -> void {
      std::lock_guard lock {get_global_mutex()};
      mag_ctx_manual_seed(get_ctx(), seed);
    }, "seed"_a, "Set RNG seed for reproducibility.");
    /* Resolves like set_default_device does, so 'is_device_available(d)' answering true guarantees that
       'set_default_device(d)' with the same string succeeds. */
    context.def("is_device_available", [](const std::string &device) -> bool {
      std::lock_guard lock {get_global_mutex()};
      std::optional<mag_device_id_t> device_id = resolve_device_id_str(std::string {device});
      if (!device_id) return false;
      return mag_ctx_is_device_available(get_ctx(), *device_id);
    });
    context.def("get_default_device", []() -> std::string {
      std::lock_guard lock {get_global_mutex()};
      return get_default_device_unlocked();
    }, "Get the default device string, with the ordinal resolved for backends that have one (e.g., 'cpu', 'cuda:1').");
    context.def("set_default_device", [](const std::string &device) -> void {
      std::lock_guard lock {get_global_mutex()};
      std::optional<mag_device_id_t> device_id = resolve_device_id_str(device);
      if (!device_id) throw std::invalid_argument {"Invalid device id: '" + device + "'"};
      mag_error_t err {};
      throw_if_error(mag_ctx_set_default_device(&err, get_ctx(), *device_id), err);
    }, "device"_a, "Set the default device (e.g., 'cpu', 'cuda', 'cuda:1'). A backend without an ordinal selects its best device.");
    context.def("best_device", [](const std::string &backend) -> std::string {
      std::lock_guard lock {get_global_mutex()};
      std::optional<mag_device_id_t> device_id = parse_device_id_str(backend);
      if (!device_id) throw std::invalid_argument {"Invalid backend: '" + backend + "'"};
      mag_device_id_t best {};
      mag_error_t err {};
      throw_if_error(mag_ctx_best_device(&err, get_ctx(), device_id->type, &best), err);
      char buf[32];
      mag_device_id_to_str(best, &buf);
      return std::string {buf};
    }, "backend"_a, "Fastest device of a backend, ranked by peak compute and memory bandwidth (e.g. best_device('cuda') -> 'cuda:1').");
    context.def("get_default_dtype", []() -> dtype_wrapper {
      std::lock_guard lock {get_global_mutex()};
      return dtype_wrapper{mag_ctx_default_dtype(get_ctx())};
    });
    context.def("set_default_dtype", [](const dtype_wrapper &dtype) -> void {
      std::lock_guard lock {get_global_mutex()};
      if (!mag_ctx_set_default_dtype(get_ctx(), *dtype))
        throw std::logic_error {"Only floating-point types are supported as the default type"};
    });
  }
}
