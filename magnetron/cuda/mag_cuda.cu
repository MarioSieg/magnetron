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

#include "mag_cuda.cuh"

#include <algorithm>

#include "cpu/mag_cpu.h"

#include <core/mag_alloc.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace mag {
  class physical_device;

  class cuda_backend final : public mag_backend_t {
  public:
    explicit cuda_backend(mag_context_t *ctx) : mag_backend_t{} {
      this->ctx = ctx;
      this->impl = this;
      this->init = +[](mag_error_t *err, mag_backend_t *bck, mag_context_t *ctx) -> mag_status_t {
        if (!static_cast<cuda_backend *>(bck->impl)->initialize(ctx)) {
          return mag_set_error(err, MAG_ERR_OOM, "Failed to initialize CUDA backend.");
        }
        return MAG_OK;
      };
      this->shutdown = +[](mag_error_t *err, mag_backend_t *bck) -> mag_status_t {
        if (!static_cast<cuda_backend *>(bck->impl)->destroy()) {
          return mag_set_error(err, MAG_ERR_OOM, "Failed to deinitialize CUDA backend.");
        }
        return MAG_OK;
      };
      backend_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_CUDA_BACKEND_VERSION; };
      runtime_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_VERSION; };
      id = +[](mag_backend_t *bck) noexcept -> const char* { return "cuda"; };
      num_devices = +[](mag_backend_t *bck) noexcept -> uint32_t { return static_cast<cuda_backend *>(bck->impl)->device_count(); };
      best_device_id = +[](mag_backend_t *bck) noexcept -> uint32_t { return 0; };
      get_device = +[](mag_backend_t *bck, uint32_t idx) -> mag_device_t* {
        auto &dvc = static_cast<cuda_backend *>(bck->impl)->devices();
        if (idx >= dvc.size()) {
          mag_log_error("Invalid device index %u (max %zu)", idx, dvc.size()-1);
          return nullptr;
        }
        return &*dvc[idx];
      };
    }

    [[nodiscard]] uint32_t device_count() const noexcept { return static_cast<uint32_t>(m_devices.size()); }
    [[nodiscard]] uint32_t active_device_idx() const noexcept { return m_active_device_idx; }
    [[nodiscard]] uint32_t best_device_idx() const noexcept { return m_best_device_idx; }
    [[nodiscard]] const physical_device &active_device() const noexcept { return *m_devices.at(m_active_device_idx); }
    [[nodiscard]] const physical_device &best_device() const noexcept { return *m_devices.at(m_best_device_idx); }
    [[nodiscard]] const std::vector<std::shared_ptr<physical_device>> &devices() const noexcept { return m_devices; }

  private:
    [[nodiscard]] bool initialize(mag_context_t *ctx) {
      if (cuInit(0) != CUDA_SUCCESS) {
        mag_log_error("Failed to initialize CUDA driver API.");
        return false;
      }
      int num_devices = 0;
      if (cudaGetDeviceCount(&num_devices) != cudaSuccess || num_devices <= 0) { // No GPUs found, backend cannot be used
        mag_log_error("No CUDA-capable devices found.");
        return false;
      }
      m_devices.reserve(num_devices);
      for (int device_ordinal=0; device_ordinal < num_devices; ++device_ordinal) {
        try {
          mag_error_t err {};
          std::shared_ptr<physical_device> device = nullptr;
          mag_status_t stat = physical_device::create(&err, device, ctx, device_ordinal);
          if (mag_iserr(stat) || !device) {
            mag_log_error("Failed to initialize CUDA device %d: %s", device_ordinal, err.message); // TODO: propagate error message bet
            continue;
          }
          m_devices.emplace_back(std::move(device));
        } catch (const std::exception &e) {
          mag_log_error("Failed to initialize CUDA device %d: %s", device_ordinal, e.what());
        } catch (...) {
          mag_log_error("Unknown error while initializing CUDA device %d", device_ordinal);
        }
      }
      bool alloc_async = !m_devices.empty() && std::all_of(m_devices.begin(), m_devices.end(), [](const auto &dvc) noexcept -> bool {
        return dvc->features() & device_features::mem_pool;
      });
      global_async_alloc.store(alloc_async, std::memory_order_relaxed);
      return true;
    }

    [[nodiscard]] bool destroy() {
      global_async_alloc.store(false, std::memory_order_relaxed);
      m_devices.clear();
      return true;
    }

    uint32_t m_active_device_idx = 0;
    uint32_t m_best_device_idx = 0;
    std::vector<std::shared_ptr<physical_device>> m_devices = {};
  };
}

uint32_t MAG_BACKEND_SYM_ABI_COOKIE(){
  return mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
}

mag_status_t MAG_BACKEND_SYM_INIT(mag_error_t *err, mag_backend_t **out, mag_context_t *ctx)
try {
  *out = new mag::cuda_backend {ctx};
  return MAG_OK;
} catch (const std::exception &e) {
  *out = nullptr;
  return mag_set_error(err, MAG_ERR_BACKEND, "cuda: C++ exception during backend initialization: %s", e.what());
} catch (...) {
  *out = nullptr;
  return mag_set_error(err, MAG_ERR_BACKEND, "cuda: C++ exception during backend initialization.");
}

mag_status_t MAG_BACKEND_SYM_SHUTDOWN(mag_error_t *err, mag_backend_t *backend)
try {
  delete static_cast<mag::cuda_backend *>(backend);
  return MAG_OK;
} catch (const std::exception &e) {
  return mag_set_error(err, MAG_ERR_BACKEND, "cuda: C++ exception during backend shutdown: %s", e.what());
} catch (...) {
  return mag_set_error(err, MAG_ERR_BACKEND, "cuda: C++ exception during backend shutdown.");
}
