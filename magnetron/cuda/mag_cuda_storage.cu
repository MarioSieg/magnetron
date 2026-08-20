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

#include "mag_cuda_storage.cuh"

namespace mag {
  mag_status_t alloc_storage_buffer(
    mag_error_t *err,
    mag_device_t *dvc,
    mag_storage_buffer_t **out,
    size_t size
  ) {
    mag_context_t *ctx = dvc->ctx;
    uintptr_t base;
    int ordinal = static_cast<int>(dvc->id.device_ordinal);
    const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
    if (mag_status_t stat = phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
    mag_cu_rt_check(err, cudaSetDevice(ordinal), "failed to set active device");
    if (cudaError_t res = stream_alloc(reinterpret_cast<void **>(&base), size, phys_device.stream()); mag_unlikely(res != cudaSuccess)) {
      double amount = 0.0;
      const char *unit = "";
      mag_humanize_memory_size(size, &amount, &unit);
      return mag_set_error(err, MAG_ERR_OOM, "cuda: failed to allocate %.03f %s of device memory: %s.", amount, unit, cudaGetErrorString(res));
    }
    *out = static_cast<mag_storage_buffer_t *>(mag_slab_alloc(&ctx->storage_slab));
    new (*out) mag_storage_buffer_t {
      .__rcb = {},
      .ctx = ctx,
      .device = dvc,
      .flags = MAG_STORAGE_FLAG_ACCESS_W,
      .alignment = 256, // cudaMalloc guarantees this
      .base = base,
      .size = size,
      .aux = {},
    };
    mag_rc_init_object(*out, +[](void *self) -> mag_status_t {
      auto *buffer = static_cast<mag_storage_buffer_t *>(self);
      mag_context_t *ctx = buffer->ctx;
      mag_device_t *dvc = buffer->device;
      auto *base = reinterpret_cast<void *>(buffer->base);
      mag_assert(ctx->telemetry.num_alive_storages > 0, "cuda: double free detected on CUDA storage buffer.");
      --ctx->telemetry.num_alive_storages;
      mag_slab_free(&ctx->storage_slab, buffer);
      if (mag_unlikely(cudaSetDevice(static_cast<int>(dvc->id.device_ordinal)) != cudaSuccess))
        return MAG_ERR_FREE;
      const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
      return stream_free(base, phys_device.stream()) == cudaSuccess ? MAG_OK : MAG_ERR_FREE;
    });
    ++ctx->telemetry.num_alive_storages;
    return MAG_OK;
  }
}
