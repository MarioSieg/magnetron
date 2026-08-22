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

#include "mag_cuda_transfer.cuh"

namespace mag {
  mag_status_t bidirectional_transfer(
    mag_error_t *err,
    mag_device_t *dvc,
    mag_transfer_dir_t dir,
    mag_tensor_t *src,
    mag_tensor_t *dst
  ) {
    size_t nb = mag_tensor_numbytes(src);
    if (mag_unlikely(nb != mag_tensor_numbytes(dst)))
      return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: source and destination byte sizes differ.");
    if (mag_unlikely(!(mag_tensor_is_contiguous(src) && mag_tensor_is_contiguous(dst))))
      return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: requires contiguous tensors.");
    switch (dir) {
      case MAG_TRANSFER_DIR_H2D: {
        int ordinal = static_cast<int>(dvc->id.device_ordinal);
        const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
        if (mag_status_t stat = phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
        cudaStream_t stream = phys_device.stream();
        if (mag_unlikely(!(src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE)))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: source storage must be host-visible.");
        if (mag_unlikely(dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: destination storage must be device-local.");
        if (mag_unlikely(dst->meta.device != dvc))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: destination device mismatch.");
        mag_cu_rt_check(err, cudaSetDevice(ordinal), "failed to set active device");
        mag_cu_rt_check(err, cudaMemcpyAsync(
          reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst)),
          reinterpret_cast<const void *>(mag_tensor_data_ptr(src)),
          nb,
          cudaMemcpyHostToDevice,
          stream
        ), "failed to enqueue async H2D copy");
        mag_cu_rt_check(err, cudaStreamSynchronize(stream), "failed to synchronize stream after H2D copy");
        return MAG_OK;
      } break;
      case MAG_TRANSFER_DIR_D2H: {
        int ordinal = static_cast<int>(dvc->id.device_ordinal);
        const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
        if (mag_status_t stat = phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
        cudaStream_t stream = phys_device.stream();
        if (mag_unlikely(src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: source storage must be device-local.");
        if (mag_unlikely(src->meta.device != dvc))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: source device mismatch.");
        if (mag_unlikely(!(dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE)))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: destination storage must be host-visible.");
        mag_cu_rt_check(err, cudaSetDevice(ordinal), "failed to set active device");
        mag_cu_rt_check(err, cudaMemcpyAsync(
          reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst)),
          reinterpret_cast<const void *>(mag_tensor_data_ptr(src)),
          nb,
          cudaMemcpyDeviceToHost,
          stream
        ), "failed to enqueue async D2H copy");
        mag_cu_rt_check(err, cudaStreamSynchronize(stream), "failed to synchronize stream after D2H copy");
        return MAG_OK;
      } break;
      case MAG_TRANSFER_DIR_D2D: {
        if (mag_unlikely((src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE) || (dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE)))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: both storages must be device-local.");
        int src_ordinal = static_cast<int>(src->meta.device->id.device_ordinal);
        int dst_ordinal = static_cast<int>(dst->meta.device->id.device_ordinal);
        const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
        if (mag_status_t stat = phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
        cudaStream_t stream = phys_device.stream();
        if (mag_unlikely(dst->storage->device != dvc))
          return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: destination device mismatch.");
        mag_cu_rt_check(err, cudaSetDevice(dst_ordinal), "failed to set active device");
        if (mag_device_t *src_dvc = src->meta.device; src_dvc != dvc) {
          const auto &src_phys_device = *static_cast<const physical_device *>(src_dvc->impl);
          if (mag_status_t stat = src_phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
          mag_cu_rt_check(err, cudaSetDevice(src_ordinal), "failed to set active device");
          cudaEvent_t src_done = src_phys_device.event();
          mag_cu_rt_check(err, cudaEventRecord(src_done, src_phys_device.stream()), "failed to record event on source device");
          mag_cu_rt_check(err, cudaSetDevice(dst_ordinal), "failed to set active device");
          mag_cu_rt_check(err, cudaStreamWaitEvent(phys_device.stream(), src_done, 0), "failed to wait for event on destination device");
        }
        mag_cu_rt_check(err, cudaMemcpyPeerAsync(
          reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst)),
          dst_ordinal,
          reinterpret_cast<const void *>(mag_tensor_data_ptr(src)),
          src_ordinal,
          nb,
          stream
          ), "failed to enqueue async D2D copy"
        );
        mag_cu_rt_check(err, cudaStreamSynchronize(stream), "failed to synchronize stream after D2D copy");
        return MAG_OK;
      } break;
    }
    return mag_set_error(err, MAG_ERR_PARAM, "cuda transfer: invalid transfer direction.");
  }
}