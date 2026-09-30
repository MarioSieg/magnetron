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

#include "mag_cuda_interp.cuh"

#include <core/mag_interp_plan.h>

#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

namespace mag {
  constexpr unsigned INTERP_BLOCK_SIZE = 256;

  struct interp_dev_axis final {
    int64_t in;
    int64_t out;
    int64_t max_taps;
    const int64_t *ntaps;
    const int64_t *idx;
    const float *w;
  };

  struct interp_dev_plan final {
    int64_t planes;
    interp_dev_axis ax[3];
  };

  template <typename T>
  __global__ static void interp_copy_kernel(int64_t total, interp_dev_plan p, T *__restrict__ br, const T *__restrict__ bx) {
    int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
    int64_t I0 = p.ax[0].in, I1 = p.ax[1].in, I2 = p.ax[2].in;
    int64_t O0 = p.ax[0].out, O1 = p.ax[1].out, O2 = p.ax[2].out;
    for (; idx < total; idx += step) {
      int64_t tmp = idx;
      int64_t o2 = tmp % O2; tmp /= O2;
      int64_t o1 = tmp % O1; tmp /= O1;
      int64_t o0 = tmp % O0; tmp /= O0;
      int64_t plane = tmp;
      int64_t i0 = p.ax[0].idx[o0*p.ax[0].max_taps];
      int64_t i1 = p.ax[1].idx[o1*p.ax[1].max_taps];
      int64_t i2 = p.ax[2].idx[o2*p.ax[2].max_taps];
      br[idx] = bx[((plane*I0 + i0)*I1 + i1)*I2 + i2];
    }
  }

  template <typename T>
  __global__ static void interp_weighted_kernel(int64_t total, interp_dev_plan p, T *__restrict__ br, const T *__restrict__ bx) {
    int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
    int64_t I0 = p.ax[0].in, I1 = p.ax[1].in, I2 = p.ax[2].in;
    int64_t O0 = p.ax[0].out, O1 = p.ax[1].out, O2 = p.ax[2].out;
    for (; idx < total; idx += step) {
      int64_t tmp = idx;
      int64_t o2 = tmp % O2; tmp /= O2;
      int64_t o1 = tmp % O1; tmp /= O1;
      int64_t o0 = tmp % O0; tmp /= O0;
      int64_t plane = tmp;
      int64_t n0 = p.ax[0].ntaps[o0], n1 = p.ax[1].ntaps[o1], n2 = p.ax[2].ntaps[o2];
      const int64_t *i0 = p.ax[0].idx + o0*p.ax[0].max_taps;
      const int64_t *i1 = p.ax[1].idx + o1*p.ax[1].max_taps;
      const int64_t *i2 = p.ax[2].idx + o2*p.ax[2].max_taps;
      const float *w0 = p.ax[0].w + o0*p.ax[0].max_taps;
      const float *w1 = p.ax[1].w + o1*p.ax[1].max_taps;
      const float *w2 = p.ax[2].w + o2*p.ax[2].max_taps;
      float acc = 0.0f;
      for (int64_t k0=0; k0 < n0; ++k0) {
        for (int64_t k1=0; k1 < n1; ++k1) {
          float ww = w0[k0]*w1[k1];
          const T *xrow = bx + ((plane*I0 + i0[k0])*I1 + i1[k1])*I2;
          for (int64_t k2=0; k2 < n2; ++k2) acc += ww*w2[k2]*static_cast<float>(xrow[i2[k2]]);
        }
      }
      br[idx] = static_cast<T>(acc);
    }
  }

  static void *interp_host_alloc(void *ud, size_t nb) {
    auto *blocks = static_cast<std::vector<void *> *>(ud);
    void *p = std::malloc(nb ? nb : 1);
    if (p) blocks->push_back(p);
    return p;
  }

  static int64_t grid_for(int64_t total) {
    int64_t blocks = (total + INTERP_BLOCK_SIZE - 1)/INTERP_BLOCK_SIZE;
    return blocks < 1 ? 1 : (blocks > 65535*16 ? 65535*16 : blocks);
  }

  static mag_status_t interp_upload_plan(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool transposed, interp_dev_plan &dev) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *small = transposed ? r : x;
    const mag_tensor_t *big = transposed ? x : r;
    std::vector<void *> blocks;
    mag_interp_plan_t plan {};
    bool ok = mag_interp_plan_build(&plan, cmd.params, small->meta.coords.shape, big->meta.coords.shape, small->meta.coords.rank, transposed, &interp_host_alloc, &blocks);
    mag_status_t status = MAG_OK;
    if (!ok) {
      status = mag_set_error(err, MAG_ERR_OOM, "cuda: interpolate: failed to build resampling plan.");
    } else {
      size_t bytes = 0;
      for (const auto &ax : plan.axes) {
        bytes += static_cast<size_t>(ax.out)*sizeof(int64_t);
        bytes += static_cast<size_t>(ax.out*ax.max_taps)*sizeof(int64_t);
        bytes += static_cast<size_t>(ax.out*ax.max_taps)*sizeof(float);
        bytes = (bytes + 15) & ~static_cast<size_t>(15);
      }
      auto *dvc = static_cast<physical_device *>(r->meta.device->impl);
      status = dvc->reserve_scratch(err, bytes);
      if (mag_isok(status)) {
        auto *base = static_cast<char *>(dvc->scratch());
        size_t off = 0;
        dev.planes = plan.planes;
        for (int i=0; i < 3; ++i) {
          const auto &ax = plan.axes[i];
          size_t nb_ntaps = static_cast<size_t>(ax.out)*sizeof(int64_t);
          size_t nb_idx = static_cast<size_t>(ax.out*ax.max_taps)*sizeof(int64_t);
          size_t nb_w = static_cast<size_t>(ax.out*ax.max_taps)*sizeof(float);
          dev.ax[i].in = ax.in;
          dev.ax[i].out = ax.out;
          dev.ax[i].max_taps = ax.max_taps;
          dev.ax[i].ntaps = reinterpret_cast<const int64_t *>(base + off);
          cudaMemcpyAsync(base + off, ax.ntaps, nb_ntaps, cudaMemcpyHostToDevice, stream);
          off += nb_ntaps;
          dev.ax[i].idx = reinterpret_cast<const int64_t *>(base + off);
          cudaMemcpyAsync(base + off, ax.idx, nb_idx, cudaMemcpyHostToDevice, stream);
          off += nb_idx;
          dev.ax[i].w = reinterpret_cast<const float *>(base + off);
          cudaMemcpyAsync(base + off, ax.w, nb_w, cudaMemcpyHostToDevice, stream);
          off += nb_w;
          off = (off + 15) & ~static_cast<size_t>(15);
        }
        mag_cu_rt_check(err, cudaGetLastError(), "interpolate plan upload");
      }
    }
    for (void *p : blocks) std::free(p);
    return status;
  }

  template <typename T>
  static mag_status_t launch_interp_copy(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    int64_t total = mag_tensor_numel(r);
    if (!total) return MAG_OK;
    interp_dev_plan dev {};
    mag_status_t status = interp_upload_plan(err, cmd, stream, false, dev);
    if (mag_iserr(status)) return status;
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    interp_copy_kernel<T><<<grid_for(total), INTERP_BLOCK_SIZE, 0, stream>>>(total, dev, br, bx);
    mag_cu_rt_check(err, cudaGetLastError(), "interpolate");
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t launch_interp_weighted(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool transposed) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    int64_t total = mag_tensor_numel(r);
    if (!total) return MAG_OK;
    interp_dev_plan dev {};
    mag_status_t status = interp_upload_plan(err, cmd, stream, transposed, dev);
    if (mag_iserr(status)) return status;
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    interp_weighted_kernel<T><<<grid_for(total), INTERP_BLOCK_SIZE, 0, stream>>>(total, dev, br, bx);
    mag_cu_rt_check(err, cudaGetLastError(), transposed ? "interpolate_back" : "interpolate");
    return MAG_OK;
  }

  static mag_status_t interp_weighted_dispatch(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool transposed) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return launch_interp_weighted<float>(err, cmd, stream, transposed);
      case MAG_DTYPE_FLOAT16: return launch_interp_weighted<half>(err, cmd, stream, transposed);
      case MAG_DTYPE_BFLOAT16: return launch_interp_weighted<__nv_bfloat16>(err, cmd, stream, transposed);
      case MAG_DTYPE_FLOAT8_E4M3FN: return launch_interp_weighted<__nv_fp8_e4m3>(err, cmd, stream, transposed);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: interpolate: unsupported dtype: %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }

  mag_status_t interp_op_interpolate(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    const mag_tensor_t *x = cmd.in[0];
    if (!mag_interp_mode_is_nearest(static_cast<mag_interp_mode_t>(cmd.params->interp.mode)))
      return interp_weighted_dispatch(err, cmd, stream, false);
    switch (mag_tensor_numbytes(x)/mag_tensor_numel(x)) {
      case 1: return launch_interp_copy<uint8_t>(err, cmd, stream);
      case 2: return launch_interp_copy<uint16_t>(err, cmd, stream);
      case 4: return launch_interp_copy<uint32_t>(err, cmd, stream);
      case 8: return launch_interp_copy<uint64_t>(err, cmd, stream);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: interpolate: unsupported dtype: %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }

  mag_status_t interp_op_interpolate_back(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    return interp_weighted_dispatch(err, cmd, stream, true);
  }
}
