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

#include "mag_cuda_fill.cuh"

#include <core/mag_prng_philox4x32.h>
#include <core/mag_u128.h>

namespace mag {
  template <typename T, const bool C>
  __global__ static void fill_kernel(
    int n,
    T *__restrict__ r,
    T v,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    if constexpr (C) {
      if (ti >= n) return;
      r[ti] = v;
    } else {
      int64_t step = blockDim.x*gridDim.x;
      for (; ti < n; ti += step) {
        int ri = mag_coords_iter_to_offset(&rc, ti);
        r[ri] = v;
      }
    }
  }

  template <typename T, const bool C>
  __global__ static void masked_fill_kernel(
    int n,
    T *__restrict__ r,
    const T *__restrict__ x,
    const uint8_t *__restrict__ m,
    T v,
    mag_coords_iter_t rc,
    mag_coords_iter_t xc,
    mag_coords_iter_t mc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    for (; ti < n; ti += step) {
      int ri = C ? ti : mag_coords_iter_to_offset(&rc, ti);
      int xi = C ? ti : mag_coords_iter_broadcast(&rc, &xc, ti);
      int mi = mag_coords_iter_broadcast(&rc, &mc, ti);
      r[ri] = m[mi] ? v : x[xi];
    }
  }

  template <typename T>
  static void launch_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd, cudaStream_t stream, const mag_tensor_t *mask = nullptr) {
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto v = unpack_scalar<T>(cmd.params->fill.value);
    bool cont = mag_tensor_is_contiguous(r);
    int n = mag_tensor_numel(r); // TODO: i64 numel fix
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    if (mask) {
      const mag_tensor_t *xt = cmd.in[0];
      const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(xt));
      const auto *pm = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(mask));
      mag_coords_iter_t mc;
      mag_coords_iter_init(&mc, &mask->meta.coords);
      mag_coords_iter_t xc;
      mag_coords_iter_init(&xc, &xt->meta.coords);
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      if (cont && mag_tensor_is_contiguous(xt)) {
        masked_fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, px, pm, v, rc, xc, mc);
      } else {
        masked_fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, px, pm, v, rc, xc, mc);
      }
    } else {
      if (cont) {
        fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, v, {});
      } else {
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->meta.coords);
        fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, v, rc);
      }
    }
  }

  template <typename T, const bool C, const bool NormDist>
  __global__ static void fill_random_kernel(
    int n,
    T *__restrict__ r,
    T p0,
    T p1,
    uint64_t seed,
    uint64_t subseq,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    int nb = (n+3)>>2;
    if (ti >= nb) return;
    mag_philox4x32_stream_t stream;
    mag_philox4x32_stream_seed(&stream, seed, subseq + static_cast<uint64_t>(ti));
    for (int b=ti; b < nb; b += step) {
      int base = b<<2;
      mag_philox4x32_float32x4_t rr;
      if constexpr (NormDist) rr = mag_philox4x32_next_float32x4_normal(&stream, static_cast<float>(p0), static_cast<float>(p1));
      else rr = mag_philox4x32_next_float32x4_uniform(&stream, static_cast<float>(p0), static_cast<float>(p1));
      int mk = n-base;
      if (mk > 4) mk = 4;
      if constexpr (C) {
        #pragma unroll
        for (int k=0; k < mk; ++k)
          r[base+k] = static_cast<T>(rr.v[k]);
      } else {
        #pragma unroll
        for (int k=0; k < mk; ++k) {
          int ri = mag_coords_iter_to_offset(&rc, base+k);
          r[ri] = static_cast<T>(rr.v[k]);
        }
      }
    }
  }

  template <typename T, typename UT>
  __device__ __forceinline__ T mag_cuda_vrand_uniform_one(
    mag_philox4x32_stream_t *stream,
    T min,
    T max
  ) {
    UT umin = static_cast<UT>(min);
    UT umax = static_cast<UT>(max);
    uint64_t span64 = static_cast<uint64_t>(static_cast<UT>(umax - umin)) + 1ull;
    if (!span64) return static_cast<T>(static_cast<UT>(mag_philox4x32_next_uint64(stream)));
    if constexpr (sizeof(UT) <= 4) {
      uint32_t span = static_cast<uint32_t>(span64);
      uint32_t thresh = static_cast<uint32_t>(0u - span) % span;
      for (;;) {
        uint32_t x = mag_philox4x32_next_uint32(stream);
        uint64_t m = static_cast<uint64_t>(x) * static_cast<uint64_t>(span);
        uint32_t lo = static_cast<uint32_t>(m);
        if (lo < thresh) continue;
        uint32_t hi = static_cast<uint32_t>(m >> 32);
        return static_cast<T>(static_cast<UT>(umin + hi));
      }
    } else {
      uint64_t span = span64;
      uint64_t thresh = (0ull - span) % span;
      for (;;) {
        uint64_t x = mag_philox4x32_next_uint64(stream);
        mag_uint128_t m = mag_uint128_mul128(x, span);
        if (mag_uint128_lo(m) < thresh) continue;
        return static_cast<T>(static_cast<UT>(umin + static_cast<UT>(mag_uint128_hi(m))));
      }
    }
  }

  template <typename T, typename UT, const bool C>
  __global__ static void fill_random_uniform_int_kernel(
    int n,
    T *__restrict__ r,
    T min,
    T max,
    uint64_t seed,
    uint64_t subseq,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    mag_philox4x32_stream_t stream;
    mag_philox4x32_stream_seed(&stream, seed, subseq + static_cast<uint64_t>(ti));
    if constexpr (C) {
      for (; ti < n; ti += step)
        r[ti] = mag_cuda_vrand_uniform_one<T, UT>(&stream, min, max);
    } else {
      for (; ti < n; ti += step) {
        int ri = mag_coords_iter_to_offset(&rc, ti);
        r[ri] = mag_cuda_vrand_uniform_one<T, UT>(&stream, min, max);
      }
    }
  }

  template <typename T, typename UT>
  static void launch_rand_fill_uniform_int_kernel(mag_tensor_t *r, const mag_command_t &cmd, cudaStream_t stream) {
    auto *o = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto min = unpack_scalar<T>(cmd.params->uniform.low);
    auto max = unpack_scalar<T>(cmd.params->uniform.high);
    int n = mag_tensor_numel(r); // TODO: int64 support
    int blocks = (n + FILL_BLOCK_SIZE - 1) / FILL_BLOCK_SIZE;
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    if (mag_tensor_is_contiguous(r)) {
      fill_random_uniform_int_kernel<T, UT, true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, min, max, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      fill_random_uniform_int_kernel<T, UT, false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, min, max, seed, subseq, rc);
    }
  }

  template <typename T, const bool NormDist>
  static void launch_rand_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd, cudaStream_t stream) {
    auto *o = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto p0 = unpack_scalar<T>(cmd.params->uniform.low);
    auto p1 = unpack_scalar<T>(cmd.params->uniform.high);
    int n = mag_tensor_numel(r); // todo: I64 NUEML
    int blocks = (((n+3)>>2)+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    if (mag_tensor_is_contiguous(r)) {
      fill_random_kernel<T, true, NormDist><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, p0, p1, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      fill_random_kernel<T, false, NormDist><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, p0, p1, seed, subseq, rc);
    }
  }

  mag_status_t fill_op_fill(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_fill_kernel<__nv_fp8_e4m3>(r, cmd, stream); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in fill operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t fill_op_masked_fill(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *mask = cmd.in[1];
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd, stream, mask); break;
      case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd, stream, mask); break;
      case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd, stream, mask); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_fill_kernel<__nv_fp8_e4m3>(r, cmd, stream, mask); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd, stream, mask); break;
      case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd, stream, mask); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in masked_fill operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t fill_op_fill_rand_uniform(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, false>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, false>(r, cmd, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, false>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_rand_fill_kernel<__nv_fp8_e4m3, false>(r, cmd, stream); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_rand_fill_uniform_int_kernel<uint8_t, uint8_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT8: launch_rand_fill_uniform_int_kernel<int8_t, uint8_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT16: launch_rand_fill_uniform_int_kernel<uint16_t, uint16_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT16: launch_rand_fill_uniform_int_kernel<int16_t, uint16_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT32: launch_rand_fill_uniform_int_kernel<uint32_t, uint32_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT32: launch_rand_fill_uniform_int_kernel<int32_t, uint32_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT64: launch_rand_fill_uniform_int_kernel<uint64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT64: launch_rand_fill_uniform_int_kernel<int64_t, uint64_t>(r, cmd, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in rand_uniform operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t fill_op_fill_rand_normal(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, true>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, true>(r, cmd, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, true>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_rand_fill_kernel<__nv_fp8_e4m3, true>(r, cmd, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in rand_normal operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <const bool C>
  __global__ static void bernoulli_kernel(
    int n,
    uint8_t *__restrict__ r,
    float p,
    uint64_t seed,
    uint64_t subseq,
    mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    for (; ti < n; ti += step) {
      mag_philox4x32_stream_t stream;
      mag_philox4x32_stream_seed(&stream, seed, subseq + static_cast<uint64_t>(ti));
      float u = mag_philox4x32_next_float32(&stream);
      uint8_t v = !!(u<p);
      int ri = C ? ti : mag_coords_iter_to_offset(&rc, ti);
      r[ri] = v;
    }
  }

  mag_status_t fill_op_rand_bernoulli(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    (void)err;
    mag_tensor_t *r = cmd.out[0];
    mag_assert2(r->meta.dtype == MAG_DTYPE_BOOLEAN);
    auto p = cmd.params->bernoulli.p;
    auto *o = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r));
    int n = mag_tensor_numel(r); // TODO: int64 support
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    if (mag_tensor_is_contiguous(r)) {
      bernoulli_kernel<true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, p, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      bernoulli_kernel<false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, o, p, seed, subseq, rc);
    }
    return MAG_OK;
  }

  template <typename T, const bool C>
  __global__ static void rand_perm_kernel(
    int n,
    T *__restrict__ o,
    uint64_t seed,
    uint64_t subseq,
    mag_coords_iter_t rc
  ) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    mag_philox4x32_stream_t stream;
    mag_philox4x32_stream_seed(&stream, seed, subseq);
    for (int i=0; i < n; ++i) {
      int ri = C ? i : mag_coords_iter_to_offset(&rc, i);
      o[ri] = static_cast<T>(i);
    }
    for (int i=0; i < n-1; ++i) {
      int j = i + static_cast<int>(mag_philox4x32_next_uint64(&stream) % static_cast<uint64_t>(n - i));
      int off_i = C ? i : mag_coords_iter_to_offset(&rc, i);
      int off_j = C ? j : mag_coords_iter_to_offset(&rc, j);
      T tmp = o[off_i];
      o[off_i] = o[off_j];
      o[off_j] = tmp;
    }
  }

  template <typename T>
  static void launch_rand_perm_kernel(mag_tensor_t *r, const mag_command_t &cmd, cudaStream_t stream) {
    int n = mag_tensor_numel(r); // TODO: int64 support
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    auto *po = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    if (mag_tensor_is_contiguous(r)) {
      rand_perm_kernel<T, true><<<1, 1, 0, stream>>>(n, po, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      rand_perm_kernel<T, false><<<1, 1, 0, stream>>>(n, po, seed, subseq, rc);
    }
  }

  mag_status_t fill_op_rand_perm(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_UINT8: launch_rand_perm_kernel<uint8_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT8: launch_rand_perm_kernel<int8_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT16: launch_rand_perm_kernel<uint16_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT16: launch_rand_perm_kernel<int16_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT32: launch_rand_perm_kernel<uint32_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT32: launch_rand_perm_kernel<int32_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT64: launch_rand_perm_kernel<uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT64: launch_rand_perm_kernel<int64_t>(r, cmd, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type for rand_perm: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T, typename PT, typename AT, const bool C>
  __global__ static void arange_kernel(
    int n,
    T *__restrict__ r,
    PT start,
    PT step,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int istep = blockDim.x*gridDim.x;
    for (; ti < n; ti += istep) {
      AT v = static_cast<AT>(start) + static_cast<AT>(ti)*static_cast<AT>(step);
      int ri = C ? ti : mag_coords_iter_to_offset(&rc, ti);
      r[ri] = static_cast<T>(v);
    }
  }

  template <typename T, typename PT, typename AT>
  static void launch_arange(mag_tensor_t *r, const mag_command_t &cmd, cudaStream_t stream) {
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    PT start = unpack_scalar<PT>(cmd.params->arange.start);
    PT step = unpack_scalar<PT>(cmd.params->arange.step);
    int n = mag_tensor_numel(r); // TODO: i64 numel support
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    if (mag_tensor_is_contiguous(r)) {
      arange_kernel<T, PT, AT, true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, start, step, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      arange_kernel<T, PT, AT, false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(n, pr, start, step, rc);
    }
  }

  mag_status_t fill_op_arange(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_arange<float, float, float>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT16: launch_arange<half, float, float>(r, cmd, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_arange<__nv_bfloat16, float, float>(r, cmd, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_arange<__nv_fp8_e4m3, float, float>(r, cmd, stream); break;
      case MAG_DTYPE_UINT8: launch_arange<uint8_t, uint64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT8: launch_arange<int8_t, int64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT16: launch_arange<uint16_t, uint64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT16: launch_arange<int16_t, int64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT32: launch_arange<uint32_t, uint64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT32: launch_arange<int32_t, int64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_UINT64: launch_arange<uint64_t, uint64_t, uint64_t>(r, cmd, stream); break;
      case MAG_DTYPE_INT64: launch_arange<int64_t, int64_t, uint64_t>(r, cmd, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type for arange: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T, const bool C>
  __global__ static void eye_kernel(
    int total,
    int cols,
    T *pr,
    T one,
    T zero,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    for (; ti < total; ti += step) {
      int row = ti / cols;
      int col = ti - row*cols;
      int ri = C ? ti : mag_coords_iter_to_offset(&rc, ti);
      pr[ri] = row == col ? one : zero;
    }
  }

  template <typename T>
  static void launch_eye(mag_tensor_t *r, cudaStream_t stream) {
    mag_assert2(r->meta.coords.rank == 2);
    int numel = mag_tensor_numel(r); // TODO: i64 numel
    int cols = static_cast<int>(r->meta.coords.shape[1]);
    int blocks = (numel + FILL_BLOCK_SIZE - 1)/FILL_BLOCK_SIZE;
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto one = static_cast<T>(1);
    auto zero = static_cast<T>(0);
    if (mag_tensor_is_contiguous(r)) {
      eye_kernel<T, true><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(numel, cols, pr, one, zero, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->meta.coords);
      eye_kernel<T, false><<<blocks, FILL_BLOCK_SIZE, 0, stream>>>(numel, cols, pr, one, zero, {});
    }
  }

  mag_status_t fill_op_eye(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_eye<float>(r, stream); break;
      case MAG_DTYPE_FLOAT16: launch_eye<half>(r, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_eye<__nv_bfloat16>(r, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_eye<__nv_fp8_e4m3>(r, stream); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_eye<uint8_t>(r, stream); break;
      case MAG_DTYPE_INT8: launch_eye<int8_t>(r, stream); break;
      case MAG_DTYPE_UINT16: launch_eye<uint16_t>(r, stream); break;
      case MAG_DTYPE_INT16: launch_eye<int16_t>(r, stream); break;
      case MAG_DTYPE_UINT32: launch_eye<uint32_t>(r, stream); break;
      case MAG_DTYPE_INT32: launch_eye<int32_t>(r, stream); break;
      case MAG_DTYPE_UINT64: launch_eye<uint64_t>(r, stream); break;
      case MAG_DTYPE_INT64: launch_eye<int64_t>(r, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type for eye: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }
}
