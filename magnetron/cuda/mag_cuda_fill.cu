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

#include <type_traits>

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
    const uint8_t *__restrict__ m,
    T v,
    [[maybe_unused]] mag_coords_iter_t rc,
    mag_coords_iter_t mc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    if constexpr (C) {
      for (; ti < n; ti += step) {
        int mi = mag_coords_iter_broadcast(&rc, &mc, ti);
        if (m[mi]) r[ti] = v;
      }
    } else {
      for (; ti < n; ti += step) {
        int ri = mag_coords_iter_to_offset(&rc, ti);
        int mi = mag_coords_iter_broadcast(&rc, &mc, ti);
        if (m[mi]) r[ri] = v;
      }
    }
  }

  template <typename T>
  static void launch_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd, const mag_tensor_t *mask = nullptr) {
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto v = unpack_param<T>(cmd.attrs, 0);
    bool cont = mag_tensor_is_contiguous(r);
    int n = numel_i32(r);
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    if (mask) {
      const auto *pm = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(mask));
      mag_coords_iter_t mc;
      mag_coords_iter_init(&mc, &mask->coords);
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->coords);
      if (cont) masked_fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, pm, v, rc, mc);
      else masked_fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, pm, v, rc, mc);
    } else {
      if (cont) {
        fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, v, {});
      } else {
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, v, rc);
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

  template <typename T, const bool NormDist>
  static void launch_rand_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd) {
    auto *o = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto p0 = unpack_param<T>(cmd.attrs, 0);
    auto p1 = unpack_param<T>(cmd.attrs, 1);
    int n = numel_i32(r);
    int blocks = (((n+3)>>2)+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    if (mag_tensor_is_contiguous(r)) {
      fill_random_kernel<T, true, NormDist><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->coords);
      fill_random_kernel<T, false, NormDist><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, rc);
    }
  }

  void fill_op_fill(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd); break;
      case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_fill_kernel<__nv_fp8_e4m3>(r, cmd); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd); break;
      case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd); break;
      case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd); break;
      case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd); break;
      case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd); break;
      case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd); break;
      case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd); break;
      case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd); break;
      default: mag_assert(false, "Unsupported data type in binary operation");
    }
  }

  void fill_op_masked_fill(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    auto *mask = static_cast<mag_tensor_t *>(mag_op_attr_unwrap_ptr(cmd.attrs[0])); // TODO: pass in cmd in why the fuck are these here
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd, mask); break;
      case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd, mask); break;
      case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd, mask); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_fill_kernel<__nv_fp8_e4m3>(r, cmd, mask); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd, mask); break;
      case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd, mask); break;
      case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd, mask); break;
      case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd, mask); break;
      case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd, mask); break;
      case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd, mask); break;
      case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd, mask); break;
      case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd, mask); break;
      default: mag_assert(false, "Unsupported data type in binary operation");
    }
  }

  void fill_op_fill_rand_uniform(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, false>(r, cmd); break;
      case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, false>(r, cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, false>(r, cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_rand_fill_kernel<__nv_fp8_e4m3, false>(r, cmd); break;
      default: mag_assert(false, "Unsupported data type in binary operation");
    }
  }

  void fill_op_fill_rand_normal(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, true>(r, cmd); break;
      case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, true>(r, cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, true>(r, cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_rand_fill_kernel<__nv_fp8_e4m3, true>(r, cmd); break;
      default: mag_assert(false, "Unsupported data type in binary operation");
    }
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

  void fill_op_rand_bernoulli(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    mag_assert2(r->dtype == MAG_DTYPE_BOOLEAN);
    auto p = static_cast<float>(mag_op_attr_unwrap_float64(cmd.attrs[0]));
    auto *o = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r));
    int n = numel_i32(r);
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    if (mag_tensor_is_contiguous(r)) {
      bernoulli_kernel<true><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->coords);
      bernoulli_kernel<false><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p, seed, subseq, rc);
    }
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
  static void launch_rand_perm_kernel(mag_tensor_t *r, const mag_command_t &cmd) {
    int n = numel_i32(r);
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    auto *po = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    if (mag_tensor_is_contiguous(r)) {
      rand_perm_kernel<T, true><<<1, 1>>>(n, po, seed, subseq, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->coords);
      rand_perm_kernel<T, false><<<1, 1>>>(n, po, seed, subseq, rc);
    }
  }

  void fill_op_rand_perm(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->dtype) {
      case MAG_DTYPE_UINT8: launch_rand_perm_kernel<uint8_t>(r, cmd); break;
      case MAG_DTYPE_INT8: launch_rand_perm_kernel<int8_t>(r, cmd); break;
      case MAG_DTYPE_UINT16: launch_rand_perm_kernel<uint16_t>(r, cmd); break;
      case MAG_DTYPE_INT16: launch_rand_perm_kernel<int16_t>(r, cmd); break;
      case MAG_DTYPE_UINT32: launch_rand_perm_kernel<uint32_t>(r, cmd); break;
      case MAG_DTYPE_INT32: launch_rand_perm_kernel<int32_t>(r, cmd); break;
      case MAG_DTYPE_UINT64: launch_rand_perm_kernel<uint64_t>(r, cmd); break;
      case MAG_DTYPE_INT64: launch_rand_perm_kernel<int64_t>(r, cmd); break;
      default: mag_assert(false, "Unsupported dtype for rand_perm");
    }
  }

  template <typename T, const bool C>
  __global__ static void arange_kernel(
    int n,
    T *__restrict__ r,
    float start,
    float step,
    [[maybe_unused]] mag_coords_iter_t rc
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int istep = blockDim.x*gridDim.x;
    for (; ti < n; ti += istep) {
      auto v = start + static_cast<float>(ti)*step;
      int ri = C ? ti : mag_coords_iter_to_offset(&rc, ti);
      r[ri] = static_cast<T>(v);
    }
  }

  template <typename T>
  static void launch_arange(mag_tensor_t *r, const mag_command_t &cmd) {
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    auto start = static_cast<float>(mag_op_attr_unwrap_float64(cmd.attrs[0]));
    auto step = static_cast<float>(mag_op_attr_unwrap_float64(cmd.attrs[1]));
    int n = numel_i32(r);
    int blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
    if (mag_tensor_is_contiguous(r)) {
      arange_kernel<T, true><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, start, step, {});
    } else {
      mag_coords_iter_t rc;
      mag_coords_iter_init(&rc, &r->coords);
      arange_kernel<T, false><<<blocks, FILL_BLOCK_SIZE>>>(n, pr, start, step, rc);
    }
  }

  void fill_op_arange(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_arange<float>(r, cmd); break;
      case MAG_DTYPE_FLOAT16: launch_arange<half>(r, cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_arange<__nv_bfloat16>(r, cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_arange<__nv_fp8_e4m3>(r, cmd); break;
      case MAG_DTYPE_UINT8: launch_arange<uint8_t>(r, cmd); break;
      case MAG_DTYPE_INT8: launch_arange<int8_t>(r, cmd); break;
      case MAG_DTYPE_UINT16: launch_arange<uint16_t>(r, cmd); break;
      case MAG_DTYPE_INT16: launch_arange<int16_t>(r, cmd); break;
      case MAG_DTYPE_UINT32: launch_arange<uint32_t>(r, cmd); break;
      case MAG_DTYPE_INT32: launch_arange<int32_t>(r, cmd); break;
      case MAG_DTYPE_UINT64: launch_arange<uint64_t>(r, cmd); break;
      case MAG_DTYPE_INT64: launch_arange<int64_t>(r, cmd); break;
      default: mag_assert(false, "Unsupported dtype for arange");
    }
  }
}
