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

#include "mag_cuda_misc.cuh"

#include "mag_cuda_unary.cuh"

#include <core/mag_prng_philox4x32.h>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <type_traits>

namespace mag {
  constexpr int MISC_BLOCK_SIZE = 256;

  static void cuda_check(cudaError_t e, const char *what) {
    if (mag_unlikely(e != cudaSuccess))
      mag_panic("%s: %s", what, cudaGetErrorString(e));
  }

  template <const bool SameLayout>
  __global__ static void one_hot_kernel(
    int total,
    int nc,
    int64_t *__restrict__ pr,
    const int64_t *__restrict__ pi,
    mag_coords_iter_t it
  ) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    for (; i < total; i += step) {
      int pir = SameLayout ? i : mag_coords_iter_to_offset(&it, i);
      int cls = pi[pir];
      if (static_cast<unsigned>(cls) < static_cast<unsigned>(nc)) {
        int off = i*nc + cls;
        pr[off] = 1;
      }
    }
  }

  mag_status_t misc_op_one_hot(mag_error_t *err, const mag_command_t &cmd) {
    (void)err;
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *idx = cmd.in[0];
    mag_assert2(r->meta.dtype == MAG_DTYPE_INT64 && idx->meta.dtype == MAG_DTYPE_INT64);
    int nc = cmd.params->one_hot.num_classes;
    int n = numel_i32(idx);
    auto *pr = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
    const auto *pidx = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(idx));
    int blocks = (n+MISC_BLOCK_SIZE-1)/MISC_BLOCK_SIZE;
    if (std::array<const mag_tensor_t *, 2> tensors{r, idx}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      one_hot_kernel<true><<<blocks, MISC_BLOCK_SIZE>>>(n, nc, pr, pidx, {});
    } else {
      mag_coords_iter_t it;
      mag_coords_iter_init(&it, &idx->meta.coords);
      one_hot_kernel<false><<<blocks, MISC_BLOCK_SIZE>>>(n, nc, pr, pidx, it);
    }
    return MAG_OK;
  }

  template <typename T, const bool upper>
  __global__ static void tri_mask_kernel(
    int64_t total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    mag_coords_iter_t cr,
    mag_coords_iter_t cx,
    int64_t diag
  ) {
    T z{};
    if constexpr (std::is_same_v<T, float>) z = 0.f;
    else if constexpr (std::is_same_v<T, half>) z = __float2half(0.f);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) z = __float2bfloat16(0.f);
    else z = T{};
    int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    int64_t cols = cr.shape[cr.rank-1];
    int64_t rows = cr.shape[cr.rank-2];
    int64_t mat = rows*cols;
    for (; ti < total; ti += step) {
      int64_t inner = ti % mat;
      int64_t row = inner / cols;
      int64_t col = inner - row*cols;
      int ri, xi;
      mag_coords_iter_offset2(&cr, &cx, ti, &ri, &xi);
      bool keep = upper ? (col - row) >= diag : (col - row) <= diag;
      br[ri] = keep ? bx[xi] : z;
    }
  }

  template <typename T, const bool upper>
  static void launch_tri_mask(mag_tensor_t *r, const mag_tensor_t *x, int64_t diag) {
    int64_t n = mag_tensor_numel(r);
    int64_t blocks = (n + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->meta.coords);
    mag_coords_iter_init(&cx, &x->meta.coords);
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    tri_mask_kernel<T, upper><<<blocks, MISC_BLOCK_SIZE>>>(n, br, bx, cr, cx, diag);
  }

  mag_status_t misc_op_tril(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    int64_t diag = cmd.params->trilu.diag;
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_tri_mask<float, false>(r, x, diag); break;
      case MAG_DTYPE_FLOAT16: launch_tri_mask<half, false>(r, x, diag); break;
      case MAG_DTYPE_BFLOAT16: launch_tri_mask<__nv_bfloat16, false>(r, x, diag); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_tri_mask<__nv_fp8_e4m3, false>(r, x, diag); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_tri_mask<uint8_t, false>(r, x, diag); break;
      case MAG_DTYPE_INT8: launch_tri_mask<int8_t, false>(r, x, diag); break;
      case MAG_DTYPE_UINT16: launch_tri_mask<uint16_t, false>(r, x, diag); break;
      case MAG_DTYPE_INT16: launch_tri_mask<int16_t, false>(r, x, diag); break;
      case MAG_DTYPE_UINT32: launch_tri_mask<uint32_t, false>(r, x, diag); break;
      case MAG_DTYPE_INT32: launch_tri_mask<int32_t, false>(r, x, diag); break;
      case MAG_DTYPE_UINT64: launch_tri_mask<uint64_t, false>(r, x, diag); break;
      case MAG_DTYPE_INT64: launch_tri_mask<int64_t, false>(r, x, diag); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: tril: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t misc_op_triu(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    int64_t diag = cmd.params->trilu.diag;
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_tri_mask<float, true>(r, x, diag); break;
      case MAG_DTYPE_FLOAT16: launch_tri_mask<half, true>(r, x, diag); break;
      case MAG_DTYPE_BFLOAT16: launch_tri_mask<__nv_bfloat16, true>(r, x, diag); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_tri_mask<__nv_fp8_e4m3, true>(r, x, diag); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_tri_mask<uint8_t, true>(r, x, diag); break;
      case MAG_DTYPE_INT8: launch_tri_mask<int8_t, true>(r, x, diag); break;
      case MAG_DTYPE_UINT16: launch_tri_mask<uint16_t, true>(r, x, diag); break;
      case MAG_DTYPE_INT16: launch_tri_mask<int16_t, true>(r, x, diag); break;
      case MAG_DTYPE_UINT32: launch_tri_mask<uint32_t, true>(r, x, diag); break;
      case MAG_DTYPE_INT32: launch_tri_mask<int32_t, true>(r, x, diag); break;
      case MAG_DTYPE_UINT64: launch_tri_mask<uint64_t, true>(r, x, diag); break;
      case MAG_DTYPE_INT64: launch_tri_mask<int64_t, true>(r, x, diag); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: triu: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T, const bool C>
  __global__ static void where_kernel(
    int n,
    T *__restrict__ br,
    const uint8_t *__restrict__ bc,
    const T *__restrict__ bx,
    const T *__restrict__ by,
    [[maybe_unused]] mag_coords_iter_t cr,
    [[maybe_unused]] mag_coords_iter_t cc,
    [[maybe_unused]] mag_coords_iter_t cx,
    [[maybe_unused]] mag_coords_iter_t cy
  ) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    if constexpr (C) {
      for (; i < n; i += step) {
        br[i] = bc[i] ? bx[i] : by[i];
      }
    } else {
      for (; i < n; i += step) {
        int ri, ci, xi, yi;
        mag_coords_iter_offset4(&cr, &cc, &cx, &cy, i, &ri, &ci, &xi, &yi);
        br[ri] = bc[ci] ? bx[xi] : by[yi];
      }
    }
  }

  template <typename T>
  static void launch_where(mag_tensor_t *r, const mag_tensor_t *cond, const mag_tensor_t *x, const mag_tensor_t *y) {
    int n = numel_i32(r);
    int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bc = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(cond));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
    if (std::array<const mag_tensor_t *, 4> tensors{r, cond, x, y}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      where_kernel<T, true><<<blocks, UNARY_BLOCK_SIZE>>>(n, br, bc, bx, by, {}, {}, {}, {});
    } else {
      mag_coords_iter_t cr, cc, cx, cy;
      mag_coords_iter_init(&cr, &r->meta.coords);
      mag_coords_iter_init(&cc, &cond->meta.coords);
      mag_coords_iter_init(&cx, &x->meta.coords);
      mag_coords_iter_init(&cy, &y->meta.coords);
      where_kernel<T, false><<<blocks, UNARY_BLOCK_SIZE>>>(n, br, bc, bx, by, cr, cc, cx, cy);
    }
  }

  mag_status_t misc_op_where(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *cond = cmd.in[0];
    const mag_tensor_t *x = cmd.in[1];
    const mag_tensor_t *y = cmd.in[2];
    mag_assert2(cond->meta.dtype == MAG_DTYPE_BOOLEAN);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_where<float>(r, cond, x, y); break;
      case MAG_DTYPE_FLOAT16: launch_where<half>(r, cond, x, y); break;
      case MAG_DTYPE_BFLOAT16: launch_where<__nv_bfloat16>(r, cond, x, y); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_where<__nv_fp8_e4m3>(r, cond, x, y); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_where<uint8_t>(r, cond, x, y); break;
      case MAG_DTYPE_INT8: launch_where<int8_t>(r, cond, x, y); break;
      case MAG_DTYPE_UINT16: launch_where<uint16_t>(r, cond, x, y); break;
      case MAG_DTYPE_INT16: launch_where<int16_t>(r, cond, x, y); break;
      case MAG_DTYPE_UINT32: launch_where<uint32_t>(r, cond, x, y); break;
      case MAG_DTYPE_INT32: launch_where<int32_t>(r, cond, x, y); break;
      case MAG_DTYPE_UINT64: launch_where<uint64_t>(r, cond, x, y); break;
      case MAG_DTYPE_INT64: launch_where<int64_t>(r, cond, x, y); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: where: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T>
  __global__ static void repeat_back_kernel(
    int rn,
    int xn,
    T *__restrict__ br,
    const T *__restrict__ bx,
    mag_coords_iter_t cr,
    mag_coords_iter_t cx
  ) {
    for (int64_t i=0; i < rn; ++i) {
      int ri = mag_coords_iter_to_offset(&cr, i);
      br[ri] = static_cast<T>(0.f);
    }
    for (int i=0; i < xn; ++i) {
      int xi = mag_coords_iter_to_offset(&cx, i);
      int ri = mag_coords_iter_repeat(&cr, &cx, i);
      br[ri] = static_cast<T>(static_cast<float>(br[ri]) + static_cast<float>(bx[xi]));
    }
  }

  mag_status_t misc_op_repeat_back(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->meta.coords);
    mag_coords_iter_init(&cx, &x->meta.coords);
    int rn = numel_i32(r);
    int xn = numel_i32(x);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: repeat_back_kernel<float><<<1, 1>>>(
        rn, xn,
        reinterpret_cast<float *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const float *>(mag_tensor_data_ptr(x)),
        cr, cx
      ); return MAG_OK;
      case MAG_DTYPE_FLOAT16: repeat_back_kernel<half><<<1, 1>>>(
        rn, xn,
        reinterpret_cast<half *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const half *>(mag_tensor_data_ptr(x)),
        cr, cx
      ); return MAG_OK;
      case MAG_DTYPE_BFLOAT16: repeat_back_kernel<__nv_bfloat16><<<1, 1>>>(
        rn, xn,
        reinterpret_cast<__nv_bfloat16 *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(x)),
        cr, cx
      ); return MAG_OK;
        case MAG_DTYPE_FLOAT8_E4M3FN: repeat_back_kernel<__nv_fp8_e4m3><<<1, 1>>>(
        rn, xn,
        reinterpret_cast<__nv_fp8_e4m3 *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const __nv_fp8_e4m3 *>(mag_tensor_data_ptr(x)),
        cr, cx
      ); return MAG_OK;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: repeat_back: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
  }

  template <typename T>
  __global__ static void gather_kernel_3d(
    T *__restrict__ br,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    int64_t inner,
    int64_t out_ax,
    int64_t ax
  ) {
    int64_t k = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t j = static_cast<int64_t>(blockIdx.y);
    int64_t o = static_cast<int64_t>(blockIdx.z);
    if (k >= inner) return;
    int64_t flat = (o*out_ax + j)*inner + k;
    int64_t g = __ldg(bi + flat);
    if (g < 0) g += ax;
    br[flat] = bx[(o*ax + g)*inner + k];
  }

  template <typename T>
  __global__ static void gather_kernel_flat(
    int64_t on,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    int64_t inner,
    int64_t out_ax,
    int64_t ax
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    for (; flat < on; flat += step) {
      int64_t g = __ldg(bi + flat);
      if (g < 0) g += ax;
      int64_t k = flat % inner;
      int64_t t = flat / inner;
      int64_t o = t / out_ax;
      br[flat] = bx[(o*ax + g)*inner + k];
    }
  }

  template <typename T>
  __global__ static void gather_kernel_strided(
    int64_t on,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    mag_tensor_t src,
    mag_tensor_t index,
    mag_tensor_t out,
    int64_t axis_in,
    int64_t ax
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    for (; flat < on; flat += step) {
      int64_t oc[MAG_MAX_DIMS];
      int64_t tmp = flat;
      for (int64_t d = out.meta.coords.rank-1; d >= 0; --d) {
        oc[d] = tmp % out.meta.coords.shape[d];
        tmp /= out.meta.coords.shape[d];
      }
      int64_t index_offset = 0;
      for (int64_t d = 0; d < index.meta.coords.rank; ++d) index_offset += oc[d]*index.meta.coords.strides[d];
      int64_t g = __ldg(bi + index_offset);
      if (g < 0) g += ax;
      int64_t src_off = 0, dst_off = 0;
      for (int64_t d = 0; d < src.meta.coords.rank; ++d) src_off += (d == axis_in ? g : oc[d])*src.meta.coords.strides[d];
      for (int64_t d = 0; d < out.meta.coords.rank; ++d) dst_off += oc[d]*out.meta.coords.strides[d];
      br[dst_off] = bx[src_off];
    }
  }

  template <typename T>
  static void launch_gather(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *src = cmd.in[0];
    const mag_tensor_t *index = cmd.in[1];
    int64_t axis = cmd.params->gather.dim;
    if (axis < 0) axis += src->meta.coords.rank;
    mag_assert2(axis >= 0 && axis < src->meta.coords.rank);
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(src));
    const auto *bi = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index));
    int64_t ax = src->meta.coords.shape[axis];
    if (mag_tensor_is_contiguous(src) && mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(index)) {
      int64_t inner = 1;
      for (int64_t d = axis+1; d < src->meta.coords.rank; ++d) inner *= src->meta.coords.shape[d];
      int64_t out_ax = r->meta.coords.shape[axis];
      int64_t outer = r->meta.numel / (out_ax*inner);
      constexpr int64_t MAX_DIM = 65535;
      if (outer <= MAX_DIM && out_ax <= MAX_DIM) {
        dim3 block(MISC_BLOCK_SIZE, 1, 1);
        dim3 grid(
          static_cast<unsigned>((inner + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE),
          static_cast<unsigned>(out_ax),
          static_cast<unsigned>(outer)
        );
        gather_kernel_3d<T><<<grid, block>>>(br, bx, bi, inner, out_ax, ax);
      } else {
        int64_t on = r->meta.numel;
        int64_t blocks = (on + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        gather_kernel_flat<T><<<blocks, MISC_BLOCK_SIZE>>>(on, br, bx, bi, inner, out_ax, ax);
      }
    } else {
      int64_t on = r->meta.numel;
      int64_t blocks = (on + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
      gather_kernel_strided<T><<<blocks, MISC_BLOCK_SIZE>>>(on, br, bx, bi, *src, *index, *r, axis, ax);
    }
  }

  mag_status_t misc_op_gather(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_gather<float>(cmd); break;
      case MAG_DTYPE_FLOAT16: launch_gather<half>(cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_gather<__nv_bfloat16>(cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_gather<__nv_fp8_e4m3>(cmd); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_gather<uint8_t>(cmd); break;
      case MAG_DTYPE_INT8: launch_gather<int8_t>(cmd); break;
      case MAG_DTYPE_UINT16: launch_gather<uint16_t>(cmd); break;
      case MAG_DTYPE_INT16: launch_gather<int16_t>(cmd); break;
      case MAG_DTYPE_UINT32: launch_gather<uint32_t>(cmd); break;
      case MAG_DTYPE_INT32: launch_gather<int32_t>(cmd); break;
      case MAG_DTYPE_UINT64: launch_gather<uint64_t>(cmd); break;
      case MAG_DTYPE_INT64: launch_gather<int64_t>(cmd); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: gather: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T>
  __global__ static void embedding_kernel_2d(
    T *__restrict__ br,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    int64_t row_size,
    int64_t vocab_size
  ) {
    int64_t col = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t row = static_cast<int64_t>(blockIdx.y);
    if (col >= row_size) return;
    int64_t g = __ldg(bi + row);
    if (g < 0) g += vocab_size;
    br[row*row_size + col] = bx[g*row_size + col];
  }

  template <typename T>
  __global__ static void embedding_kernel_flat(
    int64_t on,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    int64_t row_size,
    int64_t vocab_size
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    for (; flat < on; flat += step) {
      int64_t row = flat / row_size;
      int64_t col = flat % row_size;
      int64_t g = __ldg(bi + row);
      if (g < 0) g += vocab_size;
      br[flat] = bx[g*row_size + col];
    }
  }

  template <typename T>
  static void launch_embedding(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *weight = cmd.in[0];
    const mag_tensor_t *indices = cmd.in[1];
    int64_t vocab_size = weight->meta.coords.shape[0];
    int64_t row_size = weight->meta.numel / vocab_size;
    int64_t n_indices = indices->meta.numel;
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(weight));
    const auto *bi = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(indices));
    constexpr int64_t MAX_DIM = 65535;
    if (n_indices <= MAX_DIM) {
      dim3 block(MISC_BLOCK_SIZE, 1, 1);
      dim3 grid(
        static_cast<unsigned>((row_size + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE),
        static_cast<unsigned>(n_indices)
      );
      embedding_kernel_2d<T><<<grid, block>>>(br, bx, bi, row_size, vocab_size);
    } else {
      int64_t on = n_indices*row_size;
      int64_t blocks = (on + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
      embedding_kernel_flat<T><<<blocks, MISC_BLOCK_SIZE>>>(on, br, bx, bi, row_size, vocab_size);
    }
  }

  mag_status_t misc_op_embedding(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32:       launch_embedding<float>(cmd); break;
      case MAG_DTYPE_FLOAT16:       launch_embedding<half>(cmd); break;
      case MAG_DTYPE_BFLOAT16:      launch_embedding<__nv_bfloat16>(cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_embedding<__nv_fp8_e4m3>(cmd); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8:         launch_embedding<uint8_t>(cmd); break;
      case MAG_DTYPE_INT8:          launch_embedding<int8_t>(cmd); break;
      case MAG_DTYPE_UINT16:        launch_embedding<uint16_t>(cmd); break;
      case MAG_DTYPE_INT16:         launch_embedding<int16_t>(cmd); break;
      case MAG_DTYPE_UINT32:        launch_embedding<uint32_t>(cmd); break;
      case MAG_DTYPE_INT32:         launch_embedding<int32_t>(cmd); break;
      case MAG_DTYPE_UINT64:        launch_embedding<uint64_t>(cmd); break;
      case MAG_DTYPE_INT64:         launch_embedding<int64_t>(cmd); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: embedding: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T>
  __global__ static void cat_kernel_2d(
    T *__restrict__ br, const T *__restrict__ bx,
    int64_t jk_total, int64_t out_jk_base, int64_t out_jk_stride
  ) {
    int64_t jk = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t o  = static_cast<int64_t>(blockIdx.y);
    if (jk >= jk_total) return;
    br[o*out_jk_stride + out_jk_base + jk] = bx[o*jk_total + jk];
  }

  template <typename T>
  __global__ static void cat_kernel_2d_strided(
    T *__restrict__ br, const T *__restrict__ bx,
    int64_t jk_total, int64_t out_jk_base, int64_t out_jk_stride, int64_t outer
  ) {
    int64_t jk = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (jk >= jk_total) return;
    for (int64_t o = blockIdx.y; o < outer; o += gridDim.y)
      br[o*out_jk_stride + out_jk_base + jk] = bx[o*jk_total + jk];
  }

  template <typename T>
  __global__ static void cat_kernel_nc(
    int64_t numel,
    T *__restrict__ br, const T *__restrict__ bx,
    int64_t inner, int64_t xi_dim, int64_t out_dim, int64_t dst_off, int64_t outer,
    mag_tensor_t src_t, int64_t axis
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*blockIdx.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
    for (; flat < numel; flat += step) {
      int64_t k = flat % inner;
      int64_t t = flat / inner;
      int64_t j = t % xi_dim;
      int64_t o = t / xi_dim;
      int64_t src_off = j*src_t.meta.coords.strides[axis];
      int64_t o_rem = o;
      for (int64_t d = axis - 1; d >= 0; --d) {
        src_off += (o_rem % src_t.meta.coords.shape[d])*src_t.meta.coords.strides[d];
        o_rem /= src_t.meta.coords.shape[d];
      }
      int64_t k_rem = k;
      for (int64_t d = src_t.meta.coords.rank - 1; d > axis; --d) {
        src_off += (k_rem % src_t.meta.coords.shape[d])*src_t.meta.coords.strides[d];
        k_rem /= src_t.meta.coords.shape[d];
      }
      br[(o*out_dim + dst_off + j)*inner + k] = bx[src_off];
    }
  }

  template <typename T>
  static void launch_cat(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    mag_assert2(mag_tensor_is_contiguous(r));
    const int64_t dim = cmd.params->cat.dim;
    const int64_t R = r->meta.coords.rank;
    T *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    int64_t inner = 1;
    for (int64_t d = dim+1; d < R; ++d) inner *= r->meta.coords.shape[d];
    int64_t outer = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= r->meta.coords.shape[d];
    int64_t out_dim = r->meta.coords.shape[dim];
    constexpr int64_t MAX_OUTER = 65535;
    int64_t dst_off = 0;
    for (uint32_t i = 0; i < cmd.num_in; ++i) {
      const mag_tensor_t *x = cmd.in[i];
      int64_t xi_dim = x->meta.coords.shape[dim];
      const T *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
      if (mag_tensor_is_contiguous(x)) {
        int64_t jk_total    = xi_dim*inner;
        int64_t out_jk_base = dst_off*inner;
        int64_t out_jk_str  = out_dim*inner;
        dim3 block(MISC_BLOCK_SIZE, 1, 1);
        unsigned gx = static_cast<unsigned>((jk_total + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE);
        if (outer <= MAX_OUTER) {
          dim3 grid(gx, static_cast<unsigned>(outer), 1);
          cat_kernel_2d<T><<<grid, block>>>(br, bx, jk_total, out_jk_base, out_jk_str);
        } else {
          dim3 grid(gx, static_cast<unsigned>(MAX_OUTER), 1);
          cat_kernel_2d_strided<T><<<grid, block>>>(br, bx, jk_total, out_jk_base, out_jk_str, outer);
        }
      } else {
        int64_t numel  = outer*xi_dim*inner;
        int64_t blocks = (numel + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        cat_kernel_nc<T><<<blocks, MISC_BLOCK_SIZE>>>(numel, br, bx, inner, xi_dim, out_dim, dst_off, outer, *x, dim);
      }
      dst_off += xi_dim;
    }
  }

  mag_status_t misc_op_cat(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32:       launch_cat<float>(cmd); break;
      case MAG_DTYPE_FLOAT16:       launch_cat<half>(cmd); break;
      case MAG_DTYPE_BFLOAT16:      launch_cat<__nv_bfloat16>(cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_cat<__nv_fp8_e4m3>(cmd); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8:         launch_cat<uint8_t>(cmd); break;
      case MAG_DTYPE_INT8:          launch_cat<int8_t>(cmd); break;
      case MAG_DTYPE_UINT16:        launch_cat<uint16_t>(cmd); break;
      case MAG_DTYPE_INT16:         launch_cat<int16_t>(cmd); break;
      case MAG_DTYPE_UINT32:        launch_cat<uint32_t>(cmd); break;
      case MAG_DTYPE_INT32:         launch_cat<int32_t>(cmd); break;
      case MAG_DTYPE_UINT64:        launch_cat<uint64_t>(cmd); break;
      case MAG_DTYPE_INT64:         launch_cat<int64_t>(cmd); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: cat: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T>
  [[nodiscard]] static __device__ __forceinline__ float topk_cmp_val(T x) {
    if constexpr (std::is_same_v<T, float>) return x;
    else if constexpr (std::is_same_v<T, half>) return __half2float(x);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(x);
    else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) return static_cast<float>(x);
    else return static_cast<float>(x);
  }

  template <typename T>
  __global__ static void topk_rows_kernel(
    int64_t outer_count,
    int64_t dim_size,
    int64_t k,
    bool largest,
    int64_t R,
    int64_t dim,
    int64_t stride_x_dim,
    int64_t stride_v_dim,
    mag_tensor_t x_t,
    mag_tensor_t v_t,
    mag_tensor_t i_t,
    const T *bx,
    T *bv,
    int64_t *bi,
    char *scratch_base,
    size_t row_bytes
  ) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= outer_count || threadIdx.x != 0) return;
    const int64_t *shape_x = x_t.meta.coords.shape;
    const int64_t *str_x = x_t.meta.coords.strides;
    const int64_t *str_v = v_t.meta.coords.strides;
    (void)i_t;
    T *vals_buf = reinterpret_cast<T *>(scratch_base + static_cast<size_t>(row)*row_bytes);
    int64_t *idx_buf = reinterpret_cast<int64_t *>(vals_buf + dim_size);
    int64_t outer_rank = R - 1;
    int64_t shape_outer[MAG_MAX_DIMS];
    int64_t mult_outer[MAG_MAX_DIMS];
    int64_t outer_to_full[MAG_MAX_DIMS];
    {
      int64_t t=0;
      for (int64_t d=0; d < R; ++d) {
        if (d == dim) continue;
        shape_outer[t] = shape_x[d];
        outer_to_full[t] = d;
        ++t;
      }
      for (int64_t t2=0; t2 < outer_rank; ++t2) {
        int64_t m=1;
        for (int64_t k2=t2+1; k2 < outer_rank; ++k2)
          m *= shape_outer[k2];
        mult_outer[t2] = m;
      }
    }
    int64_t rtmp = row;
    int64_t base_idx[MAG_MAX_DIMS] = {0};
    for (int64_t d=0; d < R; ++d) base_idx[d] = 0;
    for (int64_t t=0; t < outer_rank; ++t) {
      int64_t q = mult_outer[t] == 0 ? 0 : rtmp / mult_outer[t];
      if (mult_outer[t] != 0) rtmp = rtmp % mult_outer[t];
      int64_t fd = outer_to_full[t];
      base_idx[fd] = q;
    }
    base_idx[dim] = 0;
    int64_t off_x0=0;
    int64_t off_v0=0;
    for (int64_t d=0; d < R; ++d) {
      off_x0 += base_idx[d]*str_x[d];
      off_v0 += base_idx[d]*str_v[d];
    }
    for (int64_t p = 0; p < dim_size; ++p) {
      int64_t off_x = off_x0 + p*stride_x_dim;
      vals_buf[p] = bx[off_x];
      idx_buf[p] = p;
    }
    for (int64_t rr=0; rr < k; ++rr) {
      int64_t best = rr;
      for (int64_t p = rr+1; p < dim_size; ++p) {
        T vp = vals_buf[p];
        T vb = vals_buf[best];
        float fvp = topk_cmp_val(vp);
        float fvb = topk_cmp_val(vb);
        bool better;
        if (largest) better = (fvp > fvb) || ((fvp == fvb) && (idx_buf[p] < idx_buf[best]));
        else better = (fvp < fvb) || ((fvp == fvb) && (idx_buf[p] < idx_buf[best]));
        if (better) best = p;
      }
      if (best != rr) {
        T tv = vals_buf[rr];
        vals_buf[rr] = vals_buf[best];
        vals_buf[best] = tv;
        int64_t ti2 = idx_buf[rr];
        idx_buf[rr] = idx_buf[best];
        idx_buf[best] = ti2;
      }
    }
    for (int64_t rr=0; rr < k; ++rr) {
      int64_t off_v = off_v0 + rr*stride_v_dim;
      bv[off_v] = vals_buf[rr];
      bi[off_v] = idx_buf[rr];
    }
  }

  template <typename T>
  static mag_status_t launch_topk(mag_error_t *err, const mag_command_t &cmd) {
    const mag_tensor_t *x = cmd.in[0];
    mag_tensor_t *v = cmd.out[0];
    mag_tensor_t *idx = cmd.out[1];
    const int64_t k = cmd.params->topk.k;
    int64_t dim = cmd.params->topk.dim;
    bool largest = cmd.params->topk.largest;
    int64_t R = x->meta.coords.rank;
    mag_assert2(dim >= 0 && dim < R);
    const int64_t dim_size = x->meta.coords.shape[dim];
    mag_assert2(k > 0 && k <= dim_size);
    int64_t outer_count = x->meta.numel / dim_size;
    if (outer_count <= 0) return MAG_OK;
    size_t row_bytes = static_cast<size_t>(dim_size)*(sizeof(T) + sizeof(int64_t));
    size_t scratch_bytes = row_bytes*static_cast<size_t>(outer_count);
    void *d_scratch = nullptr;
    if (cudaError_t ce = cudaMalloc(&d_scratch, scratch_bytes); mag_unlikely(ce != cudaSuccess)) return mag_set_error(err, MAG_ERR_OOM, "cuda: topk device allocation of %zu bytes failed: %s.", scratch_bytes, cudaGetErrorString(ce));
    const T *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    T *bv = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(v));
    int64_t *bi = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(idx));
    int64_t stride_x_dim = x->meta.coords.strides[dim];
    int64_t stride_v_dim = v->meta.coords.strides[dim];
    topk_rows_kernel<T><<<static_cast<unsigned>(outer_count), 1>>>(
      outer_count, dim_size, k, largest, R, dim, stride_x_dim, stride_v_dim,
      *x, *v, *idx, bx, bv, bi, reinterpret_cast<char *>(d_scratch), row_bytes);
    cuda_check(cudaFree(d_scratch), "topk cudaFree");
    return MAG_OK;
  }

  mag_status_t misc_op_topk(mag_error_t *err, const mag_command_t &cmd) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return launch_topk<float>(err, cmd);
      case MAG_DTYPE_FLOAT16: return launch_topk<half>(err, cmd);
      case MAG_DTYPE_BFLOAT16: return launch_topk<__nv_bfloat16>(err, cmd);
      case MAG_DTYPE_FLOAT8_E4M3FN: return launch_topk<__nv_fp8_e4m3>(err, cmd);
      case MAG_DTYPE_UINT8: return launch_topk<uint8_t>(err, cmd);
      case MAG_DTYPE_INT8: return launch_topk<int8_t>(err, cmd);
      case MAG_DTYPE_UINT16: return launch_topk<uint16_t>(err, cmd);
      case MAG_DTYPE_INT16: return launch_topk<int16_t>(err, cmd);
      case MAG_DTYPE_UINT32: return launch_topk<uint32_t>(err, cmd);
      case MAG_DTYPE_INT32: return launch_topk<int32_t>(err, cmd);
      case MAG_DTYPE_UINT64: return launch_topk<uint64_t>(err, cmd);
      case MAG_DTYPE_INT64: return launch_topk<int64_t>(err, cmd);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: topk: unsupported dtype: %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }

  struct mag_discrete_sample_pair_d {
    float score;
    int64_t idx;
  };

  __device__ __forceinline__ void insertion_sort_pairs_desc(mag_discrete_sample_pair_d *arr, int64_t n) {
    for (int64_t i = 1; i < n; ++i) {
      mag_discrete_sample_pair_d key = arr[i];
      int64_t j = i - 1;
      while (j >= 0 && (arr[j].score < key.score || (arr[j].score == key.score && arr[j].idx > key.idx))) {
        arr[j+1] = arr[j];
        --j;
      }
      arr[j+1] = key;
    }
  }

  template <typename T>
  __global__ static void multinomial_rows_kernel(
    int64_t B,
    int64_t K,
    int64_t num_samples,
    const T *bx,
    int64_t *br,
    uint64_t seed,
    uint64_t subseq0,
    mag_discrete_sample_pair_d *workspace
  ) {
    int64_t b = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (b >= B) return;
    mag_philox4x32_stream_t stream;
    mag_philox4x32_stream_seed(&stream, seed, subseq0 + static_cast<uint64_t>(b));
    const T *w = bx + b*K;
    int64_t *o = br + b*num_samples;
    float sumw = 0.f;
    int64_t nnz = 0;
    for (int64_t i=0; i < K; ++i) {
      auto wi = static_cast<float>(w[i]);
      if (!isfinite(wi) || wi <= 0.f) wi = 0.f;
      sumw += wi;
      if (wi > 0.f) ++nnz;
    }
    mag_discrete_sample_pair_d *arr = workspace + b*K;
    if (!(sumw > 0.f) || nnz == 0) {
      for (int64_t s=0; s < num_samples; ++s) o[s] = -1;
      return;
    }
    int64_t kout = num_samples;
    if (kout > nnz) kout = nnz;
    if (kout <= 0) {
      for (int64_t s=0; s < num_samples; ++s) o[s] = -1;
      return;
    }
    int64_t m = 0;
    for (int64_t i=0; i < K; ++i) {
      auto wi = static_cast<float>(w[i]);
      if (!isfinite(wi) || wi <= 0.f) continue;
      float u = mag_philox4x32_next_float32(&stream);
      u = fmaxf(u, 1e-37f);
      float g = -logf(-logf(u));
      arr[m].score = logf(wi) + g;
      arr[m].idx = i;
      ++m;
    }
    insertion_sort_pairs_desc(arr, m);
    for (int64_t s=0; s < kout; ++s) o[s] = arr[s].idx;
    for (int64_t s=kout; s < num_samples; ++s) o[s] = -1;
  }

  mag_status_t misc_op_multinomial(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    mag_assert2(r->meta.dtype == MAG_DTYPE_INT64);
    int64_t num_samples = cmd.params->multinomial.samples;
    int64_t K = x->meta.coords.shape[x->meta.coords.rank-1];
    if (K <= 0) return MAG_OK;
    int64_t B = x->meta.numel / K;
    if (B <= 0) return MAG_OK;
    size_t ws = static_cast<size_t>(B)*static_cast<size_t>(K)*sizeof(mag_discrete_sample_pair_d);
    void *d_ws = nullptr;
    if (cudaError_t ce = cudaMalloc(&d_ws, ws); mag_unlikely(ce != cudaSuccess)) return mag_set_error(err, MAG_ERR_OOM, "cuda: multinomial device allocation of %zu bytes failed: %s.", ws, cudaGetErrorString(ce));
    uint64_t seed = global_seed.load(std::memory_order_relaxed);
    uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
    int64_t *br = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
    int64_t blocks = (B + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32:
        multinomial_rows_kernel<float><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const float *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
        break;
      case MAG_DTYPE_FLOAT16:
        multinomial_rows_kernel<half><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const half *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
        break;
      case MAG_DTYPE_BFLOAT16:
        multinomial_rows_kernel<__nv_bfloat16><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
        break;
      case MAG_DTYPE_FLOAT8_E4M3FN:
        multinomial_rows_kernel<__nv_fp8_e4m3><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const __nv_fp8_e4m3 *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
        break;
      default:
        cuda_check(cudaFree(d_ws), "multinomial cudaFree");
        return mag_set_error(err, MAG_ERR_KERNEL, "cuda: multinomial: unsupported dtype: %s.", mag_type_trait(x->meta.dtype)->name);
    }
    cuda_check(cudaFree(d_ws), "multinomial cudaFree");
    return MAG_OK;
  }

  [[nodiscard]] __device__ __forceinline__ static int pad_reflect_index(int i, int size) {
    if (size <= 1) return 0;
    int period = (size - 1)<<1;
    i %= period;
    if (i < 0) i += period;
    if (i >= size) i = period - i;
    return i;
  }

  [[nodiscard]] __device__ __forceinline__ static int pad_replicate_index(int i, int size) {
    if (size <= 0) return 0;
    if (i < 0) return 0;
    if (i >= size) return size - 1;
    return i;
  }

  [[nodiscard]] __device__ __forceinline__ static int pad_circular_index(int i, int size) {
    if (size <= 0) return 0;
    i %= size;
    if (i < 0) i += size;
    return i;
  }

  template <typename T, int MODE, bool C>
  __global__ static void pad_kernel(
    int total,
    int R,
    mag_op_params_t plan,
    [[maybe_unused]] mag_coords_iter_t cr,
    mag_coords_iter_t cx,
    T *br,
    const T *bx
  ) {
    int ti = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    const int *in_shape = cx.shape;
    const int *in_stride = cx.strides;
    const int *out_shape = cr.shape;
    T fill = unpack_scalar<T>(plan.pad.value);
    for (; ti < total; ti += step) {
      int ri = C ? ti : mag_coords_iter_to_offset(&cr, ti);
      int tmp = ti;
      int oc[MAG_MAX_DIMS];
      for (int dim = R-1; dim >= 0; --dim) {
        oc[dim] = tmp % out_shape[dim];
        tmp /= out_shape[dim];
      }
      int si[MAG_MAX_DIMS];
      if constexpr (MODE == MAG_PAD_MODE_CONSTANT) {
        bool outside = false;
        for (int dim = 0; dim < R; ++dim) {
          int ic = oc[dim] - plan.pad.pad_before[dim];
          if (ic < 0 || ic >= in_shape[dim]) {
            outside = true;
            break;
          }
          si[dim] = ic;
        }
        if (outside) {
          br[ri] = fill;
          continue;
        }
      } else {
        for (int dim=0; dim < R; ++dim) {
          int ic = oc[dim] - plan.pad.pad_before[dim];
          if constexpr (MODE == MAG_PAD_MODE_REFLECT)
            si[dim] = pad_reflect_index(ic, in_shape[dim]);
          else if constexpr (MODE == MAG_PAD_MODE_REPLICATE)
            si[dim] = pad_replicate_index(ic, in_shape[dim]);
          else if constexpr (MODE == MAG_PAD_MODE_CIRCULAR)
            si[dim] = pad_circular_index(ic, in_shape[dim]);
        }
      }
      int xi = 0;
      for (int dim=0; dim < R; ++dim)
        xi += si[dim]*in_stride[dim];
      br[ri] = bx[xi];
    }
  }

  template <typename T, bool C>
  static mag_status_t launch_pad_mode(
    mag_error_t *err,
    int blocks,
    int n,
    int R,
    const mag_op_params_t &plan,
    mag_coords_iter_t cr,
    mag_coords_iter_t cx,
    T *br,
    const T *bx
  ) {
    switch (plan.pad.mode) {
      case MAG_PAD_MODE_CONSTANT: pad_kernel<T, MAG_PAD_MODE_CONSTANT, C><<<blocks, MISC_BLOCK_SIZE>>>(n, R, plan, cr, cx, br, bx); break;
      case MAG_PAD_MODE_REFLECT: pad_kernel<T, MAG_PAD_MODE_REFLECT, C> <<<blocks, MISC_BLOCK_SIZE>>>(n, R, plan, cr, cx, br, bx); break;
      case MAG_PAD_MODE_REPLICATE: pad_kernel<T, MAG_PAD_MODE_REPLICATE, C> <<<blocks, MISC_BLOCK_SIZE>>>(n, R, plan, cr, cx, br, bx); break;
      case MAG_PAD_MODE_CIRCULAR: pad_kernel<T, MAG_PAD_MODE_CIRCULAR, C> <<<blocks, MISC_BLOCK_SIZE>>>(n, R, plan, cr, cx, br, bx); break;
      default: return mag_set_error(err, MAG_ERR_PARAM, "cuda: pad: unsupported mode: %d.", plan.pad.mode);
    }
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t launch_pad(mag_error_t *err, mag_tensor_t *r, const mag_tensor_t *x, const mag_op_params_t &plan) {
    int n = static_cast<int>(mag_tensor_numel(r));
    int blocks = (n + MISC_BLOCK_SIZE - 1)/MISC_BLOCK_SIZE;
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->meta.coords);
    mag_coords_iter_init(&cx, &x->meta.coords);
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    if (mag_tensor_is_contiguous(r)) {
      return launch_pad_mode<T, true>(err, blocks, n, plan.pad.rank, plan, cr, cx, br, bx);
    } else {
      return launch_pad_mode<T, false>(err, blocks, n, plan.pad.rank, plan, cr, cx, br, bx);
    }
  }

  mag_status_t misc_op_pad(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const auto &plan = *cmd.params;
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return launch_pad<float>(err, r, x, plan);
      case MAG_DTYPE_FLOAT16: return launch_pad<half>(err, r, x, plan);
      case MAG_DTYPE_BFLOAT16: return launch_pad<__nv_bfloat16>(err, r, x, plan);
      case MAG_DTYPE_FLOAT8_E4M3FN: return launch_pad<__nv_fp8_e4m3>(err, r, x, plan);
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: return launch_pad<uint8_t>(err, r, x, plan);
      case MAG_DTYPE_INT8: return launch_pad<int8_t>(err, r, x, plan);
      case MAG_DTYPE_UINT16: return launch_pad<uint16_t>(err, r, x, plan);
      case MAG_DTYPE_INT16: return launch_pad<int16_t>(err, r, x, plan);
      case MAG_DTYPE_UINT32: return launch_pad<uint32_t>(err, r, x, plan);
      case MAG_DTYPE_INT32: return launch_pad<int32_t>(err, r, x, plan);
      case MAG_DTYPE_UINT64: return launch_pad<uint64_t>(err, r, x, plan);
      case MAG_DTYPE_INT64: return launch_pad<int64_t>(err, r, x, plan);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: pad: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
  }

  template <typename T, typename ACC, bool is_prod>
  __global__ static void cu_scan_rows_kernel(
    int outer_count,
    int dim_size,
    int R,
    int dim,
    int stride_x_dim,
    int stride_r_dim,
    mag_tensor_t x_t, // TODO
    mag_tensor_t r_t,
    const T *bx,
    T *br
  ) {
    int row = blockIdx.x;
    if (row >= outer_count || threadIdx.x != 0) return;
    const int64_t *shape_x = x_t.meta.coords.shape;
    const int64_t *str_x = x_t.meta.coords.strides;
    const int64_t *str_r = r_t.meta.coords.strides;
    int outer_rank = R - 1;
    int shape_outer[MAG_MAX_DIMS];
    int mult_outer[MAG_MAX_DIMS];
    int outer_to_full[MAG_MAX_DIMS];
    {
      int t=0;
      for (int d=0; d < R; ++d) {
        if (d == dim) continue;
        shape_outer[t] = shape_x[d];
        outer_to_full[t] = d;
        ++t;
      }
      for (int t2=0; t2 < outer_rank; ++t2) {
        int m=1;
        for (int k2=t2+1; k2 < outer_rank; ++k2) m *= shape_outer[k2];
        mult_outer[t2] = m;
      }
    }
    int rtmp = row;
    int base_idx[MAG_MAX_DIMS] = {0};
    for (int t=0; t < outer_rank; ++t) {
      int q = mult_outer[t] == 0 ? 0 : rtmp / mult_outer[t];
      if (mult_outer[t] != 0) rtmp = rtmp % mult_outer[t];
      base_idx[outer_to_full[t]] = q;
    }
    base_idx[dim] = 0;
    int off_x0=0;
    int off_r0=0;
    for (int d=0; d < R; ++d) {
      off_x0 += base_idx[d]*str_x[d];
      off_r0 += base_idx[d]*str_r[d];
    }
    auto acc = is_prod ? static_cast<ACC>(1) : static_cast<ACC>(0);
    for (int p=0; p < dim_size; ++p) {
      int off_x = off_x0 + p*stride_x_dim;
      int off_r = off_r0 + p*stride_r_dim;
      T xv = bx[off_x];
      if constexpr (std::is_floating_point_v<ACC>) {
        auto fv = static_cast<float>(xv);
        if constexpr (is_prod) acc = acc*static_cast<ACC>(fv);
        else acc = acc + static_cast<ACC>(fv);
      } else {
        if constexpr (is_prod) acc = acc*static_cast<ACC>(xv);
        else acc = acc + static_cast<ACC>(xv);
      }
      br[off_r] = static_cast<T>(acc);
    }
  }

  template <typename T, bool is_max>
  __global__ static void cu_ext_rows_kernel(
    int64_t outer_count,
    int64_t dim_size,
    int64_t R,
    int64_t dim,
    int64_t stride_x_dim,
    int64_t stride_v_dim,
    int64_t stride_i_dim,
    mag_tensor_t x_t, // TODO
    mag_tensor_t v_t,
    mag_tensor_t i_t,
    const T *bx,
    T *bv,
    int64_t *bi
  ) {
    int row = blockIdx.x;
    if (row >= outer_count || threadIdx.x != 0) return;
    const int64_t *shape_x = x_t.meta.coords.shape;
    const int64_t *str_x = x_t.meta.coords.strides;
    const int64_t *str_v = v_t.meta.coords.strides;
    const int64_t *str_i = i_t.meta.coords.strides;
    int outer_rank = R - 1;
    int shape_outer[MAG_MAX_DIMS];
    int mult_outer[MAG_MAX_DIMS];
    int outer_to_full[MAG_MAX_DIMS];
    {
      int t=0;
      for (int d=0; d < R; ++d) {
        if (d == dim) continue;
        shape_outer[t] = shape_x[d];
        outer_to_full[t] = d;
        ++t;
      }
      for (int t2=0; t2 < outer_rank; ++t2) {
        int m=1;
        for (int k2=t2+1; k2 < outer_rank; ++k2) m *= shape_outer[k2];
        mult_outer[t2] = m;
      }
    }
    int rtmp = row;
    int base_idx[MAG_MAX_DIMS] = {0};
    for (int t=0; t < outer_rank; ++t) {
      int q = mult_outer[t] == 0 ? 0 : rtmp / mult_outer[t];
      if (mult_outer[t] != 0) rtmp = rtmp % mult_outer[t];
      base_idx[outer_to_full[t]] = q;
    }
    base_idx[dim] = 0;
    int off_x0=0;
    int off_v0=0;
    int off_i0=0;
    for (int d=0; d < R; ++d) {
      off_x0 += base_idx[d]*str_x[d];
      off_v0 += base_idx[d]*str_v[d];
      off_i0 += base_idx[d]*str_i[d];
    }
    T best{};
    int best_idx = 0;
    for (int p=0; p < dim_size; ++p) {
      int off_x = off_x0 + p*stride_x_dim;
      int off_v = off_v0 + p*stride_v_dim;
      int off_i = off_i0 + p*stride_i_dim;
      T xv = bx[off_x];
      if (p == 0) {
        best = xv;
        best_idx = 0;
      } else {
        auto xvc = static_cast<float>(xv);
        auto bestc = static_cast<float>(best);
        bool better = is_max ? xvc > bestc : xvc < bestc;
        if (better) { best = xv; best_idx = p; }
      }
      bv[off_v] = best;
      bi[off_i] = best_idx;
    }
  }

  template <typename T, typename ACC, bool is_prod>
  static void launch_cu_scan(const mag_command_t &cmd) {
    const mag_tensor_t *x = cmd.in[0];
    mag_tensor_t *r = cmd.out[0];
    int dim = cmd.params->cumu.dim;
    if (dim < 0) dim += static_cast<int>(x->meta.coords.rank);
    int R = static_cast<int>(x->meta.coords.rank);
    int dim_size = static_cast<int>(x->meta.coords.shape[dim]);
    if (dim_size <= 0) return;
    int outer_count = numel_i32(x) / dim_size;
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    cu_scan_rows_kernel<T, ACC, is_prod><<<static_cast<unsigned>(outer_count), 1>>>(outer_count, dim_size, R, dim, x->meta.coords.strides[dim], r->meta.coords.strides[dim], *x, *r, bx, br);
  }

  template <typename T, bool is_max>
  static void launch_cu_ext(const mag_command_t &cmd) {
    const mag_tensor_t *x = cmd.in[0];
    mag_tensor_t *v = cmd.out[0];
    mag_tensor_t *idx = cmd.out[1];
    int dim = cmd.params->cumu.dim;
    if (dim < 0) dim += static_cast<int>(x->meta.coords.rank);
    int R = static_cast<int>(x->meta.coords.rank);
    int dim_size = static_cast<int>(x->meta.coords.shape[dim]);
    if (dim_size <= 0) return;
    int outer_count = numel_i32(x) / dim_size;
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    auto *bv = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(v));
    auto *bi = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(idx));
    cu_ext_rows_kernel<T, is_max><<<static_cast<unsigned>(outer_count), 1>>>(
      outer_count, dim_size, R, dim, x->meta.coords.strides[dim], v->meta.coords.strides[dim], idx->meta.coords.strides[dim],
      *x, *v, *idx, bx, bv, bi
    );
  }

  static mag_status_t impl_cu_scan(mag_error_t *err, const mag_command_t &cmd, bool is_prod) {
    mag_tensor_t *r = cmd.out[0];
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32:
        if (is_prod) launch_cu_scan<float, double, true>(cmd);
        else launch_cu_scan<float, double, false>(cmd);
        break;
      case MAG_DTYPE_FLOAT16:
        if (is_prod) launch_cu_scan<half, float, true>(cmd);
        else launch_cu_scan<half, float, false>(cmd);
        break;
      case MAG_DTYPE_BFLOAT16:
        if (is_prod) launch_cu_scan<__nv_bfloat16, float, true>(cmd);
        else launch_cu_scan<__nv_bfloat16, float, false>(cmd);
        break;
      case MAG_DTYPE_FLOAT8_E4M3FN:
        if (is_prod) launch_cu_scan<__nv_fp8_e4m3, float, true>(cmd);
        else launch_cu_scan<__nv_fp8_e4m3, float, false>(cmd);
        break;
      case MAG_DTYPE_UINT8:
        if (is_prod) launch_cu_scan<uint8_t, uint64_t, true>(cmd);
        else launch_cu_scan<uint8_t, uint64_t, false>(cmd);
        break;
      case MAG_DTYPE_INT8:
        if (is_prod) launch_cu_scan<int8_t, int64_t, true>(cmd);
        else launch_cu_scan<int8_t, int64_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT16:
        if (is_prod) launch_cu_scan<uint16_t, uint64_t, true>(cmd);
        else launch_cu_scan<uint16_t, uint64_t, false>(cmd);
        break;
      case MAG_DTYPE_INT16:
        if (is_prod) launch_cu_scan<int16_t, int64_t, true>(cmd);
        else launch_cu_scan<int16_t, int64_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT32:
        if (is_prod) launch_cu_scan<uint32_t, uint64_t, true>(cmd);
        else launch_cu_scan<uint32_t, uint64_t, false>(cmd);
        break;
      case MAG_DTYPE_INT32:
        if (is_prod) launch_cu_scan<int32_t, int64_t, true>(cmd);
        else launch_cu_scan<int32_t, int64_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT64:
        if (is_prod) launch_cu_scan<uint64_t, uint64_t, true>(cmd);
        else launch_cu_scan<uint64_t, uint64_t, false>(cmd);
        break;
      case MAG_DTYPE_INT64:
        if (is_prod) launch_cu_scan<int64_t, int64_t, true>(cmd);
        else launch_cu_scan<int64_t, int64_t, false>(cmd);
        break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: cu*: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  static mag_status_t impl_cu_ext(mag_error_t *err, const mag_command_t &cmd, bool is_max) {
    mag_tensor_t *v = cmd.out[0];
    switch (v->meta.dtype) {
      case MAG_DTYPE_FLOAT32:
        if (is_max) launch_cu_ext<float, true>(cmd);
        else launch_cu_ext<float, false>(cmd);
        break;
      case MAG_DTYPE_FLOAT16:
        if (is_max) launch_cu_ext<half, true>(cmd);
        else launch_cu_ext<half, false>(cmd);
        break;
      case MAG_DTYPE_BFLOAT16:
        if (is_max) launch_cu_ext<__nv_bfloat16, true>(cmd);
        else launch_cu_ext<__nv_bfloat16, false>(cmd);
        break;
      case MAG_DTYPE_FLOAT8_E4M3FN:
        if (is_max) launch_cu_ext<__nv_fp8_e4m3, true>(cmd);
        else launch_cu_ext<__nv_fp8_e4m3, false>(cmd);
        break;
      case MAG_DTYPE_UINT8:
        if (is_max) launch_cu_ext<uint8_t, true>(cmd);
        else launch_cu_ext<uint8_t, false>(cmd);
        break;
      case MAG_DTYPE_INT8:
        if (is_max) launch_cu_ext<int8_t, true>(cmd);
        else launch_cu_ext<int8_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT16:
        if (is_max) launch_cu_ext<uint16_t, true>(cmd);
        else launch_cu_ext<uint16_t, false>(cmd);
        break;
      case MAG_DTYPE_INT16:
        if (is_max) launch_cu_ext<int16_t, true>(cmd);
        else launch_cu_ext<int16_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT32:
        if (is_max) launch_cu_ext<uint32_t, true>(cmd);
        else launch_cu_ext<uint32_t, false>(cmd);
        break;
      case MAG_DTYPE_INT32:
        if (is_max) launch_cu_ext<int32_t, true>(cmd);
        else launch_cu_ext<int32_t, false>(cmd);
        break;
      case MAG_DTYPE_UINT64:
        if (is_max) launch_cu_ext<uint64_t, true>(cmd);
        else launch_cu_ext<uint64_t, false>(cmd);
        break;
      case MAG_DTYPE_INT64:
        if (is_max) launch_cu_ext<int64_t, true>(cmd);
        else launch_cu_ext<int64_t, false>(cmd);
        break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: cu*: unsupported dtype: %s.", mag_type_trait(v->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t misc_op_cusum(mag_error_t *err, const mag_command_t &cmd) { return impl_cu_scan(err, cmd, false); }
  mag_status_t misc_op_cuprod(mag_error_t *err, const mag_command_t &cmd) { return impl_cu_scan(err, cmd, true); }
  mag_status_t misc_op_cumax(mag_error_t *err, const mag_command_t &cmd) { return impl_cu_ext(err, cmd, true); }
  mag_status_t misc_op_cumin(mag_error_t *err, const mag_command_t &cmd) { return impl_cu_ext(err, cmd, false); }

  [[nodiscard]] __device__ static int repeat_in_elem_offset_dev(
    int64_t flat_out,
    const mag_op_params_t *plan,
    const mag_coords_iter_t *cx
  ) {
    int64_t tmp = flat_out;
    int64_t off = 0;
    for (int64_t d = plan->repeat.rank - 1; d >= 0; --d) {
      int64_t oc = tmp % plan->repeat.out_shape[d];
      tmp /= plan->repeat.out_shape[d];
      int64_t ic = oc % plan->repeat.in_shape[d];
      int64_t id = d - (plan->repeat.rank - plan->repeat.in_rank);
      if (id >= 0)
        off += ic*cx->strides[id];
    }
    return off;
  }

  template <typename T>
  __global__ static void repeat_kernel(
    int64_t on,
    T *__restrict__ br,
    const T *__restrict__ bx,
    mag_op_params_t plan,
    mag_coords_iter_t cr,
    mag_coords_iter_t cx
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    for (; flat < on; flat += step) {
      int64_t ri = mag_coords_iter_to_offset(&cr, flat);
      int64_t xi = repeat_in_elem_offset_dev(flat, &plan, &cx);
      br[ri] = bx[xi];
    }
  }

  static mag_status_t launch_repeat(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const auto *plan = cmd.params;
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->meta.coords);
    mag_coords_iter_init(&cx, &x->meta.coords);
    int64_t on = r->meta.numel;
    unsigned block = 256;
    unsigned grid = static_cast<unsigned>((on + block - 1)/block);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: repeat_kernel<float><<<grid, block>>>(
        on,
        reinterpret_cast<float *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const float *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_FLOAT16: repeat_kernel<half><<<grid, block>>>(
        on,
        reinterpret_cast<half *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const half *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_BFLOAT16: repeat_kernel<__nv_bfloat16><<<grid, block>>>(
        on,
        reinterpret_cast<__nv_bfloat16 *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: repeat_kernel<__nv_fp8_e4m3><<<grid, block>>>(
        on,
        reinterpret_cast<__nv_fp8_e4m3 *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const __nv_fp8_e4m3 *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: repeat_kernel<uint8_t><<<grid, block>>>(
        on,
        reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_INT8: repeat_kernel<int8_t><<<grid, block>>>(
        on,
        reinterpret_cast<int8_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const int8_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_UINT16: repeat_kernel<uint16_t><<<grid, block>>>(
        on,
        reinterpret_cast<uint16_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const uint16_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_INT16: repeat_kernel<int16_t><<<grid, block>>>(
        on,
        reinterpret_cast<int16_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const int16_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_UINT32: repeat_kernel<uint32_t><<<grid, block>>>(
        on,
        reinterpret_cast<uint32_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const uint32_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_INT32: repeat_kernel<int32_t><<<grid, block>>>(
        on,
        reinterpret_cast<int32_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const int32_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_UINT64: repeat_kernel<uint64_t><<<grid, block>>>(
        on,
        reinterpret_cast<uint64_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const uint64_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      case MAG_DTYPE_INT64: repeat_kernel<int64_t><<<grid, block>>>(
        on,
        reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(x)),
        *plan, cr, cx
      ); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: repeat: unsupported dtype: %s.", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t misc_op_repeat(mag_error_t *err, const mag_command_t &cmd) { return launch_repeat(err, cmd); }

  mag_status_t misc_op_repeat_interleave(mag_error_t *err, const mag_command_t &cmd) {
    (void)err;
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const auto *plan = cmd.params;
    mag_assert2(mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x));
    size_t elsz = static_cast<size_t>(mag_tensor_numbytes(r) / mag_tensor_numel(r));
    if (plan->repeat_interleave.flatten) {
      int64_t n = x->meta.numel;
      int64_t out_i = 0;
      for (int64_t i=0; i < n; ++i) {
        int64_t rep = plan->repeat_interleave.count_len == 1 ? plan->repeat_interleave.counts[0] : plan->repeat_interleave.counts[i];
        for (int64_t k=0; k < rep; ++k) {
          const uint8_t *src = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(x)) + i*static_cast<int64_t>(elsz);
          uint8_t *dst = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r)) + out_i*static_cast<int64_t>(elsz);
          cudaMemcpy(dst, src, elsz, cudaMemcpyDeviceToDevice);
          ++out_i;
        }
      }
      return MAG_OK;
    }
    int64_t dim = plan->repeat_interleave.dim;
    int64_t R = x->meta.coords.rank;
    int64_t inner_block = 1;
    for (int64_t d = dim+1; d < R; ++d) inner_block *= x->meta.coords.shape[d];
    int64_t outer_count = 1;
    for (int64_t d=0; d < dim; ++d) outer_count *= x->meta.coords.shape[d];
    int64_t axis_len = x->meta.coords.shape[dim];
    int64_t mult[MAG_MAX_DIMS];
    for (int64_t d = 0; d < dim; ++d) {
      int64_t m = 1;
      for (int64_t k = d + 1; k < dim; ++k) m *= x->meta.coords.shape[k];
      mult[d] = m;
    }
    for (int64_t p=0; p < outer_count; ++p) {
      int64_t idx_prefix[MAG_MAX_DIMS];
      int64_t rtmp = p;
      for (int64_t d = 0; d < dim; ++d) {
        int64_t q = !mult[d] ? 0 : rtmp/mult[d];
        if (mult[d] != 0) rtmp = rtmp%mult[d];
        idx_prefix[d] = q;
      }
      int64_t moff = 0;
      for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->meta.coords.strides[d];
      int64_t smoff = 0;
      for (int64_t d=0; d < dim; ++d) smoff += idx_prefix[d]*x->meta.coords.strides[d];
      int64_t cur = 0;
      for (int64_t a=0; a < axis_len; ++a) {
        int64_t rep = plan->repeat_interleave.count_len == 1 ? plan->repeat_interleave.counts[0] : plan->repeat_interleave.counts[a];
        int64_t oel = moff + cur*r->meta.coords.strides[dim];
        int64_t sel = smoff + a*x->meta.coords.strides[dim];
        const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(x)) + sel*static_cast<int64_t>(elsz);
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r)) + oel*static_cast<int64_t>(elsz);
        for (int64_t k=0; k < rep; ++k) {
          cudaMemcpy(
            dst_ptr + k*inner_block*static_cast<int64_t>(elsz),
            src_ptr,
            static_cast<size_t>(inner_block)*elsz,
            cudaMemcpyDeviceToDevice
          );
        }
        cur += rep;
      }
    }
    return MAG_OK;
  }

  template <typename T, bool is_int>
  __global__ static void index_add_kernel(
    int64_t total,
    int64_t R,
    int64_t axis,
    int64_t self_ax,
    T *__restrict__ bs,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    mag_tensor_t self,
    mag_tensor_t source,
    mag_tensor_t index,
    double alpha
  ) {
    for (int64_t flat=0; flat < total; ++flat) {
      int64_t tmp = flat;
      int64_t sc[MAG_MAX_DIMS];
      for (int64_t d = R-1; d >= 0; --d) {
        sc[d] = tmp % source.meta.coords.shape[d];
        tmp /= source.meta.coords.shape[d];
      }
      int64_t j = sc[axis];
      int64_t idx_off = j*index.meta.coords.strides[0];
      int64_t g = bi[idx_off];
      if (g < 0) g += self_ax;
      int64_t src_off = 0;
      for (int64_t d=0; d < R; ++d) src_off += sc[d]*source.meta.coords.strides[d];
      sc[axis] = g;
      int64_t dst_off = 0;
      for (int64_t d=0; d < R; ++d) dst_off += sc[d]*self.meta.coords.strides[d];
      if (is_int) {
        auto cur = static_cast<int64_t>(bs[dst_off]);
        auto add = static_cast<int64_t>(bx[src_off])*static_cast<int64_t>(alpha);
        bs[dst_off] = static_cast<T>(cur + add);
      } else {
        auto cur = static_cast<float>(bs[dst_off]);
        auto add = static_cast<float>(bx[src_off])*static_cast<float>(alpha);
        bs[dst_off] = static_cast<T>(cur + add);
      }
    }
  }

  static mag_status_t launch_index_add(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *self = cmd.out[0];
    const mag_tensor_t *source = cmd.in[1];
    const mag_tensor_t *index = cmd.in[2];
    int64_t axis = cmd.params->index_add.dim;
    double alpha = cmd.params->index_add.alpha;
    if (axis < 0) axis += self->meta.coords.rank;
    int64_t R = self->meta.coords.rank;
    int64_t total = source->meta.numel;
    int64_t self_ax = self->meta.coords.shape[axis];
    switch (self->meta.dtype) {
      case MAG_DTYPE_FLOAT32: index_add_kernel<float, false><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<float *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const float *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_FLOAT16: index_add_kernel<half, false><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<half *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const half *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_BFLOAT16: index_add_kernel<__nv_bfloat16, false><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<__nv_bfloat16 *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: index_add_kernel<__nv_fp8_e4m3, false><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<__nv_fp8_e4m3 *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const __nv_fp8_e4m3 *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_UINT8: index_add_kernel<uint8_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_INT8: index_add_kernel<int8_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<int8_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const int8_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_UINT16: index_add_kernel<uint16_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<uint16_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const uint16_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_INT16: index_add_kernel<int16_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<int16_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const int16_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_UINT32: index_add_kernel<uint32_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<uint32_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const uint32_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_INT32: index_add_kernel<int32_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<int32_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const int32_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_UINT64: index_add_kernel<uint64_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<uint64_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const uint64_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      case MAG_DTYPE_INT64: index_add_kernel<int64_t, true><<<1, 1>>>(
        total, R, axis, self_ax,
        reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(self)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(source)),
        reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
        *self, *source, *index, alpha
      ); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: index_add_: unsupported dtype: %s.", mag_type_trait(self->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t misc_op_index_add(mag_error_t *err, const mag_command_t &cmd) { return launch_index_add(err, cmd); }

  template <typename T>
  __global__ static void scatter_kernel(
    int64_t total,
    int64_t rank,
    int64_t axis,
    int64_t self_ax,
    T *__restrict__ bs,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    mag_tensor_t self,
    mag_tensor_t src,
    mag_tensor_t index
  ) {
    int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    for (; flat < total; flat += step) {
      int64_t ic[MAG_MAX_DIMS];
      int64_t tmp = flat;
      for (int64_t d = rank-1; d >= 0; --d) { ic[d] = tmp % index.meta.coords.shape[d]; tmp /= index.meta.coords.shape[d]; }
      int64_t idx_off = 0, src_off = 0, dst_off = 0;
      for (int64_t d=0; d < rank; ++d) {
        idx_off += ic[d]*index.meta.coords.strides[d];
        src_off += ic[d]*src.meta.coords.strides[d];
        if (d != axis) dst_off += ic[d]*self.meta.coords.strides[d];
      }
      int64_t g = __ldg(bi + idx_off);
      if (g < 0) g += self_ax;
      dst_off += g*self.meta.coords.strides[axis];
      bs[dst_off] = bx[src_off];
    }
  }

  template <typename T>
  static void launch_scatter(const mag_command_t &cmd) {
    mag_tensor_t *self = cmd.out[0];
    const mag_tensor_t *src = cmd.in[1];
    const mag_tensor_t *index = cmd.in[2];
    int64_t axis = cmd.params->scatter.dim;
    if (axis < 0) axis += self->meta.coords.rank;
    mag_assert2(axis >= 0 && axis < self->meta.coords.rank);
    int64_t rank = index->meta.coords.rank;
    int64_t total = index->meta.numel;
    if (total <= 0) return;
    int64_t self_ax = self->meta.coords.shape[axis];
    int64_t blocks = (total + MISC_BLOCK_SIZE - 1)/MISC_BLOCK_SIZE;
    scatter_kernel<T><<<blocks, MISC_BLOCK_SIZE>>>(
      total, rank, axis, self_ax,
      reinterpret_cast<T *>(mag_tensor_data_ptr_mut(self)),
      reinterpret_cast<const T *>(mag_tensor_data_ptr(src)),
      reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
      *self, *src, *index
    );
  }

  mag_status_t misc_op_scatter(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *self = cmd.out[0];
    switch (self->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_scatter<float>(cmd); break;
      case MAG_DTYPE_FLOAT16: launch_scatter<half>(cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_scatter<__nv_bfloat16>(cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_scatter<__nv_fp8_e4m3>(cmd); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_scatter<uint8_t>(cmd); break;
      case MAG_DTYPE_INT8: launch_scatter<int8_t>(cmd); break;
      case MAG_DTYPE_UINT16: launch_scatter<uint16_t>(cmd); break;
      case MAG_DTYPE_INT16: launch_scatter<int16_t>(cmd); break;
      case MAG_DTYPE_UINT32: launch_scatter<uint32_t>(cmd); break;
      case MAG_DTYPE_INT32: launch_scatter<int32_t>(cmd); break;
      case MAG_DTYPE_UINT64: launch_scatter<uint64_t>(cmd); break;
      case MAG_DTYPE_INT64: launch_scatter<int64_t>(cmd); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: scatter: unsupported dtype: %s.", mag_type_trait(self->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <typename T, bool is_int>
  __global__ static void scatter_add_kernel(
    int64_t num_rows,
    int64_t rank,
    int64_t axis,
    int64_t self_ax,
    int64_t s_axis,
    T *__restrict__ bs,
    const T *__restrict__ bx,
    const int64_t *__restrict__ bi,
    mag_tensor_t self,
    mag_tensor_t src,
    mag_tensor_t index
  ) {
    int64_t row = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    int64_t ist = index.meta.coords.strides[axis];
    int64_t xst = src.meta.coords.strides[axis];
    int64_t rst = self.meta.coords.strides[axis];
    for (; row < num_rows; row += step) {
      int64_t c[MAG_MAX_DIMS];
      int64_t rem = row;
      for (int64_t d = rank-1; d >= 0; --d) { if (d == axis) continue; c[d] = rem % index.meta.coords.shape[d]; rem /= index.meta.coords.shape[d]; }
      int64_t idx_row = 0, src_row = 0, dst_row = 0;
      for (int64_t d=0; d < rank; ++d) { if (d == axis) continue; idx_row += c[d]*index.meta.coords.strides[d]; src_row += c[d]*src.meta.coords.strides[d]; dst_row += c[d]*self.meta.coords.strides[d]; }
      for (int64_t j=0; j < s_axis; ++j) {
        int64_t g = bi[idx_row + j*ist];
        if (g < 0) g += self_ax;
        int64_t dst_off = dst_row + g*rst;
        int64_t src_off = src_row + j*xst;
        if (is_int) {
          auto cur = static_cast<int64_t>(bs[dst_off]);
          auto add = static_cast<int64_t>(bx[src_off]);
          bs[dst_off] = static_cast<T>(cur + add);
        } else {
          auto cur = static_cast<float>(bs[dst_off]);
          auto add = static_cast<float>(bx[src_off]);
          bs[dst_off] = static_cast<T>(cur + add);
        }
      }
    }
  }

  template <typename T, bool is_int>
  static void launch_scatter_add_typed(const mag_command_t &cmd) {
    mag_tensor_t *self = cmd.out[0];
    const mag_tensor_t *src = cmd.in[1];
    const mag_tensor_t *index = cmd.in[2];
    int64_t axis = cmd.params->scatter.dim;
    if (axis < 0) axis += self->meta.coords.rank;
    int64_t rank = index->meta.coords.rank;
    int64_t total = index->meta.numel;
    if (total <= 0) return;
    int64_t s_axis = index->meta.coords.shape[axis];
    int64_t self_ax = self->meta.coords.shape[axis];
    int64_t num_rows = total/s_axis;
    int64_t blocks = (num_rows + MISC_BLOCK_SIZE - 1)/MISC_BLOCK_SIZE;
    scatter_add_kernel<T, is_int><<<blocks, MISC_BLOCK_SIZE>>>(
      num_rows, rank, axis, self_ax, s_axis,
      reinterpret_cast<T *>(mag_tensor_data_ptr_mut(self)),
      reinterpret_cast<const T *>(mag_tensor_data_ptr(src)),
      reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index)),
      *self, *src, *index
    );
  }

  mag_status_t misc_op_scatter_add(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *self = cmd.out[0];
    switch (self->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_scatter_add_typed<float, false>(cmd); break;
      case MAG_DTYPE_FLOAT16: launch_scatter_add_typed<half, false>(cmd); break;
      case MAG_DTYPE_BFLOAT16: launch_scatter_add_typed<__nv_bfloat16, false>(cmd); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_scatter_add_typed<__nv_fp8_e4m3, false>(cmd); break;
      case MAG_DTYPE_UINT8: launch_scatter_add_typed<uint8_t, true>(cmd); break;
      case MAG_DTYPE_INT8: launch_scatter_add_typed<int8_t, true>(cmd); break;
      case MAG_DTYPE_UINT16: launch_scatter_add_typed<uint16_t, true>(cmd); break;
      case MAG_DTYPE_INT16: launch_scatter_add_typed<int16_t, true>(cmd); break;
      case MAG_DTYPE_UINT32: launch_scatter_add_typed<uint32_t, true>(cmd); break;
      case MAG_DTYPE_INT32: launch_scatter_add_typed<int32_t, true>(cmd); break;
      case MAG_DTYPE_UINT64: launch_scatter_add_typed<uint64_t, true>(cmd); break;
      case MAG_DTYPE_INT64: launch_scatter_add_typed<int64_t, true>(cmd); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: scatter_add: unsupported dtype: %s.", mag_type_trait(self->meta.dtype)->name);
    }
    return MAG_OK;
  }
}
