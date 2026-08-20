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

#include "mag_cuda_unary.cuh"

#include <array>

namespace mag {
  template <typename Src, typename Dst, typename I, const bool C>
  __global__ static void cast_kernel(I numel, Dst *__restrict__ o, const Src *__restrict__ x, [[maybe_unused]] coords_iter<I> xc) {
    I i = static_cast<I>(blockDim.x)*static_cast<I>(blockIdx.x) + static_cast<I>(threadIdx.x);
    if constexpr (C) {
      if (i >= numel) return;
      o[i] = static_cast<Dst>(x[i]);
    } else {
      I step = static_cast<I>(blockDim.x)*static_cast<I>(gridDim.x);
      for (; i < numel; i += step)
        o[i] = static_cast<Dst>(x[xc(i)]);
    }
  }

  template <typename Src, typename Dst, typename I>
  static void launch_cast_indexed(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    auto blocks = static_cast<unsigned>(std::min((numel+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE, static_cast<int64_t>(std::numeric_limits<int>::max())));
    auto *pr = reinterpret_cast<Dst *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const Src *>(mag_tensor_data_ptr(x));
    if (mag_tensor_is_contiguous(x)) {
      cast_kernel<Src, Dst, I, true><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(static_cast<I>(numel), pr, px, {});
    } else {
      coords_iter<I> xc {x};
      cast_kernel<Src, Dst, I, false><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(static_cast<I>(numel), pr, px, xc);
    }
  }

  template <typename Src, typename Dst>
  static void mag_cast_launcher(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    if (mag_unlikely(numel <= 0)) return;
    if (mag_likely(can_all_use_i32_indexes(r, x))) {
      launch_cast_indexed<Src, Dst, int32_t>(r, x, stream);
    } else {
      launch_cast_indexed<Src, Dst, int64_t>(r, x, stream);
    }
  }

  mag_status_t unary_op_cast(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    using cast_fn = void (mag_tensor_t *, const mag_tensor_t *, cudaStream_t);
    static constexpr void (*const cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(mag_tensor_t *, const mag_tensor_t *, cudaStream_t) = {
      [MAG_DTYPE_FLOAT32] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<float, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<float, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<float, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<float, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<float, uint8_t>,   // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<float, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<float, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<float, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<float, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<float, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<float, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<float, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<float, int64_t>,
      },
      [MAG_DTYPE_FLOAT16] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<half, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<half, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<half, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<half, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<half, uint8_t>,   // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<half, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<half, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<half, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<half, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<half, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<half, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<half, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<half, int64_t>,
      },
      [MAG_DTYPE_BFLOAT16] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<__nv_bfloat16, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<__nv_bfloat16, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<__nv_bfloat16, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<__nv_bfloat16, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<__nv_bfloat16, uint8_t>,   // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<__nv_bfloat16, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<__nv_bfloat16, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<__nv_bfloat16, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<__nv_bfloat16, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<__nv_bfloat16, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<__nv_bfloat16, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<__nv_bfloat16, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<__nv_bfloat16, int64_t>,
      },
      [MAG_DTYPE_FLOAT8_E4M3FN] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<__nv_fp8_e4m3, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<__nv_fp8_e4m3, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<__nv_fp8_e4m3, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<__nv_fp8_e4m3, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<__nv_fp8_e4m3, uint8_t>,   // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<__nv_fp8_e4m3, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<__nv_fp8_e4m3, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<__nv_fp8_e4m3, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<__nv_fp8_e4m3, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<__nv_fp8_e4m3, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<__nv_fp8_e4m3, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<__nv_fp8_e4m3, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<__nv_fp8_e4m3, int64_t>,
      },
      [MAG_DTYPE_BOOLEAN] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint8_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint8_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint8_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<uint8_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint8_t, uint8_t>,     // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint8_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<uint8_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint8_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<uint8_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint8_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<uint8_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint8_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<uint8_t, int64_t>,
      },
      [MAG_DTYPE_UINT8] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint8_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint8_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint8_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<uint8_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint8_t, uint8_t>,     // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint8_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<uint8_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint8_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<uint8_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint8_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<uint8_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint8_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<uint8_t, int64_t>,
      },
      [MAG_DTYPE_INT8] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int8_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int8_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int8_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<int8_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int8_t, uint8_t>,      // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<int8_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<int8_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<int8_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<int8_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<int8_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<int8_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<int8_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<int8_t, int64_t>,
      },
      [MAG_DTYPE_UINT16] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint16_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint16_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint16_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<uint16_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint16_t, uint8_t>,    // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint16_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<uint16_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint16_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<uint16_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint16_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<uint16_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint16_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<uint16_t, int64_t>,
      },
      [MAG_DTYPE_INT16] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int16_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int16_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int16_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<int16_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int16_t, uint8_t>,     // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<int16_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<int16_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<int16_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<int16_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<int16_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<int16_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<int16_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<int16_t, int64_t>,
      },
      [MAG_DTYPE_UINT32] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint32_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint32_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint32_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<uint32_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint32_t, uint8_t>,    // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint32_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<uint32_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint32_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<uint32_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint32_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<uint32_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint32_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<uint32_t, int64_t>,
      },
      [MAG_DTYPE_INT32] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int32_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int32_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int32_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<int32_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int32_t, uint8_t>,     // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<int32_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<int32_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<int32_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<int32_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<int32_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<int32_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<int32_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<int32_t, int64_t>,
      },
      [MAG_DTYPE_UINT64] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint64_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint64_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint64_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<uint64_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint64_t, uint8_t>,    // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint64_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<uint64_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint64_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<uint64_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint64_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<uint64_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint64_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<uint64_t, int64_t>,
      },
      [MAG_DTYPE_INT64] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int64_t, float>,
        [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int64_t, half>,
        [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int64_t, __nv_bfloat16>,
        [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_cast_launcher<int64_t, __nv_fp8_e4m3>,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int64_t, uint8_t>,     // bool uses uint8_t kernels
        [MAG_DTYPE_UINT8] = &mag_cast_launcher<int64_t, uint8_t>,
        [MAG_DTYPE_INT8] = &mag_cast_launcher<int64_t, int8_t>,
        [MAG_DTYPE_UINT16] = &mag_cast_launcher<int64_t, uint16_t>,
        [MAG_DTYPE_INT16] = &mag_cast_launcher<int64_t, int16_t>,
        [MAG_DTYPE_UINT32] = &mag_cast_launcher<int64_t, uint32_t>,
        [MAG_DTYPE_INT32] = &mag_cast_launcher<int64_t, int32_t>,
        [MAG_DTYPE_UINT64] = &mag_cast_launcher<int64_t, uint64_t>,
        [MAG_DTYPE_INT64] = &mag_cast_launcher<int64_t, int64_t>,
      },
    };
    static_assert(std::size(cast_table_2D) == static_cast<size_t>(MAG_DTYPE__NUM));
    static_assert([]() -> bool {
      for (auto *fn : cast_table_2D) if (!fn) return false;
      return true;
    }());
    mag_dtype_t src = x->meta.dtype;
    mag_dtype_t dst = r->meta.dtype;
    cast_fn *kernel = cast_table_2D[src][dst];
    if (!kernel) return mag_set_error(err, MAG_ERR_KERNEL, "cuda: no kernel found for type cast: %s -> %s", mag_type_trait(src)->name, mag_type_trait(dst)->name);
    (*kernel)(r, x, stream);
    return MAG_OK;
  }

  template <typename T, typename I>
  __global__ static void clone_strided_kernel(I n, T *o, const T *x, coords_iter<I> rc, coords_iter<I> xc) {
    I i = static_cast<I>(blockIdx.x)*static_cast<I>(blockDim.x) + static_cast<I>(threadIdx.x);
    I step = static_cast<I>(blockDim.x)*static_cast<I>(gridDim.x);
    for (; i < n; i += step)
      o[rc(i)] = x[xc(i)];
  }

  template <typename T, typename I>
  static void launch_clone_indexed(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) { // TODO: Can be relaxed to non-shape
      cudaMemcpyAsync(pr, px, numel*sizeof(T), cudaMemcpyDeviceToDevice, stream);
      return;
    }
    auto blocks = static_cast<unsigned>(std::min((numel+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE, static_cast<int64_t>(std::numeric_limits<int>::max())));
    coords_iter<I> rc {r};
    coords_iter<I> xc {x};
    clone_strided_kernel<T, I><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(static_cast<I>(numel), pr, px, rc, xc);
  }

  template <typename T>
  static void launch_clone(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    if (mag_unlikely(numel <= 0)) return;
    if (mag_likely(can_all_use_i32_indexes(r, x))) { /* Counting elements is not enough: a view's strides can outrun its numel. */
      launch_clone_indexed<T, int32_t>(r, x, stream);
    } else {
      launch_clone_indexed<T, int64_t>(r, x, stream);
    }
  }

  mag_status_t unary_op_clone(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    mag_assert2(r->meta.dtype == x->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_clone<float>(r, x, stream); break;
      case MAG_DTYPE_FLOAT16: launch_clone<half>(r, x, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_clone<__nv_bfloat16>(r, x, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_clone<__nv_fp8_e4m3>(r, x, stream); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_clone<uint8_t>(r, x, stream); break;
      case MAG_DTYPE_INT8: launch_clone<int8_t>(r, x, stream); break;
      case MAG_DTYPE_UINT16: launch_clone<uint16_t>(r, x, stream); break;
      case MAG_DTYPE_INT16: launch_clone<int16_t>(r, x, stream); break;
      case MAG_DTYPE_UINT32: launch_clone<uint32_t>(r, x, stream); break;
      case MAG_DTYPE_INT32: launch_clone<int32_t>(r, x, stream); break;
      case MAG_DTYPE_UINT64: launch_clone<uint64_t>(r, x, stream); break;
      case MAG_DTYPE_INT64: launch_clone<int64_t>(r, x, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported dtype for unary op.");
    }
    return MAG_OK;
  }

  constexpr float INVSQRT2 = 0.707106781186547524400844362104849039284835937f /* 1/√2 */;

  template <typename T>
  struct op_abs {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(fabsf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_abs_int {
    static_assert(std::is_integral_v<T>);
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return x < T(0) ? static_cast<Out>(-x) : x;
    }
  };

  template <typename T>
  struct op_sgn {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      return xf32 > 0.f ? static_cast<Out>(1.f) : xf32 < 0.f ? static_cast<Out>(-1.f) : static_cast<Out>(0.f);
    }
  };

  template <typename T>
  struct op_neg {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(-static_cast<float>(x));
    }
  };

  template <typename T>
  struct op_not {
    static_assert(std::is_integral_v<T>);
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return ~x;
    }
  };

  template <typename T>
  struct op_log {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__logf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_log10 {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__log10f(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_log1p {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(log1pf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_log2 {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(log2f(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_sqr {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      return static_cast<Out>(xf32*xf32);
    }
  };

  template <typename T>
  struct op_rcp {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(1.f/static_cast<float>(x));
    }
  };

  template <typename T>
  struct op_sqrt {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(sqrtf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_rsqrt {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(1.f/sqrtf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_sin {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__sinf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_cos {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__cosf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_tan {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__tanf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_asin {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(asinf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_acos {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(acosf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_atan {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(atanf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_sinh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(sinhf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_cosh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(coshf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_tanh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(tanhf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_asinh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(asinhf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_acosh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(acoshf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_atanh {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(atanhf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_step {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<float>(x) > .0f ? static_cast<Out>(1.f) : static_cast<Out>(.0f);
    }
  };

  template <typename T>
  struct op_erf {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(erff(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_erfc {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(erfcf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_exp {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__expf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_exp2 {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(exp2f(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_expm1 {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(expm1f(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_floor {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(floorf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_ceil {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(ceilf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_round {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(roundf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_trunc {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(truncf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_softmax {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__expf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_softmax_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(__expf(static_cast<float>(x)));
    }
  };

  template <typename T>
  struct op_sigmoid {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      return static_cast<Out>(1.f/(1.f + __expf(-xf32)));
    }
  };

  template <typename T>
  struct op_sigmoid_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      float sig = 1.f/(1.f + __expf(-static_cast<float>(x)));
      return static_cast<Out>(sig*(1.f-sig));
    }
  };

  template <typename T>
  struct op_hard_sigmoid {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(fminf(1.f, fmaxf(.0f, (static_cast<float>(x) + 3.f)/6.f)));
    }
  };

  template <typename T>
  struct op_silu {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      return static_cast<Out>(xf32*(1.f/(1.f + __expf(-xf32))));
    }
  };

  template <typename T>
  struct op_silu_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      float sig = 1.f/(1.f + __expf(-xf32));
      return static_cast<Out>(sig + xf32*sig);
    }
  };

  template <typename T>
  struct op_tanh_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      float th = __tanhf(static_cast<float>(x));
      return static_cast<Out>(1.f - th*th);
    }
  };

  template <typename T>
  struct op_relu {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<Out>(fmaxf(static_cast<float>(x),0.f));
    }
  };

  template <typename T>
  struct op_relu_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      return static_cast<float>(x) > 0.f ? static_cast<Out>(1.f) : static_cast<Out>(0.f);
    }
  };

  template <typename T>
  struct op_gelu {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      return static_cast<Out>(.5f*xf32*(1.f+erff(xf32*INVSQRT2)));
    }
  };

  template <typename T>
  struct op_gelu_dv {
    using In = T;
    using Out = T;
    [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const {
      auto xf32 = static_cast<float>(x);
      float th = __tanhf(xf32);
      return static_cast<Out>(.5f*(1.f + th) + .5f*xf32*(1.f - th*th));
    }
  };

  template <typename Op, typename I, const bool Contig>
  __global__ static void unary_op_kernel(
    Op op,
    I n,
    typename Op::Out *r,
    const typename Op::In *x,
    [[maybe_unused]] coords_iter<I> rc,
    [[maybe_unused]] coords_iter<I> xc
  ) {
    I i = static_cast<I>(blockDim.x)*static_cast<I>(blockIdx.x) + static_cast<I>(threadIdx.x);
    if constexpr (Contig) {
      if (i >= n) return;
      r[i] = static_cast<typename Op::Out>(op(static_cast<typename Op::In>(x[i])));
    } else {
      I step = static_cast<I>(blockDim.x)*static_cast<I>(gridDim.x);
      for (; i < n; i += step)
        r[rc(i)] = op(x[rc.broadcast(xc, i)]);
    }
  }

  template <typename Op>
  static void launch_unary_op(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    auto blocks = static_cast<unsigned>(std::min((numel+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE, static_cast<int64_t>(std::numeric_limits<int>::max())));
    auto *pr = reinterpret_cast<typename Op::Out *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const typename Op::In *>(mag_tensor_data_ptr(x));
    if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      unary_op_kernel<Op, int64_t, true><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(Op{}, numel, pr, px, {}, {});
    } else {
      coords_iter<int64_t> rc {r};
      coords_iter<int64_t> xc {x};
      unary_op_kernel<Op, int64_t, false><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(Op{}, numel, pr, px, rc, xc);
    }
  }

  template <template <typename> typename Op>
  static mag_status_t impl_unary_op_fp(mag_error_t *err, mag_tensor_t *r, mag_tensor_t *x, cudaStream_t stream) {
    mag_assert2(r->meta.dtype == x->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_unary_op<Op<float>>(r, x, stream); break;
      case MAG_DTYPE_FLOAT16: launch_unary_op<Op<half>>(r, x, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_unary_op<Op<__nv_bfloat16>>(r, x, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_unary_op<Op<__nv_fp8_e4m3>>(r, x, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in unary operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <template <typename> typename Op>
  static mag_status_t impl_unary_op_int(mag_error_t *err, mag_tensor_t *r, mag_tensor_t *x, cudaStream_t stream) {
    mag_assert2(r->meta.dtype == x->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_unary_op<Op<uint8_t>>(r, x, stream); break;
      case MAG_DTYPE_INT8: launch_unary_op<Op<int8_t>>(r, x, stream); break;
      case MAG_DTYPE_UINT16: launch_unary_op<Op<uint16_t>>(r, x, stream); break;
      case MAG_DTYPE_INT16: launch_unary_op<Op<int16_t>>(r, x, stream); break;
      case MAG_DTYPE_UINT32: launch_unary_op<Op<uint32_t>>(r, x, stream); break;
      case MAG_DTYPE_INT32: launch_unary_op<Op<int32_t>>(r, x, stream); break;
      case MAG_DTYPE_UINT64: launch_unary_op<Op<uint64_t>>(r, x, stream); break;
      case MAG_DTYPE_INT64: launch_unary_op<Op<int64_t>>(r, x, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in unary operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t unary_op_abs(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    if (mag_type_category_is_integral(r->meta.dtype)) return impl_unary_op_int<op_abs_int>(err, r, x, stream);
    else return impl_unary_op_fp<op_abs>(err, r, x, stream);
  }
  mag_status_t unary_op_sgn(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sgn>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_neg(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_neg>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_not(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_int<op_not>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_log(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_log>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_log10(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_log10>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_log1p(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_log1p>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_log2(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_log2>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sqr(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sqr>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_rcp(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_rcp>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sqrt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sqrt>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_rsqrt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_rsqrt>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sin(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sin>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_cos(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_cos>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_tan(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_tan>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_asin(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_asin>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_acos(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_acos>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_atan(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_atan>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sinh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sinh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_cosh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_cosh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_tanh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_tanh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_asinh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_asinh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_acosh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_acosh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_atanh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_atanh>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_step(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_step>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_erf(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_erf>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_erfc(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_erfc>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_exp(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_exp>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_exp2(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_exp2>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_expm1(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_expm1>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_floor(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_floor>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_ceil(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_ceil>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_round(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_round>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_trunc(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_trunc>(err, cmd.out[0], cmd.in[0], stream); }

  template <typename T>
  __global__ static void softmax_kernel(int64_t rows, int64_t last_dim, T *__restrict__ r, const T *__restrict__ x) {
    int64_t row = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (row >= rows) return;
    const T *row_in = x + row * last_dim;
    T *row_out = r + row * last_dim;
    auto maxv = static_cast<float>(row_in[0]);
    for (int64_t i=1; i < last_dim; ++i) {
      auto v = static_cast<float>(row_in[i]);
      if (v > maxv) maxv = v;
    }
    double sum = 0.0;
    for (int64_t i=0; i < last_dim; ++i)
      sum += static_cast<double>(__expf(static_cast<float>(row_in[i]) - maxv));
    if (!std::isfinite(sum) || sum <= 0.0) {
      float inv = 1.0f / static_cast<float>(last_dim);
      for (int64_t i=0; i < last_dim; ++i)
        row_out[i] = static_cast<T>(inv);
    } else {
      auto inv = static_cast<float>(1.0 / sum);
      for (int64_t i=0; i < last_dim; ++i)
        row_out[i] = static_cast<T>(__expf(static_cast<float>(row_in[i]) - maxv)*inv);
    }
  }

  template <typename T>
  static void launch_softmax(mag_tensor_t *r, const mag_tensor_t *x, cudaStream_t stream) {
    int64_t rank = mag_tensor_rank(r);
    int64_t numel = mag_tensor_numel(r);
    if (mag_unlikely(numel <= 0)) return;
    int64_t last_dim = rank == 0 ? 1 : r->meta.coords.shape[rank-1];
    int64_t rows = rank == 0 ? 1 : numel/last_dim;
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    auto blocks = static_cast<unsigned>(std::min((numel+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE, static_cast<int64_t>(std::numeric_limits<int>::max())));
    softmax_kernel<T><<<blocks, UNARY_BLOCK_SIZE, 0, stream>>>(rows, last_dim, pr, px);
  }

  mag_status_t unary_op_softmax(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    mag_assert2(mag_tensor_is_contiguous(r));
    mag_assert2(mag_isok(mag_contiguous(nullptr, &x, x))); // Softmax requires contig x and r for now
    mag_assert2(r->meta.dtype == x->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_softmax<float>(r, x, stream); break;
      case MAG_DTYPE_FLOAT16: launch_softmax<half>(r, x, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_softmax<__nv_bfloat16>(r, x, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_softmax<__nv_fp8_e4m3>(r, x, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported dtype for softmax.");
    }
    mag_tensor_decref(x);
    return MAG_OK;
  }
  mag_status_t unary_op_softmax_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_softmax_dv>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sigmoid(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sigmoid>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_sigmoid_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_sigmoid_dv>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_hard_sigmoid(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_hard_sigmoid>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_silu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_silu>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_silu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_silu_dv>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_tanh_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_tanh_dv>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_relu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_relu>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_relu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_relu_dv>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_gelu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_gelu>(err, cmd.out[0], cmd.in[0], stream); }
  mag_status_t unary_op_gelu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_unary_op_fp<op_gelu_dv>(err, cmd.out[0], cmd.in[0], stream); }
}
