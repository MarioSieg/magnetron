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
  template <typename Src, typename Dst, const bool C>
  __global__ static void cast_kernel(int n, Dst *__restrict__ o, const Src *__restrict__ x, [[maybe_unused]] mag_coords_iter_t xc) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if constexpr (C) {
      if (i >= n) return;
      o[i] = static_cast<Dst>(x[i]);
    } else {
      int step = blockDim.x*gridDim.x;
      for (; i < n; i += step)
        o[i] = static_cast<Dst>(x[mag_coords_iter_to_offset(&xc, i)]);
    }
  }

  template <typename Src, typename Dst>
  static void mag_cast_launcher(mag_tensor_t *r, const mag_tensor_t *x) {
    int n = numel_i32(r);
    auto *pr = reinterpret_cast<Dst *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const Src *>(mag_tensor_data_ptr(x));
    int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    if (mag_tensor_is_contiguous(x)) {
      cast_kernel<Src, Dst, true><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, {});
    } else {
      mag_coords_iter_t xc;
      mag_coords_iter_init(&xc, &x->coords);
      cast_kernel<Src, Dst, false><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, xc);
    }
  }

  void unary_op_cast(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    using cast_fn = void (mag_tensor_t *, const mag_tensor_t *);
    static constexpr void (*const cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(mag_tensor_t *, const mag_tensor_t *) = {
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
    mag_dtype_t src = x->dtype;
    mag_dtype_t dst = r->dtype;
    cast_fn *kernel = cast_table_2D[src][dst];
    mag_assert(kernel, "No kernel found for type cast: %s -> %s", mag_type_trait(src)->name, mag_type_trait(dst)->name);
    (*kernel)(r, x);
  }

  template <typename T>
  __global__ static void clone_strided_kernel(int n, T *o, const T *x, mag_coords_iter_t rc, mag_coords_iter_t xc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    for (; i < n; i += step) {
      int ri = mag_coords_iter_to_offset(&rc, i);
      int xi = mag_coords_iter_to_offset(&xc, i);
      o[ri] = x[xi];
    }
  }

  template <typename T>
  static void launch_clone(mag_tensor_t *r, const mag_tensor_t *x) {
    int n = numel_i32(r);
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) { // TODO: Can be relaxed to non-shape
      cudaMemcpy(pr, px, n*sizeof(T), cudaMemcpyDeviceToDevice);
      return;
    }
    mag_coords_iter_t rc, xc;
    mag_coords_iter_init(&rc, &r->coords);
    mag_coords_iter_init(&xc, &x->coords);
    int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    clone_strided_kernel<T><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, rc, xc);
  }

  void unary_op_clone(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    mag_assert2(r->dtype == x->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_clone<float>(r, x); break;
      case MAG_DTYPE_FLOAT16: launch_clone<half>(r, x); break;
      case MAG_DTYPE_BFLOAT16: launch_clone<__nv_bfloat16>(r, x); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_clone<__nv_fp8_e4m3>(r, x); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_clone<uint8_t>(r, x); break;
      case MAG_DTYPE_INT8: launch_clone<int8_t>(r, x); break;
      case MAG_DTYPE_UINT16: launch_clone<uint16_t>(r, x); break;
      case MAG_DTYPE_INT16: launch_clone<int16_t>(r, x); break;
      case MAG_DTYPE_UINT32: launch_clone<uint32_t>(r, x); break;
      case MAG_DTYPE_INT32: launch_clone<int32_t>(r, x); break;
      case MAG_DTYPE_UINT64: launch_clone<uint64_t>(r, x); break;
      case MAG_DTYPE_INT64: launch_clone<int64_t>(r, x); break;
      default: mag_assert(false, "Unsupported dtype for unary op");
    }
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

  template <typename Op, const bool C>
  __global__ static void unary_op_kernel(
    Op op,
    int n,
    typename Op::Out *r,
    const typename Op::In *x,
    [[maybe_unused]] mag_coords_iter_t rc,
    [[maybe_unused]] mag_coords_iter_t xc
  ) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if constexpr (C) {
      if (i >= n) return;
      r[i] = static_cast<typename Op::Out>(op(static_cast<typename Op::In>(x[i])));
    } else {
      int step = blockDim.x*gridDim.x;
      for (; i < n; i += step) {
        int ri = mag_coords_iter_to_offset(&rc, i);
        int xi = mag_coords_iter_broadcast(&rc, &xc, i);
        r[ri] = op(x[xi]);
      }
    }
  }

  template <typename Op>
  static void launch_unary_op(mag_tensor_t *r, const mag_tensor_t *x) {
    int n = numel_i32(r);
    int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    mag_coords_iter_t rc, xc;
    mag_coords_iter_init(&rc, &r->coords);
    mag_coords_iter_init(&xc, &x->coords);
    auto *pr = reinterpret_cast<typename Op::Out *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const typename Op::In *>(mag_tensor_data_ptr(x));
    if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      unary_op_kernel<Op, true><<<blocks, UNARY_BLOCK_SIZE>>>(Op{}, n, pr, px, rc, xc);
    } else {
      unary_op_kernel<Op, false><<<blocks, UNARY_BLOCK_SIZE>>>(Op{}, n, pr, px, rc, xc);
    }
  }

  template <template <typename> typename Op>
  static void impl_unary_op_fp(mag_tensor_t *r, mag_tensor_t *x) {
    mag_assert2(r->dtype == x->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_unary_op<Op<float>>(r, x); break;
      case MAG_DTYPE_FLOAT16: launch_unary_op<Op<half>>(r, x); break;
      case MAG_DTYPE_BFLOAT16: launch_unary_op<Op<__nv_bfloat16>>(r, x); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_unary_op<Op<__nv_fp8_e4m3>>(r, x); break;
      default: mag_assert(false, "Unsupported data type in unary operation: %s", mag_type_trait(r->dtype)->name);
    }
  }

  template <template <typename> typename Op>
  static void impl_unary_op_int(mag_tensor_t *r, mag_tensor_t *x) {
    mag_assert2(r->dtype == x->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_unary_op<Op<uint8_t>>(r, x); break;
      case MAG_DTYPE_INT8: launch_unary_op<Op<int8_t>>(r, x); break;
      case MAG_DTYPE_UINT16: launch_unary_op<Op<uint16_t>>(r, x); break;
      case MAG_DTYPE_INT16: launch_unary_op<Op<int16_t>>(r, x); break;
      case MAG_DTYPE_UINT32: launch_unary_op<Op<uint32_t>>(r, x); break;
      case MAG_DTYPE_INT32: launch_unary_op<Op<int32_t>>(r, x); break;
      case MAG_DTYPE_UINT64: launch_unary_op<Op<uint64_t>>(r, x); break;
      case MAG_DTYPE_INT64: launch_unary_op<Op<int64_t>>(r, x); break;
      default: mag_assert(false, "Unsupported data type in unary operation: %s", mag_type_trait(r->dtype)->name);
    }
  }

  void unary_op_abs(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    if (mag_type_category_is_integral(r->dtype)) impl_unary_op_int<op_abs_int>(r, x);
    else impl_unary_op_fp<op_abs>(r, x);
  }
  void unary_op_sgn(const mag_command_t &cmd) { impl_unary_op_fp<op_sgn>(cmd.out[0], cmd.in[0]); }
  void unary_op_neg(const mag_command_t &cmd) { impl_unary_op_fp<op_neg>(cmd.out[0], cmd.in[0]); }
  void unary_op_not(const mag_command_t &cmd) { impl_unary_op_int<op_not>(cmd.out[0], cmd.in[0]); }
  void unary_op_log(const mag_command_t &cmd) { impl_unary_op_fp<op_log>(cmd.out[0], cmd.in[0]); }
  void unary_op_log10(const mag_command_t &cmd) { impl_unary_op_fp<op_log10>(cmd.out[0], cmd.in[0]); }
  void unary_op_log1p(const mag_command_t &cmd) { impl_unary_op_fp<op_log1p>(cmd.out[0], cmd.in[0]); }
  void unary_op_log2(const mag_command_t &cmd) { impl_unary_op_fp<op_log2>(cmd.out[0], cmd.in[0]); }
  void unary_op_sqr(const mag_command_t &cmd) { impl_unary_op_fp<op_sqr>(cmd.out[0], cmd.in[0]); }
  void unary_op_rcp(const mag_command_t &cmd) { impl_unary_op_fp<op_rcp>(cmd.out[0], cmd.in[0]); }
  void unary_op_sqrt(const mag_command_t &cmd) { impl_unary_op_fp<op_sqrt>(cmd.out[0], cmd.in[0]); }
  void unary_op_rsqrt(const mag_command_t &cmd) { impl_unary_op_fp<op_rsqrt>(cmd.out[0], cmd.in[0]); }
  void unary_op_sin(const mag_command_t &cmd) { impl_unary_op_fp<op_sin>(cmd.out[0], cmd.in[0]); }
  void unary_op_cos(const mag_command_t &cmd) { impl_unary_op_fp<op_cos>(cmd.out[0], cmd.in[0]); }
  void unary_op_tan(const mag_command_t &cmd) { impl_unary_op_fp<op_tan>(cmd.out[0], cmd.in[0]); }
  void unary_op_asin(const mag_command_t &cmd) { impl_unary_op_fp<op_asin>(cmd.out[0], cmd.in[0]); }
  void unary_op_acos(const mag_command_t &cmd) { impl_unary_op_fp<op_acos>(cmd.out[0], cmd.in[0]); }
  void unary_op_atan(const mag_command_t &cmd) { impl_unary_op_fp<op_atan>(cmd.out[0], cmd.in[0]); }
  void unary_op_sinh(const mag_command_t &cmd) { impl_unary_op_fp<op_sinh>(cmd.out[0], cmd.in[0]); }
  void unary_op_cosh(const mag_command_t &cmd) { impl_unary_op_fp<op_cosh>(cmd.out[0], cmd.in[0]); }
  void unary_op_tanh(const mag_command_t &cmd) { impl_unary_op_fp<op_tanh>(cmd.out[0], cmd.in[0]); }
  void unary_op_asinh(const mag_command_t &cmd) { impl_unary_op_fp<op_asinh>(cmd.out[0], cmd.in[0]); }
  void unary_op_acosh(const mag_command_t &cmd) { impl_unary_op_fp<op_acosh>(cmd.out[0], cmd.in[0]); }
  void unary_op_atanh(const mag_command_t &cmd) { impl_unary_op_fp<op_atanh>(cmd.out[0], cmd.in[0]); }
  void unary_op_step(const mag_command_t &cmd) { impl_unary_op_fp<op_step>(cmd.out[0], cmd.in[0]); }
  void unary_op_erf(const mag_command_t &cmd) { impl_unary_op_fp<op_erf>(cmd.out[0], cmd.in[0]); }
  void unary_op_erfc(const mag_command_t &cmd) { impl_unary_op_fp<op_erfc>(cmd.out[0], cmd.in[0]); }
  void unary_op_exp(const mag_command_t &cmd) { impl_unary_op_fp<op_exp>(cmd.out[0], cmd.in[0]); }
  void unary_op_exp2(const mag_command_t &cmd) { impl_unary_op_fp<op_exp2>(cmd.out[0], cmd.in[0]); }
  void unary_op_expm1(const mag_command_t &cmd) { impl_unary_op_fp<op_expm1>(cmd.out[0], cmd.in[0]); }
  void unary_op_floor(const mag_command_t &cmd) { impl_unary_op_fp<op_floor>(cmd.out[0], cmd.in[0]); }
  void unary_op_ceil(const mag_command_t &cmd) { impl_unary_op_fp<op_ceil>(cmd.out[0], cmd.in[0]); }
  void unary_op_round(const mag_command_t &cmd) { impl_unary_op_fp<op_round>(cmd.out[0], cmd.in[0]); }
  void unary_op_trunc(const mag_command_t &cmd) { impl_unary_op_fp<op_trunc>(cmd.out[0], cmd.in[0]); }

  template <typename T>
  __global__ static void softmax_kernel(int rows, int last_dim, T *__restrict__ r, const T *__restrict__ x) {
    int64_t row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T *row_in = x + row * last_dim;
    T *row_out = r + row * last_dim;
    auto maxv = static_cast<float>(row_in[0]);
    for (int i=1; i < last_dim; ++i) {
      auto v = static_cast<float>(row_in[i]);
      if (v > maxv) maxv = v;
    }
    double sum = 0.0;
    for (int i=0; i < last_dim; ++i)
      sum += static_cast<double>(__expf(static_cast<float>(row_in[i]) - maxv));
    if (!std::isfinite(sum) || sum <= 0.0) {
      float inv = 1.0f / static_cast<float>(last_dim);
      for (int i=0; i < last_dim; ++i)
        row_out[i] = static_cast<T>(inv);
    } else {
      auto inv = static_cast<float>(1.0 / sum);
      for (int i=0; i < last_dim; ++i)
        row_out[i] = static_cast<T>(__expf(static_cast<float>(row_in[i]) - maxv)*inv);
    }
  }

  template <typename T>
  static void launch_softmax(mag_tensor_t *r, const mag_tensor_t *x) {
    int rank = static_cast<int>(r->coords.rank);
    int n = numel_i32(r);
    if (mag_unlikely(!n)) return;
    int last_dim = rank == 0 ? 1 : static_cast<int>(r->coords.shape[rank-1]);
    int rows = rank == 0 ? 1 : n/last_dim;
    auto *pr = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    int blocks = (rows+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    softmax_kernel<T><<<blocks, UNARY_BLOCK_SIZE>>>(rows, last_dim, pr, px);
  }

  void unary_op_softmax(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    mag_assert2(mag_tensor_is_contiguous(r));
    mag_assert2(mag_isok(mag_contiguous(nullptr, &x, x))); // Softmax requires contig x and r for now
    mag_assert2(r->dtype == x->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_softmax<float>(r, x); break;
      case MAG_DTYPE_FLOAT16: launch_softmax<half>(r, x); break;
      case MAG_DTYPE_BFLOAT16: launch_softmax<__nv_bfloat16>(r, x); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_softmax<__nv_fp8_e4m3>(r, x); break;
      default: mag_assert(false, "Unsupported dtype for softmax");
    }
    mag_tensor_decref(x);
  }
  void unary_op_softmax_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_softmax_dv>(cmd.out[0], cmd.in[0]); }
  void unary_op_sigmoid(const mag_command_t &cmd) { impl_unary_op_fp<op_sigmoid>(cmd.out[0], cmd.in[0]); }
  void unary_op_sigmoid_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_sigmoid_dv>(cmd.out[0], cmd.in[0]); }
  void unary_op_hard_sigmoid(const mag_command_t &cmd) { impl_unary_op_fp<op_hard_sigmoid>(cmd.out[0], cmd.in[0]); }
  void unary_op_silu(const mag_command_t &cmd) { impl_unary_op_fp<op_silu>(cmd.out[0], cmd.in[0]); }
  void unary_op_silu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_silu_dv>(cmd.out[0], cmd.in[0]); }
  void unary_op_tanh_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_tanh_dv>(cmd.out[0], cmd.in[0]); }
  void unary_op_relu(const mag_command_t &cmd) { impl_unary_op_fp<op_relu>(cmd.out[0], cmd.in[0]); }
  void unary_op_relu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_relu_dv>(cmd.out[0], cmd.in[0]); }
  void unary_op_gelu(const mag_command_t &cmd) { impl_unary_op_fp<op_gelu>(cmd.out[0], cmd.in[0]); }
  void unary_op_gelu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_gelu_dv>(cmd.out[0], cmd.in[0]); }
}
