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

#include "mag_cuda_binary.cuh"

#include <array>
#include <cmath>
#include <cuda/std/tuple>

namespace mag {
  template <typename In, typename Out>
  struct op_add {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return static_cast<OutT>(x + y);
      return static_cast<OutT>(static_cast<float>(x) + static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_sub {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return static_cast<OutT>(x - y);
      return static_cast<OutT>(static_cast<float>(x) - static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_mul {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return static_cast<OutT>(x * y);
      return static_cast<OutT>(static_cast<float>(x) * static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_div {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return static_cast<OutT>(x / y);
      return static_cast<OutT>(static_cast<float>(x) / static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_floordiv {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) {
        if constexpr (std::is_unsigned_v<InT>) {
          return x/y;
        } else {
          return (x - mag_remi(static_cast<int64_t>(x), static_cast<int64_t>(y)))/y;
        }
      } else {
        return static_cast<OutT>(floorf(static_cast<float>(x)/static_cast<float>(y)));
      }
    }
  };

  template <typename In, typename Out>
  struct op_mod {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) {
        if constexpr (std::is_unsigned_v<InT>) {
          return x % y;
        } else {
          int64_t r = x % y;
          if (r != 0 && (r < 0) != (y < 0)) r += y;
          return static_cast<OutT>(r);
        }
      } else {
        auto xf32 = static_cast<float>(x);
        auto yf32 = static_cast<float>(y);
        float r = fmodf(xf32, yf32);
        if (r != 0.0f && (r < 0.0f) != (yf32 < 0.0f)) r += yf32;
        return static_cast<OutT>(r);
      }
    }
  };

  template <typename In, typename Out>
  struct op_pow {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) {
        if constexpr (std::is_unsigned_v<InT>) {
          return static_cast<OutT>(mag_powu(static_cast<uint64_t>(x), static_cast<uint64_t>(y)));
        } else {
          return static_cast<OutT>(mag_powi(static_cast<int64_t>(x), static_cast<int64_t>(y)));
        }
      } else {
        return static_cast<OutT>(pow(static_cast<double>(x), static_cast<double>(y)));
      }
    }
  };

  template <typename In, typename Out>
  struct op_and {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return x&y; }
  };

  template <typename In, typename Out>
  struct op_or {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return x|y; }
  };

  template <typename In, typename Out>
  struct op_xor {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return x^y; }
  };

  template <typename In, typename Out>
  struct op_shl {
    using InT = In;
    using OutT = Out;
    static constexpr InT bits = sizeof(InT)*8;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      return mag_unlikely(y < 0 || y >= bits) ? 0 : x<<y;
    }
  };

  template <typename In, typename Out>
  struct op_shr {
    using InT = In;
    using OutT = Out;
    static constexpr InT bits = sizeof(InT)*8;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if (mag_unlikely(y < 0 || y >= bits)) {
        if constexpr (std::is_signed_v<InT>) return mag_unlikely(x < 0) ? -1 : 0; // SAR
        else return 0; // SHR
      }
      return x>>y;
    }
  };

  template <typename In, typename Out>
  struct op_eq {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x == y;
      return static_cast<float>(x) == static_cast<float>(y);
    }
  };

  template <typename In, typename Out>
  struct op_ne {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x != y;
      return static_cast<float>(x) != static_cast<float>(y);
    }
  };

  template <typename In, typename Out>
  struct op_le {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x <= y;
      return static_cast<float>(x) <= static_cast<float>(y);
    }
  };

  template <typename In, typename Out>
  struct op_ge {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x >= y;
      return static_cast<float>(x) >= static_cast<float>(y);
    }
  };

  template <typename In, typename Out>
  struct op_lt {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x < y;
      return static_cast<float>(x) < static_cast<float>(y);
    }
  };

  template <typename In, typename Out>
  struct op_gt {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      if constexpr (std::is_integral_v<InT>) return x > y;
      return static_cast<float>(x) > static_cast<float>(y);
    }
  };

  template <typename Op, const bool Contig>
  __global__ static void binary_op_kernel(
    Op op,
    int64_t numel,
    typename Op::OutT *r,
    const typename Op::InT *x,
    const typename Op::InT *y,
    [[maybe_unused]] coords_iter<int64_t> rc,
    [[maybe_unused]] coords_iter<int64_t> xc,
    [[maybe_unused]] coords_iter<int64_t> yc
  ) {
    int64_t i = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(threadIdx.x);
    int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
    if constexpr (Contig) {
      for (; i < numel; i += step)
        r[i] = op(x[i], y[i]);
    } else {
      for (; i < numel; i += step)
        r[rc(i)] = op(x[rc.broadcast(xc, i)], y[rc.broadcast(yc, i)]);
    }
  }

  template <typename Op>
  static void launch_binary_op(mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y, cudaStream_t stream) {
    int64_t numel = mag_tensor_numel(r);
    auto blocks = static_cast<unsigned>(std::min((numel+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE, static_cast<int64_t>(std::numeric_limits<int>::max())));
    auto *pr = reinterpret_cast<typename Op::OutT *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const typename Op::InT *>(mag_tensor_data_ptr(x));
    const auto *py = reinterpret_cast<const typename Op::InT *>(mag_tensor_data_ptr(y));
    if (std::array<const mag_tensor_t *, 3> tensors {r, x, y}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      binary_op_kernel<Op, true><<<blocks, BINARY_BLOCK_SIZE, 0, stream>>>(Op {}, numel, pr, px, py, {}, {}, {});
    } else {
      coords_iter<int64_t> rc {r};
      coords_iter<int64_t> xc {x};
      coords_iter<int64_t> yc {y};
      binary_op_kernel<Op, false><<<blocks, BINARY_BLOCK_SIZE, 0, stream>>>(Op {}, numel, pr, px, py, rc, xc, yc);
    }
  }

  template <template <typename, typename> typename Op>
  static mag_status_t impl_binary_op_numeric(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->meta.dtype == x->meta.dtype && r->meta.dtype == y->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_binary_op<Op<float, float>>(r, x, y, stream); break;
      case MAG_DTYPE_FLOAT16: launch_binary_op<Op<half, half>>(r, x, y, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_binary_op<Op<__nv_bfloat16, __nv_bfloat16>>(r, x, y, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_binary_op<Op<__nv_fp8_e4m3, __nv_fp8_e4m3>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, int8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint16_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, int16_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint32_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint64_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, int64_t>>(r, x, y, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in binary operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <template <typename, typename> typename Op>
  static mag_status_t impl_binary_op_logical(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->meta.dtype == x->meta.dtype && r->meta.dtype == y->meta.dtype);
    switch (r->meta.dtype) {
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, int8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint16_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, int16_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint32_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint64_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, int64_t>>(r, x, y, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in binary operation: %s", mag_type_trait(r->meta.dtype)->name);
    }
    return MAG_OK;
  }

  template <template <typename, typename> typename Op>
  static mag_status_t impl_binary_op_cmp(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->meta.dtype == MAG_DTYPE_BOOLEAN && x->meta.dtype == y->meta.dtype);
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: launch_binary_op<Op<float, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_FLOAT16: launch_binary_op<Op<half, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_BFLOAT16: launch_binary_op<Op<__nv_bfloat16, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_binary_op<Op<__nv_fp8_e4m3, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint8_t>>(r, x, y, stream); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, uint8_t>>(r, x, y, stream); break;
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: unsupported data type in binary operation: %s", mag_type_trait(x->meta.dtype)->name);
    }
    return MAG_OK;
  }

  mag_status_t binary_op_add(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_add>(err, cmd, stream); }
  mag_status_t binary_op_sub(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_sub>(err, cmd, stream); }
  mag_status_t binary_op_mul(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_mul>(err, cmd, stream); }
  mag_status_t binary_op_div(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_div>(err, cmd, stream); }
  mag_status_t binary_op_floordiv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_floordiv>(err, cmd, stream); }
  mag_status_t binary_op_mod(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_mod>(err, cmd, stream); }
  mag_status_t binary_op_pow(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_numeric<op_pow>(err, cmd, stream); }
  mag_status_t binary_op_and(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_logical<op_and>(err, cmd, stream); }
  mag_status_t binary_op_or(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream)  { return impl_binary_op_logical<op_or>(err, cmd, stream); }
  mag_status_t binary_op_xor(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_logical<op_xor>(err, cmd, stream); }
  mag_status_t binary_op_shl(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_logical<op_shl>(err, cmd, stream); }
  mag_status_t binary_op_shr(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_logical<op_shr>(err, cmd, stream); }
  mag_status_t binary_op_eq(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_eq>(err, cmd, stream); }
  mag_status_t binary_op_ne(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_ne>(err, cmd, stream); }
  mag_status_t binary_op_le(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_le>(err, cmd, stream); }
  mag_status_t binary_op_ge(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_ge>(err, cmd, stream); }
  mag_status_t binary_op_lt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_lt>(err, cmd, stream); }
  mag_status_t binary_op_gt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) { return impl_binary_op_cmp<op_gt>(err, cmd, stream); }
}
