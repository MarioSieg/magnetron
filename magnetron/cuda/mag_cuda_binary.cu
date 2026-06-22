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
#include <cuda/std/tuple>

namespace mag {
  template <typename In, typename Out>
  struct op_add {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      return static_cast<OutT>(static_cast<float>(x)+static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_sub {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      return static_cast<OutT>(static_cast<float>(x)-static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_mul {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      return static_cast<OutT>(static_cast<float>(x)*static_cast<float>(y));
    }
  };

  template <typename In, typename Out>
  struct op_div {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const {
      return static_cast<OutT>(static_cast<float>(x)/static_cast<float>(y));
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
        return static_cast<OutT>(__powf(static_cast<float>(x), static_cast<float>(y)));
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
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)==static_cast<float>(y); }
  };

  template <typename In, typename Out>
  struct op_ne {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)!=static_cast<float>(y); }
  };

  template <typename In, typename Out>
  struct op_le {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)<=static_cast<float>(y); }
  };

  template <typename In, typename Out>
  struct op_ge {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)>=static_cast<float>(y); }
  };

  template <typename In, typename Out>
  struct op_lt {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)<static_cast<float>(y); }
  };

  template <typename In, typename Out>
  struct op_gt {
    using InT = In;
    using OutT = Out;
    [[nodiscard]] __device__ __forceinline__ OutT operator()(InT x, InT y) const { return static_cast<float>(x)>static_cast<float>(y); }
  };

  template <typename Op, const bool C>
  __global__ static void binary_op_kernel(
    Op op,
    int n,
    typename Op::OutT *r,
    const typename Op::InT *x,
    const typename Op::InT *y,
    [[maybe_unused]] mag_coords_iter_t rc,
    [[maybe_unused]] mag_coords_iter_t xc,
    [[maybe_unused]] mag_coords_iter_t yc
  ) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    if constexpr (C) {
      for (; i < n; i += step)
        r[i] = op(x[i], y[i]);
    } else {
      for (; i < n; i += step) {
        int ri = mag_coords_iter_to_offset(&rc, i);
        int xi = mag_coords_iter_broadcast(&rc, &xc, i);
        int yi = mag_coords_iter_broadcast(&rc, &yc, i);
        r[ri] = op(x[xi], y[yi]);
      }
    }
  }

  template <typename Op>
  static void launch_binary_op(mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y) {
    int n = numel_i32(r);
    int blocks = (n+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE;
    auto *pr = reinterpret_cast<typename Op::OutT *>(mag_tensor_data_ptr_mut(r));
    const auto *px = reinterpret_cast<const typename Op::InT *>(mag_tensor_data_ptr(x));
    const auto *py = reinterpret_cast<const typename Op::InT *>(mag_tensor_data_ptr(y));
    if (std::array<const mag_tensor_t *, 3> tensors {r, x, y}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
      binary_op_kernel<Op, true><<<blocks, BINARY_BLOCK_SIZE>>>(Op {}, n, pr, px, py, {}, {}, {});
    } else {
      mag_coords_iter_t rc, xc, yc;
      mag_coords_iter_init(&rc, &r->coords);
      mag_coords_iter_init(&xc, &x->coords);
      mag_coords_iter_init(&yc, &y->coords);
      binary_op_kernel<Op, false><<<blocks, BINARY_BLOCK_SIZE>>>(Op {}, n, pr, px, py, rc, xc, yc);
    }
  }

  template <template <typename, typename> typename Op>
  static void impl_binary_op_numeric(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_FLOAT32: launch_binary_op<Op<float, float>>(r, x, y); break;
      case MAG_DTYPE_FLOAT16: launch_binary_op<Op<half, half>>(r, x, y); break;
      case MAG_DTYPE_BFLOAT16: launch_binary_op<Op<__nv_bfloat16, __nv_bfloat16>>(r, x, y); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_binary_op<Op<__nv_fp8_e4m3, __nv_fp8_e4m3>>(r, x, y); break;
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, int8_t>>(r, x, y); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint16_t>>(r, x, y); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, int16_t>>(r, x, y); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint32_t>>(r, x, y); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint64_t>>(r, x, y); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, int64_t>>(r, x, y); break;
      default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
    }
  }

  template <template <typename, typename> typename Op>
  static void impl_binary_op_logical(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
    switch (r->dtype) {
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, int8_t>>(r, x, y); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint16_t>>(r, x, y); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, int16_t>>(r, x, y); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint32_t>>(r, x, y); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint64_t>>(r, x, y); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, int64_t>>(r, x, y); break;
      default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
    }
  }

  template <template <typename, typename> typename Op>
  static void impl_binary_op_cmp(const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *y = cmd.in[1];
    mag_assert2(r->dtype == MAG_DTYPE_BOOLEAN && x->dtype == y->dtype);
    switch (x->dtype) {
      case MAG_DTYPE_FLOAT32: launch_binary_op<Op<float, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_FLOAT16: launch_binary_op<Op<half, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_BFLOAT16: launch_binary_op<Op<__nv_bfloat16, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_FLOAT8_E4M3FN: launch_binary_op<Op<__nv_fp8_e4m3, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_BOOLEAN:
      case MAG_DTYPE_UINT8: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT8: launch_binary_op<Op<int8_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_UINT16: launch_binary_op<Op<uint16_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT16: launch_binary_op<Op<int16_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_UINT32: launch_binary_op<Op<uint32_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT32: launch_binary_op<Op<int32_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_UINT64: launch_binary_op<Op<uint64_t, uint8_t>>(r, x, y); break;
      case MAG_DTYPE_INT64: launch_binary_op<Op<int64_t, uint8_t>>(r, x, y); break;
      default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
    }
  }

  void binary_op_add(const mag_command_t &cmd) { impl_binary_op_numeric<op_add>(cmd); }
  void binary_op_sub(const mag_command_t &cmd) { impl_binary_op_numeric<op_sub>(cmd); }
  void binary_op_mul(const mag_command_t &cmd) { impl_binary_op_numeric<op_mul>(cmd); }
  void binary_op_div(const mag_command_t &cmd) { impl_binary_op_numeric<op_div>(cmd); }
  void binary_op_floordiv(const mag_command_t &cmd) { impl_binary_op_numeric<op_floordiv>(cmd); }
  void binary_op_mod(const mag_command_t &cmd) { impl_binary_op_numeric<op_mod>(cmd); }
  void binary_op_pow(const mag_command_t &cmd) { impl_binary_op_numeric<op_pow>(cmd); }
  void binary_op_and(const mag_command_t &cmd) { impl_binary_op_logical<op_and>(cmd); }
  void binary_op_or(const mag_command_t &cmd)  { impl_binary_op_logical<op_or>(cmd); }
  void binary_op_xor(const mag_command_t &cmd) { impl_binary_op_logical<op_xor>(cmd); }
  void binary_op_shl(const mag_command_t &cmd) { impl_binary_op_logical<op_shl>(cmd); }
  void binary_op_shr(const mag_command_t &cmd) { impl_binary_op_logical<op_shr>(cmd); }
  void binary_op_eq(const mag_command_t &cmd) { impl_binary_op_cmp<op_eq>(cmd); }
  void binary_op_ne(const mag_command_t &cmd) { impl_binary_op_cmp<op_ne>(cmd); }
  void binary_op_le(const mag_command_t &cmd) { impl_binary_op_cmp<op_le>(cmd); }
  void binary_op_ge(const mag_command_t &cmd) { impl_binary_op_cmp<op_ge>(cmd); }
  void binary_op_lt(const mag_command_t &cmd) { impl_binary_op_cmp<op_lt>(cmd); }
  void binary_op_gt(const mag_command_t &cmd) { impl_binary_op_cmp<op_gt>(cmd); }
}
