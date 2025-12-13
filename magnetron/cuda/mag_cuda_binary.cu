/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_cuda_binary.cuh"

#include <cuda/std/tuple>

namespace mag {
    template <typename scalar_in_t, typename scalar_out_t>
    struct op_add {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x+y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_sub {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x-y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_mul {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x*y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_div {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x/y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_mod {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const {
            if constexpr (std::is_integral_v<in_t>) return x%y;
            else return fmodf(x, y);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_and {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x&y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_or {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x|y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_xor {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x^y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_shl {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        static constexpr scalar_in_t mask = (8*sizeof(scalar_in_t))-1;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x<<(y&mask); }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_shr {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        static constexpr scalar_in_t mask = (8*sizeof(scalar_in_t))-1;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x>>(y&mask); }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_eq {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x==y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_ne {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x!=y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_le {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x<=y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_ge {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x>=y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_lt {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x<y; }
    };

    template <typename scalar_in_t, typename scalar_out_t>
    struct op_gt {
        using in_t = scalar_in_t;
        using out_t = scalar_out_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x, in_t y) const { return x>y; }
    };

    template <typename op_t, const bool contig>
    __global__ static void binary_op_kernel(
        op_t op,
        int64_t n,
        typename op_t::out_t *o,
        const typename op_t::in_t *x,
        const typename op_t::in_t *y,
        [[maybe_unused]] mag_coords_iter_t rc,
        [[maybe_unused]] mag_coords_iter_t xc,
        [[maybe_unused]] mag_coords_iter_t yc
    ) {
        int64_t i = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        if constexpr (contig) {
            for (; i < n; i += step)
                o[i] = op(x[i], y[i]);
        } else {
            for (; i < n; i += step) {
                int64_t ri = mag_coords_iter_to_offset(&rc, i);
                int64_t xi = mag_coords_iter_broadcast(&rc, &xc, i);
                int64_t yi = mag_coords_iter_broadcast(&rc, &yc, i);
                o[ri] = op(x[xi], y[yi]);
            }
        }
    }

    template <typename op_t>
    static void launch_binary_op(
        mag_tensor_t *r,
        const mag_tensor_t *x,
        const mag_tensor_t *y
    ) {
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE;
        mag_coords_iter_t rc, xc, yc;
        mag_coords_iter_init(&rc, &r->coords);
        mag_coords_iter_init(&xc, &x->coords);
        mag_coords_iter_init(&yc, &y->coords);
        auto *pr = reinterpret_cast<typename op_t::out_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const typename op_t::in_t *>(mag_tensor_data_ptr(x));
        const auto *py = reinterpret_cast<const typename op_t::in_t *>(mag_tensor_data_ptr(y));
        if (mag_full_cont3(r, x, y)) binary_op_kernel<op_t, true><<<blocks, BINARY_BLOCK_SIZE>>>(op_t {}, n, pr, px, py, rc, xc, yc);
        else binary_op_kernel<op_t, false><<<blocks, BINARY_BLOCK_SIZE>>>(op_t {}, n, pr, px, py, rc, xc, yc);
    }

    template <template <typename, typename> typename op_t>
    static void impl_binary_op_numeric(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_binary_op<op_t<float, float>>(r, x, y); break;
            case MAG_DTYPE_FLOAT16: launch_binary_op<op_t<half, half>>(r, x, y); break;
            case MAG_DTYPE_UINT8: launch_binary_op<op_t<uint8_t, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_INT8: launch_binary_op<op_t<int8_t, int8_t>>(r, x, y); break;
            case MAG_DTYPE_UINT16: launch_binary_op<op_t<uint16_t, uint16_t>>(r, x, y); break;
            case MAG_DTYPE_INT16: launch_binary_op<op_t<int16_t, int16_t>>(r, x, y); break;
            case MAG_DTYPE_UINT32: launch_binary_op<op_t<uint32_t, uint32_t>>(r, x, y); break;
            case MAG_DTYPE_INT32: launch_binary_op<op_t<int32_t, int32_t>>(r, x, y); break;
            case MAG_DTYPE_UINT64: launch_binary_op<op_t<uint64_t, uint64_t>>(r, x, y); break;
            case MAG_DTYPE_INT64: launch_binary_op<op_t<int64_t, int64_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
        }
    }

    template <template <typename, typename> typename op_t>
    static void impl_binary_op_logical(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_binary_op<op_t<uint8_t, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_INT8: launch_binary_op<op_t<int8_t, int8_t>>(r, x, y); break;
            case MAG_DTYPE_UINT16: launch_binary_op<op_t<uint16_t, uint16_t>>(r, x, y); break;
            case MAG_DTYPE_INT16: launch_binary_op<op_t<int16_t, int16_t>>(r, x, y); break;
            case MAG_DTYPE_UINT32: launch_binary_op<op_t<uint32_t, uint32_t>>(r, x, y); break;
            case MAG_DTYPE_INT32: launch_binary_op<op_t<int32_t, int32_t>>(r, x, y); break;
            case MAG_DTYPE_UINT64: launch_binary_op<op_t<uint64_t, uint64_t>>(r, x, y); break;
            case MAG_DTYPE_INT64: launch_binary_op<op_t<int64_t, int64_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
        }
    }

    template <template <typename, typename> typename op_t>
    static void impl_binary_op_cmp(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == MAG_DTYPE_BOOLEAN && x->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_binary_op<op_t<float, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_FLOAT16: launch_binary_op<op_t<half, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_binary_op<op_t<uint8_t, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_INT8: launch_binary_op<op_t<int8_t, int8_t>>(r, x, y); break;
            case MAG_DTYPE_UINT16: launch_binary_op<op_t<uint16_t, uint16_t>>(r, x, y); break;
            case MAG_DTYPE_INT16: launch_binary_op<op_t<int16_t, int16_t>>(r, x, y); break;
            case MAG_DTYPE_UINT32: launch_binary_op<op_t<uint32_t, uint32_t>>(r, x, y); break;
            case MAG_DTYPE_INT32: launch_binary_op<op_t<int32_t, int32_t>>(r, x, y); break;
            case MAG_DTYPE_UINT64: launch_binary_op<op_t<uint64_t, uint64_t>>(r, x, y); break;
            case MAG_DTYPE_INT64: launch_binary_op<op_t<int64_t, int64_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_type_trait(r->dtype)->name);
        }
    }

    void binary_op_add(const mag_command_t *cmd) { impl_binary_op_numeric<op_add>(cmd); }
    void binary_op_sub(const mag_command_t *cmd) { impl_binary_op_numeric<op_sub>(cmd); }
    void binary_op_mul(const mag_command_t *cmd) { impl_binary_op_numeric<op_mul>(cmd); }
    void binary_op_div(const mag_command_t *cmd) { impl_binary_op_numeric<op_div>(cmd); }
    void binary_op_mod(const mag_command_t *cmd) { impl_binary_op_numeric<op_mod>(cmd); }
    void binary_op_and(const mag_command_t *cmd) { impl_binary_op_logical<op_and>(cmd); }
    void binary_op_or(const mag_command_t *cmd)  { impl_binary_op_logical<op_or>(cmd); }
    void binary_op_xor(const mag_command_t *cmd) { impl_binary_op_logical<op_xor>(cmd); }
    void binary_op_shl(const mag_command_t *cmd) { impl_binary_op_logical<op_shl>(cmd); }
    void binary_op_shr(const mag_command_t *cmd) { impl_binary_op_logical<op_shr>(cmd); }
    void binary_op_eq(const mag_command_t *cmd) { impl_binary_op_cmp<op_eq>(cmd); }
    void binary_op_ne(const mag_command_t *cmd) { impl_binary_op_cmp<op_ne>(cmd); }
    void binary_op_le(const mag_command_t *cmd) { impl_binary_op_cmp<op_le>(cmd); }
    void binary_op_ge(const mag_command_t *cmd) { impl_binary_op_cmp<op_ge>(cmd); }
    void binary_op_lt(const mag_command_t *cmd) { impl_binary_op_cmp<op_lt>(cmd); }
    void binary_op_gt(const mag_command_t *cmd) { impl_binary_op_cmp<op_gt>(cmd); }
}
