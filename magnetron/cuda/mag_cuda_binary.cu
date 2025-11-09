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
    struct tensor_coords final {
        int rank = 0;
        int shape[MAG_MAX_DIMS] = {};
        int strides[MAG_MAX_DIMS] = {};
        int storage_offset = 0;

        explicit tensor_coords(const mag_tensor_t *base) {
            rank = static_cast<int>(base->rank);
            for (int i=0; i < rank; i++) {
                shape[i] = static_cast<int>(base->shape[i]);
                strides[i] = static_cast<int>(base->strides[i]);
            }
            storage_offset = static_cast<int>(base->storage_offset);
        }

        [[nodiscard]] __device__ static cuda::std::tuple<int, int, int> index_to_offsets(
            int i,
            const tensor_coords& r,
            const tensor_coords& x,
            const tensor_coords& y
        ) {
            int t = i;
            int off_r = r.storage_offset;
            int off_x = x.storage_offset;
            int off_y = y.storage_offset;
            for (int d=r.rank-1; d >= 0; --d) {
                int size_d = r.shape[d];
                int i_d = t % size_d;
                t /= size_d;
                off_r += i_d*r.strides[d];
                int ix = x.shape[d] == 1 ? 0 : i_d;
                int iy = y.shape[d] == 1 ? 0 : i_d;
                off_x += ix*x.strides[d];
                off_y += iy*y.strides[d];
            }
            return {off_r, off_x, off_y};
        }
    };

    template <typename In, typename Out>
    struct op_add {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a+b; }
    };

    template <typename In, typename Out>
    struct op_sub {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a-b; }
    };

    template <typename In, typename Out>
    struct op_mul {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a*b; }
    };

    template <typename In, typename Out>
    struct op_div {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a/b; }
    };

    template <typename In, typename Out>
    struct op_and {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a&b; }
    };

    template <typename In, typename Out>
    struct op_or {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a|b; }
    };

    template <typename In, typename Out>
    struct op_xor {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a^b; }
    };

    template <typename In, typename Out>
    struct op_shl {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a<<b; }
    };

    template <typename In, typename Out>
    struct op_shr {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a>>b; }
    };

    template <typename In, typename Out>
    struct op_eq {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a==b; }
    };

    template <typename In, typename Out>
    struct op_ne {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a!=b; }
    };

    template <typename In, typename Out>
    struct op_le {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a<=b; }
    };

    template <typename In, typename Out>
    struct op_ge {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a>=b; }
    };

    template <typename In, typename Out>
    struct op_lt {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a<b; }
    };

    template <typename In, typename Out>
    struct op_gt {
        using TIn = In;
        using TOut = Out;
        [[nodiscard]] __device__ __forceinline__ static Out eval(In a, In b) { return a>b; }
    };

    template <typename Op>
    __global__ static void binary_op_kernel_contig(int n, typename Op::TOut *o, const typename Op::TIn *x, const typename Op::TIn *y) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < n; i += step)
            o[i] = Op::eval(x[i], y[i]);
    }

    template <typename Op>
    __global__ static void binary_op_kernel_strided(
        typename Op::TOut *o,
        const typename Op::TIn *x,
        const typename Op::TIn *y,
        const tensor_coords& rc,
        const tensor_coords& xc,
        const tensor_coords& yc,
        int total
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < total; i += step) {
            auto [off_r, off_x, off_y] = tensor_coords::index_to_offsets(i, rc, xc, yc);
            o[off_r] = Op::eval(x[off_x], y[off_y]);
        }
    }

    template <typename Op>
    static void launch_binary_op(
        mag_tensor_t *r,
        const mag_tensor_t *x,
        const mag_tensor_t *y
    ) {
        int total = static_cast<int>(mag_tensor_get_numel(r));
        int blocks = (total+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE;
        if (mag_full_cont3(r, x, y)) {
            binary_op_kernel_contig<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                static_cast<int>(total),
                static_cast<typename Op::TOut *>(mag_tensor_get_data_ptr(r)),
                static_cast<const typename Op::TIn *>(mag_tensor_get_data_ptr(x)),
                static_cast<const typename Op::TIn *>(mag_tensor_get_data_ptr(y))
            );
        } else {
            binary_op_kernel_strided<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                static_cast<typename Op::TOut *>(mag_tensor_get_data_ptr(r)),
               static_cast<const typename Op::TIn *>(mag_tensor_get_data_ptr(x)),
               static_cast<const typename Op::TIn *>(mag_tensor_get_data_ptr(y)),
                tensor_coords {r},
                tensor_coords {x},
                tensor_coords {y},
                total
            );
        }
    }

    template <template <typename, typename> typename Op>
    static void impl_binary_op_numeric(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<Op<mag_e8m23_t, mag_e8m23_t>>(r, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<Op<half, half>>(r, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_dtype_meta_of(r->dtype));
        }
    }

    template <template <typename, typename> typename Op>
    static void impl_binary_op_logical(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == x->dtype && r->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_I32: launch_binary_op<Op<int32_t, int32_t>>(r, x, y); break;
            case MAG_DTYPE_BOOL: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_dtype_meta_of(r->dtype));
        }
    }

    template <template <typename, typename> typename Op>
    static void impl_binary_op_cmp(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(r->dtype == MAG_DTYPE_BOOL && x->dtype == y->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<Op<mag_e8m23_t, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<Op<half, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_BOOL: launch_binary_op<Op<uint8_t, uint8_t>>(r, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<Op<int32_t, uint8_t>>(r, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation: %s", mag_dtype_meta_of(r->dtype));
        }
    }

    void binary_op_add(const mag_command_t *cmd) { impl_binary_op_numeric<op_add>(cmd); }
    void binary_op_sub(const mag_command_t *cmd) { impl_binary_op_numeric<op_sub>(cmd); }
    void binary_op_mul(const mag_command_t *cmd) { impl_binary_op_numeric<op_mul>(cmd); }
    void binary_op_div(const mag_command_t *cmd) { impl_binary_op_numeric<op_div>(cmd); }
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
