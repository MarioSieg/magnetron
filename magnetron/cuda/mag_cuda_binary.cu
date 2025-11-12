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
#include "mag_cuda_coords.cuh"

#include <cuda/std/tuple>


namespace mag {
    template <typename TIn, typename TOut>
    struct op_add {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x+y; }
    };

    template <typename TIn, typename TOut>
    struct op_sub {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x-y; }
    };

    template <typename TIn, typename TOut>
    struct op_mul {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x*y; }
    };

    template <typename TIn, typename TOut>
    struct op_div {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x/y; }
    };

    template <typename TIn, typename TOut>
    struct op_and {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x&y; }
    };

    template <typename TIn, typename TOut>
    struct op_or {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x|y; }
    };

    template <typename TIn, typename TOut>
    struct op_xor {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x^y; }
    };

    template <typename TIn, typename TOut>
    struct op_shl {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x<<y; }
    };

    template <typename TIn, typename TOut>
    struct op_shr {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x>>y; }
    };

    template <typename TIn, typename TOut>
    struct op_eq {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x==y; }
    };

    template <typename TIn, typename TOut>
    struct op_ne {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x!=y; }
    };

    template <typename TIn, typename TOut>
    struct op_le {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x<=y; }
    };

    template <typename TIn, typename TOut>
    struct op_ge {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x>=y; }
    };

    template <typename TIn, typename TOut>
    struct op_lt {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x<y; }
    };

    template <typename TIn, typename TOut>
    struct op_gt {
        using In = TIn;
        using Out = TOut;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x, In y) const { return x>y; }
    };

    template <typename Op>
    __global__ static void binary_op_kernel_contig(
        Op op,
        int n,
        typename Op::Out *o,
        const typename Op::In *x,
        const typename Op::In *y
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < n; i += step)
            o[i] = op(x[i], y[i]);
    }

    template <typename Op>
    __global__ static void binary_op_kernel_strided(
        Op op,
        int n,
        typename Op::Out *o,
        const typename Op::In *x,
        const typename Op::In *y,
        tensor_coords rc,
        tensor_coords xc,
        tensor_coords yc
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < n; i += step) {
            int ri = rc.to_offset(i);
            int xi = rc.broadcast(xc, i);
            int yi = rc.broadcast(yc, i);
            o[ri] = op(x[xi], y[yi]);
        }
    }

    template <typename Op>
    static void launch_binary_op(
        mag_tensor_t *r,
        const mag_tensor_t *x,
        const mag_tensor_t *y
    ) {
        int n = static_cast<int>(mag_tensor_get_numel(r));
        int blocks = (n+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE;
        if (mag_full_cont3(r, x, y)) {
            binary_op_kernel_contig<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                Op{},
                n,
                static_cast<typename Op::Out *>(mag_tensor_get_data_ptr(r)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(x)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(y))
            );
        } else {
            binary_op_kernel_strided<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                Op{},
                n,
                static_cast<typename Op::Out *>(mag_tensor_get_data_ptr(r)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(x)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(y)),
                tensor_coords {r->coords},
                tensor_coords {x->coords},
                tensor_coords {y->coords}
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
