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

namespace mag {
    struct tensor_coords final {
        int rank = 0;
        int shape[MAG_MAX_DIMS] = {};
        int strides[MAG_MAX_DIMS] = {};
        int storage_offset = 0;

        explicit tensor_coords(const mag_tensor_t *base) {
            rank = static_cast<int>(base->rank);
            for (int i = 0; i < rank; i++) {
                shape[i] = static_cast<int>(base->shape[i]);
                strides[i] = static_cast<int>(base->strides[i]);
            }
            storage_offset = static_cast<int>(base->storage_offset);
        }
    };

    static __device__ void mag_index_to_offsets(
        int64_t linear,
        const tensor_coords& r,
        const tensor_coords& x,
        const tensor_coords& y,
        int& off_r,
        int& off_x,
        int& off_y
    ) {
        int64_t tmp = linear;
        off_r = r.storage_offset;
        off_x = x.storage_offset;
        off_y = y.storage_offset;
        for (int d=r.rank-1; d >= 0; --d) {
            int64_t size_d = r.shape[d];
            int i_d = tmp % size_d;
            tmp /= size_d;
            off_r += i_d*r.strides[d];
            int64_t ix = x.shape[d] == 1 ? 0 : i_d;
            int64_t iy = y.shape[d] == 1 ? 0 : i_d;
            off_x += ix*x.strides[d];
            off_y += iy*y.strides[d];
        }
    }

    template <typename T> struct op_add { using scalar = T; __device__ __forceinline__ T operator()(T a, T b) const { return a+b; } };
    template <typename T> struct op_sub { using scalar = T; __device__ __forceinline__ T operator()(T a, T b) const { return a-b; } };
    template <typename T> struct op_mul { using scalar = T; __device__ __forceinline__ T operator()(T a, T b) const { return a*b; } };
    template <typename T> struct op_div { using scalar = T; __device__ __forceinline__ T operator()(T a, T b) const { return a/b; } };

    template <typename Op>
    static __global__ void binary_op_kernel_contig(Op op, int n, typename Op::scalar  *o, const typename Op::scalar  *x, const typename Op::scalar  *y) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < n; i += step)
            o[i] = op(x[i], y[i]);
    }

    template <typename Op>
    static __global__ void binary_op_kernel_strided(
        Op op,
        typename Op::scalar *o,
        const typename Op::scalar  *x,
        const typename Op::scalar  *y,
        const tensor_coords& rc,
        const tensor_coords& xc,
        const tensor_coords& yc,
        int64_t total
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < total; i += step) {
            int off_r, off_x, off_y;
            mag_index_to_offsets(i, rc, xc, yc, off_r, off_x, off_y);
            o[off_r] = op(x[off_x], y[off_y]);
        }
    }

    template <typename Op>
    static void launch_binary_op(
        mag_tensor_t *out,
        const mag_tensor_t *x,
        const mag_tensor_t *y
    ) {
        int64_t total = mag_tensor_get_numel(out);
        int blocks = (total+BINARY_BLOCK_SIZE-1)/BINARY_BLOCK_SIZE;
        bool contig =
            mag_tensor_is_contiguous(out)
            && mag_tensor_is_contiguous(x)
            && mag_tensor_is_contiguous(y)
            && x->numel == total
            && y->numel == total;
        if (contig) {
            binary_op_kernel_contig<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                Op {},
                static_cast<int>(total),
                static_cast<Op::scalar*>(mag_tensor_get_data_ptr(out)),
                static_cast<Op::scalar*>(mag_tensor_get_data_ptr(x)),
                static_cast<Op::scalar*>(mag_tensor_get_data_ptr(y))
            );
        } else {
            binary_op_kernel_strided<Op><<<blocks, BINARY_BLOCK_SIZE>>>(
                Op {},
                static_cast<Op::scalar*>(mag_tensor_get_data_ptr(out)),
               static_cast<Op::scalar*>(mag_tensor_get_data_ptr(x)),
               static_cast<Op::scalar*>(mag_tensor_get_data_ptr(y)),
                tensor_coords {out},
                tensor_coords {x},
                tensor_coords {y},
                total
            );
        }
    }

    void binary_op_add(const mag_command_t *cmd) {
        mag_tensor_t *out = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(out->dtype == x->dtype && out->dtype == y->dtype);
        switch (out->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<op_add<mag_e8m23_t>>(out, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<op_add<half>>(out, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<op_add<int32_t>>(out, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void binary_op_sub(const mag_command_t *cmd) {
        mag_tensor_t *out = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(out->dtype == x->dtype && out->dtype == y->dtype);
        switch (out->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<op_sub<mag_e8m23_t>>(out, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<op_sub<half>>(out, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<op_sub<int32_t>>(out, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void binary_op_mul(const mag_command_t *cmd) {
        mag_tensor_t *out = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(out->dtype == x->dtype && out->dtype == y->dtype);
        switch (out->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<op_mul<mag_e8m23_t>>(out, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<op_mul<half>>(out, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<op_mul<int32_t>>(out, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void binary_op_div(const mag_command_t *cmd) {
        mag_tensor_t *out = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        mag_assert2(out->dtype == x->dtype && out->dtype == y->dtype);
        switch (out->dtype) {
            case MAG_DTYPE_E8M23: launch_binary_op<op_div<mag_e8m23_t>>(out, x, y); break;
            case MAG_DTYPE_E5M10: launch_binary_op<op_div<half>>(out, x, y); break;
            case MAG_DTYPE_I32: launch_binary_op<op_div<int32_t>>(out, x, y); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }
}