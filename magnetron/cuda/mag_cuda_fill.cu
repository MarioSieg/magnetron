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

#include "mag_cuda_fill.cuh"
#include "mag_cuda_coords.cuh"

namespace mag {
    template <typename T>
    __global__ static void fill_kernel(int64_t n, T *__restrict__ o, T v, tensor_coords rc) {
        int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
        for (; i < n; i += step) {
            int64_t ri = rc.to_offset(i);
            o[ri] = v;
        }
    }

    template <typename T>
    __global__ static void masked_fill_kernel(
        int64_t n,
        T *__restrict__ o,
        const uint8_t *__restrict__ mask,
        T v,
        tensor_coords rc,
        tensor_coords mc
    ) {
        int64_t i = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (; i < n; i += step) {
            int64_t ri = rc.to_offset(i);
            int64_t mi = rc.broadcast(mc, i);
            if (mask[mi]) o[ri] = v;
        }
    }

    template <typename T>
    static void launch_fill_kernel(int64_t n, T *__restrict__ o, T v, const tensor_coords &rc) {
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        fill_kernel<T><<<blocks, FILL_BLOCK_SIZE>>>(n, o, v, rc);
    }

    template <typename T>
    static void launch_masked_fill_kernel(int64_t n, T *__restrict__ o, T v, const tensor_coords &rc, const mag_tensor_t *mask) {
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        auto *pmask = static_cast<const uint8_t *>(mag_tensor_get_data_ptr(mask));
        masked_fill_kernel<T><<<blocks, FILL_BLOCK_SIZE>>>(n, o, pmask,  v, rc, tensor_coords{mask->coords});
    }

    void fill_op_fill(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        int64_t n = mag_tensor_get_numel(r);
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: {
                auto *o = static_cast<mag_e8m23_t *>(mag_tensor_get_data_ptr(r));
                mag_e8m23_t v = mag_op_param_unpack_e8m23_or_panic(cmd->params[0]);
                launch_fill_kernel<mag_e8m23_t>(n, o, v, tensor_coords{r->coords});
            } break;
            case MAG_DTYPE_E5M10: {
                auto *o = static_cast<half *>(mag_tensor_get_data_ptr(r));
                mag_e8m23_t v = mag_op_param_unpack_e8m23_or_panic(cmd->params[0]);
                launch_fill_kernel<half>(n, o, v, tensor_coords{r->coords});
            } break;
            case MAG_DTYPE_I32: {
                auto *o = static_cast<int32_t *>(mag_tensor_get_data_ptr(r));
                int32_t v = static_cast<int32_t>(mag_op_param_unpack_i64_or_panic(cmd->params[0]));
                launch_fill_kernel<int32_t>(n, o, v, tensor_coords{r->coords});
            } break;
            case MAG_DTYPE_BOOL: {
                auto *o = static_cast<uint8_t *>(mag_tensor_get_data_ptr(r));
                uint8_t v = !!mag_op_param_unpack_i64_or_panic(cmd->params[0]);
                launch_fill_kernel<uint8_t>(n, o, v, tensor_coords{r->coords});
            } break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_masked_fill(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        mag_tensor_t *mask = reinterpret_cast<mag_tensor_t *>(static_cast<uintptr_t>(mag_op_param_unpack_i64_or_panic(cmd->params[0]))); // TODO: pass in cmd in why the fuck are these here
        int64_t n = mag_tensor_get_numel(r);
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: {
                auto *o = static_cast<mag_e8m23_t *>(mag_tensor_get_data_ptr(r));
                mag_e8m23_t v = mag_op_param_unpack_e8m23_or_panic(cmd->params[0]);
                launch_masked_fill_kernel<mag_e8m23_t>(n, o, v, tensor_coords{r->coords}, mask);
            } break;
            case MAG_DTYPE_E5M10: {
                auto *o = static_cast<half *>(mag_tensor_get_data_ptr(r));
                mag_e8m23_t v = mag_op_param_unpack_e8m23_or_panic(cmd->params[0]);
                launch_masked_fill_kernel<half>(n, o, v, tensor_coords{r->coords}, mask);
            } break;
            case MAG_DTYPE_I32: {
                auto *o = static_cast<int32_t *>(mag_tensor_get_data_ptr(r));
                int32_t v = static_cast<int32_t>(mag_op_param_unpack_i64_or_panic(cmd->params[0]));
                launch_masked_fill_kernel<int32_t>(n, o, v, tensor_coords{r->coords}, mask);
            } break;
            case MAG_DTYPE_BOOL: {
                auto *o = static_cast<uint8_t *>(mag_tensor_get_data_ptr(r));
                uint8_t v = !!mag_op_param_unpack_i64_or_panic(cmd->params[0]);
                launch_masked_fill_kernel<uint8_t>(n, o, v, tensor_coords{r->coords}, mask);
            } break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }
}
