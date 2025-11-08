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

namespace mag {
    template <typename T>
    static __global__ void fill_kernel(T *data, int size, T value) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < size) data[idx] = value;
    }

    template <typename T>
    static void launch_fill_kernel(T *data, int size, T value) {
        int blocks = (size+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        fill_kernel<T><<<blocks, UNARY_BLOCK_SIZE>>>(data, size, value);
    }

    void fill_op_fill(const mag_command_t *cmd) {
        mag_tensor_t *out = cmd->in[0];
        int numel = static_cast<int>(mag_tensor_get_numel(out));
        switch (out->dtype) {
            case MAG_DTYPE_E8M23: launch_fill_kernel<float>(static_cast<mag_e8m23_t *>(mag_tensor_get_data_ptr(out)), numel, mag_op_param_unpack_e8m23_or_panic(cmd->params[0])); break;
            case MAG_DTYPE_E5M10: launch_fill_kernel<half>(static_cast<half *>(mag_tensor_get_data_ptr(out)), numel, mag_op_param_unpack_e8m23_or_panic(cmd->params[0])); break;
            case MAG_DTYPE_I32: launch_fill_kernel<int32_t>(static_cast<int32_t *>(mag_tensor_get_data_ptr(out)), numel, mag_op_param_unpack_i64_or_panic(cmd->params[0])); break;
            case MAG_DTYPE_BOOL: launch_fill_kernel<uint8_t>(static_cast<uint8_t *>(mag_tensor_get_data_ptr(out)), numel, !!mag_op_param_unpack_i64_or_panic(cmd->params[0])); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }
}
