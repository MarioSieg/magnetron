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

#include "mag_cuda_unary.cuh"

namespace mag {
    constexpr mag_e8m23_t INVSQRT2 = 0.707106781186547524400844362104849039284835937f /* 1/√2 */;

    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_abs(mag_e8m23_t x) { return abs(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sgn(mag_e8m23_t x) { return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_neg(mag_e8m23_t x) { return -x; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_log(mag_e8m23_t x) { return logf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sqr(mag_e8m23_t x) { return x*x; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sqrt(mag_e8m23_t x) { return sqrtf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sin(mag_e8m23_t x) { return sinf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_cos(mag_e8m23_t x) { return cosf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_step(mag_e8m23_t x) { return x > 0.f; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_exp(mag_e8m23_t x) { return expf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_floor(mag_e8m23_t x) { return floorf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_ceil(mag_e8m23_t x) { return ceilf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_round(mag_e8m23_t x) { return roundf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_softmax(mag_e8m23_t x) { return exp(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_softmax_dv(mag_e8m23_t x) { return exp(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sigmoid(mag_e8m23_t x) { return 1.f/(1.f + expf(-x)); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_sigmoid_dv(mag_e8m23_t x) { mag_e8m23_t sig = 1.f/(1.f + expf(-x)); return sig*(1.f-sig); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_hard_sigmoid(mag_e8m23_t x) { return fminf(1.f, fmaxf(0.0f, (x + 3.0f)/6.0f)); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_silu(mag_e8m23_t x) { return x*(1.f/(1.f + expf(-x))); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_silu_dv(mag_e8m23_t x) { mag_e8m23_t sig = 1.f/(1.f + expf(-x)); return sig + x*sig; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_tanh(mag_e8m23_t x) { return tanhf(x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_tanh_dv(mag_e8m23_t x) { mag_e8m23_t th = tanhf(x); return 1.f - th*th; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_relu(mag_e8m23_t x) { return fmax(0.f, x); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_relu_dv(mag_e8m23_t x) { return x > 0.f ? 1.f : 0.f; }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_gelu(mag_e8m23_t x) { return .5f*x*(1.f+erff(x*INVSQRT2)); }
    [[nodiscard]] static __device__ __forceinline__ mag_e8m23_t op_gelu_dv(mag_e8m23_t x) { mag_e8m23_t th = tanhf(x); return .5f*(1.f + th) + .5f*x*(1.f - th*th); }

    template <mag_e8m23_t (&op)(mag_e8m23_t), typename T>
    static __global__ void unary_op_kernel(int n, T *o, const T *x) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        o[i] = static_cast<T>(op(static_cast<mag_e8m23_t>(x[i])));
    }

    template <mag_e8m23_t (&op)(mag_e8m23_t)>
    static void impl_unary_op(mag_tensor_t *o, mag_tensor_t *x) {
        mag_assert2(o->numel == x->numel);
        mag_assert2(mag_tensor_is_contiguous(o));
        mag_assert2(mag_tensor_is_contiguous(x));
        mag_assert2(mag_tensor_is_floating_point_typed(o));
        mag_assert2(mag_tensor_is_floating_point_typed(x));
        mag_assert2(o->dtype == x->dtype);
        int n = static_cast<int>(o->numel);
        int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        const auto launch = [=]<typename T>() {
            mag_assert2(mag_dtype_meta_of(o->dtype)->size == sizeof(T));
            auto *xo = static_cast<T *>(mag_tensor_get_data_ptr(o));
            const auto *xx = static_cast<const T *>(mag_tensor_get_data_ptr(x));
            unary_op_kernel<op><<<blocks, UNARY_BLOCK_SIZE, 0>>>(n, xo, xx);
        };
        switch (o->dtype) {
            case MAG_DTYPE_E8M23: launch.template operator()<mag_e8m23_t>(); break;
            case MAG_DTYPE_E5M10: launch.template operator()<half>(); break;
            default: mag_assert(false, "Unsupported dtype for unary op");
        }
    }

    void unary_op_abs(const mag_command_t *cmd) { impl_unary_op<op_abs>(cmd->out[0], cmd->in[0]); }
    void unary_op_sgn(const mag_command_t *cmd) { impl_unary_op<op_sgn>(cmd->out[0], cmd->in[0]); }
    void unary_op_neg(const mag_command_t *cmd) { impl_unary_op<op_neg>(cmd->out[0], cmd->in[0]); }
    void unary_op_log(const mag_command_t *cmd) { impl_unary_op<op_log>(cmd->out[0], cmd->in[0]); }
    void unary_op_sqr(const mag_command_t *cmd) { impl_unary_op<op_sqr>(cmd->out[0], cmd->in[0]); }
    void unary_op_sqrt(const mag_command_t *cmd) { impl_unary_op<op_sqrt>(cmd->out[0], cmd->in[0]); }
    void unary_op_sin(const mag_command_t *cmd) { impl_unary_op<op_sin>(cmd->out[0], cmd->in[0]); }
    void unary_op_cos(const mag_command_t *cmd) { impl_unary_op<op_cos>(cmd->out[0], cmd->in[0]); }
    void unary_op_step(const mag_command_t *cmd) { impl_unary_op<op_step>(cmd->out[0], cmd->in[0]); }
    void unary_op_exp(const mag_command_t *cmd) { impl_unary_op<op_exp>(cmd->out[0], cmd->in[0]); }
    void unary_op_floor(const mag_command_t *cmd) { impl_unary_op<op_floor>(cmd->out[0], cmd->in[0]); }
    void unary_op_ceil(const mag_command_t *cmd) { impl_unary_op<op_ceil>(cmd->out[0], cmd->in[0]); }
    void unary_op_round(const mag_command_t *cmd) { impl_unary_op<op_round>(cmd->out[0], cmd->in[0]); }
    void unary_op_softmax(const mag_command_t *cmd) { impl_unary_op<op_softmax>(cmd->out[0], cmd->in[0]); }
    void unary_op_softmax_dv(const mag_command_t *cmd) { impl_unary_op<op_softmax_dv>(cmd->out[0], cmd->in[0]); }
    void unary_op_sigmoid(const mag_command_t *cmd) { impl_unary_op<op_sigmoid>(cmd->out[0], cmd->in[0]); }
    void unary_op_sigmoid_dv(const mag_command_t *cmd) { impl_unary_op<op_sigmoid_dv>(cmd->out[0], cmd->in[0]); }
    void unary_op_hard_sigmoid(const mag_command_t *cmd) { impl_unary_op<op_hard_sigmoid>(cmd->out[0], cmd->in[0]); }
    void unary_op_silu(const mag_command_t *cmd) { impl_unary_op<op_silu>(cmd->out[0], cmd->in[0]); }
    void unary_op_silu_dv(const mag_command_t *cmd) { impl_unary_op<op_silu_dv>(cmd->out[0], cmd->in[0]); }
    void unary_op_tanh(const mag_command_t *cmd) { impl_unary_op<op_tanh>(cmd->out[0], cmd->in[0]); }
    void unary_op_tanh_dv(const mag_command_t *cmd) { impl_unary_op<op_tanh_dv>(cmd->out[0], cmd->in[0]); }
    void unary_op_relu(const mag_command_t *cmd) { impl_unary_op<op_relu>(cmd->out[0], cmd->in[0]); }
    void unary_op_relu_dv(const mag_command_t *cmd) { impl_unary_op<op_relu_dv>(cmd->out[0], cmd->in[0]); }
    void unary_op_gelu(const mag_command_t *cmd) { impl_unary_op<op_gelu>(cmd->out[0], cmd->in[0]); }
    void unary_op_gelu_dv(const mag_command_t *cmd) { impl_unary_op<op_gelu_dv>(cmd->out[0], cmd->in[0]); }
}
