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
#include "mag_cuda_coords.cuh"

namespace mag {
    constexpr mag_e8m23_t INVSQRT2 = 0.707106781186547524400844362104849039284835937f /* 1/√2 */;

    template <typename T>
    struct op_abs {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return fabs(x); }
    };

    template <typename T>
    struct op_sgn {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f; }
    };

    template <typename T>
    struct op_neg {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return -x; }
    };

    template <typename T>
    struct op_log {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __logf(x); }
    };

    template <typename T>
    struct op_sqr {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return x*x; }
    };

    template <typename T>
    struct op_sqrt {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return sqrtf(x); }
    };

    template <typename T>
    struct op_sin {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __sinf(x); }
    };

    template <typename T>
    struct op_cos {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __cosf(x); }
    };

    template <typename T>
    struct op_step {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return x > .0f ? 1.f : .0f; }
    };

    template <typename T>
    struct op_exp {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __expf(x); }
    };

    template <typename T>
    struct op_floor {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return floorf(x); }
    };

    template <typename T>
    struct op_ceil {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return ceilf(x); }
    };

    template <typename T>
    struct op_round {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return roundf(x); }
    };

    template <typename T>
    struct op_softmax {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __expf(x); }
    };

    template <typename T>
    struct op_softmax_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __expf(x); }
    };

    template <typename T>
    struct op_sigmoid {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return 1.f/(1.f + __expf(-x)); }
    };

    template <typename T>
    struct op_sigmoid_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { mag_e8m23_t sig = 1.f/(1.f + __expf(-x)); return sig*(1.f-sig); }
    };

    template <typename T>
    struct op_hard_sigmoid {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return fminf(1.f, fmaxf(.0f, (x + 3.f)/6.f)); }
    };

    template <typename T>
    struct op_silu {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return x*(1.f/(1.f + __expf(-x))); }
    };

    template <typename T>
    struct op_silu_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { mag_e8m23_t sig = 1.f/(1.f + __expf(-x)); return sig + x*sig; }
    };

    template <typename T>
    struct op_tanh {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return __tanhf(x); }
    };

    template <typename T>
    struct op_tanh_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { mag_e8m23_t th = __tanhf(x); return 1.f - th*th; }
    };

    template <typename T>
    struct op_relu {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return fmaxf(x,0.f); }
    };

    template <typename T>
    struct op_relu_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return x > 0.f ? 1.f : 0.f; }
    };

    template <typename T>
    struct op_gelu {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { return .5f*x*(1.f+erff(x*INVSQRT2)); }
    };

    template <typename T>
    struct op_gelu_dv {
        using In = mag_e8m23_t;
        using Out = mag_e8m23_t;
        [[nodiscard]] __device__ __forceinline__ Out operator()(In x) const { mag_e8m23_t th = __tanhf(x); return .5f*(1.f + th) + .5f*x*(1.f - th*th); }
    };

    template <typename Op>
    __global__ static void unary_op_kernel_contig(
        Op op,
        int n,
        typename Op::Out *o,
        const typename Op::In *x
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        o[i] = static_cast<typename Op::Out>(op(static_cast<typename Op::In>(x[i])));
    }

    template <typename Op>
    __global__ static void unary_op_kernel_strided(
        Op op,
        int n,
        typename Op::Out *o,
        const typename Op::In *x,
        tensor_coords rc,
        tensor_coords xc
    ) {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;
        for (; i < n; i += step) {
            int ri = rc.to_offset(i);
            int xi = rc.broadcast(xc, i);
            o[ri] = static_cast<typename Op::Out>(op(static_cast<typename Op::In>(x[xi])));
        }
    }

    template <typename Op>
    static void launch_unary_op(
        mag_tensor_t *r,
        const mag_tensor_t *x
    ) {
        int n = static_cast<int>(mag_tensor_get_numel(r));
        int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        if (mag_full_cont2(r, x)) {
            unary_op_kernel_contig<Op><<<blocks, UNARY_BLOCK_SIZE>>>(
                Op{},
                n,
                static_cast<typename Op::Out *>(mag_tensor_get_data_ptr(r)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(x))
            );
        } else {
            unary_op_kernel_strided<Op><<<blocks, UNARY_BLOCK_SIZE>>>(
                Op{},
                n,
                static_cast<typename Op::Out *>(mag_tensor_get_data_ptr(r)),
                static_cast<const typename Op::In *>(mag_tensor_get_data_ptr(x)),
                tensor_coords {r->coords},
                tensor_coords {x->coords}
            );
        }
    }

    template <template <typename> typename Op>
    static void impl_unary_op(mag_tensor_t *r, mag_tensor_t *x) {
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_unary_op<Op<mag_e8m23_t>>(r, x); break;
            case MAG_DTYPE_E5M10: launch_unary_op<Op<half>>(r, x); break;
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
    void unary_op_not(const mag_command_t *cmd) { }
}
