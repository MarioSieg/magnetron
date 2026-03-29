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

#include "mag_cuda_unary.cuh"

#include <array>

namespace mag {
    template <typename src_t, typename dst_t, const bool contig>
    __global__ static void cast_kernel(
        int64_t n,
        dst_t *__restrict__ o,
        const src_t *__restrict__ x,
        mag_coords_iter_t xc
    ) {
        int64_t i = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x) + threadIdx.x;
        if constexpr (contig) {
            if (i >= n) return;
            o[i] = static_cast<dst_t>(x[i]);
        } else {
            int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
            for (; i < n; i += step) {
                int64_t xi = mag_coords_iter_to_offset(&xc, i);
                o[i] = static_cast<dst_t>(x[xi]);
            }
        }
    }

    template <typename src_t, typename dst_t>
    static void mag_cast_launcher(
        mag_tensor_t *r,
        const mag_tensor_t *x
    ) {
        int64_t n = mag_tensor_numel(r);
        auto *pr = reinterpret_cast<dst_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const src_t *>(mag_tensor_data_ptr(x));
        int64_t blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        mag_coords_iter_t xc;
        mag_coords_iter_init(&xc, &x->coords);
        if (mag_tensor_is_contiguous(x)) cast_kernel<src_t, dst_t, true><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, xc);
        else cast_kernel<src_t, dst_t, false><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, xc);
    }

    void unary_op_cast(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        using cast_fn = void (mag_tensor_t *, const mag_tensor_t *);
        static constexpr void (*const cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(mag_tensor_t *, const mag_tensor_t *) = {
            [MAG_DTYPE_FLOAT32] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<float, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<float, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<float, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<float, uint8_t>,   // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<float, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<float, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<float, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<float, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<float, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<float, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<float, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<float, int64_t>,
            },
            [MAG_DTYPE_FLOAT16] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<half, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<half, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<half, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<half, uint8_t>,   // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<half, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<half, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<half, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<half, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<half, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<half, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<half, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<half, int64_t>,
            },
            [MAG_DTYPE_BFLOAT16] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<__nv_bfloat16, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<__nv_bfloat16, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<__nv_bfloat16, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<__nv_bfloat16, uint8_t>,   // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<__nv_bfloat16, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<__nv_bfloat16, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<__nv_bfloat16, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<__nv_bfloat16, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<__nv_bfloat16, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<__nv_bfloat16, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<__nv_bfloat16, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<__nv_bfloat16, int64_t>,
            },
            [MAG_DTYPE_BOOLEAN] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint8_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint8_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint8_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint8_t, uint8_t>,     // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint8_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<uint8_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint8_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<uint8_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint8_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<uint8_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint8_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<uint8_t, int64_t>,
            },
            [MAG_DTYPE_UINT8] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint8_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint8_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint8_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint8_t, uint8_t>,     // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint8_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<uint8_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint8_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<uint8_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint8_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<uint8_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint8_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<uint8_t, int64_t>,
            },
            [MAG_DTYPE_INT8] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int8_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int8_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int8_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int8_t, uint8_t>,      // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<int8_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<int8_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<int8_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<int8_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<int8_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<int8_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<int8_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<int8_t, int64_t>,
            },
            [MAG_DTYPE_UINT16] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint16_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint16_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint16_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint16_t, uint8_t>,    // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint16_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<uint16_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint16_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<uint16_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint16_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<uint16_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint16_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<uint16_t, int64_t>,
            },
            [MAG_DTYPE_INT16] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int16_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int16_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int16_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int16_t, uint8_t>,     // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<int16_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<int16_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<int16_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<int16_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<int16_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<int16_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<int16_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<int16_t, int64_t>,
            },
            [MAG_DTYPE_UINT32] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint32_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint32_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint32_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint32_t, uint8_t>,    // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint32_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<uint32_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint32_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<uint32_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint32_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<uint32_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint32_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<uint32_t, int64_t>,
            },
            [MAG_DTYPE_INT32] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int32_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int32_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int32_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int32_t, uint8_t>,     // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<int32_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<int32_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<int32_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<int32_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<int32_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<int32_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<int32_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<int32_t, int64_t>,
            },
            [MAG_DTYPE_UINT64] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<uint64_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<uint64_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<uint64_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<uint64_t, uint8_t>,    // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<uint64_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<uint64_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<uint64_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<uint64_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<uint64_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<uint64_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<uint64_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<uint64_t, int64_t>,
            },
            [MAG_DTYPE_INT64] = {
                [MAG_DTYPE_FLOAT32] = &mag_cast_launcher<int64_t, float>,
                [MAG_DTYPE_FLOAT16] = &mag_cast_launcher<int64_t, half>,
                [MAG_DTYPE_BFLOAT16] = &mag_cast_launcher<int64_t, __nv_bfloat16>,
                [MAG_DTYPE_BOOLEAN] = &mag_cast_launcher<int64_t, uint8_t>,     // bool uses uint8_t kernels
                [MAG_DTYPE_UINT8] = &mag_cast_launcher<int64_t, uint8_t>,
                [MAG_DTYPE_INT8] = &mag_cast_launcher<int64_t, int8_t>,
                [MAG_DTYPE_UINT16] = &mag_cast_launcher<int64_t, uint16_t>,
                [MAG_DTYPE_INT16] = &mag_cast_launcher<int64_t, int16_t>,
                [MAG_DTYPE_UINT32] = &mag_cast_launcher<int64_t, uint32_t>,
                [MAG_DTYPE_INT32] = &mag_cast_launcher<int64_t, int32_t>,
                [MAG_DTYPE_UINT64] = &mag_cast_launcher<int64_t, uint64_t>,
                [MAG_DTYPE_INT64] = &mag_cast_launcher<int64_t, int64_t>,
            },
        };
        static_assert(std::size(cast_table_2D) == static_cast<size_t>(MAG_DTYPE__NUM));
        static_assert([]() -> bool {
            for (auto *fn : cast_table_2D) if (!fn) return false;
            return true;
        }());
        mag_dtype_t src = x->dtype;
        mag_dtype_t dst = r->dtype;
        cast_fn *kernel = cast_table_2D[src][dst];
        mag_assert(kernel, "No kernel found for type cast: %s -> %s", mag_type_trait(src)->name, mag_type_trait(dst)->name);
        (*kernel)(r, x);
    }

    template <typename scalar_t>
    __global__ static void clone_strided_kernel(
        int64_t n,
        scalar_t *o,
        const scalar_t *x,
        mag_coords_iter_t rc,
        mag_coords_iter_t xc
    ) {
        int64_t i = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
        for (; i < n; i += step) {
            int64_t ri = mag_coords_iter_to_offset(&rc, i);
            int64_t xi = mag_coords_iter_to_offset(&xc, i);
            o[ri] = x[xi];
        }
    }

    template <typename scalar_t>
    static void launch_clone(mag_tensor_t *r, const mag_tensor_t *x) {
        int64_t n = mag_tensor_numel(r);
        auto *pr = reinterpret_cast<scalar_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const scalar_t *>(mag_tensor_data_ptr(x));
        if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) { // TODO: Can be relaxed to non-shape
            cudaMemcpy(pr, px, n*sizeof(scalar_t), cudaMemcpyDeviceToDevice);
            return;
        }
        mag_coords_iter_t rc, xc;
        mag_coords_iter_init(&rc, &r->coords);
        mag_coords_iter_init(&xc, &x->coords);
        int64_t blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        clone_strided_kernel<scalar_t><<<blocks, UNARY_BLOCK_SIZE>>>(n, pr, px, rc, xc);
    }

    void unary_op_clone(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_clone<float>(r, x); break;
            case MAG_DTYPE_FLOAT16: launch_clone<half>(r, x); break;
            case MAG_DTYPE_BFLOAT16: launch_clone<__nv_bfloat16>(r, x); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_clone<uint8_t>(r, x); break;
            case MAG_DTYPE_INT8: launch_clone<int8_t>(r, x); break;
            case MAG_DTYPE_UINT16: launch_clone<uint16_t>(r, x); break;
            case MAG_DTYPE_INT16: launch_clone<int16_t>(r, x); break;
            case MAG_DTYPE_UINT32: launch_clone<uint32_t>(r, x); break;
            case MAG_DTYPE_INT32: launch_clone<int32_t>(r, x); break;
            case MAG_DTYPE_UINT64: launch_clone<uint64_t>(r, x); break;
            case MAG_DTYPE_INT64: launch_clone<int64_t>(r, x); break;
            default: mag_assert(false, "Unsupported dtype for unary op");
        }
    }

    constexpr float INVSQRT2 = 0.707106781186547524400844362104849039284835937f /* 1/√2 */;

    template <typename scalar_t>
    struct op_abs {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<float>(fabsf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_sgn {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            return xf32 > 0.f ? static_cast<out_t>(1.f) : xf32 < 0.f ? static_cast<out_t>(-1.f) : static_cast<out_t>(0.f);
        }
    };

    template <typename scalar_t>
    struct op_neg {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(-static_cast<float>(x));
        }
    };

    template <typename scalar_t>
    struct op_not {
        static_assert(std::is_integral_v<scalar_t>);
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return ~x;
        }
    };

    template <typename scalar_t>
    struct op_log {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__logf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_log10 {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__log10f(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_log1p {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(log1pf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_log2 {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(log2f(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_sqr {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            return static_cast<out_t>(xf32*xf32);
        }
    };

    template <typename scalar_t>
    struct op_rcp {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(1.f/static_cast<float>(x));
        }
    };

    template <typename scalar_t>
    struct op_sqrt {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(sqrtf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_rsqrt {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(1.f/sqrtf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_sin {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__sinf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_cos {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__cosf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_tan {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__tanf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_asin {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(asinf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_acos {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(acosf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_atan {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(atanf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_sinh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(sinhf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_cosh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(coshf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_tanh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(tanhf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_asinh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(asinhf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_acosh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(acoshf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_atanh {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(atanhf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_step {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<float>(x) > .0f ? static_cast<out_t>(1.f) : static_cast<out_t>(.0f);
        }
    };

    template <typename scalar_t>
    struct op_erf {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(erff(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_erfc {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(erfcf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_exp {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__expf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_exp2 {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(exp2f(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_expm1 {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(expm1f(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_floor {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(floorf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_ceil {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(ceilf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_round {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(roundf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_trunc {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(truncf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_softmax {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__expf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_softmax_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(__expf(static_cast<float>(x)));
        }
    };

    template <typename scalar_t>
    struct op_sigmoid {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            return static_cast<out_t>(1.f/(1.f + __expf(-xf32)));
        }
    };

    template <typename scalar_t>
    struct op_sigmoid_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            float sig = 1.f/(1.f + __expf(-static_cast<float>(x)));
            return static_cast<out_t>(sig*(1.f-sig));
        }
    };

    template <typename scalar_t>
    struct op_hard_sigmoid {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(fminf(1.f, fmaxf(.0f, (static_cast<float>(x) + 3.f)/6.f)));
        }
    };

    template <typename scalar_t>
    struct op_silu {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            return static_cast<out_t>(xf32*(1.f/(1.f + __expf(-xf32))));
        }
    };

    template <typename scalar_t>
    struct op_silu_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            float sig = 1.f/(1.f + __expf(-xf32));
            return static_cast<out_t>(sig + xf32*sig);
        }
    };

    template <typename scalar_t>
    struct op_tanh_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            float th = __tanhf(static_cast<float>(x));
            return static_cast<out_t>(1.f - th*th);
        }
    };

    template <typename scalar_t>
    struct op_relu {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<out_t>(fmaxf(static_cast<float>(x),0.f));
        }
    };

    template <typename scalar_t>
    struct op_relu_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            return static_cast<float>(x) > 0.f ? static_cast<out_t>(1.f) : static_cast<out_t>(0.f);
        }
    };

    template <typename scalar_t>
    struct op_gelu {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            return static_cast<out_t>(.5f*xf32*(1.f+erff(xf32*INVSQRT2)));
        }
    };

    template <typename scalar_t>
    struct op_gelu_dv {
        using in_t = scalar_t;
        using out_t = scalar_t;
        [[nodiscard]] __device__ __forceinline__ out_t operator()(in_t x) const {
            auto xf32 = static_cast<float>(x);
            float th = __tanhf(xf32);
            return static_cast<out_t>(.5f*(1.f + th) + .5f*xf32*(1.f - th*th));
        }
    };

    template <typename op_t, const bool contig>
    __global__ static void unary_op_kernel(
        op_t op,
        int64_t n,
        typename op_t::out_t *o,
        const typename op_t::in_t *x,
        [[maybe_unused]] mag_coords_iter_t rc,
        [[maybe_unused]] mag_coords_iter_t xc
    ) {
        int64_t i = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
        if constexpr (contig) {
            if (i >= n) return;
            o[i] = static_cast<typename op_t::out_t>(op(static_cast<typename op_t::in_t>(x[i])));
        } else {
            int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
            for (; i < n; i += step) {
                int64_t ri = mag_coords_iter_to_offset(&rc, i);
                int64_t xi = mag_coords_iter_broadcast(&rc, &xc, i);
                o[ri] = op(x[xi]);
            }
        }
    }

    template <typename op_t>
    static void launch_unary_op(
        mag_tensor_t *r,
        const mag_tensor_t *x
    ) {
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
        mag_coords_iter_t rc, xc;
        mag_coords_iter_init(&rc, &r->coords);
        mag_coords_iter_init(&xc, &x->coords);
        auto *pr = reinterpret_cast<typename op_t::out_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const typename op_t::in_t *>(mag_tensor_data_ptr(x));
        if (std::array<const mag_tensor_t *, 2> tensors {r, x}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size())) {
            unary_op_kernel<op_t, true><<<blocks, UNARY_BLOCK_SIZE>>>(op_t{}, n, pr, px, rc, xc);
        } else {
            unary_op_kernel<op_t, false><<<blocks, UNARY_BLOCK_SIZE>>>(op_t{}, n, pr, px, rc, xc);
        }
    }

    template <template <typename> typename op_t>
    static void impl_unary_op_fp(mag_tensor_t *r, mag_tensor_t *x) {
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_unary_op<op_t<float>>(r, x); break;
            case MAG_DTYPE_FLOAT16: launch_unary_op<op_t<half>>(r, x); break;
            case MAG_DTYPE_BFLOAT16: launch_unary_op<op_t<__nv_bfloat16>>(r, x); break;
            default: mag_assert(false, "Unsupported data type in unary operation: %s", mag_type_trait(r->dtype)->name);
        }
    }

    template <template <typename> typename op_t>
    static void impl_unary_op_int(mag_tensor_t *r, mag_tensor_t *x) {
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_unary_op<op_t<uint8_t>>(r, x); break;
            case MAG_DTYPE_INT8: launch_unary_op<op_t<int8_t>>(r, x); break;
            case MAG_DTYPE_UINT16: launch_unary_op<op_t<uint16_t>>(r, x); break;
            case MAG_DTYPE_INT16: launch_unary_op<op_t<int16_t>>(r, x); break;
            case MAG_DTYPE_UINT32: launch_unary_op<op_t<uint32_t>>(r, x); break;
            case MAG_DTYPE_INT32: launch_unary_op<op_t<int32_t>>(r, x); break;
            case MAG_DTYPE_UINT64: launch_unary_op<op_t<uint64_t>>(r, x); break;
            case MAG_DTYPE_INT64: launch_unary_op<op_t<int64_t>>(r, x); break;
            default: mag_assert(false, "Unsupported data type in unary operation: %s", mag_type_trait(r->dtype)->name);
        }
    }

    void unary_op_abs(const mag_command_t &cmd) { impl_unary_op_fp<op_abs>(cmd.out[0], cmd.in[0]); }
    void unary_op_sgn(const mag_command_t &cmd) { impl_unary_op_fp<op_sgn>(cmd.out[0], cmd.in[0]); }
    void unary_op_neg(const mag_command_t &cmd) { impl_unary_op_fp<op_neg>(cmd.out[0], cmd.in[0]); }
    void unary_op_not(const mag_command_t &cmd) { impl_unary_op_int<op_not>(cmd.out[0], cmd.in[0]); }
    void unary_op_log(const mag_command_t &cmd) { impl_unary_op_fp<op_log>(cmd.out[0], cmd.in[0]); }
    void unary_op_log10(const mag_command_t &cmd) { impl_unary_op_fp<op_log10>(cmd.out[0], cmd.in[0]); }
    void unary_op_log1p(const mag_command_t &cmd) { impl_unary_op_fp<op_log1p>(cmd.out[0], cmd.in[0]); }
    void unary_op_log2(const mag_command_t &cmd) { impl_unary_op_fp<op_log2>(cmd.out[0], cmd.in[0]); }
    void unary_op_sqr(const mag_command_t &cmd) { impl_unary_op_fp<op_sqr>(cmd.out[0], cmd.in[0]); }
    void unary_op_rcp(const mag_command_t &cmd) { impl_unary_op_fp<op_rcp>(cmd.out[0], cmd.in[0]); }
    void unary_op_sqrt(const mag_command_t &cmd) { impl_unary_op_fp<op_sqrt>(cmd.out[0], cmd.in[0]); }
    void unary_op_rsqrt(const mag_command_t &cmd) { impl_unary_op_fp<op_rsqrt>(cmd.out[0], cmd.in[0]); }
    void unary_op_sin(const mag_command_t &cmd) { impl_unary_op_fp<op_sin>(cmd.out[0], cmd.in[0]); }
    void unary_op_cos(const mag_command_t &cmd) { impl_unary_op_fp<op_cos>(cmd.out[0], cmd.in[0]); }
    void unary_op_tan(const mag_command_t &cmd) { impl_unary_op_fp<op_tan>(cmd.out[0], cmd.in[0]); }
    void unary_op_asin(const mag_command_t &cmd) { impl_unary_op_fp<op_asin>(cmd.out[0], cmd.in[0]); }
    void unary_op_acos(const mag_command_t &cmd) { impl_unary_op_fp<op_acos>(cmd.out[0], cmd.in[0]); }
    void unary_op_atan(const mag_command_t &cmd) { impl_unary_op_fp<op_atan>(cmd.out[0], cmd.in[0]); }
    void unary_op_sinh(const mag_command_t &cmd) { impl_unary_op_fp<op_sinh>(cmd.out[0], cmd.in[0]); }
    void unary_op_cosh(const mag_command_t &cmd) { impl_unary_op_fp<op_cosh>(cmd.out[0], cmd.in[0]); }
    void unary_op_tanh(const mag_command_t &cmd) { impl_unary_op_fp<op_tanh>(cmd.out[0], cmd.in[0]); }
    void unary_op_asinh(const mag_command_t &cmd) { impl_unary_op_fp<op_asinh>(cmd.out[0], cmd.in[0]); }
    void unary_op_acosh(const mag_command_t &cmd) { impl_unary_op_fp<op_acosh>(cmd.out[0], cmd.in[0]); }
    void unary_op_atanh(const mag_command_t &cmd) { impl_unary_op_fp<op_atanh>(cmd.out[0], cmd.in[0]); }
    void unary_op_step(const mag_command_t &cmd) { impl_unary_op_fp<op_step>(cmd.out[0], cmd.in[0]); }
    void unary_op_erf(const mag_command_t &cmd) { impl_unary_op_fp<op_erf>(cmd.out[0], cmd.in[0]); }
    void unary_op_erfc(const mag_command_t &cmd) { impl_unary_op_fp<op_erfc>(cmd.out[0], cmd.in[0]); }
    void unary_op_exp(const mag_command_t &cmd) { impl_unary_op_fp<op_exp>(cmd.out[0], cmd.in[0]); }
    void unary_op_exp2(const mag_command_t &cmd) { impl_unary_op_fp<op_exp2>(cmd.out[0], cmd.in[0]); }
    void unary_op_expm1(const mag_command_t &cmd) { impl_unary_op_fp<op_expm1>(cmd.out[0], cmd.in[0]); }
    void unary_op_floor(const mag_command_t &cmd) { impl_unary_op_fp<op_floor>(cmd.out[0], cmd.in[0]); }
    void unary_op_ceil(const mag_command_t &cmd) { impl_unary_op_fp<op_ceil>(cmd.out[0], cmd.in[0]); }
    void unary_op_round(const mag_command_t &cmd) { impl_unary_op_fp<op_round>(cmd.out[0], cmd.in[0]); }
    void unary_op_trunc(const mag_command_t &cmd) { impl_unary_op_fp<op_trunc>(cmd.out[0], cmd.in[0]); }

    template <typename T>
    [[nodiscard]] __device__ __forceinline__ float to_f32(T x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __half2float(x);
        else return __bfloat162float(x);
    }

    template <typename T>
    [[nodiscard]] __device__ __forceinline__ T from_f32(float x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __float2half(x);
        else return __float2bfloat16(x);
    }

    template <typename scalar_t>
    __global__ static void softmax_lastdim_kernel(
        int64_t rows,
        int64_t last_dim,
        scalar_t *__restrict__ out,
        const scalar_t *__restrict__ in
    ) {
        int64_t row = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
        if (row >= rows) return;
        const scalar_t *row_in = in + row * last_dim;
        scalar_t *row_out = out + row * last_dim;
        float maxv = to_f32(row_in[0]);
        for (int64_t i = 1; i < last_dim; ++i) {
            float v = to_f32(row_in[i]);
            if (v > maxv) maxv = v;
        }
        double sum = 0.0;
        for (int64_t i = 0; i < last_dim; ++i) {
            float e = __expf(to_f32(row_in[i]) - maxv);
            row_out[i] = from_f32<scalar_t>(e);
            sum += e;
        }
        if (!isfinite(sum) || sum <= 0.0) {
            float inv = 1.0f / static_cast<float>(last_dim);
            for (int64_t i = 0; i < last_dim; ++i)
                row_out[i] = from_f32<scalar_t>(inv);
        } else {
            float inv = static_cast<float>(1.0 / sum);
            for (int64_t i = 0; i < last_dim; ++i)
                row_out[i] = from_f32<scalar_t>(to_f32(row_out[i]) * inv);
        }
    }

    template <typename scalar_t>
    static void launch_softmax_lastdim(mag_tensor_t *r, const mag_tensor_t *x) {
        mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
        mag_assert(mag_tensor_is_contiguous(r), "Softmax output tensor must be contiguous");
        int64_t rank = r->coords.rank;
        int64_t numel = r->numel;
        if (mag_unlikely(!numel)) return;
        int64_t last_dim = rank == 0 ? 1 : r->coords.shape[rank - 1];
        int64_t rows = rank == 0 ? 1 : (numel / last_dim);
        auto *o = reinterpret_cast<scalar_t *>(mag_tensor_data_ptr_mut(r));
        const auto *i = reinterpret_cast<const scalar_t *>(mag_tensor_data_ptr(x));
        int64_t blocks = (rows + UNARY_BLOCK_SIZE - 1) / UNARY_BLOCK_SIZE;
        softmax_lastdim_kernel<scalar_t><<<blocks, UNARY_BLOCK_SIZE>>>(rows, last_dim, o, i);
    }

    void unary_op_softmax(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_softmax_lastdim<float>(r, x); break;
            case MAG_DTYPE_FLOAT16: launch_softmax_lastdim<half>(r, x); break;
            case MAG_DTYPE_BFLOAT16: launch_softmax_lastdim<__nv_bfloat16>(r, x); break;
            default: mag_assert(false, "Unsupported dtype for softmax");
        }
    }
    void unary_op_softmax_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_softmax_dv>(cmd.out[0], cmd.in[0]); }
    void unary_op_sigmoid(const mag_command_t &cmd) { impl_unary_op_fp<op_sigmoid>(cmd.out[0], cmd.in[0]); }
    void unary_op_sigmoid_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_sigmoid_dv>(cmd.out[0], cmd.in[0]); }
    void unary_op_hard_sigmoid(const mag_command_t &cmd) { impl_unary_op_fp<op_hard_sigmoid>(cmd.out[0], cmd.in[0]); }
    void unary_op_silu(const mag_command_t &cmd) { impl_unary_op_fp<op_silu>(cmd.out[0], cmd.in[0]); }
    void unary_op_silu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_silu_dv>(cmd.out[0], cmd.in[0]); }
    void unary_op_tanh_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_tanh_dv>(cmd.out[0], cmd.in[0]); }
    void unary_op_relu(const mag_command_t &cmd) { impl_unary_op_fp<op_relu>(cmd.out[0], cmd.in[0]); }
    void unary_op_relu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_relu_dv>(cmd.out[0], cmd.in[0]); }
    void unary_op_gelu(const mag_command_t &cmd) { impl_unary_op_fp<op_gelu>(cmd.out[0], cmd.in[0]); }
    void unary_op_gelu_dv(const mag_command_t &cmd) { impl_unary_op_fp<op_gelu_dv>(cmd.out[0], cmd.in[0]); }
}
