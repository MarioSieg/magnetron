/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_cpu.h"

#include <core/mag_tensor.h>
#include <core/mag_cpuid.h>
#include <core/mag_alloc.h>
#include <core/mag_float16.h>
#include <core/mag_coords.h>
#include <core/mag_coords_iter.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

#include <float.h>
#include <math.h>

#define mag_cvt_nop(x) (x)
#define mag_cvt_int_to_bool(x) (!!(x))
#define mag_cvt_int32_to_int64(x) ((int32_t)(x))

#define mag_cmd_in(i) (payload->cmd->in[(i)])
#define mag_cmd_out(i) (payload->cmd->out[(i)])
#define mag_cmd_attr(i) (payload->cmd->attrs[(i)])

/* Remainder functions that adjust for sign, as in Python and MATLAB. C's % operator does not do this 😿 */
static MAG_AINLINE float mag_remf(float x, float y) {
    float r = fmodf(x, y);
    if (r != 0.0f && (r < 0.0f) != (y < 0.0f)) r += y;
    return r;
}

static MAG_AINLINE int64_t mag_remi(int64_t x, int64_t y) {
    int64_t r = x % y;
    if (r != 0 && (r < 0) != (y < 0)) r += y;
    return r;
}

#define mag_remu(x, y) ((x) % (y)) /* Unsigned remainder is the same as in C */

/* Arithmetic / logical shifts with defined behavior for out-of-bounds shift amounts */

#define mag_sal(x, y, bits) (mag_unlikely((y) < 0 || (y) >= (bits)) ? 0 : (x)<<(y))
#define mag_sar(x, y, bits) (mag_unlikely((y) < 0 || (y) >= (bits)) ? ((x) < 0 ? -1 : 0) : (x)>>(y))
#define mag_shl(x, y, bits) (mag_unlikely((y) < 0 || (y) >= (bits)) ? 0 : (x)<<(y))
#define mag_shr(x, y, bits) (mag_unlikely((y) < 0 || (y) >= (bits)) ? 0 : (x)>>(y))

/* Alias names for kernel macros based on signess suffix (i or u) */

#define mag_shlu(x, y, bits) mag_shl(x, y, bits)
#define mag_shru(x, y, bits) mag_shr(x, y, bits)
#define mag_shli(x, y, bits) mag_sal(x, y, bits)
#define mag_shri(x, y, bits) mag_sar(x, y, bits)

#define mag_floordivi(x, y) (((x) - mag_remi((x), (y)))/(y))
#define mag_floordivu(x, y) ((x)/(y))
#define mag_floordivf(x, y) (floorf((x)/(y)))

#define MAG_MM_SCRATCH_ALIGN MAG_DESTRUCTIVE_INTERFERENCE_SIZE

typedef struct mag_scratch_buf_t {
    void *top;
    size_t cap;
} mag_scratch_buf_t;

static MAG_THREAD_LOCAL mag_scratch_buf_t mag_tls_scratch = {0};

static void *mag_sb_acquire(size_t size) {
    mag_scratch_buf_t *sb = &mag_tls_scratch;
    if (size <= sb->cap) return sb->top; /* Enough space allocated */
    sb->top = (*mag_alloc)(sb->top, size, MAG_MM_SCRATCH_ALIGN); /* Reallocate */
    sb->cap = size;
    void *p = sb->top;
#ifndef _MSC_VER
    p = __builtin_assume_aligned(p, MAG_MM_SCRATCH_ALIGN);
#endif
    return p;
}

static void mag_sb_release(void) {
    mag_scratch_buf_t *sb = &mag_tls_scratch;
    if (sb->top) (*mag_alloc)(sb->top, 0, MAG_MM_SCRATCH_ALIGN);
    sb->top = NULL;
    sb->cap = 0;
}

#ifdef __AVX512F__ /* Vector register width in bytes */
#define MAG_VREG_WIDTH 64
#elif defined(__AVX__)
#define MAG_VREG_WIDTH 32
#elif defined(__SSE2__)
#define MAG_VREG_WIDTH 16
#elif defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON))
#define MAG_VREG_WIDTH 16
#else
#define MAG_VREG_WIDTH 16
#endif

#if defined(_MSC_VER)
typedef uint16_t __fp16; /* MSVC does not support __fp16. */
#ifdef __AVX2__ /*MSVC does not define FMA and F16C with AVX 2*/
#define __FMA__ 1
#define __F16C__ 1
#endif
#endif

#include "mag_cpu_impl_cast.inl"
#include "mag_cpu_impl_approxfn.inl"
#include "mag_cpu_impl_rand.inl"
#include "mag_cpu_impl_veclib.inl"
#include "mag_cpu_impl_ops_fill.inl"
#include "mag_cpu_impl_ops_unary.inl"
#include "mag_cpu_impl_ops_multinomial.inl"
#include "mag_cpu_impl_ops_other.inl"
#include "mag_cpu_impl_ops_prim.inl"
#include "mag_cpu_impl_ops_reduce.inl"
#include "mag_cpu_impl_ops_matmul.inl"

static void (*const mag_lut_eval_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_kernel_payload_t *) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_FLOAT32] = &mag_nop,
        [MAG_DTYPE_FLOAT16] = &mag_nop,
        [MAG_DTYPE_BOOLEAN] = &mag_nop,
        [MAG_DTYPE_UINT8] = &mag_nop,
        [MAG_DTYPE_INT8] = &mag_nop,
        [MAG_DTYPE_UINT16] = &mag_nop,
        [MAG_DTYPE_INT16] = &mag_nop,
        [MAG_DTYPE_UINT32] = &mag_nop,
        [MAG_DTYPE_INT32] = &mag_nop,
        [MAG_DTYPE_UINT64] = &mag_nop,
        [MAG_DTYPE_INT64] = &mag_nop,
    },
    [MAG_OP_FILL] = {
        [MAG_DTYPE_FLOAT32] = &mag_fill_float32,
        [MAG_DTYPE_FLOAT16] = &mag_fill_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_fill_uint8,
        [MAG_DTYPE_UINT8] = &mag_fill_uint8,
        [MAG_DTYPE_INT8] = &mag_fill_int8,
        [MAG_DTYPE_UINT16] = &mag_fill_uint16,
        [MAG_DTYPE_INT16] = &mag_fill_int16,
        [MAG_DTYPE_UINT32] = &mag_fill_uint32,
        [MAG_DTYPE_INT32] = &mag_fill_int32,
        [MAG_DTYPE_UINT64] = &mag_fill_uint64,
        [MAG_DTYPE_INT64] = &mag_fill_int64,
    },
    [MAG_OP_MASKED_FILL] = {
        [MAG_DTYPE_FLOAT32] = &mag_masked_fill_float32,
        [MAG_DTYPE_FLOAT16] = &mag_masked_fill_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_masked_fill_uint8,
        [MAG_DTYPE_UINT8] = &mag_masked_fill_uint8,
        [MAG_DTYPE_INT8] = &mag_masked_fill_int8,
        [MAG_DTYPE_UINT16] = &mag_masked_fill_uint16,
        [MAG_DTYPE_INT16] = &mag_masked_fill_int16,
        [MAG_DTYPE_UINT32] = &mag_masked_fill_uint32,
        [MAG_DTYPE_INT32] = &mag_masked_fill_int32,
        [MAG_DTYPE_UINT64] = &mag_masked_fill_uint64,
        [MAG_DTYPE_INT64] = &mag_masked_fill_int64,
    },
    [MAG_OP_RAND_UNIFORM] = {
        [MAG_DTYPE_FLOAT32] = &mag_fill_rand_uniform_float32,
        [MAG_DTYPE_FLOAT16] = &mag_fill_rand_uniform_float16,
        [MAG_DTYPE_UINT8] = &mag_fill_rand_uniform_uint8,
        [MAG_DTYPE_INT8] = &mag_fill_rand_uniform_int8,
        [MAG_DTYPE_UINT16] = &mag_fill_rand_uniform_uint16,
        [MAG_DTYPE_INT16] = &mag_fill_rand_uniform_int16,
        [MAG_DTYPE_UINT32] = &mag_fill_rand_uniform_uint32,
        [MAG_DTYPE_INT32] = &mag_fill_rand_uniform_int32,
        [MAG_DTYPE_UINT64] = &mag_fill_rand_uniform_uint64,
        [MAG_DTYPE_INT64] = &mag_fill_rand_uniform_int64,
    },
    [MAG_OP_RAND_NORMAL] = {
        [MAG_DTYPE_FLOAT32] = &mag_fill_rand_normal_float32,
        [MAG_DTYPE_FLOAT16] = &mag_fill_rand_normal_float16,
    },
    [MAG_OP_RAND_BERNOULLI] = {
        [MAG_DTYPE_BOOLEAN] = &mag_fill_rand_bernoulli_bool,
    },
    [MAG_OP_RAND_PERM] = {
        [MAG_DTYPE_UINT8] = &mag_rand_perm_uint8,
        [MAG_DTYPE_INT8] = &mag_rand_perm_int8,
        [MAG_DTYPE_UINT16] = &mag_rand_perm_uint16,
        [MAG_DTYPE_INT16] = &mag_rand_perm_int16,
        [MAG_DTYPE_UINT32] = &mag_rand_perm_uint32,
        [MAG_DTYPE_INT32] = &mag_rand_perm_int32,
        [MAG_DTYPE_UINT64] = &mag_rand_perm_uint64,
        [MAG_DTYPE_INT64] = &mag_rand_perm_int64,
    },
    [MAG_OP_ARANGE] = {
        [MAG_DTYPE_FLOAT32] = &mag_arange_float32,
        [MAG_DTYPE_FLOAT16] = &mag_arange_float16,
        [MAG_DTYPE_UINT8] = &mag_arange_uint8,
        [MAG_DTYPE_INT8] = &mag_arange_int8,
        [MAG_DTYPE_UINT16] = &mag_arange_uint16,
        [MAG_DTYPE_INT16] = &mag_arange_int16,
        [MAG_DTYPE_UINT32] = &mag_arange_uint32,
        [MAG_DTYPE_INT32] = &mag_arange_int32,
        [MAG_DTYPE_UINT64] = &mag_arange_uint64,
        [MAG_DTYPE_INT64] = &mag_arange_int64,
    },
    [MAG_OP_ONE_HOT] = {
        [MAG_DTYPE_INT64] = &mag_one_hot_int64,
    },
    [MAG_OP_CAST] = {
        [MAG_DTYPE_FLOAT32] = &mag_cast_generic,
        [MAG_DTYPE_FLOAT16] = &mag_cast_generic,
        [MAG_DTYPE_BOOLEAN] = &mag_cast_generic,
        [MAG_DTYPE_UINT8] = &mag_cast_generic,
        [MAG_DTYPE_INT8] = &mag_cast_generic,
        [MAG_DTYPE_UINT16] = &mag_cast_generic,
        [MAG_DTYPE_INT16] = &mag_cast_generic,
        [MAG_DTYPE_UINT32] = &mag_cast_generic,
        [MAG_DTYPE_INT32] = &mag_cast_generic,
        [MAG_DTYPE_UINT64] = &mag_cast_generic,
        [MAG_DTYPE_INT64] = &mag_cast_generic,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_FLOAT32] = &mag_clone_float32,
        [MAG_DTYPE_FLOAT16] = &mag_clone_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_clone_uint8,
        [MAG_DTYPE_UINT8] = &mag_clone_uint8,
        [MAG_DTYPE_INT8] = &mag_clone_int8,
        [MAG_DTYPE_UINT16] = &mag_clone_uint16,
        [MAG_DTYPE_INT16] = &mag_clone_int16,
        [MAG_DTYPE_UINT32] = &mag_clone_uint32,
        [MAG_DTYPE_INT32] = &mag_clone_int32,
        [MAG_DTYPE_UINT64] = &mag_clone_uint64,
        [MAG_DTYPE_INT64] = &mag_clone_int64,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_FLOAT32] = &mag_nop,
        [MAG_DTYPE_FLOAT16] = &mag_nop,
        [MAG_DTYPE_BOOLEAN] = &mag_nop,
        [MAG_DTYPE_UINT8] = &mag_nop,
        [MAG_DTYPE_INT8] = &mag_nop,
        [MAG_DTYPE_UINT16] = &mag_nop,
        [MAG_DTYPE_INT16] = &mag_nop,
        [MAG_DTYPE_UINT32] = &mag_nop,
        [MAG_DTYPE_INT32] = &mag_nop,
        [MAG_DTYPE_UINT64] = &mag_nop,
        [MAG_DTYPE_INT64] = &mag_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_FLOAT32] = &mag_nop,
        [MAG_DTYPE_FLOAT16] = &mag_nop,
        [MAG_DTYPE_BOOLEAN] = &mag_nop,
        [MAG_DTYPE_UINT8] = &mag_nop,
        [MAG_DTYPE_INT8] = &mag_nop,
        [MAG_DTYPE_UINT16] = &mag_nop,
        [MAG_DTYPE_INT16] = &mag_nop,
        [MAG_DTYPE_UINT32] = &mag_nop,
        [MAG_DTYPE_INT32] = &mag_nop,
        [MAG_DTYPE_UINT64] = &mag_nop,
        [MAG_DTYPE_INT64] = &mag_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_FLOAT32] = &mag_nop,
        [MAG_DTYPE_FLOAT16] = &mag_nop,
        [MAG_DTYPE_BOOLEAN] = &mag_nop,
        [MAG_DTYPE_UINT8] = &mag_nop,
        [MAG_DTYPE_INT8] = &mag_nop,
        [MAG_DTYPE_UINT16] = &mag_nop,
        [MAG_DTYPE_INT16] = &mag_nop,
        [MAG_DTYPE_UINT32] = &mag_nop,
        [MAG_DTYPE_INT32] = &mag_nop,
        [MAG_DTYPE_UINT64] = &mag_nop,
        [MAG_DTYPE_INT64] = &mag_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_FLOAT32] = &mag_mean_float32,
        [MAG_DTYPE_FLOAT16] = &mag_mean_float16,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_FLOAT32] = &mag_min_float32,
        [MAG_DTYPE_FLOAT16] = &mag_min_float16,
        [MAG_DTYPE_UINT8] = &mag_min_uint8,
        [MAG_DTYPE_INT8] = &mag_min_int8,
        [MAG_DTYPE_UINT16] = &mag_min_uint16,
        [MAG_DTYPE_INT16] = &mag_min_int16,
        [MAG_DTYPE_UINT32] = &mag_min_uint32,
        [MAG_DTYPE_INT32] = &mag_min_int32,
        [MAG_DTYPE_UINT64] = &mag_min_uint64,
        [MAG_DTYPE_INT64] = &mag_min_int64,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_FLOAT32] = &mag_max_float32,
        [MAG_DTYPE_FLOAT16] = &mag_max_float16,
        [MAG_DTYPE_UINT8] = &mag_max_uint8,
        [MAG_DTYPE_INT8] = &mag_max_int8,
        [MAG_DTYPE_UINT16] = &mag_max_uint16,
        [MAG_DTYPE_INT16] = &mag_max_int16,
        [MAG_DTYPE_UINT32] = &mag_max_uint32,
        [MAG_DTYPE_INT32] = &mag_max_int32,
        [MAG_DTYPE_UINT64] = &mag_max_uint64,
        [MAG_DTYPE_INT64] = &mag_max_int64,
    },
    [MAG_OP_ARGMIN] = {
        [MAG_DTYPE_FLOAT32] = &mag_argmin_float32,
        [MAG_DTYPE_FLOAT16] = &mag_argmin_float16,
        [MAG_DTYPE_UINT8] = &mag_argmin_uint8,
        [MAG_DTYPE_INT8] = &mag_argmin_int8,
        [MAG_DTYPE_UINT16] = &mag_argmin_uint16,
        [MAG_DTYPE_INT16] = &mag_argmin_int16,
        [MAG_DTYPE_UINT32] = &mag_argmin_uint32,
        [MAG_DTYPE_INT32] = &mag_argmin_int32,
        [MAG_DTYPE_UINT64] = &mag_argmin_uint64,
        [MAG_DTYPE_INT64] = &mag_argmin_int64,
    },
    [MAG_OP_ARGMAX] = {
        [MAG_DTYPE_FLOAT32] = &mag_argmax_float32,
        [MAG_DTYPE_FLOAT16] = &mag_argmax_float16,
        [MAG_DTYPE_UINT8] = &mag_argmax_uint8,
        [MAG_DTYPE_INT8] = &mag_argmax_int8,
        [MAG_DTYPE_UINT16] = &mag_argmax_uint16,
        [MAG_DTYPE_INT16] = &mag_argmax_int16,
        [MAG_DTYPE_UINT32] = &mag_argmax_uint32,
        [MAG_DTYPE_INT32] = &mag_argmax_int32,
        [MAG_DTYPE_UINT64] = &mag_argmax_uint64,
        [MAG_DTYPE_INT64] = &mag_argmax_int64,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_FLOAT32] = &mag_sum_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sum_float16,
        [MAG_DTYPE_UINT8] = &mag_sum_uint8,
        [MAG_DTYPE_INT8] = &mag_sum_int8,
        [MAG_DTYPE_UINT16] = &mag_sum_uint16,
        [MAG_DTYPE_INT16] = &mag_sum_int16,
        [MAG_DTYPE_UINT32] = &mag_sum_uint32,
        [MAG_DTYPE_INT32] = &mag_sum_int32,
        [MAG_DTYPE_UINT64] = &mag_sum_uint64,
        [MAG_DTYPE_INT64] = &mag_sum_int64,
    },
    [MAG_OP_PROD] = {
        [MAG_DTYPE_FLOAT32] = &mag_prod_float32,
        [MAG_DTYPE_FLOAT16] = &mag_prod_float16,
        [MAG_DTYPE_UINT8] = &mag_prod_uint8,
        [MAG_DTYPE_INT8] = &mag_prod_int8,
        [MAG_DTYPE_UINT16] = &mag_prod_uint16,
        [MAG_DTYPE_INT16] = &mag_prod_int16,
        [MAG_DTYPE_UINT32] = &mag_prod_uint32,
        [MAG_DTYPE_INT32] = &mag_prod_int32,
        [MAG_DTYPE_UINT64] = &mag_prod_uint64,
        [MAG_DTYPE_INT64] = &mag_prod_int64,
    },
    [MAG_OP_ALL] = {
        [MAG_DTYPE_FLOAT32] = &mag_all_float32,
        [MAG_DTYPE_FLOAT16] = &mag_all_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_all_uint8,
        [MAG_DTYPE_UINT8] = &mag_all_uint8,
        [MAG_DTYPE_INT8] = &mag_all_int8,
        [MAG_DTYPE_UINT16] = &mag_all_uint16,
        [MAG_DTYPE_INT16] = &mag_all_int16,
        [MAG_DTYPE_UINT32] = &mag_all_uint32,
        [MAG_DTYPE_INT32] = &mag_all_int32,
        [MAG_DTYPE_UINT64] = &mag_all_uint64,
        [MAG_DTYPE_INT64] = &mag_all_int64,
    },
    [MAG_OP_ANY] = {
        [MAG_DTYPE_FLOAT32] = &mag_any_float32,
        [MAG_DTYPE_FLOAT16] = &mag_any_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_any_uint8,
        [MAG_DTYPE_UINT8] = &mag_any_uint8,
        [MAG_DTYPE_INT8] = &mag_any_int8,
        [MAG_DTYPE_UINT16] = &mag_any_uint16,
        [MAG_DTYPE_INT16] = &mag_any_int16,
        [MAG_DTYPE_UINT32] = &mag_any_uint32,
        [MAG_DTYPE_INT32] = &mag_any_int32,
        [MAG_DTYPE_UINT64] = &mag_any_uint64,
        [MAG_DTYPE_INT64] = &mag_any_int64,
    },
    [MAG_OP_TOPK] = {
        [MAG_DTYPE_FLOAT32] = &mag_topk_float32,
        [MAG_DTYPE_FLOAT16] = &mag_topk_float16,
        [MAG_DTYPE_UINT8] = &mag_topk_uint8,
        [MAG_DTYPE_INT8] = &mag_topk_int8,
        [MAG_DTYPE_UINT16] = &mag_topk_uint16,
        [MAG_DTYPE_INT16] = &mag_topk_int16,
        [MAG_DTYPE_UINT32] = &mag_topk_uint32,
        [MAG_DTYPE_INT32] = &mag_topk_int32,
        [MAG_DTYPE_UINT64] = &mag_topk_uint64,
        [MAG_DTYPE_INT64] = &mag_topk_int64,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_FLOAT32] = &mag_abs_float32,
        [MAG_DTYPE_FLOAT16] = &mag_abs_float16,
    },
    [MAG_OP_SGN] = {
        [MAG_DTYPE_FLOAT32] = &mag_sgn_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sgn_float16,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_FLOAT32] = &mag_neg_float32,
        [MAG_DTYPE_FLOAT16] = &mag_neg_float16,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_FLOAT32] = &mag_log_float32,
        [MAG_DTYPE_FLOAT16] = &mag_log_float16,
    },
    [MAG_OP_LOG10] = {
        [MAG_DTYPE_FLOAT32] = &mag_log10_float32,
        [MAG_DTYPE_FLOAT16] = &mag_log10_float16,
    },
    [MAG_OP_LOG1P] = {
        [MAG_DTYPE_FLOAT32] = &mag_log1p_float32,
        [MAG_DTYPE_FLOAT16] = &mag_log1p_float16,
    },
    [MAG_OP_LOG2] = {
        [MAG_DTYPE_FLOAT32] = &mag_log2_float32,
        [MAG_DTYPE_FLOAT16] = &mag_log2_float16,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_FLOAT32] = &mag_sqr_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sqr_float16,
    },
    [MAG_OP_RCP] = {
        [MAG_DTYPE_FLOAT32] = &mag_rcp_float32,
        [MAG_DTYPE_FLOAT16] = &mag_rcp_float16,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_FLOAT32] = &mag_sqrt_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sqrt_float16,
    },
    [MAG_OP_RSQRT] = {
        [MAG_DTYPE_FLOAT32] = &mag_rsqrt_float32,
        [MAG_DTYPE_FLOAT16] = &mag_rsqrt_float16,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_FLOAT32] = &mag_sin_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sin_float16,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_FLOAT32] = &mag_cos_float32,
        [MAG_DTYPE_FLOAT16] = &mag_cos_float16,
    },
    [MAG_OP_TAN] = {
        [MAG_DTYPE_FLOAT32] = &mag_tan_float32,
        [MAG_DTYPE_FLOAT16] = &mag_tan_float16,
    },
    [MAG_OP_SINH] = {
        [MAG_DTYPE_FLOAT32] = &mag_sinh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sinh_float16,
    },
    [MAG_OP_COSH] = {
        [MAG_DTYPE_FLOAT32] = &mag_cosh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_cosh_float16,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_FLOAT32] = &mag_tanh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_tanh_float16,
    },
    [MAG_OP_ASIN] = {
        [MAG_DTYPE_FLOAT32] = &mag_asin_float32,
        [MAG_DTYPE_FLOAT16] = &mag_asin_float16,
    },
    [MAG_OP_ACOS] = {
        [MAG_DTYPE_FLOAT32] = &mag_acos_float32,
        [MAG_DTYPE_FLOAT16] = &mag_acos_float16,
    },
    [MAG_OP_ATAN] = {
        [MAG_DTYPE_FLOAT32] = &mag_atan_float32,
        [MAG_DTYPE_FLOAT16] = &mag_atan_float16,
    },
    [MAG_OP_ASINH] = {
        [MAG_DTYPE_FLOAT32] = &mag_asinh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_asinh_float16,
    },
    [MAG_OP_ACOSH] = {
        [MAG_DTYPE_FLOAT32] = &mag_acosh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_acosh_float16,
    },
    [MAG_OP_ATANH] = {
        [MAG_DTYPE_FLOAT32] = &mag_atanh_float32,
        [MAG_DTYPE_FLOAT16] = &mag_atanh_float16,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_FLOAT32] = &mag_step_float32,
        [MAG_DTYPE_FLOAT16] = &mag_step_float16,
    },
    [MAG_OP_ERF] = {
        [MAG_DTYPE_FLOAT32] = &mag_erf_float32,
        [MAG_DTYPE_FLOAT16] = &mag_erf_float16,
    },
    [MAG_OP_ERFC] = {
        [MAG_DTYPE_FLOAT32] = &mag_erfc_float32,
        [MAG_DTYPE_FLOAT16] = &mag_erfc_float16,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_FLOAT32] = &mag_exp_float32,
        [MAG_DTYPE_FLOAT16] = &mag_exp_float16,
    },
    [MAG_OP_EXP2] = {
        [MAG_DTYPE_FLOAT32] = &mag_exp2_float32,
        [MAG_DTYPE_FLOAT16] = &mag_exp2_float16,
    },
    [MAG_OP_EXPM1] = {
        [MAG_DTYPE_FLOAT32] = &mag_expm1_float32,
        [MAG_DTYPE_FLOAT16] = &mag_expm1_float16,
    },
    [MAG_OP_FLOOR] = {
        [MAG_DTYPE_FLOAT32] = &mag_floor_float32,
        [MAG_DTYPE_FLOAT16] = &mag_floor_float16,
    },
    [MAG_OP_CEIL] = {
        [MAG_DTYPE_FLOAT32] = &mag_ceil_float32,
        [MAG_DTYPE_FLOAT16] = &mag_ceil_float16,
    },
    [MAG_OP_ROUND] = {
        [MAG_DTYPE_FLOAT32] = &mag_round_float32,
        [MAG_DTYPE_FLOAT16] = &mag_round_float16,
    },
    [MAG_OP_TRUNC] = {
        [MAG_DTYPE_FLOAT32] = &mag_trunc_float32,
        [MAG_DTYPE_FLOAT16] = &mag_trunc_float16,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_FLOAT32] = &mag_softmax_float32,
        [MAG_DTYPE_FLOAT16] = &mag_softmax_float16,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_softmax_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_softmax_dv_float16,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_FLOAT32] = &mag_sigmoid_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sigmoid_float16,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_sigmoid_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sigmoid_dv_float16,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_FLOAT32] = &mag_hard_sigmoid_float32,
        [MAG_DTYPE_FLOAT16] = &mag_hard_sigmoid_float16,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_FLOAT32] = &mag_silu_float32,
        [MAG_DTYPE_FLOAT16] = &mag_silu_float16,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_silu_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_silu_dv_float16,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_tanh_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_tanh_dv_float16,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_FLOAT32] = &mag_relu_float32,
        [MAG_DTYPE_FLOAT16] = &mag_relu_float16,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_relu_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_relu_dv_float16,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_FLOAT32] = &mag_gelu_float32,
        [MAG_DTYPE_FLOAT16] = &mag_gelu_float16,
    },
    [MAG_OP_GELU_APPROX] = {
        [MAG_DTYPE_FLOAT32] = &mag_gelu_approx_float32,
        [MAG_DTYPE_FLOAT16] = &mag_gelu_approx_float16,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_FLOAT32] = &mag_gelu_dv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_gelu_dv_float16,
    },
    [MAG_OP_TRIL] = {
        [MAG_DTYPE_FLOAT32] = &mag_tril_float32,
        [MAG_DTYPE_FLOAT16] = &mag_tril_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_tril_uint8,
        [MAG_DTYPE_UINT8] = &mag_tril_uint8,
        [MAG_DTYPE_INT8] = &mag_tril_int8,
        [MAG_DTYPE_UINT16] = &mag_tril_uint16,
        [MAG_DTYPE_INT16] = &mag_tril_int16,
        [MAG_DTYPE_UINT32] = &mag_tril_uint32,
        [MAG_DTYPE_INT32] = &mag_tril_int32,
        [MAG_DTYPE_UINT64] = &mag_tril_uint64,
        [MAG_DTYPE_INT64] = &mag_tril_int64,
    },
    [MAG_OP_TRIU] = {
        [MAG_DTYPE_FLOAT32] = &mag_triu_float32,
        [MAG_DTYPE_FLOAT16] = &mag_triu_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_triu_uint8,
        [MAG_DTYPE_UINT8] = &mag_triu_uint8,
        [MAG_DTYPE_INT8] = &mag_triu_int8,
        [MAG_DTYPE_UINT16] = &mag_triu_uint16,
        [MAG_DTYPE_INT16] = &mag_triu_int16,
        [MAG_DTYPE_UINT32] = &mag_triu_uint32,
        [MAG_DTYPE_INT32] = &mag_triu_int32,
        [MAG_DTYPE_UINT64] = &mag_triu_uint64,
        [MAG_DTYPE_INT64] = &mag_triu_int64,
    },
    [MAG_OP_MULTINOMIAL] = {
        [MAG_DTYPE_FLOAT32] = &mag_multinomial_float32,
        [MAG_DTYPE_FLOAT16] = &mag_multinomial_float16,
    },
    [MAG_OP_CAT] = {
        [MAG_DTYPE_FLOAT32] = &mag_cat_float32,
        [MAG_DTYPE_FLOAT16] = &mag_cat_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_cat_uint8,
        [MAG_DTYPE_UINT8] = &mag_cat_uint8,
        [MAG_DTYPE_INT8] = &mag_cat_int8,
        [MAG_DTYPE_UINT16] = &mag_cat_uint16,
        [MAG_DTYPE_INT16] = &mag_cat_int16,
        [MAG_DTYPE_UINT32] = &mag_cat_uint32,
        [MAG_DTYPE_INT32] = &mag_cat_int32,
        [MAG_DTYPE_UINT64] = &mag_cat_uint64,
        [MAG_DTYPE_INT64] = &mag_cat_int64,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_FLOAT32] = &mag_add_float32,
        [MAG_DTYPE_FLOAT16] = &mag_add_float16,
        [MAG_DTYPE_UINT8] = &mag_add_uint8,
        [MAG_DTYPE_INT8] = &mag_add_int8,
        [MAG_DTYPE_UINT16] = &mag_add_uint16,
        [MAG_DTYPE_INT16] = &mag_add_int16,
        [MAG_DTYPE_UINT32] = &mag_add_uint32,
        [MAG_DTYPE_INT32] = &mag_add_int32,
        [MAG_DTYPE_UINT64] = &mag_add_uint64,
        [MAG_DTYPE_INT64] = &mag_add_int64,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_FLOAT32] = &mag_sub_float32,
        [MAG_DTYPE_FLOAT16] = &mag_sub_float16,
        [MAG_DTYPE_UINT8] = &mag_sub_uint8,
        [MAG_DTYPE_INT8] = &mag_sub_int8,
        [MAG_DTYPE_UINT16] = &mag_sub_uint16,
        [MAG_DTYPE_INT16] = &mag_sub_int16,
        [MAG_DTYPE_UINT32] = &mag_sub_uint32,
        [MAG_DTYPE_INT32] = &mag_sub_int32,
        [MAG_DTYPE_UINT64] = &mag_sub_uint64,
        [MAG_DTYPE_INT64] = &mag_sub_int64,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_FLOAT32] = &mag_mul_float32,
        [MAG_DTYPE_FLOAT16] = &mag_mul_float16,
        [MAG_DTYPE_UINT8] = &mag_mul_uint8,
        [MAG_DTYPE_INT8] = &mag_mul_int8,
        [MAG_DTYPE_UINT16] = &mag_mul_uint16,
        [MAG_DTYPE_INT16] = &mag_mul_int16,
        [MAG_DTYPE_UINT32] = &mag_mul_uint32,
        [MAG_DTYPE_INT32] = &mag_mul_int32,
        [MAG_DTYPE_UINT64] = &mag_mul_uint64,
        [MAG_DTYPE_INT64] = &mag_mul_int64,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_FLOAT32] = &mag_div_float32,
        [MAG_DTYPE_FLOAT16] = &mag_div_float16,
        [MAG_DTYPE_UINT8] = &mag_div_uint8,
        [MAG_DTYPE_INT8] = &mag_div_int8,
        [MAG_DTYPE_UINT16] = &mag_div_uint16,
        [MAG_DTYPE_INT16] = &mag_div_int16,
        [MAG_DTYPE_UINT32] = &mag_div_uint32,
        [MAG_DTYPE_INT32] = &mag_div_int32,
        [MAG_DTYPE_UINT64] = &mag_div_uint64,
        [MAG_DTYPE_INT64] = &mag_div_int64,
    },
    [MAG_OP_FLOORDIV] = {
        [MAG_DTYPE_FLOAT32] = &mag_floordiv_float32,
        [MAG_DTYPE_FLOAT16] = &mag_floordiv_float16,
        [MAG_DTYPE_UINT8] = &mag_floordiv_uint8,
        [MAG_DTYPE_INT8] = &mag_floordiv_int8,
        [MAG_DTYPE_UINT16] = &mag_floordiv_uint16,
        [MAG_DTYPE_INT16] = &mag_floordiv_int16,
        [MAG_DTYPE_UINT32] = &mag_floordiv_uint32,
        [MAG_DTYPE_INT32] = &mag_floordiv_int32,
        [MAG_DTYPE_UINT64] = &mag_floordiv_uint64,
        [MAG_DTYPE_INT64] = &mag_floordiv_int64,
    },
    [MAG_OP_MOD] = {
        [MAG_DTYPE_FLOAT32] = &mag_mod_float32,
        [MAG_DTYPE_FLOAT16] = &mag_mod_float16,
        [MAG_DTYPE_UINT8] = &mag_mod_uint8,
        [MAG_DTYPE_INT8] = &mag_mod_int8,
        [MAG_DTYPE_UINT16] = &mag_mod_uint16,
        [MAG_DTYPE_INT16] = &mag_mod_int16,
        [MAG_DTYPE_UINT32] = &mag_mod_uint32,
        [MAG_DTYPE_INT32] = &mag_mod_int32,
        [MAG_DTYPE_UINT64] = &mag_mod_uint64,
        [MAG_DTYPE_INT64] = &mag_mod_int64,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_FLOAT32] = &mag_matmul_float32,
        [MAG_DTYPE_FLOAT16] = &mag_matmul_float16,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_FLOAT32] = &mag_repeat_back_float32,
        [MAG_DTYPE_FLOAT16] = &mag_repeat_back_float16,
    },
    [MAG_OP_GATHER] = {
        [MAG_DTYPE_FLOAT32] = &mag_gather_float32,
        [MAG_DTYPE_FLOAT16] = &mag_gather_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_gather_uint8,
        [MAG_DTYPE_UINT8] = &mag_gather_uint8,
        [MAG_DTYPE_INT8] = &mag_gather_int8,
        [MAG_DTYPE_UINT16] = &mag_gather_uint16,
        [MAG_DTYPE_INT16] = &mag_gather_int16,
        [MAG_DTYPE_UINT32] = &mag_gather_uint32,
        [MAG_DTYPE_INT32] = &mag_gather_int32,
        [MAG_DTYPE_UINT64] = &mag_gather_uint64,
        [MAG_DTYPE_INT64] = &mag_gather_int64,
    },
    [MAG_OP_AND] = {
        [MAG_DTYPE_BOOLEAN] = &mag_and_uint8,
        [MAG_DTYPE_UINT8] = &mag_and_uint8,
        [MAG_DTYPE_INT8] = &mag_and_int8,
        [MAG_DTYPE_UINT16] = &mag_and_uint16,
        [MAG_DTYPE_INT16] = &mag_and_int16,
        [MAG_DTYPE_UINT32] = &mag_and_uint32,
        [MAG_DTYPE_INT32] = &mag_and_int32,
        [MAG_DTYPE_UINT64] = &mag_and_uint64,
        [MAG_DTYPE_INT64] = &mag_and_int64,
    },
    [MAG_OP_OR] = {
        [MAG_DTYPE_BOOLEAN] = &mag_or_uint8,
        [MAG_DTYPE_UINT8] = &mag_or_uint8,
        [MAG_DTYPE_INT8] = &mag_or_int8,
        [MAG_DTYPE_UINT16] = &mag_or_uint16,
        [MAG_DTYPE_INT16] = &mag_or_int16,
        [MAG_DTYPE_UINT32] = &mag_or_uint32,
        [MAG_DTYPE_INT32] = &mag_or_int32,
        [MAG_DTYPE_UINT64] = &mag_or_uint64,
        [MAG_DTYPE_INT64] = &mag_or_int64,
    },
    [MAG_OP_XOR] = {
        [MAG_DTYPE_BOOLEAN] = &mag_xor_uint8,
        [MAG_DTYPE_UINT8] = &mag_xor_uint8,
        [MAG_DTYPE_INT8] = &mag_xor_int8,
        [MAG_DTYPE_UINT16] = &mag_xor_uint16,
        [MAG_DTYPE_INT16] = &mag_xor_int16,
        [MAG_DTYPE_UINT32] = &mag_xor_uint32,
        [MAG_DTYPE_INT32] = &mag_xor_int32,
        [MAG_DTYPE_UINT64] = &mag_xor_uint64,
        [MAG_DTYPE_INT64] = &mag_xor_int64,
    },
    [MAG_OP_NOT] = {
        [MAG_DTYPE_BOOLEAN] = &mag_not_uint8,
        [MAG_DTYPE_UINT8] = &mag_not_uint8,
        [MAG_DTYPE_INT8] = &mag_not_int8,
        [MAG_DTYPE_UINT16] = &mag_not_uint16,
        [MAG_DTYPE_INT16] = &mag_not_int16,
        [MAG_DTYPE_UINT32] = &mag_not_uint32,
        [MAG_DTYPE_INT32] = &mag_not_int32,
        [MAG_DTYPE_UINT64] = &mag_not_uint64,
        [MAG_DTYPE_INT64] = &mag_not_int64,
    },
    [MAG_OP_SHL] = {
        [MAG_DTYPE_UINT8] = &mag_shl_uint8,
        [MAG_DTYPE_INT8] = &mag_shl_int8,
        [MAG_DTYPE_UINT16] = &mag_shl_uint16,
        [MAG_DTYPE_INT16] = &mag_shl_int16,
        [MAG_DTYPE_UINT32] = &mag_shl_uint32,
        [MAG_DTYPE_INT32] = &mag_shl_int32,
        [MAG_DTYPE_UINT64] = &mag_shl_uint64,
        [MAG_DTYPE_INT64] = &mag_shl_int64,
    },
    [MAG_OP_SHR] = {
        [MAG_DTYPE_UINT8] = &mag_shr_uint8,
        [MAG_DTYPE_INT8] = &mag_shr_int8,
        [MAG_DTYPE_UINT16] = &mag_shr_uint16,
        [MAG_DTYPE_INT16] = &mag_shr_int16,
        [MAG_DTYPE_UINT32] = &mag_shr_uint32,
        [MAG_DTYPE_INT32] = &mag_shr_int32,
        [MAG_DTYPE_UINT64] = &mag_shr_uint64,
        [MAG_DTYPE_INT64] = &mag_shr_int64,
    },
    [MAG_OP_EQ] = {
        [MAG_DTYPE_FLOAT32] = &mag_eq_float32,
        [MAG_DTYPE_FLOAT16] = &mag_eq_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_eq_uint8,
        [MAG_DTYPE_UINT8] = &mag_eq_uint8,
        [MAG_DTYPE_INT8] = &mag_eq_int8,
        [MAG_DTYPE_UINT16] = &mag_eq_uint16,
        [MAG_DTYPE_INT16] = &mag_eq_int16,
        [MAG_DTYPE_UINT32] = &mag_eq_uint32,
        [MAG_DTYPE_INT32] = &mag_eq_int32,
        [MAG_DTYPE_UINT64] = &mag_eq_uint64,
        [MAG_DTYPE_INT64] = &mag_eq_int64,
    },
    [MAG_OP_NE] = {
        [MAG_DTYPE_FLOAT32] = &mag_ne_float32,
        [MAG_DTYPE_FLOAT16] = &mag_ne_float16,
        [MAG_DTYPE_BOOLEAN] = &mag_ne_uint8,
        [MAG_DTYPE_UINT8] = &mag_ne_uint8,
        [MAG_DTYPE_INT8] = &mag_ne_int8,
        [MAG_DTYPE_UINT16] = &mag_ne_uint16,
        [MAG_DTYPE_INT16] = &mag_ne_int16,
        [MAG_DTYPE_UINT32] = &mag_ne_uint32,
        [MAG_DTYPE_INT32] = &mag_ne_int32,
        [MAG_DTYPE_UINT64] = &mag_ne_uint64,
        [MAG_DTYPE_INT64] = &mag_ne_int64,
    },
    [MAG_OP_LE] = {
        [MAG_DTYPE_FLOAT32] = &mag_le_float32,
        [MAG_DTYPE_FLOAT16] = &mag_le_float16,
        [MAG_DTYPE_UINT8] = &mag_le_uint8,
        [MAG_DTYPE_INT8] = &mag_le_int8,
        [MAG_DTYPE_UINT16] = &mag_le_uint16,
        [MAG_DTYPE_INT16] = &mag_le_int16,
        [MAG_DTYPE_UINT32] = &mag_le_uint32,
        [MAG_DTYPE_INT32] = &mag_le_int32,
        [MAG_DTYPE_UINT64] = &mag_le_uint64,
        [MAG_DTYPE_INT64] = &mag_le_int64,
    },
    [MAG_OP_GE] = {
        [MAG_DTYPE_FLOAT32] = &mag_ge_float32,
        [MAG_DTYPE_FLOAT16] = &mag_ge_float16,
        [MAG_DTYPE_UINT8] = &mag_ge_uint8,
        [MAG_DTYPE_INT8] = &mag_ge_int8,
        [MAG_DTYPE_UINT16] = &mag_ge_uint16,
        [MAG_DTYPE_INT16] = &mag_ge_int16,
        [MAG_DTYPE_UINT32] = &mag_ge_uint32,
        [MAG_DTYPE_INT32] = &mag_ge_int32,
        [MAG_DTYPE_UINT64] = &mag_ge_uint64,
        [MAG_DTYPE_INT64] = &mag_ge_int64,
    },
    [MAG_OP_LT] = {
        [MAG_DTYPE_FLOAT32] = &mag_lt_float32,
        [MAG_DTYPE_FLOAT16] = &mag_lt_float16,
        [MAG_DTYPE_UINT8] = &mag_lt_uint8,
        [MAG_DTYPE_INT8] = &mag_lt_int8,
        [MAG_DTYPE_UINT16] = &mag_lt_uint16,
        [MAG_DTYPE_INT16] = &mag_lt_int16,
        [MAG_DTYPE_UINT32] = &mag_lt_uint32,
        [MAG_DTYPE_INT32] = &mag_lt_int32,
        [MAG_DTYPE_UINT64] = &mag_lt_uint64,
        [MAG_DTYPE_INT64] = &mag_lt_int64,
    },
    [MAG_OP_GT] = {
        [MAG_DTYPE_FLOAT32] = &mag_gt_float32,
        [MAG_DTYPE_FLOAT16] = &mag_gt_float16,
        [MAG_DTYPE_UINT8] = &mag_gt_uint8,
        [MAG_DTYPE_INT8] = &mag_gt_int8,
        [MAG_DTYPE_UINT16] = &mag_gt_uint16,
        [MAG_DTYPE_INT16] = &mag_gt_int16,
        [MAG_DTYPE_UINT32] = &mag_gt_uint32,
        [MAG_DTYPE_INT32] = &mag_gt_int32,
        [MAG_DTYPE_UINT64] = &mag_gt_uint64,
        [MAG_DTYPE_INT64] = &mag_gt_int64,
    },
};

static size_t mag_vreg_width(void) {
    return MAG_VREG_WIDTH;
}

static void mag_impl_init(void) {

}

static void mag_impl_deinit(void) {
    mag_sb_release();
}

void MAG_BLAS_SPECIALIZATION(mag_kernel_registry_t *kernels) {
    kernels->init = &mag_impl_init;
    kernels->deinit = &mag_impl_deinit;
    for (int i=0; i < MAG_OP__NUM; ++i) {
        for (int j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->operators[i][j] = mag_lut_eval_kernels[i][j];
        }
    }
    kernels->vreg_width = &mag_vreg_width;
}

#ifndef MAG_BLAS_SPECIALIZATION
#error "BLAS specialization undefined"
#endif
#ifndef MAG_BLAS_SPECIALIZATION_FEAT_REQUEST
#error "Feature request routine undefined"
#endif

#if defined(__x86_64__) || defined(_M_X64)
/*
** x86-64 specific feature detection.
** This function is always called, so it must run down to SSE2 at least.
** This means that there should be no fancy instructions or extensions.
** There was a bug where the backend with Intel APX enabled used
** the instruction: pushp %rbp (d5 08 55) for function prologue, whis is Intel APX and crashes on older CPUs.
** This is why this function should really only return one single integer scalar in the return register, according to the calling convention,
** and NO other code or logic. The function is marked naked to supress the prologue/epilogue generation and associated extension instructions.
*/
mag_amd64_cap_bitset_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST() {
    mag_amd64_cap_bitset_t caps = 0;
#ifdef __SSE__
    caps|=mag_amd64_cap(SSE);
#endif
#ifdef __SSE2__
    caps|=mag_amd64_cap(SSE2);
#endif
#ifdef __SSE3__
    caps|=mag_amd64_cap(SSE3);
#endif
#ifdef __SSSE3__
    caps|=mag_amd64_cap(SSSE3);
#endif
#ifdef __SSE4_1__
    caps|=mag_amd64_cap(SSE41);
#endif
#ifdef __SSE4_2__
    caps|=mag_amd64_cap(SSE42);
#endif
#ifdef __SSE4A__
    caps|=mag_amd64_cap(SSE4A);
#endif

#ifdef __AVX__
    caps|=mag_amd64_cap(AVX);
#endif
#ifdef __FMA__
    caps|=mag_amd64_cap(FMA);
#endif
#ifdef __AVX2__
    caps|=mag_amd64_cap(AVX2);
#endif
#ifdef __F16C__
    caps|=mag_amd64_cap(F16C);
#endif
#ifdef __AVXVNNI__
    caps|=mag_amd64_cap(AVX_VNNI);
#endif
#ifdef __AVXVNNIINT8__
    caps|=mag_amd64_cap(AVX_VNNI_INT8);
#endif
#ifdef __AVXNECONVERT__
    caps|=mag_amd64_cap(AVX_NE_CONVERT);
#endif
#ifdef __AVXIFMA__
    caps|=mag_amd64_cap(AVX_IFMA);
#endif
#ifdef __AVXVNNIINT16__
    caps|=mag_amd64_cap(AVX_VNNI_INT16);
#endif
#ifdef __AVX10__
    caps|=mag_amd64_cap(AVX10);
#endif

#ifdef __AVX512F__
    caps|=mag_amd64_cap(AVX512_F);
#endif
#ifdef __AVX512DQ__
    caps|=mag_amd64_cap(AVX512_DQ);
#endif
#ifdef __AVX512IFMA__
    caps|=mag_amd64_cap(AVX512_IFMA);
#endif
#ifdef __AVX512PF__
    caps|=mag_amd64_cap(AVX512_PF);
#endif
#ifdef __AVX512ER__
    caps|=mag_amd64_cap(AVX512_ER);
#endif
#ifdef __AVX512CD__
    caps|=mag_amd64_cap(AVX512_CD);
#endif
#ifdef __AVX512BW__
    caps|=mag_amd64_cap(AVX512_BW);
#endif
#ifdef __AVX512VL__
    caps|=mag_amd64_cap(AVX512_VL);
#endif
#ifdef __AVX512VBMI__
    caps|=mag_amd64_cap(AVX512_VBMI);
#endif
#ifdef __AVX5124VNNIW__
    caps|=mag_amd64_cap(AVX512_4VNNIW);
#endif
#ifdef __AVX5124FMAPS__
    caps|=mag_amd64_cap(AVX512_4FMAPS);
#endif
#ifdef __AVX512VBMI2__
    caps|=mag_amd64_cap(AVX512_VBMI2);
#endif
#ifdef __AVX512VNNI__
    caps|=mag_amd64_cap(AVX512_VNNI);
#endif
#ifdef __AVX512BITALG__
    caps|=mag_amd64_cap(AVX512_BITALG);
#endif
#ifdef __AVX512VPOPCNTDQ__
    caps|=mag_amd64_cap(AVX512_VPOPCNTDQ);
#endif
#ifdef __AVX512BF16__
    caps|=mag_amd64_cap(AVX512_BF16);
#endif
#ifdef __AVX512VP2INTERSECT__
    caps|=mag_amd64_cap(AVX512_VP2INTERSECT);
#endif
#ifdef __AVX512FP16__
    caps|=mag_amd64_cap(AVX512_FP16);
#endif

#ifdef __AMX_TILE__
    caps|=mag_amd64_cap(AMX_TILE);
#endif
#ifdef __AMX_INT8__
    caps|=mag_amd64_cap(AMX_INT8);
#endif
#ifdef __AMX_BF16__
    caps|=mag_amd64_cap(AMX_BF16);
#endif
#ifdef __AMX_FP16__
    caps|=mag_amd64_cap(AMX_FP16);
#endif
#ifdef __AMX_TRANSPOSE__
    caps|=mag_amd64_cap(AMX_TRANSPOSE);
#endif
#ifdef __AMX_TF32__
    caps|=mag_amd64_cap(AMX_TF32);
#endif
#ifdef __AMX_AVX512__
    caps|=mag_amd64_cap(AMX_AVX512);
#endif
#ifdef __AMX_MOVRS__
    caps|=mag_amd64_cap(AMX_MOVRS);
#endif
#ifdef __AMX_FP8__
    caps|=mag_amd64_cap(AMX_FP8);
#endif


#ifdef __BMI__
    caps|=mag_amd64_cap(BMI1);
#endif
#ifdef __BMI2__
    caps|=mag_amd64_cap(BMI2);
#endif

#ifdef __GFNI__
    caps|=mag_amd64_cap(GFNI);
#endif
#ifdef __APXF__
    caps|=mag_amd64_cap(APX_F);
#endif

    return caps;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

mag_arm64_cap_bitset_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST(void) {
    mag_arm64_cap_bitset_t caps = 0;
#ifdef __ARM_NEON
    caps|=mag_arm64_cap(NEON);
#endif
#ifdef __ARM_FEATURE_DOTPROD
    caps|=mag_arm64_cap(DOTPROD);
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
    caps|=mag_arm64_cap(I8MM);
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    caps|=mag_arm64_cap(F16VECTOR);
    caps|=mag_arm64_cap(F16SCALAR);
    caps|=mag_arm64_cap(F16CVT);
#elif defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    caps|=mag_arm64_cap(F16SCALAR);
    caps|=mag_arm64_cap(F16CVT);
#endif
#ifdef __ARM_FEATURE_BF16
    caps|=mag_arm64_cap(BF16);
#endif
#ifdef __ARM_FEATURE_SVE
    caps|=mag_arm64_cap(SVE);
#endif
#ifdef __ARM_FEATURE_SVE2
    caps|=mag_arm64_cap(SVE2);
#endif
    return caps;
}

#endif
