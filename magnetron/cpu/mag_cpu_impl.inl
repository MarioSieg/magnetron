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

#include "mag_cpu.h"

#include <core/mag_tensor.h>
#include <core/mag_cpuid.h>
#include <core/mag_alloc.h>
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
#define mag_cvt_i642bool(x) (!!(x))
#define mag_cvt_i642i32(x) ((int32_t)(x))

#define mag_cmd_in(i) (payload->cmd->in[(i)])
#define mag_cmd_out(i) (payload->cmd->out[(i)])
#define mag_cmd_attr(i) (payload->cmd->attrs[(i)])

/* Uniform names for macro expansion */
typedef uint8_t mag_u8_t;
typedef int8_t mag_i8_t;
typedef uint16_t mag_u16_t;
typedef int16_t mag_i16_t;
typedef uint32_t mag_u32_t;
typedef int32_t mag_i32_t;
typedef uint64_t mag_u64_t;
typedef int64_t mag_i64_t;
typedef uint8_t mag_bool_t;

#define mag_e8m23p(t) ((const mag_e8m23_t*)mag_tensor_get_data_ptr(t))
#define mag_e8m23p_mut(t) ((mag_e8m23_t*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p(t) ((const mag_e5m10_t*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p_mut(t) ((mag_e5m10_t*)mag_tensor_get_data_ptr(t))
#define mag_boolp(t) ((const mag_bool_t*)mag_tensor_get_data_ptr(t))
#define mag_boolp_mut(t) ((mag_bool_t*)mag_tensor_get_data_ptr(t))
#define mag_u8p(t) ((const uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_u8p_mut(t) ((uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_i8p(t) ((const int8_t*)mag_tensor_get_data_ptr(t))
#define mag_i8p_mut(t) ((int8_t*)mag_tensor_get_data_ptr(t))
#define mag_u16p(t) ((const uint16_t*)mag_tensor_get_data_ptr(t))
#define mag_u16p_mut(t) ((uint16_t*)mag_tensor_get_data_ptr(t))
#define mag_i16p(t) ((const int16_t*)mag_tensor_get_data_ptr(t))
#define mag_i16p_mut(t) ((int16_t*)mag_tensor_get_data_ptr(t))
#define mag_u32p(t) ((const uint32_t*)mag_tensor_get_data_ptr(t))
#define mag_u32p_mut(t) ((uint32_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p(t) ((const int32_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p_mut(t) ((int32_t*)mag_tensor_get_data_ptr(t))
#define mag_u64p(t) ((const uint64_t*)mag_tensor_get_data_ptr(t))
#define mag_u64p_mut(t) ((uint64_t*)mag_tensor_get_data_ptr(t))
#define mag_i64p(t) ((const int64_t*)mag_tensor_get_data_ptr(t))
#define mag_i64p_mut(t) ((int64_t*)mag_tensor_get_data_ptr(t))

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
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_FILL] = {
        [MAG_DTYPE_E8M23] = &mag_fill_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_e5m10,
        [MAG_DTYPE_BOOL] = &mag_fill_bool,
        [MAG_DTYPE_U8] = &mag_fill_u8,
        [MAG_DTYPE_I8] = &mag_fill_i8,
        [MAG_DTYPE_U16] = &mag_fill_u16,
        [MAG_DTYPE_I16] = &mag_fill_i16,
        [MAG_DTYPE_U32] = &mag_fill_u32,
        [MAG_DTYPE_I32] = &mag_fill_i32,
        [MAG_DTYPE_U64] = &mag_fill_u64,
        [MAG_DTYPE_I64] = &mag_fill_i64,
    },
    [MAG_OP_MASKED_FILL] = {
        [MAG_DTYPE_E8M23] = &mag_masked_fill_e8m23,
        [MAG_DTYPE_E5M10] = &mag_masked_fill_e5m10,
        [MAG_DTYPE_BOOL] = &mag_masked_fill_bool,
        [MAG_DTYPE_U8]  = &mag_masked_fill_u8,
        [MAG_DTYPE_I8]  = &mag_masked_fill_i8,
        [MAG_DTYPE_U16] = &mag_masked_fill_u16,
        [MAG_DTYPE_I16] = &mag_masked_fill_i16,
        [MAG_DTYPE_U32] = &mag_masked_fill_u32,
        [MAG_DTYPE_I32] = &mag_masked_fill_i32,
        [MAG_DTYPE_U64] = &mag_masked_fill_u64,
        [MAG_DTYPE_I64] = &mag_masked_fill_i64,
    },
    [MAG_OP_RAND_UNIFORM] = {
        [MAG_DTYPE_E8M23] = &mag_fill_rand_uniform_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_rand_uniform_e5m10,
        [MAG_DTYPE_I32] = &mag_fill_rand_uniform_i32
    },
    [MAG_OP_RAND_NORMAL] = {
        [MAG_DTYPE_E8M23] = &mag_fill_rand_normal_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_rand_normal_e5m10,
    },
    [MAG_OP_RAND_BERNOULLI] = {
        [MAG_DTYPE_BOOL] = &mag_fill_rand_bernoulli_bool,
    },
    [MAG_OP_ARANGE] = {
        [MAG_DTYPE_E8M23] = &mag_fill_arange_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_arange_e5m10,
        [MAG_DTYPE_U8] = &mag_fill_arange_u8,
        [MAG_DTYPE_I8] = &mag_fill_arange_i8,
        [MAG_DTYPE_U16] = &mag_fill_arange_u16,
        [MAG_DTYPE_I16] = &mag_fill_arange_i16,
        [MAG_DTYPE_U32] = &mag_fill_arange_u32,
        [MAG_DTYPE_I32] = &mag_fill_arange_i32,
        [MAG_DTYPE_U64] = &mag_fill_arange_u64,
        [MAG_DTYPE_I64] = &mag_fill_arange_i64,
    },
    [MAG_OP_CAST] = {
        [MAG_DTYPE_E8M23] = &mag_cast_generic,
        [MAG_DTYPE_E5M10] = &mag_cast_generic,
        [MAG_DTYPE_BOOL] = &mag_cast_generic,
        [MAG_DTYPE_U8] = &mag_cast_generic,
        [MAG_DTYPE_I8] = &mag_cast_generic,
        [MAG_DTYPE_U16] = &mag_cast_generic,
        [MAG_DTYPE_I16] = &mag_cast_generic,
        [MAG_DTYPE_U32] = &mag_cast_generic,
        [MAG_DTYPE_I32] = &mag_cast_generic,
        [MAG_DTYPE_U64] = &mag_cast_generic,
        [MAG_DTYPE_I64] = &mag_cast_generic,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_clone_e8m23,
        [MAG_DTYPE_E5M10] = &mag_clone_e5m10,
        [MAG_DTYPE_BOOL] = &mag_clone_bool,
        [MAG_DTYPE_U8] = &mag_clone_u8,
        [MAG_DTYPE_I8] = &mag_clone_i8,
        [MAG_DTYPE_U16] = &mag_clone_u16,
        [MAG_DTYPE_I16] = &mag_clone_i16,
        [MAG_DTYPE_U32] = &mag_clone_u32,
        [MAG_DTYPE_I32] = &mag_clone_i32,
        [MAG_DTYPE_U64] = &mag_clone_u64,
        [MAG_DTYPE_I64] = &mag_clone_i64,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_E8M23] = &mag_mean_e8m23,
        [MAG_DTYPE_E5M10] = &mag_mean_e5m10,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_min_e8m23,
        [MAG_DTYPE_E5M10] = &mag_min_e5m10,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_max_e8m23,
        [MAG_DTYPE_E5M10] = &mag_max_e5m10,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_sum_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sum_e5m10,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_E8M23] = &mag_abs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_abs_e5m10,
    },
    [MAG_OP_SGN] = {
        [MAG_DTYPE_E8M23] = &mag_sgn_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sgn_e5m10,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_E8M23] = &mag_neg_e8m23,
        [MAG_DTYPE_E5M10] = &mag_neg_e5m10,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_E8M23] = &mag_log_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log_e5m10,
    },
    [MAG_OP_LOG10] = {
        [MAG_DTYPE_E8M23] = &mag_log10_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log10_e5m10,
    },
    [MAG_OP_LOG1P] = {
        [MAG_DTYPE_E8M23] = &mag_log1p_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log1p_e5m10,
    },
    [MAG_OP_LOG2] = {
        [MAG_DTYPE_E8M23] = &mag_log2_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log2_e5m10,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_E8M23] = &mag_sqr_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sqr_e5m10,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_E8M23] = &mag_sqrt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sqrt_e5m10,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_E8M23] = &mag_sin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sin_e5m10,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_E8M23] = &mag_cos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cos_e5m10,
    },
    [MAG_OP_TAN] = {
        [MAG_DTYPE_E8M23] = &mag_tan_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tan_e5m10,
    },
    [MAG_OP_SINH] = {
        [MAG_DTYPE_E8M23] = &mag_sinh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sinh_e5m10,
    },
    [MAG_OP_COSH] = {
        [MAG_DTYPE_E8M23] = &mag_cosh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cosh_e5m10,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_E8M23] = &mag_tanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tanh_e5m10,
    },
    [MAG_OP_ASIN] = {
        [MAG_DTYPE_E8M23] = &mag_asin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_asin_e5m10,
    },
    [MAG_OP_ACOS] = {
        [MAG_DTYPE_E8M23] = &mag_acos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_acos_e5m10,
    },
    [MAG_OP_ATAN] = {
        [MAG_DTYPE_E8M23] = &mag_atan_e8m23,
        [MAG_DTYPE_E5M10] = &mag_atan_e5m10,
    },
    [MAG_OP_ASINH] = {
        [MAG_DTYPE_E8M23] = &mag_asinh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_asinh_e5m10,
    },
    [MAG_OP_ACOSH] = {
        [MAG_DTYPE_E8M23] = &mag_acosh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_acosh_e5m10,
    },
    [MAG_OP_ATANH] = {
        [MAG_DTYPE_E8M23] = &mag_atanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_atanh_e5m10,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_E8M23] = &mag_step_e8m23,
        [MAG_DTYPE_E5M10] = &mag_step_e5m10,
    },
    [MAG_OP_ERF] = {
        [MAG_DTYPE_E8M23] = &mag_erf_e8m23,
        [MAG_DTYPE_E5M10] = &mag_erf_e5m10,
    },
    [MAG_OP_ERFC] = {
        [MAG_DTYPE_E8M23] = &mag_erfc_e8m23,
        [MAG_DTYPE_E5M10] = &mag_erfc_e5m10,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_E8M23] = &mag_exp_e8m23,
        [MAG_DTYPE_E5M10] = &mag_exp_e5m10,
    },
    [MAG_OP_EXP2] = {
        [MAG_DTYPE_E8M23] = &mag_exp2_e8m23,
        [MAG_DTYPE_E5M10] = &mag_exp2_e5m10,
    },
    [MAG_OP_EXPM1] = {
        [MAG_DTYPE_E8M23] = &mag_expm1_e8m23,
        [MAG_DTYPE_E5M10] = &mag_expm1_e5m10,
    },
    [MAG_OP_FLOOR] = {
        [MAG_DTYPE_E8M23] = &mag_floor_e8m23,
        [MAG_DTYPE_E5M10] = &mag_floor_e5m10,
    },
    [MAG_OP_CEIL] = {
        [MAG_DTYPE_E8M23] = &mag_ceil_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ceil_e5m10,
    },
    [MAG_OP_ROUND] = {
        [MAG_DTYPE_E8M23] = &mag_round_e8m23,
        [MAG_DTYPE_E5M10] = &mag_round_e5m10,
    },
    [MAG_OP_TRUNC] = {
        [MAG_DTYPE_E8M23] = &mag_trunc_e8m23,
        [MAG_DTYPE_E5M10] = &mag_trunc_e5m10,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_E8M23] = &mag_softmax_e8m23,
        [MAG_DTYPE_E5M10] = &mag_softmax_e5m10,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_E8M23] = &mag_softmax_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_softmax_dv_e5m10,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sigmoid_e5m10,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_E8M23] = &mag_sigmoid_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sigmoid_dv_e5m10,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_hard_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_hard_sigmoid_e5m10,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_E8M23] = &mag_silu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_silu_e5m10,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_silu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_silu_dv_e5m10,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_E8M23] = &mag_tanh_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tanh_dv_e5m10,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_E8M23] = &mag_relu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_relu_e5m10,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_relu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_relu_dv_e5m10,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_e5m10,
    },
    [MAG_OP_GELU_APPROX] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_approx_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_approx_e5m10,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_dv_e5m10,
    },
    [MAG_OP_TRIL] = {
        [MAG_DTYPE_E8M23] = &mag_tril_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tril_e5m10,
        [MAG_DTYPE_BOOL] = &mag_tril_bool,
        [MAG_DTYPE_U8]  = &mag_tril_u8,
        [MAG_DTYPE_I8]  = &mag_tril_i8,
        [MAG_DTYPE_U16] = &mag_tril_u16,
        [MAG_DTYPE_I16] = &mag_tril_i16,
        [MAG_DTYPE_U32] = &mag_tril_u32,
        [MAG_DTYPE_I32] = &mag_tril_i32,
        [MAG_DTYPE_U64] = &mag_tril_u64,
        [MAG_DTYPE_I64] = &mag_tril_i64,
    },
    [MAG_OP_TRIU] = {
        [MAG_DTYPE_E8M23] = &mag_triu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_triu_e5m10,
        [MAG_DTYPE_BOOL] = &mag_triu_bool,
        [MAG_DTYPE_U8]  = &mag_triu_u8,
        [MAG_DTYPE_I8]  = &mag_triu_i8,
        [MAG_DTYPE_U16] = &mag_triu_u16,
        [MAG_DTYPE_I16] = &mag_triu_i16,
        [MAG_DTYPE_U32] = &mag_triu_u32,
        [MAG_DTYPE_I32] = &mag_triu_i32,
        [MAG_DTYPE_U64] = &mag_triu_u64,
        [MAG_DTYPE_I64] = &mag_triu_i64,
    },
    [MAG_OP_MULTINOMIAL] = {
        [MAG_DTYPE_E8M23] = &mag_multinomial_e8m23,
        [MAG_DTYPE_E5M10] = &mag_multinomial_e5m10,
    },
    [MAG_OP_CAT] = {
        [MAG_DTYPE_E8M23] = &mag_cat_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cat_e5m10,
        [MAG_DTYPE_BOOL] = &mag_cat_bool,
        [MAG_DTYPE_U8]  = &mag_cat_u8,
        [MAG_DTYPE_I8]  = &mag_cat_i8,
        [MAG_DTYPE_U16] = &mag_cat_u16,
        [MAG_DTYPE_I16] = &mag_cat_i16,
        [MAG_DTYPE_U32] = &mag_cat_u32,
        [MAG_DTYPE_I32] = &mag_cat_i32,
        [MAG_DTYPE_U64] = &mag_cat_u64,
        [MAG_DTYPE_I64] = &mag_cat_i64,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_E8M23] = &mag_add_e8m23,
        [MAG_DTYPE_E5M10] = &mag_add_e5m10,
        [MAG_DTYPE_U8]  = &mag_add_u8,
        [MAG_DTYPE_I8]  = &mag_add_i8,
        [MAG_DTYPE_U16] = &mag_add_u16,
        [MAG_DTYPE_I16] = &mag_add_i16,
        [MAG_DTYPE_U32] = &mag_add_u32,
        [MAG_DTYPE_I32] = &mag_add_i32,
        [MAG_DTYPE_U64] = &mag_add_u64,
        [MAG_DTYPE_I64] = &mag_add_i64,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_E8M23] = &mag_sub_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sub_e5m10,
        [MAG_DTYPE_U8]  = &mag_sub_u8,
        [MAG_DTYPE_I8]  = &mag_sub_i8,
        [MAG_DTYPE_U16] = &mag_sub_u16,
        [MAG_DTYPE_I16] = &mag_sub_i16,
        [MAG_DTYPE_U32] = &mag_sub_u32,
        [MAG_DTYPE_I32] = &mag_sub_i32,
        [MAG_DTYPE_U64] = &mag_sub_u64,
        [MAG_DTYPE_I64] = &mag_sub_i64,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_E8M23] = &mag_mul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_mul_e5m10,
        [MAG_DTYPE_U8]  = &mag_mul_u8,
        [MAG_DTYPE_I8]  = &mag_mul_i8,
        [MAG_DTYPE_U16] = &mag_mul_u16,
        [MAG_DTYPE_I16] = &mag_mul_i16,
        [MAG_DTYPE_U32] = &mag_mul_u32,
        [MAG_DTYPE_I32] = &mag_mul_i32,
        [MAG_DTYPE_U64] = &mag_mul_u64,
        [MAG_DTYPE_I64] = &mag_mul_i64,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_E8M23] = &mag_div_e8m23,
        [MAG_DTYPE_E5M10] = &mag_div_e5m10,
        [MAG_DTYPE_U8]  = &mag_div_u8,
        [MAG_DTYPE_I8]  = &mag_div_i8,
        [MAG_DTYPE_U16] = &mag_div_u16,
        [MAG_DTYPE_I16] = &mag_div_i16,
        [MAG_DTYPE_U32] = &mag_div_u32,
        [MAG_DTYPE_I32] = &mag_div_i32,
        [MAG_DTYPE_U64] = &mag_div_u64,
        [MAG_DTYPE_I64] = &mag_div_i64,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_matmul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_matmul_e5m10,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_repeat_back_e5m10,
    },
    [MAG_OP_GATHER] = {
        [MAG_DTYPE_E8M23] = &mag_gather_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gather_e5m10,
        [MAG_DTYPE_BOOL] = &mag_gather_bool,
        [MAG_DTYPE_U8]  = &mag_gather_u8,
        [MAG_DTYPE_I8]  = &mag_gather_i8,
        [MAG_DTYPE_U16] = &mag_gather_u16,
        [MAG_DTYPE_I16] = &mag_gather_i16,
        [MAG_DTYPE_U32] = &mag_gather_u32,
        [MAG_DTYPE_I32] = &mag_gather_i32,
        [MAG_DTYPE_U64] = &mag_gather_u64,
        [MAG_DTYPE_I64] = &mag_gather_i64,
    },
    [MAG_OP_AND] = {
        [MAG_DTYPE_BOOL] = &mag_and_bool,
        [MAG_DTYPE_U8]  = &mag_and_u8,
        [MAG_DTYPE_I8]  = &mag_and_i8,
        [MAG_DTYPE_U16] = &mag_and_u16,
        [MAG_DTYPE_I16] = &mag_and_i16,
        [MAG_DTYPE_U32] = &mag_and_u32,
        [MAG_DTYPE_I32] = &mag_and_i32,
        [MAG_DTYPE_U64] = &mag_and_u64,
        [MAG_DTYPE_I64] = &mag_and_i64,
    },
    [MAG_OP_OR] = {
        [MAG_DTYPE_BOOL] = &mag_or_bool,
        [MAG_DTYPE_U8]  = &mag_or_u8,
        [MAG_DTYPE_I8]  = &mag_or_i8,
        [MAG_DTYPE_U16] = &mag_or_u16,
        [MAG_DTYPE_I16] = &mag_or_i16,
        [MAG_DTYPE_U32] = &mag_or_u32,
        [MAG_DTYPE_I32] = &mag_or_i32,
        [MAG_DTYPE_U64] = &mag_or_u64,
        [MAG_DTYPE_I64] = &mag_or_i64,
    },
    [MAG_OP_XOR] = {
        [MAG_DTYPE_BOOL] = &mag_xor_bool,
        [MAG_DTYPE_U8]  = &mag_xor_u8,
        [MAG_DTYPE_I8]  = &mag_xor_i8,
        [MAG_DTYPE_U16] = &mag_xor_u16,
        [MAG_DTYPE_I16] = &mag_xor_i16,
        [MAG_DTYPE_U32] = &mag_xor_u32,
        [MAG_DTYPE_I32] = &mag_xor_i32,
        [MAG_DTYPE_U64] = &mag_xor_u64,
        [MAG_DTYPE_I64] = &mag_xor_i64,
    },
    [MAG_OP_NOT] = {
        [MAG_DTYPE_BOOL] = &mag_not_bool,
        [MAG_DTYPE_U8]  = &mag_not_u8,
        [MAG_DTYPE_I8]  = &mag_not_i8,
        [MAG_DTYPE_U16] = &mag_not_u16,
        [MAG_DTYPE_I16] = &mag_not_i16,
        [MAG_DTYPE_U32] = &mag_not_u32,
        [MAG_DTYPE_I32] = &mag_not_i32,
        [MAG_DTYPE_U64] = &mag_not_u64,
        [MAG_DTYPE_I64] = &mag_not_i64,
    },
    [MAG_OP_SHL] = {
        [MAG_DTYPE_U8]  = &mag_shl_u8,
        [MAG_DTYPE_I8]  = &mag_shl_i8,
        [MAG_DTYPE_U16] = &mag_shl_u16,
        [MAG_DTYPE_I16] = &mag_shl_i16,
        [MAG_DTYPE_U32] = &mag_shl_u32,
        [MAG_DTYPE_I32] = &mag_shl_i32,
        [MAG_DTYPE_U64] = &mag_shl_u64,
        [MAG_DTYPE_I64] = &mag_shl_i64,
    },
    [MAG_OP_SHR] = {
        [MAG_DTYPE_U8]  = &mag_shr_u8,
        [MAG_DTYPE_I8]  = &mag_shr_i8,
        [MAG_DTYPE_U16] = &mag_shr_u16,
        [MAG_DTYPE_I16] = &mag_shr_i16,
        [MAG_DTYPE_U32] = &mag_shr_u32,
        [MAG_DTYPE_I32] = &mag_shr_i32,
        [MAG_DTYPE_U64] = &mag_shr_u64,
        [MAG_DTYPE_I64] = &mag_shr_i64,
    },
    [MAG_OP_EQ] = {
        [MAG_DTYPE_E8M23] = &mag_eq_e8m23,
        [MAG_DTYPE_E5M10] = &mag_eq_e5m10,
        [MAG_DTYPE_BOOL] = &mag_eq_bool,
        [MAG_DTYPE_U8]  = &mag_eq_u8,
        [MAG_DTYPE_I8]  = &mag_eq_i8,
        [MAG_DTYPE_U16] = &mag_eq_u16,
        [MAG_DTYPE_I16] = &mag_eq_i16,
        [MAG_DTYPE_U32] = &mag_eq_u32,
        [MAG_DTYPE_I32] = &mag_eq_i32,
        [MAG_DTYPE_U64] = &mag_eq_u64,
        [MAG_DTYPE_I64] = &mag_eq_i64,
    },
    [MAG_OP_NE] = {
        [MAG_DTYPE_E8M23] = &mag_ne_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ne_e5m10,
        [MAG_DTYPE_BOOL] = &mag_ne_bool,
        [MAG_DTYPE_U8]  = &mag_ne_u8,
        [MAG_DTYPE_I8]  = &mag_ne_i8,
        [MAG_DTYPE_U16] = &mag_ne_u16,
        [MAG_DTYPE_I16] = &mag_ne_i16,
        [MAG_DTYPE_U32] = &mag_ne_u32,
        [MAG_DTYPE_I32] = &mag_ne_i32,
        [MAG_DTYPE_U64] = &mag_ne_u64,
        [MAG_DTYPE_I64] = &mag_ne_i64,
    },
    [MAG_OP_LE] = {
        [MAG_DTYPE_E8M23] = &mag_le_e8m23,
        [MAG_DTYPE_E5M10] = &mag_le_e5m10,
        [MAG_DTYPE_U8]  = &mag_le_u8,
        [MAG_DTYPE_I8]  = &mag_le_i8,
        [MAG_DTYPE_U16] = &mag_le_u16,
        [MAG_DTYPE_I16] = &mag_le_i16,
        [MAG_DTYPE_U32] = &mag_le_u32,
        [MAG_DTYPE_I32] = &mag_le_i32,
        [MAG_DTYPE_U64] = &mag_le_u64,
        [MAG_DTYPE_I64] = &mag_le_i64,
    },
    [MAG_OP_GE] = {
        [MAG_DTYPE_E8M23] = &mag_ge_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ge_e5m10,
        [MAG_DTYPE_U8]  = &mag_ge_u8,
        [MAG_DTYPE_I8]  = &mag_ge_i8,
        [MAG_DTYPE_U16] = &mag_ge_u16,
        [MAG_DTYPE_I16] = &mag_ge_i16,
        [MAG_DTYPE_U32] = &mag_ge_u32,
        [MAG_DTYPE_I32] = &mag_ge_i32,
        [MAG_DTYPE_U64] = &mag_ge_u64,
        [MAG_DTYPE_I64] = &mag_ge_i64,
    },
    [MAG_OP_LT] = {
        [MAG_DTYPE_E8M23] = &mag_lt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_lt_e5m10,
        [MAG_DTYPE_U8]  = &mag_lt_u8,
        [MAG_DTYPE_I8]  = &mag_lt_i8,
        [MAG_DTYPE_U16] = &mag_lt_u16,
        [MAG_DTYPE_I16] = &mag_lt_i16,
        [MAG_DTYPE_U32] = &mag_lt_u32,
        [MAG_DTYPE_I32] = &mag_lt_i32,
        [MAG_DTYPE_U64] = &mag_lt_u64,
        [MAG_DTYPE_I64] = &mag_lt_i64,
    },
    [MAG_OP_GT] = {
        [MAG_DTYPE_E8M23] = &mag_gt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gt_e5m10,
        [MAG_DTYPE_U8]  = &mag_gt_u8,
        [MAG_DTYPE_I8]  = &mag_gt_i8,
        [MAG_DTYPE_U16] = &mag_gt_u16,
        [MAG_DTYPE_I16] = &mag_gt_i16,
        [MAG_DTYPE_U32] = &mag_gt_u32,
        [MAG_DTYPE_I32] = &mag_gt_i32,
        [MAG_DTYPE_U64] = &mag_gt_u64,
        [MAG_DTYPE_I64] = &mag_gt_i64,
    },
};

static void (*const mag_lut_cast_kernels[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(int64_t, void *, const void *) = {
    [MAG_DTYPE_E8M23] = {
        [MAG_DTYPE_E5M10] = &mag_vcast_e8m23_e5m10,
        [MAG_DTYPE_I32] = &mag_vcast_e8m23_i32,
    },
    [MAG_DTYPE_E5M10] = {
        [MAG_DTYPE_E8M23] = &mag_vcast_e5m10_e8m23,
        [MAG_DTYPE_I32] = &mag_vcast_e5m10_i32,
    },
    [MAG_DTYPE_I32] = {
        [MAG_DTYPE_E8M23] = &mag_vcast_i32_e8m23,
        [MAG_DTYPE_E5M10] = &mag_vcast_i32_e5m10,
    },
};

static void MAG_HOTPROC mag_vector_cast_stub(size_t nb, const void *src, mag_dtype_t src_t, void *dst, mag_dtype_t dst_t) {
    mag_assert2(dst_t != src_t); /* src and dst types must differ */
    size_t nbs = mag_dtype_meta_of(src_t)->size;
    size_t nbd = mag_dtype_meta_of(dst_t)->size;
    mag_assert2(!((uintptr_t)src&(nbs-1)));     /* src must be aligned */
    mag_assert2(!((uintptr_t)dst&(nbd-1)));     /* dst must be aligned */
    mag_assert2(!(nb&(nbs-1)));                 /* size must be aligned */
    int64_t numel = (int64_t)(nb/nbs);          /* byte -> elems */
    void (*kern)(int64_t, void *, const void *) = mag_lut_cast_kernels[src_t][dst_t];
    mag_assert(kern, "invalid cast dtypes %s -> %s", mag_dtype_meta_of(src_t)->name, mag_dtype_meta_of(dst_t)->name);
    (*kern)(numel, dst, src);
}

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
    kernels->vector_cast = &mag_vector_cast_stub;
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
** the instruction: pushp  %rbp (d5 08 55) for function prologue, whis is Intel APX and crashes on older CPUs.
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
