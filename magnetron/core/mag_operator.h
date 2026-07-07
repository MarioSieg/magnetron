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

#ifndef MAG_OPERATOR_H
#define MAG_OPERATOR_H

#include "mag_def.h"
#include "mag_reduce_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum mag_opflags_t {
  MAG_OP_FLAG_NONE = 0,
  MAG_OP_FLAG_SUPPORTS_INPLACE = 1<<0,                /* Allows to be executed inplace on the input tensor. */
  MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING = 1<<1,      /* Supports multithreading on CPU. */
} mag_opflags_t;

#define MAG_OP_FLAGS_COMMON (MAG_OP_FLAG_SUPPORTS_INPLACE+MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING)
#define MAG_OP_INOUT_DYN (UINT32_MAX-1) /* Flags flexible input/output count. Used for operations that can have arbitrary number of inputs/outputs such as split or cat. */

typedef enum mag_pad_mode_t {
  MAG_PAD_MODE_CONSTANT = 0,
  MAG_PAD_MODE_REFLECT = 1,
  MAG_PAD_MODE_REPLICATE = 2,
  MAG_PAD_MODE_CIRCULAR = 3,
} mag_pad_mode_t;

typedef union mag_op_params_t {
  struct {
    int64_t original_axes[MAG_MAX_DIMS];
  } transpose;
  struct {
    int64_t rank;
    int64_t axes[MAG_MAX_DIMS];
  } permute;
  struct {
    int64_t ndims;               /* Number of flipped axes. */
    int64_t dims[MAG_MAX_DIMS];  /* Normalized axes that were reversed. */
  } flip;
  struct {
    int64_t diag;
  } trilu;
  struct {
    mag_scalar_t start;
    mag_scalar_t step;
  } arange;
  struct {
    mag_reduce_plan_t red_plan;
  } reduction;
  struct {
    int64_t k;
    int64_t dim;
    bool largest : 1;
    bool sorted : 1;
  } topk;
  struct {
    int64_t dim;
  } cumu;
  struct {
    int64_t rank;
    int64_t pad_before[MAG_MAX_DIMS];
    int64_t pad_after[MAG_MAX_DIMS];
    mag_pad_mode_t mode;
    mag_scalar_t value;
  } pad;
  struct {
    int64_t samples;
    bool replacement;
  } multinomial;
  struct {
    int64_t dim;
  } cat;
  struct {
    int64_t dim;
    int64_t start;
    int64_t len;
    int64_t step;
  } slice;
  struct {
    int64_t num_classes;
  } one_hot;
  struct {
    int64_t rank;
    int64_t in_rank;
    int64_t in_shape[MAG_MAX_DIMS];
    int64_t out_shape[MAG_MAX_DIMS];
  } repeat;
  struct {
    bool flatten;
    int64_t dim;
    int64_t rank;
    int64_t out_shape[MAG_MAX_DIMS];
    const int64_t *counts;
    int64_t count_len;
  } repeat_interleave;
  struct {
    int64_t dim;
  } gather;
  struct {
    int64_t dim;
    double alpha;
  } index_add;
  struct {
    int64_t dim;
  } scatter;
  struct {
    mag_scalar_t value;
  } fill;
  struct {
    mag_scalar_t low;
    mag_scalar_t high;
  } uniform;
  struct {
    mag_scalar_t mean;
    mag_scalar_t std;
  } normal;
  struct {
    double p;
  } bernoulli;
} mag_op_params_t;

#define mag_opdef(_, __)\
  _(NOP, 0, 0, NONE, MAG_OP_FLAG_NONE, NULL)__\
  _(FILL, 0, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(MASKED_FILL, 2, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, masked_fill)__\
  _(RAND_UNIFORM, 0, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(RAND_NORMAL, 0, 1, FP, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(RAND_BERNOULLI, 0, 1, BOOL, MAG_OP_FLAG_NONE, NULL)__\
  _(RAND_PERM, 0, 1, INTEGER, MAG_OP_FLAG_NONE, NULL)__\
  _(ARANGE, 0, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(ONE_HOT, 1, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(CLONE, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, clone)__\
  _(CAST, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, cast)__\
  _(VIEW, 1, 1, ALL, MAG_OP_FLAG_NONE, view)__\
  _(TRANSPOSE, 1, 1, ALL, MAG_OP_FLAG_NONE, transpose)__\
  _(PERMUTE, 1, 1, ALL, MAG_OP_FLAG_NONE, permute)__\
  _(MEAN, 1, 1, FP, MAG_OP_FLAG_NONE, mean)__\
  _(MINIMA, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(MAXIMA, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(ARGMIN, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(ARGMAX, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(SUM, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, sum)__\
  _(PROD, 1, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(ALL, 1, 1, ALL, MAG_OP_FLAG_NONE, NULL)__\
  _(ANY, 1, 1, ALL, MAG_OP_FLAG_NONE, NULL)__\
  _(TOPK, 1, 2, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(ABS, 1, 1, NUMERIC, MAG_OP_FLAGS_COMMON, abs)__\
  _(SGN, 1, 1, NUMERIC, MAG_OP_FLAGS_COMMON, NULL)__\
  _(NEG, 1, 1, NUMERIC, MAG_OP_FLAGS_COMMON, neg)__\
  _(LOG, 1, 1, FP, MAG_OP_FLAGS_COMMON, log)__\
  _(LOG10, 1, 1, FP, MAG_OP_FLAGS_COMMON, log10)__\
  _(LOG1P, 1, 1, FP, MAG_OP_FLAGS_COMMON, log1p)__\
  _(LOG2, 1, 1, FP, MAG_OP_FLAGS_COMMON, log2)__\
  _(SQR, 1, 1, NUMERIC, MAG_OP_FLAGS_COMMON, sqr)__\
  _(RCP, 1, 1, FP, MAG_OP_FLAGS_COMMON, rcp)__\
  _(SQRT, 1, 1, FP, MAG_OP_FLAGS_COMMON, sqrt)__\
  _(RSQRT, 1, 1, FP, MAG_OP_FLAGS_COMMON, rsqrt)__\
  _(SIN, 1, 1, FP, MAG_OP_FLAGS_COMMON, sin)__\
  _(COS, 1, 1, FP, MAG_OP_FLAGS_COMMON, cos)__\
  _(TAN, 1, 1, FP, MAG_OP_FLAGS_COMMON, tan)__\
  _(SINH, 1, 1, FP, MAG_OP_FLAGS_COMMON, sinh)__\
  _(COSH, 1, 1, FP, MAG_OP_FLAGS_COMMON, cosh)__\
  _(TANH, 1, 1, FP, MAG_OP_FLAGS_COMMON, tanh)__\
  _(ASIN, 1, 1, FP, MAG_OP_FLAGS_COMMON, asin)__\
  _(ACOS, 1, 1, FP, MAG_OP_FLAGS_COMMON, acos)__\
  _(ATAN, 1, 1, FP, MAG_OP_FLAGS_COMMON, atan)__\
  _(ASINH, 1, 1, FP, MAG_OP_FLAGS_COMMON, asinh)__\
  _(ACOSH, 1, 1, FP, MAG_OP_FLAGS_COMMON, acosh)__\
  _(ATANH, 1, 1, FP, MAG_OP_FLAGS_COMMON, atanh)__\
  _(STEP, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(ERF, 1, 1, FP, MAG_OP_FLAGS_COMMON, erf)__\
  _(ERFC, 1, 1, FP, MAG_OP_FLAGS_COMMON, erfc)__\
  _(EXP, 1, 1, FP, MAG_OP_FLAGS_COMMON, exp)__\
  _(EXP2, 1, 1, FP, MAG_OP_FLAGS_COMMON, exp2)__\
  _(EXPM1, 1, 1, FP, MAG_OP_FLAGS_COMMON, expm1)__\
  _(FLOOR, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(CEIL, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(ROUND, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(TRUNC, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(SOFTMAX, 1, 1, FP, MAG_OP_FLAGS_COMMON, softmax)__\
  _(SOFTMAX_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(SIGMOID, 1, 1, FP, MAG_OP_FLAGS_COMMON, sigmoid)__\
  _(SIGMOID_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(HARD_SIGMOID, 1, 1, FP, MAG_OP_FLAGS_COMMON, hard_sigmoid)__\
  _(SILU, 1, 1, FP, MAG_OP_FLAGS_COMMON, silu)__\
  _(SILU_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(TANH_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(RELU, 1, 1, FP, MAG_OP_FLAGS_COMMON, relu)__\
  _(RELU_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(GELU, 1, 1, FP, MAG_OP_FLAGS_COMMON, gelu)__\
  _(GELU_APPROX, 1, 1, FP, MAG_OP_FLAGS_COMMON, gelu)__\
  _(GELU_DV, 1, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(TRIL, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, tril)__\
  _(TRIU, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, triu)__\
  _(MULTINOMIAL, 1, 1, FP, MAG_OP_FLAG_NONE, NULL)__\
  _(CAT, MAG_OP_INOUT_DYN, 1, ALL, MAG_OP_FLAGS_COMMON, cat)__\
  _(ADD, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, add)__\
  _(SUB, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, sub)__\
  _(MUL, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, mul)__\
  _(DIV, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, div)__\
  _(FLOORDIV, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, NULL)__\
  _(MOD, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, NULL)__\
  _(POW, 2, 1, NUMERIC, MAG_OP_FLAGS_COMMON, pow)__\
  _(MATMUL, 2, 1, FP, MAG_OP_FLAGS_COMMON, matmul)__\
  _(REPEAT_BACK, 2, 1, FP, MAG_OP_FLAGS_COMMON, NULL)__\
  _(GATHER, 2, 1, ALL, MAG_OP_FLAG_NONE, gather)__\
  _(AND, 2, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(OR, 2, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(XOR, 2, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(NOT, 1, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(SHL, 2, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(SHR, 2, 1, INTEGRAL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(EQ, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(NE, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(LE, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(GE, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(LT, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(GT, 2, 1, ALL, MAG_OP_FLAGS_COMMON, NULL)__\
  _(WHERE, 3, 1, ALL, MAG_OP_FLAGS_COMMON, where)__\
  _(MIN, 2, 1, ALL, MAG_OP_FLAGS_COMMON, min)__\
  _(MAX, 2, 1, ALL, MAG_OP_FLAGS_COMMON, max)__\
  _(CLAMP, 3, 1, ALL, MAG_OP_FLAGS_COMMON, clamp)__\
  _(PAD, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(EYE, 0, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(CUSUM, 1, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(CUPROD, 1, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(CUMAX, 1, 2, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(CUMIN, 1, 2, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(REPEAT, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, repeat)__\
  _(REPEAT_INTERLEAVE, 1, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(INDEX_ADD, 3, 1, NUMERIC, MAG_OP_FLAG_NONE, NULL)__\
  _(EMBEDDING, 2, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, embedding)__\
  _(SCATTER, 3, 1, ALL, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(SCATTER_ADD, 3, 1, NUMERIC, MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING, NULL)__\
  _(FLIP, 1, 1, ALL, MAG_OP_FLAG_NONE, flip)__\
  _(SLICE, 1, 1, ALL, MAG_OP_FLAG_NONE, slice)__

/* Standard opcodes, not including initialization operators. */
typedef enum mag_opcode_t {
#define _(enu, in, out, dtm, flags, diff) MAG_OP_##enu
  mag_opdef(_, MAG_SEP)
#undef _
  MAG_OP__NUM
} mag_opcode_t;
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_SLICE+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff); /* Must fit in one byte */

typedef uint16_t mag_dtype_mask_t; /* Bitmask of supported dtypes, 1 bit per dtype. */
mag_static_assert(MAG_DTYPE__NUM <= 16); /* Must fit in 8 bits, if this fails increase the type of dtpe_mask. */
#define mag_dtype_bit(x) (((mag_dtype_mask_t)1)<<((x)&((sizeof(mag_dtype_mask_t)<<3)-1)))
#define mag_dtype_mask(enume) mag_dtype_bit(MAG_DTYPE_##enume)
#define MAG_DTYPE_MASK_NONE 0
#define MAG_DTYPE_MASK_FP (mag_dtype_mask(FLOAT32)|mag_dtype_mask(FLOAT16)|mag_dtype_mask(BFLOAT16)|mag_dtype_mask(FLOAT8_E4M3FN))
#define MAG_DTYPE_MASK_UINT (mag_dtype_mask(UINT8)|mag_dtype_mask(UINT16)|mag_dtype_mask(UINT32)|mag_dtype_mask(UINT64))
#define MAG_DTYPE_MASK_SINT (mag_dtype_mask(INT8)|mag_dtype_mask(INT16)|mag_dtype_mask(INT32)|mag_dtype_mask(INT64))
#define MAG_DTYPE_MASK_INTEGER (MAG_DTYPE_MASK_UINT|MAG_DTYPE_MASK_SINT)
#define MAG_DTYPE_MASK_INTEGRAL (mag_dtype_mask(BOOLEAN)|MAG_DTYPE_MASK_INTEGER)
#define MAG_DTYPE_MASK_NUMERIC (MAG_DTYPE_MASK_INTEGER|MAG_DTYPE_MASK_FP)
#define MAG_DTYPE_MASK_BOOL (mag_dtype_mask(BOOLEAN))
#define MAG_DTYPE_MASK_ALL (MAG_DTYPE_MASK_NUMERIC|MAG_DTYPE_MASK_BOOL)

typedef struct mag_au_state_t mag_au_state_t;

/* Stores operator metadata such as operation type, number of inputs and parameters, and the types of the parameters. */
typedef struct mag_op_traits_t {
  const char *const mnemonic;
  const uint32_t in;
  const uint32_t out;
  const mag_dtype_mask_t dtype_mask;
  const mag_opflags_t flags;
  mag_status_t (*const backward)(mag_error_t *, mag_au_state_t *, mag_tensor_t **);
} mag_op_traits_t;

extern MAG_EXPORT const mag_op_traits_t *mag_op_trait(mag_opcode_t op); /* Get operation metadata for a specific opcode. */

#ifdef __cplusplus
}
#endif

#endif
