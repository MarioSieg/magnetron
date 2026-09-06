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

#include "mag_cpu_autotune.h"
#include "mag_cpu.h"

#include <core/mag_context.h>
#include <core/mag_envcfg.h>
#include <core/mag_tensor.h>

#ifndef MAG_MATMUL_FLOPS_PER_WORKER /* Flops a GEMM worker must get before adding another one. */
  #define MAG_MATMUL_FLOPS_PER_WORKER (1<<21)
#endif

/*
** Element counts at which intra-op multithreading starts paying for itself.
**
** Fan-out plus barrier costs on the order of 20us regardless of tensor size, so an op only
** benefits once its single-threaded time clears that. The thresholds therefore track arithmetic
** cost per element: the more work an op does per element, the sooner threading wins. Cheap
** elementwise ops are memory-bandwidth bound, and one core already saturates a large share of
** that bandwidth, so they need to be very large before extra threads help at all.
**
** Measured on an Apple M3 (8 cores) with benchmark/python/tune_intraop.py. Re-derive on another
** machine by sweeping MAG_CPU_INTRAOP_MIN_ELEMS; the tool prints a suggested table.
*/
#define MAG_INTRAOP_BANDWIDTH   (1<<20) /* 1048576. Measured: ADD, SUB, MUL, DIV, SQR, SQRT, ABS, NEG, RELU, CAST. */
#define MAG_INTRAOP_TRANSCEND   (1<<18) /* 262144.  Measured: EXP, TANH, SIGMOID, SILU. */
#define MAG_INTRAOP_EXPENSIVE   (1<<16) /* 65536.   Measured: LOG, GELU_APPROX. */
#define MAG_INTRAOP_VERY_EXP    (1<<12) /* 4096.    Measured: GELU (exact, erf based). */
#define MAG_INTRAOP_NEVER       INT64_MAX /* Measured: FILL, CLONE, SUM, MEAN, SOFTMAX never win, even at 64M elements. */

/*
** The thresholds above assume the contiguous kernel, which walks memory straight through and is
** bandwidth bound. When an operand is strided the kernel instead computes an index per element,
** which makes it compute bound and worth spreading across cores far sooner. The gap is large:
** a strided clone measured 2.2x faster threaded at 64K elements and 4.2x at 1M, while a contiguous
** clone never wins at any size. Anything above this cap is lowered to it when an operand is strided.
*/
#define MAG_INTRAOP_STRIDED     (1<<15) /* 32768. Measured: CLONE through a transposed view. */

static bool mag_cpu_all_operands_contiguous(const mag_command_t *cmd) {
  for (uint32_t i=0; i < cmd->num_in; ++i)
    if (!mag_tensor_is_contiguous(cmd->in[i])) return false;
  for (uint32_t i=0; i < cmd->num_out; ++i)
    if (!mag_tensor_is_contiguous(cmd->out[i])) return false;
  return true;
}

mag_op_thread_scaling_info mag_cpu_get_op_thread_scaling_info(mag_opcode_t op) {
  static const mag_op_thread_scaling_info scaling_table[MAG_OP__NUM] = {
    [MAG_OP_NOP] = {0.0, 0},
    [MAG_OP_FILL] = {0.5, MAG_INTRAOP_NEVER},
    [MAG_OP_MASKED_FILL] = {0.5, MAG_INTRAOP_NEVER},
    [MAG_OP_RAND_UNIFORM] = {0.8, 10000},
    [MAG_OP_RAND_NORMAL] = {1.0, 10000},
    [MAG_OP_RAND_BERNOULLI] = {0.0, 0},
    [MAG_OP_RAND_PERM] = {0.0, 0},
    [MAG_OP_ARANGE] = {0.4, 10000},
    [MAG_OP_ONE_HOT] = {0.4, 10000},
    [MAG_OP_CLONE] = {0.4, MAG_INTRAOP_NEVER},
    [MAG_OP_CAST] = {0.4, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_STRIDED_VIEW] = {0.0, 0},
    [MAG_OP_MEAN] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_MINIMA] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_MAXIMA] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_MIN] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_MAX] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_ARGMIN] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_ARGMAX] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_SUM] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_PROD] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_ALL] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_ANY] = {3.5, MAG_INTRAOP_NEVER},
    [MAG_OP_ABS] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SGN] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_NEG] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_LOG] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_LOG10] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_LOG1P] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_LOG2] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_SQR] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_RCP] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SQRT] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_RSQRT] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SIN] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_COS] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_TAN] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_SINH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_COSH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_TANH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ASIN] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ACOS] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ATAN] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ASINH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ACOSH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ATANH] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_STEP] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_ERF] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_ERFC] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_EXP] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_EXP2] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_EXPM1] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_FLOOR] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_CEIL] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_ROUND] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_TRUNC] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SOFTMAX] = {0.9, MAG_INTRAOP_NEVER},
    [MAG_OP_SOFTMAX_DV] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_SIGMOID] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_SIGMOID_DV] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_HARD_SIGMOID] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SILU] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_SILU_DV] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_TANH_DV] = {0.5, MAG_INTRAOP_TRANSCEND},
    [MAG_OP_RELU] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_RELU_DV] = {0.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_GELU] = {0.9, MAG_INTRAOP_VERY_EXP},
    [MAG_OP_GELU_APPROX] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_GELU_DV] = {0.5, MAG_INTRAOP_EXPENSIVE},
    [MAG_OP_TRIL] = {0.5, 10000},
    [MAG_OP_TRIU] = {0.5, 10000},
    [MAG_OP_MULTINOMIAL] = {0.5, 25000},
    [MAG_OP_CAT] = {0.8, 10000},
    [MAG_OP_ADD] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SUB] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_MUL] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_DIV] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_MOD] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_MATMUL] = {0.4, 1000},
    [MAG_OP_REPEAT_BACK] = {0.5, 25000},
    [MAG_OP_GATHER] = {0.0, 0},
    [MAG_OP_AND] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_OR] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_XOR] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_NOT] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SHL] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_SHR] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_EQ] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_NE] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_LE] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_GE] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_LT] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_GT] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_WHERE] = {3.5, MAG_INTRAOP_BANDWIDTH},
    [MAG_OP_PAD] = {0.5, 10000},
    [MAG_OP_EYE] = {0.5, 10000},
    [MAG_OP_CUSUM] = {0.5, 10000},
    [MAG_OP_CUPROD] = {0.5, 10000},
    [MAG_OP_CUMAX] = {0.5, 10000},
    [MAG_OP_CUMIN] = {0.5, 10000},
    [MAG_OP_REPEAT] = {0.5, 10000},
    [MAG_OP_REPEAT_INTERLEAVE] = {0.5, 10000},
    [MAG_OP_INDEX_ADD] = {0.0, 0},
    [MAG_OP_EMBEDDING] = {0.5, 10000},
    [MAG_OP_SCATTER] = {0.5, 10000},
    [MAG_OP_SCATTER_ADD] = {0.5, 10000},
  };
  return scaling_table[op];
}

uint32_t mag_cpu_tune_eager_intra_op_worker_count(const mag_command_t *cmd, mag_device_t *dvc) {
  mag_cpu_device_t *cpu_dvc = dvc->impl;
  int64_t max_numel = INT64_MIN;
  for (uint32_t i=0; i < cmd->num_in; ++i) max_numel = mag_xmax(max_numel, cmd->in[i]->meta.numel);
  for (uint32_t i=0; i < cmd->num_out; ++i) max_numel = mag_xmax(max_numel, cmd->out[i]->meta.numel);
  mag_opcode_t op = cmd->op;
  uint32_t allocated_workers = cpu_dvc->num_allocated_workers;
  const mag_op_traits_t *meta = mag_op_trait(op);
  mag_op_thread_scaling_info info = mag_cpu_get_op_thread_scaling_info(op);
  if (info.thread_treshold > MAG_INTRAOP_STRIDED && !mag_cpu_all_operands_contiguous(cmd))
    info.thread_treshold = MAG_INTRAOP_STRIDED; /* Strided kernels are compute bound, see above. */
  static int64_t threshold_override = -2; /* -2 = not yet read, -1 = unset, >=0 = pinned. */
  if (mag_unlikely(threshold_override == -2)) threshold_override = mag_envcfg_cpu_intraop_min_elems();
  if (threshold_override >= 0) info.thread_treshold = threshold_override;
  if (allocated_workers <= 1 || !(meta->flags & MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING) || max_numel < info.thread_treshold)  /* Use a single worker (main thread). */
    return 1;
  if (op == MAG_OP_MATMUL) { /* Special case for matmul */
    const mag_tensor_t *x = cmd->in[0];
    const mag_tensor_t *y = cmd->in[1];
    mag_matmul_type_t matmul_type = mag_matmul_type_detect(x, y);
    switch (matmul_type) {
      case MAG_MATMUL_TYPE_DOT:
      case MAG_MATMUL_TYPE_BMM_DOT: return 1;
      case MAG_MATMUL_TYPE_GEMV_VEC_MAT:
      case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
      case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT:
      case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC: {
        int64_t K = x->meta.coords.shape[x->meta.coords.rank-1];
        int64_t N = y->meta.coords.shape[y->meta.coords.rank-1];
        int64_t work_bytes = N*K*mag_type_trait(x->meta.dtype)->size;
        int64_t workers = 1;
        if (work_bytes >= 4LL   << 20) workers = 4;
        if (work_bytes >= 16LL  << 20) workers = 8;
        if (work_bytes >= 32LL  << 20) workers = 16;
        if (work_bytes >= 96LL  << 20) workers = 32;
        if (work_bytes >= 256LL << 20) workers = 64;
        return mag_xmin(workers, allocated_workers);
      }
      default: { /* GEMM/BMM: spread over workers only once the flops amortize the barrier and packing. */
        int64_t K = x->meta.coords.shape[x->meta.coords.rank-1];
        double flops = 2.0*(double)cmd->out[0]->meta.numel*(double)K;
        int64_t workers = (int64_t)(flops/(double)MAG_MATMUL_FLOPS_PER_WORKER);
        return (uint32_t)mag_xmin((int64_t)allocated_workers, mag_xmax(1, workers));
      }
    }
  }
  max_numel -= info.thread_treshold;
  uint32_t workers = (uint32_t)ceil(info.growth * log2((double)max_numel)); /* Logarithmic scaling */
  workers = mag_xmin(allocated_workers, mag_xmax(1, workers));
  return workers;
}
