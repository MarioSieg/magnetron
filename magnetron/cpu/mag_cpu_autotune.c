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
#include <core/mag_tensor.h>

mag_op_thread_scaling_info mag_cpu_get_op_thread_scaling_info(mag_opcode_t op) {
  static const mag_op_thread_scaling_info scaling_table[MAG_OP__NUM] = {
    [MAG_OP_NOP] = {0.0, 0},
    [MAG_OP_FILL] = {0.5, 10000},
    [MAG_OP_MASKED_FILL] = {0.5, 10000},
    [MAG_OP_RAND_UNIFORM] = {0.8, 10000},
    [MAG_OP_RAND_NORMAL] = {1.0, 10000},
    [MAG_OP_RAND_BERNOULLI] = {0.0, 0},
    [MAG_OP_RAND_PERM] = {0.0, 0},
    [MAG_OP_ARANGE] = {0.4, 10000},
    [MAG_OP_ONE_HOT] = {0.4, 10000},
    [MAG_OP_CLONE] = {0.4, 10000},
    [MAG_OP_CAST] = {0.4, 10000},
    [MAG_OP_VIEW] = {0.0, 0},
    [MAG_OP_TRANSPOSE] = {0.0, 0},
    [MAG_OP_PERMUTE] = {0.0, 0},
    [MAG_OP_MEAN] = {0.0, 0},
    [MAG_OP_MIN] = {0.0, 0},
    [MAG_OP_MAX] = {0.0, 0},
    [MAG_OP_ARGMIN] = {0.0, 0},
    [MAG_OP_ARGMAX] = {0.0, 0},
    [MAG_OP_SUM] = {0.0, 0},
    [MAG_OP_PROD] = {0.0, 0},
    [MAG_OP_ALL] = {0.0, 0},
    [MAG_OP_ANY] = {0.0, 0},
    [MAG_OP_ABS] = {0.5, 25000},
    [MAG_OP_SGN] = {0.5, 25000},
    [MAG_OP_NEG] = {0.5, 25000},
    [MAG_OP_LOG] = {0.5, 25000},
    [MAG_OP_LOG10] = {0.5, 25000},
    [MAG_OP_LOG1P] = {0.5, 25000},
    [MAG_OP_LOG2] = {0.5, 25000},
    [MAG_OP_SQR] = {0.5, 25000},
    [MAG_OP_RCP] = {0.5, 25000},
    [MAG_OP_SQRT] = {0.5, 25000},
    [MAG_OP_RSQRT] = {0.5, 25000},
    [MAG_OP_SIN] = {0.5, 25000},
    [MAG_OP_COS] = {0.5, 25000},
    [MAG_OP_TAN] = {0.5, 25000},
    [MAG_OP_SINH] = {0.5, 25000},
    [MAG_OP_COSH] = {0.5, 25000},
    [MAG_OP_TANH] = {0.5, 25000},
    [MAG_OP_ASIN] = {0.5, 25000},
    [MAG_OP_ACOS] = {0.5, 25000},
    [MAG_OP_ATAN] = {0.5, 25000},
    [MAG_OP_ASINH] = {0.5, 25000},
    [MAG_OP_ACOSH] = {0.5, 25000},
    [MAG_OP_ATANH] = {0.5, 25000},
    [MAG_OP_STEP] = {0.5, 25000},
    [MAG_OP_ERF] = {0.5, 25000},
    [MAG_OP_ERFC] = {0.5, 25000},
    [MAG_OP_EXP] = {0.5, 25000},
    [MAG_OP_EXP2] = {0.5, 25000},
    [MAG_OP_EXPM1] = {0.5, 25000},
    [MAG_OP_FLOOR] = {0.5, 25000},
    [MAG_OP_CEIL] = {0.5, 25000},
    [MAG_OP_ROUND] = {0.5, 25000},
    [MAG_OP_TRUNC] = {0.5, 25000},
    [MAG_OP_SOFTMAX] = {0.9, 25000},
    [MAG_OP_SOFTMAX_DV] = {0.5, 25000},
    [MAG_OP_SIGMOID] = {0.5, 25000},
    [MAG_OP_SIGMOID_DV] = {0.5, 25000},
    [MAG_OP_HARD_SIGMOID] = {0.5, 25000},
    [MAG_OP_SILU] = {0.5, 25000},
    [MAG_OP_SILU_DV] = {0.5, 25000},
    [MAG_OP_TANH_DV] = {0.5, 25000},
    [MAG_OP_RELU] = {0.5, 25000},
    [MAG_OP_RELU_DV] = {0.5, 25000},
    [MAG_OP_GELU] = {0.9, 10000},
    [MAG_OP_GELU_APPROX] = {0.5, 25000},
    [MAG_OP_GELU_DV] = {0.5, 25000},
    [MAG_OP_TRIL] = {0.5, 10000},
    [MAG_OP_TRIU] = {0.5, 10000},
    [MAG_OP_MULTINOMIAL] = {0.5, 25000},
    [MAG_OP_CAT] = {0.8, 10000},
    [MAG_OP_ADD] = {3.5, 10000},
    [MAG_OP_SUB] = {3.5, 10000},
    [MAG_OP_MUL] = {3.5, 10000},
    [MAG_OP_DIV] = {3.5, 10000},
    [MAG_OP_MOD] = {3.5, 10000},
    [MAG_OP_MATMUL] = {0.4, 1000},
    [MAG_OP_REPEAT_BACK] = {0.5, 25000},
    [MAG_OP_GATHER] = {0.0, 0},
    [MAG_OP_AND] = {3.5, 10000},
    [MAG_OP_OR] = {3.5, 10000},
    [MAG_OP_XOR] = {3.5, 10000},
    [MAG_OP_NOT] = {3.5, 10000},
    [MAG_OP_SHL] = {3.5, 10000},
    [MAG_OP_SHR] = {3.5, 10000},
    [MAG_OP_EQ] = {3.5, 10000},
    [MAG_OP_NE] = {3.5, 10000},
    [MAG_OP_LE] = {3.5, 10000},
    [MAG_OP_GE] = {3.5, 10000},
    [MAG_OP_LT] = {3.5, 10000},
    [MAG_OP_GT] = {3.5, 10000},
    [MAG_OP_WHERE] = {3.5, 10000},
    [MAG_OP_PAD] = {0.5, 10000},
    [MAG_OP_EYE] = {0.5, 10000},
    [MAG_OP_CUSUM] = {0.5, 10000},
    [MAG_OP_CUPROD] = {0.5, 10000},
    [MAG_OP_CUMAX] = {0.5, 10000},
    [MAG_OP_CUMIN] = {0.5, 10000},
    [MAG_OP_REPEAT] = {0.5, 10000},
    [MAG_OP_REPEAT_INTERLEAVE] = {0.5, 10000},
    [MAG_OP_INDEX_ADD] = {0.0, 0},
  };
  return scaling_table[op];
}

uint32_t mag_cpu_tune_eager_intra_op_worker_count(const mag_command_t *cmd, mag_device_t *dvc) {
  mag_cpu_device_t *cpu_dvc = dvc->impl;
  int64_t max_numel = INT64_MIN;
  for (uint32_t i=0; i < cmd->num_in; ++i) max_numel = mag_xmax(max_numel, cmd->in[i]->numel);
  for (uint32_t i=0; i < cmd->num_out; ++i) max_numel = mag_xmax(max_numel, cmd->out[i]->numel);
  mag_opcode_t op = cmd->op;
  uint32_t allocated_workers = cpu_dvc->num_allocated_workers;
  const mag_op_traits_t *meta = mag_op_trait(op);
  mag_op_thread_scaling_info info = mag_cpu_get_op_thread_scaling_info(op);
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
        int64_t K = x->coords.shape[x->coords.rank-1];
        int64_t N = y->coords.shape[y->coords.rank-1];
        int64_t work_bytes = N*K*mag_type_trait(x->dtype)->size;
        int64_t workers = 1;
        if (work_bytes >= 4LL   << 20) workers = 4;
        if (work_bytes >= 16LL  << 20) workers = 8;
        if (work_bytes >= 32LL  << 20) workers = 16;
        if (work_bytes >= 96LL  << 20) workers = 32;
        if (work_bytes >= 256LL << 20) workers = 64;
        return mag_xmin(workers, allocated_workers);
      }
      default: return allocated_workers;
    }
  }
  max_numel -= info.thread_treshold;
  uint32_t workers = (uint32_t)ceil(info.growth * log2((double)max_numel)); /* Logarithmic scaling */
  workers = mag_xmin(allocated_workers, mag_xmax(1, workers));
  return workers;
}
