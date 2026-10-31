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
#include "mag_cpu_acc.h"

#include <core/mag_context.h>
#include <core/mag_tensor.h>

#define MAG_MATMUL_FLOPS_PER_WORKER (1<<21)           /* Flops a worker must get before adding another one */
#define MAG_MATMUL_BYTES_PER_WORKER (512<<10)         /* Bytes a worker must stream from DRAM before adding another one */
#define MAG_MATMUL_CACHED_BYTES_PER_WORKER (64<<10)   /* Bytes a worker must read from cache before adding another one */
#define MAG_MATMUL_SMT_FLOPS_THRESHOLD 6e10

static uint32_t mag_cpu_matmul_worker_cap(const mag_command_t *cmd, const mag_cpu_device_t *cpu_dvc) {
  const mag_tensor_t *x = cmd->in[0];
  double flops = 2.0*(double)cmd->out[0]->meta.numel*(double)x->meta.coords.shape[x->meta.coords.rank-1];
  return flops >= MAG_MATMUL_SMT_FLOPS_THRESHOLD ? cpu_dvc->num_allocated_workers : cpu_dvc->num_core_workers;
}

static uint32_t mag_cpu_matmul_static_workers(const mag_command_t *cmd, const mag_cpu_device_t *cpu_dvc) {
  const mag_tensor_t *x = cmd->in[0];
  const mag_tensor_t *y = cmd->in[1];
  const mag_tensor_t *r = cmd->out[0];
  double flops = 2.0*(double)r->meta.numel*(double)x->meta.coords.shape[x->meta.coords.rank-1];
  double bytes = (double)(mag_tensor_numbytes(x) + mag_tensor_numbytes(y) + mag_tensor_numbytes(r));
  const mag_machine_info_t *machine = &cpu_dvc->ctx->machine;
  double core_cache = (double)(machine->cpu_l2_size+machine->cpu_l3_size);
  double quota = bytes <= core_cache ? MAG_MATMUL_CACHED_BYTES_PER_WORKER : MAG_MATMUL_BYTES_PER_WORKER;
  double workers = mag_vmax(flops/(double)MAG_MATMUL_FLOPS_PER_WORKER, bytes/quota);
  uint32_t cap = mag_cpu_matmul_worker_cap(cmd, cpu_dvc);
  return workers >= (double)cap ? cap : (uint32_t)mag_vmax(1.0, workers);
}

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
    [MAG_OP_STRIDED_VIEW] = {0.0, 0},
    [MAG_OP_MEAN] = {3.5, 10000},
    [MAG_OP_MINIMA] = {3.5, 10000},
    [MAG_OP_MAXIMA] = {3.5, 10000},
    [MAG_OP_MIN] = {3.5, 10000},
    [MAG_OP_MAX] = {3.5, 10000},
    [MAG_OP_ARGMIN] = {3.5, 10000},
    [MAG_OP_ARGMAX] = {3.5, 10000},
    [MAG_OP_SUM] = {3.5, 10000},
    [MAG_OP_PROD] = {3.5, 10000},
    [MAG_OP_ALL] = {3.5, 10000},
    [MAG_OP_ANY] = {3.5, 10000},
    [MAG_OP_ABS] = {0.5, 10000},
    [MAG_OP_SGN] = {0.5, 10000},
    [MAG_OP_NEG] = {0.5, 10000},
    [MAG_OP_LOG] = {0.5, 4096},
    [MAG_OP_LOG10] = {0.5, 10000},
    [MAG_OP_LOG1P] = {0.5, 2048},
    [MAG_OP_LOG2] = {0.5, 10000},
    [MAG_OP_SQR] = {0.5, 10000},
    [MAG_OP_RCP] = {0.5, 10000},
    [MAG_OP_SQRT] = {0.5, 8192},
    [MAG_OP_RSQRT] = {0.5, 8192},
    [MAG_OP_SIN] = {0.5, 4096},
    [MAG_OP_COS] = {0.5, 4096},
    [MAG_OP_TAN] = {0.5, 4096},
    [MAG_OP_SINH] = {0.5, 8192},
    [MAG_OP_COSH] = {0.5, 8192},
    [MAG_OP_TANH] = {0.5, 4096},
    [MAG_OP_ASIN] = {0.5, 8192},
    [MAG_OP_ACOS] = {0.5, 8192},
    [MAG_OP_ATAN] = {0.5, 8192},
    [MAG_OP_ASINH] = {0.5, 8192},
    [MAG_OP_ACOSH] = {0.5, 8192},
    [MAG_OP_ATANH] = {0.5, 8192},
    [MAG_OP_STEP] = {0.5, 10000},
    [MAG_OP_ERF] = {0.5, 1024},
    [MAG_OP_ERFC] = {0.5, 1024},
    [MAG_OP_EXP] = {0.5, 4096},
    [MAG_OP_EXP2] = {0.5, 2048},
    [MAG_OP_EXPM1] = {0.5, 1024},
    [MAG_OP_FLOOR] = {0.5, 10000},
    [MAG_OP_CEIL] = {0.5, 10000},
    [MAG_OP_ROUND] = {0.5, 10000},
    [MAG_OP_TRUNC] = {0.5, 10000},
    [MAG_OP_SOFTMAX] = {0.9, 2048},
    [MAG_OP_SIGMOID] = {0.5, 4096},
    [MAG_OP_HARD_SIGMOID] = {0.5, 10000},
    [MAG_OP_SILU] = {0.5, 4096},
    [MAG_OP_RELU] = {0.5, 10000},
    [MAG_OP_GELU] = {0.9, 1024},
    [MAG_OP_GELU_APPROX] = {0.5, 4096},
    [MAG_OP_TRIL] = {0.5, 1024},
    [MAG_OP_TRIU] = {0.5, 1024},
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
    [MAG_OP_WHERE] = {3.5, 1024},
    [MAG_OP_PAD] = {0.5, 10000},
    [MAG_OP_EYE] = {0.5, 10000},
    [MAG_OP_CUSUM] = {0.5, 10000},
    [MAG_OP_CUPROD] = {0.5, 10000},
    [MAG_OP_CUMAX] = {0.5, 10000},
    [MAG_OP_CUMIN] = {0.5, 10000},
    [MAG_OP_SORT] = {0.8, 10000},
    [MAG_OP_ARGSORT] = {0.8, 10000},
    [MAG_OP_BINCOUNT] = {0.0, 0},
    [MAG_OP_NONZERO] = {0.0, 0},
    [MAG_OP_REPEAT] = {0.5, 10000},
    [MAG_OP_REPEAT_INTERLEAVE] = {0.5, 10000},
    [MAG_OP_INDEX_ADD] = {0.0, 0},
    [MAG_OP_EMBEDDING] = {0.5, 10000},
    [MAG_OP_SCATTER] = {0.5, 10000},
    [MAG_OP_SCATTER_ADD] = {0.5, 10000},
    [MAG_OP_CONV] = {0.8, 4096},
    [MAG_OP_CONV_T] = {0.8, 4096},
    [MAG_OP_CONV_WGRAD] = {0.8, 4096},
    [MAG_OP_INTERPOLATE] = {0.5, 10000},
    [MAG_OP_INTERPOLATE_BACK] = {0.5, 10000},
  };
  return scaling_table[op];
}

uint32_t mag_cpu_tune_eager_intra_op_worker_count(const mag_command_t *cmd, mag_device_t *dvc) {
  mag_cpu_device_t *cpu_dvc = dvc->impl;
  int64_t numel_lim = INT64_MIN;
  for (uint32_t i=0; i < cmd->num_in; ++i) numel_lim = mag_vmax(numel_lim, cmd->in[i]->meta.numel);
  for (uint32_t i=0; i < cmd->num_out; ++i) numel_lim = mag_vmax(numel_lim, cmd->out[i]->meta.numel);
  mag_opcode_t op = cmd->op;
  uint32_t wk_alloced = cpu_dvc->num_core_workers;
  const mag_op_traits_t *meta = mag_op_trait(op);
  mag_op_thread_scaling_info info = mag_cpu_get_op_thread_scaling_info(op);
  if (wk_alloced <= 1 || !(meta->flags & MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING) || numel_lim <= info.thread_treshold)
    return 1;
  if (op == MAG_OP_MATMUL) {
    const mag_tensor_t *x = cmd->in[0];
    const mag_tensor_t *y = cmd->in[1];
    if (mag_accel_matmul_supported(x, y, cmd->out[0])) return 1;
    if (mag_matmul_type_detect(x, y) == MAG_MATMUL_TYPE_DOT) return 1;
    return mag_cpu_matmul_static_workers(cmd, cpu_dvc);
  }
  numel_lim -= info.thread_treshold;
  uint32_t workers = (uint32_t)ceil(info.growth*log2((double)numel_lim));
  workers = mag_vmin(wk_alloced, mag_vmax(1, workers));
  return workers;
}
