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

#include "mag_cuda_exec.cuh"

#include "mag_cuda_unary.cuh"
#include "mag_cuda_binary.cuh"
#include "mag_cuda_fill.cuh"
#include "mag_cuda_reduction.cuh"
#include "mag_cuda_misc.cuh"

namespace mag {
  static constexpr mag_status_t(*k_kernel_dispatch_table[])(mag_error_t *, const mag_command_t &, cudaStream_t) = {
    [MAG_OP_NOP] = +[](mag_error_t *, const mag_command_t &, cudaStream_t) -> mag_status_t { return MAG_OK; },
    [MAG_OP_FILL] = &fill_op_fill,
    [MAG_OP_MASKED_FILL] = &fill_op_masked_fill,
    [MAG_OP_RAND_UNIFORM] = &fill_op_fill_rand_uniform,
    [MAG_OP_RAND_NORMAL] = &fill_op_fill_rand_normal,
    [MAG_OP_RAND_BERNOULLI] = &fill_op_rand_bernoulli,
    [MAG_OP_RAND_PERM] = &fill_op_rand_perm,
    [MAG_OP_ARANGE] = &fill_op_arange,
    [MAG_OP_ONE_HOT] = &misc_op_one_hot,
    [MAG_OP_CLONE] = &unary_op_clone,
    [MAG_OP_CAST] = &unary_op_cast,
    [MAG_OP_MEAN] = &reduce_op_mean,
    [MAG_OP_MINIMA] = &reduce_op_minima,
    [MAG_OP_MAXIMA] = &reduce_op_maxima,
    [MAG_OP_ARGMIN] = &reduce_op_argmin,
    [MAG_OP_ARGMAX] = &reduce_op_argmax,
    [MAG_OP_SUM] = &reduce_op_sum,
    [MAG_OP_PROD] = &reduce_op_prod,
    [MAG_OP_ALL] = &reduce_op_all,
    [MAG_OP_ANY] = &reduce_op_any,
    [MAG_OP_TOPK] = &misc_op_topk,
    [MAG_OP_ABS] = &unary_op_abs,
    [MAG_OP_SGN] = &unary_op_sgn,
    [MAG_OP_NEG] = &unary_op_neg,
    [MAG_OP_LOG] = &unary_op_log,
    [MAG_OP_LOG10] = &unary_op_log10,
    [MAG_OP_LOG1P] = &unary_op_log1p,
    [MAG_OP_LOG2] = &unary_op_log2,
    [MAG_OP_SQR] = &unary_op_sqr,
    [MAG_OP_RCP] = &unary_op_rcp,
    [MAG_OP_SQRT] = &unary_op_sqrt,
    [MAG_OP_RSQRT] = &unary_op_rsqrt,
    [MAG_OP_SIN] = &unary_op_sin,
    [MAG_OP_COS] = &unary_op_cos,
    [MAG_OP_TAN] = &unary_op_tan,
    [MAG_OP_SINH] = &unary_op_sinh,
    [MAG_OP_COSH] = &unary_op_cosh,
    [MAG_OP_TANH] = &unary_op_tanh,
    [MAG_OP_ASIN] = &unary_op_asin,
    [MAG_OP_ACOS] = &unary_op_acos,
    [MAG_OP_ATAN] = &unary_op_atan,
    [MAG_OP_ASINH] = &unary_op_asinh,
    [MAG_OP_ACOSH] = &unary_op_acosh,
    [MAG_OP_ATANH] = &unary_op_atanh,
    [MAG_OP_STEP] = &unary_op_step,
    [MAG_OP_ERF] = &unary_op_erf,
    [MAG_OP_ERFC] = &unary_op_erfc,
    [MAG_OP_EXP] = &unary_op_exp,
    [MAG_OP_EXP2] = &unary_op_exp2,
    [MAG_OP_EXPM1] = &unary_op_expm1,
    [MAG_OP_FLOOR] = &unary_op_floor,
    [MAG_OP_CEIL] = &unary_op_ceil,
    [MAG_OP_ROUND] = &unary_op_round,
    [MAG_OP_TRUNC] = &unary_op_trunc,
    [MAG_OP_SOFTMAX] = &unary_op_softmax,
    [MAG_OP_SOFTMAX_DV] = &unary_op_softmax_dv,
    [MAG_OP_SIGMOID] = &unary_op_sigmoid,
    [MAG_OP_SIGMOID_DV] = &unary_op_sigmoid_dv,
    [MAG_OP_HARD_SIGMOID] = &unary_op_hard_sigmoid,
    [MAG_OP_SILU] = &unary_op_silu,
    [MAG_OP_SILU_DV] = &unary_op_silu_dv,
    [MAG_OP_TANH_DV] = &unary_op_tanh_dv,
    [MAG_OP_RELU] = &unary_op_relu,
    [MAG_OP_RELU_DV] = &unary_op_relu_dv,
    [MAG_OP_GELU] = &unary_op_gelu,
    [MAG_OP_GELU_APPROX] = &unary_op_gelu_approx,
    [MAG_OP_GELU_DV] = &unary_op_gelu_dv,
    [MAG_OP_TRIL] = &misc_op_tril,
    [MAG_OP_TRIU] = &misc_op_triu,
    [MAG_OP_MULTINOMIAL] = &misc_op_multinomial,
    [MAG_OP_CAT] = &misc_op_cat,
    [MAG_OP_ADD] = &binary_op_add,
    [MAG_OP_SUB] = &binary_op_sub,
    [MAG_OP_MUL] = &binary_op_mul,
    [MAG_OP_DIV] = &binary_op_div,
    [MAG_OP_FLOORDIV] = &binary_op_floordiv,
    [MAG_OP_MOD] = &binary_op_mod,
    [MAG_OP_POW] = &binary_op_pow,
    [MAG_OP_MATMUL] = &misc_op_matmul,
    [MAG_OP_REPEAT_BACK] = &misc_op_repeat_back,
    [MAG_OP_GATHER] = &misc_op_gather,
    [MAG_OP_AND] = &binary_op_and,
    [MAG_OP_OR] = &binary_op_or,
    [MAG_OP_XOR] = &binary_op_xor,
    [MAG_OP_NOT] = &unary_op_not,
    [MAG_OP_SHL] = &binary_op_shl,
    [MAG_OP_SHR] = &binary_op_shr,
    [MAG_OP_EQ] = &binary_op_eq,
    [MAG_OP_NE] = &binary_op_ne,
    [MAG_OP_LE] = &binary_op_le,
    [MAG_OP_GE] = &binary_op_ge,
    [MAG_OP_LT] = &binary_op_lt,
    [MAG_OP_GT] = &binary_op_gt,
    [MAG_OP_WHERE] = &misc_op_where,
    [MAG_OP_MIN] = nullptr,
    [MAG_OP_MAX] = nullptr,
    [MAG_OP_CLAMP] = nullptr,
    [MAG_OP_PAD] = &misc_op_pad,
    [MAG_OP_EYE] = &fill_op_eye,
    [MAG_OP_CUSUM] = &misc_op_cusum,
    [MAG_OP_CUPROD] = &misc_op_cuprod,
    [MAG_OP_CUMAX] = &misc_op_cumax,
    [MAG_OP_CUMIN] = &misc_op_cumin,
    [MAG_OP_REPEAT] = &misc_op_repeat,
    [MAG_OP_REPEAT_INTERLEAVE] = &misc_op_repeat_interleave,
    [MAG_OP_INDEX_ADD] = &misc_op_index_add,
    [MAG_OP_EMBEDDING] = &misc_op_embedding,
    [MAG_OP_SCATTER] = &misc_op_scatter,
    [MAG_OP_SCATTER_ADD] = &misc_op_scatter_add,
    [MAG_OP_STRIDED_VIEW] = +[](mag_error_t *, const mag_command_t &, cudaStream_t) -> mag_status_t { return MAG_OK; },
  };
  static_assert(std::size(k_kernel_dispatch_table) == MAG_OP__NUM, "Dispatch table size mismatch");
  //static_assert([] -> bool {
  //    for (auto *fn : dispatch_table)
  //        if (!fn) return false;
  //    return true;
  //}());

  mag_status_t submit_op(mag_error_t *err, mag_device_t *dvc, const mag_command_t *cmd) {
    int ordinal = static_cast<int>(dvc->id.device_ordinal);
    const auto &phys_device = *static_cast<const physical_device *>(dvc->impl);
    if (mag_status_t stat = phys_device.ensure_initialized(err); mag_iserr(stat)) return stat;
    mag_cu_rt_check(err, cudaSetDevice(ordinal), "failed to set active device");
    cudaStream_t stream = phys_device.stream();
    auto *kernel = k_kernel_dispatch_table[cmd->op];
    if (mag_unlikely(kernel == nullptr))
      return mag_set_error(err, MAG_ERR_KERNEL, "cuda: operator %s not implemented in CUDA backend.", mag_op_trait(cmd->op)->mnemonic);
    if (mag_status_t stat = (*kernel)(err, *cmd, stream); mag_unlikely(mag_iserr(stat)))
      return stat;
    mag_cu_rt_check(err, cudaGetLastError(), "kernel execution failed for operator");
    if constexpr (sync_after_each_op)
      mag_cu_rt_check(err, cudaStreamSynchronize(stream), "failed to synchronize stream");
    return MAG_OK;
  }
}
