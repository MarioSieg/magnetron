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

#ifndef MAG_OP_HELPERS_H
#define MAG_OP_HELPERS_H

#include "mag_def.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

static void mag_norm_axis(int64_t *ax, int64_t ra) {
  if (*ax < 0) *ax += ra;
}


extern bool mag_arange_numel_i64(int64_t start, int64_t stop, int64_t step, int64_t *numel);
extern bool mag_arange_numel_u64(uint64_t start, uint64_t stop, uint64_t step, int64_t *numel);
extern bool mag_arange_numel_float(double start, double end, double step, int64_t *numel);

extern mag_status_t mag_op_stub_reduction(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

extern mag_status_t mag_op_stub_unary(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  const mag_op_params_t *params,
  bool inplace
);

extern mag_status_t mag_op_stub_cu(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  const char *ext,
  mag_tensor_t *x,
  int64_t dim
);

extern mag_status_t mag_op_stub_cu_ex(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_opcode_t op,
  const char *ext,
  mag_tensor_t *x,
  int64_t dim
);

typedef enum mag_binop_flags_t {
  MAG_BINOP_NONE = 0,
  MAG_BINOP_LOGICAL = 1<<0,
  MAG_BINOP_INPLACE = 1<<1
} mag_binop_flags_t;

extern mag_status_t mag_op_stub_binary(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  mag_tensor_t *y,
  mag_binop_flags_t flags
);

extern mag_status_t mag_matmul_verify_shapes(
  mag_error_t *err,
  int64_t *rb,
  int64_t *xb,
  int64_t *yb,
  const mag_tensor_t *x,
  const mag_tensor_t *y
);

extern mag_status_t mag_matmul_alloc_res(
  mag_error_t *err,
  mag_tensor_t **res,
  int64_t rb,
  int64_t *xb,
  int64_t *yb,
  mag_tensor_t *x,
  mag_tensor_t *y
);

extern void MAG_COLDPROC mag_dbg_trace_op_ir(
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out
);

extern mag_status_t mag_check_dtype_and_device_compat(mag_error_t *err, mag_opcode_t op, mag_tensor_t **inputs, uint32_t num_in_dyn);
extern mag_status_t mag_check_inplace_grad_ok(mag_error_t *err, const mag_tensor_t *result);

#ifdef __cplusplus
}
#endif
#endif
