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

#include "mag_cpu_kernels_matmul_dot.h"
#include "mag_cpu_kernels_matmul_gemv_vec_mat.h"
#include "mag_cpu_kernels_matmul_gemv_mat_vec.h"

static mag_status_t mag_matmul_generic(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *y = mag_cmd_in(1);
  mag_matmul_type_t type = mag_matmul_type_detect(x, y);
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID:
      mag_contract(err, ERR_OPERATOR_IMPOSSIBLE, {}, type != MAG_MATMUL_TYPE_INVALID, "Invalid matmul type detected for the given input tensors");
      break;
    case MAG_MATMUL_TYPE_DOT:
      mag_matmul_dot(r, x, y);
      break;
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT:
      mag_matmul_gemv_vec_mat(r, x, y);
      break;
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
      mag_matmul_gemv_mat_vec(r, x, y);
      break;
    case MAG_MATMUL_TYPE_GEMM:
      mag_panic("NYI!");
      break;
    case MAG_MATMUL_TYPE_BMM:
      mag_panic("NYI!");
      break;
    default:
      mag_panic("NYI!");
  }
  return MAG_STATUS_OK;
}