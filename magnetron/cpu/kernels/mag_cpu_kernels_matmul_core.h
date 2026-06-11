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

static MAG_HOTPROC mag_status_t mag_matmul_generic(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_matmul_type_t type = mag_matmul_type_detect(payload->cmd->in[0], payload->cmd->in[1]);
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID: mag_contract(err, ERR_OPERATOR_IMPOSSIBLE, {}, type != MAG_MATMUL_TYPE_INVALID, "Invalid matmul type detected for the given input tensors"); break;
    case MAG_MATMUL_TYPE_DOT: mag_matmul_dot(payload); break;
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT: mag_matmul_gemv_vec_mat(payload); break;
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC: mag_matmul_gemv_mat_vec(payload); break;
    case MAG_MATMUL_TYPE_GEMM: mag_matmul_gemm(payload); break;
    case MAG_MATMUL_TYPE_BMM_DOT:
    case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT:
    case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:
    case MAG_MATMUL_TYPE_BMM_GEMM:
      mag_matmul_bmm(payload, type); break;
    default: mag_panic("NYI!");
  }
  return MAG_STATUS_OK;
}

static MAG_HOTPROC mag_status_t mag_matmul_fp8w_scaled(mag_error_t *err, const mag_kernel_payload_t *payload) {
  /*
  (void)err;
  mag_matmul_type_t type = mag_matmul_type_detect(payload->cmd->in[0], payload->cmd->in[1]);
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID: mag_contract(err, ERR_OPERATOR_IMPOSSIBLE, {}, type != MAG_MATMUL_TYPE_INVALID, "Invalid matmul type detected for the given input tensors"); break;
    case MAG_MATMUL_TYPE_DOT: mag_matmul_fp8w_scaled_dot(payload); break;
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT: mag_matmul_fp8w_scaled_gemv_vec_mat(payload); break;
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC: mag_matmul_fp8w_scaled_gemv_mat_vec(payload); break;
    case MAG_MATMUL_TYPE_GEMM: mag_matmul_fp8w_scaled_gemm(payload); break;
    case MAG_MATMUL_TYPE_BMM: mag_matmul_fp8w_scaled_bmm(payload); break;
    default: mag_panic("NYI!");
  }*/
  return MAG_STATUS_OK;
}
