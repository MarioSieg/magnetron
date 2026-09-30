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

mag_static_assert(MAG_DTYPE_FLOAT32 == 0); /* We use LUTs for the dtype kernels from 0..=3 */
mag_static_assert(MAG_DTYPE_FLOAT8_E4M3FN == 3);

static MAG_HOTPROC bool mag_matmul_vec_mat_via_gemm(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t xr = x->meta.coords.rank;
  int64_t yr = y->meta.coords.rank;
  if (yr < 2 || x->meta.coords.strides[xr-1] != 1) return false;
  int64_t K = x->meta.coords.shape[xr-1];
  int64_t N = y->meta.coords.shape[yr-1];
  int64_t sy0 = y->meta.coords.strides[yr-2];
  int64_t sy1 = y->meta.coords.strides[yr-1];
  if (!((sy1 == K && sy0 == 1) || (sy0 == N && sy1 == 1))) return false;
  if (xr >= 2) {
    mag_matmul_bmm_gemm(payload);
    return true;
  }
  if (yr != 2) return false;
  mag_matmul_gemm_impl(
    payload, r->meta.dtype, 1, N, K, 1, NULL, NULL,
    (void *)mag_tensor_data_ptr_mut(r),
    (const void *)mag_tensor_data_ptr(x), K, 1,
    (const void *)mag_tensor_data_ptr(y), sy0, sy1
  );
  return true;
}

static MAG_HOTPROC mag_status_t mag_matmul_generic(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_kernel_payload_t solo;
  if (mag_accel_matmul_supported(payload->cmd->in[0], payload->cmd->in[1], payload->cmd->out[0])) {
    if (payload->thread_idx != 0) return MAG_OK;
    if (mag_accel_matmul(payload->cmd->out[0], payload->cmd->in[0], payload->cmd->in[1])) return MAG_OK;
    solo = *payload;
    solo.thread_num = 1;
    payload = &solo;
  }
  mag_matmul_type_t type = mag_matmul_type_detect(payload->cmd->in[0], payload->cmd->in[1]);
  switch (type) {
    case MAG_MATMUL_TYPE_DOT: mag_matmul_dot(payload); break;
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT: if (!mag_matmul_vec_mat_via_gemm(payload)) mag_matmul_gemv_vec_mat(payload); break;
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC: mag_matmul_gemv_mat_vec(payload); break;
    case MAG_MATMUL_TYPE_GEMM: mag_matmul_gemm(payload); break;
    case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT: if (!mag_matmul_vec_mat_via_gemm(payload)) mag_matmul_bmm(payload, type); break;
    case MAG_MATMUL_TYPE_BMM_DOT:
    case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:
    case MAG_MATMUL_TYPE_BMM_GEMM: mag_matmul_bmm(payload, type); break;
    default: mag_panic("matmul: unsupported kernel type %d.", (int)type);
  }
  return MAG_OK;
}
