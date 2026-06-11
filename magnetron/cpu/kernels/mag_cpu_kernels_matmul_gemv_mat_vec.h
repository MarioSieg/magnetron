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

typedef void (mag_gemv_mat_vec_kernel_contig_t)(int64_t M, int64_t K, void *r, const void *px, const void *py);
#define mag_gemv_mat_vec_kernel_contig_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_mat_vec_kernel_contig_##dtype(int64_t M, int64_t K, void *pr, const void *px, const void *py) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t i=0; i < M; ++i) { \
      float acc = 0.f; \
      for (int64_t j=0; j < K; ++j) \
        acc += TtoF32(x[i*K + j])*TtoF32(y[j]); \
      r[i] = F32toT(acc); \
    } \
  }
mag_gemv_mat_vec_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_mat_vec_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_mat_vec_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_mat_vec_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_mat_vec_kernel_contig_impl

typedef void (mag_gemv_mat_vec_kernel_strided_t)(int64_t M, int64_t K, void *r, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy);
#define mag_gemv_mat_vec_kernel_strided_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_mat_vec_kernel_strided_##dtype(int64_t M, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t i=0; i < M; ++i) { \
      float acc = 0.f; \
      for (int64_t j=0; j < K; ++j) \
        acc += TtoF32(x[i*sx0 + j*sx1])*TtoF32(y[j*sy]); \
      r[i] = F32toT(acc); \
    } \
  }
mag_gemv_mat_vec_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_mat_vec_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_mat_vec_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_mat_vec_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_mat_vec_kernel_strided_impl

static MAG_HOTPROC void mag_matmul_gemv_mat_vec_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t tc,
  int64_t M,
  int64_t K,
  void *pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy
) {
  static mag_gemv_mat_vec_kernel_contig_t *const kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_mat_vec_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_mat_vec_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_mat_vec_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_mat_vec_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_mat_vec_kernel_strided_t *const kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_mat_vec_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_mat_vec_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_mat_vec_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_mat_vec_kernel_strided_mag_float8_e4m3fn_t
  };
  int64_t chunk = (M+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(M, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Mt = end-start;
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  pr = (uint8_t *)pr + start*el;
  px = (const uint8_t *)px + start*sx0*el;
  if (sx0 == K && sx1 == 1 && sy == 1) /* Contig fast path */
    (*kernel_lut_contig[dtype])(Mt, K, pr, px, py);
  else
    (*kernel_lut_strided[dtype])(Mt, K, pr, px, py, sx0, sx1, sy);
}

static MAG_HOTPROC void mag_matmul_gemv_mat_vec(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t M = x->coords.shape[0];
  int64_t K = x->coords.shape[1];
  int64_t sx0 = x->coords.strides[0];
  int64_t sx1 = x->coords.strides[1];
  int64_t sy = y->coords.strides[0];
  void *pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemv_mat_vec_impl(r->dtype, payload->thread_idx, payload->thread_num, M, K, pr, px, sx0, sx1, py, sy);
}
