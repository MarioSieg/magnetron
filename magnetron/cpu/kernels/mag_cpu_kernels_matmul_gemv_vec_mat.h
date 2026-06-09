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

typedef void (mag_gemv_vec_mat_kernel_contig_t)(int64_t K, int64_t N, void *r, const void *px, const void *py, int64_t sy);
#define mag_gemv_vec_mat_kernel_contig_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_contig_##dtype(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t j=0; j < N; ++j) { \
      float acc = 0.f; \
      for (int64_t i=0; i < K; ++i) \
        acc += TtoF32(x[i])*TtoF32(y[i*sy + j]); \
      r[j] = F32toT(acc); \
    } \
  }
mag_gemv_vec_mat_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_vec_mat_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_vec_mat_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_vec_mat_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_vec_mat_kernel_contig_impl

typedef void (mag_gemv_vec_mat_kernel_strided_t)(int64_t K, int64_t N, void *r, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1);
#define mag_gemv_vec_mat_kernel_strided_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_strided_##dtype(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t j=0; j < N; ++j) { \
      float acc = 0.f; \
      for (int64_t i=0; i < K; ++i) \
        acc += TtoF32(x[i*sx])*TtoF32(y[i*sy0 + j*sy1]); \
      r[j] = F32toT(acc); \
    } \
  }
mag_gemv_vec_mat_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_vec_mat_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_vec_mat_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_vec_mat_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_vec_mat_kernel_strided_impl

static void mag_matmul_gemv_vec_mat(const mag_kernel_payload_t *payload) {
  static mag_gemv_vec_mat_kernel_contig_t *const kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_vec_mat_kernel_strided_t *const kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_kernel_strided_mag_float8_e4m3fn_t
  };
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t sx = x->coords.strides[0];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t K = x->coords.shape[0];
  int64_t N = y->coords.shape[1];
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t chunk = (N+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(N, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Nt = end-start;
  int64_t el = (int64_t)mag_type_trait(r->dtype)->size;
  void *pr = (uint8_t *)mag_tensor_data_ptr_mut(r) + start*el;
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const uint8_t *)mag_tensor_data_ptr(y) + start*sy1*el;
  if (sx == 1 && sy0 == N && sy1 == 1) /* Contig fast path */
    (*kernel_lut_contig[r->dtype])(K, Nt, pr, px, py, N);
  else
    (*kernel_lut_strided[r->dtype])(K, Nt, pr, px, py, sx, sy0, sy1);
}
