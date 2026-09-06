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
#define mag_gemv_mat_vec_kernel_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemv_mat_vec_kernel_contig_##T(int64_t M, int64_t K, void *pr, const void *px, const void *py) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    int64_t i = 0; \
    for (; i+3 < M; i += 4) { \
      const T *x0 = x + (i+0)*K; \
      const T *x1 = x + (i+1)*K; \
      const T *x2 = x + (i+2)*K; \
      const T *x3 = x + (i+3)*K; \
      mag_vf32_t a0 = mag_vf32_zero(); \
      mag_vf32_t a1 = mag_vf32_zero(); \
      mag_vf32_t a2 = mag_vf32_zero(); \
      mag_vf32_t a3 = mag_vf32_zero(); \
      int64_t k = 0; \
      for (; k+MAG_VF32_LANES-1 < K; k += MAG_VF32_LANES) { \
        mag_vf32_t yv = LoadTtoF32(y + k); \
        a0 = mag_vf32_fmadd(LoadTtoF32(x0 + k), yv, a0); \
        a1 = mag_vf32_fmadd(LoadTtoF32(x1 + k), yv, a1); \
        a2 = mag_vf32_fmadd(LoadTtoF32(x2 + k), yv, a2); \
        a3 = mag_vf32_fmadd(LoadTtoF32(x3 + k), yv, a3); \
      } \
      float s0 = mag_vf32_reduce_add(a0); \
      float s1 = mag_vf32_reduce_add(a1); \
      float s2 = mag_vf32_reduce_add(a2); \
      float s3 = mag_vf32_reduce_add(a3); \
      for (; k < K; ++k) { \
        float yv = TtoF32(y[k]); \
        s0 += TtoF32(x0[k])*yv; \
        s1 += TtoF32(x1[k])*yv; \
        s2 += TtoF32(x2[k])*yv; \
        s3 += TtoF32(x3[k])*yv; \
      } \
      r[i+0] = F32toT(s0); \
      r[i+1] = F32toT(s1); \
      r[i+2] = F32toT(s2); \
      r[i+3] = F32toT(s3); \
    } \
    for (; i < M; ++i) { \
      const T *x0 = x + i*K; \
      mag_vf32_t a0 = mag_vf32_zero(); \
      int64_t k = 0; \
      for (; k+MAG_VF32_LANES-1 < K; k += MAG_VF32_LANES) \
        a0 = mag_vf32_fmadd(LoadTtoF32(x0 + k), LoadTtoF32(y + k), a0); \
      float s0 = mag_vf32_reduce_add(a0); \
      for (; k < K; ++k) s0 += TtoF32(x0[k])*TtoF32(y[k]); \
      r[i] = F32toT(s0); \
    } \
  }
mag_gemv_mat_vec_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gemv_mat_vec_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gemv_mat_vec_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemv_mat_vec_kernel_contig_impl

static MAG_HOTPROC void mag_gemv_mat_vec_kernel_contig_mag_bfloat16_t(int64_t M, int64_t K, void *pr, const void *px, const void *py) {
  mag_bfloat16_t *restrict r = (mag_bfloat16_t *)pr;
  const mag_bfloat16_t *x = (const mag_bfloat16_t *)px;
  const mag_bfloat16_t *y = (const mag_bfloat16_t *)py;
  int64_t i = 0;
  for (; i+3 < M; i += 4) {
    const mag_bfloat16_t *x0 = x + (i+0)*K, *x1 = x + (i+1)*K, *x2 = x + (i+2)*K, *x3 = x + (i+3)*K;
    mag_vf32_t a0 = mag_vf32_zero(), a1 = mag_vf32_zero(), a2 = mag_vf32_zero(), a3 = mag_vf32_zero();
    int64_t k = 0;
    for (; k+MAG_VBF16_LANES-1 < K; k += MAG_VBF16_LANES) {
      mag_vbf16_t yv = mag_vbf16_loadu(y+k);
      a0 = mag_vf32_dpbf16(a0, mag_vbf16_loadu(x0+k), yv);
      a1 = mag_vf32_dpbf16(a1, mag_vbf16_loadu(x1+k), yv);
      a2 = mag_vf32_dpbf16(a2, mag_vbf16_loadu(x2+k), yv);
      a3 = mag_vf32_dpbf16(a3, mag_vbf16_loadu(x3+k), yv);
    }
    float s0 = mag_vf32_reduce_add(a0), s1 = mag_vf32_reduce_add(a1), s2 = mag_vf32_reduce_add(a2), s3 = mag_vf32_reduce_add(a3);
    for (; k < K; ++k) {
      float yv = mag_bfloat16_to_float32(y[k]);
      s0 += mag_bfloat16_to_float32(x0[k])*yv;
      s1 += mag_bfloat16_to_float32(x1[k])*yv;
      s2 += mag_bfloat16_to_float32(x2[k])*yv;
      s3 += mag_bfloat16_to_float32(x3[k])*yv;
    }
    r[i+0] = mag_float32_to_bfloat16(s0);
    r[i+1] = mag_float32_to_bfloat16(s1);
    r[i+2] = mag_float32_to_bfloat16(s2);
    r[i+3] = mag_float32_to_bfloat16(s3);
  }
  for (; i < M; ++i) r[i] = mag_float32_to_bfloat16(mag_bf16_dot(x + i*K, y, K));
}

typedef void (mag_gemv_mat_vec_kernel_strided_t)(int64_t M, int64_t K, void *r, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy);
#define mag_gemv_mat_vec_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_mat_vec_kernel_strided_##T(int64_t M, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
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
  int64_t end = mag_vmin(M, start+chunk);
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
  int64_t M = x->meta.coords.shape[0];
  int64_t K = x->meta.coords.shape[1];
  int64_t sx0 = x->meta.coords.strides[0];
  int64_t sx1 = x->meta.coords.strides[1];
  int64_t sy = y->meta.coords.strides[0];
  void *pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemv_mat_vec_impl(r->meta.dtype, payload->thread_idx, payload->thread_num, M, K, pr, px, sx0, sx1, py, sy);
}
