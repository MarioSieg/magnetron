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

typedef void (mag_gemv_vec_mat_kernel_contig_t)(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy);
#define mag_gemv_vec_mat_kernel_contig_impl(T, TtoF32, F32toT, LoadTtoF32, StoreF32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_contig_##T(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    int64_t j = 0; \
    for (; j+4*MAG_VF32_LANES-1 < N; j += 4*MAG_VF32_LANES) { \
      mag_vf32_t a0 = mag_vf32_zero(); \
      mag_vf32_t a1 = mag_vf32_zero(); \
      mag_vf32_t a2 = mag_vf32_zero(); \
      mag_vf32_t a3 = mag_vf32_zero(); \
      for (int64_t k=0; k < K; ++k) { \
        mag_vf32_t xv = mag_vf32_splat(TtoF32(x[k])); \
        const T *row = y + k*sy + j; \
        a0 = mag_vf32_fmadd(xv, LoadTtoF32(row + 0*MAG_VF32_LANES), a0); \
        a1 = mag_vf32_fmadd(xv, LoadTtoF32(row + 1*MAG_VF32_LANES), a1); \
        a2 = mag_vf32_fmadd(xv, LoadTtoF32(row + 2*MAG_VF32_LANES), a2); \
        a3 = mag_vf32_fmadd(xv, LoadTtoF32(row + 3*MAG_VF32_LANES), a3); \
      } \
      StoreF32toT(r + j + 0*MAG_VF32_LANES, a0); \
      StoreF32toT(r + j + 1*MAG_VF32_LANES, a1); \
      StoreF32toT(r + j + 2*MAG_VF32_LANES, a2); \
      StoreF32toT(r + j + 3*MAG_VF32_LANES, a3); \
    } \
    for (; j+MAG_VF32_LANES-1 < N; j += MAG_VF32_LANES) { \
      mag_vf32_t a0 = mag_vf32_zero(); \
      for (int64_t k=0; k < K; ++k) \
        a0 = mag_vf32_fmadd(mag_vf32_splat(TtoF32(x[k])), LoadTtoF32(y + k*sy + j), a0); \
      StoreF32toT(r + j, a0); \
    } \
    for (; j < N; ++j) { \
      float acc = 0.f; \
      for (int64_t k=0; k < K; ++k) \
        acc += TtoF32(x[k])*TtoF32(y[k*sy + j]); \
      r[j] = F32toT(acc); \
    } \
  }
mag_gemv_vec_mat_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu, mag_vf32_storeu)
mag_gemv_vec_mat_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16, mag_vf32_storeu_f16)
mag_gemv_vec_mat_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16, mag_vf32_storeu_bf16)
mag_gemv_vec_mat_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_storeu_float8_e4m3fn)
#undef mag_gemv_vec_mat_kernel_contig_impl

typedef void (mag_gemv_vec_mat_kernel_rhs_transposed_contig_t)(int64_t K, int64_t N, void *pr, const void *px, const void *py);
#define mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_rhs_transposed_contig_##T( \
    int64_t K, int64_t N, void *pr, const void *px, const void *py \
  ) { \
    T *restrict r = (T *)pr; \
    const T *restrict x = (const T *)px; \
    const T *restrict y = (const T *)py; \
    int64_t j = 0; \
    for (; j+3 < N; j += 4) { \
      const T *restrict pw0 = y + (j + 0)*K; \
      const T *restrict pw1 = y + (j + 1)*K; \
      const T *restrict pw2 = y + (j + 2)*K; \
      const T *restrict pw3 = y + (j + 3)*K; \
      const T *restrict xp = x; \
      mag_vf32_t vacc0 = mag_vf32_zero(); \
      mag_vf32_t vacc1 = mag_vf32_zero(); \
      mag_vf32_t vacc2 = mag_vf32_zero(); \
      mag_vf32_t vacc3 = mag_vf32_zero(); \
      int64_t i=0; \
      for (; i+MAG_VF32_LANES-1 < K; i += MAG_VF32_LANES) { \
        mag_vf32_t xv = LoadTtoF32(xp); \
        vacc0 = mag_vf32_fmadd(xv, LoadTtoF32(pw0), vacc0); \
        vacc1 = mag_vf32_fmadd(xv, LoadTtoF32(pw1), vacc1); \
        vacc2 = mag_vf32_fmadd(xv, LoadTtoF32(pw2), vacc2); \
        vacc3 = mag_vf32_fmadd(xv, LoadTtoF32(pw3), vacc3); \
        xp += MAG_VF32_LANES; \
        pw0 += MAG_VF32_LANES; \
        pw1 += MAG_VF32_LANES; \
        pw2 += MAG_VF32_LANES; \
        pw3 += MAG_VF32_LANES; \
      } \
      float sacc0 = mag_vf32_reduce_add(vacc0); \
      float sacc1 = mag_vf32_reduce_add(vacc1); \
      float sacc2 = mag_vf32_reduce_add(vacc2); \
      float sacc3 = mag_vf32_reduce_add(vacc3); \
      for (; i < K; ++i) { \
        float xv = TtoF32(x[i]); \
        sacc0 += xv*TtoF32((y + (j + 0)*K)[i]); \
        sacc1 += xv*TtoF32((y + (j + 1)*K)[i]); \
        sacc2 += xv*TtoF32((y + (j + 2)*K)[i]); \
        sacc3 += xv*TtoF32((y + (j + 3)*K)[i]); \
      } \
      r[j+0] = F32toT(sacc0); \
      r[j+1] = F32toT(sacc1); \
      r[j+2] = F32toT(sacc2); \
      r[j+3] = F32toT(sacc3); \
    } \
    for (; j < N; ++j) { \
      const T *restrict wp = y + j*K; \
      const T *restrict xp = x; \
      mag_vf32_t acc = mag_vf32_zero(); \
      int64_t i = 0; \
      for (; i+MAG_VF32_LANES-1 < K; i += MAG_VF32_LANES) { \
        acc = mag_vf32_fmadd(LoadTtoF32(xp), LoadTtoF32(wp), acc); \
        xp += MAG_VF32_LANES; \
        wp += MAG_VF32_LANES; \
      } \
      float s = mag_vf32_reduce_add(acc); \
      for (; i < K; ++i) \
        s += TtoF32(x[i])*TtoF32((y + j*K)[i]); \
      r[j] = F32toT(s); \
    } \
  }
mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl
static MAG_HOTPROC void mag_gemv_vec_mat_kernel_rhs_transposed_contig_mag_bfloat16_t(int64_t K, int64_t N, void *pr, const void *px, const void *py) {
  mag_bfloat16_t *restrict r = (mag_bfloat16_t *)pr;
  const mag_bfloat16_t *x = (const mag_bfloat16_t *)px;
  const mag_bfloat16_t *y = (const mag_bfloat16_t *)py;
  int64_t j = 0;
  for (; j+3 < N; j += 4) {
    const mag_bfloat16_t *w0 = y + (j+0)*K, *w1 = y + (j+1)*K, *w2 = y + (j+2)*K, *w3 = y + (j+3)*K;
    mag_vf32_t a0 = mag_vf32_zero(), a1 = mag_vf32_zero(), a2 = mag_vf32_zero(), a3 = mag_vf32_zero();
    int64_t i = 0;
    for (; i+MAG_VBF16_LANES-1 < K; i += MAG_VBF16_LANES) {
      mag_vbf16_t xv = mag_vbf16_loadu(x+i);
      a0 = mag_vf32_dpbf16(a0, xv, mag_vbf16_loadu(w0+i));
      a1 = mag_vf32_dpbf16(a1, xv, mag_vbf16_loadu(w1+i));
      a2 = mag_vf32_dpbf16(a2, xv, mag_vbf16_loadu(w2+i));
      a3 = mag_vf32_dpbf16(a3, xv, mag_vbf16_loadu(w3+i));
    }
    float s0 = mag_vf32_reduce_add(a0), s1 = mag_vf32_reduce_add(a1), s2 = mag_vf32_reduce_add(a2), s3 = mag_vf32_reduce_add(a3);
    for (; i < K; ++i) {
      float xv = mag_bfloat16_to_float32(x[i]);
      s0 += xv*mag_bfloat16_to_float32(w0[i]);
      s1 += xv*mag_bfloat16_to_float32(w1[i]);
      s2 += xv*mag_bfloat16_to_float32(w2[i]);
      s3 += xv*mag_bfloat16_to_float32(w3[i]);
    }
    r[j+0] = mag_float32_to_bfloat16(s0);
    r[j+1] = mag_float32_to_bfloat16(s1);
    r[j+2] = mag_float32_to_bfloat16(s2);
    r[j+3] = mag_float32_to_bfloat16(s3);
  }
  for (; j < N; ++j) r[j] = mag_float32_to_bfloat16(mag_bf16_dot(x, y + j*K, K));
}

typedef void (mag_gemv_vec_mat_kernel_strided_t)(int64_t K, int64_t N, void *r, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1);
#define mag_gemv_vec_mat_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_strided_##T(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
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

static MAG_HOTPROC void mag_matmul_gemv_vec_mat_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t tc,
  int64_t N,
  int64_t K,
  void *restrict pr,
  const void *px, int64_t sx,
  const void *py, int64_t sy0, int64_t sy1
) {
  static mag_gemv_vec_mat_kernel_contig_t *const kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_vec_mat_kernel_rhs_transposed_contig_t *const kernel_lut_rhs_transposed_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_kernel_rhs_transposed_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_kernel_rhs_transposed_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_kernel_rhs_transposed_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_kernel_rhs_transposed_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_vec_mat_kernel_strided_t *const kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_kernel_strided_mag_float8_e4m3fn_t
  };
  int64_t nr = 4*MAG_VF32_LANES;
  int64_t chunk = (N + tc - 1) / tc;
  chunk = ((chunk + nr - 1) / nr) * nr;
  int64_t start = ti * chunk;
  int64_t end = mag_xmin(N, start + chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Nt = end-start;
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  pr = (uint8_t *)pr + start*el;
  py = (const uint8_t *)py + start*sy1*el;
  if (sx == 1 && sy0 == N && sy1 == 1) /* Contig fast path */
    (*kernel_lut_contig[dtype])(K, Nt, pr, px, py, N);
  else if (sx == 1 && sy0 == 1 && sy1 == K)
    (*kernel_lut_rhs_transposed_contig[dtype])(K, Nt, pr, px, py);
  else
    (*kernel_lut_strided[dtype])(K, Nt, pr, px, py, sx, sy0, sy1);
}

static MAG_HOTPROC void mag_matmul_gemv_vec_mat(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t sx = x->meta.coords.strides[0];
  int64_t sy0 = y->meta.coords.strides[0];
  int64_t sy1 = y->meta.coords.strides[1];
  int64_t K = x->meta.coords.shape[0];
  int64_t N = y->meta.coords.shape[1];
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemv_vec_mat_impl(r->meta.dtype, payload->thread_idx, payload->thread_num, N, K, pr, px, sx, py, sy0, sy1);
}
