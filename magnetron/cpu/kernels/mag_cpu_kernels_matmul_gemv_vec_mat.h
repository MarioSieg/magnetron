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
#define mag_gemv_vec_mat_kernel_contig_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_kernel_contig_##T(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
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
mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemv_vec_mat_kernel_rhs_transposed_contig_impl

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
  int64_t nr = 8;
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
  int64_t sx = x->coords.strides[0];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t K = x->coords.shape[0];
  int64_t N = y->coords.shape[1];
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemv_vec_mat_impl(r->dtype, payload->thread_idx, payload->thread_num, N, K, pr, px, sx, py, sy0, sy1);
}

typedef void (mag_gemv_vec_mat_fp8w_scaled_kernel_contig_t)(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy, float scale);
#define mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_fp8w_scaled_kernel_contig_##T(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sy, float scale) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const mag_float8_e4m3fn_t *y = (const mag_float8_e4m3fn_t *)py; \
    for (int64_t j=0; j < N; ++j) { \
      float acc = 0.f; \
      for (int64_t i=0; i < K; ++i) \
        acc += TtoF32(x[i])*mag_float8_e4m3fn_to_float32(y[i*sy + j]); \
      r[j] = F32toT(acc*scale); \
    } \
  }
mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_vec_mat_fp8w_scaled_kernel_contig_impl

typedef void (mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_t)(
  int64_t K, int64_t N, void *pr, const void *px, const void *py, float scale
);

#ifndef mag_vf32_add
#define mag_vf32_add(a, b) ((a) + (b))
#endif

#define MAG_VF32_SUM3(a, b, c) mag_vf32_add(mag_vf32_add((a), (b)), (c))

#define mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_##T( \
    int64_t K, int64_t N, void *pr, const void *px, const void *py, float scale \
  ) { \
    enum { VL = MAG_VF32_LANES }; \
    T *restrict r = (T *)pr; \
    const T *restrict x = (const T *)px; \
    const mag_float8_e4m3fn_t *restrict y = (const mag_float8_e4m3fn_t *)py; \
    int64_t j = 0; \
    for (; j + 7 < N; j += 8) { \
      const mag_float8_e4m3fn_t *restrict pw0 = y + (j + 0) * K; \
      const mag_float8_e4m3fn_t *restrict pw1 = y + (j + 1) * K; \
      const mag_float8_e4m3fn_t *restrict pw2 = y + (j + 2) * K; \
      const mag_float8_e4m3fn_t *restrict pw3 = y + (j + 3) * K; \
      const mag_float8_e4m3fn_t *restrict pw4 = y + (j + 4) * K; \
      const mag_float8_e4m3fn_t *restrict pw5 = y + (j + 5) * K; \
      const mag_float8_e4m3fn_t *restrict pw6 = y + (j + 6) * K; \
      const mag_float8_e4m3fn_t *restrict pw7 = y + (j + 7) * K; \
      const T *restrict xp = x; \
      mag_vf32_t a00 = mag_vf32_zero(), a01 = mag_vf32_zero(), a02 = mag_vf32_zero(); \
      mag_vf32_t a10 = mag_vf32_zero(), a11 = mag_vf32_zero(), a12 = mag_vf32_zero(); \
      mag_vf32_t a20 = mag_vf32_zero(), a21 = mag_vf32_zero(), a22 = mag_vf32_zero(); \
      mag_vf32_t a30 = mag_vf32_zero(), a31 = mag_vf32_zero(), a32 = mag_vf32_zero(); \
      mag_vf32_t a40 = mag_vf32_zero(), a41 = mag_vf32_zero(), a42 = mag_vf32_zero(); \
      mag_vf32_t a50 = mag_vf32_zero(), a51 = mag_vf32_zero(), a52 = mag_vf32_zero(); \
      mag_vf32_t a60 = mag_vf32_zero(), a61 = mag_vf32_zero(), a62 = mag_vf32_zero(); \
      mag_vf32_t a70 = mag_vf32_zero(), a71 = mag_vf32_zero(), a72 = mag_vf32_zero(); \
      int64_t i = 0; \
      for (; i + 3 * VL - 1 < K; i += 3 * VL) { \
        mag_vf32_t x0 = LoadTtoF32(xp + 0 * VL); \
        mag_vf32_t x1 = LoadTtoF32(xp + 1 * VL); \
        mag_vf32_t x2 = LoadTtoF32(xp + 2 * VL); \
        a00 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw0 + 0 * VL), a00); \
        a01 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw0 + 1 * VL), a01); \
        a02 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw0 + 2 * VL), a02); \
        a10 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw1 + 0 * VL), a10); \
        a11 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw1 + 1 * VL), a11); \
        a12 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw1 + 2 * VL), a12); \
        a20 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw2 + 0 * VL), a20); \
        a21 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw2 + 1 * VL), a21); \
        a22 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw2 + 2 * VL), a22); \
        a30 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw3 + 0 * VL), a30); \
        a31 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw3 + 1 * VL), a31); \
        a32 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw3 + 2 * VL), a32); \
        a40 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw4 + 0 * VL), a40); \
        a41 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw4 + 1 * VL), a41); \
        a42 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw4 + 2 * VL), a42); \
        a50 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw5 + 0 * VL), a50); \
        a51 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw5 + 1 * VL), a51); \
        a52 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw5 + 2 * VL), a52); \
        a60 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw6 + 0 * VL), a60); \
        a61 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw6 + 1 * VL), a61); \
        a62 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw6 + 2 * VL), a62); \
        a70 = mag_vf32_fmadd(x0, mag_vf32_loadu_float8_e4m3fn(pw7 + 0 * VL), a70); \
        a71 = mag_vf32_fmadd(x1, mag_vf32_loadu_float8_e4m3fn(pw7 + 1 * VL), a71); \
        a72 = mag_vf32_fmadd(x2, mag_vf32_loadu_float8_e4m3fn(pw7 + 2 * VL), a72); \
        xp  += 3 * VL; \
        pw0 += 3 * VL; pw1 += 3 * VL; pw2 += 3 * VL; pw3 += 3 * VL; \
        pw4 += 3 * VL; pw5 += 3 * VL; pw6 += 3 * VL; pw7 += 3 * VL; \
      } \
      for (; i + VL - 1 < K; i += VL) { \
        mag_vf32_t xv = LoadTtoF32(xp); \
        a00 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw0), a00); \
        a10 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw1), a10); \
        a20 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw2), a20); \
        a30 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw3), a30); \
        a40 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw4), a40); \
        a50 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw5), a50); \
        a60 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw6), a60); \
        a70 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(pw7), a70); \
        xp += VL; \
        pw0 += VL; pw1 += VL; pw2 += VL; pw3 += VL; \
        pw4 += VL; pw5 += VL; pw6 += VL; pw7 += VL; \
      } \
      float s0 = mag_vf32_reduce_add(MAG_VF32_SUM3(a00, a01, a02)); \
      float s1 = mag_vf32_reduce_add(MAG_VF32_SUM3(a10, a11, a12)); \
      float s2 = mag_vf32_reduce_add(MAG_VF32_SUM3(a20, a21, a22)); \
      float s3 = mag_vf32_reduce_add(MAG_VF32_SUM3(a30, a31, a32)); \
      float s4 = mag_vf32_reduce_add(MAG_VF32_SUM3(a40, a41, a42)); \
      float s5 = mag_vf32_reduce_add(MAG_VF32_SUM3(a50, a51, a52)); \
      float s6 = mag_vf32_reduce_add(MAG_VF32_SUM3(a60, a61, a62)); \
      float s7 = mag_vf32_reduce_add(MAG_VF32_SUM3(a70, a71, a72)); \
      for (; i < K; ++i) { \
        float xv = TtoF32(x[i]); \
        s0 += xv * mag_float8_e4m3fn_to_float32(y[(j + 0) * K + i]); \
        s1 += xv * mag_float8_e4m3fn_to_float32(y[(j + 1) * K + i]); \
        s2 += xv * mag_float8_e4m3fn_to_float32(y[(j + 2) * K + i]); \
        s3 += xv * mag_float8_e4m3fn_to_float32(y[(j + 3) * K + i]); \
        s4 += xv * mag_float8_e4m3fn_to_float32(y[(j + 4) * K + i]); \
        s5 += xv * mag_float8_e4m3fn_to_float32(y[(j + 5) * K + i]); \
        s6 += xv * mag_float8_e4m3fn_to_float32(y[(j + 6) * K + i]); \
        s7 += xv * mag_float8_e4m3fn_to_float32(y[(j + 7) * K + i]); \
      } \
      r[j + 0] = F32toT(s0 * scale); \
      r[j + 1] = F32toT(s1 * scale); \
      r[j + 2] = F32toT(s2 * scale); \
      r[j + 3] = F32toT(s3 * scale); \
      r[j + 4] = F32toT(s4 * scale); \
      r[j + 5] = F32toT(s5 * scale); \
      r[j + 6] = F32toT(s6 * scale); \
      r[j + 7] = F32toT(s7 * scale); \
    } \
    for (; j < N; ++j) { \
      const mag_float8_e4m3fn_t *restrict wp = y + j * K; \
      const T *restrict xp = x; \
      mag_vf32_t a0 = mag_vf32_zero(); \
      mag_vf32_t a1 = mag_vf32_zero(); \
      mag_vf32_t a2 = mag_vf32_zero(); \
      int64_t i = 0; \
      for (; i + 3 * VL - 1 < K; i += 3 * VL) { \
        a0 = mag_vf32_fmadd(LoadTtoF32(xp + 0 * VL), mag_vf32_loadu_float8_e4m3fn(wp + 0 * VL), a0); \
        a1 = mag_vf32_fmadd(LoadTtoF32(xp + 1 * VL), mag_vf32_loadu_float8_e4m3fn(wp + 1 * VL), a1); \
        a2 = mag_vf32_fmadd(LoadTtoF32(xp + 2 * VL), mag_vf32_loadu_float8_e4m3fn(wp + 2 * VL), a2); \
        xp += 3 * VL; \
        wp += 3 * VL; \
      } \
      for (; i + VL - 1 < K; i += VL) { \
        a0 = mag_vf32_fmadd(LoadTtoF32(xp), mag_vf32_loadu_float8_e4m3fn(wp), a0); \
        xp += VL; \
        wp += VL; \
      } \
      float s = mag_vf32_reduce_add(MAG_VF32_SUM3(a0, a1, a2)); \
      for (; i < K; ++i) \
        s += TtoF32(x[i]) * mag_float8_e4m3fn_to_float32(y[j * K + i]); \
      r[j] = F32toT(s * scale); \
    } \
  }

mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl(
  float,
  mag_cvt_nop,
  mag_cvt_nop,
  mag_vf32_loadu
)

mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl(
  mag_float16_t,
  mag_float16_to_float32,
  mag_float32_to_float16,
  mag_vf32_loadu_f16
)

mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl(
  mag_bfloat16_t,
  mag_bfloat16_to_float32,
  mag_float32_to_bfloat16,
  mag_vf32_loadu_bf16
)

mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl(
  mag_float8_e4m3fn_t,
  mag_float8_e4m3fn_to_float32,
  mag_float32_to_float8_e4m3fn,
  mag_vf32_loadu_float8_e4m3fn
)

#undef mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_impl
#undef MAG_VF32_SUM3

typedef void (mag_gemv_vec_mat_fp8w_scaled_kernel_strided_t)(int64_t K, int64_t N, void *r, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1, float scale);
#define mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemv_vec_mat_fp8w_scaled_kernel_strided_##T(int64_t K, int64_t N, void *pr, const void *px, const void *py, int64_t sx, int64_t sy0, int64_t sy1, float scale) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const mag_float8_e4m3fn_t *y = (const mag_float8_e4m3fn_t *)py; \
    for (int64_t j=0; j < N; ++j) { \
      float acc = 0.f; \
      for (int64_t i=0; i < K; ++i) \
        acc += TtoF32(x[i*sx])*mag_float8_e4m3fn_to_float32(y[i*sy0 + j*sy1]); \
      r[j] = F32toT(acc*scale); \
    } \
  }
mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemv_vec_mat_fp8w_scaled_kernel_strided_impl

static MAG_HOTPROC void mag_matmul_gemv_vec_mat_fp8w_scaled_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t tc,
  int64_t N,
  int64_t K,
  void *restrict pr,
  const void *px, int64_t sx,
  const void *py, int64_t sy0, int64_t sy1,
  float scale
) {
  static mag_gemv_vec_mat_fp8w_scaled_kernel_contig_t *const kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_fp8w_scaled_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_fp8w_scaled_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_t *const kernel_lut_rhs_transposed_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_mag_float8_e4m3fn_t
  };
  static mag_gemv_vec_mat_fp8w_scaled_kernel_strided_t *const kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemv_vec_mat_fp8w_scaled_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemv_vec_mat_fp8w_scaled_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemv_vec_mat_fp8w_scaled_kernel_strided_mag_float8_e4m3fn_t
  };
  int64_t nr = 8;
  int64_t chunk = (N + tc - 1) / tc;
  chunk = ((chunk + nr - 1) / nr) * nr;
  int64_t start = ti * chunk;
  int64_t end = mag_xmin(N, start + chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Nt = end-start;
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  pr = (uint8_t *)pr + start*el;
  py = (const uint8_t *)py + start*sy1;
  if (sx == 1 && sy0 == N && sy1 == 1) /* Contig fast path */
    (*kernel_lut_contig[dtype])(K, Nt, pr, px, py, N, scale);
  else if (sx == 1 && sy0 == 1 && sy1 == K)
    (*kernel_lut_rhs_transposed_contig[dtype])(K, Nt, pr, px, py, scale);
  else
    (*kernel_lut_strided[dtype])(K, Nt, pr, px, py, sx, sy0, sy1, scale);
}

static MAG_HOTPROC void mag_matmul_gemv_vec_mat_fp8w_scaled(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  const mag_tensor_t *s = payload->cmd->in[2];
  int64_t sx = x->coords.strides[0];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t K = x->coords.shape[0];
  int64_t N = y->coords.shape[1];
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  float scale = *(const float *)mag_tensor_data_ptr(s);
  mag_matmul_gemv_vec_mat_fp8w_scaled_impl(r->dtype, payload->thread_idx, payload->thread_num, N, K, pr, px, sx, py, sy0, sy1, scale);
}

