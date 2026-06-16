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

typedef void (mag_gemm_kernel_contig_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py);
#define mag_gemm_kernel_contig_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_kernel_contig_##T(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*K + k])*TtoF32(y[k*N + j]); \
          r[i*N + j] = F32toT(acc); \
        } \
      } \
  }
mag_gemm_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_kernel_contig_impl

typedef void (mag_gemm_kernel_rhs_transposed_contig_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py);
#define mag_gemm_kernel_rhs_transposed_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemm_kernel_rhs_transposed_contig_##T( \
    int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py \
  ) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    for (int64_t i = 0; i < M; ++i) { \
      const T *xrow = x + i*K; \
      T *restrict rrow = r + i*N; \
      int64_t j = 0; \
      for (; j + 3 < N; j += 4) { \
        const T *w0 = y + (j + 0)*K; \
        const T *w1 = y + (j + 1)*K; \
        const T *w2 = y + (j + 2)*K; \
        const T *w3 = y + (j + 3)*K; \
        mag_vf32_t acc0 = mag_vf32_zero(); \
        mag_vf32_t acc1 = mag_vf32_zero(); \
        mag_vf32_t acc2 = mag_vf32_zero(); \
        mag_vf32_t acc3 = mag_vf32_zero(); \
        int64_t k = 0; \
        for (; k + MAG_VF32_LANES-1 < K; k += MAG_VF32_LANES) { \
          mag_vf32_t xv = LoadTtoF32(xrow + k); \
          acc0 = mag_vf32_fmadd(xv, LoadTtoF32(w0 + k), acc0); \
          acc1 = mag_vf32_fmadd(xv, LoadTtoF32(w1 + k), acc1); \
          acc2 = mag_vf32_fmadd(xv, LoadTtoF32(w2 + k), acc2); \
          acc3 = mag_vf32_fmadd(xv, LoadTtoF32(w3 + k), acc3); \
        } \
        float s0 = mag_vf32_reduce_add(acc0); \
        float s1 = mag_vf32_reduce_add(acc1); \
        float s2 = mag_vf32_reduce_add(acc2); \
        float s3 = mag_vf32_reduce_add(acc3); \
        for (; k < K; ++k) { \
          float xv = TtoF32(xrow[k]); \
          s0 += xv * TtoF32(w0[k]); \
          s1 += xv * TtoF32(w1[k]); \
          s2 += xv * TtoF32(w2[k]); \
          s3 += xv * TtoF32(w3[k]); \
        } \
        rrow[j + 0] = F32toT(s0); \
        rrow[j + 1] = F32toT(s1); \
        rrow[j + 2] = F32toT(s2); \
        rrow[j + 3] = F32toT(s3); \
      } \
      for (; j < N; ++j) { \
        const T *restrict w = y + j*K; \
        mag_vf32_t acc = mag_vf32_zero(); \
        int64_t k = 0; \
        for (; k + MAG_VF32_LANES - 1 < K; k += MAG_VF32_LANES) \
          acc = mag_vf32_fmadd(LoadTtoF32(xrow + k), LoadTtoF32(w + k), acc); \
        float s = mag_vf32_reduce_add(acc); \
        for (; k < K; ++k) \
          s += TtoF32(xrow[k]) * TtoF32(w[k]); \
        rrow[j] = F32toT(s); \
      } \
    } \
  }
mag_gemm_kernel_rhs_transposed_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gemm_kernel_rhs_transposed_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gemm_kernel_rhs_transposed_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gemm_kernel_rhs_transposed_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemm_kernel_rhs_transposed_contig_impl

typedef void (mag_gemm_kernel_strided_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1);
#define mag_gemm_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_kernel_strided_##T(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*sx0 + k*sx1])*TtoF32(y[k*sy0 + j*sy1]); \
          r[i*N + j] = F32toT(acc); \
        } \
      } \
  }
mag_gemm_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_kernel_strided_impl

static MAG_HOTPROC void mag_matmul_gemm_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t tc,
  int64_t M,
  int64_t N,
  int64_t K,
  void *restrict pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy0, int64_t sy1
) {
  static mag_gemm_kernel_contig_t *const mag_gemm_kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemm_kernel_rhs_transposed_contig_t *const mag_gemm_kernel_lut_rhs_transposed_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_kernel_rhs_transposed_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_kernel_rhs_transposed_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_kernel_rhs_transposed_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_kernel_rhs_transposed_contig_mag_float8_e4m3fn_t
  };
  static mag_gemm_kernel_strided_t *const mag_gemm_kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_kernel_strided_mag_float8_e4m3fn_t
  };
  int64_t chunk = (M+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(M, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Mt = end-start;
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  pr = (uint8_t *)pr + start*N*el;
  px = (const uint8_t *)px + start*sx0*el;
  if (sx0 == K && sx1 == 1 && sy0 == N && sy1 == 1)
    (*mag_gemm_kernel_lut_contig[dtype])(Mt, N, K, pr, px, py);
  else if (sx0 == K && sx1 == 1 && sy0 == 1 && sy1 == K)
    (*mag_gemm_kernel_lut_rhs_transposed_contig[dtype])(Mt, N, K, pr, px, py);
  else
    (*mag_gemm_kernel_lut_strided[dtype])(Mt, N, K, pr, px, py, sx0, sx1, sy0, sy1);
}

static MAG_HOTPROC void mag_matmul_gemm(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t M = x->coords.shape[0];
  int64_t K = x->coords.shape[1];
  int64_t N = y->coords.shape[1];
  int64_t sx0 = x->coords.strides[0];
  int64_t sx1 = x->coords.strides[1];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemm_impl(r->dtype, ti, tc, M, N, K, pr, px, sx0, sx1, py, sy0, sy1);
}

typedef void (mag_gemm_fp8w_scaled_kernel_contig_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, float scale);
#define mag_gemm_fp8w_scaled_kernel_contig_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_fp8w_scaled_kernel_contig_##T(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, float scale) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const mag_float8_e4m3fn_t *y = (const mag_float8_e4m3fn_t *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*K + k])*mag_float8_e4m3fn_to_float32(y[k*N + j]); \
          r[i*N + j] = F32toT(acc*scale); \
        } \
      } \
  }
mag_gemm_fp8w_scaled_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_fp8w_scaled_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_fp8w_scaled_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_fp8w_scaled_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_fp8w_scaled_kernel_contig_impl

typedef void (mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, float scale);
#define mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_##T(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, float scale) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const mag_float8_e4m3fn_t *y = (const mag_float8_e4m3fn_t *)py; \
    for (int64_t i=0; i < M; ++i) { \
      const T *xrow = x + i*K; \
      T *restrict rrow = r + i*N; \
      int64_t j=0; \
      for (; j+3 < N; j += 4) { \
        const mag_float8_e4m3fn_t *w0 = y + (j + 0)*K; \
        const mag_float8_e4m3fn_t *w1 = y + (j + 1)*K; \
        const mag_float8_e4m3fn_t *w2 = y + (j + 2)*K; \
        const mag_float8_e4m3fn_t *w3 = y + (j + 3)*K; \
        mag_vf32_t acc0 = mag_vf32_zero(); \
        mag_vf32_t acc1 = mag_vf32_zero(); \
        mag_vf32_t acc2 = mag_vf32_zero(); \
        mag_vf32_t acc3 = mag_vf32_zero(); \
        int64_t k = 0; \
        for (; k + MAG_VF32_LANES-1 < K; k += MAG_VF32_LANES) { \
          mag_vf32_t xv = LoadTtoF32(xrow + k); \
          acc0 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(w0 + k), acc0); \
          acc1 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(w1 + k), acc1); \
          acc2 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(w2 + k), acc2); \
          acc3 = mag_vf32_fmadd(xv, mag_vf32_loadu_float8_e4m3fn(w3 + k), acc3); \
        } \
        float s0 = mag_vf32_reduce_add(acc0); \
        float s1 = mag_vf32_reduce_add(acc1); \
        float s2 = mag_vf32_reduce_add(acc2); \
        float s3 = mag_vf32_reduce_add(acc3); \
        for (; k < K; ++k) { \
          float xv = TtoF32(xrow[k]); \
          s0 += xv * mag_float8_e4m3fn_to_float32(w0[k]); \
          s1 += xv * mag_float8_e4m3fn_to_float32(w1[k]); \
          s2 += xv * mag_float8_e4m3fn_to_float32(w2[k]); \
          s3 += xv * mag_float8_e4m3fn_to_float32(w3[k]); \
        } \
        rrow[j+0] = F32toT(s0*scale); \
        rrow[j+1] = F32toT(s1*scale); \
        rrow[j+2] = F32toT(s2*scale); \
        rrow[j+3] = F32toT(s3*scale); \
      } \
      for (; j < N; ++j) { \
        const mag_float8_e4m3fn_t *restrict w = y + j*K; \
        mag_vf32_t acc = mag_vf32_zero(); \
        int64_t k = 0; \
        for (; k + MAG_VF32_LANES-1 < K; k += MAG_VF32_LANES) \
          acc = mag_vf32_fmadd(LoadTtoF32(xrow + k), mag_vf32_loadu_float8_e4m3fn(w + k), acc); \
        float s = mag_vf32_reduce_add(acc); \
        for (; k < K; ++k) \
          s += TtoF32(xrow[k]) * mag_float8_e4m3fn_to_float32(w[k]); \
        rrow[j] = F32toT(s*scale); \
      } \
    } \
  }
mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_impl

typedef void (mag_gemm_fp8w_scaled_kernel_strided_t)(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1, float scale);
#define mag_gemm_fp8w_scaled_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_fp8w_scaled_kernel_strided_##T(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1, float scale) { \
    T *restrict r = (T *)pr; \
    const T *x = (const T *)px; \
    const mag_float8_e4m3fn_t *y = (const mag_float8_e4m3fn_t *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*sx0 + k*sx1])*mag_float8_e4m3fn_to_float32(y[k*sy0 + j*sy1]); \
          r[i*N + j] = F32toT(acc*scale); \
        } \
      } \
  }
mag_gemm_fp8w_scaled_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_fp8w_scaled_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_fp8w_scaled_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_fp8w_scaled_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_fp8w_scaled_kernel_strided_impl

static MAG_HOTPROC void mag_matmul_gemm_fp8w_scaled_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t tc,
  int64_t M,
  int64_t N,
  int64_t K,
  void *restrict pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy0, int64_t sy1,
  float scale
) {
  static mag_gemm_fp8w_scaled_kernel_contig_t *const mag_gemm_fp8w_scaled_kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_fp8w_scaled_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_fp8w_scaled_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_fp8w_scaled_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_fp8w_scaled_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_t *const mag_gemm_fp8w_scaled_kernel_lut_rhs_transposed_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_fp8w_scaled_kernel_rhs_transposed_contig_mag_float8_e4m3fn_t
  };
  static mag_gemm_fp8w_scaled_kernel_strided_t *const mag_gemm_fp8w_scaled_kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_fp8w_scaled_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_fp8w_scaled_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_fp8w_scaled_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_fp8w_scaled_kernel_strided_mag_float8_e4m3fn_t
  };
  int64_t chunk = (M+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(M, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Mt = end-start;
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  pr = (uint8_t *)pr + start*N*el;
  px = (const uint8_t *)px + start*sx0*el;
  if (sx0 == K && sx1 == 1 && sy0 == N && sy1 == 1)
    (*mag_gemm_fp8w_scaled_kernel_lut_contig[dtype])(Mt, N, K, pr, px, py, scale);
  else if (sx0 == K && sx1 == 1 && sy0 == 1 && sy1 == K)
    (*mag_gemm_fp8w_scaled_kernel_lut_rhs_transposed_contig[dtype])(Mt, N, K, pr, px, py, scale);
  else
    (*mag_gemm_fp8w_scaled_kernel_lut_strided[dtype])(Mt, N, K, pr, px, py, sx0, sx1, sy0, sy1, scale);
}

static MAG_HOTPROC void mag_matmul_gemm_fp8w_scaled(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  const mag_tensor_t *s = payload->cmd->in[2];
  int64_t M = x->coords.shape[0];
  int64_t K = x->coords.shape[1];
  int64_t N = y->coords.shape[1];
  int64_t sx0 = x->coords.strides[0];
  int64_t sx1 = x->coords.strides[1];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  float scale = *(const float *)mag_tensor_data_ptr(s);
  mag_matmul_gemm_fp8w_scaled_impl(r->dtype, ti, tc, M, N, K, pr, px, sx0, sx1, py, sy0, sy1, scale);
}

