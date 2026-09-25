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

#define mag_gemm_align_up(x) mag_align_up(x, MAG_MM_SCRATCH_ALIGN)

#if defined(__AVX512F__)
  #define MAG_GEMM_MR 8
  #define MAG_GEMM_NRV 2
#elif defined(__AVX2__)
  #define MAG_GEMM_MR 6
  #define MAG_GEMM_NRV 2
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
  #define MAG_GEMM_MR 8
  #define MAG_GEMM_NRV 3
#elif defined(__loongarch_asx)
  #define MAG_GEMM_MR 6
  #define MAG_GEMM_NRV 2
#elif defined(__SSE2__) || defined(__loongarch_sx)
  #define MAG_GEMM_MR 6
  #define MAG_GEMM_NRV 2
#else
  #define MAG_GEMM_MR 4
  #define MAG_GEMM_NRV 1
#endif
#define MAG_GEMM_NR (MAG_GEMM_NRV*MAG_VF32_LANES)

#ifndef MAG_GEMM_KC
  #define MAG_GEMM_KC 256
#endif
#ifndef MAG_GEMM_MC
  #define MAG_GEMM_MC 192
#endif
#ifndef MAG_GEMM_NC
  #define MAG_GEMM_NC 256
#endif
#ifndef MAG_GEMM_TILES_PER_THREAD
  #define MAG_GEMM_TILES_PER_THREAD 4
#endif
#ifndef MAG_GEMM_MIN_TILE_FLOPS
  #define MAG_GEMM_MIN_TILE_FLOPS (1<<20)
#endif
#ifndef MAG_GEMM_THIN_MIN_TILE_BYTES
  #define MAG_GEMM_THIN_MIN_TILE_BYTES (256<<10) /* Thin (bandwidth bound) tiles stop shrinking below this many bytes of B. */
#endif
#define MAG_GEMM_MIN_MT (MAG_GEMM_MR*4)
#define MAG_GEMM_MIN_NT (MAG_GEMM_NR*2)
#define mag_gemm_round_up(x, m) (((x)+(m)-1)/(m)*(m))

#if MAG_GEMM_MR == 4
  #define mag_gemm_unroll_mr(X) X(0) X(1) X(2) X(3)
#elif MAG_GEMM_MR == 6
  #define mag_gemm_unroll_mr(X) X(0) X(1) X(2) X(3) X(4) X(5)
#elif MAG_GEMM_MR == 8
  #define mag_gemm_unroll_mr(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#else
  #error "unsupported MAG_GEMM_MR"
#endif

#if MAG_GEMM_NRV == 1
  #define mag_gemm_unroll_nrv(X, i) X(i, 0)
#elif MAG_GEMM_NRV == 2
  #define mag_gemm_unroll_nrv(X, i) X(i, 0) X(i, 1)
#elif MAG_GEMM_NRV == 3
  #define mag_gemm_unroll_nrv(X, i) X(i, 0) X(i, 1) X(i, 2)
#else
  #error "unsupported MAG_GEMM_NRV"
#endif

static MAG_HOTPROC void mag_gemm_ukernel(
  int64_t kc,
  const float *restrict ap,
  const float *restrict bp,
  float *restrict cc,
  int64_t ldc
) {
  mag_vf32_t acc[MAG_GEMM_MR][MAG_GEMM_NRV];
  #define mag_gemm_uk_zero(i, j) acc[i][j] = mag_vf32_zero();
  #define mag_gemm_uk_zero_row(i) mag_gemm_unroll_nrv(mag_gemm_uk_zero, i)
  mag_gemm_unroll_mr(mag_gemm_uk_zero_row)
  for (int64_t k=0; k < kc; ++k) {
    mag_vf32_t b[MAG_GEMM_NRV];
    #define mag_gemm_uk_loadb(i, j) b[j] = mag_vf32_loadu(bp + (j)*MAG_VF32_LANES);
    mag_gemm_unroll_nrv(mag_gemm_uk_loadb, 0)
    #define mag_gemm_uk_fma(i, j) acc[i][j] = mag_vf32_fmadd(a, b[j], acc[i][j]);
    #define mag_gemm_uk_fma_row(i) { \
      mag_vf32_t a = mag_vf32_broadcast(ap + (i)); \
      mag_gemm_unroll_nrv(mag_gemm_uk_fma, i) \
    }
    mag_gemm_unroll_mr(mag_gemm_uk_fma_row)
    ap += MAG_GEMM_MR;
    bp += MAG_GEMM_NR;
  }
  #define mag_gemm_uk_store(i, j) \
    mag_vf32_storeu(cc + (i)*ldc + (j)*MAG_VF32_LANES, \
      mag_vf32_add(mag_vf32_loadu(cc + (i)*ldc + (j)*MAG_VF32_LANES), acc[i][j]));
  #define mag_gemm_uk_store_row(i) mag_gemm_unroll_nrv(mag_gemm_uk_store, i)
  mag_gemm_unroll_mr(mag_gemm_uk_store_row)
  #undef mag_gemm_uk_zero
  #undef mag_gemm_uk_zero_row
  #undef mag_gemm_uk_loadb
  #undef mag_gemm_uk_fma
  #undef mag_gemm_uk_fma_row
  #undef mag_gemm_uk_store
  #undef mag_gemm_uk_store_row
}

typedef void (mag_gemm_pack_a_t)(float *restrict ap, const void *px, int64_t mc, int64_t kc, int64_t sx0, int64_t sx1);
#define mag_gemm_pack_a_impl(T, TtoF32) \
  static MAG_HOTPROC void mag_gemm_pack_a_##T(float *restrict ap, const void *px, int64_t mc, int64_t kc, int64_t sx0, int64_t sx1) { \
    const T *x = (const T *)px; \
    for (int64_t q=0; q < mc; q += MAG_GEMM_MR) { \
      int64_t mm = mag_vmin((int64_t)MAG_GEMM_MR, mc-q); \
      const T *src = x + q*sx0; \
      for (int64_t k=0; k < kc; ++k) { \
        int64_t i=0; \
        for (; i < mm; ++i) ap[i] = TtoF32(src[i*sx0 + k*sx1]); \
        for (; i < MAG_GEMM_MR; ++i) ap[i] = 0.f; \
        ap += MAG_GEMM_MR; \
      } \
    } \
  }
mag_gemm_pack_a_impl(float, mag_cvt_nop)
mag_gemm_pack_a_impl(mag_float16_t, mag_float16_to_float32)
mag_gemm_pack_a_impl(mag_bfloat16_t, mag_bfloat16_to_float32)
mag_gemm_pack_a_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32)
#undef mag_gemm_pack_a_impl

typedef void (mag_gemm_pack_b_t)(float *restrict bp, const void *py, int64_t kc, int64_t nc, int64_t sy0, int64_t sy1);
#define mag_gemm_pack_b_impl(T, TtoF32, LoadTtoF32) \
  static MAG_HOTPROC void mag_gemm_pack_b_##T(float *restrict bp, const void *py, int64_t kc, int64_t nc, int64_t sy0, int64_t sy1) { \
    const T *y = (const T *)py; \
    for (int64_t p=0; p < nc; p += MAG_GEMM_NR) { \
      int64_t nn = mag_vmin((int64_t)MAG_GEMM_NR, nc-p); \
      const T *src = y + p*sy1; \
      for (int64_t k=0; k < kc; ++k) { \
        const T *row = src + k*sy0; \
        int64_t j = 0; \
        if (sy1 == 1) \
          for (; j+MAG_VF32_LANES-1 < nn; j += MAG_VF32_LANES) \
            mag_vf32_storeu(bp + j, LoadTtoF32(row + j)); \
        for (; j < nn; ++j) bp[j] = TtoF32(row[j*sy1]); \
        for (; j < MAG_GEMM_NR; ++j) bp[j] = 0.f; \
        bp += MAG_GEMM_NR; \
      } \
    } \
  }
mag_gemm_pack_b_impl(float, mag_cvt_nop, mag_vf32_loadu)
mag_gemm_pack_b_impl(mag_float16_t, mag_float16_to_float32, mag_vf32_loadu_f16)
mag_gemm_pack_b_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_vf32_loadu_bf16)
mag_gemm_pack_b_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_vf32_loadu_float8_e4m3fn)
#undef mag_gemm_pack_b_impl

typedef void (mag_gemm_store_c_t)(void *pr, int64_t ldr, const float *cc, int64_t ldc, int64_t m, int64_t n);
#define mag_gemm_store_c_impl(T, F32toT, StoreF32toT) \
  static MAG_HOTPROC void mag_gemm_store_c_##T(void *pr, int64_t ldr, const float *cc, int64_t ldc, int64_t m, int64_t n) { \
    T *r = (T *)pr; \
    for (int64_t i=0; i < m; ++i) { \
      T *restrict dst = r + i*ldr; \
      const float *src = cc + i*ldc; \
      int64_t j = 0; \
      for (; j+MAG_VF32_LANES-1 < n; j += MAG_VF32_LANES) \
        StoreF32toT(dst + j, mag_vf32_loadu(src + j)); \
      for (; j < n; ++j) dst[j] = F32toT(src[j]); \
    } \
  }
mag_gemm_store_c_impl(float, mag_cvt_nop, mag_vf32_storeu)
mag_gemm_store_c_impl(mag_float16_t, mag_float32_to_float16, mag_vf32_storeu_f16)
mag_gemm_store_c_impl(mag_bfloat16_t, mag_float32_to_bfloat16, mag_vf32_storeu_bf16)
mag_gemm_store_c_impl(mag_float8_e4m3fn_t, mag_float32_to_float8_e4m3fn, mag_vf32_storeu_float8_e4m3fn)
#undef mag_gemm_store_c_impl

#ifndef MAG_GEMM_THIN_MAX_M
  #define MAG_GEMM_THIN_MAX_M 32
#endif
#define MAG_GEMM_THIN_MR 4
#define MAG_GEMM_THIN_NRD 4
#define MAG_GEMM_THIN_NRV 2

typedef void (mag_gemm_thin_nt_tail_t)(int64_t md, int64_t nn, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb);
#define mag_gemm_thin_nt_tail_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_thin_nt_tail_##T(int64_t md, int64_t nn, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb) { \
    T *restrict r = (T *)pr; \
    const T *a = (const T *)pa; \
    const T *b = (const T *)pb; \
    for (int64_t i=0; i < md; ++i) { \
      for (int64_t j=0; j < nn; ++j) { \
        float s = 0.f; \
        for (int64_t k=0; k < K; ++k) s += TtoF32(a[i*lda + k])*TtoF32(b[j*ldb + k]); \
        r[i*ldr + j] = F32toT(s); \
      } \
    } \
  }
mag_gemm_thin_nt_tail_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_thin_nt_tail_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_thin_nt_tail_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_thin_nt_tail_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_thin_nt_tail_impl

#define mag_gemm_dot_fma(acc, a, b) mag_vf32_fmadd(a, b, acc)

typedef void (mag_gemm_thin_nt_t)(int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb);
#define mag_gemm_thin_nt_impl(NAME, T, MD, KU, VT, LANES, LOAD, DOT, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_thin_nt_##NAME##_md##MD(int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb) { \
    T *restrict r = (T *)pr; \
    const T *a = (const T *)pa; \
    const T *b = (const T *)pb; \
    mag_vf32_t acc[KU][MD][MAG_GEMM_THIN_NRD]; \
    for (int u=0; u < KU; ++u) for (int i=0; i < MD; ++i) for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) acc[u][i][j] = mag_vf32_zero(); \
    int64_t k = 0; \
    for (; k+KU*LANES-1 < K; k += KU*LANES) { \
      VT bv[KU][MAG_GEMM_THIN_NRD]; \
      for (int u=0; u < KU; ++u) for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) bv[u][j] = LOAD(b + j*ldb + k + u*LANES); \
      for (int i=0; i < MD; ++i) { \
        for (int u=0; u < KU; ++u) { \
          VT av = LOAD(a + i*lda + k + u*LANES); \
          for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) acc[u][i][j] = DOT(acc[u][i][j], av, bv[u][j]); \
        } \
      } \
    } \
    for (; k+LANES-1 < K; k += LANES) { \
      VT bv[MAG_GEMM_THIN_NRD]; \
      for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) bv[j] = LOAD(b + j*ldb + k); \
      for (int i=0; i < MD; ++i) { \
        VT av = LOAD(a + i*lda + k); \
        for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) acc[0][i][j] = DOT(acc[0][i][j], av, bv[j]); \
      } \
    } \
    for (int i=0; i < MD; ++i) { \
      for (int j=0; j < MAG_GEMM_THIN_NRD; ++j) { \
        mag_vf32_t v = acc[0][i][j]; \
        for (int u=1; u < KU; ++u) v = mag_vf32_add(v, acc[u][i][j]); \
        float s = mag_vf32_reduce_add(v); \
        for (int64_t kk=k; kk < K; ++kk) s += TtoF32(a[i*lda + kk])*TtoF32(b[j*ldb + kk]); \
        r[i*ldr + j] = F32toT(s); \
      } \
    } \
  }
#define mag_gemm_thin_nt_impl_all(NAME, T, VT, LANES, LOAD, DOT, TtoF32, F32toT) \
  mag_gemm_thin_nt_impl(NAME, T, 1, 2, VT, LANES, LOAD, DOT, TtoF32, F32toT) \
  mag_gemm_thin_nt_impl(NAME, T, 2, 2, VT, LANES, LOAD, DOT, TtoF32, F32toT) \
  mag_gemm_thin_nt_impl(NAME, T, 3, 1, VT, LANES, LOAD, DOT, TtoF32, F32toT) \
  mag_gemm_thin_nt_impl(NAME, T, 4, 1, VT, LANES, LOAD, DOT, TtoF32, F32toT)
mag_gemm_thin_nt_impl_all(float, float, mag_vf32_t, MAG_VF32_LANES, mag_vf32_loadu, mag_gemm_dot_fma, mag_cvt_nop, mag_cvt_nop)
mag_gemm_thin_nt_impl_all(mag_float16_t, mag_float16_t, mag_vf32_t, MAG_VF32_LANES, mag_vf32_loadu_f16, mag_gemm_dot_fma, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_thin_nt_impl_all(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, mag_vf32_t, MAG_VF32_LANES, mag_vf32_loadu_float8_e4m3fn, mag_gemm_dot_fma, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
mag_gemm_thin_nt_impl_all(mag_bfloat16_t, mag_bfloat16_t, mag_vf32_t, MAG_VF32_LANES, mag_vf32_loadu_bf16, mag_gemm_dot_fma, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
#if MAG_HAS_NATIVE_DPBF16
  mag_gemm_thin_nt_impl_all(bf16dp, mag_bfloat16_t, mag_vbf16_t, MAG_VBF16_LANES, mag_vbf16_loadu, mag_vf32_dpbf16, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
#endif
#undef mag_gemm_thin_nt_impl_all
#undef mag_gemm_thin_nt_impl

#define mag_gemm_thin_nt_lut_row(T) { NULL, &mag_gemm_thin_nt_##T##_md1, &mag_gemm_thin_nt_##T##_md2, &mag_gemm_thin_nt_##T##_md3, &mag_gemm_thin_nt_##T##_md4 }

typedef void (mag_gemm_thin_nn_t)(int64_t md, int64_t nv, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb);
#define mag_gemm_thin_nn_impl(T, TtoF32, F32toT, LoadTtoF32, StoreF32toT) \
  static MAG_HOTPROC void mag_gemm_thin_nn_##T(int64_t md, int64_t nv, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb) { \
    T *restrict r = (T *)pr; \
    const T *a = (const T *)pa; \
    const T *b = (const T *)pb; \
    mag_vf32_t vacc[MAG_GEMM_THIN_MR][MAG_GEMM_THIN_NRV]; \
    for (int64_t i=0; i < md; ++i) for (int64_t j=0; j < nv; ++j) vacc[i][j] = mag_vf32_zero(); \
    for (int64_t k=0; k < K; ++k) { \
      mag_vf32_t bv[MAG_GEMM_THIN_NRV]; \
      for (int64_t j=0; j < nv; ++j) bv[j] = LoadTtoF32(b + k*ldb + j*MAG_VF32_LANES); \
      for (int64_t i=0; i < md; ++i) { \
        mag_vf32_t av = mag_vf32_splat(TtoF32(a[i*lda + k])); \
        for (int64_t j=0; j < nv; ++j) vacc[i][j] = mag_vf32_fmadd(av, bv[j], vacc[i][j]); \
      } \
    } \
    for (int64_t i=0; i < md; ++i) \
      for (int64_t j=0; j < nv; ++j) StoreF32toT(r + i*ldr + j*MAG_VF32_LANES, vacc[i][j]); \
  }
mag_gemm_thin_nn_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu, mag_vf32_storeu)
mag_gemm_thin_nn_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16, mag_vf32_storeu_f16)
mag_gemm_thin_nn_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16, mag_vf32_storeu_bf16)
mag_gemm_thin_nn_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_storeu_float8_e4m3fn)
#undef mag_gemm_thin_nn_impl

typedef void (mag_gemm_thin_nn_tail_t)(int64_t md, int64_t n, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb);
#define mag_gemm_thin_nn_tail_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_thin_nn_tail_##T(int64_t md, int64_t n, int64_t K, void *pr, int64_t ldr, const void *pa, int64_t lda, const void *pb, int64_t ldb) { \
    T *restrict r = (T *)pr; \
    const T *a = (const T *)pa; \
    const T *b = (const T *)pb; \
    for (int64_t i=0; i < md; ++i) { \
      for (int64_t j=0; j < n; ++j) { \
        float acc = 0.f; \
        for (int64_t k=0; k < K; ++k) acc += TtoF32(a[i*lda + k])*TtoF32(b[k*ldb + j]); \
        r[i*ldr + j] = F32toT(acc); \
      } \
    } \
  }
mag_gemm_thin_nn_tail_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_thin_nn_tail_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_thin_nn_tail_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_thin_nn_tail_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_thin_nn_tail_impl

static MAG_HOTPROC void mag_gemm_thin(
  mag_dtype_t dtype,
  bool nt,
  bool bf16_dp,
  int64_t m0, int64_t m1, int64_t n0, int64_t n1,
  int64_t N, int64_t K,
  void *pr,
  const void *px, int64_t sx0,
  const void *py, int64_t sy0, int64_t sy1
) {
  static mag_gemm_thin_nt_t *const lut_nt[4][MAG_GEMM_THIN_MR+1] = {
    [MAG_DTYPE_FLOAT32] = mag_gemm_thin_nt_lut_row(float),
    [MAG_DTYPE_FLOAT16] = mag_gemm_thin_nt_lut_row(mag_float16_t),
    [MAG_DTYPE_BFLOAT16] = mag_gemm_thin_nt_lut_row(mag_bfloat16_t),
    [MAG_DTYPE_FLOAT8_E4M3FN] = mag_gemm_thin_nt_lut_row(mag_float8_e4m3fn_t)
  };
  #if MAG_HAS_NATIVE_DPBF16
    static mag_gemm_thin_nt_t *const lut_nt_bf16dp[MAG_GEMM_THIN_MR+1] = mag_gemm_thin_nt_lut_row(bf16dp);
  #endif
  static mag_gemm_thin_nt_tail_t *const lut_nt_tail[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_thin_nt_tail_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_thin_nt_tail_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_thin_nt_tail_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_thin_nt_tail_mag_float8_e4m3fn_t
  };
  static mag_gemm_thin_nn_t *const lut_nn[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_thin_nn_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_thin_nn_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_thin_nn_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_thin_nn_mag_float8_e4m3fn_t
  };
  static mag_gemm_thin_nn_tail_t *const lut_nn_tail[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_thin_nn_tail_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_thin_nn_tail_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_thin_nn_tail_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_thin_nn_tail_mag_float8_e4m3fn_t
  };
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  const uint8_t *xb = px;
  const uint8_t *yb = py;
  uint8_t *rb = pr;
  #define mag_gemm_thin_rows(call) \
    for (int64_t i=m0; i < m1; i += MAG_GEMM_THIN_MR) { \
      int64_t md = mag_vmin((int64_t)MAG_GEMM_THIN_MR, m1-i); \
      const void *arow = xb + i*sx0*el; \
      uint8_t *rrow = rb + i*N*el; \
      call; \
    }
  if (nt) {
    mag_gemm_thin_nt_t *const *kern = lut_nt[dtype];
    #if MAG_HAS_NATIVE_DPBF16
      if (bf16_dp && dtype == MAG_DTYPE_BFLOAT16) kern = lut_nt_bf16dp;
    #else
      (void)bf16_dp;
    #endif
    int64_t j = n0;
    for (; j+MAG_GEMM_THIN_NRD-1 < n1; j += MAG_GEMM_THIN_NRD)
      mag_gemm_thin_rows((*kern[md])(K, rrow + j*el, N, arow, sx0, yb + j*sy1*el, sy1))
    if (j < n1)
      mag_gemm_thin_rows((*lut_nt_tail[dtype])(md, n1-j, K, rrow + j*el, N, arow, sx0, yb + j*sy1*el, sy1))
  } else {
    int64_t j = n0;
    for (; j+MAG_GEMM_THIN_NRV*MAG_VF32_LANES-1 < n1; j += MAG_GEMM_THIN_NRV*MAG_VF32_LANES)
      mag_gemm_thin_rows((*lut_nn[dtype])(md, (int64_t)MAG_GEMM_THIN_NRV, K, rrow + j*el, N, arow, sx0, yb + j*el, sy0))
    for (; j+MAG_VF32_LANES-1 < n1; j += MAG_VF32_LANES)
      mag_gemm_thin_rows((*lut_nn[dtype])(md, 1, K, rrow + j*el, N, arow, sx0, yb + j*el, sy0))
    if (j < n1)
      mag_gemm_thin_rows((*lut_nn_tail[dtype])(md, n1-j, K, rrow + j*el, N, arow, sx0, yb + j*el, sy0))
  }
  #undef mag_gemm_thin_rows
}

#define MAG_GEMM_BF16_NPAIRS(kc) (((kc)+1)>>1)

static MAG_HOTPROC void mag_gemm_pack_a_bf16_dp(mag_bfloat16_t *restrict ap, const mag_bfloat16_t *x, int64_t mc, int64_t kc, int64_t sx0, int64_t sx1) {
  mag_bfloat16_t zero = mag_float32_to_bfloat16(0.f);
  int64_t npairs = MAG_GEMM_BF16_NPAIRS(kc);
  for (int64_t q=0; q < mc; q += MAG_GEMM_MR) {
    int64_t mm = mag_vmin((int64_t)MAG_GEMM_MR, mc-q);
    const mag_bfloat16_t *src = x + q*sx0;
    for (int64_t kp=0; kp < npairs; ++kp) {
      int64_t k0 = 2*kp, k1 = k0+1;
      for (int64_t i=0; i < MAG_GEMM_MR; ++i) {
        ap[i*2+0] = i < mm ? src[i*sx0 + k0*sx1] : zero;
        ap[i*2+1] = i < mm && k1 < kc ? src[i*sx0 + k1*sx1] : zero;
      }
      ap += MAG_GEMM_MR*2;
    }
  }
}

static MAG_HOTPROC void mag_gemm_pack_b_bf16_dp(mag_bfloat16_t *restrict bp, const mag_bfloat16_t *y, int64_t kc, int64_t nc, int64_t sy0, int64_t sy1) {
  mag_bfloat16_t zero = mag_float32_to_bfloat16(0.f);
  int64_t npairs = MAG_GEMM_BF16_NPAIRS(kc);
  for (int64_t p=0; p < nc; p += MAG_GEMM_NR) {
    int64_t nn = mag_vmin((int64_t)MAG_GEMM_NR, nc-p);
    const mag_bfloat16_t *src = y + p*sy1;
    for (int64_t kp=0; kp < npairs; ++kp) {
      int64_t k0 = kp<<1, k1 = k0+1;
      const mag_bfloat16_t *r0 = src + k0*sy0;
      const mag_bfloat16_t *r1 = src + k1*sy0;
      for (int64_t c=0; c < MAG_GEMM_NR; ++c) {
        bp[c*2+0] = c < nn ? r0[c*sy1] : zero;
        bp[c*2+1] = c < nn && k1 < kc ? r1[c*sy1] : zero;
      }
      bp += MAG_GEMM_NR*2;
    }
  }
}

static MAG_HOTPROC void mag_gemm_ukernel_bf16_dp(int64_t npairs, const mag_bfloat16_t *restrict ap, const mag_bfloat16_t *restrict bp, float *restrict cc, int64_t ldc) {
  mag_vf32_t acc[MAG_GEMM_MR][MAG_GEMM_NRV];
  #define mag_gemm_bdp_zero(i, j) acc[i][j] = mag_vf32_zero();
  #define mag_gemm_bdp_zero_row(i) mag_gemm_unroll_nrv(mag_gemm_bdp_zero, i)
  mag_gemm_unroll_mr(mag_gemm_bdp_zero_row)
  for (int64_t kp=0; kp < npairs; ++kp) {
    mag_vbf16_t b[MAG_GEMM_NRV];
    #define mag_gemm_bdp_loadb(i, j) b[j] = mag_vbf16_loadu(bp + (j)*MAG_VBF16_LANES);
    mag_gemm_unroll_nrv(mag_gemm_bdp_loadb, 0)
    #define mag_gemm_bdp_dp(i, j) acc[i][j] = mag_vf32_dpbf16(acc[i][j], a, b[j]);
    #define mag_gemm_bdp_dp_row(i) { \
      mag_vbf16_t a = mag_vbf16_broadcast_pair(ap + (i)*2); \
      mag_gemm_unroll_nrv(mag_gemm_bdp_dp, i) \
    }
    mag_gemm_unroll_mr(mag_gemm_bdp_dp_row)
    ap += MAG_GEMM_MR<<1;
    bp += MAG_GEMM_NR<<1;
  }
  #define mag_gemm_bdp_store(i, j) \
    mag_vf32_storeu(cc + (i)*ldc + (j)*MAG_VF32_LANES, \
      mag_vf32_add(mag_vf32_loadu(cc + (i)*ldc + (j)*MAG_VF32_LANES), acc[i][j]));
  #define mag_gemm_bdp_store_row(i) mag_gemm_unroll_nrv(mag_gemm_bdp_store, i)
  mag_gemm_unroll_mr(mag_gemm_bdp_store_row)
  #undef mag_gemm_bdp_zero
  #undef mag_gemm_bdp_zero_row
  #undef mag_gemm_bdp_loadb
  #undef mag_gemm_bdp_dp
  #undef mag_gemm_bdp_dp_row
  #undef mag_gemm_bdp_store
  #undef mag_gemm_bdp_store_row
}

typedef void (mag_gemm_resolve_batch_t)(const void *user, int64_t batch, void **pr, const void **px, const void **py);

static void mag_gemm_pick_tiles(int64_t tc, int64_t batch, int64_t M, int64_t N, int64_t K, int64_t el, bool split_m, int64_t *pmt, int64_t *pnt) {
  int64_t mt = split_m ? mag_vmin((int64_t)MAG_GEMM_MC, mag_gemm_round_up(M, MAG_GEMM_MR)) : M;
  int64_t nt = mag_vmin((int64_t)MAG_GEMM_NC, mag_gemm_round_up(N, MAG_GEMM_NR));
  if (tc > 1) {
    int64_t want = MAG_GEMM_TILES_PER_THREAD*tc;
    for (;;) {
      int64_t tiles = batch*((M + mt-1)/mt)*((N + nt-1)/nt);
      if (tiles >= want) break;
      bool can_m = split_m && mt > MAG_GEMM_MIN_MT;
      bool can_n = nt > MAG_GEMM_MIN_NT;
      if (!can_m && !can_n) break;
      int64_t nmt = mt, nnt = nt;
      if (can_m && (!can_n || mt >= nt)) nmt = mag_vmax((int64_t)MAG_GEMM_MIN_MT, mag_gemm_round_up((mt+1)/2, MAG_GEMM_MR));
      else nnt = mag_vmax((int64_t)MAG_GEMM_MIN_NT, mag_gemm_round_up((nt+1)/2, MAG_GEMM_NR));
      if (split_m ? 2*mag_vmin(nmt, M)*mag_vmin(nnt, N)*K < MAG_GEMM_MIN_TILE_FLOPS : mag_vmin(nnt, N)*K*el < MAG_GEMM_THIN_MIN_TILE_BYTES) break;
      mt = nmt;
      nt = nnt;
    }
  }
  *pmt = mt;
  *pnt = nt;
}

static MAG_HOTPROC void mag_gemm_tile_bf16_dp(
  int64_t m0, int64_t m1, int64_t n0, int64_t n1,
  int64_t N, int64_t K, bool pack_b,
  void *pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy0, int64_t sy1,
  mag_bfloat16_t *ap, mag_bfloat16_t *bp, float *cc
) {
  const mag_bfloat16_t *xb = px;
  const mag_bfloat16_t *yb = py;
  mag_bfloat16_t *rb = pr;
  int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
  int64_t mbt = m1-m0;
  int64_t nct = n1-n0;
  int64_t ldc = mag_gemm_round_up(nct, MAG_GEMM_NR);
  memset(cc, 0, (size_t)(mag_gemm_round_up(mbt, MAG_GEMM_MR)*ldc)*sizeof(*cc));
  for (int64_t pc=0; pc < K; pc += KC) {
    int64_t kct = mag_vmin(KC, K-pc);
    int64_t np = MAG_GEMM_BF16_NPAIRS(kct);
    if (pack_b || pc) mag_gemm_pack_b_bf16_dp(bp, yb + pc*sy0 + n0*sy1, kct, nct, sy0, sy1);
    mag_gemm_pack_a_bf16_dp(ap, xb + m0*sx0 + pc*sx1, mbt, kct, sx0, sx1);
    for (int64_t jr=0; jr < nct; jr += MAG_GEMM_NR) {
      const mag_bfloat16_t *bpp = bp + jr/MAG_GEMM_NR*np*MAG_GEMM_NR*2;
      for (int64_t ir=0; ir < mbt; ir += MAG_GEMM_MR)
        mag_gemm_ukernel_bf16_dp(np, ap + ir/MAG_GEMM_MR*np*MAG_GEMM_MR*2, bpp, cc + ir*ldc + jr, ldc);
    }
  }
  mag_gemm_store_c_mag_bfloat16_t(rb + m0*N + n0, N, cc, ldc, mbt, nct);
}

static MAG_HOTPROC void mag_gemm_tile_f32(
  mag_dtype_t dtype,
  int64_t m0, int64_t m1, int64_t n0, int64_t n1,
  int64_t N, int64_t K, bool pack_b,
  void *pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy0, int64_t sy1,
  float *ap, float *bp, float *cc
) {
  static mag_gemm_pack_a_t *const lut_pack_a[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_pack_a_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_pack_a_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_pack_a_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_pack_a_mag_float8_e4m3fn_t
  };
  static mag_gemm_pack_b_t *const lut_pack_b[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_pack_b_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_pack_b_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_pack_b_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_pack_b_mag_float8_e4m3fn_t
  };
  static mag_gemm_store_c_t *const lut_store_c[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_gemm_store_c_float,
    [MAG_DTYPE_FLOAT16] = &mag_gemm_store_c_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_gemm_store_c_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_store_c_mag_float8_e4m3fn_t
  };
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  const uint8_t *xb = px;
  const uint8_t *yb = py;
  uint8_t *rb = pr;
  int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
  int64_t mbt = m1-m0;
  int64_t nct = n1-n0;
  int64_t ldc = mag_gemm_round_up(nct, MAG_GEMM_NR);
  memset(cc, 0, (size_t)(mag_gemm_round_up(mbt, MAG_GEMM_MR)*ldc)*sizeof(*cc));
  for (int64_t pc=0; pc < K; pc += KC) {
    int64_t kct = mag_vmin(KC, K-pc);
    if (pack_b || pc) (*lut_pack_b[dtype])(bp, yb + (pc*sy0 + n0*sy1)*el, kct, nct, sy0, sy1);
    (*lut_pack_a[dtype])(ap, xb + (m0*sx0 + pc*sx1)*el, mbt, kct, sx0, sx1);
    for (int64_t jr=0; jr < nct; jr += MAG_GEMM_NR) {
      const float *bpp = bp + jr/MAG_GEMM_NR*kct*MAG_GEMM_NR;
      for (int64_t ir=0; ir < mbt; ir += MAG_GEMM_MR)
        mag_gemm_ukernel(kct, ap + ir/MAG_GEMM_MR*kct*MAG_GEMM_MR, bpp, cc + ir*ldc + jr, ldc);
    }
  }
  (*lut_store_c[dtype])(rb + (m0*N + n0)*el, N, cc, ldc, mbt, nct);
}

static MAG_HOTPROC void mag_matmul_gemm_impl(
  const mag_kernel_payload_t *payload,
  mag_dtype_t dtype,
  int64_t M,
  int64_t N,
  int64_t K,
  int64_t batch,
  mag_gemm_resolve_batch_t *resolve,
  const void *user,
  void *restrict pr,
  const void *px, int64_t sx0, int64_t sx1,
  const void *py, int64_t sy0, int64_t sy1
) {
  int64_t tc = payload->thread_num;
  bool thin = M <= MAG_GEMM_THIN_MAX_M && sx1 == 1 && (sy1 == K ? sy0 == 1 : (sy0 == N && sy1 == 1));
  bool bf16_dp = false;
  bool thin_bf16_dp = false;
  #if MAG_HAS_NATIVE_DPBF16
    bf16_dp = dtype == MAG_DTYPE_BFLOAT16;
    #if defined(__aarch64__) || defined(_M_ARM64)
      bf16_dp = bf16_dp && !payload->cmd->out[0]->ctx->machine.arm64_bf16_dot_slow;
    #endif
    thin_bf16_dp = thin && bf16_dp;
    bf16_dp = !thin && bf16_dp;
  #endif
  int64_t mt, nt;
  mag_gemm_pick_tiles(tc, batch, M, N, K, (int64_t)mag_type_trait(dtype)->size, !thin, &mt, &nt);
  int64_t tiles_m = (M + mt-1)/mt;
  int64_t tiles_n = (N + nt-1)/nt;
  int64_t tiles_per_batch = tiles_m*tiles_n;
  int64_t total = batch*tiles_per_batch;
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = NULL;
  size_t ap_nb = 0, bp_nb = 0;
  if (!thin) {
    int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
    int64_t apad = mag_gemm_round_up(mt, MAG_GEMM_MR);
    int64_t ldc_max = mag_gemm_round_up(nt, MAG_GEMM_NR);
    int64_t kdepth = bf16_dp ? MAG_GEMM_BF16_NPAIRS(KC)*2 : KC;
    size_t el = bf16_dp ? sizeof(mag_bfloat16_t) : sizeof(float);
    ap_nb = mag_gemm_align_up((size_t)(apad*kdepth)*el);
    bp_nb = mag_gemm_align_up((size_t)(ldc_max*kdepth)*el);
    size_t cc_nb = mag_gemm_align_up((size_t)(apad*ldc_max)*sizeof(float));
    blk = mag_scratch_arena_alloc(&mag_tls_arena, ap_nb + bp_nb + cc_nb);
  }
  bool b_reusable = !thin && K <= MAG_GEMM_KC;
  int64_t packed_batch = -1;
  int64_t packed_jc = -1;
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t b = t/tiles_per_batch;
    int64_t rem = t%tiles_per_batch;
    int64_t jc = rem/tiles_m;
    int64_t ic = rem%tiles_m;
    void *tpr = pr;
    const void *tpx = px;
    const void *tpy = py;
    if (resolve) (*resolve)(user, b, &tpr, &tpx, &tpy);
    int64_t m0 = ic*mt;
    int64_t m1 = mag_vmin(M, m0+mt);
    int64_t n0 = jc*nt;
    int64_t n1 = mag_vmin(N, n0+nt);
    if (thin) {
      mag_gemm_thin(dtype, sy1 == K, thin_bf16_dp, m0, m1, n0, n1, N, K, tpr, tpx, sx0, tpy, sy0, sy1);
      continue;
    }
    bool pack_b = !(b_reusable && packed_batch == b && packed_jc == jc);
    if (bf16_dp)
      mag_gemm_tile_bf16_dp(m0, m1, n0, n1, N, K, pack_b, tpr, tpx, sx0, sx1, tpy, sy0, sy1, (mag_bfloat16_t *)blk, (mag_bfloat16_t *)(blk + ap_nb), (float *)(blk + ap_nb + bp_nb));
    else
      mag_gemm_tile_f32(dtype, m0, m1, n0, n1, N, K, pack_b, tpr, tpx, sx0, sx1, tpy, sy0, sy1, (float *)blk, (float *)(blk + ap_nb), (float *)(blk + ap_nb + bp_nb));
    packed_batch = b;
    packed_jc = jc;
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
}

static MAG_HOTPROC void mag_matmul_gemm(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t M = x->meta.coords.shape[0];
  int64_t K = x->meta.coords.shape[1];
  int64_t N = y->meta.coords.shape[1];
  int64_t sx0 = x->meta.coords.strides[0];
  int64_t sx1 = x->meta.coords.strides[1];
  int64_t sy0 = y->meta.coords.strides[0];
  int64_t sy1 = y->meta.coords.strides[1];
  void *restrict pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  mag_matmul_gemm_impl(payload, r->meta.dtype, M, N, K, 1, NULL, NULL, pr, px, sx0, sx1, py, sy0, sy1);
}
