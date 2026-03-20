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

#include "mat_cpu_ops_matmul_kernels.inl"

static int64_t mag_offset_rmn(const mag_tensor_t *t, int64_t flat, int64_t i, int64_t j) {
    int64_t ra = t->coords.rank;
    const int64_t *restrict td = t->coords.shape;
    const int64_t *restrict ts = t->coords.strides;
    if (mag_likely(ra <= 3)) { /* Fast path */
        switch (ra) {
            case 0: return 0;
            case 1: return flat*ts[0];
            case 2: return i*ts[0] + j*ts[1];
            case 3: return flat*ts[0] + i*ts[1] + j*ts[2];
        }
    }
    int64_t off = 0, rem = flat;
    for (int64_t d = ra-3; d >= 0; --d) {
        int64_t idx = rem % td[d];
        rem /= td[d];
        off += idx*ts[d];
    }
    off += i*ts[ra-2];
    off += j*ts[ra-1];
    return off;
}

static MAG_AINLINE void mag_mm_tile_8x8_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    const __mmask16 m8 = (__mmask16)0x00ff;

    __m512 C0, C1, C2, C3, C4, C5, C6, C7;
    if (acc) {
        __m256i t0 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 0*ldc));
        __m256i t1 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 1*ldc));
        __m256i t2 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 2*ldc));
        __m256i t3 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 3*ldc));
        __m256i t4 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 4*ldc));
        __m256i t5 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 5*ldc));
        __m256i t6 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 6*ldc));
        __m256i t7 = _mm256_maskz_loadu_epi16(m8, (const void *)(c + 7*ldc));
        C0 = _mm512_cvtpbh_ps((__m256bh)t0);
        C1 = _mm512_cvtpbh_ps((__m256bh)t1);
        C2 = _mm512_cvtpbh_ps((__m256bh)t2);
        C3 = _mm512_cvtpbh_ps((__m256bh)t3);
        C4 = _mm512_cvtpbh_ps((__m256bh)t4);
        C5 = _mm512_cvtpbh_ps((__m256bh)t5);
        C6 = _mm512_cvtpbh_ps((__m256bh)t6);
        C7 = _mm512_cvtpbh_ps((__m256bh)t7);
    } else {
        __m512 z = _mm512_setzero_ps();
        C0 = C1 = C2 = C3 = C4 = C5 = C6 = C7 = z;
    }

    int64_t k = 0;
    for (; k + 7 < kc; k += 8) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L1) << 3));
            mag_simd_prefetch_t1(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L2) << 3));
        }

#define MAG_STEP8(KI) do { \
            __m256i bi = _mm256_maskz_loadu_epi16(m8, (const void *)(b + (k + (KI))*ldb)); \
            __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi); \
            __m256i ai = _mm256_maskz_loadu_epi16(m8, (const void *)(a + (k + (KI))*8)); \
            __m512 Av = _mm512_cvtpbh_ps((__m256bh)ai); \
            __m512 A0 = _mm512_permutexvar_ps(_mm512_set1_epi32(0), Av); \
            __m512 A1 = _mm512_permutexvar_ps(_mm512_set1_epi32(1), Av); \
            __m512 A2 = _mm512_permutexvar_ps(_mm512_set1_epi32(2), Av); \
            __m512 A3 = _mm512_permutexvar_ps(_mm512_set1_epi32(3), Av); \
            __m512 A4 = _mm512_permutexvar_ps(_mm512_set1_epi32(4), Av); \
            __m512 A5 = _mm512_permutexvar_ps(_mm512_set1_epi32(5), Av); \
            __m512 A6 = _mm512_permutexvar_ps(_mm512_set1_epi32(6), Av); \
            __m512 A7 = _mm512_permutexvar_ps(_mm512_set1_epi32(7), Av); \
            C0 = _mm512_fmadd_ps(A0, Bv, C0); \
            C1 = _mm512_fmadd_ps(A1, Bv, C1); \
            C2 = _mm512_fmadd_ps(A2, Bv, C2); \
            C3 = _mm512_fmadd_ps(A3, Bv, C3); \
            C4 = _mm512_fmadd_ps(A4, Bv, C4); \
            C5 = _mm512_fmadd_ps(A5, Bv, C5); \
            C6 = _mm512_fmadd_ps(A6, Bv, C6); \
            C7 = _mm512_fmadd_ps(A7, Bv, C7); \
        } while (0)

        MAG_STEP8(0);
        MAG_STEP8(1);
        MAG_STEP8(2);
        MAG_STEP8(3);
        MAG_STEP8(4);
        MAG_STEP8(5);
        MAG_STEP8(6);
        MAG_STEP8(7);
#undef MAG_STEP8
    }
    for (; k < kc; ++k) {
        __m256i bi = _mm256_maskz_loadu_epi16(m8, (const void *)(b + k*ldb));
        __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi);
        __m256i ai = _mm256_maskz_loadu_epi16(m8, (const void *)(a + k*8));
        __m512 Av = _mm512_cvtpbh_ps((__m256bh)ai);

        __m512 A0 = _mm512_permutexvar_ps(_mm512_set1_epi32(0), Av);
        __m512 A1 = _mm512_permutexvar_ps(_mm512_set1_epi32(1), Av);
        __m512 A2 = _mm512_permutexvar_ps(_mm512_set1_epi32(2), Av);
        __m512 A3 = _mm512_permutexvar_ps(_mm512_set1_epi32(3), Av);
        __m512 A4 = _mm512_permutexvar_ps(_mm512_set1_epi32(4), Av);
        __m512 A5 = _mm512_permutexvar_ps(_mm512_set1_epi32(5), Av);
        __m512 A6 = _mm512_permutexvar_ps(_mm512_set1_epi32(6), Av);
        __m512 A7 = _mm512_permutexvar_ps(_mm512_set1_epi32(7), Av);

        C0 = _mm512_fmadd_ps(A0, Bv, C0);
        C1 = _mm512_fmadd_ps(A1, Bv, C1);
        C2 = _mm512_fmadd_ps(A2, Bv, C2);
        C3 = _mm512_fmadd_ps(A3, Bv, C3);
        C4 = _mm512_fmadd_ps(A4, Bv, C4);
        C5 = _mm512_fmadd_ps(A5, Bv, C5);
        C6 = _mm512_fmadd_ps(A6, Bv, C6);
        C7 = _mm512_fmadd_ps(A7, Bv, C7);
    }

    __m256bh o0 = _mm512_cvtneps_pbh(C0);
    __m256bh o1 = _mm512_cvtneps_pbh(C1);
    __m256bh o2 = _mm512_cvtneps_pbh(C2);
    __m256bh o3 = _mm512_cvtneps_pbh(C3);
    __m256bh o4 = _mm512_cvtneps_pbh(C4);
    __m256bh o5 = _mm512_cvtneps_pbh(C5);
    __m256bh o6 = _mm512_cvtneps_pbh(C6);
    __m256bh o7 = _mm512_cvtneps_pbh(C7);

    _mm256_mask_storeu_epi16((void *)(c + 0*ldc), m8, (__m256i)o0);
    _mm256_mask_storeu_epi16((void *)(c + 1*ldc), m8, (__m256i)o1);
    _mm256_mask_storeu_epi16((void *)(c + 2*ldc), m8, (__m256i)o2);
    _mm256_mask_storeu_epi16((void *)(c + 3*ldc), m8, (__m256i)o3);
    _mm256_mask_storeu_epi16((void *)(c + 4*ldc), m8, (__m256i)o4);
    _mm256_mask_storeu_epi16((void *)(c + 5*ldc), m8, (__m256i)o5);
    _mm256_mask_storeu_epi16((void *)(c + 6*ldc), m8, (__m256i)o6);
    _mm256_mask_storeu_epi16((void *)(c + 7*ldc), m8, (__m256i)o7);

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)) || defined(_M_ARM64)
    float32x4_t C[8][2];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            bfloat16x8_t cv = vld1q_bf16((const bfloat16_t *)(c + r*ldc));
            C[r][0] = vcvt_f32_bf16(vget_low_bf16(cv));
            C[r][1] = vcvt_f32_bf16(vget_high_bf16(cv));
        }
    } else {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            C[r][0] = vdupq_n_f32(0.f);
            C[r][1] = vdupq_n_f32(0.f);
        }
    }

    int64_t k = 0;
    for (; k + 7 < kc; k += 8) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            __builtin_prefetch(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb, 0, 3);
            __builtin_prefetch(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb, 0, 2);
            __builtin_prefetch(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L1) << 3), 0, 3);
            __builtin_prefetch(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L2) << 3), 0, 2);
        }

#define MAG_STEP8N(KI) do { \
            bfloat16x8_t bv = vld1q_bf16((const bfloat16_t *)(b + (k + (KI))*ldb)); \
            float32x4_t B0 = vcvt_f32_bf16(vget_low_bf16(bv)); \
            float32x4_t B1 = vcvt_f32_bf16(vget_high_bf16(bv)); \
            bfloat16x8_t av = vld1q_bf16((const bfloat16_t *)(a + (k + (KI))*8)); \
            float32x4_t A0 = vcvt_f32_bf16(vget_low_bf16(av)); \
            float32x4_t A1 = vcvt_f32_bf16(vget_high_bf16(av)); \
            C[0][0] = vfmaq_laneq_f32(C[0][0], B0, A0, 0); C[0][1] = vfmaq_laneq_f32(C[0][1], B1, A0, 0); \
            C[1][0] = vfmaq_laneq_f32(C[1][0], B0, A0, 1); C[1][1] = vfmaq_laneq_f32(C[1][1], B1, A0, 1); \
            C[2][0] = vfmaq_laneq_f32(C[2][0], B0, A0, 2); C[2][1] = vfmaq_laneq_f32(C[2][1], B1, A0, 2); \
            C[3][0] = vfmaq_laneq_f32(C[3][0], B0, A0, 3); C[3][1] = vfmaq_laneq_f32(C[3][1], B1, A0, 3); \
            C[4][0] = vfmaq_laneq_f32(C[4][0], B0, A1, 0); C[4][1] = vfmaq_laneq_f32(C[4][1], B1, A1, 0); \
            C[5][0] = vfmaq_laneq_f32(C[5][0], B0, A1, 1); C[5][1] = vfmaq_laneq_f32(C[5][1], B1, A1, 1); \
            C[6][0] = vfmaq_laneq_f32(C[6][0], B0, A1, 2); C[6][1] = vfmaq_laneq_f32(C[6][1], B1, A1, 2); \
            C[7][0] = vfmaq_laneq_f32(C[7][0], B0, A1, 3); C[7][1] = vfmaq_laneq_f32(C[7][1], B1, A1, 3); \
        } while (0)

        MAG_STEP8N(0);
        MAG_STEP8N(1);
        MAG_STEP8N(2);
        MAG_STEP8N(3);
        MAG_STEP8N(4);
        MAG_STEP8N(5);
        MAG_STEP8N(6);
        MAG_STEP8N(7);
#undef MAG_STEP8N
    }
    for (; k < kc; ++k) {
        bfloat16x8_t bv = vld1q_bf16((const bfloat16_t *)(b + k*ldb));
        float32x4_t B0 = vcvt_f32_bf16(vget_low_bf16(bv));
        float32x4_t B1 = vcvt_f32_bf16(vget_high_bf16(bv));
        bfloat16x8_t av = vld1q_bf16((const bfloat16_t *)(a + k*8));
        float32x4_t A0 = vcvt_f32_bf16(vget_low_bf16(av));
        float32x4_t A1 = vcvt_f32_bf16(vget_high_bf16(av));
        C[0][0] = vfmaq_laneq_f32(C[0][0], B0, A0, 0); C[0][1] = vfmaq_laneq_f32(C[0][1], B1, A0, 0);
        C[1][0] = vfmaq_laneq_f32(C[1][0], B0, A0, 1); C[1][1] = vfmaq_laneq_f32(C[1][1], B1, A0, 1);
        C[2][0] = vfmaq_laneq_f32(C[2][0], B0, A0, 2); C[2][1] = vfmaq_laneq_f32(C[2][1], B1, A0, 2);
        C[3][0] = vfmaq_laneq_f32(C[3][0], B0, A0, 3); C[3][1] = vfmaq_laneq_f32(C[3][1], B1, A0, 3);
        C[4][0] = vfmaq_laneq_f32(C[4][0], B0, A1, 0); C[4][1] = vfmaq_laneq_f32(C[4][1], B1, A1, 0);
        C[5][0] = vfmaq_laneq_f32(C[5][0], B0, A1, 1); C[5][1] = vfmaq_laneq_f32(C[5][1], B1, A1, 1);
        C[6][0] = vfmaq_laneq_f32(C[6][0], B0, A1, 2); C[6][1] = vfmaq_laneq_f32(C[6][1], B1, A1, 2);
        C[7][0] = vfmaq_laneq_f32(C[7][0], B0, A1, 3); C[7][1] = vfmaq_laneq_f32(C[7][1], B1, A1, 3);
    }

#pragma GCC unroll 8
    for (int r=0; r < 8; ++r) {
        bfloat16x4_t o0 = vcvt_bf16_f32(C[r][0]);
        bfloat16x4_t o1 = vcvt_bf16_f32(C[r][1]);
        bfloat16x8_t out = vcombine_bf16(o0, o1);
        vst1q_bf16((bfloat16_t *)(c + r*ldc), out);
    }

#elif defined(__AVX2__) && defined(__FMA__)
    /* AVX2 bf16: soft bf16->fp32 then 8x8 FMA (no AVX512-BF16 on AVX2-only CPUs) */
    __m256 C[8];
    if (acc) {
        for (int r = 0; r < 8; ++r) {
            __m128i cv = _mm_loadu_si128((const __m128i *)(c + r*ldc));
            __m256i cu = _mm256_cvtepu16_epi32(cv);
            C[r] = _mm256_castsi256_ps(_mm256_slli_epi32(cu, 16));
        }
    } else {
        __m256 z = _mm256_setzero_ps();
        for (int r = 0; r < 8; ++r) C[r] = z;
    }
    for (int64_t k = 0; k < kc; ++k) {
        __m128i av = _mm_loadu_si128((const __m128i *)(a + k*8));
        __m256 a_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(av), 16));
        __m128i bv = _mm_loadu_si128((const __m128i *)(b + k*ldb));
        __m256 b_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bv), 16));
        for (int r = 0; r < 8; ++r)
            C[r] = _mm256_fmadd_ps(_mm256_permutevar8x32_ps(a_f, _mm256_set1_epi32(r)), b_f, C[r]);
    }
    for (int r = 0; r < 8; ++r) {
        __m256i ci = _mm256_castps_si256(C[r]);
        __m256i sh = _mm256_srli_epi32(ci, 16);
        __m128i lo = _mm256_castsi256_si128(sh);
        __m128i hi = _mm256_extracti128_si256(sh, 1);
        _mm_storeu_si128((__m128i *)(c + r*ldc), _mm_packus_epi32(lo, hi));
    }

#else
    /* scalar fallback (allowed conversions) */
    for (int64_t rr = 0; rr < 8; ++rr) {
        for (int64_t cc = 0; cc < 8; ++cc) {
            float sum = acc ? mag_bfloat16_to_float32(c[rr*ldc + cc]) : 0.f;
            for (int64_t k = 0; k < kc; ++k) {
                float av = mag_bfloat16_to_float32(a[k*8 + rr]);
                float bv = mag_bfloat16_to_float32(b[k*ldb + cc]);
                sum += av * bv;
            }
            c[rr*ldc + cc] = mag_float32_to_bfloat16(sum);
        }
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_8x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    __m512 C[8];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m256i t = _mm256_loadu_si256((const __m256i *)(c + r*ldc));
            C[r] = _mm512_cvtpbh_ps((__m256bh)t);
        }
    } else {
        __m512 z = _mm512_setzero_ps();
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) C[r] = z;
    }

    int64_t k = 0;
    for (; k + 7 < kc; k += 8) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L1) << 3));
            mag_simd_prefetch_t1(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L2) << 3));
        }

#define MAG_STEP16(KI) do { \
            __m256i bi = _mm256_loadu_si256((const __m256i *)(b + (k + (KI))*ldb)); \
            __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi); \
            __m256i ai = _mm256_loadu_si256((const __m256i *)(a + (k + (KI))*8)); \
            __m512 Av = _mm512_cvtpbh_ps((__m256bh)ai); \
            __m512 A0 = _mm512_permutexvar_ps(_mm512_set1_epi32(0), Av); \
            __m512 A1 = _mm512_permutexvar_ps(_mm512_set1_epi32(1), Av); \
            __m512 A2 = _mm512_permutexvar_ps(_mm512_set1_epi32(2), Av); \
            __m512 A3 = _mm512_permutexvar_ps(_mm512_set1_epi32(3), Av); \
            __m512 A4 = _mm512_permutexvar_ps(_mm512_set1_epi32(4), Av); \
            __m512 A5 = _mm512_permutexvar_ps(_mm512_set1_epi32(5), Av); \
            __m512 A6 = _mm512_permutexvar_ps(_mm512_set1_epi32(6), Av); \
            __m512 A7 = _mm512_permutexvar_ps(_mm512_set1_epi32(7), Av); \
            C[0] = _mm512_fmadd_ps(A0, Bv, C[0]); \
            C[1] = _mm512_fmadd_ps(A1, Bv, C[1]); \
            C[2] = _mm512_fmadd_ps(A2, Bv, C[2]); \
            C[3] = _mm512_fmadd_ps(A3, Bv, C[3]); \
            C[4] = _mm512_fmadd_ps(A4, Bv, C[4]); \
            C[5] = _mm512_fmadd_ps(A5, Bv, C[5]); \
            C[6] = _mm512_fmadd_ps(A6, Bv, C[6]); \
            C[7] = _mm512_fmadd_ps(A7, Bv, C[7]); \
        } while (0)

        MAG_STEP16(0);
        MAG_STEP16(1);
        MAG_STEP16(2);
        MAG_STEP16(3);
        MAG_STEP16(4);
        MAG_STEP16(5);
        MAG_STEP16(6);
        MAG_STEP16(7);
#undef MAG_STEP16
    }
    for (; k < kc; ++k) {
        __m256i bi = _mm256_loadu_si256((const __m256i *)(b + k*ldb));
        __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi);
        __m256i ai = _mm256_loadu_si256((const __m256i *)(a + k*8));
        __m512 Av = _mm512_cvtpbh_ps((__m256bh)ai);

        __m512 A0 = _mm512_permutexvar_ps(_mm512_set1_epi32(0), Av);
        __m512 A1 = _mm512_permutexvar_ps(_mm512_set1_epi32(1), Av);
        __m512 A2 = _mm512_permutexvar_ps(_mm512_set1_epi32(2), Av);
        __m512 A3 = _mm512_permutexvar_ps(_mm512_set1_epi32(3), Av);
        __m512 A4 = _mm512_permutexvar_ps(_mm512_set1_epi32(4), Av);
        __m512 A5 = _mm512_permutexvar_ps(_mm512_set1_epi32(5), Av);
        __m512 A6 = _mm512_permutexvar_ps(_mm512_set1_epi32(6), Av);
        __m512 A7 = _mm512_permutexvar_ps(_mm512_set1_epi32(7), Av);

        C[0] = _mm512_fmadd_ps(A0, Bv, C[0]);
        C[1] = _mm512_fmadd_ps(A1, Bv, C[1]);
        C[2] = _mm512_fmadd_ps(A2, Bv, C[2]);
        C[3] = _mm512_fmadd_ps(A3, Bv, C[3]);
        C[4] = _mm512_fmadd_ps(A4, Bv, C[4]);
        C[5] = _mm512_fmadd_ps(A5, Bv, C[5]);
        C[6] = _mm512_fmadd_ps(A6, Bv, C[6]);
        C[7] = _mm512_fmadd_ps(A7, Bv, C[7]);
    }

#pragma GCC unroll 8
    for (int r=0; r < 8; ++r) {
        __m256bh out = _mm512_cvtneps_pbh(C[r]);
        _mm256_storeu_si256((__m256i *)(c + r*ldc), (__m256i)out);
    }

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)) || defined(_M_ARM64)
    float32x4_t C[8][4];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            bfloat16x8_t c0 = vld1q_bf16((const bfloat16_t *)(c + r*ldc + 0));
            bfloat16x8_t c1 = vld1q_bf16((const bfloat16_t *)(c + r*ldc + 8));
            C[r][0] = vcvt_f32_bf16(vget_low_bf16(c0));
            C[r][1] = vcvt_f32_bf16(vget_high_bf16(c0));
            C[r][2] = vcvt_f32_bf16(vget_low_bf16(c1));
            C[r][3] = vcvt_f32_bf16(vget_high_bf16(c1));
        }
    } else {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r)
            C[r][0] = C[r][1] = C[r][2] = C[r][3] = vdupq_n_f32(0.f);
    }

    int64_t k = 0;
    for (; k + 7 < kc; k += 8) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            __builtin_prefetch(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb, 0, 3);
            __builtin_prefetch(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb, 0, 2);
            __builtin_prefetch(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L1) << 3), 0, 3);
            __builtin_prefetch(a + (int64_t)((k + MAG_PREFETCH_RANGE_A_L2) << 3), 0, 2);
        }

#define MAG_STEP16N(KI) do { \
            const bfloat16_t *Bk = (const bfloat16_t *)(b + (k + (KI))*ldb); \
            bfloat16x8_t b0 = vld1q_bf16(Bk + 0); \
            bfloat16x8_t b1 = vld1q_bf16(Bk + 8); \
            float32x4_t B00 = vcvt_f32_bf16(vget_low_bf16(b0)); \
            float32x4_t B01 = vcvt_f32_bf16(vget_high_bf16(b0)); \
            float32x4_t B10 = vcvt_f32_bf16(vget_low_bf16(b1)); \
            float32x4_t B11 = vcvt_f32_bf16(vget_high_bf16(b1)); \
            bfloat16x8_t av = vld1q_bf16((const bfloat16_t *)(a + (k + (KI))*8)); \
            float32x4_t A0 = vcvt_f32_bf16(vget_low_bf16(av)); \
            float32x4_t A1 = vcvt_f32_bf16(vget_high_bf16(av)); \
            C[0][0] = vfmaq_laneq_f32(C[0][0], B00, A0, 0); C[0][1] = vfmaq_laneq_f32(C[0][1], B01, A0, 0); C[0][2] = vfmaq_laneq_f32(C[0][2], B10, A0, 0); C[0][3] = vfmaq_laneq_f32(C[0][3], B11, A0, 0); \
            C[1][0] = vfmaq_laneq_f32(C[1][0], B00, A0, 1); C[1][1] = vfmaq_laneq_f32(C[1][1], B01, A0, 1); C[1][2] = vfmaq_laneq_f32(C[1][2], B10, A0, 1); C[1][3] = vfmaq_laneq_f32(C[1][3], B11, A0, 1); \
            C[2][0] = vfmaq_laneq_f32(C[2][0], B00, A0, 2); C[2][1] = vfmaq_laneq_f32(C[2][1], B01, A0, 2); C[2][2] = vfmaq_laneq_f32(C[2][2], B10, A0, 2); C[2][3] = vfmaq_laneq_f32(C[2][3], B11, A0, 2); \
            C[3][0] = vfmaq_laneq_f32(C[3][0], B00, A0, 3); C[3][1] = vfmaq_laneq_f32(C[3][1], B01, A0, 3); C[3][2] = vfmaq_laneq_f32(C[3][2], B10, A0, 3); C[3][3] = vfmaq_laneq_f32(C[3][3], B11, A0, 3); \
            C[4][0] = vfmaq_laneq_f32(C[4][0], B00, A1, 0); C[4][1] = vfmaq_laneq_f32(C[4][1], B01, A1, 0); C[4][2] = vfmaq_laneq_f32(C[4][2], B10, A1, 0); C[4][3] = vfmaq_laneq_f32(C[4][3], B11, A1, 0); \
            C[5][0] = vfmaq_laneq_f32(C[5][0], B00, A1, 1); C[5][1] = vfmaq_laneq_f32(C[5][1], B01, A1, 1); C[5][2] = vfmaq_laneq_f32(C[5][2], B10, A1, 1); C[5][3] = vfmaq_laneq_f32(C[5][3], B11, A1, 1); \
            C[6][0] = vfmaq_laneq_f32(C[6][0], B00, A1, 2); C[6][1] = vfmaq_laneq_f32(C[6][1], B01, A1, 2); C[6][2] = vfmaq_laneq_f32(C[6][2], B10, A1, 2); C[6][3] = vfmaq_laneq_f32(C[6][3], B11, A1, 2); \
            C[7][0] = vfmaq_laneq_f32(C[7][0], B00, A1, 3); C[7][1] = vfmaq_laneq_f32(C[7][1], B01, A1, 3); C[7][2] = vfmaq_laneq_f32(C[7][2], B10, A1, 3); C[7][3] = vfmaq_laneq_f32(C[7][3], B11, A1, 3); \
        } while (0)

        MAG_STEP16N(0);
        MAG_STEP16N(1);
        MAG_STEP16N(2);
        MAG_STEP16N(3);
        MAG_STEP16N(4);
        MAG_STEP16N(5);
        MAG_STEP16N(6);
        MAG_STEP16N(7);
#undef MAG_STEP16N
    }
    for (; k < kc; ++k) {
        const bfloat16_t *Bk = (const bfloat16_t *)(b + k*ldb);
        bfloat16x8_t b0 = vld1q_bf16(Bk + 0);
        bfloat16x8_t b1 = vld1q_bf16(Bk + 8);
        float32x4_t B00 = vcvt_f32_bf16(vget_low_bf16(b0));
        float32x4_t B01 = vcvt_f32_bf16(vget_high_bf16(b0));
        float32x4_t B10 = vcvt_f32_bf16(vget_low_bf16(b1));
        float32x4_t B11 = vcvt_f32_bf16(vget_high_bf16(b1));
        bfloat16x8_t av = vld1q_bf16((const bfloat16_t *)(a + k*8));
        float32x4_t A0 = vcvt_f32_bf16(vget_low_bf16(av));
        float32x4_t A1 = vcvt_f32_bf16(vget_high_bf16(av));
        C[0][0] = vfmaq_laneq_f32(C[0][0], B00, A0, 0); C[0][1] = vfmaq_laneq_f32(C[0][1], B01, A0, 0); C[0][2] = vfmaq_laneq_f32(C[0][2], B10, A0, 0); C[0][3] = vfmaq_laneq_f32(C[0][3], B11, A0, 0);
        C[1][0] = vfmaq_laneq_f32(C[1][0], B00, A0, 1); C[1][1] = vfmaq_laneq_f32(C[1][1], B01, A0, 1); C[1][2] = vfmaq_laneq_f32(C[1][2], B10, A0, 1); C[1][3] = vfmaq_laneq_f32(C[1][3], B11, A0, 1);
        C[2][0] = vfmaq_laneq_f32(C[2][0], B00, A0, 2); C[2][1] = vfmaq_laneq_f32(C[2][1], B01, A0, 2); C[2][2] = vfmaq_laneq_f32(C[2][2], B10, A0, 2); C[2][3] = vfmaq_laneq_f32(C[2][3], B11, A0, 2);
        C[3][0] = vfmaq_laneq_f32(C[3][0], B00, A0, 3); C[3][1] = vfmaq_laneq_f32(C[3][1], B01, A0, 3); C[3][2] = vfmaq_laneq_f32(C[3][2], B10, A0, 3); C[3][3] = vfmaq_laneq_f32(C[3][3], B11, A0, 3);
        C[4][0] = vfmaq_laneq_f32(C[4][0], B00, A1, 0); C[4][1] = vfmaq_laneq_f32(C[4][1], B01, A1, 0); C[4][2] = vfmaq_laneq_f32(C[4][2], B10, A1, 0); C[4][3] = vfmaq_laneq_f32(C[4][3], B11, A1, 0);
        C[5][0] = vfmaq_laneq_f32(C[5][0], B00, A1, 1); C[5][1] = vfmaq_laneq_f32(C[5][1], B01, A1, 1); C[5][2] = vfmaq_laneq_f32(C[5][2], B10, A1, 1); C[5][3] = vfmaq_laneq_f32(C[5][3], B11, A1, 1);
        C[6][0] = vfmaq_laneq_f32(C[6][0], B00, A1, 2); C[6][1] = vfmaq_laneq_f32(C[6][1], B01, A1, 2); C[6][2] = vfmaq_laneq_f32(C[6][2], B10, A1, 2); C[6][3] = vfmaq_laneq_f32(C[6][3], B11, A1, 2);
        C[7][0] = vfmaq_laneq_f32(C[7][0], B00, A1, 3); C[7][1] = vfmaq_laneq_f32(C[7][1], B01, A1, 3); C[7][2] = vfmaq_laneq_f32(C[7][2], B10, A1, 3); C[7][3] = vfmaq_laneq_f32(C[7][3], B11, A1, 3);
    }

#pragma GCC unroll 8
    for (int r=0; r < 8; ++r) {
        bfloat16x8_t o0 = vcombine_bf16(vcvt_bf16_f32(C[r][0]), vcvt_bf16_f32(C[r][1]));
        bfloat16x8_t o1 = vcombine_bf16(vcvt_bf16_f32(C[r][2]), vcvt_bf16_f32(C[r][3]));
        vst1q_bf16((bfloat16_t *)(c + r*ldc + 0), o0);
        vst1q_bf16((bfloat16_t *)(c + r*ldc + 8), o1);
    }

#elif defined(__AVX2__) && defined(__FMA__)
    /* AVX2 bf16 8x16: soft bf16->fp32 then FMA */
    __m256 C[8][2];
    if (acc) {
        for (int r = 0; r < 8; ++r) {
            __m128i c0 = _mm_loadu_si128((const __m128i *)(c + r*ldc + 0));
            __m128i c1 = _mm_loadu_si128((const __m128i *)(c + r*ldc + 8));
            C[r][0] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(c0), 16));
            C[r][1] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(c1), 16));
        }
    } else {
        __m256 z = _mm256_setzero_ps();
        for (int r = 0; r < 8; ++r) C[r][0] = C[r][1] = z;
    }
    for (int64_t k = 0; k < kc; ++k) {
        __m128i av = _mm_loadu_si128((const __m128i *)(a + k*8));
        __m256 a_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(av), 16));
        __m256i bv = _mm256_loadu_si256((const __m256i *)(b + k*ldb));
        __m256 b_f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(bv)), 16));
        __m256 b_f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(bv, 1)), 16));
        for (int r = 0; r < 8; ++r) {
            __m256 ar = _mm256_permutevar8x32_ps(a_f, _mm256_set1_epi32(r));
            C[r][0] = _mm256_fmadd_ps(ar, b_f0, C[r][0]);
            C[r][1] = _mm256_fmadd_ps(ar, b_f1, C[r][1]);
        }
    }
    for (int r = 0; r < 8; ++r) {
        for (int j = 0; j < 2; ++j) {
            __m256i ci = _mm256_castps_si256(C[r][j]);
            __m256i sh = _mm256_srli_epi32(ci, 16);
            __m128i lo = _mm256_castsi256_si128(sh);
            __m128i hi = _mm256_extracti128_si256(sh, 1);
            _mm_storeu_si128((__m128i *)(c + r*ldc + j*8), _mm_packus_epi32(lo, hi));
        }
    }

#else
    for (int64_t rr = 0; rr < 8; ++rr) {
        for (int64_t cc = 0; cc < 16; ++cc) {
            float sum = acc ? mag_bfloat16_to_float32(c[rr*ldc + cc]) : 0.f;
            for (int64_t k = 0; k < kc; ++k) {
                float av = mag_bfloat16_to_float32(a[k*8 + rr]);
                float bv = mag_bfloat16_to_float32(b[k*ldb + cc]);
                sum += av * bv;
            }
            c[rr*ldc + cc] = mag_float32_to_bfloat16(sum);
        }
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_8x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_8x16_bfloat16(kc, a, lda, b,     ldb, c,     ldc, acc);
    mag_mm_tile_8x16_bfloat16(kc, a, lda, b+16,  ldb, c+16,  ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    const __mmask16 m8 = (__mmask16)0x00ff;

    __m256i ct = acc ? _mm256_maskz_loadu_epi16(m8, (const void *)c) : _mm256_setzero_si256();
    __m512 C = acc ? _mm512_cvtpbh_ps((__m256bh)ct) : _mm512_setzero_ps();

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }

#define MAG_STEP1x8(KI) do { \
            uint16_t abits = a[k + (KI)].bits; \
            __m256bh Abh = (__m256bh)_mm256_set1_epi16((short)abits); \
            __m512 A = _mm512_cvtpbh_ps(Abh); \
            __m256i bi = _mm256_maskz_loadu_epi16(m8, (const void *)(b + (k + (KI))*ldb)); \
            __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi); \
            C = _mm512_fmadd_ps(A, Bv, C); \
        } while (0)

        MAG_STEP1x8(0);
        MAG_STEP1x8(1);
        MAG_STEP1x8(2);
        MAG_STEP1x8(3);
#undef MAG_STEP1x8
    }
    for (; k < kc; ++k) {
        uint16_t abits = a[k].bits;
        __m256bh Abh = (__m256bh)_mm256_set1_epi16((short)abits);
        __m512 A = _mm512_cvtpbh_ps(Abh);

        __m256i bi = _mm256_maskz_loadu_epi16(m8, (const void *)(b + k*ldb));
        __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi);

        C = _mm512_fmadd_ps(A, Bv, C);
    }

    __m256bh out = _mm512_cvtneps_pbh(C);
    _mm256_mask_storeu_epi16((void *)c, m8, (__m256i)out);

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)) || defined(_M_ARM64)
    float32x4_t C0, C1;
    if (acc) {
        bfloat16x8_t cv = vld1q_bf16((const bfloat16_t *)c);
        C0 = vcvt_f32_bf16(vget_low_bf16(cv));
        C1 = vcvt_f32_bf16(vget_high_bf16(cv));
    } else {
        C0 = vdupq_n_f32(0.f);
        C1 = vdupq_n_f32(0.f);
    }

    for (int64_t k = 0; k < kc; ++k) {
        bfloat16x8_t bv = vld1q_bf16((const bfloat16_t *)(b + k*ldb));
        float32x4_t B0 = vcvt_f32_bf16(vget_low_bf16(bv));
        float32x4_t B1 = vcvt_f32_bf16(vget_high_bf16(bv));

        bfloat16x4_t av = vdup_n_bf16(*(const bfloat16_t *)&a[k]);
        float32x4_t A = vcvt_f32_bf16(av);

        C0 = vfmaq_f32(C0, B0, A);
        C1 = vfmaq_f32(C1, B1, A);
    }

    bfloat16x8_t out = vcombine_bf16(vcvt_bf16_f32(C0), vcvt_bf16_f32(C1));
    vst1q_bf16((bfloat16_t *)c, out);

#else
    for (int64_t j=0; j < 8; ++j) {
        float sum = acc ? mag_bfloat16_to_float32(c[j]) : 0.f;
        for (int64_t k=0; k < kc; ++k)
            sum += mag_bfloat16_to_float32(a[k]) * mag_bfloat16_to_float32(b[k*ldb + j]);
        c[j] = mag_float32_to_bfloat16(sum);
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    __m512 C;
    if (acc) {
        __m256i ct = _mm256_loadu_si256((const __m256i *)c);
        C = _mm512_cvtpbh_ps((__m256bh)ct);
    } else {
        C = _mm512_setzero_ps();
    }
    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & (MAG_PREFETCH_GROUPS - 1))) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }
#define MAG_STEP1x16(KI) do { \
            uint16_t abits = a[k + (KI)].bits; \
            __m256bh Abh = (__m256bh)_mm256_set1_epi16((short)abits); \
            __m512 A = _mm512_cvtpbh_ps(Abh); \
            __m256i bi = _mm256_loadu_si256((const __m256i *)(b + (k + (KI))*ldb)); \
            __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi); \
            C = _mm512_fmadd_ps(A, Bv, C); \
        } while (0)

        MAG_STEP1x16(0);
        MAG_STEP1x16(1);
        MAG_STEP1x16(2);
        MAG_STEP1x16(3);
#undef MAG_STEP1x16
    }
    for (; k < kc; ++k) {
        uint16_t abits = a[k].bits;
        __m256bh Abh = (__m256bh)_mm256_set1_epi16((short)abits);
        __m512 A = _mm512_cvtpbh_ps(Abh);

        __m256i bi = _mm256_loadu_si256((const __m256i *)(b + k*ldb));
        __m512 Bv = _mm512_cvtpbh_ps((__m256bh)bi);

        C = _mm512_fmadd_ps(A, Bv, C);
    }

    __m256bh out = _mm512_cvtneps_pbh(C);
    _mm256_storeu_si256((__m256i *)c, (__m256i)out);

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)) || defined(_M_ARM64)
    float32x4_t C0, C1, C2, C3;
    if (acc) {
        bfloat16x8_t c0 = vld1q_bf16((const bfloat16_t *)(c + 0));
        bfloat16x8_t c1 = vld1q_bf16((const bfloat16_t *)(c + 8));
        C0 = vcvt_f32_bf16(vget_low_bf16(c0));
        C1 = vcvt_f32_bf16(vget_high_bf16(c0));
        C2 = vcvt_f32_bf16(vget_low_bf16(c1));
        C3 = vcvt_f32_bf16(vget_high_bf16(c1));
    } else {
        C0 = C1 = C2 = C3 = vdupq_n_f32(0.f);
    }

    for (int64_t k = 0; k < kc; ++k) {
        const bfloat16_t *Bk = (const bfloat16_t *)(b + k*ldb);
        bfloat16x8_t b0 = vld1q_bf16(Bk + 0);
        bfloat16x8_t b1 = vld1q_bf16(Bk + 8);
        float32x4_t B00 = vcvt_f32_bf16(vget_low_bf16(b0));
        float32x4_t B01 = vcvt_f32_bf16(vget_high_bf16(b0));
        float32x4_t B10 = vcvt_f32_bf16(vget_low_bf16(b1));
        float32x4_t B11 = vcvt_f32_bf16(vget_high_bf16(b1));

        bfloat16x4_t av = vdup_n_bf16(*(const bfloat16_t *)&a[k]);
        float32x4_t A = vcvt_f32_bf16(av);

        C0 = vfmaq_f32(C0, B00, A);
        C1 = vfmaq_f32(C1, B01, A);
        C2 = vfmaq_f32(C2, B10, A);
        C3 = vfmaq_f32(C3, B11, A);
    }

    bfloat16x8_t o0 = vcombine_bf16(vcvt_bf16_f32(C0), vcvt_bf16_f32(C1));
    bfloat16x8_t o1 = vcombine_bf16(vcvt_bf16_f32(C2), vcvt_bf16_f32(C3));
    vst1q_bf16((bfloat16_t *)(c + 0), o0);
    vst1q_bf16((bfloat16_t *)(c + 8), o1);

#else
    for (int64_t j=0; j < 16; ++j) {
        float sum = acc ? mag_bfloat16_to_float32(c[j]) : 0.f;
        for (int64_t k=0; k < kc; ++k)
            sum += mag_bfloat16_to_float32(a[k]) * mag_bfloat16_to_float32(b[k*ldb + j]);
        c[j] = mag_float32_to_bfloat16(sum);
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
    mag_mm_tile_1x16_bfloat16(kc, a, b,     ldb, c,     acc);
    mag_mm_tile_1x16_bfloat16(kc, a, b+16,  ldb, c+16,  acc);
}

static MAG_AINLINE void mag_mm_tile_16x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_8x16_bfloat16(kc, a,         lda, b, ldb, c,         ldc, acc);
    mag_mm_tile_8x16_bfloat16(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_16x16_bfloat16(kc, a, lda, b,     ldb, c,     ldc, acc);
    mag_mm_tile_16x16_bfloat16(kc, a, lda, b+16,  ldb, c+16,  ldc, acc);
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_bfloat16(
    int64_t kc, int64_t nc,
    const mag_bfloat16_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN,
    mag_bfloat16_t *restrict Bp
) {
    if (strideN == 1) {
        for (int64_t k = 0; k < kc; ++k) {
            if (k + 1 < kc)
                mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1) * strideK));
            const mag_bfloat16_t *src = Bsrc + k * strideK;
            mag_bfloat16_t *dst = Bp + k * nc;

#if defined(__AVX512F__) && defined(__AVX512BF16__)
            int64_t j = 0;
            for (; j + 255 < nc; j += 256) {
                mag_simd_prefetch_t0((const char*)(src + j) + 512);
                mag_simd_prefetch_t1((const char*)(src + j) + 2048);
                __m512i w0 = _mm512_loadu_si512((const __m512i*)(src + j +  0));
                __m512i w1 = _mm512_loadu_si512((const __m512i*)(src + j + 32));
                __m512i w2 = _mm512_loadu_si512((const __m512i*)(src + j + 64));
                __m512i w3 = _mm512_loadu_si512((const __m512i*)(src + j + 96));
                __m512i w4 = _mm512_loadu_si512((const __m512i*)(src + j + 128));
                __m512i w5 = _mm512_loadu_si512((const __m512i*)(src + j + 160));
                __m512i w6 = _mm512_loadu_si512((const __m512i*)(src + j + 192));
                __m512i w7 = _mm512_loadu_si512((const __m512i*)(src + j + 224));
                _mm512_storeu_si512((__m512i*)(dst + j +  0), w0);
                _mm512_storeu_si512((__m512i*)(dst + j + 32), w1);
                _mm512_storeu_si512((__m512i*)(dst + j + 64), w2);
                _mm512_storeu_si512((__m512i*)(dst + j + 96), w3);
                _mm512_storeu_si512((__m512i*)(dst + j + 128), w4);
                _mm512_storeu_si512((__m512i*)(dst + j + 160), w5);
                _mm512_storeu_si512((__m512i*)(dst + j + 192), w6);
                _mm512_storeu_si512((__m512i*)(dst + j + 224), w7);
            }
            for (; j + 127 < nc; j += 128) {
                mag_simd_prefetch_t0((const char*)(src + j) + 256);
                mag_simd_prefetch_t1((const char*)(src + j) + 1024);

                __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + j +  0));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + j + 16));
                __m256i v2 = _mm256_loadu_si256((const __m256i*)(src + j + 32));
                __m256i v3 = _mm256_loadu_si256((const __m256i*)(src + j + 48));
                __m256i v4 = _mm256_loadu_si256((const __m256i*)(src + j + 64));
                __m256i v5 = _mm256_loadu_si256((const __m256i*)(src + j + 80));
                __m256i v6 = _mm256_loadu_si256((const __m256i*)(src + j + 96));
                __m256i v7 = _mm256_loadu_si256((const __m256i*)(src + j + 112));

                _mm256_storeu_si256((__m256i*)(dst + j +  0), v0);
                _mm256_storeu_si256((__m256i*)(dst + j + 16), v1);
                _mm256_storeu_si256((__m256i*)(dst + j + 32), v2);
                _mm256_storeu_si256((__m256i*)(dst + j + 48), v3);
                _mm256_storeu_si256((__m256i*)(dst + j + 64), v4);
                _mm256_storeu_si256((__m256i*)(dst + j + 80), v5);
                _mm256_storeu_si256((__m256i*)(dst + j + 96), v6);
                _mm256_storeu_si256((__m256i*)(dst + j + 112), v7);
            }
            for (; j+63 < nc; j += 64) {
                __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + j +  0));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + j + 16));
                __m256i v2 = _mm256_loadu_si256((const __m256i*)(src + j + 32));
                __m256i v3 = _mm256_loadu_si256((const __m256i*)(src + j + 48));
                _mm256_storeu_si256((__m256i*)(dst + j +  0), v0);
                _mm256_storeu_si256((__m256i*)(dst + j + 16), v1);
                _mm256_storeu_si256((__m256i*)(dst + j + 32), v2);
                _mm256_storeu_si256((__m256i*)(dst + j + 48), v3);
            }
            for (; j+31 < nc; j += 32) {
                __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + j +  0));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + j + 16));
                _mm256_storeu_si256((__m256i*)(dst + j +  0), v0);
                _mm256_storeu_si256((__m256i*)(dst + j + 16), v1);
            }
            for (; j+15 < nc; j += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + j));
                _mm256_storeu_si256((__m256i*)(dst + j), v);
            }
            if (j < nc) {
                for (; j < nc; ++j) dst[j] = src[j];
            }

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
            if (k + 1 < kc)
                __builtin_prefetch((const char *)(Bsrc + (k + 1) * strideK), 0, 3);
            int64_t j = 0;
            for (; j + 63 < nc; j += 64) {
                __builtin_prefetch((const char *)(src + j + 128), 0, 3);
                __builtin_prefetch((const char *)(src + j + 256), 0, 2);
                uint16x8_t v0 = vld1q_u16((const uint16_t*)(src + j +  0));
                uint16x8_t v1 = vld1q_u16((const uint16_t*)(src + j +  8));
                uint16x8_t v2 = vld1q_u16((const uint16_t*)(src + j + 16));
                uint16x8_t v3 = vld1q_u16((const uint16_t*)(src + j + 24));
                uint16x8_t v4 = vld1q_u16((const uint16_t*)(src + j + 32));
                uint16x8_t v5 = vld1q_u16((const uint16_t*)(src + j + 40));
                uint16x8_t v6 = vld1q_u16((const uint16_t*)(src + j + 48));
                uint16x8_t v7 = vld1q_u16((const uint16_t*)(src + j + 56));
                vst1q_u16((uint16_t*)(dst + j +  0), v0);
                vst1q_u16((uint16_t*)(dst + j +  8), v1);
                vst1q_u16((uint16_t*)(dst + j + 16), v2);
                vst1q_u16((uint16_t*)(dst + j + 24), v3);
                vst1q_u16((uint16_t*)(dst + j + 32), v4);
                vst1q_u16((uint16_t*)(dst + j + 40), v5);
                vst1q_u16((uint16_t*)(dst + j + 48), v6);
                vst1q_u16((uint16_t*)(dst + j + 56), v7);
            }
            for (; j+31 < nc; j += 32) {
                __builtin_prefetch((const char *)(src + j + 64), 0, 3);
                uint16x8_t v0 = vld1q_u16((const uint16_t*)(src + j +  0));
                uint16x8_t v1 = vld1q_u16((const uint16_t*)(src + j +  8));
                uint16x8_t v2 = vld1q_u16((const uint16_t*)(src + j + 16));
                uint16x8_t v3 = vld1q_u16((const uint16_t*)(src + j + 24));
                vst1q_u16((uint16_t*)(dst + j +  0), v0);
                vst1q_u16((uint16_t*)(dst + j +  8), v1);
                vst1q_u16((uint16_t*)(dst + j + 16), v2);
                vst1q_u16((uint16_t*)(dst + j + 24), v3);
            }
            for (; j+15 < nc; j += 16) {
                uint16x8_t v0 = vld1q_u16((const uint16_t*)(src + j + 0));
                uint16x8_t v1 = vld1q_u16((const uint16_t*)(src + j + 8));
                vst1q_u16((uint16_t*)(dst + j + 0), v0);
                vst1q_u16((uint16_t*)(dst + j + 8), v1);
            }
            for (; j + 7 < nc; j += 8) {
                uint16x8_t v = vld1q_u16((const uint16_t*)(src + j));
                vst1q_u16((uint16_t*)(dst + j), v);
            }
            for (; j < nc; ++j) dst[j] = src[j];

#else
            /* portable: contiguous bf16 copy */
            memcpy(dst, src, (size_t)nc * sizeof(*Bsrc));
#endif
        }
    } else {
        /* non-contiguous in N: AVX-512 gather; index via iota*strideN+base, pack via cvtepi32_epi16 when BW */
#if defined(__AVX512F__)
        /* Hoist constant index pattern: [0,1,...,15]*strideN computed once */
        __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m512i strideN_v = _mm512_set1_epi32((int)strideN);
        __m512i step16 = _mm512_set1_epi32(16 * (int)strideN);
        __m512i step32 = _mm512_set1_epi32(32 * (int)strideN);
        __m512i step48 = _mm512_set1_epi32(48 * (int)strideN);
        for (int64_t k = 0; k < kc; ++k) {
            if (k + 1 < kc)
                mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1) * strideK));
            if (k + 2 < kc)
                mag_simd_prefetch_t1((const char *)(Bsrc + (k + 2) * strideK));
            const mag_bfloat16_t *src = Bsrc + k * strideK;
            mag_bfloat16_t *dst = Bp + k * nc;
            int64_t j = 0;
            for (; j + 64 <= nc; j += 64) {
                int base = (int)(j * (ptrdiff_t)strideN);
                __m512i idx0 = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_mullo_epi32(iota, strideN_v));
                __m512i idx1 = _mm512_add_epi32(idx0, step16);
                __m512i idx2 = _mm512_add_epi32(idx0, step32);
                __m512i idx3 = _mm512_add_epi32(idx0, step48);
                __m512i v0 = _mm512_i32gather_epi32(idx0, (const void *)src, 2);
                __m512i v1 = _mm512_i32gather_epi32(idx1, (const void *)src, 2);
                __m512i v2 = _mm512_i32gather_epi32(idx2, (const void *)src, 2);
                __m512i v3 = _mm512_i32gather_epi32(idx3, (const void *)src, 2);
                v0 = _mm512_and_si512(v0, _mm512_set1_epi32(0xFFFF));
                v1 = _mm512_and_si512(v1, _mm512_set1_epi32(0xFFFF));
                v2 = _mm512_and_si512(v2, _mm512_set1_epi32(0xFFFF));
                v3 = _mm512_and_si512(v3, _mm512_set1_epi32(0xFFFF));
#if defined(__AVX512BW__)
                __m256i p0 = _mm512_cvtepi32_epi16(v0);
                __m256i p1 = _mm512_cvtepi32_epi16(v1);
                __m256i p2 = _mm512_cvtepi32_epi16(v2);
                __m256i p3 = _mm512_cvtepi32_epi16(v3);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j), p0);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j + 16), p1);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j + 32), p2);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j + 48), p3);
#else
                __m256i v00 = _mm512_castsi512_si256(v0), v01 = _mm512_extracti32x8_epi32(v0, 1);
                __m256i v10 = _mm512_castsi512_si256(v1), v11 = _mm512_extracti32x8_epi32(v1, 1);
                __m256i v20 = _mm512_castsi512_si256(v2), v21 = _mm512_extracti32x8_epi32(v2, 1);
                __m256i v30 = _mm512_castsi512_si256(v3), v31 = _mm512_extracti32x8_epi32(v3, 1);
                __m128i lo, hi;
                lo = _mm256_castsi256_si128(v00); hi = _mm256_extracti128_si256(v00, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v01); hi = _mm256_extracti128_si256(v01, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 8), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v10); hi = _mm256_extracti128_si256(v10, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 16), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v11); hi = _mm256_extracti128_si256(v11, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 24), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v20); hi = _mm256_extracti128_si256(v20, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 32), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v21); hi = _mm256_extracti128_si256(v21, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 40), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v30); hi = _mm256_extracti128_si256(v30, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 48), _mm_packus_epi32(lo, hi));
                lo = _mm256_castsi256_si128(v31); hi = _mm256_extracti128_si256(v31, 1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 56), _mm_packus_epi32(lo, hi));
#endif
            }
            for (; j + 32 <= nc; j += 32) {
                int base = (int)(j * (ptrdiff_t)strideN);
                __m512i idx_a = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_mullo_epi32(iota, strideN_v));
                __m512i idx_b = _mm512_add_epi32(idx_a, step16);
                __m512i va = _mm512_i32gather_epi32(idx_a, (const void *)src, 2);
                __m512i vb = _mm512_i32gather_epi32(idx_b, (const void *)src, 2);
                va = _mm512_and_si512(va, _mm512_set1_epi32(0xFFFF));
                vb = _mm512_and_si512(vb, _mm512_set1_epi32(0xFFFF));
#if defined(__AVX512BW__)
                __m256i pa = _mm512_cvtepi32_epi16(va);
                __m256i pb = _mm512_cvtepi32_epi16(vb);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j), pa);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j + 16), pb);
#else
                __m256i va0 = _mm512_castsi512_si256(va), va1 = _mm512_extracti32x8_epi32(va, 1);
                __m256i vb0 = _mm512_castsi512_si256(vb), vb1 = _mm512_extracti32x8_epi32(vb, 1);
                __m128i lo0 = _mm256_castsi256_si128(va0), hi0 = _mm256_extracti128_si256(va0, 1);
                __m128i lo1 = _mm256_castsi256_si128(va1), hi1 = _mm256_extracti128_si256(va1, 1);
                __m128i p0 = _mm_packus_epi32(lo0, hi0);
                __m128i p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 8), p1);
                lo0 = _mm256_castsi256_si128(vb0); hi0 = _mm256_extracti128_si256(vb0, 1);
                lo1 = _mm256_castsi256_si128(vb1); hi1 = _mm256_extracti128_si256(vb1, 1);
                p0 = _mm_packus_epi32(lo0, hi0);
                p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 16), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 24), p1);
#endif
            }
            for (; j + 16 <= nc; j += 16) {
                int base = (int)(j * (ptrdiff_t)strideN);
                __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_mullo_epi32(iota, strideN_v));
                __m512i v = _mm512_i32gather_epi32(idx, (const void *)src, 2);
                v = _mm512_and_si512(v, _mm512_set1_epi32(0xFFFF));
#if defined(__AVX512BW__)
                __m256i packed = _mm512_cvtepi32_epi16(v);
                _mm256_storeu_si256((__m256i *)(void *)(dst + j), packed);
#else
                __m256i v0 = _mm512_castsi512_si256(v);
                __m256i v1 = _mm512_extracti32x8_epi32(v, 1);
                __m128i lo0 = _mm256_castsi256_si128(v0), hi0 = _mm256_extracti128_si256(v0, 1);
                __m128i lo1 = _mm256_castsi256_si128(v1), hi1 = _mm256_extracti128_si256(v1, 1);
                __m128i p0 = _mm_packus_epi32(lo0, hi0);
                __m128i p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 8), p1);
#endif
            }
            for (; j + 8 <= nc; j += 8) {
                __m256i idx = _mm256_set_epi32(
                    (int)((j + 7) * (ptrdiff_t)strideN), (int)((j + 6) * (ptrdiff_t)strideN),
                    (int)((j + 5) * (ptrdiff_t)strideN), (int)((j + 4) * (ptrdiff_t)strideN),
                    (int)((j + 3) * (ptrdiff_t)strideN), (int)((j + 2) * (ptrdiff_t)strideN),
                    (int)((j + 1) * (ptrdiff_t)strideN), (int)(j * (ptrdiff_t)strideN));
                __m256i v = _mm256_i32gather_epi32((const int *)(const void *)src, idx, 2);
                v = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                __m128i packed = _mm_packus_epi32(lo, hi);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), packed);
            }
            for (; j < nc; ++j)
                dst[j] = src[j * strideN];
        }
#elif defined(__AVX2__)
        for (int64_t k = 0; k < kc; ++k) {
            if (k + 1 < kc)
                mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1) * strideK));
            if (k + 2 < kc)
                mag_simd_prefetch_t1((const char *)(Bsrc + (k + 2) * strideK));
            const mag_bfloat16_t *src = Bsrc + k * strideK;
            mag_bfloat16_t *dst = Bp + k * nc;
            int64_t j = 0;
            for (; j + 32 <= nc; j += 32) {
                __m256i idx0 = _mm256_set_epi32(
                    (int)((j + 7) * (ptrdiff_t)strideN), (int)((j + 6) * (ptrdiff_t)strideN),
                    (int)((j + 5) * (ptrdiff_t)strideN), (int)((j + 4) * (ptrdiff_t)strideN),
                    (int)((j + 3) * (ptrdiff_t)strideN), (int)((j + 2) * (ptrdiff_t)strideN),
                    (int)((j + 1) * (ptrdiff_t)strideN), (int)(j * (ptrdiff_t)strideN));
                __m256i idx1 = _mm256_set_epi32(
                    (int)((j + 15) * (ptrdiff_t)strideN), (int)((j + 14) * (ptrdiff_t)strideN),
                    (int)((j + 13) * (ptrdiff_t)strideN), (int)((j + 12) * (ptrdiff_t)strideN),
                    (int)((j + 11) * (ptrdiff_t)strideN), (int)((j + 10) * (ptrdiff_t)strideN),
                    (int)((j + 9) * (ptrdiff_t)strideN),  (int)((j + 8) * (ptrdiff_t)strideN));
                __m256i v0 = _mm256_i32gather_epi32((const int *)(const void *)src, idx0, 2);
                __m256i v1 = _mm256_i32gather_epi32((const int *)(const void *)src, idx1, 2);
                v0 = _mm256_and_si256(v0, _mm256_set1_epi32(0xFFFF));
                v1 = _mm256_and_si256(v1, _mm256_set1_epi32(0xFFFF));
                __m128i lo0 = _mm256_castsi256_si128(v0), hi0 = _mm256_extracti128_si256(v0, 1);
                __m128i lo1 = _mm256_castsi256_si128(v1), hi1 = _mm256_extracti128_si256(v1, 1);
                __m128i p0 = _mm_packus_epi32(lo0, hi0);
                __m128i p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 8), p1);
                __m256i idx2 = _mm256_set_epi32(
                    (int)((j + 23) * (ptrdiff_t)strideN), (int)((j + 22) * (ptrdiff_t)strideN),
                    (int)((j + 21) * (ptrdiff_t)strideN), (int)((j + 20) * (ptrdiff_t)strideN),
                    (int)((j + 19) * (ptrdiff_t)strideN), (int)((j + 18) * (ptrdiff_t)strideN),
                    (int)((j + 17) * (ptrdiff_t)strideN), (int)((j + 16) * (ptrdiff_t)strideN));
                __m256i idx3 = _mm256_set_epi32(
                    (int)((j + 31) * (ptrdiff_t)strideN), (int)((j + 30) * (ptrdiff_t)strideN),
                    (int)((j + 29) * (ptrdiff_t)strideN), (int)((j + 28) * (ptrdiff_t)strideN),
                    (int)((j + 27) * (ptrdiff_t)strideN), (int)((j + 26) * (ptrdiff_t)strideN),
                    (int)((j + 25) * (ptrdiff_t)strideN), (int)((j + 24) * (ptrdiff_t)strideN));
                __m256i v2 = _mm256_i32gather_epi32((const int *)(const void *)src, idx2, 2);
                __m256i v3 = _mm256_i32gather_epi32((const int *)(const void *)src, idx3, 2);
                v2 = _mm256_and_si256(v2, _mm256_set1_epi32(0xFFFF));
                v3 = _mm256_and_si256(v3, _mm256_set1_epi32(0xFFFF));
                lo0 = _mm256_castsi256_si128(v2); hi0 = _mm256_extracti128_si256(v2, 1);
                lo1 = _mm256_castsi256_si128(v3); hi1 = _mm256_extracti128_si256(v3, 1);
                p0 = _mm_packus_epi32(lo0, hi0);
                p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 16), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 24), p1);
            }
            for (; j + 16 <= nc; j += 16) {
                __m256i idx0 = _mm256_set_epi32(
                    (int)((j + 7) * (ptrdiff_t)strideN), (int)((j + 6) * (ptrdiff_t)strideN),
                    (int)((j + 5) * (ptrdiff_t)strideN), (int)((j + 4) * (ptrdiff_t)strideN),
                    (int)((j + 3) * (ptrdiff_t)strideN), (int)((j + 2) * (ptrdiff_t)strideN),
                    (int)((j + 1) * (ptrdiff_t)strideN), (int)(j * (ptrdiff_t)strideN));
                __m256i idx1 = _mm256_set_epi32(
                    (int)((j + 15) * (ptrdiff_t)strideN), (int)((j + 14) * (ptrdiff_t)strideN),
                    (int)((j + 13) * (ptrdiff_t)strideN), (int)((j + 12) * (ptrdiff_t)strideN),
                    (int)((j + 11) * (ptrdiff_t)strideN), (int)((j + 10) * (ptrdiff_t)strideN),
                    (int)((j + 9) * (ptrdiff_t)strideN),  (int)((j + 8) * (ptrdiff_t)strideN));
                __m256i v0 = _mm256_i32gather_epi32((const int *)(const void *)src, idx0, 2);
                __m256i v1 = _mm256_i32gather_epi32((const int *)(const void *)src, idx1, 2);
                v0 = _mm256_and_si256(v0, _mm256_set1_epi32(0xFFFF));
                v1 = _mm256_and_si256(v1, _mm256_set1_epi32(0xFFFF));
                __m128i lo0 = _mm256_castsi256_si128(v0), hi0 = _mm256_extracti128_si256(v0, 1);
                __m128i lo1 = _mm256_castsi256_si128(v1), hi1 = _mm256_extracti128_si256(v1, 1);
                __m128i p0 = _mm_packus_epi32(lo0, hi0);
                __m128i p1 = _mm_packus_epi32(lo1, hi1);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), p0);
                _mm_storeu_si128((__m128i *)(void *)(dst + j + 8), p1);
            }
            for (; j + 8 <= nc; j += 8) {
                __m256i idx = _mm256_set_epi32(
                    (int)((j + 7) * (ptrdiff_t)strideN), (int)((j + 6) * (ptrdiff_t)strideN),
                    (int)((j + 5) * (ptrdiff_t)strideN), (int)((j + 4) * (ptrdiff_t)strideN),
                    (int)((j + 3) * (ptrdiff_t)strideN), (int)((j + 2) * (ptrdiff_t)strideN),
                    (int)((j + 1) * (ptrdiff_t)strideN), (int)(j * (ptrdiff_t)strideN));
                __m256i v = _mm256_i32gather_epi32((const int *)(const void *)src, idx, 2);
                v = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                __m128i packed = _mm_packus_epi32(lo, hi);
                _mm_storeu_si128((__m128i *)(void *)(dst + j), packed);
            }
            for (; j < nc; ++j)
                dst[j] = src[j * strideN];
        }
#elif (defined(__aarch64__) && defined(__ARM_NEON))
        /* ARM64 NEON: prefetch k+1/k+2; gather 8/16 elements per step, vectorized store. */
        for (int64_t k = 0; k < kc; ++k) {
            if (k + 1 < kc)
                __builtin_prefetch((const char *)(Bsrc + (k + 1) * strideK), 0, 3);
            if (k + 2 < kc)
                __builtin_prefetch((const char *)(Bsrc + (k + 2) * strideK), 0, 2);
            const mag_bfloat16_t *src = Bsrc + k * strideK;
            mag_bfloat16_t *dst = Bp + k * nc;
            int64_t j = 0;
            for (; j + 16 <= nc; j += 16) {
                const mag_bfloat16_t *p0 = src + (j + 0) * strideN;
                const mag_bfloat16_t *p8 = src + (j + 8) * strideN;
                uint16_t t0[8], t1[8];
                for (int i = 0; i < 8; ++i) {
                    t0[i] = p0[i * strideN].bits;
                    t1[i] = p8[i * strideN].bits;
                }
                vst1q_u16((uint16_t *)(dst + j),      vld1q_u16(t0));
                vst1q_u16((uint16_t *)(dst + j + 8), vld1q_u16(t1));
            }
            for (; j + 8 <= nc; j += 8) {
                const mag_bfloat16_t *p = src + j * strideN;
                uint16_t t[8];
                for (int i = 0; i < 8; ++i) t[i] = p[i * strideN].bits;
                vst1q_u16((uint16_t *)(dst + j), vld1q_u16(t));
            }
            for (; j < nc; ++j)
                dst[j] = src[j * strideN];
        }
#else
        for (int64_t k = 0; k < kc; ++k) {
            const mag_bfloat16_t *src = Bsrc + k * strideK;
            for (int64_t j = 0; j < nc; ++j) {
                Bp[k * nc + j] = src[j * strideN];
            }
        }
#endif
    }
}

static MAG_AINLINE void mag_mm_pack_B_vec_bfloat16(
    int64_t kc, int64_t nc,
    const mag_bfloat16_t *restrict yvec,
    mag_bfloat16_t *restrict Bp
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    for (int64_t k = 0; k < kc; ++k) {
        __m512bh val = (__m512bh)_mm512_set1_epi16((int)yvec[k].bits);
        mag_bfloat16_t *dst = Bp + k * nc;
        int64_t j = 0;
        for (; j+63 < nc; j += 64) {
            _mm512_storeu_si512((void*)(dst + j +  0), (__m512i)val);
            _mm512_storeu_si512((void*)(dst + j + 32), (__m512i)val);
        }
        for (; j+31 < nc; j += 32) {
            _mm512_storeu_si512((void*)(dst + j), (__m512i)val);
        }
        for (; j+15 < nc; j += 16) {
            _mm256_storeu_si256((__m256i*)(dst + j), _mm512_castsi512_si256((__m512i)val));
        }
        if (j < nc) {
            int64_t rem = nc - j;
            __mmask16 m = (rem == 16) ? (__mmask16)0xffff : (__mmask16)((1u << rem) - 1u);
            _mm256_mask_storeu_epi16((void*)(dst + j), m, _mm512_castsi512_si256((__m512i)val));
        }
    }

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
    for (int64_t k = 0; k < kc; ++k) {
        uint16_t u = yvec[k].bits;
        bfloat16x8_t val = vreinterpretq_bf16_u16(vdupq_n_u16(u));
        mag_bfloat16_t *dst = Bp + k * nc;
        int64_t j = 0;
        for (; j+31 < nc; j += 32) {
            vst1q_u16((uint16_t*)(dst + j +  0), vreinterpretq_u16_bf16(val));
            vst1q_u16((uint16_t*)(dst + j +  8), vreinterpretq_u16_bf16(val));
            vst1q_u16((uint16_t*)(dst + j + 16), vreinterpretq_u16_bf16(val));
            vst1q_u16((uint16_t*)(dst + j + 24), vreinterpretq_u16_bf16(val));
        }
        for (; j+15 < nc; j += 16) {
            vst1q_u16((uint16_t*)(dst + j + 0), vreinterpretq_u16_bf16(val));
            vst1q_u16((uint16_t*)(dst + j + 8), vreinterpretq_u16_bf16(val));
        }
        for (; j+7 < nc; j += 8) {
            vst1q_u16((uint16_t*)(dst + j), vreinterpretq_u16_bf16(val));
        }
        for (; j < nc; ++j) {
            dst[j] = yvec[k];
        }
    }

#else
    for (int64_t k = 0; k < kc; ++k) {
        mag_bfloat16_t v = yvec[k];
        for (int64_t j = 0; j < nc; ++j)
            Bp[k*nc + j] = v;
    }
#endif
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_float32(int64_t kc, int64_t mr, const float *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx, float *restrict pa) {
    int64_t m8 = mr&~7;
    for (int64_t i=0; i < m8; i += 8) {
        const float *p0 = ra + (i+0)*sMx;
        const float *p1 = ra + (i+1)*sMx;
        const float *p2 = ra + (i+2)*sMx;
        const float *p3 = ra + (i+3)*sMx;
        const float *p4 = ra + (i+4)*sMx;
        const float *p5 = ra + (i+5)*sMx;
        const float *p6 = ra + (i+6)*sMx;
        const float *p7 = ra + (i+7)*sMx;
        float *dst = pa + i*kc;
        int64_t k = 0;
        for (; k+1 < kc; k += 2) {
            if ((k & ((MAG_PREFETCH_GROUPS<<1)-1)) == 0) {
                mag_simd_prefetch_t0(p0 + (int64_t)MAG_PREFETCH_RANGE_A_L1*sKx);
                mag_simd_prefetch_t0(p4 + (int64_t)MAG_PREFETCH_RANGE_A_L1*sKx);
                mag_simd_prefetch_t1(p0 + (int64_t)MAG_PREFETCH_RANGE_A_L2*sKx);
                mag_simd_prefetch_t1(p4 + (int64_t)MAG_PREFETCH_RANGE_A_L2*sKx);
            }
            float s00 = p0[0];
            float s10 = p1[0];
            float s20 = p2[0];
            float s30 = p3[0];
            float s40 = p4[0];
            float s50 = p5[0];
            float s60 = p6[0];
            float s70 = p7[0];
            p0 += sKx;
            p1 += sKx;
            p2 += sKx;
            p3 += sKx;
            p4 += sKx;
            p5 += sKx;
            p6 += sKx;
            p7 += sKx;
            float s01 = p0[0];
            float s11 = p1[0];
            float s21 = p2[0];
            float s31 = p3[0];
            float s41 = p4[0];
            float s51 = p5[0];
            float s61 = p6[0];
            float s71 = p7[0];
            p0 += sKx;
            p1 += sKx;
            p2 += sKx;
            p3 += sKx;
            p4 += sKx;
            p5 += sKx;
            p6 += sKx;
            p7 += sKx;
#if defined(__AVX512F__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            __m256 v1 = _mm256_setr_ps(s01,s11,s21,s31,s41,s51,s61,s71);
            __m512 vv = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v1, 1);
            _mm512_storeu_ps(dst + k*8, vv);
#elif defined(__AVX2__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            __m256 v1 = _mm256_setr_ps(s01,s11,s21,s31,s41,s51,s61,s71);
            _mm256_storeu_ps(dst + (k+0)*8, v0);
            _mm256_storeu_ps(dst + (k+1)*8, v1);
#elif defined(__SSE4_1__) || defined(__SSE2__)
            __m128 v00 = _mm_setr_ps(s00,s10,s20,s30);
            __m128 v01 = _mm_setr_ps(s40,s50,s60,s70);
            __m128 v10 = _mm_setr_ps(s01,s11,s21,s31);
            __m128 v11 = _mm_setr_ps(s41,s51,s61,s71);
            _mm_storeu_ps(dst + (k+0)*8 + 0, v00);
            _mm_storeu_ps(dst + (k+0)*8 + 4, v01);
            _mm_storeu_ps(dst + (k+1)*8 + 0, v10);
            _mm_storeu_ps(dst + (k+1)*8 + 4, v11);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            float32x4_t v00 = vdupq_n_f32(0.f);
            v00 = vsetq_lane_f32(s00, v00, 0);
            v00 = vsetq_lane_f32(s10, v00, 1);
            v00 = vsetq_lane_f32(s20, v00, 2);
            v00 = vsetq_lane_f32(s30, v00, 3);
            float32x4_t v01 = vdupq_n_f32(0.f);
            v01 = vsetq_lane_f32(s40, v01, 0);
            v01 = vsetq_lane_f32(s50, v01, 1);
            v01 = vsetq_lane_f32(s60, v01, 2);
            v01 = vsetq_lane_f32(s70, v01, 3);
            float32x4_t v10 = vdupq_n_f32(0.f);
            v10 = vsetq_lane_f32(s01, v10, 0);
            v10 = vsetq_lane_f32(s11, v10, 1);
            v10 = vsetq_lane_f32(s21, v10, 2);
            v10 = vsetq_lane_f32(s31, v10, 3);
            float32x4_t v11 = vdupq_n_f32(0.f);
            v11 = vsetq_lane_f32(s41, v11, 0);
            v11 = vsetq_lane_f32(s51, v11, 1);
            v11 = vsetq_lane_f32(s61, v11, 2);
            v11 = vsetq_lane_f32(s71, v11, 3);
            vst1q_f32(dst + (k+0)*8 + 0, v00);
            vst1q_f32(dst + (k+0)*8 + 4, v01);
            vst1q_f32(dst + (k+1)*8 + 0, v10);
            vst1q_f32(dst + (k+1)*8 + 4, v11);

#else
            float *d0 = dst + (k+0)*8;
            float *d1 = dst + (k+1)*8;
            d0[0]=s00;
            d0[1]=s10;
            d0[2]=s20;
            d0[3]=s30;
            d0[4]=s40;
            d0[5]=s50;
            d0[6]=s60;
            d0[7]=s70;
            d1[0]=s01;
            d1[1]=s11;
            d1[2]=s21;
            d1[3]=s31;
            d1[4]=s41;
            d1[5]=s51;
            d1[6]=s61;
            d1[7]=s71;
#endif
        }
        if (k < kc) {
            float s00 = p0[0];
            float s10 = p1[0];
            float s20 = p2[0];
            float s30 = p3[0];
            float s40 = p4[0];
            float s50 = p5[0];
            float s60 = p6[0];
            float s70 = p7[0];
#if defined(__AVX512F__) || defined(__AVX2__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            _mm256_storeu_ps(dst + k*8, v0);
#elif defined(__SSE4_1__) || defined(__SSE2__)
            __m128 v00 = _mm_setr_ps(s00,s10,s20,s30);
            __m128 v01 = _mm_setr_ps(s40,s50,s60,s70);
            _mm_storeu_ps(dst + k*8 + 0, v00);
            _mm_storeu_ps(dst + k*8 + 4, v01);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            float32x4_t v00 = vdupq_n_f32(0.f);
            v00 = vsetq_lane_f32(s00, v00, 0);
            v00 = vsetq_lane_f32(s10, v00, 1);
            v00 = vsetq_lane_f32(s20, v00, 2);
            v00 = vsetq_lane_f32(s30, v00, 3);
            float32x4_t v01 = vdupq_n_f32(0.f);
            v01 = vsetq_lane_f32(s40, v01, 0);
            v01 = vsetq_lane_f32(s50, v01, 1);
            v01 = vsetq_lane_f32(s60, v01, 2);
            v01 = vsetq_lane_f32(s70, v01, 3);
            vst1q_f32(dst + k*8 + 0, v00);
            vst1q_f32(dst + k*8 + 4, v01);
#else
            float *d0 = dst + k*8;
            d0[0]=s00;
            d0[1]=s10;
            d0[2]=s20;
            d0[3]=s30;
            d0[4]=s40;
            d0[5]=s50;
            d0[6]=s60;
            d0[7]=s70;
#endif
        }
    }
    for (int64_t i=m8; i < mr; ++i) {
        const float *src = ra + i*sMx;
        float *dst = pa + i*kc;
#if defined(__AVX512F__)
        int64_t k = 0;
        for (; k+15 < kc; k += 16) {
            mag_simd_prefetch_t0(src + (k + MAG_PREFETCH_RANGE_A_L1)*sKx);
            mag_simd_prefetch_t1(src + (k + MAG_PREFETCH_RANGE_A_L2)*sKx);
            __m512 v = _mm512_set_ps(
                           src[(k+15)*sKx], src[(k+14)*sKx], src[(k+13)*sKx], src[(k+12)*sKx],
                           src[(k+11)*sKx], src[(k+10)*sKx], src[(k+9)*sKx], src[(k+8)*sKx],
                           src[(k+7)*sKx], src[(k+6)*sKx], src[(k+5)*sKx], src[(k+4)*sKx],
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm512_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif defined(__AVX2__)
        int64_t k=0;
        for (; k+7 < kc; k += 8) {
            __m256 v = _mm256_set_ps(
                           src[(k+7)*sKx], src[(k+6)*sKx], src[(k+5)*sKx], src[(k+4)*sKx],
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm256_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif defined(__SSE4_1__) || defined(__SSE2__)
        int64_t k = 0;
        for (; k+3 < kc; k += 4) {
            __m128 v = _mm_set_ps(
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        int64_t k=0;
        for (; k+3 < kc; k += 4) {
            float32x4_t v;
            v = vsetq_lane_f32(src[(k+0)*sKx], vdupq_n_f32(0.f), 0);
            v = vsetq_lane_f32(src[(k+1)*sKx], v, 1);
            v = vsetq_lane_f32(src[(k+2)*sKx], v, 2);
            v = vsetq_lane_f32(src[(k+3)*sKx], v, 3);
            vst1q_f32(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#else
        for (int64_t k=0; k < kc; ++k) dst[k] = src[k*sKx];
#endif
    }
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_bfloat16(
    int64_t kc, int64_t mr,
    const mag_bfloat16_t *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx,
    mag_bfloat16_t *restrict pa
) {
    int64_t m8 = mr & ~7;
    for (int64_t i = 0; i < m8; i += 8) {
        const mag_bfloat16_t *p0 = ra + (i + 0) * sMx;
        const mag_bfloat16_t *p1 = ra + (i + 1) * sMx;
        const mag_bfloat16_t *p2 = ra + (i + 2) * sMx;
        const mag_bfloat16_t *p3 = ra + (i + 3) * sMx;
        const mag_bfloat16_t *p4 = ra + (i + 4) * sMx;
        const mag_bfloat16_t *p5 = ra + (i + 5) * sMx;
        const mag_bfloat16_t *p6 = ra + (i + 6) * sMx;
        const mag_bfloat16_t *p7 = ra + (i + 7) * sMx;
        mag_bfloat16_t *dst = pa + i * kc;
        int64_t k = 0;
        for (; k+1 < kc; k += 2) {
            if ((k & ((MAG_PREFETCH_GROUPS << 1) - 1)) == 0) {
                mag_simd_prefetch_t0(p0 + (int64_t)MAG_PREFETCH_RANGE_A_L1 * sKx);
                mag_simd_prefetch_t0(p4 + (int64_t)MAG_PREFETCH_RANGE_A_L1 * sKx);
                mag_simd_prefetch_t1(p0 + (int64_t)MAG_PREFETCH_RANGE_A_L2 * sKx);
                mag_simd_prefetch_t1(p4 + (int64_t)MAG_PREFETCH_RANGE_A_L2 * sKx);
            }
            uint16_t s00 = p0[0].bits, s10 = p1[0].bits, s20 = p2[0].bits, s30 = p3[0].bits;
            uint16_t s40 = p4[0].bits, s50 = p5[0].bits, s60 = p6[0].bits, s70 = p7[0].bits;
            p0 += sKx; p1 += sKx; p2 += sKx; p3 += sKx;
            p4 += sKx; p5 += sKx; p6 += sKx; p7 += sKx;
            uint16_t s01 = p0[0].bits, s11 = p1[0].bits, s21 = p2[0].bits, s31 = p3[0].bits;
            uint16_t s41 = p4[0].bits, s51 = p5[0].bits, s61 = p6[0].bits, s71 = p7[0].bits;
            p0 += sKx; p1 += sKx; p2 += sKx; p3 += sKx;
            p4 += sKx; p5 += sKx; p6 += sKx; p7 += sKx;
#if defined(__AVX512F__) && defined(__AVX512BF16__)
            __m256i v16 = _mm256_setr_epi16(
                (int)s00,(int)s10,(int)s20,(int)s30,(int)s40,(int)s50,(int)s60,(int)s70,
                (int)s01,(int)s11,(int)s21,(int)s31,(int)s41,(int)s51,(int)s61,(int)s71
            );
            _mm256_storeu_si256((__m256i*)(dst + k * 8), v16);

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
            uint16x8_t v0 = (uint16x8_t){ s00,s10,s20,s30,s40,s50,s60,s70 };
            uint16x8_t v1 = (uint16x8_t){ s01,s11,s21,s31,s41,s51,s61,s71 };
            vst1q_u16((uint16_t*)(dst + (k + 0) * 8), v0);
            vst1q_u16((uint16_t*)(dst + (k + 1) * 8), v1);

#else
            mag_bfloat16_t *d0 = dst + (k + 0) * 8;
            mag_bfloat16_t *d1 = dst + (k + 1) * 8;
            d0[0].bits = s00; d0[1].bits = s10; d0[2].bits = s20; d0[3].bits = s30;
            d0[4].bits = s40; d0[5].bits = s50; d0[6].bits = s60; d0[7].bits = s70;
            d1[0].bits = s01; d1[1].bits = s11; d1[2].bits = s21; d1[3].bits = s31;
            d1[4].bits = s41; d1[5].bits = s51; d1[6].bits = s61; d1[7].bits = s71;
#endif
        }

        if (k < kc) {
            uint16_t s00 = p0[0].bits, s10 = p1[0].bits, s20 = p2[0].bits, s30 = p3[0].bits;
            uint16_t s40 = p4[0].bits, s50 = p5[0].bits, s60 = p6[0].bits, s70 = p7[0].bits;

#if defined(__AVX512F__) && defined(__AVX512BF16__)
            __m128i v8 = _mm_setr_epi16(
                (int)s00,(int)s10,(int)s20,(int)s30,(int)s40,(int)s50,(int)s60,(int)s70
            );
            _mm_storeu_si128((__m128i*)(dst + k * 8), v8);

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
            uint16x8_t v0 = (uint16x8_t){ s00,s10,s20,s30,s40,s50,s60,s70 };
            vst1q_u16((uint16_t*)(dst + k * 8), v0);

#else
            mag_bfloat16_t *d0 = dst + k * 8;
            d0[0].bits = s00; d0[1].bits = s10; d0[2].bits = s20; d0[3].bits = s30;
            d0[4].bits = s40; d0[5].bits = s50; d0[6].bits = s60; d0[7].bits = s70;
#endif
        }
    }
    for (int64_t i = m8; i < mr; ++i) {
        const mag_bfloat16_t *src = ra + i * sMx;
        mag_bfloat16_t *dst = pa + i * kc;

#if defined(__AVX512F__) && defined(__AVX512BF16__)
        int64_t k = 0;
        for (; k + 31 < kc; k += 32) {
            mag_simd_prefetch_t0(src + (k + MAG_PREFETCH_RANGE_A_L1) * sKx);
            mag_simd_prefetch_t1(src + (k + MAG_PREFETCH_RANGE_A_L2) * sKx);

            /* gather via scalar-lane load into vectors (stride may be != 1) */
            __m256i v0 = _mm256_set_epi16(
                (int)src[(k+15)*sKx].bits,(int)src[(k+14)*sKx].bits,(int)src[(k+13)*sKx].bits,(int)src[(k+12)*sKx].bits,
                (int)src[(k+11)*sKx].bits,(int)src[(k+10)*sKx].bits,(int)src[(k+ 9)*sKx].bits,(int)src[(k+ 8)*sKx].bits,
                (int)src[(k+ 7)*sKx].bits,(int)src[(k+ 6)*sKx].bits,(int)src[(k+ 5)*sKx].bits,(int)src[(k+ 4)*sKx].bits,
                (int)src[(k+ 3)*sKx].bits,(int)src[(k+ 2)*sKx].bits,(int)src[(k+ 1)*sKx].bits,(int)src[(k+ 0)*sKx].bits
            );
            __m256i v1 = _mm256_set_epi16(
                (int)src[(k+31)*sKx].bits,(int)src[(k+30)*sKx].bits,(int)src[(k+29)*sKx].bits,(int)src[(k+28)*sKx].bits,
                (int)src[(k+27)*sKx].bits,(int)src[(k+26)*sKx].bits,(int)src[(k+25)*sKx].bits,(int)src[(k+24)*sKx].bits,
                (int)src[(k+23)*sKx].bits,(int)src[(k+22)*sKx].bits,(int)src[(k+21)*sKx].bits,(int)src[(k+20)*sKx].bits,
                (int)src[(k+19)*sKx].bits,(int)src[(k+18)*sKx].bits,(int)src[(k+17)*sKx].bits,(int)src[(k+16)*sKx].bits
            );
            _mm256_storeu_si256((__m256i*)(dst + k +  0), v0);
            _mm256_storeu_si256((__m256i*)(dst + k + 16), v1);
        }
        for (; k + 15 < kc; k += 16) {
            mag_simd_prefetch_t0(src + (k + MAG_PREFETCH_RANGE_A_L1) * sKx);
            mag_simd_prefetch_t1(src + (k + MAG_PREFETCH_RANGE_A_L2) * sKx);

            __m256i v = _mm256_set_epi16(
                (int)src[(k+15)*sKx].bits,(int)src[(k+14)*sKx].bits,(int)src[(k+13)*sKx].bits,(int)src[(k+12)*sKx].bits,
                (int)src[(k+11)*sKx].bits,(int)src[(k+10)*sKx].bits,(int)src[(k+ 9)*sKx].bits,(int)src[(k+ 8)*sKx].bits,
                (int)src[(k+ 7)*sKx].bits,(int)src[(k+ 6)*sKx].bits,(int)src[(k+ 5)*sKx].bits,(int)src[(k+ 4)*sKx].bits,
                (int)src[(k+ 3)*sKx].bits,(int)src[(k+ 2)*sKx].bits,(int)src[(k+ 1)*sKx].bits,(int)src[(k+ 0)*sKx].bits
            );
            _mm256_storeu_si256((__m256i*)(dst + k), v);
        }
        for (; k < kc; ++k) dst[k] = src[k * sKx];

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
        int64_t k = 0;
        for (; k + 15 < kc; k += 16) {
            mag_simd_prefetch_t0(src + (k + MAG_PREFETCH_RANGE_A_L1) * sKx);
            mag_simd_prefetch_t1(src + (k + MAG_PREFETCH_RANGE_A_L2) * sKx);
            uint16x8_t v0 = (uint16x8_t){
                src[(k+0)*sKx].bits, src[(k+1)*sKx].bits, src[(k+2)*sKx].bits, src[(k+3)*sKx].bits,
                src[(k+4)*sKx].bits, src[(k+5)*sKx].bits, src[(k+6)*sKx].bits, src[(k+7)*sKx].bits
            };
            uint16x8_t v1 = (uint16x8_t){
                src[(k+8)*sKx].bits, src[(k+9)*sKx].bits, src[(k+10)*sKx].bits, src[(k+11)*sKx].bits,
                src[(k+12)*sKx].bits, src[(k+13)*sKx].bits, src[(k+14)*sKx].bits, src[(k+15)*sKx].bits
            };
            vst1q_u16((uint16_t*)(dst + k + 0), v0);
            vst1q_u16((uint16_t*)(dst + k + 8), v1);
        }
        for (; k + 7 < kc; k += 8) {
            uint16x8_t v0 = (uint16x8_t){
                src[(k+0)*sKx].bits, src[(k+1)*sKx].bits, src[(k+2)*sKx].bits, src[(k+3)*sKx].bits,
                src[(k+4)*sKx].bits, src[(k+5)*sKx].bits, src[(k+6)*sKx].bits, src[(k+7)*sKx].bits
            };
            vst1q_u16((uint16_t*)(dst + k), v0);
        }
        for (; k < kc; ++k) dst[k] = src[k * sKx];

#else
        for (int64_t k = 0; k < kc; ++k) dst[k] = src[k * sKx];
#endif
    }
}

static MAG_AINLINE void mag_gemv_bfloat16(
    int64_t K, int64_t N,
    const mag_bfloat16_t *restrict A,
    const mag_bfloat16_t *restrict B,
    int64_t ldb,
    mag_bfloat16_t *restrict C
) {
#if defined(__AVX512F__) && defined(__AVX512BF16__)
    int64_t j = 0;
    for (; j + 127 < N; j += 128) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        __m512 s7 = _mm512_setzero_ps();
        const mag_bfloat16_t *restrict brow = B + j;
        int64_t k = 0;
        for (; k + 1 < K; k += 2, brow += 2 * ldb) {
            uint16_t a0 = A[k + 0].bits;
            uint16_t a1 = A[k + 1].bits;
            uint32_t apair = (uint32_t)a0 | ((uint32_t)a1 << 16);
            __m512bh avec = (__m512bh)_mm512_set1_epi32((int)apair);

#define DP_STEP(acc, off) do { \
                const mag_bfloat16_t *r0 = brow + (k + 0 - k) * ldb + (off); \
                const mag_bfloat16_t *r1 = brow + (k + 1 - k) * ldb + (off); \
                __m256i b0 = _mm256_loadu_si256((const __m256i*)r0); \
                __m256i b1 = _mm256_loadu_si256((const __m256i*)r1); \
                __m256i lo = _mm256_unpacklo_epi16(b0, b1); \
                __m256i hi = _mm256_unpackhi_epi16(b0, b1); \
                __m512i  bi = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); \
                __m512bh bvec = (__m512bh)bi; \
                acc = _mm512_dpbf16_ps(acc, avec, bvec); \
            } while (0)

            DP_STEP(s0,   0);
            DP_STEP(s1,  16);
            DP_STEP(s2,  32);
            DP_STEP(s3,  48);
            DP_STEP(s4,  64);
            DP_STEP(s5,  80);
            DP_STEP(s6,  96);
            DP_STEP(s7, 112);

#undef DP_STEP
        }
        if (k < K) {
            float a = mag_bfloat16_to_float32(A[k]);
            __m512 av = _mm512_set1_ps(a);
#define FMA_TAIL(acc, off) do { \
                const mag_bfloat16_t *rp = brow + (k - k) * ldb + (off); \
                __m256i b16 = _mm256_loadu_si256((const __m256i*)rp); \
                __m256bh bb = (__m256bh)b16; \
                __m512 bv = _mm512_cvtpbh_ps(bb); \
                acc = _mm512_fmadd_ps(av, bv, acc); \
            } while (0)
            FMA_TAIL(s0,   0);
            FMA_TAIL(s1,  16);
            FMA_TAIL(s2,  32);
            FMA_TAIL(s3,  48);
            FMA_TAIL(s4,  64);
            FMA_TAIL(s5,  80);
            FMA_TAIL(s6,  96);
            FMA_TAIL(s7, 112);
#undef FMA_TAIL
        }
        _mm256_storeu_si256((__m256i*)(C + j +   0), (__m256i)_mm512_cvtneps_pbh(s0));
        _mm256_storeu_si256((__m256i*)(C + j +  16), (__m256i)_mm512_cvtneps_pbh(s1));
        _mm256_storeu_si256((__m256i*)(C + j +  32), (__m256i)_mm512_cvtneps_pbh(s2));
        _mm256_storeu_si256((__m256i*)(C + j +  48), (__m256i)_mm512_cvtneps_pbh(s3));
        _mm256_storeu_si256((__m256i*)(C + j +  64), (__m256i)_mm512_cvtneps_pbh(s4));
        _mm256_storeu_si256((__m256i*)(C + j +  80), (__m256i)_mm512_cvtneps_pbh(s5));
        _mm256_storeu_si256((__m256i*)(C + j +  96), (__m256i)_mm512_cvtneps_pbh(s6));
        _mm256_storeu_si256((__m256i*)(C + j + 112), (__m256i)_mm512_cvtneps_pbh(s7));
    }
    for (; j+63 < N; j += 64) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        const mag_bfloat16_t *restrict brow = B + j;
        int64_t k = 0;
        for (; k + 1 < K; k += 2, brow += 2 * ldb) {
            uint16_t a0 = A[k + 0].bits;
            uint16_t a1 = A[k + 1].bits;
            uint32_t apair = (uint32_t)a0 | ((uint32_t)a1 << 16);
            __m512bh avec = (__m512bh)_mm512_set1_epi32((int)apair);
#define DP_STEP(acc, off) do { \
                const mag_bfloat16_t *r0 = brow + (off); \
                const mag_bfloat16_t *r1 = brow + ldb + (off); \
                __m256i b0 = _mm256_loadu_si256((const __m256i*)r0); \
                __m256i b1 = _mm256_loadu_si256((const __m256i*)r1); \
                __m256i lo = _mm256_unpacklo_epi16(b0, b1); \
                __m256i hi = _mm256_unpackhi_epi16(b0, b1); \
                __m512i  bi = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); \
                __m512bh bvec = (__m512bh)bi; \
                acc = _mm512_dpbf16_ps(acc, avec, bvec); \
            } while (0)
            DP_STEP(s0,  0);
            DP_STEP(s1, 16);
            DP_STEP(s2, 32);
            DP_STEP(s3, 48);
#undef DP_STEP
        }
        if (k < K) {
            float a = mag_bfloat16_to_float32(A[k]);
            __m512 av = _mm512_set1_ps(a);
#define FMA_TAIL(acc, off) do { \
                const mag_bfloat16_t *rp = brow + (off); \
                __m256i b16 = _mm256_loadu_si256((const __m256i*)rp); \
                __m256bh bb = (__m256bh)b16; \
                __m512 bv = _mm512_cvtpbh_ps(bb); \
                acc = _mm512_fmadd_ps(av, bv, acc); \
            } while (0)
            FMA_TAIL(s0,  0);
            FMA_TAIL(s1, 16);
            FMA_TAIL(s2, 32);
            FMA_TAIL(s3, 48);
#undef FMA_TAIL
        }
        _mm256_storeu_si256((__m256i*)(C + j +  0), (__m256i)_mm512_cvtneps_pbh(s0));
        _mm256_storeu_si256((__m256i*)(C + j + 16), (__m256i)_mm512_cvtneps_pbh(s1));
        _mm256_storeu_si256((__m256i*)(C + j + 32), (__m256i)_mm512_cvtneps_pbh(s2));
        _mm256_storeu_si256((__m256i*)(C + j + 48), (__m256i)_mm512_cvtneps_pbh(s3));
    }
    for (; j+15 < N; j += 16) {
        __m512 s = _mm512_setzero_ps();
        const mag_bfloat16_t *restrict brow = B + j;
        int64_t k = 0;
        for (; k + 1 < K; k += 2, brow += 2 * ldb) {
            uint16_t a0 = A[k + 0].bits;
            uint16_t a1 = A[k + 1].bits;
            uint32_t apair = (uint32_t)a0 | ((uint32_t)a1 << 16);
            __m512bh avec = (__m512bh)_mm512_set1_epi32((int)apair);
            const mag_bfloat16_t *r0 = brow;
            const mag_bfloat16_t *r1 = brow + ldb;
            __m256i b0 = _mm256_loadu_si256((const __m256i*)r0);
            __m256i b1 = _mm256_loadu_si256((const __m256i*)r1);
            __m256i lo = _mm256_unpacklo_epi16(b0, b1);
            __m256i hi = _mm256_unpackhi_epi16(b0, b1);
            __m512i bi = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            __m512bh bvec = (__m512bh)bi;
            s = _mm512_dpbf16_ps(s, avec, bvec);
        }
        if (k < K) {
            float a = mag_bfloat16_to_float32(A[k]);
            __m512 av = _mm512_set1_ps(a);
            __m256i b16 = _mm256_loadu_si256((const __m256i*)brow);
            __m256bh bb = (__m256bh)b16;
            __m512 bv = _mm512_cvtpbh_ps(bb);
            s = _mm512_fmadd_ps(av, bv, s);
        }
        _mm256_storeu_si256((__m256i*)(C + j), (__m256i)_mm512_cvtneps_pbh(s));
    }
    for (; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float a = mag_bfloat16_to_float32(A[k]);
            float b = mag_bfloat16_to_float32(B[k * ldb + j]);
            sum += a * b;
        }
        C[j] = mag_float32_to_bfloat16(sum);
    }

#elif (defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16))
    int64_t j = 0;
    for (; j + 7 < N; j += 8) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        const mag_bfloat16_t *restrict brow = B + j;
        for (int64_t k = 0; k < K; ++k, brow += ldb) {
            uint16_t ab = A[k].bits;
            uint16x4_t au16 = vdup_n_u16(ab);
            bfloat16x4_t a4  = vreinterpret_bf16_u16(au16);
            bfloat16x8_t a8  = vcombine_bf16(a4, a4);
            bfloat16x8_t b8 = vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)brow));
            acc0 = vbfmlalbq_f32(acc0, a8, b8);
            acc1 = vbfmlaltq_f32(acc1, a8, b8);
        }
        bfloat16x8_t out = vcombine_bf16(vcvt_bf16_f32(acc0), vcvt_bf16_f32(acc1));
        vst1q_u16((uint16_t*)(C + j), vreinterpretq_u16_bf16(out));
    }
    for (; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float a = mag_bfloat16_to_float32(A[k]);
            float b = mag_bfloat16_to_float32(B[k * ldb + j]);
            sum += a * b;
        }
        C[j] = mag_float32_to_bfloat16(sum);
    }

#else
    for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float a = mag_bfloat16_to_float32(A[k]);
            float b = mag_bfloat16_to_float32(B[k * ldb + j]);
            sum += a * b;
        }
        C[j] = mag_float32_to_bfloat16(sum);
    }
#endif
}

static MAG_HOTPROC void mag_mm_block_bfloat16(
    int64_t kc, int64_t mr, int64_t nr,
    const mag_bfloat16_t *A, int64_t lda,
    const mag_bfloat16_t *B, int64_t ldb,
    mag_bfloat16_t *C, int64_t ldc,
    bool acc
) {
    int64_t j = 0;
    for (; nr - j >= 32; j += 32) {
        int64_t i = 0;
        for (; mr - i >= 16; i += 16) mag_mm_tile_16x32_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; mr - i >=  8; i +=  8) mag_mm_tile_8x32_bfloat16 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x32_bfloat16 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr - j >= 16; j += 16) {
        int64_t i = 0;
        for (; mr - i >= 8; i += 8) mag_mm_tile_8x16_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x16_bfloat16(kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr - j >= 8; j += 8) {
        int64_t i = 0;
        for (; mr - i >= 8; i += 8) mag_mm_tile_8x8_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x8_bfloat16(kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    int64_t rem = nr - j;
    if (!rem) return;
    for (int64_t i2 = 0; i2 < mr; ++i2) {
        const mag_bfloat16_t *ap = A + i2 * lda;
        mag_bfloat16_t *cp = C + i2*ldc + j;
        for (int64_t jj = 0; jj < rem; ++jj) {
            float sum = 0.0f;
            if (acc) sum = mag_bfloat16_to_float32(cp[jj]);
            for (int64_t k = 0; k < kc; ++k) {
                float ax  = mag_bfloat16_to_float32(ap[k]);
                float byv = mag_bfloat16_to_float32(B[k*ldb + (j + jj)]);
                sum += ax*byv;
            }
            cp[jj] = mag_float32_to_bfloat16(sum);
        }
    }
}

MAG_HOTPROC static void mag_matmul_bfloat16(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    const mag_tensor_t *y = mag_cmd_in(1);
    const mag_bfloat16_t *bx = (const mag_bfloat16_t *)mag_tensor_data_ptr(x);
    const mag_bfloat16_t *by = (const mag_bfloat16_t *)mag_tensor_data_ptr(y);
    mag_bfloat16_t *br = (mag_bfloat16_t *)mag_tensor_data_ptr_mut(r);
    int64_t MR = payload->mm_params.MR;
    int64_t MC = payload->mm_params.MC;
    int64_t KC = payload->mm_params.KC;
    int64_t NC = payload->mm_params.NC;
    int64_t NR = payload->mm_params.NR;
    int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
    int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
    int64_t K = x->coords.shape[x->coords.rank-1];
    int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d)
        batch_total *= r->coords.shape[d];
    if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { /* Detect GEMV */
        int64_t nth = payload->thread_num;
        int64_t tid = payload->thread_idx;
        int64_t j_per_thread = (N + nth - 1) / nth;
        int64_t j0 = tid*j_per_thread;
        int64_t j1 = mag_xmin(N, j0 + j_per_thread);
        for (int64_t batch = 0; batch < batch_total; ++batch) {
            const mag_bfloat16_t *A = bx + mag_offset_rmn(x, batch, 0, 0);
            const mag_bfloat16_t *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
            mag_bfloat16_t *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
            mag_gemv_bfloat16(K, j1 - j0, A, B, N, C);
        }
        return;
    }
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t tic = (M+MC-1)/MC;
    int64_t tjc = (N+NC-1)/NC;
    int64_t tpb = tic*tjc;
    int64_t tt = batch_total*tpb;
    mag_scratch_arena_clear(&mag_tls_arena);
    mag_bfloat16_t *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));
    mag_bfloat16_t *Bp = scratch;
    mag_bfloat16_t *Ap = Bp + KC*NC;
    for (;;) {
        int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
        if (tile >= tt) break;
        int64_t batch_idx = tile / tpb;
        int64_t rem = tile % tpb;
        int64_t jc = rem % tjc;
        int64_t ic = rem / tjc;
        int64_t idx_r[MAG_MAX_DIMS] = {0};
        for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
            idx_r[d] = t % r->coords.shape[d];
            t /= r->coords.shape[d];
        }
        int64_t xb_flat = 0;
        for (int64_t d=0; d < bdx; ++d) {
            int64_t rd = bdr - bdx + d;
            xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        int64_t yb_flat = 0;
        for (int64_t d=0; d < bdy; ++d) {
            int64_t rd = bdr - bdy + d;
            yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        bool yv = y->coords.rank == 1;
        const mag_bfloat16_t *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        const mag_bfloat16_t *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
        mag_bfloat16_t *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
        int64_t i0 = ic*MC;
        int64_t mc = i0+MC <= M ? MC : M-i0;
        int64_t j0 = jc*NC;
        int64_t nc = j0+NC <= N ? NC : N-j0;
        int64_t sMx = x->coords.strides[x->coords.rank-2];
        int64_t sKx = x->coords.strides[x->coords.rank-1];
        int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
        int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = mag_xmin(KC, K - pc);
            if (y->coords.rank == 1) mag_mm_pack_B_vec_bfloat16(kc, nc, py_base + pc, Bp);
            else mag_mm_pack_B_kc_nc_bfloat16(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
            mag_mm_pack_A_mc_kc_panel8_bfloat16(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
            for (int64_t ir=0; ir < mc; ir += MR)
                for (int64_t jr=0; jr < nc; jr += NR)
                    mag_mm_block_bfloat16(
                        kc,
                        mag_xmin(MR, mc - ir),
                        mag_xmin(NR, nc - jr),
                        Ap + ir*kc,
                        kc,
                        Bp + jr,
                        nc,
                        pr_base + (i0 + ir)*N + (j0 + jr),
                        N,
                        pc);
        }
    }
    mag_scratch_arena_clear(&mag_tls_arena);
}

static MAG_AINLINE float mag_load_x_f16_as_f32(
    const mag_tensor_t *x, const mag_float16_t *bx,
    int64_t xb_flat, int64_t i, int64_t k
) {
    if (x->coords.rank == 1) {
        int64_t off = mag_offset_rmn(x, k, 0, 0);
        return mag_float16_to_float32(bx[off]);
    } else {
        int64_t off = mag_offset_rmn(x, xb_flat, i, k);
        return mag_float16_to_float32(bx[off]);
    }
}

static MAG_AINLINE float mag_load_y_f16_as_f32(
    const mag_tensor_t *y, const mag_float16_t *by,
    int64_t yb_flat, int64_t k, int64_t n
) {
    if (y->coords.rank == 1) {
        int64_t off = mag_offset_rmn(y, k, 0, 0);
        return mag_float16_to_float32(by[off]);
    } else {
        int64_t off = mag_offset_rmn(y, yb_flat, k, n);
        return mag_float16_to_float32(by[off]);
    }
}

static MAG_AINLINE void mag_store_r_f16_from_f32(
    mag_tensor_t *r, mag_float16_t *br,
    int64_t rb_flat, int64_t i, int64_t n, float v
) {
    if (r->coords.rank == 0) {
        br[0] = mag_float32_to_float16(v);
    } else if (r->coords.rank == 1) {
        int64_t off = mag_offset_rmn(r, n, 0, 0);
        br[off] = mag_float32_to_float16(v);
    } else {
        int64_t off = mag_offset_rmn(r, rb_flat, i, n);
        br[off] = mag_float32_to_float16(v);
    }
}

static MAG_HOTPROC void mag_matmul_float16(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    const mag_tensor_t *y = mag_cmd_in(1);
    mag_float16_t *br = (mag_float16_t *)mag_tensor_data_ptr_mut(r);
    const mag_float16_t *bx = (const mag_float16_t *)mag_tensor_data_ptr(x);
    const mag_float16_t *by = (const mag_float16_t *)mag_tensor_data_ptr(y);
    int64_t M = (x->coords.rank == 1) ? 1 : x->coords.shape[x->coords.rank - 2];
    int64_t N = (y->coords.rank == 1) ? 1 : y->coords.shape[y->coords.rank - 1];
    int64_t K = x->coords.shape[x->coords.rank - 1];
    int64_t bdr = (r->coords.rank > 2) ? (r->coords.rank - 2) : 0;
    int64_t batch_total = 1;
    for (int64_t d = 0; d < bdr; ++d) batch_total *= r->coords.shape[d];
    int64_t bdx = (x->coords.rank > 2) ? (x->coords.rank - 2) : 0;
    int64_t bdy = (y->coords.rank > 2) ? (y->coords.rank - 2) : 0;
    int64_t tid = payload->thread_idx;
    int64_t nth = payload->thread_num;
    int64_t work_total = batch_total * M;
    for (int64_t work = tid; work < work_total; work += nth) {
        int64_t b = work / M;
        int64_t i = work - b * M;
        int64_t idx_r[MAG_MAX_DIMS] = {0};
        {
            int64_t rem = b;
            for (int64_t d = bdr - 1; d >= 0; --d) {
                idx_r[d] = rem % r->coords.shape[d];
                rem /= r->coords.shape[d];
            }
        }
        int64_t xb_flat = 0;
        for (int64_t d = 0; d < bdx; ++d) {
            int64_t rd  = bdr - bdx + d;
            int64_t idx = (x->coords.shape[d] == 1) ? 0 : idx_r[rd];
            xb_flat = xb_flat * x->coords.shape[d] + idx;
        }
        int64_t yb_flat = 0;
        for (int64_t d = 0; d < bdy; ++d) {
            int64_t rd  = bdr - bdy + d;
            int64_t idx = (y->coords.shape[d] == 1) ? 0 : idx_r[rd];
            yb_flat = yb_flat * y->coords.shape[d] + idx;
        }
        int64_t rb_flat = 0;
        for (int64_t d = 0; d < bdr; ++d)
            rb_flat = rb_flat * r->coords.shape[d] + idx_r[d];
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float ax = mag_load_x_f16_as_f32(x, bx, xb_flat, i, k);
                const float byv = mag_load_y_f16_as_f32(y, by, yb_flat, k, n);
                sum += ax * byv;
            }
            mag_store_r_f16_from_f32(r, br, rb_flat, i, n, sum);
        }
    }
}



/* == fp32 == */

static MAG_HOTPROC void mag_mm_block_float32(int64_t kc, int64_t mr, int64_t nr, const float *restrict A, const float *restrict B, int64_t ldb, float *restrict C, int64_t ldc, bool acc) {
    int64_t j = 0;
    int64_t m8 = mr&~7;
    for (; nr-j >= 32; j += 32) {
        int64_t i=0;
        for (; i+15 < m8; i += 16)
            mag_mm_tile_16x32_f32(kc, A + i*kc, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i+7 < m8; i += 8)
            mag_mm_tile_8x32_f32(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i)
            mag_mm_tile_1x32_f32(kc, A + i*kc, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr - j >= 16; j += 16) {
        int64_t i=0;
        for (; i+7 < m8; i += 8)
            mag_mm_tile_8x16_f32(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i)
            mag_mm_tile_1x16_f32(kc, A + i*kc, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr-j >= 8; j += 8) {
        int64_t i=0;
        for (; i+7 < m8; i += 8)
            mag_mm_tile_8x8_f32(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i)
            mag_mm_tile_1x8_f32(kc, A + i*kc, B + j, ldb, C + i*ldc + j, acc);
    }
    int64_t rem = nr - j;
    if (!rem) return;
    for (int64_t i2 = 0; i2 < mr; ++i2) {
        float *cp = C + i2 * ldc + j;
        if (i2 < m8) {
            int64_t panel = i2 >> 3;
            int64_t r = i2 & 7;
            const float *ap = A + panel*(kc<<3);
            for (int64_t jj = 0; jj < rem; ++jj) {
                float sum = acc ? cp[jj] : 0.0f;
                for (int64_t k = 0; k < kc; ++k)
                    sum += ap[k * MAG_MR + r] * B[k * ldb + (j + jj)];
                cp[jj] = sum;
            }
        } else {
            const float *ap = A + i2 * kc;
            for (int64_t jj = 0; jj < rem; ++jj) {
                float sum = acc ? cp[jj] : 0.0f;
                for (int64_t k = 0; k < kc; ++k)
                    sum += ap[k] * B[k * ldb + (j + jj)];
                cp[jj] = sum;
            }
        }
    }
}

MAG_HOTPROC static void mag_matmul_float32(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    const mag_tensor_t *y = mag_cmd_in(1);
    const float *bx = (const float *)mag_tensor_data_ptr(x);
    const float *by = (const float *)mag_tensor_data_ptr(y);
    float *br = (float *)mag_tensor_data_ptr_mut(r);
    int64_t MR = payload->mm_params.MR;
    int64_t MC = payload->mm_params.MC;
    int64_t KC = payload->mm_params.KC;
    int64_t NC = payload->mm_params.NC;
    int64_t NR = payload->mm_params.NR;
    int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
    int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
    int64_t K = x->coords.shape[x->coords.rank-1];
    int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d)
        batch_total *= r->coords.shape[d];
    if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { /* Detect GEMV */
        int64_t nth = payload->thread_num;
        int64_t tid = payload->thread_idx;
        int64_t j_per_thread = (N + nth - 1) / nth;
        int64_t j0 = tid*j_per_thread;
        int64_t j1 = mag_xmin(N, j0 + j_per_thread);
        for (int64_t batch = 0; batch < batch_total; ++batch) {
            const float *A = bx + mag_offset_rmn(x, batch, 0, 0);
            const float *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
            float *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
            mag_gemv_f32(K, j1 - j0, A, B, N, C);
        }
        return;
    }
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t tic = (M+MC-1)/MC;
    int64_t tjc = (N+NC-1)/NC;
    int64_t tpb = tic*tjc;
    int64_t tt = batch_total*tpb;
    mag_scratch_arena_clear(&mag_tls_arena);
    float *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));
    float *Bp = scratch;
    float *Ap = Bp + KC*NC;
    for (;;) {
        int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
        if (tile >= tt) break;
        int64_t batch_idx = tile / tpb;
        int64_t rem = tile % tpb;
        int64_t jc = rem % tjc;
        int64_t ic = rem / tjc;
        int64_t idx_r[MAG_MAX_DIMS] = {0};
        for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
            idx_r[d] = t % r->coords.shape[d];
            t /= r->coords.shape[d];
        }
        int64_t xb_flat = 0;
        for (int64_t d=0; d < bdx; ++d) {
            int64_t rd = bdr - bdx + d;
            xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        int64_t yb_flat = 0;
        for (int64_t d=0; d < bdy; ++d) {
            int64_t rd = bdr - bdy + d;
            yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        bool yv = y->coords.rank == 1;
        const float *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        const float *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
        float *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
        int64_t i0 = ic*MC;
        int64_t mc = i0+MC <= M ? MC : M-i0;
        int64_t j0 = jc*NC;
        int64_t nc = j0+NC <= N ? NC : N-j0;
        int64_t sMx = x->coords.strides[x->coords.rank-2];
        int64_t sKx = x->coords.strides[x->coords.rank-1];
        int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
        int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = mag_xmin(KC, K - pc);
            if (y->coords.rank == 1) {
                mag_mm_pack_B_vec_f32(kc, nc, py_base + pc, Bp);
            } else {
                mag_mm_pack_B_kc_nc_f32(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
            }
            mag_mm_pack_A_mc_kc_panel8_float32(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
            for (int64_t ir=0; ir < mc; ir += MR) {
                for (int64_t jr=0; jr < nc; jr += NR) {
                    int64_t mr = mag_xmin(MR, mc - ir);
                    int64_t nr = mag_xmin(NR, nc - jr);
                    const float *restrict A = Ap + ir*kc;
                    const float *restrict B = Bp + jr;
                    float *restrict C = pr_base + (i0 + ir) * N + (j0 + jr);
                    mag_mm_block_float32(kc, mr, nr, A, B, nc, C, N, pc != 0);
                }
            }
        }
    }
    mag_scratch_arena_clear(&mag_tls_arena);
}

