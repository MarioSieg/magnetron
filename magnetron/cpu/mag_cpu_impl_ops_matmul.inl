/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

static int64_t mag_offset_rmn(const mag_tensor_t *t, int64_t flat, int64_t i, int64_t j) {
    int64_t ra = t->coords.rank;
    const int64_t *restrict td = t->coords.shape;
    const int64_t *restrict ts = t->coords.strides;
    if (mag_likely(ra <= 3)) { /* Fast path */
        switch (ra) {
        case 1:
            return i*ts[0];
        case 2:
            return i*ts[0] + j*ts[1];
        case 3:
            return flat*ts[0] + i*ts[1] + j*ts[2];
        default:
            mag_panic("invalid rank: %" PRIi64, ra);
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

static MAG_HOTPROC mag_float16_t *mag_mm_pack_x_float16(mag_float16_t *xbuf, int64_t M, int64_t K, int64_t xb, const mag_tensor_t *x, const mag_float16_t *px) {
    for (int64_t i=0; i < M; ++i) {
        for (int64_t k=0; k < K; ++k) {
            size_t j = mag_offset_rmn(x, xb, i, k);
            mag_bnd_chk(px+j, px, mag_tensor_numbytes(x));
            mag_bnd_chk(xbuf + i*K + k, xbuf, M*K*sizeof(*xbuf));
            xbuf[i*K + k] = px[j];
        }
    }
    return xbuf;
}

static MAG_HOTPROC mag_float16_t *mag_mm_pack_y_float16(mag_float16_t *ybuf, int64_t K, int64_t N, int64_t yb, const mag_tensor_t *y, const mag_float16_t *py) {
    if (y->coords.rank == 1) {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[k];
            }
        }
    } else {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[mag_offset_rmn(y, yb, k, n)];
            }
        }
    }
    return ybuf;
}

#define MAG_PREFETCH_SPAN 8
#define MAG_PF_GROUP 8
#define MAG_PFDIST_B_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PFDIST_B_L2 (MAG_PREFETCH_SPAN*12)
#define MAG_PFDIST_A_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PFDIST_A_L2 (MAG_PREFETCH_SPAN*10)

#if defined(__GNUC__) || defined(__clang__)
#define mag_prefetchw(addr) __builtin_prefetch((addr), 1, 3)
#else
#define mag_prefetchw(addr) ((void)0)
#endif

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FMA
#define mag_vfmadd_float32(acc, a, b) vfmaq_f32((acc), (a), (b))
#else
#define mag_vfmadd_float32(acc, a, b) vmlaq_f32((acc), (a), (b))
#endif
#define mag_prefetcht0(p) __builtin_prefetch((const char*)(p), 0, 3)
#define mag_prefetcht1(p) __builtin_prefetch((const char*)(p), 0, 2)
#else
#define mag_prefetcht0(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#define mag_prefetcht1(p) _mm_prefetch((const char*)(p), _MM_HINT_T1)
#endif

static MAG_AINLINE void mag_mm_tile_8x8_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
#ifdef __AVX512F__
    __m512 C01, C23, C45, C67;
    if (acc) {
        __m256 c0 = _mm256_loadu_ps(c + 0*ldc);
        __m256 c1 = _mm256_loadu_ps(c + 1*ldc);
        __m256 c2 = _mm256_loadu_ps(c + 2*ldc);
        __m256 c3 = _mm256_loadu_ps(c + 3*ldc);
        __m256 c4 = _mm256_loadu_ps(c + 4*ldc);
        __m256 c5 = _mm256_loadu_ps(c + 5*ldc);
        __m256 c6 = _mm256_loadu_ps(c + 6*ldc);
        __m256 c7 = _mm256_loadu_ps(c + 7*ldc);
        C01 = _mm512_insertf32x8(_mm512_castps256_ps512(c0), c1, 1);
        C23 = _mm512_insertf32x8(_mm512_castps256_ps512(c2), c3, 1);
        C45 = _mm512_insertf32x8(_mm512_castps256_ps512(c4), c5, 1);
        C67 = _mm512_insertf32x8(_mm512_castps256_ps512(c6), c7, 1);
    } else {
        __m512 z = _mm512_setzero_ps();
        C01 = z;
        C23 = z;
        C45 = z;
        C67 = z;
    }
    __m512 P01e = _mm512_setzero_ps();
    __m512 P23e = _mm512_setzero_ps();
    __m512 P45e = _mm512_setzero_ps();
    __m512 P67e = _mm512_setzero_ps();
    __m512 P01o = _mm512_setzero_ps();
    __m512 P23o = _mm512_setzero_ps();
    __m512 P45o = _mm512_setzero_ps();
    __m512 P67o = _mm512_setzero_ps();
#define mag_plat_idx_pair(lo,hi) _mm512_set_epi32((hi),(hi),(hi),(hi),(hi),(hi),(hi),(hi),(lo),(lo),(lo),(lo),(lo),(lo),(lo),(lo))
    __m512i i01 = mag_plat_idx_pair(0,1);
    __m512i i23 = mag_plat_idx_pair(2,3);
    __m512i i45 = mag_plat_idx_pair(4,5);
    __m512i i67 = mag_plat_idx_pair(6,7);
#undef mag_plat_idx_pair
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + ((k + MAG_PFDIST_A_L1)<<3));
            mag_prefetcht1(a + ((k + MAG_PFDIST_A_L2)<<3));
        }
        __m256 b0_256 = _mm256_loadu_ps(b + (k + 0)  *ldb);
#ifdef __AVX512DQ__
        __m512 b0 = _mm512_broadcast_f32x8(b0_256);
#else
        __m512 b0 = _mm512_insertf32x8(_mm512_castps256_ps512(b0_256), b0_256, 1);
#endif
        __m256 av0 = _mm256_loadu_ps(a + k*8);
        __m512 Adup0 = _mm512_broadcast_f32x8(av0);
        __m512 A01_0 = _mm512_permutexvar_ps(i01, Adup0);
        __m512 A23_0 = _mm512_permutexvar_ps(i23, Adup0);
        __m512 A45_0 = _mm512_permutexvar_ps(i45, Adup0);
        __m512 A67_0 = _mm512_permutexvar_ps(i67, Adup0);
        P01e = _mm512_fmadd_ps(A01_0, b0, P01e);
        P23e = _mm512_fmadd_ps(A23_0, b0, P23e);
        P45e = _mm512_fmadd_ps(A45_0, b0, P45e);
        P67e = _mm512_fmadd_ps(A67_0, b0, P67e);
        __m256 b1_256 = _mm256_loadu_ps(b + (k + 1)*ldb);
#ifdef __AVX512DQ__
        __m512 b1 = _mm512_broadcast_f32x8(b1_256);
#else
        __m512 b1 = _mm512_insertf32x8(_mm512_castps256_ps512(b1_256), b1_256, 1);
#endif
        __m256 av1 = _mm256_loadu_ps(a + (k + 1)*8);
        __m512 Adup1 = _mm512_broadcast_f32x8(av1);
        __m512 A01_1 = _mm512_permutexvar_ps(i01, Adup1);
        __m512 A23_1 = _mm512_permutexvar_ps(i23, Adup1);
        __m512 A45_1 = _mm512_permutexvar_ps(i45, Adup1);
        __m512 A67_1 = _mm512_permutexvar_ps(i67, Adup1);
        P01o = _mm512_fmadd_ps(A01_1, b1, P01o);
        P23o = _mm512_fmadd_ps(A23_1, b1, P23o);
        P45o = _mm512_fmadd_ps(A45_1, b1, P45o);
        P67o = _mm512_fmadd_ps(A67_1, b1, P67o);
        __m256 b2_256 = _mm256_loadu_ps(b + (k + 2)*ldb);
#ifdef __AVX512DQ__
        __m512 b2 = _mm512_broadcast_f32x8(b2_256);
#else
        __m512 b2 = _mm512_insertf32x8(_mm512_castps256_ps512(b2_256), b2_256, 1);
#endif
        __m256 av2 = _mm256_loadu_ps(a + (k + 2)*8);
        __m512 Adup2 = _mm512_broadcast_f32x8(av2);
        __m512 A01_2 = _mm512_permutexvar_ps(i01, Adup2);
        __m512 A23_2 = _mm512_permutexvar_ps(i23, Adup2);
        __m512 A45_2 = _mm512_permutexvar_ps(i45, Adup2);
        __m512 A67_2 = _mm512_permutexvar_ps(i67, Adup2);
        P01e = _mm512_fmadd_ps(A01_2, b2, P01e);
        P23e = _mm512_fmadd_ps(A23_2, b2, P23e);
        P45e = _mm512_fmadd_ps(A45_2, b2, P45e);
        P67e = _mm512_fmadd_ps(A67_2, b2, P67e);
        __m256 b3_256 = _mm256_loadu_ps(b + (k + 3)*ldb);
#ifdef __AVX512DQ__
        __m512 b3 = _mm512_broadcast_f32x8(b3_256);
#else
        __m512 b3 = _mm512_insertf32x8(_mm512_castps256_ps512(b3_256), b3_256, 1);
#endif
        __m256 av3 = _mm256_loadu_ps(a + (k + 3)*8);
        __m512 Adup3 = _mm512_broadcast_f32x8(av3);
        __m512 A01_3 = _mm512_permutexvar_ps(i01, Adup3);
        __m512 A23_3 = _mm512_permutexvar_ps(i23, Adup3);
        __m512 A45_3 = _mm512_permutexvar_ps(i45, Adup3);
        __m512 A67_3 = _mm512_permutexvar_ps(i67, Adup3);
        P01o = _mm512_fmadd_ps(A01_3, b3, P01o);
        P23o = _mm512_fmadd_ps(A23_3, b3, P23o);
        P45o = _mm512_fmadd_ps(A45_3, b3, P45o);
        P67o = _mm512_fmadd_ps(A67_3, b3, P67o);
    }
    for (; k < kc; ++k) {
        __m256 bk_256 = _mm256_loadu_ps(b + k*ldb);
#ifdef __AVX512DQ__
        __m512 bk = _mm512_broadcast_f32x8(bk_256);
#else
        __m512 bk = _mm512_insertf32x8(_mm512_castps256_ps512(bk_256), bk_256, 1);
#endif
        __m256 av = _mm256_loadu_ps(a + k  *8);
        __m512 Adup = _mm512_broadcast_f32x8(av);
        __m512 A01 = _mm512_permutexvar_ps(i01, Adup);
        __m512 A23 = _mm512_permutexvar_ps(i23, Adup);
        __m512 A45 = _mm512_permutexvar_ps(i45, Adup);
        __m512 A67 = _mm512_permutexvar_ps(i67, Adup);
        if (k & 1) {
            P01o = _mm512_fmadd_ps(A01, bk, P01o);
            P23o = _mm512_fmadd_ps(A23, bk, P23o);
            P45o = _mm512_fmadd_ps(A45, bk, P45o);
            P67o = _mm512_fmadd_ps(A67, bk, P67o);
        } else {
            P01e = _mm512_fmadd_ps(A01, bk, P01e);
            P23e = _mm512_fmadd_ps(A23, bk, P23e);
            P45e = _mm512_fmadd_ps(A45, bk, P45e);
            P67e = _mm512_fmadd_ps(A67, bk, P67e);
        }
    }
    C01 = _mm512_add_ps(C01, _mm512_add_ps(P01e, P01o));
    C23 = _mm512_add_ps(C23, _mm512_add_ps(P23e, P23o));
    C45 = _mm512_add_ps(C45, _mm512_add_ps(P45e, P45o));
    C67 = _mm512_add_ps(C67, _mm512_add_ps(P67e, P67o));
    _mm256_storeu_ps(c + 0*ldc, _mm512_extractf32x8_ps(C01, 0));
    _mm256_storeu_ps(c + 1*ldc, _mm512_extractf32x8_ps(C01, 1));
    _mm256_storeu_ps(c + 2*ldc, _mm512_extractf32x8_ps(C23, 0));
    _mm256_storeu_ps(c + 3*ldc, _mm512_extractf32x8_ps(C23, 1));
    _mm256_storeu_ps(c + 4*ldc, _mm512_extractf32x8_ps(C45, 0));
    _mm256_storeu_ps(c + 5*ldc, _mm512_extractf32x8_ps(C45, 1));
    _mm256_storeu_ps(c + 6*ldc, _mm512_extractf32x8_ps(C67, 0));
    _mm256_storeu_ps(c + 7*ldc, _mm512_extractf32x8_ps(C67, 1));
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 C[8];
    if (acc) {
#pragma GCC unroll 8
        for (int r = 0; r < 8; ++r)
            C[r] = _mm256_loadu_ps(c + r*ldc);
    } else {
        __m256 z = _mm256_setzero_ps();
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r)
            C[r] = z;
    }
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if ((k & (MAG_PF_GROUP - 1)) == 0) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (int64_t)((k + MAG_PFDIST_A_L1)<<3));
            mag_prefetcht1(a + (int64_t)((k + MAG_PFDIST_A_L2)<<3));
        }
        __m256 B0 = _mm256_loadu_ps(b + (k + 0)*ldb);
        __m256 B1 = _mm256_loadu_ps(b + (k + 1)*ldb);
        __m256 B2 = _mm256_loadu_ps(b + (k + 2)*ldb);
        __m256 B3 = _mm256_loadu_ps(b + (k + 3)*ldb);
        const float *a0 = a + (k + 0)*8;
        const float *a1 = a + (k + 1)*8;
        const float *a2 = a + (k + 2)*8;
        const float *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m256 A;
            A = _mm256_broadcast_ss(a0 + r);
            C[r] = _mm256_fmadd_ps(A, B0, C[r]);
            A = _mm256_broadcast_ss(a1 + r);
            C[r] = _mm256_fmadd_ps(A, B1, C[r]);
            A = _mm256_broadcast_ss(a2 + r);
            C[r] = _mm256_fmadd_ps(A, B2, C[r]);
            A = _mm256_broadcast_ss(a3 + r);
            C[r] = _mm256_fmadd_ps(A, B3, C[r]);
        }
    }
    for (; k < kc; ++k) {
        __m256 Bk = _mm256_loadu_ps(b + k*ldb);
        const float *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m256 A = _mm256_broadcast_ss(ak + r);
            C[r] = _mm256_fmadd_ps(A, Bk, C[r]);
        }
    }
#pragma GCC unroll 8
    for (int r=0; r < 8; ++r)
        _mm256_storeu_ps(c + r*ldc, C[r]);
#elif defined(__SSE2__)
#define mm_fmadd_ps(a,b,c) _mm_add_ps((c), _mm_mul_ps((a),(b)))
    __m128 C[8][2];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            C[r][0] = _mm_loadu_ps(c + r*ldc + 0);
            C[r][1] = _mm_loadu_ps(c + r*ldc + 4);
        }
    } else {
        __m128 z = _mm_setzero_ps();
#pragma GCC unroll 8
        for (int r = 0; r < 8; ++r) C[r][0] = C[r][1] = z;
    }
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if ((k & (MAG_PF_GROUP - 1)) == 0) {
            _mm_prefetch((const char *)(b + (k + MAG_PFDIST_B_L1)*ldb), _MM_HINT_T0);
            _mm_prefetch((const char *)(b + (k + MAG_PFDIST_B_L2)*ldb), _MM_HINT_T1);
            _mm_prefetch((const char *)(a + (int64_t)((k + MAG_PFDIST_A_L1)*8)), _MM_HINT_T0);
            _mm_prefetch((const char *)(a + (int64_t)((k + MAG_PFDIST_A_L2)*8)), _MM_HINT_T1);
        }
        __m128 B0_0 = _mm_loadu_ps(b + (k + 0)*ldb + 0);
        __m128 B0_1 = _mm_loadu_ps(b + (k + 0)*ldb + 4);
        __m128 B1_0 = _mm_loadu_ps(b + (k + 1)*ldb + 0);
        __m128 B1_1 = _mm_loadu_ps(b + (k + 1)*ldb + 4);
        __m128 B2_0 = _mm_loadu_ps(b + (k + 2)*ldb + 0);
        __m128 B2_1 = _mm_loadu_ps(b + (k + 2)*ldb + 4);
        __m128 B3_0 = _mm_loadu_ps(b + (k + 3)*ldb + 0);
        __m128 B3_1 = _mm_loadu_ps(b + (k + 3)*ldb + 4);
        const float *a0 = a + (k + 0)*8;
        const float *a1 = a + (k + 1)*8;
        const float *a2 = a + (k + 2)*8;
        const float *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m128 A;
            A = _mm_set1_ps(a0[r]);
            C[r][0] = mm_fmadd_ps(A, B0_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B0_1, C[r][1]);
            A = _mm_set1_ps(a1[r]);
            C[r][0] = mm_fmadd_ps(A, B1_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B1_1, C[r][1]);
            A = _mm_set1_ps(a2[r]);
            C[r][0] = mm_fmadd_ps(A, B2_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B2_1, C[r][1]);
            A = _mm_set1_ps(a3[r]);
            C[r][0] = mm_fmadd_ps(A, B3_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B3_1, C[r][1]);
        }
    }
    for (; k < kc; ++k) {
        __m128 B0 = _mm_loadu_ps(b + k*ldb + 0);
        __m128 B1 = _mm_loadu_ps(b + k*ldb + 4);
        const float *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m128 A = _mm_set1_ps(ak[r]);
            C[r][0] = mm_fmadd_ps(A, B0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B1, C[r][1]);
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        _mm_storeu_ps(c + r*ldc + 0, C[r][0]);
        _mm_storeu_ps(c + r*ldc + 4, C[r][1]);
    }
#undef mm_fmadd_ps
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C[8][2];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            C[r][0] = vld1q_f32(c + r*ldc + 0);
            C[r][1] = vld1q_f32(c + r*ldc + 4);
        }
    } else {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r)
            C[r][0] = C[r][1] = vdupq_n_f32(0.f);
    }
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            __builtin_prefetch(b + (k + MAG_PFDIST_B_L1)*ldb, 0, 3);
            __builtin_prefetch(b + (k + MAG_PFDIST_B_L2)*ldb, 0, 2);
            __builtin_prefetch(a + (int64_t)((k + MAG_PFDIST_A_L1)<<3), 0, 3);
            __builtin_prefetch(a + (int64_t)((k + MAG_PFDIST_A_L2)<<3), 0, 2);
        }
        float32x4_t B0_0 = vld1q_f32(b + (k + 0)*ldb + 0);
        float32x4_t B0_1 = vld1q_f32(b + (k + 0)*ldb + 4);
        float32x4_t B1_0 = vld1q_f32(b + (k + 1)*ldb + 0);
        float32x4_t B1_1 = vld1q_f32(b + (k + 1)*ldb + 4);
        float32x4_t B2_0 = vld1q_f32(b + (k + 2)*ldb + 0);
        float32x4_t B2_1 = vld1q_f32(b + (k + 2)*ldb + 4);
        float32x4_t B3_0 = vld1q_f32(b + (k + 3)*ldb + 0);
        float32x4_t B3_1 = vld1q_f32(b + (k + 3)*ldb + 4);
        const float *a0 = a + (k + 0)*8;
        const float *a1 = a + (k + 1)*8;
        const float *a2 = a + (k + 2)*8;
        const float *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            float32x4_t A;
            A = vdupq_n_f32(a0[r]);
            C[r][0] = vfmaq_f32(C[r][0], B0_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B0_1, A);
            A = vdupq_n_f32(a1[r]);
            C[r][0] = vfmaq_f32(C[r][0], B1_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B1_1, A);
            A = vdupq_n_f32(a2[r]);
            C[r][0] = vfmaq_f32(C[r][0], B2_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B2_1, A);
            A = vdupq_n_f32(a3[r]);
            C[r][0] = vfmaq_f32(C[r][0], B3_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B3_1, A);
        }
    }
    for (; k < kc; ++k) {
        float32x4_t B0 = vld1q_f32(b + k*ldb + 0);
        float32x4_t B1 = vld1q_f32(b + k*ldb + 4);
        const float *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            float32x4_t A = vdupq_n_f32(ak[r]);
            C[r][0] = vfmaq_f32(C[r][0], B0, A);
            C[r][1] = vfmaq_f32(C[r][1], B1, A);
        }
    }
#pragma GCC unroll 8
    for (int r=0; r < 8; ++r) {
        vst1q_f32(c + r*ldc + 0, C[r][0]);
        vst1q_f32(c + r*ldc + 4, C[r][1]);
    }
#else
#error "Unsupported architecture"
#endif
}

static MAG_AINLINE void mag_mm_tile_8x16_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x8_float32(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x8_float32(kc, a, lda, b+8, ldb, c+8, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x32_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x16_float32(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x16_float32(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c, bool acc) {
#ifdef __AVX512F__
    __mmask16 m8 = 0x00ff;
    __m512 C = acc ? _mm512_maskz_loadu_ps(m8, c) : _mm512_setzero_ps();
    __m512 P0 = _mm512_setzero_ps();
    __m512 P1 = _mm512_setzero_ps();
    __m512 P2 = _mm512_setzero_ps();
    __m512 P3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP-1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (k + MAG_PFDIST_A_L1));
            mag_prefetcht1(a + (k + MAG_PFDIST_A_L2));
        }
        __m512 a0 = _mm512_set1_ps(a[k + 0]);
        __m512 a1 = _mm512_set1_ps(a[k + 1]);
        __m512 a2 = _mm512_set1_ps(a[k + 2]);
        __m512 a3 = _mm512_set1_ps(a[k + 3]);
        __m512 b0 = _mm512_maskz_loadu_ps(m8, b + (k + 0)*ldb);
        __m512 b1 = _mm512_maskz_loadu_ps(m8, b + (k + 1)*ldb);
        __m512 b2 = _mm512_maskz_loadu_ps(m8, b + (k + 2)*ldb);
        __m512 b3 = _mm512_maskz_loadu_ps(m8, b + (k + 3)*ldb);
        P0 = _mm512_fmadd_ps(a0, b0, P0);
        P1 = _mm512_fmadd_ps(a1, b1, P1);
        P2 = _mm512_fmadd_ps(a2, b2, P2);
        P3 = _mm512_fmadd_ps(a3, b3, P3);
    }
    C = _mm512_add_ps(C, _mm512_add_ps(_mm512_add_ps(P0, P1), _mm512_add_ps(P2, P3)));
    for (; k < kc; ++k) {
        __m512 ak = _mm512_set1_ps(a[k]);
        __m512 bk = _mm512_maskz_loadu_ps(m8, b + k  *ldb);
        C = _mm512_fmadd_ps(ak, bk, C);
    }
    _mm512_mask_storeu_ps(c, m8, C);
#elif defined(__AVX__) && defined(__FMA__)
    __m256 C0 = acc ? _mm256_loadu_ps(c) : _mm256_setzero_ps();
    for (int64_t k=0; k < kc; ++k) {
        __m256 A = _mm256_broadcast_ss(a + k);
        __m256 B0 = _mm256_loadu_ps(b + k*ldb + 0);
        C0 = _mm256_fmadd_ps(A, B0, C0);
    }
    _mm256_storeu_ps(c, C0);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C0 = acc ? vld1q_f32(c + 0) : vdupq_n_f32(0.0f);
    float32x4_t C1 = acc ? vld1q_f32(c + 4) : vdupq_n_f32(0.0f);
    for (int64_t k = 0; k < kc; ++k) {
        float32x4_t A = vdupq_n_f32(a[k]);
        float32x4_t B0 = vld1q_f32(b + k*ldb + 0);
        float32x4_t B1 = vld1q_f32(b + k*ldb + 4);
        C0 = mag_vfmadd_float32(C0, A, B0);
        C1 = mag_vfmadd_float32(C1, A, B1);
    }
    vst1q_f32(c + 0, C0);
    vst1q_f32(c + 4, C1);
#else
#pragma GCC unroll 8
    for (int64_t j=0; j < 8; ++j)
        c[j] = acc ? c[j] : 0.f;
    for (int64_t k=0; k < kc; ++k) {
        float a0 = a[k];
#pragma GCC unroll 8
        for (int64_t j=0; j < 8; ++j)
            c[j] += a0*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x16_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c, bool acc) {
#ifdef __AVX512F__
    __m512 C = acc ? _mm512_loadu_ps(c) : _mm512_setzero_ps();
    __m512 P0 = _mm512_setzero_ps();
    __m512 P1 = _mm512_setzero_ps();
    __m512 P2 = _mm512_setzero_ps();
    __m512 P3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (k + MAG_PFDIST_A_L1));
            mag_prefetcht1(a + (k + MAG_PFDIST_A_L2));
        }
        __m512 a0 = _mm512_set1_ps(a[k + 0]);
        __m512 a1 = _mm512_set1_ps(a[k + 1]);
        __m512 a2 = _mm512_set1_ps(a[k + 2]);
        __m512 a3 = _mm512_set1_ps(a[k + 3]);
        const float *B0p = b + (k + 0)*ldb;
        const float *B1p = b + (k + 1)*ldb;
        const float *B2p = b + (k + 2)*ldb;
        const float *B3p = b + (k + 3)*ldb;
        __m512 B0 = _mm512_loadu_ps(B0p);
        __m512 B1 = _mm512_loadu_ps(B1p);
        __m512 B2 = _mm512_loadu_ps(B2p);
        __m512 B3 = _mm512_loadu_ps(B3p);
        P0 = _mm512_fmadd_ps(a0, B0, P0);
        P1 = _mm512_fmadd_ps(a1, B1, P1);
        P2 = _mm512_fmadd_ps(a2, B2, P2);
        P3 = _mm512_fmadd_ps(a3, B3, P3);
    }
    C = _mm512_add_ps(C, _mm512_add_ps(_mm512_add_ps(P0, P1), _mm512_add_ps(P2, P3)));
    for (; k < kc; ++k) {
        __m512 ak = _mm512_set1_ps(a[k]);
        __m512 bk = _mm512_loadu_ps(b + k  *ldb);
        C = _mm512_fmadd_ps(ak, bk, C);
    }
    _mm512_storeu_ps(c, C);
#elif defined(__AVX__) && defined(__FMA__)
    __m256 C0 = acc ? _mm256_loadu_ps(c) : _mm256_setzero_ps();
    __m256 C1 = acc ? _mm256_loadu_ps(c+8) : _mm256_setzero_ps();
    for (int64_t k=0; k < kc; ++k) {
        __m256 A = _mm256_broadcast_ss(a + k);
        __m256 B0 = _mm256_loadu_ps(b + k*ldb + 0);
        __m256 B1 = _mm256_loadu_ps(b + k*ldb + 8);
        C0 = _mm256_fmadd_ps(A, B0, C0);
        C1 = _mm256_fmadd_ps(A, B1, C1);
    }
    _mm256_storeu_ps(c + 0, C0);
    _mm256_storeu_ps(c + 8, C1);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C0 = acc ? vld1q_f32(c + 0) : vdupq_n_f32(0.0f);
    float32x4_t C1 = acc ? vld1q_f32(c + 4) : vdupq_n_f32(0.0f);
    float32x4_t C2 = acc ? vld1q_f32(c + 8) : vdupq_n_f32(0.0f);
    float32x4_t C3 = acc ? vld1q_f32(c + 12) : vdupq_n_f32(0.0f);
    for (int64_t k=0; k < kc; ++k) {
        float32x4_t A = vdupq_n_f32(a[k]);
        const float *Bk = b + k*ldb;
        C0 = mag_vfmadd_float32(C0, A, vld1q_f32(Bk + 0));
        C1 = mag_vfmadd_float32(C1, A, vld1q_f32(Bk + 4));
        C2 = mag_vfmadd_float32(C2, A, vld1q_f32(Bk + 8));
        C3 = mag_vfmadd_float32(C3, A, vld1q_f32(Bk + 12));
    }
    vst1q_f32(c + 0, C0);
    vst1q_f32(c + 4, C1);
    vst1q_f32(c + 8, C2);
    vst1q_f32(c + 12, C3);
#else
#pragma GCC unroll 16
    for (int64_t j=0; j < 16; ++j)
        c[j] = acc ? c[j] : 0.f;
    for (int64_t k=0; k < kc; ++k) {
        float a0 = a[k];
#pragma GCC unroll 16
        for (int64_t j=0; j < 16; ++j)
            c[j] += a0*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x32_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c,  bool acc) {
    mag_mm_tile_1x16_float32(kc, a, b, ldb, c, acc);
    mag_mm_tile_1x16_float32(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_float32(int64_t kc, int64_t nc, const float *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN, float *restrict Bp) {
    if (strideN == 1) {
        for (int64_t k=0; k < kc; ++k) {
            const float *src = Bsrc + k*strideK;
#ifdef __AVX512F__
            int64_t j=0;
            float *dst = Bp + k*nc;
            for (; j+63 < nc; j += 64) {
                mag_prefetcht0(src + j + 256);
                mag_prefetcht1(src + j + 1024);
                __m512 v0 = _mm512_loadu_ps(src + j + 0);
                __m512 v1 = _mm512_loadu_ps(src + j + 16);
                __m512 v2 = _mm512_loadu_ps(src + j + 32);
                __m512 v3 = _mm512_loadu_ps(src + j + 48);
                _mm512_storeu_ps(dst + j + 0, v0);
                _mm512_storeu_ps(dst + j + 16, v1);
                _mm512_storeu_ps(dst + j + 32, v2);
                _mm512_storeu_ps(dst + j + 48, v3);
            }
            for (; j+31 < nc; j += 32) {
                __m512 v0 = _mm512_loadu_ps(src + j +  0);
                __m512 v1 = _mm512_loadu_ps(src + j + 16);
                _mm512_storeu_ps(dst + j +  0, v0);
                _mm512_storeu_ps(dst + j + 16, v1);
            }
            for (; j+15 < nc; j += 16) {
                __m512 v = _mm512_loadu_ps(src + j);
                _mm512_storeu_ps(dst + j, v);
            }
            if (j < nc) {
                int64_t rem = nc - j;
                __mmask16 m = rem == 16 ? 0xffff : (__mmask16)((1u<<rem)-1);
                __m512 v = _mm512_maskz_loadu_ps(m, src + j);
                _mm512_mask_storeu_ps(dst + j, m, v);
            }
#elif defined(__AVX__)
            int64_t j=0;
            for (; j+31 < nc; j += 32) {
                mag_prefetcht0(src + j + 128);
                mag_prefetcht1(src + j + 512);
                __m256 v0 = _mm256_loadu_ps(src + j + 0);
                __m256 v1 = _mm256_loadu_ps(src + j + 8);
                __m256 v2 = _mm256_loadu_ps(src + j + 16);
                __m256 v3 = _mm256_loadu_ps(src + j + 24);
                _mm256_storeu_ps(Bp + k*nc + j + 0, v0);
                _mm256_storeu_ps(Bp + k*nc + j + 8, v1);
                _mm256_storeu_ps(Bp + k*nc + j + 16, v2);
                _mm256_storeu_ps(Bp + k*nc + j + 24, v3);
            }
            for (; j+15 < nc; j += 16) {
                __m256 v0 = _mm256_loadu_ps(src + j + 0);
                __m256 v1 = _mm256_loadu_ps(src + j + 8);
                _mm256_storeu_ps(Bp + k*nc + j + 0, v0);
                _mm256_storeu_ps(Bp + k*nc + j + 8, v1);
            }
            for (; j+7 < nc; j += 8) {
                __m256 v = _mm256_loadu_ps(src + j);
                _mm256_storeu_ps(Bp + k*nc + j, v);
            }
            for (; j+3 < nc; j += 4) {
                __m128 v = _mm_loadu_ps(src + j);
                _mm_storeu_ps(Bp + k*nc + j, v);
            }
            for (; j < nc; ++j) Bp[k*nc + j] = src[j];
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            int64_t j = 0;
            for (; j+15 < nc; j += 16) {
                vst1q_f32(Bp + k*nc + j + 0, vld1q_f32(src + j + 0));
                vst1q_f32(Bp + k*nc + j + 4, vld1q_f32(src + j + 4));
                vst1q_f32(Bp + k*nc + j + 8, vld1q_f32(src + j + 8));
                vst1q_f32(Bp + k*nc + j + 12, vld1q_f32(src + j + 12));
            }
            for (; j+3 < nc; j += 4)
                vst1q_f32(Bp + k*nc + j, vld1q_f32(src + j));
            for (; j < nc; ++j)
                Bp[k*nc + j] = src[j];
#else
            memcpy(Bp + k*nc, src, nc*sizeof(*Bsrc));
#endif
        }
    } else {
        for (int64_t k=0; k < kc; ++k) {
            const float *src = Bsrc + k*strideK;
            for (int64_t j=0; j < nc; ++j)
                Bp[k*nc + j] = src[j*strideN];
        }
    }
}

static MAG_AINLINE void mag_mm_pack_A_mr8_kc_float32(int64_t kc, const float *restrict Asrc, ptrdiff_t strideK, float *restrict Ap) {
    if (strideK == 1) {
#ifdef __AVX512F__
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const float *src = Asrc + i*kc;
            float *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+63 < kc; k += 64) {
                mag_prefetcht0(src + k + 256);
                mag_prefetcht1(src + k + 1024);
                __m512 v0 = _mm512_loadu_ps(src + k + 0);
                __m512 v1 = _mm512_loadu_ps(src + k + 16);
                __m512 v2 = _mm512_loadu_ps(src + k + 32);
                __m512 v3 = _mm512_loadu_ps(src + k + 48);
                _mm512_storeu_ps(dst + k + 0, v0);
                _mm512_storeu_ps(dst + k + 16, v1);
                _mm512_storeu_ps(dst + k + 32, v2);
                _mm512_storeu_ps(dst + k + 48, v3);
            }
            for (; k+31 < kc; k += 32) {
                __m512 v0 = _mm512_loadu_ps(src + k + 0);
                __m512 v1 = _mm512_loadu_ps(src + k + 16);
                _mm512_storeu_ps(dst + k + 0, v0);
                _mm512_storeu_ps(dst + k + 16, v1);
            }
            for (; k+15 < kc; k += 16) {
                __m512 v = _mm512_loadu_ps(src + k);
                _mm512_storeu_ps(dst + k, v);
            }
            if (k < kc) {
                int64_t rem = kc - k;
                __mmask16 m = (__mmask16)((1u<<rem)-1);
                __m512 v = _mm512_maskz_loadu_ps(m, src + k);
                _mm512_mask_storeu_ps(dst + k, m, v);
            }
        }
#elif defined(__AVX2__)
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const float *src = Asrc + i*kc;
            float *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+7 < kc; k += 8) {
                __m256 v = _mm256_loadu_ps(src + k);
                _mm256_storeu_ps(dst + k, v);
            }
            for (; k+3 < kc; k += 4) {
                __m128 v = _mm_loadu_ps(src + k);
                _mm_storeu_ps(dst + k, v);
            }
            for (; k < kc; ++k) dst[k] = src[k];
        }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const float *src = Asrc + i*kc;
            float *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+15 < kc; k += 16) {
                vst1q_f32(dst + k + 0, vld1q_f32(src + k + 0));
                vst1q_f32(dst + k + 4, vld1q_f32(src + k + 4));
                vst1q_f32(dst + k + 8, vld1q_f32(src + k + 8));
                vst1q_f32(dst + k + 12, vld1q_f32(src + k + 12));
            }
            for (; k+3 < kc; k += 4)
                vst1q_f32(dst + k, vld1q_f32(src + k));
            for (; k < kc; ++k)
                dst[k] = src[k];
        }
#else
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i)
            memcpy(Ap + i*kc, Asrc + i*kc, kc*sizeof(*Asrc));
#endif
    } else {
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const float *src = Asrc + i*strideK*kc; /* start of row i */
            for (int64_t k = 0; k < kc; ++k)
                Ap[i*kc + k] = src[k*strideK];
        }
    }
}

static MAG_AINLINE void mag_mm_pack_B_vec_float32(int64_t kc, int64_t nc, const float *restrict yvec, float *restrict Bp) {
#ifdef __AVX512F__
    for (int64_t k=0; k < kc; ++k) {
        __m512 val = _mm512_set1_ps(yvec[k]);
        float *dst = Bp + k*nc;
        int64_t j=0;
        for (; j+63 < nc; j += 64) {
            _mm512_storeu_ps(dst + j + 0, val);
            _mm512_storeu_ps(dst + j + 16, val);
            _mm512_storeu_ps(dst + j + 32, val);
            _mm512_storeu_ps(dst + j + 48, val);
        }
        for (; j+31 < nc; j += 32) {
            _mm512_storeu_ps(dst + j + 0, val);
            _mm512_storeu_ps(dst + j + 16, val);
        }
        for (; j+15 < nc; j += 16) {
            _mm512_storeu_ps(dst + j, val);
        }
        if (j < nc) {
            int64_t rem = nc - j;
            __mmask16 m = (__mmask16)((1u<<rem)-1);
            _mm512_mask_storeu_ps(dst + j, m, val);
        }
    }
#elif defined(__AVX2__)
    for (int64_t k=0; k < kc; ++k) {
        __m256 val = _mm256_broadcast_ss(yvec + k);
        int64_t j = 0;
        for (; j+31 < nc; j += 32) {
            _mm256_storeu_ps(Bp + k*nc + j + 0, val);
            _mm256_storeu_ps(Bp + k*nc + j + 8, val);
            _mm256_storeu_ps(Bp + k*nc + j + 16, val);
            _mm256_storeu_ps(Bp + k*nc + j + 24, val);
        }
        for (; j+15 < nc; j += 16) {
            _mm256_storeu_ps(Bp + k*nc + j, val);
            _mm256_storeu_ps(Bp + k*nc + j + 8, val);
        }
        for (; j+7 < nc; j += 8)
            _mm256_storeu_ps(Bp + k*nc + j, val);
        for (; j+3 < nc; j += 4)
            _mm_storeu_ps(Bp + k*nc + j, _mm256_castps256_ps128(val));
        for (; j < nc; ++j)
            Bp[k*nc + j] = yvec[k];
    }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (int64_t k=0; k < kc; ++k) {
        float32x4_t val = vdupq_n_f32(yvec[k]);
        int64_t j=0;
        for (; j+15 < nc; j += 16) {
            vst1q_f32(Bp + k*nc + j + 0, val);
            vst1q_f32(Bp + k*nc + j + 4, val);
            vst1q_f32(Bp + k*nc + j + 8, val);
            vst1q_f32(Bp + k*nc + j + 12, val);
        }
        for (; j+3 < nc; j += 4)
            vst1q_f32(Bp + k*nc + j, val);
        for (; j < nc; ++j)
            Bp[k*nc + j] = yvec[k];
    }
#else
    for (int64_t k = 0; k < kc; ++k) {
        float v = yvec[k];
        for (int64_t j=0; j < nc; ++j)
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
            if ((k & ((MAG_PF_GROUP<<1) - 1)) == 0) {
                mag_prefetcht0(p0 + (int64_t)MAG_PFDIST_A_L1*sKx);
                mag_prefetcht0(p4 + (int64_t)MAG_PFDIST_A_L1*sKx);
                mag_prefetcht1(p0 + (int64_t)MAG_PFDIST_A_L2*sKx);
                mag_prefetcht1(p4 + (int64_t)MAG_PFDIST_A_L2*sKx);
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
            mag_prefetcht0(src + (k + MAG_PFDIST_A_L1)*sKx);
            mag_prefetcht1(src + (k + MAG_PFDIST_A_L2)*sKx);
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

static MAG_AINLINE void mag_mv_float32(int64_t K, int64_t N, const float *restrict A, const float *restrict B, int64_t ldb, float *restrict C) {
#ifdef __AVX512F__
    int64_t j=0;
    for (; j+127 < N; j += 128) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        __m512 s7 = _mm512_setzero_ps();
        const float *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do { \
                    __m512 a = _mm512_set1_ps(A[k + (i)]); \
                    const float *bp = brow + (i)*ldb; \
                    s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 0), s0); \
                    s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 16), s1); \
                    s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 32), s2); \
                    s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 48), s3); \
                    s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 64), s4); \
                    s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 80), s5); \
                    s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 96), s6); \
                    s7 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 112), s7); \
                } while (0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=(K&~3); k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 96), s6);
            s7 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 112), s7);
        }
        _mm512_storeu_ps(C + j + 0, s0);
        _mm512_storeu_ps(C + j + 16, s1);
        _mm512_storeu_ps(C + j + 32, s2);
        _mm512_storeu_ps(C + j + 48, s3);
        _mm512_storeu_ps(C + j + 64, s4);
        _mm512_storeu_ps(C + j + 80, s5);
        _mm512_storeu_ps(C + j + 96, s6);
        _mm512_storeu_ps(C + j + 112, s7);
    }
    for (; j+63 < N; j += 64) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        const float *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do { \
                    __m512 a = _mm512_set1_ps(A[k + (i)]); \
                    const float *bp = brow + (i)*ldb; \
                    s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 0), s0); \
                    s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 16), s1); \
                    s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 32), s2); \
                    s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 48), s3); \
                } while (0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=(K&~3); k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 48), s3);
        }

        _mm512_storeu_ps(C + j + 0, s0);
        _mm512_storeu_ps(C + j + 16, s1);
        _mm512_storeu_ps(C + j + 32, s2);
        _mm512_storeu_ps(C + j + 48, s3);
    }
    for (; j+15 < N; j += 16) {
        __m512 s = _mm512_setzero_ps();
        const float *restrict brow = B + j;
        for (int64_t k=0; k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow), s);
        }
        _mm512_storeu_ps(C + j, s);
    }
    if (j < N) {
        int64_t rem = N-j;
        __mmask16 m = rem == 16 ? (__mmask16)0xffff : (__mmask16)((1u<<rem)-1);
        __m512 s = _mm512_setzero_ps();
        const float *restrict brow = B + j;
        for (int64_t k=0; k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            __m512 bv = _mm512_maskz_loadu_ps(m, brow);
            s = _mm512_fmadd_ps(a, bv, s);
        }
        _mm512_mask_storeu_ps(C + j, m, s);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    int64_t j = 0;
    for (; j+63 < N; j += 64) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        __m256 s7 = _mm256_setzero_ps();
        const float *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do {                                        \
                    __m256 a = _mm256_broadcast_ss(A + k + i);              \
                    const float *restrict bp = brow + i*ldb;          \
                    s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp +  0), s0);  \
                    s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp +  8), s1);  \
                    s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 16), s2);  \
                    s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 24), s3);  \
                    s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 32), s4);  \
                    s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 40), s5);  \
                    s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 48), s6);  \
                    s7 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 56), s7);  \
                } while(0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=K & ~3; k < K; ++k, brow += ldb) {
            __m256 a = _mm256_broadcast_ss(A + k);
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow +  0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow +  8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 48), s6);
            s7 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 56), s7);
        }
        _mm256_storeu_ps(C + j +  0, s0);
        _mm256_storeu_ps(C + j +  8, s1);
        _mm256_storeu_ps(C + j + 16, s2);
        _mm256_storeu_ps(C + j + 24, s3);
        _mm256_storeu_ps(C + j + 32, s4);
        _mm256_storeu_ps(C + j + 40, s5);
        _mm256_storeu_ps(C + j + 48, s6);
        _mm256_storeu_ps(C + j + 56, s7);
    }
    for (; j+7 < N; j += 8) {
        __m256 s = _mm256_setzero_ps();
        const float *restrict b = B + j;
        for (int64_t k=0; k < K; ++k, b += ldb)
            s = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), _mm256_loadu_ps(b), s);
        _mm256_storeu_ps(C + j, s);
    }
    for (; j < N; ++j) {
        float s = 0.f;
        for (int64_t k=0; k < K; ++k)
            s += A[k]*B[k*ldb + j];
        C[j] = s;
    }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    int64_t NN = N&-8;
    int64_t j=0;
    for (; j < NN; j += 8) {
        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        for (int64_t k=0; k < K; ++k) {
            float32x4_t b0 = vld1q_f32(B + k*ldb + j + 0);
            float32x4_t b1 = vld1q_f32(B + k*ldb + j + 4);
            float32x4_t a = vdupq_n_f32(A[k]);
            sum0 = mag_vfmadd_float32(sum0, a, b0);
            sum1 = mag_vfmadd_float32(sum1, a, b1);
        }
        vst1q_f32(C + j + 0, sum0);
        vst1q_f32(C + j + 4, sum1);
    }
    for (; j < N; ++j) {
        float sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            sum += A[k]*B[k*ldb + j];
        C[j] = sum;
    }
#else
    for (int64_t j = 0; j < N; ++j) {
        float sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            sum += A[k]*B[k*ldb + j];
        C[j] = sum;
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_16x16_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x16_float32(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x16_float32(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_16x16_float32(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_16x16_float32(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_HOTPROC void mag_mm_block_float32(int64_t kc, int64_t mr, int64_t nr, const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc, bool acc) {
    int64_t j = 0;
    for (; nr-j >= 32; j += 32) {
        int64_t i = 0;
        for (; mr-i >= 16; i += 16) mag_mm_tile_16x32_float32(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x32_float32 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x32_float32 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr-j >= 16; j += 16) {
        int64_t i = 0;
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x16_float32 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x16_float32 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr-j >= 8; j += 8) {
        int64_t i = 0;
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x8_float32 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x8_float32 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    int64_t rem = nr-j;
    if (!rem) return;
    for (int64_t i2=0; i2 < mr; ++i2) {
        const float *ap = A + i2*lda;
        float *cp = C + i2*ldc + j;
        for (int64_t jj = 0; jj < rem; ++jj) {
            float sum = acc ? cp[jj] : 0.f;
            for (int64_t k=0; k < kc; ++k)
                sum += ap[k]*B[k*ldb + (j + jj)];
            cp[jj] = sum;
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
            mag_mv_float32(K, j1 - j0, A, B, N, C);
        }
        return;
    }
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t tic = (M+MC-1)/MC;
    int64_t tjc = (N+NC-1)/NC;
    int64_t tpb = tic  *tjc;
    int64_t tt = batch_total  *tpb;
    float *scratch = mag_sb_acquire(sizeof(*scratch)*(KC*NC + MC*KC));
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
            if (y->coords.rank == 1) mag_mm_pack_B_vec_float32(kc, nc, py_base + pc, Bp);
            else mag_mm_pack_B_kc_nc_float32(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
            mag_mm_pack_A_mc_kc_panel8_float32(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
            for (int64_t ir=0; ir < mc; ir += MR)
                for (int64_t jr=0; jr < nc; jr += NR)
                    mag_mm_block_float32(
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
}

static MAG_HOTPROC void mag_matmul_float16(const mag_kernel_payload_t *payload) {
    if (payload->thread_idx != 0) return;
    mag_tensor_t *r  = mag_cmd_out(0);
    const mag_tensor_t *x  = mag_cmd_in(0);
    const mag_tensor_t *y  = mag_cmd_in(1);
    mag_float16_t *br = (mag_float16_t *)mag_tensor_data_ptr_mut(r);
    const mag_float16_t *bx = (const mag_float16_t *)mag_tensor_data_ptr(x);
    const mag_float16_t *by = (const mag_float16_t *)mag_tensor_data_ptr(y);
    int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank - 2];
    int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank - 1];
    int64_t K = x->coords.shape[x->coords.rank - 1];
    int64_t bdr = r->coords.rank > 2 ? r->coords.rank-2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d) batch_total *= r->coords.shape[d];
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    bool x_row = mag_tensor_is_contiguous(x) && x->coords.strides[x->coords.rank-1] == 1;
    mag_float16_t *scratch = mag_sb_acquire(sizeof(mag_float16_t)*(K*N + (x_row ? 0 : M*K)));
    mag_float16_t *xbuf = x_row ? NULL : scratch;
    mag_float16_t *ybuf = scratch + (x_row ? 0 : M*K);
    int64_t idx_r[4] = {0};
    for (int64_t b=0; b < batch_total; ++b) {
        int64_t rem = b;
        for (int64_t d = bdr-1; d >= 0; --d) {
            idx_r[d] = rem % r->coords.shape[d];
            rem /= r->coords.shape[d];
        }
        int64_t xb_flat = 0, yb_flat = 0;
        if (bdx) {
            for (int64_t d=0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = x->coords.shape[d] == 1 ? 0 : idx_r[rd];
                xb_flat = xb_flat  *x->coords.shape[d] + idx;
            }
        }
        if (bdy) {
            for (int64_t d=0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = y->coords.shape[d] == 1 ? 0 : idx_r[rd];
                yb_flat = yb_flat  *y->coords.shape[d] + idx;
            }
        }
        const mag_float16_t *px = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        mag_float16_t *pr = br + mag_offset_rmn(r,b, 0, 0);
        const mag_float16_t *restrict A = x_row ? px : mag_mm_pack_x_float16(xbuf, M, K, xb_flat, x, bx);
        const mag_float16_t *restrict B = mag_mm_pack_y_float16(ybuf, K, N, yb_flat, y, by);
        mag_float16_t *restrict C = pr;
        for (int64_t i=0; i < M; ++i) {
            const mag_float16_t *restrict a_row = A + i*K;
            for (int64_t n=0; n < N; ++n) {
                const mag_float16_t *restrict b_col = B + n*K;
                C[i*N + n] = mag_vdot_float16(K, b_col, a_row);
            }
        }
    }
}
