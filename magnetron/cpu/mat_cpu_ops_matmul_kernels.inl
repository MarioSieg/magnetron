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

#define MAG_PREFETCH_SPAN 8
#define MAG_PREFETCH_GROUPS 8
#define MAG_PREFETCH_GROUPS_MASK (MAG_PREFETCH_GROUPS-1)
#define MAG_PREFETCH_RANGE_B_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PREFETCH_RANGE_B_L2 (MAG_PREFETCH_SPAN*12)
#define MAG_PREFETCH_RANGE_A_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PREFETCH_RANGE_A_L2 (MAG_PREFETCH_SPAN*10)
#define MAG_MR 8
#define MAG_NR_F32 MAG_VF32_LANES

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8xnr_f32(
    int64_t kc,
    const float *restrict a, int64_t lda,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
    mag_vf32_t base[MAG_MR];
    mag_vf32_t partial0[MAG_MR];
    mag_vf32_t partial1[MAG_MR];
    mag_vf32_t partial2[MAG_MR];
    mag_vf32_t partial3[MAG_MR];
    if (do_acc) {
        #pragma GCC unroll 8
        for (int r=0; r < MAG_MR; ++r)
            base[r] = mag_vf32_loadu(c + r*ldc);
    } else {
        mag_vf32_t z = mag_vf32_zero();
        #pragma GCC unroll 8
        for (int r=0; r < MAG_MR; ++r)
            base[r] = z;
    }
    {
        mag_vf32_t z = mag_vf32_zero();
        #pragma GCC unroll 8
        for (int r=0; r < MAG_MR; ++r)
            partial0[r] = partial1[r] = partial2[r] = partial3[r] = z;
    }
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1)*lda);
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2)*lda);
        }
        mag_vf32_t b0 = mag_vf32_loadu(b + (k + 0)*ldb);
        mag_vf32_t b1 = mag_vf32_loadu(b + (k + 1)*ldb);
        mag_vf32_t b2 = mag_vf32_loadu(b + (k + 2)*ldb);
        mag_vf32_t b3 = mag_vf32_loadu(b + (k + 3)*ldb);
        const float *a0 = a + (k + 0)*lda;
        const float *a1 = a + (k + 1)*lda;
        const float *a2 = a + (k + 2)*lda;
        const float *a3 = a + (k + 3)*lda;
        #pragma GCC unroll 8
        for (int r=0; r < MAG_MR; ++r) {
            partial0[r] = mag_vf32_fmadd(mag_vf32_broadcast(a0 + r), b0, partial0[r]);
            partial1[r] = mag_vf32_fmadd(mag_vf32_broadcast(a1 + r), b1, partial1[r]);
            partial2[r] = mag_vf32_fmadd(mag_vf32_broadcast(a2 + r), b2, partial2[r]);
            partial3[r] = mag_vf32_fmadd(mag_vf32_broadcast(a3 + r), b3, partial3[r]);
        }
    }
    #pragma GCC unroll 8
    for (int r=0; r < MAG_MR; ++r) {
        base[r] = mag_vf32_add(
            base[r],
            mag_vf32_add(
                mag_vf32_add(partial0[r], partial1[r]),
                mag_vf32_add(partial2[r], partial3[r])
            )
        );
    }
    for (; k < kc; ++k) {
        mag_vf32_t bk = mag_vf32_loadu(b + k * ldb);
        const float *ak = a + k * lda;
        #pragma GCC unroll 8
        for (int r=0; r < MAG_MR; ++r)
            base[r] = mag_vf32_fmadd(mag_vf32_broadcast(ak + r), bk, base[r]);
    }
    #pragma GCC unroll 8
    for (int r=0; r < MAG_MR; ++r)
        mag_vf32_storeu(c + r * ldc, base[r]);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x8_f32(
    int64_t kc,
    const float *restrict a, int64_t lda,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
#if MAG_NR_F32 == 16
    mag_mm_tile_8xnr_f32(kc, a, lda, b, ldb, c, ldc, do_acc);
#elif MAG_NR_F32 == 8
    mag_mm_tile_8xnr_f32(kc, a, lda, b, ldb, c, ldc, do_acc);
#elif MAG_NR_F32 == 4
    mag_mm_tile_8xnr_f32(kc, a, lda, b + 0, ldb, c + 0, ldc, do_acc);
    mag_mm_tile_8xnr_f32(kc, a, lda, b + 4, ldb, c + 4, ldc, do_acc);
#else
    for (int r=0; r < 8; ++r) {
        for (int j=0; j < 8; ++j) {
            float sum = do_acc ? c[r*ldc + j] : 0.0f;
            for (int64_t k=0; k < kc; ++k)
                sum += a[k*lda + r]*b[k*ldb + j];
            c[r*ldc + j] = sum;
        }
    }
#endif
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x16_f32(
    int64_t kc,
    const float *restrict a, int64_t lda,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
#if MAG_NR_F32 == 16
    mag_mm_tile_8xnr_f32(kc, a, lda, b, ldb, c, ldc, do_acc);
#elif MAG_NR_F32 == 8
    mag_mm_tile_8xnr_f32(kc, a, lda, b + 0, ldb, c + 0, ldc, do_acc);
    mag_mm_tile_8xnr_f32(kc, a, lda, b + 8, ldb, c + 8, ldc, do_acc);
#elif MAG_NR_F32 == 4
    mag_mm_tile_8xnr_f32(kc, a, lda, b +  0, ldb, c +  0, ldc, do_acc);
    mag_mm_tile_8xnr_f32(kc, a, lda, b +  4, ldb, c +  4, ldc, do_acc);
    mag_mm_tile_8xnr_f32(kc, a, lda, b +  8, ldb, c +  8, ldc, do_acc);
    mag_mm_tile_8xnr_f32(kc, a, lda, b + 12, ldb, c + 12, ldc, do_acc);
#else
    for (int r=0; r < 8; ++r) {
        for (int j=0; j < 16; ++j) {
            float sum = do_acc ? c[r*ldc + j] : 0.0f;
            for (int64_t k=0; k < kc; ++k)
                sum += a[k*lda + r]*b[k*ldb + j];
            c[r*ldc + j] = sum;
        }
    }
#endif
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x32_f32(
    int64_t kc,
    const float *restrict a, int64_t lda,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
    mag_mm_tile_8x16_f32(kc, a, lda, b +  0, ldb, c +  0, ldc, do_acc);
    mag_mm_tile_8x16_f32(kc, a, lda, b + 16, ldb, c + 16, ldc, do_acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1xnr_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    bool do_acc
) {
    mag_vf32_t partial0 = mag_vf32_zero();
    mag_vf32_t partial1 = mag_vf32_zero();
    mag_vf32_t partial2 = mag_vf32_zero();
    mag_vf32_t partial3 = mag_vf32_zero();
    int64_t k=0;
    for (; k + 3 < kc; k += 4) {
        if ((k & MAG_PREFETCH_GROUPS_MASK) == 0) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }
        partial0 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 0), mag_vf32_loadu(b + (k + 0)*ldb), partial0 );
        partial1 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 1), mag_vf32_loadu(b + (k + 1)*ldb), partial1 );
        partial2 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 2), mag_vf32_loadu(b + (k + 2)*ldb), partial2 );
        partial3 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 3), mag_vf32_loadu(b + (k + 3)*ldb), partial3 );
    }
    mag_vf32_t acc = do_acc ? mag_vf32_loadu(c) : mag_vf32_zero();
    acc = mag_vf32_add(acc, mag_vf32_add(mag_vf32_add(partial0, partial1), mag_vf32_add(partial2, partial3)));
    for (; k < kc; ++k)
        acc = mag_vf32_fmadd(mag_vf32_broadcast(a + k), mag_vf32_loadu(b + k*ldb), acc);
    mag_vf32_storeu(c, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1xnr_masked_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    unsigned n,
    bool do_acc
) {
    mag_vf32_t partial0 = mag_vf32_zero();
    mag_vf32_t partial1 = mag_vf32_zero();
    mag_vf32_t partial2 = mag_vf32_zero();
    mag_vf32_t partial3 = mag_vf32_zero();
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }
        partial0 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 0), mag_vf32_loadu_masked(b + (k + 0)*ldb, n), partial0);
        partial1 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 1), mag_vf32_loadu_masked(b + (k + 1)*ldb, n), partial1);
        partial2 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 2), mag_vf32_loadu_masked(b + (k + 2)*ldb, n), partial2);
        partial3 = mag_vf32_fmadd(mag_vf32_broadcast(a + k + 3), mag_vf32_loadu_masked(b + (k + 3)*ldb, n), partial3);
    }
    mag_vf32_t acc = do_acc ? mag_vf32_loadu_masked(c, n) : mag_vf32_zero();
    acc = mag_vf32_add(acc, mag_vf32_add(mag_vf32_add(partial0, partial1), mag_vf32_add(partial2, partial3)));
    for (; k < kc; ++k)
        acc = mag_vf32_fmadd(mag_vf32_broadcast(a + k), mag_vf32_loadu_masked(b + k*ldb, n), acc);
    mag_vf32_storeu_masked(c, acc, n);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x8_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    bool do_acc
) {
#if MAG_NR_F32 == 16
    mag_mm_tile_1xnr_masked_f32(kc, a, b, ldb, c, 8, do_acc);
#elif MAG_NR_F32 == 8
    mag_mm_tile_1xnr_f32(kc, a, b, ldb, c, do_acc);
#elif MAG_NR_F32 == 4
    mag_mm_tile_1xnr_f32(kc, a, b + 0, ldb, c + 0, do_acc);
    mag_mm_tile_1xnr_f32(kc, a, b + 4, ldb, c + 4, do_acc);
#else
    for (int64_t j=0; j < 8; ++j)
        c[j] = do_acc ? c[j] : 0.0f;
    for (int64_t k=0; k < kc; ++k) {
        float ak = a[k];
        for (int64_t j=0; j < 8; ++j)
            c[j] += ak*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x16_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    bool do_acc
) {
#if MAG_NR_F32 == 16
    mag_mm_tile_1xnr_f32(kc, a, b, ldb, c, do_acc);
#elif MAG_NR_F32 == 8
    mag_mm_tile_1xnr_f32(kc, a, b + 0, ldb, c + 0, do_acc);
    mag_mm_tile_1xnr_f32(kc, a, b + 8, ldb, c + 8, do_acc);
#elif MAG_NR_F32 == 4
    mag_mm_tile_1xnr_f32(kc, a, b +  0, ldb, c +  0, do_acc);
    mag_mm_tile_1xnr_f32(kc, a, b +  4, ldb, c +  4, do_acc);
    mag_mm_tile_1xnr_f32(kc, a, b +  8, ldb, c +  8, do_acc);
    mag_mm_tile_1xnr_f32(kc, a, b + 12, ldb, c + 12, do_acc);
#else
    for (int64_t j=0; j < 16; ++j)
        c[j] = do_acc ? c[j] : 0.0f;
    for (int64_t k=0; k < kc; ++k) {
        float ak = a[k];
        for (int64_t j=0; j < 16; ++j)
            c[j] += ak*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x32_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    bool do_acc
) {
    mag_mm_tile_1x16_f32(kc, a, b +  0, ldb, c +  0, do_acc);
    mag_mm_tile_1x16_f32(kc, a, b + 16, ldb, c + 16, do_acc);
}

static MAG_AINLINE void mag_mm_tile_16x16_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
    mag_mm_tile_8x16_f32(kc, a + 0*(8*kc), MAG_MR, b, ldb, c + 0*ldc, ldc, do_acc);
    mag_mm_tile_8x16_f32(kc, a + 1*(8*kc), MAG_MR, b, ldb, c + 8*ldc, ldc, do_acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
    mag_mm_tile_16x16_f32(kc, a, b +  0, ldb, c +  0, ldc, do_acc);
    mag_mm_tile_16x16_f32(kc, a, b + 16, ldb, c + 16, ldc, do_acc);
}


static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_B_kc_nc_f32(
    int64_t kc,
    int64_t nc,
    const float *restrict bs,
    int64_t sk,
    int64_t sn,
    float *restrict bp
) {
    if (sn == 1) {
        for (int64_t k=0; k < kc; ++k) {
            const float *restrict src = bs + k*sk;
            float *restrict dst = bp + k*nc;
            int64_t j=0;
            for (; j+MAG_VF32_LANES-1 < nc; j += MAG_VF32_LANES) {
                if (!(j & MAG_PREFETCH_GROUPS_MASK)) {
                    mag_simd_prefetch_t0(src + j + MAG_PREFETCH_RANGE_B_L1);
                    mag_simd_prefetch_t1(src + j + MAG_PREFETCH_RANGE_B_L2);
                }
                mag_vf32_storeu(dst + j, mag_vf32_loadu(src + j));
            }
            if (j < nc) {
                unsigned rem = (unsigned)(nc - j);
                mag_vf32_storeu_masked(dst + j, mag_vf32_loadu_masked(src + j, rem), rem);
            }
        }
    } else {
        for (int64_t k=0; k < kc; ++k) {
            const float *restrict src = bs + k*sk;
            float *restrict dst = bp + k*nc;
            for (int64_t j=0; j < nc; ++j)
                dst[j] = src[j*sn];
        }
    }
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_A_mr8_kc_f32(
    int64_t kc,
    const float *restrict as,
    int64_t sk,
    float *restrict ap
) {
    if (sk == 1) {
        for (int64_t k=0; k < kc; ++k) {
            #pragma GCC unroll 8
            for (int r=0; r < MAG_MR; ++r)
                ap[k*MAG_MR + r] = as[r*kc + k];
        }
    } else {
        for (int64_t k=0; k < kc; ++k) {
            #pragma GCC unroll 8
            for (int r=0; r < MAG_MR; ++r)
                ap[k*MAG_MR + r] = as[r*sk*kc + k*sk];
        }
    }
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_B_vec_f32(
    int64_t kc,
    int64_t nc,
    const float *restrict yvec,
    float *restrict bp
) {
    for (int64_t k=0; k < kc; ++k) {
        mag_vf32_t val = mag_vf32_broadcast(yvec + k);
        float *restrict dst = bp + k*nc;
        int64_t j=0;
        int64_t step = 4*MAG_VF32_LANES;
        for (; j + step - 1 < nc; j += step) {
            mag_vf32_storeu(dst + j + 0*MAG_VF32_LANES, val);
            mag_vf32_storeu(dst + j + 1*MAG_VF32_LANES, val);
            mag_vf32_storeu(dst + j + 2*MAG_VF32_LANES, val);
            mag_vf32_storeu(dst + j + 3*MAG_VF32_LANES, val);
        }
        for (; j+MAG_VF32_LANES-1 < nc; j += MAG_VF32_LANES)
            mag_vf32_storeu(dst + j, val);
        if (j < nc) {
            unsigned rem = (unsigned)(nc - j);
            mag_vf32_storeu_masked(dst + j, val, rem);
        }
    }
}

static MAG_AINLINE MAG_HOTPROC void mag_gemv_vf32_f32(
    int64_t K,
    const float *restrict A,
    const float *restrict B,
    int64_t ldb,
    float *restrict C
) {
    mag_vf32_t partial0 = mag_vf32_zero();
    mag_vf32_t partial1 = mag_vf32_zero();
    mag_vf32_t partial2 = mag_vf32_zero();
    mag_vf32_t partial3 = mag_vf32_zero();
    const float *restrict row = B;
    int64_t k=0;
    int64_t inc = ldb << 2;
    for (; k+3 < K; k += 4, row += inc) {
        partial0 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 0), mag_vf32_loadu(row + 0*ldb), partial0);
        partial1 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 1), mag_vf32_loadu(row + 1*ldb), partial1);
        partial2 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 2), mag_vf32_loadu(row + 2*ldb), partial2);
        partial3 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 3), mag_vf32_loadu(row + 3*ldb), partial3);
    }
    mag_vf32_t acc = mag_vf32_add(mag_vf32_add(partial0, partial1), mag_vf32_add(partial2, partial3));
    for (; k < K; ++k, row += ldb)
        acc = mag_vf32_fmadd(mag_vf32_broadcast(A + k), mag_vf32_loadu(row), acc);
    mag_vf32_storeu(C, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_gemv_vf32_masked_f32(
    int64_t K,
    const float *restrict A,
    const float *restrict B,
    int64_t ldb,
    float *restrict C,
    unsigned n
) {
    mag_vf32_t partial0 = mag_vf32_zero();
    mag_vf32_t partial1 = mag_vf32_zero();
    mag_vf32_t partial2 = mag_vf32_zero();
    mag_vf32_t partial3 = mag_vf32_zero();
    const float *restrict row = B;
    int64_t k=0;
    int64_t inc = ldb << 2;
    for (; k+3 < K; k += 4, row += inc) {
        partial0 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 0), mag_vf32_loadu_masked(row + 0*ldb, n), partial0);
        partial1 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 1), mag_vf32_loadu_masked(row + 1*ldb, n), partial1);
        partial2 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 2), mag_vf32_loadu_masked(row + 2*ldb, n), partial2);
        partial3 = mag_vf32_fmadd(mag_vf32_broadcast(A + k + 3), mag_vf32_loadu_masked(row + 3*ldb, n), partial3);
    }
    mag_vf32_t acc = mag_vf32_add(mag_vf32_add(partial0, partial1), mag_vf32_add(partial2, partial3));
    for (; k < K; ++k, row += ldb)
        acc = mag_vf32_fmadd(mag_vf32_broadcast(A + k), mag_vf32_loadu_masked(row, n), acc);
    mag_vf32_storeu_masked(C, acc, n);
}

static MAG_AINLINE MAG_HOTPROC void mag_gemv_f32(
    int64_t K,
    int64_t N,
    const float *restrict A,
    const float *restrict B,
    int64_t ldb,
    float *restrict C
) {
    int64_t j=0;
    int64_t step = 4*MAG_VF32_LANES;
    for (; j + step-1 < N; j += step) {
        mag_gemv_vf32_f32(K, A, B + j + 0*MAG_VF32_LANES, ldb, C + j + 0*MAG_VF32_LANES);
        mag_gemv_vf32_f32(K, A, B + j + 1*MAG_VF32_LANES, ldb, C + j + 1*MAG_VF32_LANES);
        mag_gemv_vf32_f32(K, A, B + j + 2*MAG_VF32_LANES, ldb, C + j + 2*MAG_VF32_LANES);
        mag_gemv_vf32_f32(K, A, B + j + 3*MAG_VF32_LANES, ldb, C + j + 3*MAG_VF32_LANES);
    }
    for (; j+MAG_VF32_LANES-1 < N; j += MAG_VF32_LANES)
        mag_gemv_vf32_f32(K, A, B + j, ldb, C + j);
    if (j < N) {
        unsigned rem = (unsigned)(N - j);
        mag_gemv_vf32_masked_f32(K, A, B + j, ldb, C + j, rem);
    }
}
