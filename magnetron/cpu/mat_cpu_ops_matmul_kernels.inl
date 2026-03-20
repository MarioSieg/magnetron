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
    mag_vf32_8_t p0[MAG_MR], p1[MAG_MR], p2[MAG_MR], p3[MAG_MR];
    mag_vf32_8_t z8 = mag_vf32_8_zero();

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r) {
        if (do_acc)
            p0[r] = mag_vf32_loadu_8(c + r*ldc);
        else
            p0[r] = z8;
        p1[r] = p2[r] = p3[r] = z8;
    }

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1)*lda);
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2)*lda);
        }

        mag_vf32_8_t b0 = mag_vf32_loadu_8(b + (k + 0)*ldb);
        mag_vf32_8_t b1 = mag_vf32_loadu_8(b + (k + 1)*ldb);
        mag_vf32_8_t b2 = mag_vf32_loadu_8(b + (k + 2)*ldb);
        mag_vf32_8_t b3 = mag_vf32_loadu_8(b + (k + 3)*ldb);

        const float *a0 = a + (k + 0)*lda;
        const float *a1 = a + (k + 1)*lda;
        const float *a2 = a + (k + 2)*lda;
        const float *a3 = a + (k + 3)*lda;

#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
        /* Load 8 A row values once, then broadcast each lane (row) from register. */
        mag_vf32_t Av0 = mag_vf32_loadu_8(a0);
        mag_vf32_t Av1 = mag_vf32_loadu_8(a1);
        mag_vf32_t Av2 = mag_vf32_loadu_8(a2);
        mag_vf32_t Av3 = mag_vf32_loadu_8(a3);
#endif

        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A0 = mag_vf32_broadcast_lane(Av0, (unsigned)r);
#else
            mag_vf32_t A0 = mag_vf32_broadcast(a0 + r);
#endif
            p0[r] = mag_vf32_8_fmadd(A0, b0, p0[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A1 = mag_vf32_broadcast_lane(Av1, (unsigned)r);
#else
            mag_vf32_t A1 = mag_vf32_broadcast(a1 + r);
#endif
            p1[r] = mag_vf32_8_fmadd(A1, b1, p1[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A2 = mag_vf32_broadcast_lane(Av2, (unsigned)r);
#else
            mag_vf32_t A2 = mag_vf32_broadcast(a2 + r);
#endif
            p2[r] = mag_vf32_8_fmadd(A2, b2, p2[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A3 = mag_vf32_broadcast_lane(Av3, (unsigned)r);
#else
            mag_vf32_t A3 = mag_vf32_broadcast(a3 + r);
#endif
            p3[r] = mag_vf32_8_fmadd(A3, b3, p3[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r) {
        p0[r] = mag_vf32_8_add(
            p0[r],
            mag_vf32_8_add(
                mag_vf32_8_add(p1[r], p2[r]),
                p3[r]
            )
        );
    }

    for (; k < kc; ++k) {
        mag_vf32_8_t bk = mag_vf32_loadu_8(b + k*ldb);
        const float *ak = a + k*lda;
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
            mag_vf32_t A = mag_vf32_broadcast(ak + r);
            switch (k & 3) {
                case 0: p0[r] = mag_vf32_8_fmadd(A, bk, p0[r]); break;
                case 1: p1[r] = mag_vf32_8_fmadd(A, bk, p1[r]); break;
                case 2: p2[r] = mag_vf32_8_fmadd(A, bk, p2[r]); break;
                default: p3[r] = mag_vf32_8_fmadd(A, bk, p3[r]); break;
            }
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        mag_vf32_storeu_8(c + r*ldc, p0[r]);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x16_f32(
    int64_t kc,
    const float *restrict a, int64_t lda,
    const float *restrict b, int64_t ldb,
    float *restrict c, int64_t ldc,
    bool do_acc
) {
    mag_vf32_16_t p0[MAG_MR], p1[MAG_MR], p2[MAG_MR], p3[MAG_MR];
    mag_vf32_16_t z16 = mag_vf32_16_zero();

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r) {
        if (do_acc) p0[r] = mag_vf32_loadu_16(c + r*ldc);
        else p0[r] = z16;
        p1[r] = p2[r] = p3[r] = z16;
    }

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1)*lda);
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2)*lda);
        }

        mag_vf32_16_t b0 = mag_vf32_loadu_16(b + (k + 0)*ldb);
        mag_vf32_16_t b1 = mag_vf32_loadu_16(b + (k + 1)*ldb);
        mag_vf32_16_t b2 = mag_vf32_loadu_16(b + (k + 2)*ldb);
        mag_vf32_16_t b3 = mag_vf32_loadu_16(b + (k + 3)*ldb);

        const float *a0 = a + (k + 0)*lda;
        const float *a1 = a + (k + 1)*lda;
        const float *a2 = a + (k + 2)*lda;
        const float *a3 = a + (k + 3)*lda;

#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
        /* Load 8 A row values once, then broadcast each lane (row) from register. */
        mag_vf32_t Av0 = mag_vf32_loadu_8(a0);
        mag_vf32_t Av1 = mag_vf32_loadu_8(a1);
        mag_vf32_t Av2 = mag_vf32_loadu_8(a2);
        mag_vf32_t Av3 = mag_vf32_loadu_8(a3);
#endif

        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A0 = mag_vf32_broadcast_lane(Av0, (unsigned)r);
#else
            mag_vf32_t A0 = mag_vf32_broadcast(a0 + r);
#endif
            p0[r] = mag_vf32_16_fmadd(A0, b0, p0[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A1 = mag_vf32_broadcast_lane(Av1, (unsigned)r);
#else
            mag_vf32_t A1 = mag_vf32_broadcast(a1 + r);
#endif
            p1[r] = mag_vf32_16_fmadd(A1, b1, p1[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A2 = mag_vf32_broadcast_lane(Av2, (unsigned)r);
#else
            mag_vf32_t A2 = mag_vf32_broadcast(a2 + r);
#endif
            p2[r] = mag_vf32_16_fmadd(A2, b2, p2[r]);
        }
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A3 = mag_vf32_broadcast_lane(Av3, (unsigned)r);
#else
            mag_vf32_t A3 = mag_vf32_broadcast(a3 + r);
#endif
            p3[r] = mag_vf32_16_fmadd(A3, b3, p3[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r) {
        p0[r] = mag_vf32_16_add(
            p0[r],
            mag_vf32_16_add(
                mag_vf32_16_add(p1[r], p2[r]),
                p3[r]
            )
        );
    }

    for (; k < kc; ++k) {
        mag_vf32_16_t bk = mag_vf32_loadu_16(b + k*ldb);
        const float *ak = a + k*lda;
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
            mag_vf32_t A = mag_vf32_broadcast(ak + r);
            switch (k & 3) {
                case 0: p0[r] = mag_vf32_16_fmadd(A, bk, p0[r]); break;
                case 1: p1[r] = mag_vf32_16_fmadd(A, bk, p1[r]); break;
                case 2: p2[r] = mag_vf32_16_fmadd(A, bk, p2[r]); break;
                default: p3[r] = mag_vf32_16_fmadd(A, bk, p3[r]); break;
            }
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        mag_vf32_storeu_16(c + r*ldc, p0[r]);
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
    mag_vf32_8_t p0 = mag_vf32_8_zero();
    mag_vf32_8_t p1 = mag_vf32_8_zero();
    mag_vf32_8_t p2 = mag_vf32_8_zero();
    mag_vf32_8_t p3 = mag_vf32_8_zero();
    mag_vf32_8_t acc = mag_vf32_8_zero();

    if (do_acc)
        acc = mag_vf32_loadu_8(c);

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if ((k & MAG_PREFETCH_GROUPS_MASK) == 0) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }

        mag_vf32_8_t b0 = mag_vf32_loadu_8(b + (k + 0)*ldb);
        mag_vf32_8_t b1 = mag_vf32_loadu_8(b + (k + 1)*ldb);
        mag_vf32_8_t b2 = mag_vf32_loadu_8(b + (k + 2)*ldb);
        mag_vf32_8_t b3 = mag_vf32_loadu_8(b + (k + 3)*ldb);

        mag_vf32_t A0 = mag_vf32_broadcast(a + k + 0);
        mag_vf32_t A1 = mag_vf32_broadcast(a + k + 1);
        mag_vf32_t A2 = mag_vf32_broadcast(a + k + 2);
        mag_vf32_t A3 = mag_vf32_broadcast(a + k + 3);

        p0 = mag_vf32_8_fmadd(A0, b0, p0);
        p1 = mag_vf32_8_fmadd(A1, b1, p1);
        p2 = mag_vf32_8_fmadd(A2, b2, p2);
        p3 = mag_vf32_8_fmadd(A3, b3, p3);
    }

    acc = mag_vf32_8_add(
        acc,
        mag_vf32_8_add(
            mag_vf32_8_add(p0, p1),
            mag_vf32_8_add(p2, p3)
        )
    );

    for (; k < kc; ++k) {
        mag_vf32_8_t bk = mag_vf32_loadu_8(b + k*ldb);
        mag_vf32_t A = mag_vf32_broadcast(a + k);
        acc = mag_vf32_8_fmadd(A, bk, acc);
    }

    mag_vf32_storeu_8(c, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x16_f32(
    int64_t kc,
    const float *restrict a,
    const float *restrict b, int64_t ldb,
    float *restrict c,
    bool do_acc
) {
    mag_vf32_16_t p0 = mag_vf32_16_zero();
    mag_vf32_16_t p1 = mag_vf32_16_zero();
    mag_vf32_16_t p2 = mag_vf32_16_zero();
    mag_vf32_16_t p3 = mag_vf32_16_zero();
    mag_vf32_16_t acc = mag_vf32_16_zero();

    if (do_acc)
        acc = mag_vf32_loadu_16(c);

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if ((k & MAG_PREFETCH_GROUPS_MASK) == 0) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }

        mag_vf32_16_t b0 = mag_vf32_loadu_16(b + (k + 0)*ldb);
        mag_vf32_16_t b1 = mag_vf32_loadu_16(b + (k + 1)*ldb);
        mag_vf32_16_t b2 = mag_vf32_loadu_16(b + (k + 2)*ldb);
        mag_vf32_16_t b3 = mag_vf32_loadu_16(b + (k + 3)*ldb);

        mag_vf32_t A0 = mag_vf32_broadcast(a + k + 0);
        mag_vf32_t A1 = mag_vf32_broadcast(a + k + 1);
        mag_vf32_t A2 = mag_vf32_broadcast(a + k + 2);
        mag_vf32_t A3 = mag_vf32_broadcast(a + k + 3);

        p0 = mag_vf32_16_fmadd(A0, b0, p0);
        p1 = mag_vf32_16_fmadd(A1, b1, p1);
        p2 = mag_vf32_16_fmadd(A2, b2, p2);
        p3 = mag_vf32_16_fmadd(A3, b3, p3);
    }

    acc = mag_vf32_16_add(
        acc,
        mag_vf32_16_add(
            mag_vf32_16_add(p0, p1),
            mag_vf32_16_add(p2, p3)
        )
    );

    for (; k < kc; ++k) {
        mag_vf32_16_t bk = mag_vf32_loadu_16(b + k*ldb);
        mag_vf32_t A = mag_vf32_broadcast(a + k);
        acc = mag_vf32_16_fmadd(A, bk, acc);
    }

    mag_vf32_storeu_16(c, acc);
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


/* == bf16 (bfloat16) == */

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x8_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    (void)lda;
    mag_vf32_8_t accv[MAG_MR];
    mag_vf32_8_t z8 = mag_vf32_8_zero();
    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        accv[r] = acc ? mag_vbf16_loadu_8_to_f32(c + r*ldc) : z8;

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1) * MAG_MR);
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2) * MAG_MR);
        }

        mag_vf32_8_t b0 = mag_vbf16_loadu_8_to_f32(b + (k + 0)*ldb);
        mag_vf32_8_t b1 = mag_vbf16_loadu_8_to_f32(b + (k + 1)*ldb);
        mag_vf32_8_t b2 = mag_vbf16_loadu_8_to_f32(b + (k + 2)*ldb);
        mag_vf32_8_t b3 = mag_vbf16_loadu_8_to_f32(b + (k + 3)*ldb);

        const mag_bfloat16_t *a0 = a + (k + 0) * MAG_MR;
        const mag_bfloat16_t *a1 = a + (k + 1) * MAG_MR;
        const mag_bfloat16_t *a2 = a + (k + 2) * MAG_MR;
        const mag_bfloat16_t *a3 = a + (k + 3) * MAG_MR;

#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
        mag_vf32_t Av0 = mag_vbf16_loadu_8_to_f32(a0);
        mag_vf32_t Av1 = mag_vbf16_loadu_8_to_f32(a1);
        mag_vf32_t Av2 = mag_vbf16_loadu_8_to_f32(a2);
        mag_vf32_t Av3 = mag_vbf16_loadu_8_to_f32(a3);
#endif

        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A0 = mag_vf32_broadcast_lane(Av0, (unsigned)r);
            mag_vf32_t A1 = mag_vf32_broadcast_lane(Av1, (unsigned)r);
            mag_vf32_t A2 = mag_vf32_broadcast_lane(Av2, (unsigned)r);
            mag_vf32_t A3 = mag_vf32_broadcast_lane(Av3, (unsigned)r);
#else
            mag_vf32_t A0 = mag_vbf16_broadcast(a0 + r);
            mag_vf32_t A1 = mag_vbf16_broadcast(a1 + r);
            mag_vf32_t A2 = mag_vbf16_broadcast(a2 + r);
            mag_vf32_t A3 = mag_vbf16_broadcast(a3 + r);
#endif
            accv[r] = mag_vf32_8_fmadd(A0, b0, accv[r]);
            accv[r] = mag_vf32_8_fmadd(A1, b1, accv[r]);
            accv[r] = mag_vf32_8_fmadd(A2, b2, accv[r]);
            accv[r] = mag_vf32_8_fmadd(A3, b3, accv[r]);
        }
    }

    for (; k < kc; ++k) {
        mag_vf32_8_t bk = mag_vbf16_loadu_8_to_f32(b + k*ldb);
        const mag_bfloat16_t *ak = a + k * MAG_MR;
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
            mag_vf32_t A = mag_vbf16_broadcast(ak + r);
            accv[r] = mag_vf32_8_fmadd(A, bk, accv[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        mag_vbf16_storeu_8_from_f32(c + r*ldc, accv[r]);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    (void)lda;
    mag_vf32_16_t accv[MAG_MR];
    mag_vf32_16_t z16 = mag_vf32_16_zero();
    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        accv[r] = acc ? mag_vbf16_loadu_16_to_f32(c + r*ldc) : z16;

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1) * MAG_MR);
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2) * MAG_MR);
        }

        mag_vf32_16_t b0 = mag_vbf16_loadu_16_to_f32(b + (k + 0)*ldb);
        mag_vf32_16_t b1 = mag_vbf16_loadu_16_to_f32(b + (k + 1)*ldb);
        mag_vf32_16_t b2 = mag_vbf16_loadu_16_to_f32(b + (k + 2)*ldb);
        mag_vf32_16_t b3 = mag_vbf16_loadu_16_to_f32(b + (k + 3)*ldb);

        const mag_bfloat16_t *a0 = a + (k + 0) * MAG_MR;
        const mag_bfloat16_t *a1 = a + (k + 1) * MAG_MR;
        const mag_bfloat16_t *a2 = a + (k + 2) * MAG_MR;
        const mag_bfloat16_t *a3 = a + (k + 3) * MAG_MR;

#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
        mag_vf32_t Av0 = mag_vbf16_loadu_8_to_f32(a0);
        mag_vf32_t Av1 = mag_vbf16_loadu_8_to_f32(a1);
        mag_vf32_t Av2 = mag_vbf16_loadu_8_to_f32(a2);
        mag_vf32_t Av3 = mag_vbf16_loadu_8_to_f32(a3);
#endif

        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
#if (MAG_VF32_LANES == 8) || (MAG_VF32_LANES == 16)
            mag_vf32_t A0 = mag_vf32_broadcast_lane(Av0, (unsigned)r);
            mag_vf32_t A1 = mag_vf32_broadcast_lane(Av1, (unsigned)r);
            mag_vf32_t A2 = mag_vf32_broadcast_lane(Av2, (unsigned)r);
            mag_vf32_t A3 = mag_vf32_broadcast_lane(Av3, (unsigned)r);
#else
            mag_vf32_t A0 = mag_vbf16_broadcast(a0 + r);
            mag_vf32_t A1 = mag_vbf16_broadcast(a1 + r);
            mag_vf32_t A2 = mag_vbf16_broadcast(a2 + r);
            mag_vf32_t A3 = mag_vbf16_broadcast(a3 + r);
#endif
            accv[r] = mag_vf32_16_fmadd(A0, b0, accv[r]);
            accv[r] = mag_vf32_16_fmadd(A1, b1, accv[r]);
            accv[r] = mag_vf32_16_fmadd(A2, b2, accv[r]);
            accv[r] = mag_vf32_16_fmadd(A3, b3, accv[r]);
        }
    }

    for (; k < kc; ++k) {
        mag_vf32_16_t bk = mag_vbf16_loadu_16_to_f32(b + k*ldb);
        const mag_bfloat16_t *ak = a + k * MAG_MR;
        #pragma GCC unroll 8
        for (int r = 0; r < MAG_MR; ++r) {
            mag_vf32_t A = mag_vbf16_broadcast(ak + r);
            accv[r] = mag_vf32_16_fmadd(A, bk, accv[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < MAG_MR; ++r)
        mag_vbf16_storeu_16_from_f32(c + r*ldc, accv[r]);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_8x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_8x16_bfloat16(kc, a,    lda, b,    ldb, c,    ldc, acc);
    mag_mm_tile_8x16_bfloat16(kc, a,    lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x8_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
    mag_vf32_8_t accv = acc ? mag_vbf16_loadu_8_to_f32(c) : mag_vf32_8_zero();

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }

        mag_vf32_8_t b0 = mag_vbf16_loadu_8_to_f32(b + (k + 0)*ldb);
        mag_vf32_8_t b1 = mag_vbf16_loadu_8_to_f32(b + (k + 1)*ldb);
        mag_vf32_8_t b2 = mag_vbf16_loadu_8_to_f32(b + (k + 2)*ldb);
        mag_vf32_8_t b3 = mag_vbf16_loadu_8_to_f32(b + (k + 3)*ldb);

        mag_vf32_t A0 = mag_vbf16_broadcast(a + (k + 0));
        mag_vf32_t A1 = mag_vbf16_broadcast(a + (k + 1));
        mag_vf32_t A2 = mag_vbf16_broadcast(a + (k + 2));
        mag_vf32_t A3 = mag_vbf16_broadcast(a + (k + 3));

        accv = mag_vf32_8_fmadd(A0, b0, accv);
        accv = mag_vf32_8_fmadd(A1, b1, accv);
        accv = mag_vf32_8_fmadd(A2, b2, accv);
        accv = mag_vf32_8_fmadd(A3, b3, accv);
    }

    for (; k < kc; ++k) {
        mag_vf32_8_t bk = mag_vbf16_loadu_8_to_f32(b + k*ldb);
        mag_vf32_t A = mag_vbf16_broadcast(a + k);
        accv = mag_vf32_8_fmadd(A, bk, accv);
    }

    mag_vbf16_storeu_8_from_f32(c, accv);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
    mag_vf32_16_t accv = acc ? mag_vbf16_loadu_16_to_f32(c) : mag_vf32_16_zero();

    int64_t k = 0;
    for (; k + 3 < kc; k += 4) {
        if (!(k & MAG_PREFETCH_GROUPS_MASK)) {
            mag_simd_prefetch_t0(b + (k + MAG_PREFETCH_RANGE_B_L1)*ldb);
            mag_simd_prefetch_t1(b + (k + MAG_PREFETCH_RANGE_B_L2)*ldb);
            mag_simd_prefetch_t0(a + (k + MAG_PREFETCH_RANGE_A_L1));
            mag_simd_prefetch_t1(a + (k + MAG_PREFETCH_RANGE_A_L2));
        }
        mag_vf32_16_t b0 = mag_vbf16_loadu_16_to_f32(b + (k + 0)*ldb);
        mag_vf32_16_t b1 = mag_vbf16_loadu_16_to_f32(b + (k + 1)*ldb);
        mag_vf32_16_t b2 = mag_vbf16_loadu_16_to_f32(b + (k + 2)*ldb);
        mag_vf32_16_t b3 = mag_vbf16_loadu_16_to_f32(b + (k + 3)*ldb);

        mag_vf32_t A0 = mag_vbf16_broadcast(a + (k + 0));
        mag_vf32_t A1 = mag_vbf16_broadcast(a + (k + 1));
        mag_vf32_t A2 = mag_vbf16_broadcast(a + (k + 2));
        mag_vf32_t A3 = mag_vbf16_broadcast(a + (k + 3));

        accv = mag_vf32_16_fmadd(A0, b0, accv);
        accv = mag_vf32_16_fmadd(A1, b1, accv);
        accv = mag_vf32_16_fmadd(A2, b2, accv);
        accv = mag_vf32_16_fmadd(A3, b3, accv);
    }

    for (; k < kc; ++k) {
        mag_vf32_16_t bk = mag_vbf16_loadu_16_to_f32(b + k*ldb);
        mag_vf32_t A = mag_vbf16_broadcast(a + k);
        accv = mag_vf32_16_fmadd(A, bk, accv);
    }

    mag_vbf16_storeu_16_from_f32(c, accv);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_1x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c,
    bool acc
) {
    mag_mm_tile_1x16_bfloat16(kc, a, b,    ldb, c,    acc);
    mag_mm_tile_1x16_bfloat16(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_16x16_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_8x16_bfloat16(kc, a,         lda, b,    ldb, c,         ldc, acc);
    mag_mm_tile_8x16_bfloat16(kc, a + 8*lda, lda, b,    ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_tile_16x32_bfloat16(
    int64_t kc,
    const mag_bfloat16_t *restrict a, ptrdiff_t lda,
    const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
    mag_bfloat16_t *restrict c, ptrdiff_t ldc,
    bool acc
) {
    mag_mm_tile_16x16_bfloat16(kc, a, lda, b,    ldb, c,    ldc, acc);
    mag_mm_tile_16x16_bfloat16(kc, a, lda, b+16, ldb, c+16, ldc, acc);
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

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_A_mc_kc_panel8_float32(
    int64_t kc, int64_t mr,
    const float *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx,
    float *restrict pa
) {
    int64_t m8 = mr & ~7;

    for (int64_t i = 0; i < m8; i += 8) {
        const float *p0 = ra + (i + 0) * sMx;
        const float *p1 = ra + (i + 1) * sMx;
        const float *p2 = ra + (i + 2) * sMx;
        const float *p3 = ra + (i + 3) * sMx;
        const float *p4 = ra + (i + 4) * sMx;
        const float *p5 = ra + (i + 5) * sMx;
        const float *p6 = ra + (i + 6) * sMx;
        const float *p7 = ra + (i + 7) * sMx;
        float *dst = pa + i * kc;

        for (int64_t k = 0; k < kc; ++k) {
            dst[k * 8 + 0] = p0[0];
            dst[k * 8 + 1] = p1[0];
            dst[k * 8 + 2] = p2[0];
            dst[k * 8 + 3] = p3[0];
            dst[k * 8 + 4] = p4[0];
            dst[k * 8 + 5] = p5[0];
            dst[k * 8 + 6] = p6[0];
            dst[k * 8 + 7] = p7[0];
            p0 += sKx; p1 += sKx; p2 += sKx; p3 += sKx;
            p4 += sKx; p5 += sKx; p6 += sKx; p7 += sKx;
        }
    }

    for (int64_t i = m8; i < mr; ++i) {
        const float *src = ra + i * sMx;
        float *dst = pa + i * kc;
        for (int64_t k = 0; k < kc; ++k) {
            dst[k] = src[0];
            src += sKx;
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

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_B_kc_nc_bfloat16(
    int64_t kc, int64_t nc,
    const mag_bfloat16_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN,
    mag_bfloat16_t *restrict Bp
) {
    for (int64_t k = 0; k < kc; ++k) {
        const mag_bfloat16_t *restrict src = Bsrc + k * strideK;
        mag_bfloat16_t *restrict dst = Bp + k * nc;
        if (strideN == 1) {
            int64_t j = 0;
            for (; j + MAG_VF32_LANES <= nc; j += MAG_VF32_LANES) {
                mag_vbf16_t v = mag_vbf16_loadu(src + j);
                mag_vbf16_storeu(dst + j, v);
            }
            for (; j < nc; ++j) dst[j] = src[j];
        } else {
            for (int64_t j = 0; j < nc; ++j)
                dst[j] = src[j * strideN];
        }
    }
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_B_vec_bfloat16(
    int64_t kc, int64_t nc,
    const mag_bfloat16_t *restrict yvec,
    mag_bfloat16_t *restrict bp
) {
    for (int64_t k = 0; k < kc; ++k) {
        const mag_bfloat16_t v = yvec[k];
        mag_bfloat16_t *restrict dst = bp + k * nc;
        mag_vbf16_t vv = mag_vbf16_splat(v);
        int64_t j = 0;
        for (; j + MAG_VF32_LANES <= nc; j += MAG_VF32_LANES)
            mag_vbf16_storeu(dst + j, vv);
        for (; j < nc; ++j)
            dst[j] = v;
    }
}

static MAG_AINLINE MAG_HOTPROC void mag_mm_pack_A_mc_kc_panel8_bfloat16(
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

        for (int64_t k = 0; k < kc; ++k) {
            dst[k*8 + 0] = p0[0];
            dst[k*8 + 1] = p1[0];
            dst[k*8 + 2] = p2[0];
            dst[k*8 + 3] = p3[0];
            dst[k*8 + 4] = p4[0];
            dst[k*8 + 5] = p5[0];
            dst[k*8 + 6] = p6[0];
            dst[k*8 + 7] = p7[0];
            p0 += sKx; p1 += sKx; p2 += sKx; p3 += sKx;
            p4 += sKx; p5 += sKx; p6 += sKx; p7 += sKx;
        }
    }

    for (int64_t i = m8; i < mr; ++i) {
        const mag_bfloat16_t *src = ra + i * sMx;
        mag_bfloat16_t *dst = pa + i * kc;
        for (int64_t k = 0; k < kc; ++k)
            dst[k] = src[k * sKx];
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
