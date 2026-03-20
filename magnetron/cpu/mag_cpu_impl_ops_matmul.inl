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

