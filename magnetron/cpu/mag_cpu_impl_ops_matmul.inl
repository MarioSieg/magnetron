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

#define mag_convert_nop(x) (x)

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

#define mag_matmul_impl_inner_block_kernel(T, TN, CVTT, CVTB) \
    static MAG_HOTPROC void mag_mm_block_##TN(int64_t kc, int64_t mr, int64_t nr, const T *restrict A, const T *restrict B, int64_t ldb, T *restrict C, int64_t ldc, bool do_acc) { \
        int64_t j = 0; \
        int64_t m8 = mr&~7; \
        for (; nr-j >= 32; j += 32) { \
            int64_t i=0; \
            for (; i+15 < m8; i += 16) \
                mag_mm_tile_16x32_##TN(kc, A + i*kc, B + j, ldb, C + i*ldc + j, ldc, do_acc); \
            for (; i+7 < m8; i += 8) \
                mag_mm_tile_8x32_##TN(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, do_acc); \
            for (; i < mr; ++i) \
                mag_mm_tile_1x32_##TN(kc, A + i*kc, B + j, ldb, C + i*ldc + j, do_acc); \
        } \
        for (; nr-j >= 16; j += 16) { \
            int64_t i=0; \
            for (; i+7 < m8; i += 8) \
                mag_mm_tile_8x16_##TN(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, do_acc); \
            for (; i < mr; ++i) \
                mag_mm_tile_1x16_##TN(kc, A + i*kc, B + j, ldb, C + i*ldc + j, do_acc); \
        } \
        for (; nr-j >= 8; j += 8) { \
            int64_t i=0; \
            for (; i+7 < m8; i += 8) \
                mag_mm_tile_8x8_##TN(kc, A + i*kc, MAG_MR, B + j, ldb, C + i*ldc + j, ldc, do_acc); \
            for (; i < mr; ++i) \
                mag_mm_tile_1x8_##TN(kc, A + i*kc, B + j, ldb, C + i*ldc + j, do_acc); \
        } \
        int64_t rem = nr - j; \
        if (mag_likely(!rem)) return; \
        for (int64_t i2 = 0; i2 < mr; ++i2) { \
            T *restrict cp = C + i2*ldc + j; \
            if (i2 < m8) { \
                int64_t panel = i2>>3; \
                int64_t r = i2&7; \
                const T *restrict ap = A + panel*(kc<<3); \
                for (int64_t jj = 0; jj < rem; ++jj) { \
                    float sum = do_acc ? CVTT(cp[jj]) : 0.0f; \
                    for (int64_t k = 0; k < kc; ++k) \
                        sum += CVTT(ap[k*MAG_MR + r])*CVTT(B[k*ldb + (j + jj)]); \
                    cp[jj] = CVTB(sum); \
                } \
            } else { \
                const T *restrict ap = A + i2*kc; \
                for (int64_t jj=0; jj < rem; ++jj) { \
                    float sum = do_acc ? CVTT(cp[jj]) : 0.0f; \
                    for (int64_t k = 0; k < kc; ++k) \
                        sum += CVTT(ap[k])*CVTT(B[k*ldb + (j + jj)]); \
                    cp[jj] = CVTB(sum); \
                } \
            } \
        } \
    } \

mag_matmul_impl_inner_block_kernel(float, f32, mag_convert_nop, mag_convert_nop)
mag_matmul_impl_inner_block_kernel(mag_bfloat16_t, bf16, mag_bfloat16_to_float32, mag_float32_to_bfloat16)

#undef mag_matmul_impl_inner_block_kernel

#define mag_matmul_impl_outer_kernel_skeleton(T, TN) \
    static MAG_HOTPROC void mag_matmul_##TN(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        const T *by = (const T *)mag_tensor_data_ptr(y); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        int64_t MR = payload->mm_params.MR; \
        int64_t MC = payload->mm_params.MC; \
        int64_t KC = payload->mm_params.KC; \
        int64_t NC = payload->mm_params.NC; \
        int64_t NR = payload->mm_params.NR; \
        int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2]; \
        int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1]; \
        int64_t K = x->coords.shape[x->coords.rank-1]; \
        int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0; \
        int64_t batch_total = 1; \
        for (int64_t d=0; d < bdr; ++d) \
            batch_total *= r->coords.shape[d]; \
        if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { \
            int64_t nth = payload->thread_num; \
            int64_t tid = payload->thread_idx; \
            int64_t jpt = (N+nth-1)/nth; \
            int64_t j0 = tid*jpt; \
            int64_t j1 = mag_xmin(N, j0 + jpt); \
            for (int64_t batch = 0; batch < batch_total; ++batch) { \
                const T *A = bx + mag_offset_rmn(x, batch, 0, 0); \
                const T *B = by + mag_offset_rmn(y, batch, 0, 0) + j0; \
                T *C = br + mag_offset_rmn(r, batch, 0, 0) + j0; \
                mag_gemv_##TN(K, j1 - j0, A, B, N, C); \
            } \
            return; \
        } \
        int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0; \
        int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0; \
        int64_t tic = (M+MC-1)/MC; \
        int64_t tjc = (N+NC-1)/NC; \
        int64_t tpb = tic*tjc; \
        int64_t tt = batch_total*tpb; \
        mag_scratch_arena_clear(&mag_tls_arena); \
        T *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC)); \
        T *Bp = scratch; \
        T *Ap = Bp + KC*NC; \
        for (;;) { \
            int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED); \
            if (tile >= tt) break; \
            int64_t batch_idx = tile / tpb; \
            int64_t rem = tile % tpb; \
            int64_t jc = rem % tjc; \
            int64_t ic = rem / tjc; \
            int64_t idx_r[MAG_MAX_DIMS] = {0}; \
            for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) { \
                idx_r[d] = t % r->coords.shape[d]; \
                t /= r->coords.shape[d]; \
            } \
            int64_t xb_flat = 0; \
            for (int64_t d=0; d < bdx; ++d) { \
                int64_t rd = bdr - bdx + d; \
                xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]); \
            } \
            int64_t yb_flat = 0; \
            for (int64_t d=0; d < bdy; ++d) { \
                int64_t rd = bdr - bdy + d; \
                yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]); \
            } \
            bool yv = y->coords.rank == 1; \
            const T *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0); \
            const T *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0); \
            T *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0); \
            int64_t i0 = ic*MC; \
            int64_t mc = i0+MC <= M ? MC : M-i0; \
            int64_t j0 = jc*NC; \
            int64_t nc = j0+NC <= N ? NC : N-j0; \
            int64_t sMx = x->coords.strides[x->coords.rank-2]; \
            int64_t sKx = x->coords.strides[x->coords.rank-1]; \
            int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2]; \
            int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1]; \
            for (int64_t pc = 0; pc < K; pc += KC) { \
                int64_t kc = mag_xmin(KC, K - pc); \
                if (y->coords.rank == 1) { \
                    mag_mm_pack_B_vec_##TN(kc, nc, py_base + pc, Bp); \
                } else { \
                    mag_mm_pack_B_kc_nc_##TN(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp); \
                } \
                mag_mm_pack_A_mc_kc_panel8_##TN(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap); \
                for (int64_t ir=0; ir < mc; ir += MR) { \
                    for (int64_t jr=0; jr < nc; jr += NR) { \
                        int64_t mr = mag_xmin(MR, mc - ir); \
                        int64_t nr = mag_xmin(NR, nc - jr); \
                        const T *restrict A = Ap + ir*kc; \
                        const T *restrict B = Bp + jr; \
                        T *restrict C = pr_base + (i0 + ir)*N + (j0 + jr); \
                        mag_mm_block_##TN(kc, mr, nr, A, B, nc, C, N, pc != 0); \
                    } \
                } \
            } \
        } \
        mag_scratch_arena_clear(&mag_tls_arena); \
    }


mag_matmul_impl_outer_kernel_skeleton(float, f32)
mag_matmul_impl_outer_kernel_skeleton(mag_bfloat16_t, bf16)

#undef mag_matmul_impl_outer_kernel_skeleton

// eww TODO:

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
    int64_t work_total = batch_total*M;
    for (int64_t work = tid; work < work_total; work += nth) {
        int64_t b = work / M;
        int64_t i = work - b*M;
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
            xb_flat = xb_flat*x->coords.shape[d] + idx;
        }
        int64_t yb_flat = 0;
        for (int64_t d = 0; d < bdy; ++d) {
            int64_t rd  = bdr - bdy + d;
            int64_t idx = (y->coords.shape[d] == 1) ? 0 : idx_r[rd];
            yb_flat = yb_flat*y->coords.shape[d] + idx;
        }
        int64_t rb_flat = 0;
        for (int64_t d = 0; d < bdr; ++d)
            rb_flat = rb_flat*r->coords.shape[d] + idx_r[d];
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float ax = mag_load_x_f16_as_f32(x, bx, xb_flat, i, k);
                const float byv = mag_load_y_f16_as_f32(y, by, yb_flat, k, n);
                sum += ax*byv;
            }
            mag_store_r_f16_from_f32(r, br, rb_flat, i, n, sum);
        }
    }
}

