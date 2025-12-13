/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#define mag_gen_stub_clone(T, TF) \
    static MAG_HOTPROC void mag_clone_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total+tc-1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra+chunk, total); \
        if (mag_unlikely(rb <= ra)) return; \
        if (mag_full_cont2(r, x)) { \
            memcpy(br+ra, bx+ra, (rb-ra)*sizeof(T)); \
            return; \
        } \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi; \
            mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            br[ri] = bx[xi]; \
        } \
    }

mag_gen_stub_clone(float, float32)
mag_gen_stub_clone(mag_float16_t, float16)
mag_gen_stub_clone(uint8_t, uint8)
mag_gen_stub_clone(int8_t, int8)
mag_gen_stub_clone(uint16_t, uint16)
mag_gen_stub_clone(int16_t, int16)
mag_gen_stub_clone(uint32_t, uint32)
mag_gen_stub_clone(int32_t, int32)
mag_gen_stub_clone(uint64_t, uint64)
mag_gen_stub_clone(int64_t, int64)

#undef mag_gen_stub_clone

#define mag_gen_stub_unary(T, TF, FUNC) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont2(x, r)) { \
            mag_v##FUNC##_##TF(rb-ra, br+ra, bx+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi; \
            mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            mag_v##FUNC##_##TF(1, br+ri, bx+xi); \
        } \
    }

mag_gen_stub_unary(float, float32, abs)
mag_gen_stub_unary(mag_float16_t, float16, abs)

mag_gen_stub_unary(float, float32, sgn)
mag_gen_stub_unary(mag_float16_t, float16, sgn)

mag_gen_stub_unary(float, float32, neg)
mag_gen_stub_unary(mag_float16_t, float16, neg)

mag_gen_stub_unary(float, float32, log)
mag_gen_stub_unary(mag_float16_t, float16, log)

mag_gen_stub_unary(float, float32, log10)
mag_gen_stub_unary(mag_float16_t, float16, log10)

mag_gen_stub_unary(float, float32, log1p)
mag_gen_stub_unary(mag_float16_t, float16, log1p)

mag_gen_stub_unary(float, float32, log2)
mag_gen_stub_unary(mag_float16_t, float16, log2)

mag_gen_stub_unary(float, float32, sqr)
mag_gen_stub_unary(mag_float16_t, float16, sqr)

mag_gen_stub_unary(float, float32, rcp)
mag_gen_stub_unary(mag_float16_t, float16, rcp)

mag_gen_stub_unary(float, float32, sqrt)
mag_gen_stub_unary(mag_float16_t, float16, sqrt)

mag_gen_stub_unary(float, float32, rsqrt)
mag_gen_stub_unary(mag_float16_t, float16, rsqrt)

mag_gen_stub_unary(float, float32, sin)
mag_gen_stub_unary(mag_float16_t, float16, sin)

mag_gen_stub_unary(float, float32, cos)
mag_gen_stub_unary(mag_float16_t, float16, cos)

mag_gen_stub_unary(float, float32, tan)
mag_gen_stub_unary(mag_float16_t, float16, tan)

mag_gen_stub_unary(float, float32, asin)
mag_gen_stub_unary(mag_float16_t, float16, asin)

mag_gen_stub_unary(float, float32, acos)
mag_gen_stub_unary(mag_float16_t, float16, acos)

mag_gen_stub_unary(float, float32, atan)
mag_gen_stub_unary(mag_float16_t, float16, atan)

mag_gen_stub_unary(float, float32, sinh)
mag_gen_stub_unary(mag_float16_t, float16, sinh)

mag_gen_stub_unary(float, float32, cosh)
mag_gen_stub_unary(mag_float16_t, float16, cosh)

mag_gen_stub_unary(float, float32, tanh)
mag_gen_stub_unary(mag_float16_t, float16, tanh)

mag_gen_stub_unary(float, float32, asinh)
mag_gen_stub_unary(mag_float16_t, float16, asinh)

mag_gen_stub_unary(float, float32, acosh)
mag_gen_stub_unary(mag_float16_t, float16, acosh)

mag_gen_stub_unary(float, float32, atanh)
mag_gen_stub_unary(mag_float16_t, float16, atanh)

mag_gen_stub_unary(float, float32, step)
mag_gen_stub_unary(mag_float16_t, float16, step)

mag_gen_stub_unary(float, float32, erf)
mag_gen_stub_unary(mag_float16_t, float16, erf)

mag_gen_stub_unary(float, float32, erfc)
mag_gen_stub_unary(mag_float16_t, float16, erfc)

mag_gen_stub_unary(float, float32, exp)
mag_gen_stub_unary(mag_float16_t, float16, exp)

mag_gen_stub_unary(float, float32, exp2)
mag_gen_stub_unary(mag_float16_t, float16, exp2)

mag_gen_stub_unary(float, float32, expm1)
mag_gen_stub_unary(mag_float16_t, float16, expm1)

mag_gen_stub_unary(float, float32, floor)
mag_gen_stub_unary(mag_float16_t, float16, floor)

mag_gen_stub_unary(float, float32, ceil)
mag_gen_stub_unary(mag_float16_t, float16, ceil)

mag_gen_stub_unary(float, float32, round)
mag_gen_stub_unary(mag_float16_t, float16, round)

mag_gen_stub_unary(float, float32, trunc)
mag_gen_stub_unary(mag_float16_t, float16, trunc)

mag_gen_stub_unary(float, float32, softmax_dv)
mag_gen_stub_unary(mag_float16_t, float16, softmax_dv)

mag_gen_stub_unary(float, float32, sigmoid)
mag_gen_stub_unary(mag_float16_t, float16, sigmoid)

mag_gen_stub_unary(float, float32, sigmoid_dv)
mag_gen_stub_unary(mag_float16_t, float16, sigmoid_dv)

mag_gen_stub_unary(float, float32, hard_sigmoid)
mag_gen_stub_unary(mag_float16_t, float16, hard_sigmoid)

mag_gen_stub_unary(float, float32, silu)
mag_gen_stub_unary(mag_float16_t, float16, silu)

mag_gen_stub_unary(float, float32, silu_dv)
mag_gen_stub_unary(mag_float16_t, float16, silu_dv)

mag_gen_stub_unary(float, float32, tanh_dv)
mag_gen_stub_unary(mag_float16_t, float16, tanh_dv)

mag_gen_stub_unary(float, float32, relu)
mag_gen_stub_unary(mag_float16_t, float16, relu)

mag_gen_stub_unary(float, float32, relu_dv)
mag_gen_stub_unary(mag_float16_t, float16, relu_dv)

mag_gen_stub_unary(float, float32, gelu)
mag_gen_stub_unary(mag_float16_t, float16, gelu)

mag_gen_stub_unary(float, float32, gelu_approx)
mag_gen_stub_unary(mag_float16_t, float16, gelu_approx)

mag_gen_stub_unary(float, float32, gelu_dv)
mag_gen_stub_unary(mag_float16_t, float16, gelu_dv)

mag_gen_stub_unary(uint8_t, uint8, not)
mag_gen_stub_unary(int8_t, int8, not)

mag_gen_stub_unary(uint16_t, uint16, not)
mag_gen_stub_unary(int16_t, int16, not)

mag_gen_stub_unary(uint32_t, uint32, not)
mag_gen_stub_unary(int32_t, int32, not)

mag_gen_stub_unary(uint64_t, uint64, not)
mag_gen_stub_unary(int64_t, int64, not)

#undef mag_gen_stub_unary

static void MAG_HOTPROC mag_softmax_float32(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
    float *br = (float *)mag_tensor_data_ptr_mut(r);
    const float *bx = (const float *)mag_tensor_data_ptr(x);
    int64_t rank  = r->coords.rank;
    int64_t numel = r->numel;
    if (mag_unlikely(!numel)) return;
    if (rank == 0) {
        if (payload->thread_idx == 0) *br = 1.0f;
        return;
    }
    int64_t last_dim = r->coords.shape[rank-1];
    int64_t rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rpt = (rows + tc - 1)/tc;
    int64_t ra = ti*rpt;
    int64_t rb = mag_xmin(ra + rpt, rows);
    for (int64_t ri=ra; ri < rb; ++ri) {
        const float *row_in = bx + ri*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_numbytes(x));
        float *row_out = br + ri*last_dim;
        float max_val = row_in[0]; /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            if (row_in[i] > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
                max_val = row_in[i];
            }
        }
        double sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_numbytes(r));
            row_out[i] = expf(row_in[i] - max_val); /* -max for numerical stability */
            sum += row_out[i];
        }
        if (!isfinite(sum) || sum <= 0.0) {
            float inv = 1.0f / (float)last_dim;
            for (int64_t i=0; i < last_dim; ++i) row_out[i] = inv;
        } else {
            float inv = (float)(1.0 / sum);
            for (int64_t i=0; i < last_dim; ++i) row_out[i] *= inv;
        }
    }
}

static void MAG_HOTPROC mag_softmax_float16(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
    mag_float16_t *br = (mag_float16_t *)mag_tensor_data_ptr_mut(r);
    const mag_float16_t *bx = (const mag_float16_t *)mag_tensor_data_ptr(x);
    int64_t rank  = r->coords.rank;
    int64_t numel = r->numel;
    if (mag_unlikely(!numel)) return;
    if (rank == 0) {
        if (payload->thread_idx == 0) *br = MAG_FLOAT16_ONE;
        return;
    }
    int64_t last_dim = r->coords.shape[r->coords.rank-1];
    int64_t rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rpt = (rows + tc - 1)/tc;
    int64_t ra = ti*rpt;
    int64_t rb = mag_xmin(ra + rpt, rows);
    for (int64_t ri=ra; ri < rb; ++ri) {
        const mag_float16_t *row_in = bx + ri*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_numbytes(x));
        mag_float16_t *row_out = br + ri*last_dim;
        float max_val = mag_float16_to_float32(row_in[0]); /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            if (mag_float16_to_float32(row_in[i]) > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
                max_val = mag_float16_to_float32(row_in[i]);
            }
        }
        double sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_numbytes(r));
            float ro = expf(mag_float16_to_float32(row_in[i]) - max_val);
            row_out[i] = mag_float32_to_float16(ro); /* -max for numerical stability */
            sum += ro;
        }
        if (!isfinite(sum) || sum <= 0.0) {
            mag_float16_t inv = mag_float32_to_float16(1.0f / (float)last_dim);
            for (int64_t i=0; i < last_dim; ++i) row_out[i] = inv;
        } else {
            float inv = (float)(1.0 / sum);
            for (int64_t i=0; i < last_dim; ++i) row_out[i] = mag_float32_to_float16(mag_float16_to_float32(row_out[i]) * inv);
        }
    }
}
