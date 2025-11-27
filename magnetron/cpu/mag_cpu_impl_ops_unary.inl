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

#define mag_gen_stub_clone(T) \
    static MAG_HOTPROC void mag_clone_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        if (mag_full_cont2(r, x)) { \
            memcpy(br, bx, mag_tensor_get_data_size(r)); \
            return; \
        } \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        int64_t numel = r->numel; \
        for (int64_t i=0; i < numel; ++i) { \
            int64_t ri, xi; \
            mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = bx[xi]; \
        } \
    }

mag_gen_stub_clone(e8m23)
mag_gen_stub_clone(e5m10)
mag_gen_stub_clone(bool)
mag_gen_stub_clone(u8)
mag_gen_stub_clone(i8)
mag_gen_stub_clone(u16)
mag_gen_stub_clone(i16)
mag_gen_stub_clone(u32)
mag_gen_stub_clone(i32)
mag_gen_stub_clone(u64)
mag_gen_stub_clone(i64)

#undef mag_gen_stub_clone

#define mag_gen_stub_unary(T, FUNC) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont2(x, r)) { \
            mag_v##FUNC##_##T(rb-ra, br+ra, bx+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi; \
            mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            mag_v##FUNC##_##T(1, br+ri, bx+xi); \
        } \
    }

mag_gen_stub_unary(e8m23, abs)
mag_gen_stub_unary(e5m10, abs)
mag_gen_stub_unary(e8m23, sgn)
mag_gen_stub_unary(e5m10, sgn)
mag_gen_stub_unary(e8m23, neg)
mag_gen_stub_unary(e5m10, neg)
mag_gen_stub_unary(e8m23, log)
mag_gen_stub_unary(e5m10, log)
mag_gen_stub_unary(e8m23, log10)
mag_gen_stub_unary(e5m10, log10)
mag_gen_stub_unary(e8m23, log1p)
mag_gen_stub_unary(e5m10, log1p)
mag_gen_stub_unary(e8m23, log2)
mag_gen_stub_unary(e5m10, log2)
mag_gen_stub_unary(e8m23, sqr)
mag_gen_stub_unary(e5m10, sqr)
mag_gen_stub_unary(e8m23, sqrt)
mag_gen_stub_unary(e5m10, sqrt)
mag_gen_stub_unary(e8m23, sin)
mag_gen_stub_unary(e5m10, sin)
mag_gen_stub_unary(e8m23, cos)
mag_gen_stub_unary(e5m10, cos)
mag_gen_stub_unary(e8m23, tan)
mag_gen_stub_unary(e5m10, tan)
mag_gen_stub_unary(e8m23, asin)
mag_gen_stub_unary(e5m10, asin)
mag_gen_stub_unary(e8m23, acos)
mag_gen_stub_unary(e5m10, acos)
mag_gen_stub_unary(e8m23, atan)
mag_gen_stub_unary(e5m10, atan)
mag_gen_stub_unary(e8m23, sinh)
mag_gen_stub_unary(e5m10, sinh)
mag_gen_stub_unary(e8m23, cosh)
mag_gen_stub_unary(e5m10, cosh)
mag_gen_stub_unary(e8m23, tanh)
mag_gen_stub_unary(e5m10, tanh)
mag_gen_stub_unary(e8m23, asinh)
mag_gen_stub_unary(e5m10, asinh)
mag_gen_stub_unary(e8m23, acosh)
mag_gen_stub_unary(e5m10, acosh)
mag_gen_stub_unary(e8m23, atanh)
mag_gen_stub_unary(e5m10, atanh)
mag_gen_stub_unary(e8m23, step)
mag_gen_stub_unary(e5m10, step)
mag_gen_stub_unary(e8m23, erf)
mag_gen_stub_unary(e5m10, erf)
mag_gen_stub_unary(e8m23, erfc)
mag_gen_stub_unary(e5m10, erfc)
mag_gen_stub_unary(e8m23, exp)
mag_gen_stub_unary(e5m10, exp)
mag_gen_stub_unary(e8m23, exp2)
mag_gen_stub_unary(e5m10, exp2)
mag_gen_stub_unary(e8m23, expm1)
mag_gen_stub_unary(e5m10, expm1)
mag_gen_stub_unary(e8m23, floor)
mag_gen_stub_unary(e5m10, floor)
mag_gen_stub_unary(e8m23, ceil)
mag_gen_stub_unary(e5m10, ceil)
mag_gen_stub_unary(e8m23, round)
mag_gen_stub_unary(e5m10, round)
mag_gen_stub_unary(e8m23, trunc)
mag_gen_stub_unary(e5m10, trunc)
mag_gen_stub_unary(e8m23, softmax_dv)
mag_gen_stub_unary(e5m10, softmax_dv)
mag_gen_stub_unary(e8m23, sigmoid)
mag_gen_stub_unary(e5m10, sigmoid)
mag_gen_stub_unary(e8m23, sigmoid_dv)
mag_gen_stub_unary(e5m10, sigmoid_dv)
mag_gen_stub_unary(e8m23, hard_sigmoid)
mag_gen_stub_unary(e5m10, hard_sigmoid)
mag_gen_stub_unary(e8m23, silu)
mag_gen_stub_unary(e5m10, silu)
mag_gen_stub_unary(e8m23, silu_dv)
mag_gen_stub_unary(e5m10, silu_dv)
mag_gen_stub_unary(e8m23, tanh_dv)
mag_gen_stub_unary(e5m10, tanh_dv)
mag_gen_stub_unary(e8m23, relu)
mag_gen_stub_unary(e5m10, relu)
mag_gen_stub_unary(e8m23, relu_dv)
mag_gen_stub_unary(e5m10, relu_dv)
mag_gen_stub_unary(e8m23, gelu)
mag_gen_stub_unary(e5m10, gelu)
mag_gen_stub_unary(e8m23, gelu_approx)
mag_gen_stub_unary(e5m10, gelu_approx)
mag_gen_stub_unary(e8m23, gelu_dv)
mag_gen_stub_unary(e5m10, gelu_dv)
mag_gen_stub_unary(bool, not)
mag_gen_stub_unary(u8, not)
mag_gen_stub_unary(i8, not)
mag_gen_stub_unary(u16, not)
mag_gen_stub_unary(i16, not)
mag_gen_stub_unary(u32, not)
mag_gen_stub_unary(i32, not)
mag_gen_stub_unary(u64, not)
mag_gen_stub_unary(i64, not)

#undef mag_gen_stub_unary

static void MAG_HOTPROC mag_softmax_e8m23(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_e8m23_t *br = mag_e8m23p_mut(r);
    const mag_e8m23_t *bx = mag_e8m23p(x);
    int64_t last_dim = r->coords.shape[r->coords.rank-1];
    int64_t num_rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rows_per_thread = (num_rows + tc - 1)/tc;
    int64_t start_row = ti*rows_per_thread;
    int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
    for (int64_t row = start_row; row < end_row; ++row) {
        const mag_e8m23_t *row_in = bx + row*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
        mag_e8m23_t *row_out = br + row*last_dim;
        mag_e8m23_t max_val = row_in[0]; /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            if (row_in[i] > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = row_in[i];
            }
        }
        mag_e8m23_t sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
            row_out[i] = expf(row_in[i] - max_val); /* -max for numerical stability */
            sum += row_out[i];
        }
        for (int64_t i=0; i < last_dim; ++i) {
            row_out[i] /= sum;
        }
    }
}

static void MAG_HOTPROC mag_softmax_e5m10(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_e5m10_t *br = mag_e5m10p_mut(r);
    const mag_e5m10_t *bx = mag_e5m10p(x);
    int64_t last_dim = r->coords.shape[r->coords.rank-1];
    int64_t num_rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rows_per_thread = (num_rows + tc - 1)/tc;
    int64_t start_row = ti*rows_per_thread;
    int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
    for (int64_t row = start_row; row < end_row; ++row) {
        const mag_e5m10_t *row_in = bx + row*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
        mag_e5m10_t *row_out = br + row*last_dim;
        mag_e8m23_t max_val = mag_e5m10_to_e8m23(row_in[0]);  /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            mag_e8m23_t fp32_row = mag_e5m10_to_e8m23(row_in[i]);
            if (fp32_row > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = fp32_row;
            }
        }
        mag_e8m23_t sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
            mag_e8m23_t fp32_row = mag_e5m10_to_e8m23(row_in[i]);
            mag_e8m23_t exp = expf(fp32_row - max_val);
            row_out[i] = mag_e8m23_to_e5m10(exp); /* -max for numerical stability */
            sum += exp;
        }
        for (int64_t i=0; i < last_dim; ++i) {
            row_out[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(row_out[i]) / sum);
        }
    }
}
