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

#define mag_G(x) (x)                    /* Get scalar value */
#define mag_G_underlying(x) (x.bits)    /* Get underlying storage scalar */

#define mag_gen_stub_fill(T, G, UT, CVT) \
    static MAG_HOTPROC void mag_fill_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        mag_##T##_t val = CVT(mag_op_attr_unwrap_##UT(mag_cmd_attr(0))); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        int64_t numel = r->numel; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=0; ri < numel; ++ri) { \
                mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
                br[ri] = val; \
            } \
            return; \
        } \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        for (int64_t i=0; i < numel; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = val; \
        } \
    }

mag_gen_stub_fill(e8m23, mag_G, e8m23, mag_cvt_nop)
mag_gen_stub_fill(e5m10, mag_G_underlying, e8m23, mag_e8m23_to_e5m10)
mag_gen_stub_fill(bool, mag_G, i64, mag_cvt_i642bool)
mag_gen_stub_fill(u8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u64, mag_G, i64, mag_cvt_nop)
mag_gen_stub_fill(i64, mag_G, i64, mag_cvt_nop)

#undef mag_gen_stub_fill

#define mag_gen_stub_masked_fill(T, G, UT, CVT) \
    static MAG_HOTPROC void mag_masked_fill_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        mag_##T##_t val = CVT(mag_op_attr_unwrap_##UT(mag_cmd_attr(0))); \
        const mag_tensor_t *mask = mag_op_attr_unwrap_ptr(mag_cmd_attr(1)); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_bool_t *bm = mag_boolp(mask); \
        mag_coords_iter_t cr, cm; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cm, &mask->coords); \
        int64_t numel = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (numel+tc-1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra+chunk, numel); \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                int64_t mi = mag_coords_iter_broadcast(&cr, &cm, ri); \
                mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
                mag_bnd_chk(bm+mi, bm, mag_tensor_get_data_size(mask)); \
                if (bm[mi]) br[ri] = val; \
            } \
            return; \
        } \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, mi; \
            mag_coords_iter_offset2(&cr, &cm, i, &ri, &mi); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            mag_bnd_chk(bm+mi, bm, mag_tensor_get_data_size(mask)); \
            if (bm[mi]) br[ri] = val; \
        } \
    }

mag_gen_stub_masked_fill(e8m23, mag_G, e8m23, mag_cvt_nop)
mag_gen_stub_masked_fill(e5m10, mag_G_underlying, e8m23, mag_e8m23_to_e5m10)
mag_gen_stub_masked_fill(bool, mag_G, i64, mag_cvt_i642bool)
mag_gen_stub_masked_fill(u8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u64, mag_G, i64, mag_cvt_nop)
mag_gen_stub_masked_fill(i64, mag_G, i64, mag_cvt_nop)

#undef mag_gen_stub_masked_fill

#define mag_gen_stub_fill_rand(D, T, TS, UT) \
    static MAG_HOTPROC void mag_fill_rand_##D##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        mag_##TS##_t min = mag_op_attr_unwrap_##UT(mag_cmd_attr(0)); \
        mag_##TS##_t max = mag_op_attr_unwrap_##UT(mag_cmd_attr(1)); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        mag_philox4x32_stream_t *prng = payload->prng; \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_tensor_is_contiguous(r)) { \
            mag_vrand_##D##_##T(prng, rb-ra, br+ra, min, max); \
            return; \
        } \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            mag_vrand_##D##_##T(prng, 1, br+ri, min, max); \
        } \
    }

mag_gen_stub_fill_rand(uniform, e8m23, e8m23, e8m23)
mag_gen_stub_fill_rand(uniform, e5m10, e8m23, e8m23)
mag_gen_stub_fill_rand(uniform, u8, i64, i64)
mag_gen_stub_fill_rand(uniform, i8, i64, i64)
mag_gen_stub_fill_rand(uniform, u16, i64, i64)
mag_gen_stub_fill_rand(uniform, i16, i64, i64)
mag_gen_stub_fill_rand(uniform, u32, i64, i64)
mag_gen_stub_fill_rand(uniform, i32, i64, i64)
mag_gen_stub_fill_rand(uniform, u64, i64, i64)
mag_gen_stub_fill_rand(uniform, i64, i64, i64)
mag_gen_stub_fill_rand(normal, e8m23, e8m23, e8m23)
mag_gen_stub_fill_rand(normal, e5m10, e8m23, e8m23)

#undef mag_gen_stub_fill_rand

#define mag_gen_stub_fill_arange(T, CVT) \
    static MAG_HOTPROC void mag_fill_arange_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        mag_e8m23_t start = mag_op_attr_unwrap_e8m23(mag_cmd_attr(0)); /* TODO: Use double precision as ACC */ \
        mag_e8m23_t step = mag_op_attr_unwrap_e8m23(mag_cmd_attr(1)); \
        int64_t numel = r->numel; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=0; ri < numel; ++ri) { \
                mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
                br[ri] = CVT(start + (mag_e8m23_t)ri*step); \
            } \
            return; \
        } \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        for (int64_t i=0; i < numel; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = CVT(start + (mag_e8m23_t)i*step); \
        } \
    }

mag_gen_stub_fill_arange(e8m23, mag_cvt_nop)
mag_gen_stub_fill_arange(e5m10, mag_e8m23_to_e5m10)
mag_gen_stub_fill_arange(u8, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i8, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u16, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i16, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u32, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i32, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u64, mag_cvt_nop)
mag_gen_stub_fill_arange(i64, mag_cvt_nop)

#undef mag_gen_stub_fill_arange

#undef mag_G
#undef mag_G_underlying
