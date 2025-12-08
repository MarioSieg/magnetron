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

#define mag_gen_stub_fill(T, TF, G, UT, CVT) \
    static MAG_HOTPROC void mag_fill_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        T val = (T)CVT(mag_op_attr_unwrap_##UT(mag_cmd_attr(0))); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total+tc-1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra+chunk, total); \
        if (mag_unlikely(rb <= ra)) return; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
                br[ri] = val; \
            } \
            return; \
        } \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            br[ri] = val; \
        } \
    }

mag_gen_stub_fill(float, float32, mag_G, float64, mag_cvt_nop)
mag_gen_stub_fill(mag_float16_t, float16, mag_G_underlying, float64, mag_float32_to_float16)
mag_gen_stub_fill(uint8_t, uint8, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int8_t, int8, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint16_t, uint16, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int16_t, int16, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint32_t, uint32, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int32_t, int32, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint64_t, uint64, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int64_t, int64, mag_G, int64, mag_cvt_nop)

#undef mag_gen_stub_fill

#define mag_gen_stub_masked_fill(T, TF, G, UT, CVT) \
    static MAG_HOTPROC void mag_masked_fill_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        T val = (T)CVT(mag_op_attr_unwrap_##UT(mag_cmd_attr(0))); \
        const mag_tensor_t *mask = mag_op_attr_unwrap_ptr(mag_cmd_attr(1)); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        const uint8_t *bm = (const uint8_t *)mag_tensor_data_ptr(mask); \
        mag_coords_iter_t cr, cm; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cm, &mask->coords); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total+tc-1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra+chunk, total); \
        if (mag_unlikely(rb <= ra)) return; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                int64_t mi = mag_coords_iter_broadcast(&cr, &cm, ri); \
                mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
                mag_bnd_chk(bm+mi, bm, mag_tensor_numbytes(mask)); \
                if (bm[mi]) br[ri] = val; \
            } \
            return; \
        } \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, mi; \
            mag_coords_iter_offset2(&cr, &cm, i, &ri, &mi); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            mag_bnd_chk(bm+mi, bm, mag_tensor_numbytes(mask)); \
            if (bm[mi]) br[ri] = val; \
        } \
    }

mag_gen_stub_masked_fill(float, float32, mag_G, float64, mag_cvt_nop)
mag_gen_stub_masked_fill(mag_float16_t, float16, mag_G_underlying, float64, mag_float32_to_float16)
mag_gen_stub_masked_fill(uint8_t, uint8, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int8_t, int8, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint16_t, uint16, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int16_t, int16, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint32_t, uint32, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int32_t, int32, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint64_t, uint64, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_masked_fill(int64_t, int64, mag_G, int64, mag_cvt_nop)

#undef mag_gen_stub_masked_fill

#define mag_gen_stub_fill_rand(D, T, TS, UT, TF) \
    static MAG_HOTPROC void mag_fill_rand_##D##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        TS min = (TS)mag_op_attr_unwrap_##UT(mag_cmd_attr(0)); \
        TS max = (TS)mag_op_attr_unwrap_##UT(mag_cmd_attr(1)); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        mag_philox4x32_stream_t *prng = payload->prng; \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_unlikely(rb <= ra)) return; \
        if (mag_tensor_is_contiguous(r)) { \
            mag_vrand_##D##_##T(prng, rb-ra, br+ra, min, max); \
            return; \
        } \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            mag_vrand_##D##_##T(prng, 1, br+ri, min, max); \
        } \
    }

mag_gen_stub_fill_rand(uniform, float, float, float64, float32)
mag_gen_stub_fill_rand(uniform, mag_float16_t, float, float64, float16)
mag_gen_stub_fill_rand(uniform, uint8_t, uint64_t, uint64, uint8)
mag_gen_stub_fill_rand(uniform, int8_t, int64_t, int64, int8)
mag_gen_stub_fill_rand(uniform, uint16_t, uint64_t, uint64, uint16)
mag_gen_stub_fill_rand(uniform, int16_t, int64_t, int64, int16)
mag_gen_stub_fill_rand(uniform, uint32_t, uint64_t, uint64, uint32)
mag_gen_stub_fill_rand(uniform, int32_t, int64_t, int64, int32)
mag_gen_stub_fill_rand(uniform, uint64_t, uint64_t, uint64, uint64)
mag_gen_stub_fill_rand(uniform, int64_t, int64_t, int64, int64)

mag_gen_stub_fill_rand(normal, float, float, float64, float32)
mag_gen_stub_fill_rand(normal, mag_float16_t, float, float64, float16)

#undef mag_gen_stub_fill_rand

#define mag_gen_stub_arange(T, TF, CVT) \
    static MAG_HOTPROC void mag_arange_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        double start = mag_op_attr_unwrap_float64(mag_cmd_attr(0));  /* TODO: this looses information for int64/uint64 ranges that exceed f64 precision */\
        double step = mag_op_attr_unwrap_float64(mag_cmd_attr(1)); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_unlikely(rb <= ra)) return; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t ri=ra; ri < rb; ++ri) { \
                mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
                br[ri] = CVT(start + (double)ri*step); \
            } \
            return; \
        } \
        mag_coords_iter_t cr; \
        mag_coords_iter_init(&cr, &r->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            br[ri] = CVT(start + (double)i*step); \
        } \
    }

mag_gen_stub_arange(float, float32, mag_cvt_nop)
mag_gen_stub_arange(mag_float16_t, float16, mag_float32_to_float16)
mag_gen_stub_arange(uint8_t, uint8, mag_cvt_int32_to_int64)
mag_gen_stub_arange(int8_t, int8, mag_cvt_int32_to_int64)
mag_gen_stub_arange(uint16_t, uint16, mag_cvt_int32_to_int64)
mag_gen_stub_arange(int16_t, int16, mag_cvt_int32_to_int64)
mag_gen_stub_arange(uint32_t, uint32, mag_cvt_int32_to_int64)
mag_gen_stub_arange(int32_t, int32, mag_cvt_int32_to_int64)
mag_gen_stub_arange(uint64_t, uint64, mag_cvt_nop)
mag_gen_stub_arange(int64_t, int64, mag_cvt_nop)

#undef mag_gen_stub_arange

static MAG_HOTPROC void mag_one_hot_int64(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    mag_tensor_t *idx = mag_cmd_in(0);
    int64_t *restrict pr = (int64_t *)mag_tensor_data_ptr_mut(r);
    const int64_t *restrict pidx = (const int64_t *)mag_tensor_data_ptr(idx);
    int64_t nc = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); /* number of classes */
    int64_t total = idx->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    if (mag_unlikely(rb <= ra)) return;
    if (mag_full_cont2(r, idx)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t cls = pidx[i];
            if ((uint64_t)cls < (uint64_t)nc) {
                int64_t off = i*nc + cls;
                mag_bnd_chk(pr+off, pr, mag_tensor_numbytes(r));
                pr[off] = 1;
            }
        }
        return;
    }
    mag_coords_iter_t it;
    mag_coords_iter_init(&it, &idx->coords);
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ridx = mag_coords_iter_to_offset(&it, i);
        mag_bnd_chk(pidx+ridx, pidx, mag_tensor_numbytes(idx));
        int64_t cls = pidx[ridx];
        if ((uint64_t)cls < (uint64_t)nc) {
            int64_t off = i*nc + cls;
            mag_bnd_chk(pr+off, pr, mag_tensor_numbytes(r));
            pr[off] = 1;
        }
    }
}

#undef mag_G
#undef mag_G_underlying

static MAG_HOTPROC void mag_fill_rand_bernoulli_bool(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    float p = (float)mag_op_attr_unwrap_float64(mag_cmd_attr(0));
    uint8_t *b_r = (uint8_t *)mag_tensor_data_ptr_mut(r);
    int64_t numel = r->numel;
    mag_vrand_bernoulli_bool(payload->prng, numel, b_r, p);
}

#define mag_gen_stub_rand_perm(T, TF, CVT) \
    static MAG_HOTPROC void mag_rand_perm_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        int64_t numel = r->numel; \
        mag_philox4x32_stream_t *prng = payload->prng; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t i=0; i < numel; ++i) { \
                mag_bnd_chk(br+i, br, mag_tensor_numbytes(r)); \
                br[i] = CVT((int64_t)i); \
            } \
            for (int64_t i=0; i < numel-1; ++i) { \
                int64_t j = i+(int64_t)(mag_philox4x32_next_uint64(prng)%(uint64_t)(numel-i)); \
                mag_bnd_chk(br+j, br, mag_tensor_numbytes(r)); \
                T tmp = br[i]; \
                br[i] = br[j]; \
                br[j] = tmp; \
            } \
            return; \
        } \
        mag_coords_iter_t it; \
        mag_coords_iter_init(&it, &r->coords); \
        for (int64_t i=0; i < numel; ++i) { \
            int64_t off = mag_coords_iter_to_offset(&it, i); \
            mag_bnd_chk(br+off, br, mag_tensor_numbytes(r)); \
            br[off] = CVT((int64_t)i); \
        } \
        for (int64_t i=0; i < numel-1; ++i) { \
            int64_t j = i+(int64_t)(mag_philox4x32_next_uint64(prng)%(uint64_t)(numel-i)); \
            int64_t off_i = mag_coords_iter_to_offset(&it, i); \
            int64_t off_j = mag_coords_iter_to_offset(&it, j); \
            mag_bnd_chk(br+off_i, br, mag_tensor_numbytes(r)); \
            mag_bnd_chk(br+off_j, br, mag_tensor_numbytes(r)); \
            T tmp = br[off_i]; \
            br[off_i] = br[off_j]; \
            br[off_j] = tmp; \
        } \
    }

mag_gen_stub_rand_perm(uint8_t, uint8, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int8_t, int8, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint16_t, uint16, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int16_t, int16, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint32_t, uint32, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int32_t, int32, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint64_t, uint64, mag_cvt_nop)
mag_gen_stub_rand_perm(int64_t, int64, mag_cvt_nop)

#undef mag_gen_stub_randperm
