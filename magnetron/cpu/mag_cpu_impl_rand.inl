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

#include <core/mag_u128.h>

#define mag_gen_vrand_uniform_fp(T, CVT) \
    static void MAG_AINLINE mag_vrand_uniform_##T(mag_philox4x32_stream_t *prng, int64_t numel, mag_##T##_t *restrict o, mag_e8m23_t min, mag_e8m23_t max) {  \
        int64_t i=0;  \
        for (; i+3 < numel; i += 4) { \
            mag_philox4x32_e8m23x4_t r = mag_philox4x32_next_e8m23x4_uniform(prng, min, max); \
            for (int k=0; k < 4; ++k) \
                o[i+k] = CVT(r.v[k]); \
        }  \
        if (i < numel) {  \
            mag_philox4x32_e8m23x4_t r = mag_philox4x32_next_e8m23x4_uniform(prng, min, max); \
            for (int64_t t=0; i < numel; ++i, ++t)  \
                o[i] = CVT(r.v[t]);  \
        }  \
    }

mag_gen_vrand_uniform_fp(e8m23, mag_cvt_nop)
mag_gen_vrand_uniform_fp(e5m10, mag_e8m23_to_e5m10)

#undef mag_gen_vrand_uniform_fp

#define mag_gen_vrand_normal_fp(T, CVT) \
    static void MAG_AINLINE mag_vrand_normal_##T(mag_philox4x32_stream_t *prng, int64_t numel, mag_##T##_t *restrict o, mag_e8m23_t mean, mag_e8m23_t std) {  \
        int64_t i=0;  \
        for (; i+3 < numel; i += 4) { \
            mag_philox4x32_e8m23x4_t r = mag_philox4x32_next_e8m23x4_normal(prng, mean, std); \
            for (int k=0; k < 4; ++k) \
                o[i+k] = CVT(r.v[k]); \
        }  \
        if (i < numel) {  \
            mag_philox4x32_e8m23x4_t r = mag_philox4x32_next_e8m23x4_normal(prng, mean, std); \
            for (int64_t t=0; i < numel; ++i, ++t)  \
                o[i] = CVT(r.v[t]);  \
        }  \
    }

mag_gen_vrand_normal_fp(e8m23, mag_cvt_nop)
mag_gen_vrand_normal_fp(e5m10, mag_e8m23_to_e5m10)

#undef mag_gen_vrand_normal_fp

/* Generate N bernoulli distributed booleans. */
static void MAG_AINLINE mag_vrand_bernoulli_bool(mag_philox4x32_stream_t *prng, int64_t numel, mag_bool_t *restrict o, mag_e8m23_t p) {
    if (mag_unlikely(p <= 0.0f)) {
        memset(o, 0, sizeof(*o)*numel);
        return;
    }
    if (mag_unlikely(p >= 1.0f)) {
        for (int64_t i=0; i < numel; ++i) o[i] = 1;
        return;
    }
    uint32_t thresh = (uint32_t)(p*4294967296.f); /* 2^32 */
    int64_t i=0;
    for (; i+3 < numel; i += 4) {
        mag_philox4x32_u32x4_t r = mag_philox4x32_next_u32x4(prng);
        for (int j=0; j < 4; ++j)
            o[i+j] = r.v[j] < thresh;
    }
    if (i < numel) {
        mag_philox4x32_u32x4_t r = mag_philox4x32_next_u32x4(prng);
        for (int64_t t=0; i < numel; ++i, ++t)
            o[i] = r.v[t] < thresh;
    }
}

#define mag_gen_vrand_uniform_int(T, UT) \
    static MAG_HOTPROC void mag_vrand_uniform_##T(mag_philox4x32_stream_t *prng, int64_t numel, mag_##T##_t *restrict o, mag_##T##_t min, mag_##T##_t max) {                                                                  \
        if (mag_unlikely(max < min)) mag_swap(mag_##T##_t, min, max); \
        mag_##UT##_t umin = (mag_##UT##_t)min; \
        mag_##UT##_t umax = (mag_##UT##_t)max; \
        uint64_t span64 = (uint64_t)((mag_##UT##_t)(umax - umin))+1ull; \
        if (!span64) { \
            for (int64_t i=0; i < numel; ++i) { \
                mag_##UT##_t r = (mag_##UT##_t)mag_philox4x32_next_u64(prng); \
                o[i] = (mag_##T##_t)r; \
            } \
            return; \
        } \
        if (sizeof(mag_##UT##_t) <= 4) { \
            uint32_t span = (uint32_t)span64; \
            uint32_t thresh = (uint32_t)(0u-span)%span; \
            for (int64_t i=0; i < numel; ++i) { \
                for (;;) { \
                    uint32_t x = mag_philox4x32_next_u32(prng); \
                    uint64_t m = (uint64_t)x * (uint64_t)span; \
                    uint32_t lo = (uint32_t)m; \
                    if (mag_unlikely(lo < thresh)) continue; \
                    uint32_t hi = (uint32_t)(m>>32); \
                    mag_##UT##_t v = (mag_##UT##_t)((uint32_t)umin + hi); \
                    o[i] = (mag_##T##_t)v; \
                    break; \
                } \
            } \
        } else { \
            uint64_t span = span64; \
            uint64_t thresh = (uint64_t)(0ull-span)%span; \
            for (int64_t i=0; i < numel; ++i) { \
                for (;;) { \
                    uint64_t x = mag_philox4x32_next_u64(prng); \
                    mag_uint128_t m = mag_uint128_mul128(x, span); \
                    uint64_t lo = m.lo, hi = m.hi; \
                    if (mag_unlikely(lo < thresh)) continue; \
                    mag_##UT##_t v = (mag_##UT##_t)(umin + hi); \
                    o[i] = (mag_##T##_t)v; \
                    break; \
                } \
            } \
        } \
    }

mag_gen_vrand_uniform_int(u8, u8)
mag_gen_vrand_uniform_int(i8, u8)
mag_gen_vrand_uniform_int(u16, u16)
mag_gen_vrand_uniform_int(i16, u16)
mag_gen_vrand_uniform_int(u32, u32)
mag_gen_vrand_uniform_int(i32, u32)
mag_gen_vrand_uniform_int(u64, u64)
mag_gen_vrand_uniform_int(i64, u64)

#undef mag_gen_vrand_uniform_int

static MAG_HOTPROC void mag_fill_rand_bernoulli_bool(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    mag_e8m23_t p = mag_op_attr_unwrap_e8m23(mag_cmd_attr(0));
    mag_bool_t *b_r = mag_boolp_mut(r);
    int64_t numel = r->numel;
    mag_vrand_bernoulli_bool(payload->prng, numel, b_r, p);
}

#define mag_gen_stub_rand_perm(T, CVT) \
    static MAG_HOTPROC void mag_rand_perm_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        int64_t numel = r->numel; \
        mag_philox4x32_stream_t *prng = payload->prng; \
        if (mag_tensor_is_contiguous(r)) { \
            for (int64_t i=0; i < numel; ++i) { \
                mag_bnd_chk(br+i, br, mag_tensor_get_data_size(r)); \
                br[i] = CVT((int64_t)i); \
            } \
            for (int64_t i=0; i < numel-1; ++i) { \
                int64_t j = i+(int64_t)(mag_philox4x32_next_u64(prng)%(uint64_t)(numel-i)); \
                mag_bnd_chk(br+j, br, mag_tensor_get_data_size(r)); \
                mag_##T##_t tmp = br[i]; \
                br[i] = br[j]; \
                br[j] = tmp; \
            } \
            return; \
        } \
        mag_coords_iter_t it; \
        mag_coords_iter_init(&it, &r->coords); \
        for (int64_t i=0; i < numel; ++i) { \
            int64_t off = mag_coords_iter_to_offset(&it, i); \
            mag_bnd_chk(br+off, br, mag_tensor_get_data_size(r)); \
            br[off] = CVT((int64_t)i); \
        } \
        for (int64_t i=0; i < numel-1; ++i) { \
            int64_t j = i+(int64_t)(mag_philox4x32_next_u64(prng)%(uint64_t)(numel-i)); \
            int64_t off_i = mag_coords_iter_to_offset(&it, i); \
            int64_t off_j = mag_coords_iter_to_offset(&it, j); \
            mag_bnd_chk(br+off_i, br, mag_tensor_get_data_size(r)); \
            mag_bnd_chk(br+off_j, br, mag_tensor_get_data_size(r)); \
            mag_##T##_t tmp = br[off_i]; \
            br[off_i] = br[off_j]; \
            br[off_j] = tmp; \
        } \
    }

mag_gen_stub_rand_perm(u8, mag_cvt_i642i32)
mag_gen_stub_rand_perm(i8, mag_cvt_i642i32)
mag_gen_stub_rand_perm(u16, mag_cvt_i642i32)
mag_gen_stub_rand_perm(i16, mag_cvt_i642i32)
mag_gen_stub_rand_perm(u32, mag_cvt_i642i32)
mag_gen_stub_rand_perm(i32, mag_cvt_i642i32)
mag_gen_stub_rand_perm(u64, mag_cvt_nop)
mag_gen_stub_rand_perm(i64, mag_cvt_nop)

#undef mag_gen_stub_randperm
