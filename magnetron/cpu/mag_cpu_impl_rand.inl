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
    static void MAG_AINLINE mag_vrand_uniform_##T(mag_philox4x32_stream_t *prng, int64_t numel, T *restrict o, float min, float max) {  \
        int64_t i=0;  \
        for (; i+3 < numel; i += 4) { \
            mag_philox4x32_float32x4_t r = mag_philox4x32_next_float32x4_uniform(prng, min, max); \
            for (int k=0; k < 4; ++k) \
                o[i+k] = CVT(r.v[k]); \
        }  \
        if (i < numel) {  \
            mag_philox4x32_float32x4_t r = mag_philox4x32_next_float32x4_uniform(prng, min, max); \
            for (int64_t t=0; i < numel; ++i, ++t)  \
                o[i] = CVT(r.v[t]);  \
        }  \
    }

mag_gen_vrand_uniform_fp(float, mag_cvt_nop)
mag_gen_vrand_uniform_fp(mag_float16_t, mag_float32_to_float16)

#undef mag_gen_vrand_uniform_fp

#define mag_gen_vrand_normal_fp(T, CVT) \
    static void MAG_AINLINE mag_vrand_normal_##T(mag_philox4x32_stream_t *prng, int64_t numel, T *restrict o, float mean, float std) {  \
        int64_t i=0;  \
        for (; i+3 < numel; i += 4) { \
            mag_philox4x32_float32x4_t r = mag_philox4x32_next_float32x4_normal(prng, mean, std); \
            for (int k=0; k < 4; ++k) \
                o[i+k] = CVT(r.v[k]); \
        }  \
        if (i < numel) {  \
            mag_philox4x32_float32x4_t r = mag_philox4x32_next_float32x4_normal(prng, mean, std); \
            for (int64_t t=0; i < numel; ++i, ++t)  \
                o[i] = CVT(r.v[t]);  \
        }  \
    }

mag_gen_vrand_normal_fp(float, mag_cvt_nop)
mag_gen_vrand_normal_fp(mag_float16_t, mag_float32_to_float16)

#undef mag_gen_vrand_normal_fp

/* Generate N bernoulli distributed booleans. */
static void MAG_AINLINE mag_vrand_bernoulli_bool(mag_philox4x32_stream_t *prng, int64_t numel, uint8_t *restrict o, float p) {
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
        mag_philox4x32_uint32x4_t r = mag_philox4x32_next_uint32x4(prng);
        for (int j=0; j < 4; ++j)
            o[i+j] = r.v[j] < thresh;
    }
    if (i < numel) {
        mag_philox4x32_uint32x4_t r = mag_philox4x32_next_uint32x4(prng);
        for (int64_t t=0; i < numel; ++i, ++t)
            o[i] = r.v[t] < thresh;
    }
}

#define mag_gen_vrand_uniform_int(T, UT) \
    static MAG_HOTPROC void mag_vrand_uniform_##T(mag_philox4x32_stream_t *prng, int64_t numel, T *restrict o, T min, T max) {                                                                  \
        UT umin = (UT)min; \
        UT umax = (UT)max; \
        uint64_t span64 = (uint64_t)((UT)(umax - umin))+1ull; \
        if (!span64) { \
            for (int64_t i=0; i < numel; ++i) { \
                UT r = (UT)mag_philox4x32_next_uint64(prng); \
                o[i] = (T)r; \
            } \
            return; \
        } \
        if (sizeof(UT) <= 4) { \
            uint32_t span = (uint32_t)span64; \
            uint32_t thresh = (uint32_t)(0u-span)%span; \
            for (int64_t i=0; i < numel; ++i) { \
                for (;;) { \
                    uint32_t x = mag_philox4x32_next_uint32(prng); \
                    uint64_t m = (uint64_t)x * (uint64_t)span; \
                    uint32_t lo = (uint32_t)m; \
                    if (mag_unlikely(lo < thresh)) continue; \
                    uint32_t hi = (uint32_t)(m>>32); \
                    UT v = (UT)((uint32_t)umin + hi); \
                    o[i] = (T)v; \
                    break; \
                } \
            } \
        } else { \
            uint64_t span = span64; \
            uint64_t thresh = (uint64_t)(0ull-span)%span; \
            for (int64_t i=0; i < numel; ++i) { \
                for (;;) { \
                    uint64_t x = mag_philox4x32_next_uint64(prng); \
                    mag_uint128_t m = mag_uint128_mul128(x, span); \
                    uint64_t lo = m.lo, hi = m.hi; \
                    if (mag_unlikely(lo < thresh)) continue; \
                    UT v = (UT)(umin + hi); \
                    o[i] = (T)v; \
                    break; \
                } \
            } \
        } \
    }

mag_gen_vrand_uniform_int(uint8_t, uint8_t)
mag_gen_vrand_uniform_int(int8_t, uint8_t)
mag_gen_vrand_uniform_int(uint16_t, uint16_t)
mag_gen_vrand_uniform_int(int16_t, uint16_t)
mag_gen_vrand_uniform_int(uint32_t, uint32_t)
mag_gen_vrand_uniform_int(int32_t, uint32_t)
mag_gen_vrand_uniform_int(uint64_t, uint64_t)
mag_gen_vrand_uniform_int(int64_t, uint64_t)

#undef mag_gen_vrand_uniform_int
