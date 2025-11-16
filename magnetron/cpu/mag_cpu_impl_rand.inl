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
mag_gen_vrand_uniform_fp(e5m10, mag_e8m23_cvt_e5m10)

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
mag_gen_vrand_normal_fp(e5m10, mag_e8m23_cvt_e5m10)

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

/* Generate N uniform distributed int32s ∈ [min, max]. */
static void MAG_AINLINE mag_vrand_uniform_i32(mag_philox4x32_stream_t *prng, int64_t numel, int32_t *restrict o, int32_t min, int32_t max) {
    if (max < min) mag_swap(int32_t, min, max);
    uint64_t spa64 = (uint64_t)(uint32_t)max - (uint64_t)(uint32_t)min+1ull;
    if (!spa64) {
        for (int64_t i=0; i < numel; ++i)
            o[i] = (int32_t)mag_philox4x32_next_u32(prng);
        return;
    }
    uint32_t span = (uint32_t)spa64;
    uint32_t thresh = (uint32_t)(-span) % span;
    int64_t i=0;
    while (i < numel) {
        mag_philox4x32_u32x4_t r = mag_philox4x32_next_u32x4(prng);
        for (int64_t k=0; k < 4 && i < numel; ++k) {
            uint32_t x = r.v[k];
            uint64_t m = (uint64_t)x*(uint64_t)span;
            uint32_t lo = (uint32_t)m;
            uint32_t hi = (uint32_t)(m >> 32);
            if (mag_unlikely(lo < thresh)) { /* Rejection sampling */
                do {
                    uint32_t x2 = mag_philox4x32_next_u32(prng);
                    m = (uint64_t)x2*(uint64_t)span;
                    lo = (uint32_t)m;
                    hi = (uint32_t)(m>>32);
                } while (lo < thresh);
            }
            o[i++] = (int32_t)((uint32_t)min + hi);
        }
    }
}
