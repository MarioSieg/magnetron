/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
**
**
** !!! Make sure all functions in this file are static. This is required to correctly clone the impl for each specialized compilation unit.
** This file implements the core math for magnetron, optimized for different CPU instruction sets.
** This file is also included into different compilation units, which are all compiled with different architecture flags, thus the impl is 'cloned'.
** At runtime the best impl for the host-CPU is chose automatically, by detecting the CPU and querying the hardware features.
**
** !!! Minimum Requirements!!!
**  AMD 64 CPUs: SSE & SSE2 (any 64-bit AMD64 CPU).
**  ARM 64 CPUs: ARM v8-a (Raspberry Pi 4, 5, Apple M1-4, Neoverse/Graviton etc..)
**
** +==============+=============+==============+======================================================+
** | AMD 64 Versions and Features
** +==============+=============+==============+======================================================+
** | x86-64-v1	| CMOV, CX8, FPU, FXSR, MMX, OSFXSR, SCE, SSE, SSE2
** | x86-64-v2	| CMPXCHG16B, LAHF-SAHF, POPCNT, SSE3, SSE4_1, SSE4_2, SSSE3
** | x86-64-v3	| AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE
** | x86-64-v4	| AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL
** +==============+=============+==============+======================================================+
** Some CPUs fall inbetween those, for example my old rusty test server has four old AMD Opteron CPUs with 16 cores each. They support AVX but not AVX2.
** For CPUs like this, we still support more granular feature levels: SSE42, AVX, AVX2 and AVX512F.
**
** +==============+=============+==============+======================================================+
** | ARM 64 Versions and Features
** +==============+=============+==============+======================================================+
** | armv8-a      |  Armv8-A    |              |  +fp, +simd
** | armv8.1-a    |  Armv8.1-A  |  armv8-a,    |  +crc, +lse, +rdma
** | armv8.2-a    |  Armv8.2-A  |  armv8.1-a   |
** | armv8.3-a    |  Armv8.3-A  |  armv8.2-a,  |  +pauth, +fcma, +jscvt
** | armv8.4-a    |  Armv8.4-A  |  armv8.3-a,  |  +flagm, +fp16fml, +dotprod, +rcpc2
** | armv8.5-a    |  Armv8.5-A  |  armv8.4-a,  |  +sb, +ssbs, +predres, +frintts, +flagm2
** | armv8.6-a    |  Armv8.6-A  |  armv8.5-a,  |  +bf16, +i8mm
** | armv8.7-a    |  Armv8.7-A  |  armv8.6-a,  |  +wfxt, +xs
** | armv8.8-a    |  Armv8.8-a  |  armv8.7-a,  |  +mops
** | armv8.9-a    |  Armv8.9-a  |  armv8.8-a   |
** | armv9-a      |  Armv9-A    |  armv8.5-a,  |  +sve, +sve2
** | armv9.1-a    |  Armv9.1-A  |  armv9-a,    |  +bf16, +i8mm
** | armv9.2-a    |  Armv9.2-A  |  armv9.1-a   |
** | armv9.3-a    |  Armv9.3-A  |  armv9.2-a,  |  +mops
** | armv9.4-a    |  Armv9.4-A  |  armv9.3-a   |
** | armv8-r      |  Armv8-R    |  armv8-r     |
** +==============+=============+==============+======================================================+
*/

#include "magnetron_internal.h"

#include <math.h>

#define mag_e8m23p(t) ((const mag_E8M23*)mag_tensor_get_data_ptr(t))
#define mag_e8m23p_mut(t) ((mag_E8M23*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p(t) ((const mag_E5M10*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p_mut(t) ((mag_E5M10*)mag_tensor_get_data_ptr(t))
#define mag_boolp(t) ((const uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_boolp_mut(t) ((uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p(t) ((const int32_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p_mut(t) ((int32_t*)mag_tensor_get_data_ptr(t))

#define MAG_TAU 6.283185307179586476925286766559005768394338798f /* τ=2π */
#define MAG_INVSQRT2 0.707106781186547524400844362104849039284835937f /* 1/√2 */

#if defined(_MSC_VER)
typedef uint16_t __fp16; /* MSVC does not support __fp16. */
#ifdef __AVX2__ /*MSVC does not define FMA and F16C with AVX 2*/
#define __FMA__ 1
#define __F16C__ 1
#endif
#endif

static MAG_AINLINE mag_E5M10 mag_e8m23_cvt_e5m10(mag_E8M23 x) {
    uint16_t r;
    #ifdef __F16C__
        #ifdef _MSC_VER
            r = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0);
        #else
            r = _cvtss_sh(x, 0);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        union {
            __fp16 f;
            uint16_t u;
        } castor = {.f=(__fp16)x};
        r = castor.u;
    #else
        union {
            uint32_t u;
            mag_E8M23 f;
        } castor;
        mag_E8M23 base = fabs(x)*0x1.0p+112f*0x1.0p-110f;
        castor.f = x;
        uint32_t shl1_w = castor.u+castor.u;
        uint32_t sign = castor.u & 0x80000000u;
        castor.u = 0x07800000u+(mag_xmax(0x71000000u, shl1_w&0xff000000u)>>1);
        castor.f = base + castor.f;
        uint32_t exp_bits = (castor.u>>13) & 0x00007c00u;
        uint32_t mant_bits = castor.u & 0x00000fffu;
        uint32_t nonsign = exp_bits + mant_bits;
        r = (sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign);
    #endif
    return (mag_E5M10){.bits=r};
}

static MAG_AINLINE mag_E8M23 mag_e5m10_cvt_e8m23(mag_E5M10 x) {
    #ifdef __F16C__
        #ifdef _MSC_VER
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x.bits)));
        #else
            return _cvtsh_ss(x.bits);
        #endif
    #elif defined(__ARM_NEON) && !defined(_MSC_VER)
        union {
            __fp16 f;
            uint16_t u;
        } castor = {.u=x.bits};
        return castor.f;
    #else
        union {
            uint32_t u;
            mag_E8M23 f;
        } castor;
        uint32_t w = (uint32_t)x.bits<<16;
        uint32_t sign = w & 0x80000000u;
        uint32_t two_w = w+w;
        uint32_t offs = 0xe0u<<23;
        uint32_t t1 = (two_w>>4) + offs;
        uint32_t t2 = (two_w>>17) | (126u<<23);
        castor.u = t1;
        mag_E8M23 norm_x = castor.f*0x1.0p-112f;
        castor.u = t2;
        mag_E8M23 denorm_x = castor.f-0.5f;
        uint32_t denorm_cutoff = 1u<<27;
        uint32_t r = sign | (two_w < denorm_cutoff
            ? (castor.f = denorm_x, castor.u)
            : (castor.f = norm_x, castor.u));
        castor.u = r;
        return castor.f;
    #endif
}

static void MAG_HOTPROC mag_vcast_e8m23_e5m10(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E5M10* o = xo;
    const mag_E8M23* x = xx;
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < numel; i += 4) {
            float32x4_t v = vld1q_f32(x+i);
            vst1_f16((__fp16*)o+i, vcvt_f16_f32(v));
        }
    #elif defined(__F16C__)
        #ifdef __AVX512F__
            for (; i+15 < numel; i += 16) {
                __m512 xv = _mm512_loadu_ps(x+i);
                __m256i yv = _mm512_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
                _mm256_storeu_si256((__m256i*)(o+i), yv);
            }
        #endif
        for (; i+7 < numel; i += 8) {
            __m256 xv = _mm256_loadu_ps(x+i);
            __m128i yv = _mm256_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i*)(o+i), yv);
        }
        for (; i+3 < numel; i += 4) {
            __m128 xv = _mm_loadu_ps(x+i);
            __m128i yv = _mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
            _mm_storel_epi64((__m128i*)(o+i), yv);
        }
    #endif
    for (; i < numel; ++i) /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(x[i]);
}
static void MAG_HOTPROC mag_vcast_e5m10_e8m23(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E8M23* o = xo;
    const mag_E5M10* x = xx;
    int64_t i=0;
    #ifdef __ARM_NEON
        for (; i+3 < numel; i += 4) {
            float16x4_t v = vld1_f16((const __fp16*)x+i);
            vst1q_f32(o+i, vcvt_f32_f16(v));
        }
    #elif defined(__F16C__)
        #ifdef __AVX512F__
            for (; i+15 < numel; i += 16) {
                __m256i xv = _mm256_loadu_si256((const __m256i*)(x+i));
                __m512 yv = _mm512_cvtph_ps(xv);
                _mm512_storeu_ps(o+i, yv);
            }
        #endif
        for (; i+7 < numel; i += 8) {
            __m128i xv = _mm_loadu_si128((const __m128i*)(x+i));
            __m256 yv = _mm256_cvtph_ps(xv);
            _mm256_storeu_ps(o+i, yv);
        }
        for (; i+3 < numel; i += 4) {
            __m128i xv = _mm_loadl_epi64((const __m128i*)(x+i));
            __m128 yv = _mm_cvtph_ps(xv);
            _mm_storeu_ps(o+i, yv);
        }
    #endif
    for (; i < numel; ++i) /* Scalar drain loop */
        o[i] = mag_e5m10_cvt_e8m23(x[i]);
}

static void MAG_HOTPROC mag_vcast_e8m23_i32(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    int32_t* o = xo;
    const mag_E8M23* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (int32_t)x[i];
}
static void MAG_HOTPROC mag_vcast_i32_e8m23(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E8M23* o = xo;
    const int32_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (mag_E8M23)x[i];
}

static void MAG_HOTPROC mag_vcast_e8m23_bool(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    uint8_t* o = xo;
    const mag_E8M23* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (uint8_t)(x[i] != .0f);
}
static void MAG_HOTPROC mag_vcast_bool_e8m23(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E8M23* o = xo;
    const uint8_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vcast_i32_bool(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    uint8_t* o = xo;
    const int32_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (uint8_t)(x[i] != 0);
}
static void MAG_HOTPROC mag_vcast_bool_i32(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    int32_t* o = xo;
    const uint8_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? 1 : 0;
}

static void MAG_HOTPROC mag_vcast_e5m10_i32(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    int32_t* o = xo;
    const mag_E5M10* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (int32_t)mag_e5m10_cvt_e8m23(x[i]);
}
static void MAG_HOTPROC mag_vcast_i32_e5m10(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E5M10* o = xo;
    const int32_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10((mag_E8M23)x[i]);
}

static void MAG_HOTPROC mag_vcast_e5m10_bool(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    uint8_t* o = xo;
    const mag_E5M10* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (uint8_t)(x[i].bits != 0);
}
static void MAG_HOTPROC mag_vcast_bool_e5m10(int64_t numel, void* _Nonnull restrict xo, const void* _Nonnull restrict xx) {
    mag_E5M10* o = xo;
    const uint8_t* x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static uint32_t MAG_AINLINE mag_mt19937_step(uint32_t* _Nonnull rem, uint32_t* _Nonnull next, uint32_t* _Nonnull state) {
    if (--*rem <= 0) {
        *rem = 624;
        *next = 0;
        uint32_t y, i;
        for (i = 0; i < 624-397; ++i) {
            y = (state[i]&0x80000000u) | (state[i+1]&0x7fffffffu);
            state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0x9908b0dfu : 0);
        }
        for (; i < 624-1; ++i) {
            y = (state[i]&0x80000000u) | (state[i+1]&0x7fffffffu);
            state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0x9908b0dfu : 0);
        }
        y = (state[624-1]&0x80000000u) | (*state&0x7fffffffu);
        state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0x9908b0dfu : 0);
    }
    uint32_t y = state[(*next)++];
    y ^= y>>11;
    y ^= (y<<7) & 0x9d2c5680;
    y ^= (y<<15) & 0xefc60000;
    y ^= y>>18;
    return y;
}

static uint32_t MAG_AINLINE mag_pcg_step(uint64_t* _Nonnull state, uint64_t inc) {
    uint64_t prev = *state;
    *state = prev*6364136223846793005ull + inc;
    uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
    uint32_t rot = prev >> 59u;
    return (mixed>>rot) | (mixed << ((-rot)&31));
}

#define mag_e8m23_canonical(y) (1.f/0x1.0p23f*((mag_E8M23)((y)>>9) + 0.5f)) /* Transform u32 -> xi ∈ [0, 1) */

static void MAG_AINLINE mag_box_mueller(mag_E8M23* _Nonnull u1, mag_E8M23* _Nonnull u2, mag_E8M23 std, mag_E8M23 mean) {
    mag_E8M23 mag = std*sqrtf(-2.0f*logf(*u1));
    *u1 = mag*cosf(MAG_TAU**u2) + mean;
    *u2 = mag*sinf(MAG_TAU**u2) + mean;
}

/* Generate N uniform distributed e8m23 floats ∈ [min, max]. */
static void MAG_AINLINE mag_vrand_uniform_e8m23(mag_PRNGState* _Nonnull prng, int64_t numel, mag_E8M23* restrict _Nonnull o, mag_E8M23 min, mag_E8M23 max) {
    mag_E8M23 rescale_uniform = max - min;
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t i=0; i < numel; ++i) {
                o[i] = min + rescale_uniform*mag_e8m23_canonical(mag_mt19937_step(rem, next, state)); /* Generate canonical and rescale. */
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t inc = prng->pcg.inc;
            for (int64_t i=0; i < numel; ++i) {
                o[i] = min + rescale_uniform*mag_e8m23_canonical(mag_pcg_step(state, inc)); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N uniform distributed e5m10 floats ∈ [min, max]. */
static void MAG_AINLINE mag_vrand_uniform_e5m10(mag_PRNGState* _Nonnull prng, int64_t numel, mag_E5M10* restrict _Nonnull o, mag_E8M23 min, mag_E8M23 max) {
    mag_E8M23 rescale_uniform = max - min;
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t i=0; i < numel; ++i) {
                o[i] = mag_e8m23_cvt_e5m10(min + rescale_uniform*mag_e8m23_canonical(mag_mt19937_step(rem, next, state))); /* Generate canonical and rescale. */
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t inc = prng->pcg.inc;
            for (int64_t i=0; i < numel; ++i) {
                o[i] = mag_e8m23_cvt_e5m10(min + rescale_uniform*mag_e8m23_canonical(mag_pcg_step(state, inc))); /* Generate canonical and rescale. */
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N normal (Gauss) distributed e8m23 floats. */
static void MAG_HOTPROC mag_vrand_normal_e8m23(mag_PRNGState* _Nonnull prng, int64_t numel, mag_E8M23* restrict _Nonnull o, mag_E8M23 mean, mag_E8M23 std) {
    mag_vrand_uniform_e8m23(prng, numel, o, 0.0f, 1.f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < numel-1; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_box_mueller(o+i, o+i+1, std, mean);
    }
    if (numel & 1) {  /* Handle odd numel */
        mag_E8M23 u[2];
        mag_vrand_uniform_e8m23(prng, sizeof(u)/sizeof(*u), u, 0.0f, 1.f);
        o[numel-1] = std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean;
    }
}

/* Generate N normal (Gauss) distributed e5m10 floats. */
static void MAG_HOTPROC mag_vrand_normal_e5m10(mag_PRNGState* _Nonnull prng, int64_t numel, mag_E5M10* restrict _Nonnull o, mag_E8M23 mean, mag_E8M23 std) {
    mag_vrand_uniform_e5m10(prng, numel, o, 0.0f, 1.f); /* Generate uniform random numbers. */
    for (int64_t i=0; i < numel; i += 2) { /* Map uniform to normal dist with Box-Muller transform. TODO: Write SIMD sqrt and vectorize this. */
        mag_E8M23 u1 = mag_e5m10_cvt_e8m23(o[i]);
        mag_E8M23 u2 = mag_e5m10_cvt_e8m23(o[i+1]);
        mag_box_mueller(&u1, &u2, std, mean);
        o[i] = mag_e8m23_cvt_e5m10(u1);
        o[i+1] = mag_e8m23_cvt_e5m10(u2);
    }
    if (numel & 1) {  /* Handle odd numel */
        mag_E8M23 u[2];
        mag_vrand_uniform_e8m23(prng, sizeof(u)/sizeof(*u), u, 0.0f, 1.f);
        o[numel-1] = mag_e8m23_cvt_e5m10(std*sqrtf(-2.0f*logf(u[0]))*cosf(MAG_TAU*u[1]) + mean);
    }
}

/* Generate N bernoulli distributed e5m10 floats. */
static void MAG_AINLINE mag_vrand_bernoulli_bool(mag_PRNGState* _Nonnull prng, int64_t numel, uint8_t* restrict _Nonnull o, mag_E8M23 p) {
    uint32_t thresh = (uint32_t)(p*4294967296.f); /* 2^32 */
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t i=0; i < numel; ++i) {
                o[i] = !!(mag_mt19937_step(rem, next, state) < thresh);
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t inc = prng->pcg.inc;
            for (int64_t ii=0; ii < numel; ++ii) {
                o[ii] = !!(mag_pcg_step(state, inc) < thresh);
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

/* Generate N uniform distributed int32s ∈ [min, max]. */
static void MAG_AINLINE mag_vrand_uniform_i32(mag_PRNGState* _Nonnull prng, int64_t numel, int32_t* restrict _Nonnull o, int32_t min, int32_t max) {
    /* Rejection-sampling constants:
    ** we want r ∈ [0, 2³²) s.t. r < lim, where  lim = span*floor(2³²/span)
    ** for a bias-free mapping r % span.
    */
    uint64_t span = (uint64_t)max-(uint64_t)min+1ull; /* Interval width */
    uint64_t lim = 0x100000000ull - 0x100000000ull%span;
    switch (prng->algo) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Use Mersenne Twister. */
            uint32_t* rem = &prng->mersenne.remaining;
            uint32_t* next = &prng->mersenne.next;
            uint32_t* state = prng->mersenne.state;
            for (int64_t i=0; i < numel; ++i) {
                uint64_t r;
                do r = mag_mt19937_step(rem, next, state); /* Rejection sampling */
                while (mag_unlikely(r >= lim));
                o[i] = (int32_t)((int64_t)min + (int32_t)(r%span));
            }
        } break;
        case MAG_PRNG_PCG: { /* Use Permuted Congruential Generator. */
            uint64_t* state = &prng->pcg.state;
            uint64_t inc = prng->pcg.inc;
            for (int64_t i=0; i < numel; ++i) {
                uint64_t r;
                do r = mag_pcg_step(state, inc); /* Rejection sampling */
                while (mag_unlikely(r >= lim));
                o[i] = (int32_t)((int64_t)min + (int32_t)(r%span));
            }
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

static mag_E8M23 MAG_HOTPROC mag_vdot_e8m23(int64_t numel, const mag_E8M23* _Nonnull restrict x, const mag_E8M23* _Nonnull restrict y) {
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        int64_t k = numel & -16;
        float32x4_t acc[4] = {vdupq_n_f32(0)};
        float32x4_t vx[4];
        float32x4_t vy[4];
        for (int64_t i=0; i < k; i += 16) { /* Process STEP elements at a time */
            vx[0] = vld1q_f32(x+i+(0<<2));
            vy[0] = vld1q_f32(y+i+(0<<2));
            acc[0] = vfmaq_f32(acc[0], vx[0], vy[0]);
            vx[1] = vld1q_f32(x+i+(1<<2));
            vy[1] = vld1q_f32(y+i+(1<<2));
            acc[1] = vfmaq_f32(acc[1], vx[1], vy[1]);
            vx[2] = vld1q_f32(x+i+(2<<2));
            vy[2] = vld1q_f32(y+i+(2<<2));
            acc[2] = vfmaq_f32(acc[2], vx[2], vy[2]);
            vx[3] = vld1q_f32(x+i+(3<<2));
            vy[3] = vld1q_f32(y+i+(3<<2));
            acc[3] = vfmaq_f32(acc[3], vx[3], vy[3]);
        }
        acc[1] = vaddq_f32(acc[1], acc[3]); /* Fold acc[1] += acc[3] */
        *acc = vaddq_f32(*acc, acc[2]);     /* Fold acc[0] += acc[2] */
        *acc = vaddq_f32(*acc, acc[1]);     /* Fold acc[0] += acc[1] */
        mag_E8M23 sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
        for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
        return sum;
    #elif defined(__AVX512F__) && defined(__FMA__)
        int64_t k = numel & -64;
        __m512 acc[4] = {_mm512_setzero_ps()};
        __m512 vx[4];
        __m512 vy[4];
        for (int64_t i=0; i < k; i += 64) {
            vx[0] = _mm512_loadu_ps(x+i+(0<<4));
            vy[0] = _mm512_loadu_ps(y+i+(0<<4));
            acc[0] = _mm512_fmadd_ps(vx[0], vy[0], acc[0]);
            vx[1] = _mm512_loadu_ps(x+i+(1<<4));
            vy[1] = _mm512_loadu_ps(y+i+(1<<4));
            acc[1] = _mm512_fmadd_ps(vx[1], vy[1], acc[1]);
            vx[2] = _mm512_loadu_ps(x+i+(2<<4));
            vy[2] = _mm512_loadu_ps(y+i+(2<<4));
            acc[2] = _mm512_fmadd_ps(vx[2], vy[2], acc[2]);
            vx[3] = _mm512_loadu_ps(x+i+(3<<4));
            vy[3] = _mm512_loadu_ps(y+i+(3<<4));
            acc[3] = _mm512_fmadd_ps(vx[3], vy[3], acc[3]);
        }
        acc[1] = _mm512_add_ps(acc[1], acc[3]);
        *acc = _mm512_add_ps(*acc, acc[2]);
        *acc = _mm512_add_ps(*acc, acc[1]);
        mag_E8M23 sum = _mm512_reduce_add_ps(*acc);
        for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
        return sum;
    #elif defined(__AVX__) && defined(__FMA__)
        int64_t k = numel & -32;
        __m256 acc[4] = {_mm256_setzero_ps()};
        __m256 vx[4];
        __m256 vy[4];
        for (int64_t i=0; i < k; i += 32) {
            vx[0] = _mm256_loadu_ps(x+i+(0<<3));
            vy[0] = _mm256_loadu_ps(y+i+(0<<3));
            acc[0] = _mm256_fmadd_ps(vx[0], vy[0], acc[0]);
            vx[1] = _mm256_loadu_ps(x+i+(1<<3));
            vy[1] = _mm256_loadu_ps(y+i+(1<<3));
            acc[1] = _mm256_fmadd_ps(vx[1], vy[1], acc[1]);
            vx[2] = _mm256_loadu_ps(x+i+(2<<3));
            vy[2] = _mm256_loadu_ps(y+i+(2<<3));
            acc[2] = _mm256_fmadd_ps(vx[2], vy[2], acc[2]);
            vx[3] = _mm256_loadu_ps(x+i+(3<<3));
            vy[3] = _mm256_loadu_ps(y+i+(3<<3));
            acc[3] = _mm256_fmadd_ps(vx[3], vy[3], acc[3]);
        }
        acc[1] = _mm256_add_ps(acc[1], acc[3]);
        *acc = _mm256_add_ps(*acc, acc[2]);
        *acc = _mm256_add_ps(*acc, acc[1]);
        __m128 v0 = _mm_add_ps(_mm256_castps256_ps128(*acc), _mm256_extractf128_ps(*acc, 1));
        v0 = _mm_hadd_ps(v0, v0);
        v0 = _mm_hadd_ps(v0, v0);
        mag_E8M23 sum = _mm_cvtss_f32(v0);
        for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
        return sum;
    #elif defined(__SSE2__)
        int64_t k = numel & -16;
        __m128 acc[4] = {_mm_setzero_ps()};
        __m128 vx[4];
        __m128 vy[4];
        for (int64_t i=0; i < k; i += 16) {
            vx[0] = _mm_loadu_ps(x+i+(0<<2));
            vy[0] = _mm_loadu_ps(y+i+(0<<2));
            acc[0] = _mm_add_ps(acc[0], _mm_mul_ps(vx[0], vy[0]));
            vx[1] = _mm_loadu_ps(x+i+(1<<2));
            vy[1] = _mm_loadu_ps(y+i+(1<<2));
            acc[1] = _mm_add_ps(acc[1], _mm_mul_ps(vx[1], vy[1]));
            vx[2] = _mm_loadu_ps(x+i+(2<<2));
            vy[2] = _mm_loadu_ps(y+i+(2<<2));
            acc[2] = _mm_add_ps(acc[2], _mm_mul_ps(vx[2], vy[2]));
            vx[3] = _mm_loadu_ps(x+i+(3<<2));
            vy[3] = _mm_loadu_ps(y+i+(3<<2));
            acc[3] = _mm_add_ps(acc[3], _mm_mul_ps(vx[3], vy[3]));
        }
        #ifdef __SSE3__
            acc[1] = _mm_add_ps(acc[1], acc[3]);
            *acc = _mm_add_ps(*acc, acc[2]);
            *acc = _mm_add_ps(*acc, acc[1]);
            *acc = _mm_hadd_ps(*acc, *acc);
            *acc = _mm_hadd_ps(*acc, *acc);
            mag_E8M23 sum = _mm_cvtss_f32(*acc);
        #else
            __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(*acc, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            mag_E8M23 sum = _mm_cvtss_f32(sums);
        #endif
        for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
        return sum;
    #else
        mag_E11M52 r = 0.0;
        for (int64_t i=0; i < numel; ++i) r += (mag_E11M52)x[i]*(mag_E11M52)y[i];
        return (mag_E8M23)r;
    #endif
}

static mag_E5M10 MAG_HOTPROC mag_vdot_e5m10(int64_t numel, const mag_E5M10* _Nonnull restrict x, const mag_E5M10* _Nonnull restrict y) {
    mag_E8M23 r = .0f;
    for (int64_t i=0; i < numel; ++i) /* TODO: Optimize with SIMD */
        r += mag_e5m10_cvt_e8m23(x[i])*mag_e5m10_cvt_e8m23(y[i]);
    return mag_e8m23_cvt_e5m10(r);
}

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

static float32x4_t mag_simd_expf(float32x4_t x) {
    float32x4_t r = vdupq_n_f32(0x1.8p23f);
    float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    float32x4_t n = vsubq_f32(z, r);
    float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    float32x4_t u = vmulq_f32(b, b);
    float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
        vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
    uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
           vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static float32x4_t mag_simd_tanh(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.f);
    float32x4_t m1 = vdupq_n_f32(-1.f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t m2 = vdupq_n_f32(-2.0f);
    float32x4_t a = vmulq_f32(m2, x);
    float32x4_t b = mag_simd_expf(a);
    float32x4_t c = vaddq_f32(one, b);
    float32x4_t inv = vrecpeq_f32(c);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    return vaddq_f32(m1, vmulq_f32(two, inv));
}

static void mag_simd_sincos(float32x4_t x, float32x4_t* _Nonnull osin, float32x4_t* _Nonnull ocos) {
    uint32x4_t sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);
    float32x4_t y = vmulq_f32(x, vdupq_n_f32(1.27323954473516f));
    uint32x4_t emm2 = vcvtq_u32_f32(y);
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);
    uint32x4_t poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));
    x = vmlaq_f32(x, y, vdupq_n_f32(-0.78515625f));
    x = vmlaq_f32(x, y, vdupq_n_f32(-2.4187564849853515625e-4f));
    x = vmlaq_f32(x, y, vdupq_n_f32(-3.77489497744594108e-8f));
    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    uint32x4_t sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y1, y2;
    y1 = vmlaq_f32(vdupq_n_f32(-1.388731625493765e-003f), z, vdupq_n_f32(2.443315711809948e-005f));
    y2 = vmlaq_f32(vdupq_n_f32(8.3321608736e-3f), z, vdupq_n_f32(-1.9515295891e-4f));
    y1 = vmlaq_f32(vdupq_n_f32(4.166664568298827e-002f), y1, z);
    y2 = vmlaq_f32(vdupq_n_f32(-1.6666654611e-1f), y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y1 = vmlsq_f32(y1, z, vdupq_n_f32(0.5f));
    y2 = vmlaq_f32(x, y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));
    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *osin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ocos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

static __m512 mag_simd_expf(const __m512 x) {
    __m512 r = _mm512_set1_ps(0x1.8p23f);
    __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    __m512 n = _mm512_sub_ps(z, r);
    __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    __m512 u = _mm512_mul_ps(b, b);
    __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))), u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F))
    );
    __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) return res;
    __m512 zero = _mm512_setzero_ps();
    __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}

static __m512 mag_simd_tanh(__m512 x) {
    __m512 one = _mm512_set1_ps(1.f);
    __m512 neg_one = _mm512_set1_ps(-1.f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    __m512 a = _mm512_mul_ps(neg_two, x);
    __m512 b = mag_simd_expf(a);
    __m512 c = _mm512_add_ps(one, b);
    __m512 inv = _mm512_rcp14_ps(c);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    return _mm512_fmadd_ps(two, inv, neg_one);
}

#elif defined(__AVX2__) && defined(__FMA__)

static __m256 mag_simd_expf(const __m256 x) {
    __m256 r = _mm256_set1_ps(0x1.8p23f);
    __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    __m256 n = _mm256_sub_ps(z, r);
    __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),_mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    __m256 u = _mm256_mul_ps(b, b);
    __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,_mm256_set1_ps(0x1.573e2ep-5f)), u,_mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,_mm256_set1_ps(0x1.fffdb6p-2f))),u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
    __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),_mm256_set1_epi32(0x82000000u));
    __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
        _mm256_castsi256_ps(d),
        _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(c),
        _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k))))
    );
}

static __m256 mag_simd_tanh(__m256 x) {
    __m256 one = _mm256_set1_ps(1.f);
    __m256 neg_one = _mm256_set1_ps(-1.f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    __m256 a = _mm256_mul_ps(neg_two, x);
    __m256 b = mag_simd_expf(a);
    __m256 c = _mm256_add_ps(one, b);
    __m256 inv = _mm256_rcp_ps(c);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    return _mm256_fmadd_ps(two, inv, neg_one);
}

#elif defined(__SSE2__)

static __m128 mag_simd_expf(__m128 x) {
    __m128 r = _mm_set1_ps(0x1.8p23f);
    __m128 z = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0x1.715476p+0f)), r);
    __m128 n = _mm_sub_ps(z, r);
    __m128 b = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(0x1.62e4p-1f))), _mm_mul_ps(n, _mm_set1_ps(0x1.7f7d1cp-20f)));
    __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    __m128i c = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    __m128 u = _mm_mul_ps(b, b);
    __m128 j = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.0e4020p-7f), b), _mm_set1_ps(0x1.573e2ep-5f)),u),
    _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.555e66p-3f), b), _mm_set1_ps(0x1.fffdb6p-2f))), u),
    _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c)) return _mm_add_ps(_mm_mul_ps(j, k), k);
    __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),_mm_set1_epi32(0x82000000u));
    __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    __m128i d = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
        _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s2, j), s2), s1)),
        _mm_andnot_ps(_mm_castsi128_ps(c), _mm_add_ps(_mm_mul_ps(k, j), k))))
    );
}

static __m128 mag_simd_tanh(__m128 x) {
    __m128 one = _mm_set1_ps(1.f);
    __m128 neg_one = _mm_set1_ps(-1.f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 neg_two = _mm_set1_ps(-2.0f);
    __m128 a = _mm_mul_ps(neg_two, x);
    __m128 b = mag_simd_expf(a);
    __m128 c = _mm_add_ps(one, b);
    __m128 inv = _mm_rcp_ps(c);
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm_add_ps(neg_one, _mm_mul_ps(two, inv));
}

static void mag_simd_sincos(__m128 x, __m128* _Nonnull osin, __m128* _Nonnull ocos) {
    __m128 sign_mask_sin_ps = _mm_cmplt_ps(x, _mm_set1_ps(0.0f));
    __m128i sign_mask_sin = _mm_castps_si128(sign_mask_sin_ps);
    x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
    __m128 y = _mm_mul_ps(x, _mm_set1_ps(1.27323954473516f));
    __m128i emm2 = _mm_cvtps_epi32(y);
    emm2 = _mm_add_epi32(emm2, _mm_set1_epi32(1));
    emm2 = _mm_and_si128(emm2, _mm_set1_epi32(~1));
    y = _mm_cvtepi32_ps(emm2);
    __m128i poly_mask = _mm_cmpeq_epi32(emm2, _mm_set1_epi32(2));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-0.78515625f)));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-2.4187564849853515625e-4f)));
    x = _mm_add_ps(x, _mm_mul_ps(y, _mm_set1_ps(-3.77489497744594108e-8f)));
    __m128i ax = _mm_cmpeq_epi32(emm2, _mm_set1_epi32(4));
    sign_mask_sin = _mm_xor_si128(sign_mask_sin, ax);
    __m128i sign_mask_cos = _mm_cmpeq_epi32(_mm_sub_epi32(emm2, _mm_set1_epi32(2)), _mm_set1_epi32(4));
    __m128 z = _mm_mul_ps(x, x);
    __m128 y1 = _mm_add_ps(_mm_set1_ps(-1.388731625493765e-003f), _mm_mul_ps(z, _mm_set1_ps(2.443315711809948e-005f)));
    __m128 y2 = _mm_add_ps(_mm_set1_ps(8.3321608736e-3f), _mm_mul_ps(z, _mm_set1_ps(-1.9515295891e-4f)));
    y1 = _mm_add_ps(_mm_set1_ps(4.166664568298827e-002f), _mm_mul_ps(y1, z));
    y2 = _mm_add_ps(_mm_set1_ps(-1.6666654611e-1f), _mm_mul_ps(y2, z));
    y1 = _mm_mul_ps(y1, z);
    y2 = _mm_mul_ps(y2, z);
    y1 = _mm_mul_ps(y1, z);
    y1 = _mm_sub_ps(y1, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
    y2 = _mm_add_ps(x, _mm_mul_ps(y2, x));
    y1 = _mm_add_ps(y1, _mm_set1_ps(1.f));
    __m128 poly_mask_ps = _mm_castsi128_ps(poly_mask);
    __m128 ys = _mm_or_ps(_mm_and_ps(poly_mask_ps, y1), _mm_andnot_ps(poly_mask_ps, y2));
    __m128 yc = _mm_or_ps(_mm_and_ps(poly_mask_ps, y2), _mm_andnot_ps(poly_mask_ps, y1));
    __m128 sign_mask_sin_ps2 = _mm_castsi128_ps(sign_mask_sin);
    __m128 neg_ys = _mm_sub_ps(_mm_setzero_ps(), ys);
    __m128 osin_ps = _mm_or_ps(_mm_and_ps(sign_mask_sin_ps2, neg_ys), _mm_andnot_ps(sign_mask_sin_ps2, ys));
    __m128 sign_mask_cos_ps = _mm_castsi128_ps(sign_mask_cos);
    __m128 neg_yc = _mm_sub_ps(_mm_setzero_ps(), yc);
    __m128 ocos_ps = _mm_or_ps(_mm_and_ps(sign_mask_cos_ps, yc), _mm_andnot_ps(sign_mask_cos_ps, neg_yc));
    *osin = osin_ps;
    *ocos = ocos_ps;
}

#endif

static void MAG_HOTPROC mag_vfill_e8m23(int64_t numel, mag_E8M23* _Nonnull o, mag_E8M23 x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x;
}

static void MAG_HOTPROC mag_vacc_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] += x[i];
}

static void MAG_HOTPROC mag_vadd_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    #ifdef MAG_ACCELERATE
        vDSP_vadd(y, 1, x, 1, o, 1, numel);
    #else
        for (int64_t i=0; i < numel; ++i)
            o[i] = x[i] + y[i];
    #endif
}

static void MAG_HOTPROC mag_vadd_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vaddq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vadd_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vaddq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_add_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_add_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_add_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vsub_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    #ifdef MAG_ACCELERATE
        vDSP_vsub(y, 1, x, 1, o, 1, numel);
    #else
        for (int64_t i=0; i < numel; ++i)
            o[i] = x[i] - y[i];
    #endif
}

static void MAG_HOTPROC mag_vsub_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vsubq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vsub_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vsubq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_sub_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_sub_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_sub_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vmul_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    #ifdef MAG_ACCELERATE
        vDSP_vmul(y, 1, x, 1, o, 1, numel);
    #else
        for (int64_t i=0; i < numel; ++i)
            o[i] = x[i]*y[i];
    #endif
}

static void MAG_HOTPROC mag_vmul_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vmulq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vmul_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vmulq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_mul_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_mul_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_mul_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i])*mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vdiv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    #ifdef MAG_ACCELERATE
        vDSP_vdiv(y, 1, x, 1, o, 1, numel);
    #else
        for (int64_t i=0; i < numel; ++i)
            o[i] = x[i] / y[i];
    #endif
}

static void MAG_HOTPROC mag_vdiv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    int64_t i=0;
    #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (; i+7 < numel; i += 8) {
                float16x8_t va = vld1q_f16((const __fp16*)x+i);
                float16x8_t vb = vld1q_f16((const __fp16*)y+i);
                float16x8_t r = vdivq_f16(va, vb);
                vst1q_f16((__fp16*)o+i, r);
            }
            for (; i+3 < numel; i += 4) {
                float16x4_t va = vld1_f16((const __fp16*)x+i);
                float16x4_t vb = vld1_f16((const __fp16*)y+i);
                float16x4_t r = vdiv_f16(va, vb);
                vst1_f16((__fp16*)o+i, r);
            }
        #else
            for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
                float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)x+i));
                float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16*)y+i));
                float32x4_t r = vdivq_f32(va_f32, vb_f32);
                vst1_f16((__fp16*)o+i, vcvt_f16_f32(r));
            }
        #endif
    #elif defined(__AVX512F__) && defined(__AVX512FP16__)
        for (; i+31 < numel; i += 32) { /* Compute in fp16 precision directly. */
            __m512h xph = _mm512_loadu_ph(x+i);
            __m512h yph = _mm512_loadu_ph(y+i);
            __m512h rph = _mm512_div_ph(xph, yph);
            _mm512_storeu_ph(o+i, rph);
        }
    #elif defined(__AVX512F__)
        for (; i+15 < numel; i += 16) { /* Load, downcast, compute, upcast, store. */
            __m256i xph = _mm256_loadu_si256((const __m256i*)(x+i));
            __m256i yph = _mm256_loadu_si256((const __m256i*)(y+i));
            __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
            __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
            __m512 rps = _mm512_div_ps(xps, yps);
            _mm256_storeu_si256((__m256i*)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
        }
    #elif defined(__AVX__) && defined(__F16C__)
        for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
            __m128i xph = _mm_loadu_si128((const __m128i*)(x+i));
            __m128i yph = _mm_loadu_si128((const __m128i*)(y+i));
            __m256 xps = _mm256_cvtph_ps(xph);
            __m256 yps = _mm256_cvtph_ps(yph);
            __m256 sum = _mm256_div_ps(xps, yps);
            _mm_storeu_si128((__m128i*)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
        }
    #endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vpows_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = powf(x[i], y);
}

static void MAG_HOTPROC mag_vpows_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(powf(mag_e5m10_cvt_e8m23(x[i]), y));
}

static void MAG_HOTPROC mag_vadds_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y;
}

static void MAG_HOTPROC mag_vadds_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + y);
}

static void MAG_HOTPROC mag_vsubs_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y;
}

static void MAG_HOTPROC mag_vsubs_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - y);
}

static void MAG_HOTPROC mag_vmuls_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y;
}

static void MAG_HOTPROC mag_vmuls_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i])*y);
}

static void MAG_HOTPROC mag_vdivs_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y;
}

static void MAG_HOTPROC mag_vdivs_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x, mag_E8M23 y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / y);
}

static void MAG_HOTPROC mag_vabs_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fabs(x[i]);
}

static void MAG_HOTPROC mag_vabs_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(fabs(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsgn_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        o[i] = xi > 0.f ? 1.f : xi < 0.f ? -1.f : 0.f;
    }
}

static void MAG_HOTPROC mag_vsgn_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = xi > 0.f ? MAG_E5M10_ONE : xi < 0.f ? MAG_E5M10_NEG_ONE : MAG_E5M10_ZERO;
    }
}

static void MAG_HOTPROC mag_vneg_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = -x[i];
}

static void MAG_HOTPROC mag_vneg_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(-mag_e5m10_cvt_e8m23(x[i]));
}

static void MAG_HOTPROC mag_vlog_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = logf(x[i]);
}

static void MAG_HOTPROC mag_vlog_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(logf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsqr_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        o[i] = xi*xi;
    }
}

static void MAG_HOTPROC mag_vsqr_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(xi*xi);
    }
}

static void MAG_HOTPROC mag_vsqrt_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sqrtf(x[i]);
}

static void MAG_HOTPROC mag_vsqrt_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(sqrtf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsin_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinf(x[i]);
}

static void MAG_HOTPROC mag_vsin_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(sinf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vcos_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = cosf(x[i]);
}

static void MAG_HOTPROC mag_vcos_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(cosf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vstep_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.0f ? 1.f : 0.0f;
}

static void MAG_HOTPROC mag_vstep_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > 0.0f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_vexp_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expf(x[i]);
}

static void MAG_HOTPROC mag_vexp_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(expf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vfloor_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = floorf(x[i]);
}

static void MAG_HOTPROC mag_vfloor_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(floorf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vceil_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = ceilf(x[i]);
}

static void MAG_HOTPROC mag_vceil_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(ceilf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vround_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = rintf(x[i]);
}

static void MAG_HOTPROC mag_vround_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(rintf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsoftmax_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    mag_vexp_e8m23(numel, o, x);
}

static void MAG_HOTPROC mag_vsoftmax_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    mag_vexp_e5m10(numel, o, x);
}

static void MAG_HOTPROC mag_vsigmoid_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = 1.f/(1.f + expf(-x[i]));
}

static void MAG_HOTPROC mag_vsigmoid_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(1.f/(1.f + expf(-mag_e5m10_cvt_e8m23(x[i]))));
}

static void MAG_HOTPROC mag_vsigmoid_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 sig = 1.f/(1.f + expf(-x[i]));
        o[i] = sig*(1.f-sig);
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 sig = 1.f/(1.f + expf(-mag_e5m10_cvt_e8m23(x[i])));
        o[i] = mag_e8m23_cvt_e5m10(sig*(1.f-sig));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fminf(1.f, fmaxf(0.0f, (x[i] + 3.0f)/6.0f));
}

static void MAG_HOTPROC mag_vhard_sigmoid_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10( fminf(1.f, fmaxf(0.0f, (mag_e5m10_cvt_e8m23(x[i]) + 3.0f)/6.0f)));
}

static void MAG_HOTPROC mag_vsilu_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        o[i] = xi*(1.f/(1.f + expf(-xi)));
    }
}

static void MAG_HOTPROC mag_vsilu_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(xi*(1.f/(1.f + expf(-xi))));
    }
}

static void MAG_HOTPROC mag_vsilu_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        mag_E8M23 sig = 1.f/(1.f + expf(-xi));
        o[i] = sig + xi*sig;
    }
}

static void MAG_HOTPROC mag_vsilu_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        mag_E8M23 sig = 1.f/(1.f + expf(-xi));
        o[i] = mag_e8m23_cvt_e5m10(sig + xi*sig);
    }
}

static void MAG_HOTPROC mag_vtanh_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanhf(x[i]);
}

static void MAG_HOTPROC mag_vtanh_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(tanhf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtanh_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 th = tanhf(x[i]);
        o[i] = 1.f - th*th;
    }
}

static void MAG_HOTPROC mag_vtanh_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 th = tanhf(mag_e5m10_cvt_e8m23(x[i]));
        o[i] = mag_e8m23_cvt_e5m10(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vrelu_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmaxf(0.f, x[i]);
}

static void MAG_HOTPROC mag_vrelu_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(fmaxf(0.f, mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vrelu_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.f ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vrelu_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > 0.f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_vgelu_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        o[i] = .5f*xi*(1.f+erff(xi*MAG_INVSQRT2));
    }
}

static void MAG_HOTPROC mag_vgelu_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(.5f*xi*(1.f+erff(xi*MAG_INVSQRT2)));
    }
}

static void MAG_HOTPROC mag_vgelu_dv_e8m23(int64_t numel, mag_E8M23* _Nonnull o, const mag_E8M23* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = x[i];
        mag_E8M23 th = tanhf(xi);
        o[i] = .5f*(1.f + th) + .5f*xi*(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vgelu_dv_e5m10(int64_t numel, mag_E5M10* _Nonnull o, const mag_E5M10* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_E8M23 xi = mag_e5m10_cvt_e8m23(x[i]);
        mag_E8M23 th = tanhf(xi);
        o[i] = mag_e8m23_cvt_e5m10(.5f*(1.f + th) + .5f*xi*(1.f - th*th));
    }
}

static mag_E11M52 MAG_HOTPROC mag_vsum_f64_e8m23(int64_t numel, const mag_E8M23* _Nonnull x) {
    #ifdef MAG_ACCELERATE
        mag_E8M23 sum;
        vDSP_sve(x, 1, &sum, numel);
        return (mag_E11M52)sum;
    #else
        mag_E11M52 sum = 0.0;
        for (int64_t i=0; i < numel; ++i)
            sum += (mag_E11M52)x[i];
        return sum;
    #endif
}

static mag_E11M52 MAG_HOTPROC mag_vsum_f64_e5m10(int64_t numel, const mag_E5M10* _Nonnull x) {
    mag_E11M52 sum = 0.0;
    for (int64_t i=0; i < numel; ++i)
        sum += mag_e5m10_cvt_e8m23(x[i]);
    return sum;
}

static mag_E8M23 MAG_HOTPROC mag_vmin_e8m23(int64_t numel, const mag_E8M23* _Nonnull x) {
    mag_E8M23 min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, x[i]);
    return min;
}

static mag_E8M23 MAG_HOTPROC mag_vmin_e5m10(int64_t numel, const mag_E5M10* _Nonnull x) {
    mag_E8M23 min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, mag_e5m10_cvt_e8m23(x[i]));
    return min;
}

static mag_E8M23 MAG_HOTPROC mag_vmax_e8m23(int64_t numel, const mag_E8M23* _Nonnull x) {
    mag_E8M23 min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, x[i]);
    return min;
}

static mag_E8M23 MAG_HOTPROC mag_vmax_e5m10(int64_t numel, const mag_E5M10* _Nonnull x) {
    mag_E8M23 min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, mag_e5m10_cvt_e8m23(x[i]));
    return min;
}

static void mag_vand_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x, const uint8_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] & y[i];
}

static void mag_vor_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x, const uint8_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] | y[i];
}

static void mag_vxor_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x, const uint8_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ^ y[i];
}

static void mag_vnot_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = !x[i];
}

static void mag_vadd_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y[i];
}

static void mag_vsub_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y[i];
}

static void mag_vmul_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y[i];
}

static void mag_vdiv_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y[i];
}

static void mag_vand_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] & y[i];
}

static void mag_vor_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] | y[i];
}

static void mag_vxor_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ^ y[i];
}

static void mag_vshl_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] << (y[i]&31);
}

static void mag_vshr_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >> (y[i]&31);
}

static void mag_vnot_i32(int64_t numel, int32_t* _Nonnull o, const int32_t* _Nonnull x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = ~x[i];
}

static void MAG_HOTPROC mag_veq_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] == y[i];
    }
}

static void MAG_HOTPROC mag_veq_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits == y[i].bits;
}

static void MAG_HOTPROC mag_veq_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x, const uint8_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] == y[i];
}

static void MAG_HOTPROC mag_veq_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] == y[i];
}

static void MAG_HOTPROC mag_vne_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] != y[i];
}

static void MAG_HOTPROC mag_vne_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits != y[i].bits;
}

static void MAG_HOTPROC mag_vne_bool(int64_t numel, uint8_t* _Nonnull o, const uint8_t* _Nonnull x, const uint8_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] != y[i];
}

static void MAG_HOTPROC mag_vne_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] != y[i];
}

static void MAG_HOTPROC mag_vle_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] <= y[i];
}

static void MAG_HOTPROC mag_vle_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) <= mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vle_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] <= y[i];
}

static void MAG_HOTPROC mag_vge_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >= y[i];
}

static void MAG_HOTPROC mag_vge_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) >= mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vge_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >= y[i];
}

static void MAG_HOTPROC mag_vlt_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] < y[i];
}

static void MAG_HOTPROC mag_vlt_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) < mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vlt_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] < y[i];
}

static void MAG_HOTPROC mag_vgt_e8m23(int64_t numel, uint8_t* _Nonnull o, const mag_E8M23* _Nonnull x, const mag_E8M23* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > y[i];
}

static void MAG_HOTPROC mag_vgt_e5m10(int64_t numel, uint8_t* _Nonnull o, const mag_E5M10* _Nonnull x, const mag_E5M10* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vgt_i32(int64_t numel, uint8_t* _Nonnull o, const int32_t* _Nonnull x, const int32_t* _Nonnull y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > y[i];
}

static void mag_blas_nop(const mag_CPUKernelPayload* _Nonnull payload) { (void)payload; }

static MAG_AINLINE int64_t mag_offset_from_flat(const mag_Tensor* _Nonnull t, int64_t i) {
    int64_t off = 0;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t coord = i % t->shape[d];
        i /= t->shape[d];
        off += coord*t->strides[d];
    }
    return off;
}

static MAG_HOTPROC void mag_blas_clone_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    if (mag_likely(mag_tensor_is_contiguous(x))) { /* Fast path for contiguous X */
        memcpy(br, bx, mag_tensor_get_data_size(r));
        return;
    }
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static MAG_HOTPROC void mag_blas_clone_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    if (mag_likely(mag_tensor_is_contiguous(x))) { /* Fast path for contiguous X */
        memcpy(br, bx, mag_tensor_get_data_size(r));
        return;
    }
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static MAG_HOTPROC void mag_blas_clone_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    if (mag_likely(mag_tensor_is_contiguous(x))) { /* Fast path for contiguous X */
        memcpy(br, bx, mag_tensor_get_data_size(r));
        return;
    }
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static MAG_HOTPROC void mag_blas_clone_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    if (mag_likely(mag_tensor_is_contiguous(x))) { /* Fast path for contiguous X */
        memcpy(br, bx, mag_tensor_get_data_size(r));
        return;
    }
    for (int64_t i=0; i < r->numel; ++i) {
        int64_t off_src = mag_offset_from_flat(x, i);
        br[i] = bx[off_src];
    }
}

static MAG_HOTPROC void mag_blas_init_broadcast_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 xi = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    if (xi == 0.0f) {
        memset(b_r, 0, mag_tensor_get_data_size(r));
        return;
    }
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static MAG_HOTPROC void mag_blas_init_broadcast_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E5M10 xi = mag_e8m23_cvt_e5m10(mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]));
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static MAG_HOTPROC void mag_blas_init_broadcast_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    bool xi = mag_op_param_unpack_i64_or_panic(r->init_op_params[0]) != 0;
    uint8_t* b_r = mag_boolp_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static MAG_HOTPROC void mag_blas_init_broadcast_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    int32_t xi = (int32_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[0]);
    int32_t* b_r = mag_i32p_mut(r);
    int64_t numel = r->numel;
    for (int64_t i=0; i < numel; ++i)
        b_r[i] = xi;
}

static MAG_HOTPROC void mag_blas_init_masked_broadcast_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 yi = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    const mag_Tensor* mask = (const mag_Tensor*)(uintptr_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[1]);
    mag_E8M23* br = mag_e8m23p_mut(r);
    const uint8_t* bm = mag_boolp(mask);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t rx = r->rank - mask->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && mask->shape[dx] > 1)
                xi += coord*mask->strides[dx];
        }
        if (bm[xi]) br[i] = yi;
    }
}

static MAG_HOTPROC void mag_blas_init_masked_broadcast_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E5M10 yi = mag_e8m23_cvt_e5m10(mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]));
    const mag_Tensor* mask = (const mag_Tensor*)(uintptr_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[1]);
    mag_E5M10* br = mag_e5m10p_mut(r);
    const uint8_t* bm = mag_boolp(mask);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t rx = r->rank - mask->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && mask->shape[dx] > 1)
                xi += coord*mask->strides[dx];
        }
        if (bm[xi]) br[i] = yi;
    }
}

static MAG_HOTPROC void mag_blas_init_masked_broadcast_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    int32_t yi = (int32_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[0]);
    const mag_Tensor* mask = (const mag_Tensor*)(uintptr_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[1]);
    int32_t* br = mag_i32p_mut(r);
    const uint8_t* bm = mag_boolp(mask);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t rx = r->rank - mask->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && mask->shape[dx] > 1)
                xi += coord*mask->strides[dx];
        }
        if (bm[xi]) br[i] = yi;
    }
}

static MAG_HOTPROC void mag_blas_init_masked_broadcast_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    uint8_t yi = (uint8_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[0]);
    const mag_Tensor* mask = (const mag_Tensor*)(uintptr_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[1]);
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bm = mag_boolp(mask);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t rx = r->rank - mask->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && mask->shape[dx] > 1)
                xi += coord*mask->strides[dx];
        }
        if (bm[xi]) br[i] = yi;
    }
}

static MAG_HOTPROC void mag_blas_init_rand_uniform_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 min = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 max = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_vrand_uniform_e8m23(payload->local_prng, numel, b_r, min, max);
}

static MAG_HOTPROC void mag_blas_init_rand_uniform_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 min = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 max = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_vrand_uniform_e5m10(payload->local_prng, numel, b_r, min, max);
}

static MAG_HOTPROC void mag_blas_init_rand_uniform_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    int32_t min = (int32_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[0]);
    int32_t max = (int32_t)mag_op_param_unpack_i64_or_panic(r->init_op_params[1]);
    int32_t* b_r = mag_i32p_mut(r);
    int64_t numel = r->numel;
    mag_vrand_uniform_i32(payload->local_prng, numel, b_r, min, max);
}

static MAG_HOTPROC void mag_blas_init_rand_normal_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 mean = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 stddev = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E8M23* b_r = mag_e8m23p_mut(r);
    int64_t numel = r->numel;
    mag_vrand_normal_e8m23(payload->local_prng, numel, b_r, mean, stddev);
}

static MAG_HOTPROC void mag_blas_init_rand_normal_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 mean = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    mag_E8M23 stddev = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[1]);
    mag_E5M10* b_r = mag_e5m10p_mut(r);
    int64_t numel = r->numel;
    mag_vrand_normal_e5m10(payload->local_prng, numel, b_r, mean, stddev);
}

static MAG_HOTPROC void mag_blas_init_rand_bernoulli_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    mag_E8M23 p = mag_op_param_unpack_e8m23_or_panic(r->init_op_params[0]);
    uint8_t* b_r = mag_boolp_mut(r);
    int64_t numel = r->numel;
    mag_vrand_bernoulli_bool(payload->local_prng, numel, b_r, p);
}

#define mag_cpu_blas_impl_reduce(T, DT, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_blas_##FUNC##_##T(const mag_CPUKernelPayload* _Nonnull payload) { \
        mag_Tensor* r = payload->node; \
        const mag_Tensor* x = r->op_inputs[0]; \
        if (payload->thread_idx != 0) return; \
        const mag_##DT* bx = mag_##T##p(x); \
        mag_##DT* br = mag_##T##p_mut(r); \
        ACC_T acc = (INIT_EXPR); \
        for (int64_t i=0; i < x->numel; ++i) { \
            int64_t off = mag_offset_from_flat(x, i); \
            mag_bnd_chk(bx+off, bx, mag_tensor_get_data_size(x)); \
            UPDATE_STMT; \
        } \
        FINAL_STMT; \
    }

static void MAG_HOTPROC mag_blas_abs_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vabs_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vabs_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sgn_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsgn_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsgn_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_neg_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vneg_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vneg_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_log_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vlog_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vlog_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sqr_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsqr_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsqr_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sqrt_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsqrt_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsqrt_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sin_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsin_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsin_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_cos_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vcos_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vcos_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_step_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vstep_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vstep_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_exp_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vexp_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vexp_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_floor_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vfloor_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vfloor_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_ceil_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vceil_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vceil_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_round_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vround_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vround_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_softmax_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t last_dim = r->shape[r->rank-1];
    int64_t num_rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rows_per_thread = (num_rows + tc - 1)/tc;
    int64_t start_row = ti*rows_per_thread;
    int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
    for (int64_t row = start_row; row < end_row; ++row) {
        const mag_E8M23* row_in = bx + row*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
        mag_E8M23* row_out = br + row*last_dim;
        mag_E8M23 max_val = row_in[0]; /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            if (row_in[i] > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = row_in[i];
            }
        }
        mag_E8M23 sum = 0.0f;
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

static void MAG_HOTPROC mag_blas_softmax_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsoftmax_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsoftmax_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsigmoid_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsigmoid_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsigmoid_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsigmoid_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_hard_sigmoid_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vhard_sigmoid_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vhard_sigmoid_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_silu_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsilu_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsilu_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_silu_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsilu_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsilu_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tanh_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vtanh_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vtanh_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tanh_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vtanh_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vtanh_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_relu_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vrelu_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vrelu_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_relu_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vrelu_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vrelu_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_gelu_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vgelu_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vgelu_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_gelu_dv_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vgelu_dv_e8m23(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vgelu_dv_e8m23(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tril_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row <= diag ? bx[i] : 0.f; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row <= diag ? bx[xi] : 0.f; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_triu_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row >= diag ? bx[i] : 0.f; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row >= diag ? bx[xi] : 0.f; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_add_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vadd_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] + by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] + py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] + by[yi];
    }
}

static void MAG_HOTPROC mag_blas_sub_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vsub_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] - by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] - py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] - by[yi];
    }
}

static void MAG_HOTPROC mag_blas_mul_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vmul_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra]*by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi]*py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi]*by[yi];
    }
}

static void MAG_HOTPROC mag_blas_div_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vdiv_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] / by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        mag_E8M23* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] / py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] / by[yi];
    }
}

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, sum, mag_E11M52, 0.0, \
    acc += (mag_E11M52)bx[off];, \
    *br = (mag_E8M23)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, mean, mag_E11M52, 0.0, \
    acc += (mag_E11M52)bx[off];, \
    acc /= (mag_E11M52)x->numel; *br = (mag_E8M23)acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, min, mag_E8M23, INFINITY, \
    acc = fminf(acc, bx[off]);, \
    *br = acc; )

mag_cpu_blas_impl_reduce( \
    e8m23, E8M23, max, mag_E8M23, -INFINITY, \
    acc = fmaxf(acc, bx[off]);, \
    *br = acc; )

static void MAG_HOTPROC mag_blas_abs_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vabs_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vabs_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sgn_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsgn_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsgn_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_neg_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vneg_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vneg_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_log_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vlog_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vlog_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sqr_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsqr_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsqr_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sqrt_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsqrt_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsqrt_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sin_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsin_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsin_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_cos_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vcos_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vcos_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_step_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vstep_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vstep_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_exp_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vexp_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vexp_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_floor_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vfloor_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vfloor_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_ceil_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vceil_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vceil_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_round_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vround_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vround_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_softmax_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t last_dim = r->shape[r->rank-1];
    int64_t num_rows = r->numel / last_dim;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t rows_per_thread = (num_rows + tc - 1)/tc;
    int64_t start_row = ti*rows_per_thread;
    int64_t end_row = (start_row + rows_per_thread) < num_rows ? (start_row + rows_per_thread) : num_rows;
    for (int64_t row = start_row; row < end_row; ++row) {
        const mag_E5M10* row_in = bx + row*last_dim;
        mag_bnd_chk(row_in, bx, mag_tensor_get_data_size(x));
        mag_E5M10* row_out = br + row*last_dim;
        mag_E8M23 max_val = mag_e5m10_cvt_e8m23(row_in[0]);  /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            mag_E8M23 fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
            if (fp32_row > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = fp32_row;
            }
        }
        mag_E8M23 sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
            mag_E8M23 fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
            mag_E8M23 exp = expf(fp32_row - max_val);
            row_out[i] = mag_e8m23_cvt_e5m10(exp); /* -max for numerical stability */
            sum += exp;
        }
        for (int64_t i=0; i < last_dim; ++i) {
            row_out[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(row_out[i]) / sum);
        }
    }
}

static void MAG_HOTPROC mag_blas_softmax_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsoftmax_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsoftmax_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsigmoid_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsigmoid_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsigmoid_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsigmoid_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_hard_sigmoid_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vhard_sigmoid_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vhard_sigmoid_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_silu_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsilu_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsilu_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_silu_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vsilu_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vsilu_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tanh_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vtanh_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vtanh_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tanh_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vtanh_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vtanh_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_relu_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vrelu_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vrelu_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_relu_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vrelu_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vrelu_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_gelu_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vgelu_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vgelu_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_gelu_dv_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vgelu_dv_e5m10(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vgelu_dv_e5m10(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tril_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row <= diag ? bx[i] : MAG_E5M10_ZERO; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row <= diag ? bx[xi] : MAG_E5M10_ZERO; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_triu_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row >= diag ? bx[i] : MAG_E5M10_ZERO; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row >= diag ? bx[xi] : MAG_E5M10_ZERO; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_add_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vadd_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(px[i-ra]) + mag_e5m10_cvt_e8m23(by[yi]));
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) + mag_e5m10_cvt_e8m23(py[i-ra]));
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) + mag_e5m10_cvt_e8m23(by[yi]));
    }
}

static void MAG_HOTPROC mag_blas_sub_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vsub_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(px[i-ra]) - mag_e5m10_cvt_e8m23(by[yi]));
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) - mag_e5m10_cvt_e8m23(py[i-ra]));
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) - mag_e5m10_cvt_e8m23(by[yi]));
    }
}

static void MAG_HOTPROC mag_blas_mul_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vmul_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(px[i-ra])*mag_e5m10_cvt_e8m23(by[yi]));
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi])*mag_e5m10_cvt_e8m23(py[i-ra]));
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi])*mag_e5m10_cvt_e8m23(by[yi]));
    }
}

static void MAG_HOTPROC mag_blas_div_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vdiv_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(px[i-ra]) / mag_e5m10_cvt_e8m23(by[yi]));
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        mag_E5M10* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) / mag_e5m10_cvt_e8m23(py[i-ra]));
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(bx[xi]) / mag_e5m10_cvt_e8m23(by[yi]));
    }
}

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, sum, mag_E8M23, 0.0f, \
    acc += mag_e5m10_cvt_e8m23(bx[off]);, \
    *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, mean, mag_E8M23, 0.0f, \
    acc += mag_e5m10_cvt_e8m23(bx[off]);, \
    acc /= (mag_E8M23)x->numel; *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, min, mag_E8M23, INFINITY, \
    acc = fminf(acc, mag_e5m10_cvt_e8m23(bx[off]));, \
    *br = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_blas_impl_reduce( \
    e5m10, E5M10, max, mag_E8M23, -INFINITY, \
    acc = fmaxf(acc, mag_e5m10_cvt_e8m23(bx[off]));, \
    *br = mag_e8m23_cvt_e5m10(acc); )

#define VLA(type, name, size) \
type* name = (type*)(*mag_alloc)(NULL, (size)*sizeof(type),0)

static int64_t mag_offset_rmn(const mag_Tensor* _Nonnull t, int64_t flat, int64_t i, int64_t j) {
    int64_t ra = t->rank;
    const int64_t* restrict td = t->shape;
    const int64_t* restrict ts = t->strides;
    if (mag_likely(ra <= 3)) { /* Fast path */
        switch (ra) {
            case 1: return i*ts[0];
            case 2: return i*ts[0] + j*ts[1];
            case 3: return flat*ts[0] + i*ts[1] + j*ts[2];
            default: mag_panic("invalid rank: %" PRIi64, ra);
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

static MAG_HOTPROC mag_E8M23* _Nonnull mag_mm_pack_x_e8m23(mag_E8M23* _Nonnull xbuf, int64_t M, int64_t K, int64_t xb, const mag_Tensor* _Nonnull x, const mag_E8M23* _Nonnull px) {
    for (int64_t i=0; i < M; ++i) {
        for (int64_t k=0; k < K; ++k) {
            size_t j = mag_offset_rmn(x, xb, i, k);
            mag_bnd_chk(px+j, px, mag_tensor_get_data_size(x));
            mag_bnd_chk(xbuf + i*K + k, xbuf, M*K*sizeof(*xbuf));
            xbuf[i*K + k] = px[j];
        }
    }
    return xbuf;
}

static MAG_HOTPROC mag_E5M10* _Nonnull mag_mm_pack_x_e5m10(mag_E5M10* _Nonnull xbuf, int64_t M, int64_t K, int64_t xb, const mag_Tensor* _Nonnull x, const mag_E5M10* _Nonnull px) {
    for (int64_t i=0; i < M; ++i) {
        for (int64_t k=0; k < K; ++k) {
            size_t j = mag_offset_rmn(x, xb, i, k);
            mag_bnd_chk(px+j, px, mag_tensor_get_data_size(x));
            mag_bnd_chk(xbuf + i*K + k, xbuf, M*K*sizeof(*xbuf));
            xbuf[i*K + k] = px[j];
        }
    }
    return xbuf;
}

static MAG_HOTPROC mag_E8M23* _Nonnull mag_mm_pack_y_e8m23(mag_E8M23* _Nonnull ybuf, int64_t K, int64_t N, int64_t yb, const mag_Tensor* _Nonnull y, const mag_E8M23* _Nonnull py) {
    if (y->rank == 1) {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[k];
            }
        }
    } else {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[mag_offset_rmn(y, yb, k, n)];
            }
        }
    }
    return ybuf;
}

static MAG_HOTPROC mag_E5M10* _Nonnull mag_mm_pack_y_e5m10(mag_E5M10* _Nonnull ybuf, int64_t K, int64_t N, int64_t yb, const mag_Tensor* _Nonnull y, const mag_E5M10* _Nonnull py) {
    if (y->rank == 1) {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[k];
            }
        }
    } else {
        for (int64_t k=0; k < K; ++k) {
            for (int64_t n=0; n < N; ++n) {
                ybuf[n*K + k] = py[mag_offset_rmn(y, yb, k, n)];
            }
        }
    }
    return ybuf;
}

typedef struct mag_MMScratchBuf {
    void* _Nonnull p;
    size_t cap;
} mag_MMScratchBuf;

#define MAG_MM_SCRATCH_BUG_ALIGN MAG_DESTRUCTIVE_INTERFERENCE_SIZE

static void* _Nonnull mag_mm_scratch_acquire(mag_MMScratchBuf* _Nonnull sb, size_t size) {
    if (size <= sb->cap) return sb->p; /* We have enough space */
    void* p = (*mag_alloc)(NULL, size, MAG_MM_SCRATCH_BUG_ALIGN);
    if (sb->p) (*mag_alloc)(sb->p, 0, MAG_MM_SCRATCH_BUG_ALIGN);
    sb->p = p;
    sb->cap = size;
    return p;
}

static void mag_mm_scratch_release(mag_MMScratchBuf* _Nonnull sb) {
    if (sb->p) (*mag_alloc)(sb->p, 0, MAG_MM_SCRATCH_BUG_ALIGN);
    sb->p = NULL;
    sb->cap = 0;
}

static MAG_HOTPROC void mag_blas_matmul_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r  = payload->node;
    const mag_Tensor* x  = r->op_inputs[0];
    const mag_Tensor* y  = r->op_inputs[1];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t M = x->rank == 1 ? 1 : x->shape[x->rank - 2];
    int64_t N = y->rank == 1 ? 1 : y->shape[y->rank - 1];
    int64_t K = x->shape[x->rank - 1];
    int64_t bdr = r->rank > 2 ? r->rank-2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d) batch_total *= r->shape[d];
    int64_t bdx = x->rank > 2 ? x->rank-2 : 0;
    int64_t bdy = y->rank > 2 ? y->rank-2 : 0;
    bool x_row = mag_tensor_is_contiguous(x) && x->strides[x->rank-1] == 1;
    size_t scratch_sz = sizeof(mag_E8M23)*(K*N + (x_row ? 0 : M*K));   /* y-panel mandatory, x-panel optional */
    static MAG_THREAD_LOCAL mag_MMScratchBuf sb; /* auto-reused, TODO: free this buffer  */
    mag_E8M23* scratch = mag_mm_scratch_acquire(&sb, scratch_sz);
    mag_E8M23* xbuf = x_row ? NULL : scratch;
    mag_E8M23* ybuf = scratch + (x_row ? 0 : M*K);
    int64_t idx_r[4] = {0};
    for (int64_t b=0; b < batch_total; ++b) {
        int64_t rem = b;
        for (int64_t d = bdr-1; d >= 0; --d) {
            idx_r[d] = rem % r->shape[d];
            rem /= r->shape[d];
        }
        int64_t xb_flat = 0, yb_flat = 0;
        if (bdx) {
            for (int64_t d=0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = x->shape[d] == 1 ? 0 : idx_r[rd];
                xb_flat = xb_flat * x->shape[d] + idx;
            }
        }
        if (bdy) {
            for (int64_t d=0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = y->shape[d] == 1 ? 0 : idx_r[rd];
                yb_flat = yb_flat * y->shape[d] + idx;
            }
        }
        const mag_E8M23* px = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        mag_E8M23* pr = br + mag_offset_rmn(r,b, 0, 0);
        const mag_E8M23* restrict A = x_row ? px : mag_mm_pack_x_e8m23(xbuf, M, K, xb_flat, x, bx);
        const mag_E8M23* restrict B = mag_mm_pack_y_e8m23(ybuf, K, N, yb_flat, y, by);
        mag_E8M23* restrict C = pr;
        for (int64_t i=0; i < M; ++i) {
            const mag_E8M23* restrict a_row = A + i*K;
            for (int64_t n=0; n < N; ++n) {
                const mag_E8M23* restrict b_col = B + n*K;
                C[i*N + n] = mag_vdot_e8m23(K, b_col, a_row);
            }
        }
    }
}

static MAG_HOTPROC void mag_blas_matmul_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r  = payload->node;
    const mag_Tensor* x  = r->op_inputs[0];
    const mag_Tensor* y  = r->op_inputs[1];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t M = x->rank == 1 ? 1 : x->shape[x->rank - 2];
    int64_t N = y->rank == 1 ? 1 : y->shape[y->rank - 1];
    int64_t K = x->shape[x->rank - 1];
    int64_t bdr = r->rank > 2 ? r->rank-2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d) batch_total *= r->shape[d];
    int64_t bdx = x->rank > 2 ? x->rank-2 : 0;
    int64_t bdy = y->rank > 2 ? y->rank-2 : 0;
    bool x_row = mag_tensor_is_contiguous(x) && x->strides[x->rank-1] == 1;
    size_t scratch_sz = sizeof(mag_E5M10)*(K*N + (x_row ? 0 : M*K));   /* y-panel mandatory, x-panel optional */
    static MAG_THREAD_LOCAL mag_MMScratchBuf sb; /* auto-reused, TODO: free this buffer  */
    mag_E5M10* scratch = mag_mm_scratch_acquire(&sb, scratch_sz);
    mag_E5M10* xbuf = x_row ? NULL : scratch;
    mag_E5M10* ybuf = scratch + (x_row ? 0 : M*K);
    int64_t idx_r[4] = {0};
    for (int64_t b=0; b < batch_total; ++b) {
        int64_t rem = b;
        for (int64_t d = bdr-1; d >= 0; --d) {
            idx_r[d] = rem % r->shape[d];
            rem /= r->shape[d];
        }
        int64_t xb_flat = 0, yb_flat = 0;
        if (bdx) {
            for (int64_t d=0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = x->shape[d] == 1 ? 0 : idx_r[rd];
                xb_flat = xb_flat * x->shape[d] + idx;
            }
        }
        if (bdy) {
            for (int64_t d=0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = y->shape[d] == 1 ? 0 : idx_r[rd];
                yb_flat = yb_flat * y->shape[d] + idx;
            }
        }
        const mag_E5M10* px = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        mag_E5M10* pr = br + mag_offset_rmn(r,b, 0, 0);
        const mag_E5M10* restrict A = x_row ? px : mag_mm_pack_x_e5m10(xbuf, M, K, xb_flat, x, bx);
        const mag_E5M10* restrict B = mag_mm_pack_y_e5m10(ybuf, K, N, yb_flat, y, by);
        mag_E5M10* restrict C = pr;
        for (int64_t i=0; i < M; ++i) {
            const mag_E5M10* restrict a_row = A + i*K;
            for (int64_t n=0; n < N; ++n) {
                const mag_E5M10* restrict b_col = B + n*K;
                C[i*N + n] = mag_vdot_e5m10(K, b_col, a_row);
            }
        }
    }
}

static void MAG_HOTPROC  mag_blas_repeat_back_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E8M23* br = mag_e8m23p_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    for (int64_t i=0; i < r->numel; ++i)
        br[mag_offset_from_flat(r, i)] = 0;
    int64_t rx = r->rank;
    int64_t xx = x->rank;
    int64_t shift = xx - rx;
    for (int64_t flat=0; flat < x->numel; ++flat) {
        int64_t ax = flat;
        int64_t xoff = 0;
        int64_t roff = 0;
        for (int64_t d = xx-1; d >= 0; --d) {
            int64_t coord = ax % x->shape[d];
            ax /= x->shape[d];
            xoff += coord*x->strides[d];
            int64_t rd = d - shift;
            if (rd >= 0) {
                int64_t rcoord = coord % r->shape[rd];
                roff += rcoord*r->strides[rd];
            }
        }
        br[roff] += bx[xoff];
    }
}

static void MAG_HOTPROC mag_blas_repeat_back_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    if (payload->thread_idx != 0) return;
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    mag_E5M10* br = mag_e5m10p_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    for (int64_t i=0; i < r->numel; ++i)
        br[mag_offset_from_flat(r, i)].bits = 0;
    int64_t rx = r->rank;
    int64_t xx = x->rank;
    int64_t shift = xx - rx;
    for (int64_t flat=0; flat < x->numel; ++flat) {
        int64_t ax = flat;
        int64_t xoff = 0;
        int64_t roff = 0;
        for (int64_t d = xx-1; d >= 0; --d) {
            int64_t coord = ax % x->shape[d];
            ax /= x->shape[d];
            xoff += coord*x->strides[d];
            int64_t rd = d - shift;
            if (rd >= 0) {
                int64_t rcoord = coord % r->shape[rd];
                roff += rcoord*r->strides[rd];
            }
        }
        br[roff] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(br[roff]) + mag_e5m10_cvt_e8m23(bx[xoff]));
    }
}

static void MAG_HOTPROC mag_blas_and_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    const uint8_t* by = mag_boolp(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const uint8_t* px = bx + ra;
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vand_bool(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const uint8_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] & by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] & py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] & by[yi];
    }
}

static void MAG_HOTPROC mag_blas_or_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    const uint8_t* by = mag_boolp(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const uint8_t* px = bx + ra;
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vor_bool(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const uint8_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] | by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] | py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] | by[yi];
    }
}

static void MAG_HOTPROC mag_blas_xor_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    const uint8_t* by = mag_boolp(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const uint8_t* px = bx + ra;
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vxor_bool(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const uint8_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] ^ by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] ^ py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] ^ by[yi];
    }
}

static void MAG_HOTPROC mag_blas_not_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vnot_bool(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vnot_bool(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tril_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row <= diag ? bx[i] : 0; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row <= diag ? bx[xi] : 0; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_triu_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row >= diag ? bx[i] : 0; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row >= diag ? bx[xi] : 0; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_add_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vadd_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] + by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] + py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] + by[yi];
    }
}

static void MAG_HOTPROC mag_blas_sub_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vsub_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] - by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] - py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] - by[yi];
    }
}

static void MAG_HOTPROC mag_blas_mul_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vmul_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra]*by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi]*py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi]*by[yi];
    }
}

static void MAG_HOTPROC mag_blas_div_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vdiv_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] / by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] / py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] / by[yi];
    }
}

static void MAG_HOTPROC mag_blas_and_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vand_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] & by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] & py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] & by[yi];
    }
}

static void MAG_HOTPROC mag_blas_or_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vor_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] | by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] | py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] | by[yi];
    }
}

static void MAG_HOTPROC mag_blas_xor_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vxor_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] ^ by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] ^ py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] ^ by[yi];
    }
}

static void MAG_HOTPROC mag_blas_shl_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vshl_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] << (by[yi]&31);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] << (py[i-ra]&31);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] << (by[yi]&31);
    }
}

static void MAG_HOTPROC mag_blas_shr_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vshr_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] >> (by[yi]&31);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        int32_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] >> (py[i-ra]&31);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] >> (by[yi]&31);
    }
}

static void MAG_HOTPROC mag_blas_not_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) { /* Fast path if tensor is contiguous */
        mag_vnot_i32(rb - ra, br + ra, bx + ra);
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        mag_vnot_i32(1, br+i, bx+xi);
    }
}

static void MAG_HOTPROC mag_blas_tril_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row <= diag ? bx[i] : 0; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row <= diag ? bx[xi] : 0; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_triu_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    int32_t* br = mag_i32p_mut(r);
    const int32_t* bx = mag_i32p(x);
    int64_t diag = mag_op_param_unpack_i64_or_panic(r->op_params[0]);
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    int64_t cols = r->shape[r->rank-1];
    int64_t rows = r->shape[r->rank-2];
    int64_t mat = rows*cols;
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc)) {
        for (int64_t i=ra; i < rb; ++i) {
            int64_t inner = i % mat;
            int64_t row = inner/cols;
            int64_t col = inner - row*cols;
            br[i] = col-row >= diag ? bx[i] : 0; /* Zero-fill masked entries */
        }
        return;
    }
    int64_t rx = r->rank - x->rank;
    for (int64_t i=ra; i < rb; ++i) {
        int64_t ax = i;
        int64_t xi = 0;
        int64_t row = 0;
        int64_t col = 0;
        for (int64_t d=r->rank-1; d >= 0; --d) {
            int64_t coord = ax % r->shape[d];
            ax /= r->shape[d];
            if (d == r->rank-1) col = coord;
            else if (d == r->rank-2) row = coord;
            int64_t dx = d-rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
        }
        br[i] = col-row >= diag ? bx[xi] : 0; /* Zero-fill masked entries */
    }
}

static void MAG_HOTPROC mag_blas_eq_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_veq_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] == by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] == py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] == by[yi];
    }
}

static void MAG_HOTPROC mag_blas_eq_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_veq_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra].bits == by[yi].bits;
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi].bits == py[i-ra].bits;
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi].bits == by[yi].bits;
    }
}

static void MAG_HOTPROC mag_blas_eq_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    const uint8_t* by = mag_boolp(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const uint8_t* px = bx + ra;
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_veq_bool(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const uint8_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] == by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] == py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] == by[yi];
    }
}

static void MAG_HOTPROC mag_blas_eq_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_veq_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] == by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] == py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] == by[yi];
    }
}

static void MAG_HOTPROC mag_blas_ne_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vne_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] != by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] != py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] != by[yi];
    }
}

static void MAG_HOTPROC mag_blas_ne_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vne_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra].bits != by[yi].bits;
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi].bits != py[i-ra].bits;
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi].bits != by[yi].bits;
    }
}

static void MAG_HOTPROC mag_blas_ne_bool(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const uint8_t* bx = mag_boolp(x);
    const uint8_t* by = mag_boolp(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const uint8_t* px = bx + ra;
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vne_bool(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const uint8_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] != by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const uint8_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] != py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] != by[yi];
    }
}

static void MAG_HOTPROC mag_blas_ne_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vne_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] != by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] != py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] != by[yi];
    }
}

static void MAG_HOTPROC mag_blas_le_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vle_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] <= by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] <= py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] <= by[yi];
    }
}

static void MAG_HOTPROC mag_blas_le_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vle_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(px[i-ra]) <= mag_e5m10_cvt_e8m23(by[yi]);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(bx[xi]) <= mag_e5m10_cvt_e8m23(py[i-ra]);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e5m10_cvt_e8m23(bx[xi]) <= mag_e5m10_cvt_e8m23(by[yi]);
    }
}

static void MAG_HOTPROC mag_blas_le_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vle_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] <= by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] <= py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] <= by[yi];
    }
}

static void MAG_HOTPROC mag_blas_ge_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vge_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] >= by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] >= py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] >= by[yi];
    }
}

static void MAG_HOTPROC mag_blas_ge_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vge_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(px[i-ra]) >= mag_e5m10_cvt_e8m23(by[yi]);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(bx[xi]) >= mag_e5m10_cvt_e8m23(py[i-ra]);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e5m10_cvt_e8m23(bx[xi]) >= mag_e5m10_cvt_e8m23(by[yi]);
    }
}

static void MAG_HOTPROC mag_blas_ge_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vge_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] >= by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] >= py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] >= by[yi];
    }
}

static void MAG_HOTPROC mag_blas_gt_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vgt_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] > by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] > py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] > by[yi];
    }
}

static void MAG_HOTPROC mag_blas_gt_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vgt_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(px[i-ra]) > mag_e5m10_cvt_e8m23(by[yi]);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(bx[xi]) > mag_e5m10_cvt_e8m23(py[i-ra]);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e5m10_cvt_e8m23(bx[xi]) > mag_e5m10_cvt_e8m23(by[yi]);
    }
}

static void MAG_HOTPROC mag_blas_gt_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vgt_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] > by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] > py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] > by[yi];
    }
}

static void MAG_HOTPROC mag_blas_lt_e8m23(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E8M23* bx = mag_e8m23p(x);
    const mag_E8M23* by = mag_e8m23p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E8M23* px = bx + ra;
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vlt_e8m23(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E8M23* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] < by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E8M23* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] < py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] < by[yi];
    }
}

static void MAG_HOTPROC mag_blas_lt_e5m10(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const mag_E5M10* bx = mag_e5m10p(x);
    const mag_E5M10* by = mag_e5m10p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const mag_E5M10* px = bx + ra;
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vlt_e5m10(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const mag_E5M10* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(px[i-ra]) < mag_e5m10_cvt_e8m23(by[yi]);
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const mag_E5M10* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = mag_e5m10_cvt_e8m23(bx[xi]) < mag_e5m10_cvt_e8m23(py[i-ra]);
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = mag_e5m10_cvt_e8m23(bx[xi]) < mag_e5m10_cvt_e8m23(by[yi]);
    }
}

static void MAG_HOTPROC mag_blas_lt_i32(const mag_CPUKernelPayload* _Nonnull payload) {
    mag_Tensor* r = payload->node;
    const mag_Tensor* x = r->op_inputs[0];
    const mag_Tensor* y = r->op_inputs[1];
    uint8_t* br = mag_boolp_mut(r);
    const int32_t* bx = mag_i32p(x);
    const int32_t* by = mag_i32p(y);
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t total = r->numel;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    bool xc = mag_tensor_is_contiguous(x) && x->numel == total && mag_tensor_is_contiguous(r);
    bool yc = mag_tensor_is_contiguous(y) && y->numel == total && mag_tensor_is_contiguous(r);
    if (mag_likely(xc && yc)) { /* Fast path if both tensors are contiguous */
        const int32_t* px = bx + ra;
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        int64_t numel = rb - ra;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(py, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        mag_vlt_i32(numel, pr, px, py);
        return;
    }
    int64_t rx = r->rank - x->rank;
    int64_t ry = r->rank - y->rank;
    if (mag_likely(xc)) { /* Fast path if X is contiguous */
        const int32_t* px = bx + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t yi = 0;
            for (int64_t d=r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dy = d - ry;
                if (dy >= 0 && y->shape[dy] > 1)
                    yi += coord*y->strides[dy];
            }
            mag_bnd_chk(px+i-ra, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = px[i-ra] < by[yi];
        }
        return;
    }
    if (mag_likely(yc)) {  /* Fast path if Y is contiguous */
        const int32_t* py = by + ra;
        uint8_t* pr = br + ra;
        for (int64_t i=ra; i < rb; ++i) {
            int64_t ax = i;
            int64_t xi = 0;
            for (int64_t d = r->rank-1; d >= 0; --d) {
                int64_t dim = r->shape[d];
                int64_t coord = ax % dim;
                ax /= dim;
                int64_t dx = d - rx;
                if (dx >= 0 && x->shape[dx] > 1)
                    xi += coord*x->strides[dx];
            }
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(py+i-ra, by, mag_tensor_get_data_size(y));
            mag_bnd_chk(pr+i-ra, br, mag_tensor_get_data_size(r));
            pr[i-ra] = bx[xi] < py[i-ra];
        }
        return;
    }
    for (int64_t i=ra; i < rb; ++i) { /* General case */
        int64_t ax  = i;
        int64_t ri = 0;
        int64_t xi = 0;
        int64_t yi = 0;
        for (int64_t d = r->rank-1; d >= 0; --d) {
            int64_t dim = r->shape[d];
            int64_t coord = ax % dim;
            ax /= dim;
            ri += coord*r->strides[d];
            int64_t dx = d - rx;
            if (dx >= 0 && x->shape[dx] > 1)
                xi += coord*x->strides[dx];
            int64_t dy = d - ry;
            if (dy >= 0 && y->shape[dy] > 1)
                yi += coord*y->strides[dy];
        }
        mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y));
        mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r));
        br[ri] = bx[xi] < by[yi];
    }
}

#ifndef MAG_BLAS_SPECIALIZATION
#error "BLAS specialization undefined"
#endif
#ifndef MAG_BLAS_SPECIALIZATION_FEAT_REQUEST
#error "Feature request routine undefined"
#endif

#if defined(__x86_64__) || defined(_M_X64)
uint64_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST() {
    uint64_t caps = 1ull<<MAG_AMD64_CAP_SSE2; /* always required */
    #ifdef __AVX512F__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512F;
    #endif
    #ifdef __AVX512BW__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512BW;
    #endif
    #ifdef __AVX512CD__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512CD;
    #endif
    #ifdef __AVX512DQ__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512DQ;
    #endif
    #ifdef __AVX512ER__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512ER;
    #endif
    #ifdef __AVX512IFMA__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512IFMA;
    #endif
    #ifdef __AVX512PF__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512PF;
    #endif
    #ifdef __AVX512VBMI__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512VBMI;
    #endif
    #ifdef __AVX512VL__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512VL;
    #endif
    #ifdef __AVX512_4FMAPS__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_4FMAPS;
    #endif
    #ifdef __AVX512_4VNNIW__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_4VNNIW;
    #endif
    #ifdef __AVX512_FP16__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_FP16;
    #endif
    #ifdef __AVX512_BF16__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_BF16;
    #endif
    #ifdef __AVX512_BITALG__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_BITALG;
    #endif
    #ifdef __AVX512_VBMI2__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VBMI2;
    #endif
    #ifdef __AVX512_VNNI__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VNNI;
    #endif
    #ifdef __AVX512_VP2INTERSECT__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VP2INTERSECT;
    #endif
    #ifdef __AVX512_VPOPCNTDQ__
        caps |= 1ull<<MAG_AMD64_CAP_AVX512_VPOPCNTDQ;
    #endif
    #ifdef __AVX__
        caps |= 1ull<<MAG_AMD64_CAP_AVX;
    #endif
    #ifdef __AVX2__
        caps |= 1ull<<MAG_AMD64_CAP_AVX2;
    #endif
    #ifdef __AVXVNNI__
       caps |= 1ull<<MAG_AMD64_CAP_AVXVNNI;
    #endif
    #ifdef __AVXVNNIINT8__
        caps |= 1ull<<MAG_AMD64_CAP_AVXVNNIINT8;
    #endif
    #ifdef __AVXVNNIINT16__
        caps |= 1ull<<MAG_AMD64_CAP_AVXVNNIINT16;
    #endif
    #ifdef __BMI__
        caps |= 1ull<<MAG_AMD64_CAP_BMI;
    #endif
    #ifdef __BMI2__
        caps |= 1ull<<MAG_AMD64_CAP_BMI2;
    #endif
    #ifdef __F16C__
        caps |= 1ull<<MAG_AMD64_CAP_F16C;
    #endif
    #ifdef __FMA__
        caps |= 1ull<<MAG_AMD64_CAP_FMA;
    #endif
    #ifdef __GFNI__
        caps |= 1ull<<MAG_AMD64_CAP_GFNI;
    #endif
    #ifdef __PCLMUL__
        caps |= 1ull<<MAG_AMD64_CAP_PCLMUL;
    #endif
    #ifdef __RDRND__
        caps |= 1ull<<MAG_AMD64_CAP_RDRND;
    #endif
    #ifdef __RDSEED__
        caps |= 1ull<<MAG_AMD64_CAP_RDSEED;
    #endif
    #ifdef __RDTSCP__
        caps |= 1ull<<MAG_AMD64_CAP_RDTSCP;
    #endif
    #ifdef __SHA__
        caps |= 1ull<<MAG_AMD64_CAP_SHA;
    #endif
    #ifdef __SSE3__
        caps |= 1ull<<MAG_AMD64_CAP_SSE3;
    #endif
    #ifdef __SSE4_1__
        caps |= 1ull<<MAG_AMD64_CAP_SSE4_1;
    #endif
    #ifdef __SSE4_2__
        caps |= 1ull<<MAG_AMD64_CAP_SSE4_2;
    #endif
    #ifdef __SSSE3__
        caps |= 1ull<<MAG_AMD64_CAP_SSSE3;
    #endif
    #ifdef __VAES__
        caps |= 1ull<<MAG_AMD64_CAP_VAES;
    #endif
    #ifdef __VPCLMULQDQ__
        caps |= 1ull<<MAG_AMD64_CAP_VPCLMULQDQ;
    #endif
    #ifdef __XSAVE__
        caps |= 1ull<<MAG_AMD64_CAP_XSAVE;
    #endif
    return caps;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

uint64_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST(void) {
    uint64_t caps = 1u<<MAG_ARM64_CAP_NEON; /* Always required on arm64. */
    #ifdef __ARM_FEATURE_DOTPROD
        caps |= 1u<<MAG_ARM64_CAP_DOTPROD;
    #endif
    #ifdef __ARM_FEATURE_MATMUL_INT8
        caps |= 1u<<MAG_ARM64_CAP_I8MM;
    #endif
    #ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
        caps |= 1u<<MAG_ARM64_CAP_F16SCA;
    #endif
    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        caps |= 1u<<MAG_ARM64_CAP_F16VEC;
    #endif
    #ifdef __ARM_FEATURE_BF16
        caps |= 1u<<MAG_ARM64_CAP_BF16;
    #endif
    #ifdef __ARM_FEATURE_SVE
        caps |= 1u<<MAG_ARM64_CAP_SVE;
    #endif
    #ifdef __ARM_FEATURE_SVE2
        caps |= 1u<<MAG_ARM64_CAP_SVE2;
    #endif
    return caps;
}

#endif

static void (*_Nonnull const mag_blas_lut_init_kernels[MAG_IOP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull) = {
    [MAG_IOP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
        [MAG_DTYPE_BOOL] = &mag_blas_nop,
        [MAG_DTYPE_I32] = &mag_blas_nop,
    },
    [MAG_IOP_BROADCAST] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_broadcast_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_broadcast_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_init_broadcast_bool,
        [MAG_DTYPE_I32] = &mag_blas_init_broadcast_i32,
    },
    [MAG_IOP_MASKED_BROADCAST] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_masked_broadcast_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_masked_broadcast_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_init_masked_broadcast_bool,
        [MAG_DTYPE_I32] = &mag_blas_init_masked_broadcast_i32,
    },
    [MAG_IOP_RAND_UNIFORM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_rand_uniform_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_rand_uniform_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_init_rand_uniform_i32
    },
    [MAG_IOP_RAND_NORMAL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_init_rand_normal_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_init_rand_normal_e5m10,
    },
    [MAG_IOP_RAND_BERNOULLI] = {
        [MAG_DTYPE_BOOL] = &mag_blas_init_rand_bernoulli_bool,
    },
};

static void (*_Nonnull const mag_blas_lut_eval_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
        [MAG_DTYPE_BOOL] = &mag_blas_nop,
        [MAG_DTYPE_I32] = &mag_blas_nop,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_clone_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_clone_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_clone_bool,
        [MAG_DTYPE_I32] = &mag_blas_clone_i32,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
        [MAG_DTYPE_BOOL] = &mag_blas_nop,
        [MAG_DTYPE_I32] = &mag_blas_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
        [MAG_DTYPE_BOOL] = &mag_blas_nop,
        [MAG_DTYPE_I32] = &mag_blas_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_nop,
        [MAG_DTYPE_E5M10] = &mag_blas_nop,
        [MAG_DTYPE_BOOL] = &mag_blas_nop,
        [MAG_DTYPE_I32] = &mag_blas_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mean_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mean_e5m10,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_min_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_min_e5m10,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_max_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_max_e5m10,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sum_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sum_e5m10,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_abs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_abs_e5m10,
    },
    [MAG_OP_SGN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sgn_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sgn_e5m10,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_neg_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_neg_e5m10,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_E8M23] = &mag_blas_log_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_log_e5m10,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqr_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqr_e5m10,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sqrt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sqrt_e5m10,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sin_e5m10,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_E8M23] = &mag_blas_cos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_cos_e5m10,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_step_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_step_e5m10,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_E8M23] = &mag_blas_exp_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_exp_e5m10,
    },
    [MAG_OP_FLOOR] = {
        [MAG_DTYPE_E8M23] = &mag_blas_floor_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_floor_e5m10,
    },
    [MAG_OP_CEIL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_ceil_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_ceil_e5m10,
    },
    [MAG_OP_ROUND] = {
        [MAG_DTYPE_E8M23] = &mag_blas_round_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_round_e5m10,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_e5m10,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_softmax_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_softmax_dv_e5m10,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_e5m10,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sigmoid_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sigmoid_dv_e5m10,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_blas_hard_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_hard_sigmoid_e5m10,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_e5m10,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_silu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_silu_dv_e5m10,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_e5m10,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tanh_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tanh_dv_e5m10,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_e5m10,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_relu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_relu_dv_e5m10,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_e5m10,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gelu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gelu_dv_e5m10,
    },
    [MAG_OP_TRIL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_tril_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_tril_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_tril_bool,
        [MAG_DTYPE_I32] = &mag_blas_tril_i32,
    },
    [MAG_OP_TRIU] = {
        [MAG_DTYPE_E8M23] = &mag_blas_triu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_triu_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_triu_bool,
        [MAG_DTYPE_I32] = &mag_blas_triu_i32,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_E8M23] = &mag_blas_add_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_add_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_add_i32,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_E8M23] = &mag_blas_sub_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_sub_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_sub_i32,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_mul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_mul_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_mul_i32,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_E8M23] = &mag_blas_div_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_div_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_div_i32,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_blas_matmul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_matmul_e5m10,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_blas_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_repeat_back_e5m10,
    },
    [MAG_OP_AND] = {
        [MAG_DTYPE_BOOL] = &mag_blas_and_bool,
        [MAG_DTYPE_I32] = &mag_blas_and_i32,
    },
    [MAG_OP_OR] = {
        [MAG_DTYPE_BOOL] = &mag_blas_or_bool,
        [MAG_DTYPE_I32] = &mag_blas_or_i32,
    },
    [MAG_OP_XOR] = {
        [MAG_DTYPE_BOOL] = &mag_blas_xor_bool,
        [MAG_DTYPE_I32] = &mag_blas_xor_i32,
    },
    [MAG_OP_NOT] = {
        [MAG_DTYPE_BOOL] = &mag_blas_not_bool,
        [MAG_DTYPE_I32] = &mag_blas_not_i32,
    },
    [MAG_OP_SHL] = {
        [MAG_DTYPE_I32] = &mag_blas_shl_i32,
    },
    [MAG_OP_SHR] = {
        [MAG_DTYPE_I32] = &mag_blas_shr_i32,
    },
    [MAG_OP_EQ] = {
        [MAG_DTYPE_E8M23] = &mag_blas_eq_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_eq_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_eq_bool,
        [MAG_DTYPE_I32] = &mag_blas_eq_i32,
    },
    [MAG_OP_NE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_ne_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_ne_e5m10,
        [MAG_DTYPE_BOOL] = &mag_blas_ne_bool,
        [MAG_DTYPE_I32] = &mag_blas_ne_i32,
    },
    [MAG_OP_LE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_le_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_le_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_le_i32,
    },
    [MAG_OP_GE] = {
        [MAG_DTYPE_E8M23] = &mag_blas_ge_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_ge_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_ge_i32,
    },
    [MAG_OP_LT] = {
        [MAG_DTYPE_E8M23] = &mag_blas_lt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_lt_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_lt_i32,
    },
    [MAG_OP_GT] = {
        [MAG_DTYPE_E8M23] = &mag_blas_gt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_blas_gt_e5m10,
        [MAG_DTYPE_I32] = &mag_blas_gt_i32,
    },
};

static void (*_Nonnull const mag_blas_lut_cast_kernels[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(int64_t, void* _Nonnull, const void* _Nonnull) = {
    [MAG_DTYPE_E8M23] = {
        [MAG_DTYPE_E5M10] = &mag_vcast_e8m23_e5m10,
        [MAG_DTYPE_I32] = &mag_vcast_e8m23_i32,
        [MAG_DTYPE_BOOL] = &mag_vcast_e8m23_bool,
    },
    [MAG_DTYPE_E5M10] = {
        [MAG_DTYPE_E8M23] = &mag_vcast_e5m10_e8m23,
        [MAG_DTYPE_I32] = &mag_vcast_e5m10_i32,
        [MAG_DTYPE_BOOL] = &mag_vcast_e5m10_bool,
    },
    [MAG_DTYPE_I32] = {
        [MAG_DTYPE_E8M23] = &mag_vcast_i32_e8m23,
        [MAG_DTYPE_E5M10] = &mag_vcast_i32_e5m10,
        [MAG_DTYPE_BOOL] = &mag_vcast_i32_bool,
    },
    [MAG_DTYPE_BOOL] = {
        [MAG_DTYPE_E8M23] = &mag_vcast_bool_e8m23,
        [MAG_DTYPE_E5M10] = &mag_vcast_bool_e5m10,
        [MAG_DTYPE_I32] = &mag_vcast_bool_i32,
    }
};

static void MAG_HOTPROC mag_blas_vector_cast_stub(size_t nb, const void* _Nonnull src, mag_DType src_t, void* _Nonnull dst, mag_DType dst_t) {
    mag_assert2(dst_t != src_t); /* src and dst types must differ */
    size_t nbs = mag_dtype_meta_of(src_t)->size;
    size_t nbd = mag_dtype_meta_of(dst_t)->size;
    mag_assert2(!((uintptr_t)src&(nbs-1)));     /* src must be aligned */
    mag_assert2(!((uintptr_t)dst&(nbd-1)));     /* dst must be aligned */
    mag_assert2(!(nb&(nbs-1)));                 /* size must be aligned */
    int64_t numel = (int64_t)(nb/nbs);          /* byte -> elems */
    void (*_Nullable kern)(int64_t, void*, const void*) = mag_blas_lut_cast_kernels[src_t][dst_t];
    mag_assert(kern, "invalid cast dtypes %s -> %s", mag_dtype_meta_of(src_t)->name, mag_dtype_meta_of(dst_t)->name);
    (*kern)(numel, dst, src);
}

void MAG_BLAS_SPECIALIZATION(mag_CPUKernelRegistry* _Nonnull kernels) {
    for (int i=0; i < MAG_IOP__NUM; ++i)
        for (int j=0; j < MAG_DTYPE__NUM; ++j)
            kernels->init[i][j] = mag_blas_lut_init_kernels[i][j];
    for (int i=0; i < MAG_OP__NUM; ++i) {
        for (int j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->eval[i][j] = mag_blas_lut_eval_kernels[i][j];
        }
    }
    kernels->vector_cast = &mag_blas_vector_cast_stub;
}
