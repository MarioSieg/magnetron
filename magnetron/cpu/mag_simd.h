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

#ifndef MAG_SIMD_H
#define MAG_SIMD_H

#include <core/mag_def.h>

#include <core/mag_bfloat16.h>

#include <float.h>
#include <math.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AVX512F__)

#define MAG_SIMD_REG_WIDTH 64

typedef __m512 mag_vf32_t;
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return _mm512_setzero_ps(); }
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return _mm512_set1_ps(x); }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return _mm512_set1_ps(*p); }
/* Broadcast a single lane (0..MAG_VF32_LANES-1) across all lanes. */
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast_lane(mag_vf32_t v, unsigned lane) {
    return _mm512_permutexvar_ps(_mm512_set1_epi32((int)lane), v);
}
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) { return _mm512_load_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) { return _mm512_loadu_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, unsigned n) { return _mm512_maskz_loadu_ps((__mmask16)((1u<<n)-1u), p); }
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) { _mm512_store_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) { _mm512_storeu_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, unsigned n) { _mm512_mask_storeu_ps(p, (__mmask16)((1u<<n)-1u), v); }
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) { return _mm512_add_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) { return _mm512_sub_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) { return _mm512_mul_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) { return _mm512_div_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) { return _mm512_min_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) { return _mm512_max_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return _mm512_fmadd_ps(x, y, z); }
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return _mm512_fnmadd_ps(x, y, z); }
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) { return _mm512_andnot_ps(_mm512_set1_ps(-0.f), x); }
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t v) { return _mm512_reduce_add_ps(v); }

#elif defined(__AVX__)

#define MAG_SIMD_REG_WIDTH 32

typedef __m256 mag_vf32_t;
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return _mm256_setzero_ps(); }
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return _mm256_set1_ps(x); }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return _mm256_broadcast_ss(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast_lane(mag_vf32_t v, unsigned lane) {
#ifdef __AVX2__
    return _mm256_permutevar8x32_ps(v, _mm256_set1_epi32((int)lane));
#else
    switch (lane & 7) {
        case 0: return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0));
        case 1: return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1));
        case 2: return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2));
        case 3: return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3));
        case 4: {
            __m256 t = _mm256_permute2f128_ps(v, v, 0x11);
            return _mm256_shuffle_ps(t, t, _MM_SHUFFLE(0,0,0,0));
        } case 5: {
            __m256 t = _mm256_permute2f128_ps(v, v, 0x11);
            return _mm256_shuffle_ps(t, t, _MM_SHUFFLE(1,1,1,1));
        } case 6: {
            __m256 t = _mm256_permute2f128_ps(v, v, 0x11);
            return _mm256_shuffle_ps(t, t, _MM_SHUFFLE(2,2,2,2));
        } default: {
            __m256 t = _mm256_permute2f128_ps(v, v, 0x11);
            return _mm256_shuffle_ps(t, t, _MM_SHUFFLE(3,3,3,3));
        }
    }
#endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) { return _mm256_load_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) { return _mm256_loadu_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, unsigned n) {
    mag_alignas(32) float tmp[8] = {0};
    memcpy(tmp, p, n*sizeof(*tmp));
    return _mm256_load_ps(tmp);
}
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) { _mm256_store_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) { _mm256_storeu_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, unsigned n) {
    mag_alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    memcpy(p, tmp, n*sizeof(*tmp));
}
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) { return _mm256_add_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) { return _mm256_sub_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) { return _mm256_mul_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) { return _mm256_div_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) { return _mm256_min_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) { return _mm256_max_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
#ifdef __FMA__
    return _mm256_fmadd_ps(x, y, z);
#else
    return _mm256_add_ps(_mm256_mul_ps(x, y), z);
#endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
#ifdef __FMA__
    return _mm256_fnmadd_ps(x, y, z);
#else
    return _mm256_sub_ps(z, _mm256_mul_ps(x, y));
#endif
}
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t v) {
    __m128 y  = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    y = _mm_hadd_ps(y, y);
    y = _mm_hadd_ps(y, y);
    return _mm_cvtss_f32(y);
}
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) {
    return _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000)), x);
}

#elif defined(__SSE2__)

#define MAG_SIMD_REG_WIDTH 16

typedef __m128 mag_vf32_t;
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return _mm_setzero_ps(); }
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return _mm_set1_ps(x); }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return _mm_set1_ps(*p); }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast_lane(mag_vf32_t v, unsigned lane) {
    switch (lane & 3) {
        case 0: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0));
        case 1: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1));
        case 2: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2));
        default: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3));
    }
}
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) { return _mm_load_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) { return _mm_loadu_ps(p); }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, unsigned n) {
    mag_alignas(16) float tmp[4] = {0};
    memcpy(tmp, p, n*sizeof(*tmp));
    return _mm_load_ps(tmp);
}
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) { _mm_store_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) { _mm_storeu_ps(p, v); }
static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, unsigned n) {
    mag_alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    memcpy(p, tmp, n*sizeof(*tmp));
}
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) { return _mm_add_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) { return _mm_sub_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) { return _mm_mul_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) { return _mm_div_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) { return _mm_min_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) { return _mm_max_ps(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
#ifdef __FMA__
    return _mm_fmadd_ps(x, y, z);
#else
    return _mm_add_ps(_mm_mul_ps(x, y), z);
#endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
#ifdef __FMA__
    return _mm_fnmadd_ps(x, y, z);
#else
    return _mm_sub_ps(z, _mm_mul_ps(x, y));
#endif
}
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t v) {
#ifdef __SSE3__
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
#else
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), x); }

#else

typedef float mag_vf32_t;

/* Scalar fallback: treat as 1 lane (4 bytes). */
#define MAG_SIMD_REG_WIDTH 4

static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return 0.f; }
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return x; }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return *p; }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast_lane(mag_vf32_t v, unsigned lane) { (void)lane; return v; }
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) { return *p; }
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) { return *p; }
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) { *p = v; }
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) { *p = v; }
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) { return x + y; }
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) { return x - y; }
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) { return x * y; }
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) { return x / y; }
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) { return fminf(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) { return fmaxf(x, y); }
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return x*y + z; }
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return z - x*y; }
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t v) { return v; }
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) { return fabsf(x); }

#endif

/* Prefetching */

#if defined(__x86_64__) || defined(_M_X64)
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T0); }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T1); }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T2); }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_NTA); }
#elif defined(__GNUC__) || defined(__clang__)
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { __builtin_prefetch(p, 0, 3); }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { __builtin_prefetch(p, 0, 2); }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { __builtin_prefetch(p, 0, 1); }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { __builtin_prefetch(p, 0, 0); }
#else
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { (void)p; }
#endif

#if defined(__GNUC__) || defined(__clang__)
    static MAG_AINLINE void mag_simd_prefetchw(const void *p) { __builtin_prefetch(p, 1, 3); }
#else
    static MAG_AINLINE void mag_simd_prefetchw(const void *p) { (void)p; }
#endif


#define MAG_VF32_LANES (MAG_SIMD_REG_WIDTH/4)
#define MAG_VF32_8_PARTS ((8 + MAG_VF32_LANES - 1)/MAG_VF32_LANES)
#define MAG_VF32_16_PARTS ((16 + MAG_VF32_LANES - 1)/MAG_VF32_LANES)

/* Logical-width SIMD helpers for fp32 microkernels.
   These provide a stable logical interface (8/16 floats) regardless of the
   physical SIMD register width (MAG_VF32_LANES). */

/* --- 8-wide --- */
#if MAG_VF32_LANES == 8 || MAG_VF32_LANES == 16
typedef mag_vf32_t mag_vf32_8_t;

static MAG_AINLINE mag_vf32_8_t mag_vf32_8_zero(void) { return mag_vf32_zero(); }

static MAG_AINLINE mag_vf32_8_t mag_vf32_loadu_8(const float *p) {
#if MAG_VF32_LANES == 16
    /* AVX-512: mask to logical 8. */
    return mag_vf32_loadu_masked(p, 8);
#else
    /* AVX2: physical == logical. */
    return mag_vf32_loadu(p);
#endif
}

static MAG_AINLINE void mag_vf32_storeu_8(float *p, mag_vf32_8_t v) {
#if MAG_VF32_LANES == 16
    mag_vf32_storeu_masked(p, v, 8);
#else
    mag_vf32_storeu(p, v);
#endif
}

static MAG_AINLINE mag_vf32_8_t mag_vf32_8_add(mag_vf32_8_t x, mag_vf32_8_t y) { return mag_vf32_add(x, y); }
static MAG_AINLINE mag_vf32_8_t mag_vf32_8_fmadd(mag_vf32_t a, mag_vf32_8_t b, mag_vf32_8_t c) { return mag_vf32_fmadd(a, b, c); }

#else
typedef struct mag_vf32_8_t {
    mag_vf32_t v[MAG_VF32_8_PARTS];
} mag_vf32_8_t;

static MAG_AINLINE mag_vf32_8_t mag_vf32_8_zero(void) {
    mag_vf32_8_t out;
    mag_vf32_t z = mag_vf32_zero();
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i) out.v[i] = z;
    return out;
}

static MAG_AINLINE mag_vf32_8_t mag_vf32_loadu_8(const float *p) {
    mag_vf32_8_t out;
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i)
        out.v[i] = mag_vf32_loadu(p + (int64_t)i * MAG_VF32_LANES);
    return out;
}

static MAG_AINLINE void mag_vf32_storeu_8(float *p, mag_vf32_8_t v) {
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i)
        mag_vf32_storeu(p + (int64_t)i * MAG_VF32_LANES, v.v[i]);
}

static MAG_AINLINE mag_vf32_8_t mag_vf32_8_add(mag_vf32_8_t x, mag_vf32_8_t y) {
    mag_vf32_8_t out;
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i) out.v[i] = mag_vf32_add(x.v[i], y.v[i]);
    return out;
}

static MAG_AINLINE mag_vf32_8_t mag_vf32_8_fmadd(mag_vf32_t a, mag_vf32_8_t b, mag_vf32_8_t c) {
    mag_vf32_8_t out;
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i) out.v[i] = mag_vf32_fmadd(a, b.v[i], c.v[i]);
    return out;
}
#endif

/* --- 16-wide --- */
#if MAG_VF32_LANES == 16
typedef mag_vf32_t mag_vf32_16_t;

static MAG_AINLINE mag_vf32_16_t mag_vf32_16_zero(void) { return mag_vf32_zero(); }
static MAG_AINLINE mag_vf32_16_t mag_vf32_loadu_16(const float *p) { return mag_vf32_loadu(p); }
static MAG_AINLINE void mag_vf32_storeu_16(float *p, mag_vf32_16_t v) { mag_vf32_storeu(p, v); }
static MAG_AINLINE mag_vf32_16_t mag_vf32_16_add(mag_vf32_16_t x, mag_vf32_16_t y) { return mag_vf32_add(x, y); }
static MAG_AINLINE mag_vf32_16_t mag_vf32_16_fmadd(mag_vf32_t a, mag_vf32_16_t b, mag_vf32_16_t c) { return mag_vf32_fmadd(a, b, c); }

#else
typedef struct mag_vf32_16_t {
    mag_vf32_t v[MAG_VF32_16_PARTS];
} mag_vf32_16_t;

static MAG_AINLINE mag_vf32_16_t mag_vf32_16_zero(void) {
    mag_vf32_16_t out;
    mag_vf32_t z = mag_vf32_zero();
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i) out.v[i] = z;
    return out;
}

static MAG_AINLINE mag_vf32_16_t mag_vf32_loadu_16(const float *p) {
    mag_vf32_16_t out;
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i)
        out.v[i] = mag_vf32_loadu(p + (int64_t)i * MAG_VF32_LANES);
    return out;
}

static MAG_AINLINE void mag_vf32_storeu_16(float *p, mag_vf32_16_t v) {
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i)
        mag_vf32_storeu(p + (int64_t)i * MAG_VF32_LANES, v.v[i]);
}

static MAG_AINLINE mag_vf32_16_t mag_vf32_16_add(mag_vf32_16_t x, mag_vf32_16_t y) {
    mag_vf32_16_t out;
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i) out.v[i] = mag_vf32_add(x.v[i], y.v[i]);
    return out;
}

static MAG_AINLINE mag_vf32_16_t mag_vf32_16_fmadd(mag_vf32_t a, mag_vf32_16_t b, mag_vf32_16_t c) {
    mag_vf32_16_t out;
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i) out.v[i] = mag_vf32_fmadd(a, b.v[i], c.v[i]);
    return out;
}
#endif

/* --- bfloat16 (bf16) SIMD abstraction --- */

/* Physical bf16 vector container.
   We keep the lane count aligned to mag_vf32_t (MAG_VF32_LANES). */
#if defined(__AVX512BF16__)
typedef __m256bh mag_vbf16_t;
#elif defined(__AVX2__) || defined(__SSE2__)
typedef __m128i mag_vbf16_t;
#else
typedef mag_bfloat16_t mag_vbf16_t;
#endif

static MAG_AINLINE mag_vbf16_t mag_vbf16_zero(void) {
#if defined(__AVX512BF16__) || defined(__AVX2__) || defined(__SSE2__)
    /* Set all bf16 lanes to 0. */
#if defined(__AVX512BF16__)
    return (mag_vbf16_t)_mm256_setzero_si256();
#else
    return (mag_vbf16_t)_mm_setzero_si128();
#endif
#else
    return mag_bfloat16c(0);
#endif
}

static MAG_AINLINE mag_vbf16_t mag_vbf16_loadu(const mag_bfloat16_t *p) {
#if defined(__AVX512BF16__)
    return (mag_vbf16_t)_mm256_loadu_si256((const __m256i *)p);
#elif defined(__AVX2__) 
    return (mag_vbf16_t)_mm_loadu_si128((const __m128i *)p);
#elif defined(__SSE2__)
    uint16_t tmp[8] = {0};
    /* Under AVX (no AVX2), MAG_VF32_LANES == 8 but we still only have SSE2 loads.
       So we must load the full 8 bf16 lanes when MAG_VF32_LANES == 8. */
#if MAG_VF32_LANES == 8
    for (int i = 0; i < 8; ++i) tmp[i] = p[i].bits;
#else
    for (int i = 0; i < 4; ++i) tmp[i] = p[i].bits;
#endif
    return (mag_vbf16_t)_mm_loadu_si128((const __m128i *)tmp);
#else
    return *p;
#endif
}

static MAG_AINLINE mag_vbf16_t mag_vbf16_loadu_masked(const mag_bfloat16_t *p, unsigned n) {
    if (n == 0) return mag_vbf16_zero();
#if defined(__AVX512BF16__)
    /* AVX512 BF16 has a native 16-lane mask on the 256-bit bf16 container. */
    const __mmask16 m = (__mmask16)((n >= 16) ? 0xFFFFu : ((1u << n) - 1u));
    return (mag_vbf16_t)_mm256_maskz_loadu_epi16(m, (const void *)p);
#elif defined(__AVX2__)
    if (n >= 8) return mag_vbf16_loadu(p);
    uint16_t tmp[8] = {0};
    for (unsigned i = 0; i < n; ++i) tmp[i] = p[i].bits;
    return (mag_vbf16_t)_mm_loadu_si128((const __m128i *)tmp);
#elif defined(__SSE2__)
    /* MAG_VF32_LANES == 4 (SSE-only) or 8 (AVX without AVX2). */
#if MAG_VF32_LANES == 8
    if (n >= 8) return mag_vbf16_loadu(p);
#else
    if (n >= 4) return mag_vbf16_loadu(p);
#endif
    uint16_t tmp[8] = {0};
    for (unsigned i = 0; i < n; ++i) tmp[i] = p[i].bits;
    return (mag_vbf16_t)_mm_loadu_si128((const __m128i *)tmp);
#else
    /* Scalar fallback supports only n==1 logically. */
    return (n ? *p : mag_bfloat16c(0));
#endif
}

static MAG_AINLINE void mag_vbf16_storeu(mag_bfloat16_t *p, mag_vbf16_t v) {
#if defined(__AVX512BF16__)
    _mm256_storeu_si256((__m256i *)p, (__m256i)v);
#elif defined(__AVX2__)
    _mm_storeu_si128((__m128i *)p, v);
#elif defined(__SSE2__)
    uint16_t tmp[8];
    _mm_storeu_si128((__m128i *)tmp, v);
#if MAG_VF32_LANES == 8
    for (int i = 0; i < 8; ++i) p[i].bits = tmp[i];
#else
    for (int i = 0; i < 4; ++i) p[i].bits = tmp[i];
#endif
#else
    *p = v;
#endif
}

static MAG_AINLINE void mag_vbf16_storeu_masked(mag_bfloat16_t *p, mag_vbf16_t v, unsigned n) {
#if defined(__AVX512BF16__)
    if (n >= 16) return mag_vbf16_storeu(p, v);
    const __mmask16 m = (__mmask16)((1u << n) - 1u);
    _mm256_mask_storeu_epi16((void *)p, m, (__m256i)v);
#elif defined(__AVX2__)
    if (n >= 8) return mag_vbf16_storeu(p, v);
    uint16_t tmp[8];
    _mm_storeu_si128((__m128i *)tmp, v);
    for (unsigned i = 0; i < n; ++i) p[i].bits = tmp[i];
#elif defined(__SSE2__)
    /* MAG_VF32_LANES == 4 (SSE-only) or 8 (AVX without AVX2). */
#if MAG_VF32_LANES == 8
    if (n >= 8) return mag_vbf16_storeu(p, v);
#else
    if (n >= 4) return mag_vbf16_storeu(p, v);
#endif
    uint16_t tmp[8];
    _mm_storeu_si128((__m128i *)tmp, v);
    for (unsigned i = 0; i < n; ++i) p[i].bits = tmp[i];
#else
    if (n) p[0] = v;
#endif
}

static MAG_AINLINE mag_vf32_t mag_vbf16_to_f32(mag_vbf16_t v) {
#if defined(__AVX512BF16__)
    return _mm512_cvtpbh_ps(v);
#elif defined(__AVX512F__)
    /* AVX512F without AVX512BF16:
       Convert bf16[0..7] -> fp32[0..7] with SSE2 bit shifts, insert into __m512. */
    __m128i zero = _mm_setzero_si128();
    __m128i lo = _mm_unpacklo_epi16(v, zero);
    lo = _mm_slli_epi32(lo, 16);
    __m128i hi = _mm_unpackhi_epi16(v, zero);
    hi = _mm_slli_epi32(hi, 16);
    __m512 out = _mm512_setzero_ps();
    out = _mm512_insertf32x4(out, _mm_castsi128_ps(lo), 0);
    out = _mm512_insertf32x4(out, _mm_castsi128_ps(hi), 1);
    return out;
#elif defined(__AVX__)
    /* AVX (includes AVX2): expand 8 bf16 u16 -> 8 float32 bit-patterns via shifts. */
    __m128i zero = _mm_setzero_si128();
    __m128i lo = _mm_unpacklo_epi16(v, zero);
    lo = _mm_slli_epi32(lo, 16);
    __m128i hi = _mm_unpackhi_epi16(v, zero);
    hi = _mm_slli_epi32(hi, 16);
    __m128 lo_ps = _mm_castsi128_ps(lo);
    __m128 hi_ps = _mm_castsi128_ps(hi);
    __m256 out = _mm256_castps128_ps256(lo_ps);
    out = _mm256_insertf128_ps(out, hi_ps, 1);
    return out;
#elif defined(__SSE2__)
    /* SSE-only: expand 4 bf16 u16 -> 4 float32 bit-patterns via shift. */
    __m128i zero = _mm_setzero_si128();
    __m128i lo = _mm_unpacklo_epi16(v, zero);
    lo = _mm_slli_epi32(lo, 16);
    return _mm_castsi128_ps(lo);
#else
    return mag_vf32_splat(mag_bfloat16_to_float32_soft_fp(v));
#endif
}

static MAG_AINLINE mag_vbf16_t mag_vf32_to_bf16(mag_vf32_t v) {
#if defined(__AVX512BF16__)
    return (mag_vbf16_t)_mm512_cvtneps_pbh(v);
#elif defined(__AVX512F__)
    /* AVX512F without AVX512BF16:
       Convert low 8 floats to bf16 bits via scalar lane shifts. */
    __m128 lo_ps = _mm512_castps512_ps128(v);      /* lanes 0..3 */
    __m128 hi_ps = _mm512_extractf32x4_ps(v, 1);  /* lanes 4..7 */
    __m128i lo_ci = _mm_castps_si128(lo_ps);
    __m128i hi_ci = _mm_castps_si128(hi_ps);
    __m128i lo_sh = _mm_srli_epi32(lo_ci, 16);
    __m128i hi_sh = _mm_srli_epi32(hi_ci, 16);
    uint32_t u32[4];
    uint16_t out16[8] = {0};
    _mm_storeu_si128((__m128i *)u32, lo_sh);
    for (int i = 0; i < 4; ++i) out16[i] = (uint16_t)u32[i];
    _mm_storeu_si128((__m128i *)u32, hi_sh);
    for (int i = 0; i < 4; ++i) out16[i + 4] = (uint16_t)u32[i];
    return (mag_vbf16_t)_mm_loadu_si128((const __m128i *)out16);
#elif defined(__AVX__)
    /* AVX (includes AVX2): convert low 8 floats to 8 bf16 lanes without _mm_packus_epi32. */
    __m128 lo_ps = _mm256_castps256_ps128(v);
    __m128 hi_ps = _mm256_extractf128_ps(v, 1);
    __m128i lo_ci = _mm_castps_si128(lo_ps);
    __m128i hi_ci = _mm_castps_si128(hi_ps);
    __m128i lo_sh = _mm_srli_epi32(lo_ci, 16);
    __m128i hi_sh = _mm_srli_epi32(hi_ci, 16);

    mag_alignas(16) uint32_t u32[4];
    mag_alignas(16) uint16_t out16[8] = {0};

    _mm_store_si128((__m128i *)u32, lo_sh);
    for (int i = 0; i < 4; ++i) out16[i] = (uint16_t)u32[i];

    _mm_store_si128((__m128i *)u32, hi_sh);
    for (int i = 0; i < 4; ++i) out16[i + 4] = (uint16_t)u32[i];

    return (mag_vbf16_t)_mm_load_si128((const __m128i *)out16);
#elif defined(__SSE2__)
    /* SSE-only: convert 4 floats to 4 bf16 lanes (upper 4 lanes are unused). */
    __m128i ci = _mm_castps_si128(v);
    __m128i sh = _mm_srli_epi32(ci, 16);

    mag_alignas(16) uint32_t u32[4];
    mag_alignas(16) uint16_t out16[8] = {0};

    _mm_store_si128((__m128i *)u32, sh);
    for (int i = 0; i < 4; ++i) out16[i] = (uint16_t)u32[i];

    return (mag_vbf16_t)_mm_load_si128((const __m128i *)out16);
#else
    return mag_bfloat16_from_float32_soft_fp(v);
#endif
}

static MAG_AINLINE mag_vf32_t mag_vbf16_broadcast(const mag_bfloat16_t *p) {
#if defined(__AVX512BF16__)
    __m256bh Abh = (__m256bh)_mm256_set1_epi16((short)p->bits);
    return _mm512_cvtpbh_ps(Abh);
#elif defined(__AVX512F__)
    /* No AVX512BF16 conversion: broadcast via scalar bf16->f32. */
    return _mm512_set1_ps(mag_bfloat16_to_float32_soft_fp(*p));
#elif defined(__AVX__)
    /* AVX (includes AVX2): broadcast 8 bf16 lanes as 8 float bit-patterns. */
    __m128i Abits = _mm_set1_epi16((short)p->bits);
    __m128i zero = _mm_setzero_si128();
    __m128i lo = _mm_unpacklo_epi16(Abits, zero);
    lo = _mm_slli_epi32(lo, 16);
    __m128i hi = _mm_unpackhi_epi16(Abits, zero);
    hi = _mm_slli_epi32(hi, 16);

    __m128 lo_ps = _mm_castsi128_ps(lo);
    __m128 hi_ps = _mm_castsi128_ps(hi);
    __m256 out = _mm256_castps128_ps256(lo_ps);
    out = _mm256_insertf128_ps(out, hi_ps, 1);
    return out;
#elif defined(__SSE2__)
    /* SSE-only: broadcast 4 bf16 lanes as 4 float bit-patterns. */
    __m128i Abits = _mm_set1_epi16((short)p->bits);
    __m128i zero = _mm_setzero_si128();
    __m128i lo = _mm_unpacklo_epi16(Abits, zero);
    lo = _mm_slli_epi32(lo, 16);
    return _mm_castsi128_ps(lo);
#else
    return mag_vf32_splat(mag_bfloat16_to_float32_soft_fp(*p));
#endif
}

static MAG_AINLINE mag_vbf16_t mag_vbf16_splat(mag_bfloat16_t x) {
#if defined(__AVX512BF16__)
    return (mag_vbf16_t)_mm256_set1_epi16((short)x.bits);
#elif defined(__AVX2__) || defined(__SSE2__)
    return (mag_vbf16_t)_mm_set1_epi16((short)x.bits);
#else
    return x;
#endif
}

/* Logical-width helpers for bf16->fp32 conversion (8/16 lanes). */

static MAG_AINLINE mag_vf32_8_t mag_vbf16_loadu_8_to_f32(const mag_bfloat16_t *p) {
#if MAG_VF32_LANES == 8
    return mag_vbf16_to_f32(mag_vbf16_loadu(p));
#elif MAG_VF32_LANES == 16
    return mag_vbf16_to_f32(mag_vbf16_loadu_masked(p, 8));
#else
    mag_vf32_8_t out;
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i)
        out.v[i] = mag_vbf16_to_f32(mag_vbf16_loadu(p + (int64_t)i * MAG_VF32_LANES));
    return out;
#endif
}

static MAG_AINLINE void mag_vbf16_storeu_8_from_f32(mag_bfloat16_t *p, mag_vf32_8_t v) {
#if MAG_VF32_LANES == 8
    mag_vbf16_storeu(p, mag_vf32_to_bf16(v));
#elif MAG_VF32_LANES == 16
    mag_vbf16_storeu_masked(p, mag_vf32_to_bf16(v), 8);
#else
    for (int i = 0; i < MAG_VF32_8_PARTS; ++i)
        mag_vbf16_storeu(p + (int64_t)i * MAG_VF32_LANES, mag_vf32_to_bf16(v.v[i]));
#endif
}

static MAG_AINLINE mag_vf32_16_t mag_vbf16_loadu_16_to_f32(const mag_bfloat16_t *p) {
#if MAG_VF32_LANES == 16
    return mag_vbf16_to_f32(mag_vbf16_loadu(p));
#elif MAG_VF32_LANES == 8
    mag_vf32_16_t out;
    out.v[0] = mag_vbf16_to_f32(mag_vbf16_loadu(p + 0));
    out.v[1] = mag_vbf16_to_f32(mag_vbf16_loadu(p + MAG_VF32_LANES));
    return out;
#else
    mag_vf32_16_t out;
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i)
        out.v[i] = mag_vbf16_to_f32(mag_vbf16_loadu(p + (int64_t)i * MAG_VF32_LANES));
    return out;
#endif
}

static MAG_AINLINE void mag_vbf16_storeu_16_from_f32(mag_bfloat16_t *p, mag_vf32_16_t v) {
#if MAG_VF32_LANES == 16
    mag_vbf16_storeu(p, mag_vf32_to_bf16(v));
#elif MAG_VF32_LANES == 8
    mag_vbf16_storeu(p + 0, mag_vf32_to_bf16(v.v[0]));
    mag_vbf16_storeu(p + MAG_VF32_LANES, mag_vf32_to_bf16(v.v[1]));
#else
    for (int i = 0; i < MAG_VF32_16_PARTS; ++i)
        mag_vbf16_storeu(p + (int64_t)i * MAG_VF32_LANES, mag_vf32_to_bf16(v.v[i]));
#endif
}

#define mag_simd_for_vf32(i, n) for (int64_t i=0, lim=(n)&-MAG_VF32_LANES; i < lim; i += MAG_VF32_LANES)

#ifdef __cplusplus
}
#endif

#endif
