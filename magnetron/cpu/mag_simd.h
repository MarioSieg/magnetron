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
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return 0.f; }
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return x; }
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return *p; }
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
#define mag_simd_for_vf32(i, n) for (int64_t i=0, lim=(n)&-MAG_VF32_LANES; i < lim; i += MAG_VF32_LANES)

#ifdef __cplusplus
}
#endif

#endif
