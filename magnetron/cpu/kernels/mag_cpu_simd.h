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
#include <string.h>
#include <math.h>

/*
** Platform guards template (:
**
**#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
**  #error "TODO"
**#elif defined(__AVX512F__)
**  #error "TODO"
**#elif defined(__AVX2__)
**  #error "TODO"
**#elif defined(__SSE4_1__)
**  #error "TODO"
**#elif defined(__SSE2__)
**  #error "TODO"
**#elif defined(__loongarch_asx)
**  #error "TODO"
**#elif defined(__loongarch_sx)
**  #error "TODO"
**#else
**  #error "TODO"
**#endif
*/

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
  #elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    #include <arm_neon.h>
    #include <arm_acle.h>
  #elif defined(__loongarch_asx)
    #include <lasxintrin.h>
    #include <lsxintrin.h>
  #elif defined(__loongarch_sx)
    #include <lsxintrin.h>
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64) /* 128-bit ARM64 Neon */
  typedef uint32x4_t mag_vmask32_t;
  typedef float32x4_t mag_vf32_t;
  typedef int32x4_t mag_vi32_t;
  typedef uint16x8_t mag_vbf16_t;
#elif defined(__AVX512F__)       /* 512-bit AVX*/
  typedef __mmask16 mag_vmask32_t;
  typedef __m512 mag_vf32_t;
  typedef __m512i mag_vi32_t;
  #ifdef __AVX512BF16__
    typedef __m512bh mag_vbf16_t;
  #else
    typedef __m512i mag_vbf16_t;
  #endif
#elif defined(__AVX2__) /* TODO (not so important tbh): Make AVX1 version, emulate AVX2 int instructions using dual SSE2 pumping */
  typedef __m256i mag_vmask32_t;
  typedef __m256 mag_vf32_t;
  typedef __m256i mag_vi32_t;
  typedef __m256i mag_vbf16_t;
#elif defined(__SSE2__)         /* 128-bit SSE* */
  typedef __m128i mag_vmask32_t;
  typedef __m128 mag_vf32_t;
  typedef __m128i mag_vi32_t;
  typedef __m128i mag_vbf16_t;
#elif defined(__loongarch_asx)   /* 256-bit LASX on Loongson / Godson */
  typedef __m256i mag_vmask32_t;
  typedef __m256 mag_vf32_t;
  typedef __m256i mag_vi32_t;
#elif defined(__loongarch_sx)   /* 128-bit LSX on Loongson / Godson */
  typedef __m128i mag_vmask32_t;
  typedef __m128 mag_vf32_t;
  typedef __m128i mag_vi32_t;
#else /* Scalar fallback path */
  typedef uint32_t mag_vmask32_t;
  typedef float mag_vf32_t;
  typedef int32_t mag_vi32_t;
  typedef mag_bfloat16_t mag_vbf16_t;
#endif

#define MAG_VF32_LANES ((int64_t)(sizeof(mag_vf32_t)/sizeof(float)))
#define MAG_VBF16_LANES (MAG_VF32_LANES<<1)

#if defined(__AVX512F__) && defined(__AVX512BF16__)
  #define MAG_HAS_NATIVE_DPBF16 1
#else
  #define MAG_HAS_NATIVE_DPBF16 0
#endif

/* mask type */
static MAG_AINLINE mag_vmask32_t mag_vmask32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_u32(0);
  #elif defined(__AVX512F__)
    return 0;
  #elif defined(__AVX2__)
    return _mm256_setzero_si256();
  #elif defined(__SSE2__)
    return _mm_setzero_si128();
  #elif defined(__loongarch_asx)
    return __lasx_xvreplgr2vr_w(0);
  #elif defined(__loongarch_sx)
    return __lsx_vreplgr2vr_w(0);
  #else
    return 0;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_full(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_u32(~0u);
  #elif defined(__AVX512F__)
    return 0xffff;
  #elif defined(__AVX2__)
    return _mm256_set1_epi32(-1);
  #elif defined(__SSE2__)
    return _mm_set1_epi32(-1);
  #elif defined(__loongarch_asx)
    return __lasx_xvreplgr2vr_w(-1);
  #elif defined(__loongarch_sx)
    return __lsx_vreplgr2vr_w(-1);
  #else
    return ~0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_not(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(x);
  #elif defined(__AVX512F__)
    return ~x&0xffff;
  #elif defined(__AVX2__)
    return _mm256_xor_si256(x, _mm256_set1_epi32(-1));
  #elif defined(__SSE2__)
    return _mm_xor_si128(x, _mm_set1_epi32(-1));
  #elif defined(__loongarch_asx)
    return __lasx_xvxor_v(x, __lasx_xvreplgr2vr_w(-1));
  #elif defined(__loongarch_sx)
    return __lsx_vxor_v(x, __lsx_vreplgr2vr_w(-1));
  #else
    return ~x;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_and(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vandq_u32(x, y);
  #elif defined(__AVX512F__)
    return x&y;
  #elif defined(__AVX2__)
    return _mm256_and_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_and_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvand_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vand_v(x, y);
  #else
    return x&y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_or(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vorrq_u32(x, y);
  #elif defined(__AVX512F__)
    return x|y;
  #elif defined(__AVX2__)
    return _mm256_or_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_or_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvor_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vor_v(x, y);
  #else
    return x|y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_xor(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return veorq_u32(x, y);
  #elif defined(__AVX512F__)
    return x^y;
  #elif defined(__AVX2__)
    return _mm256_xor_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_xor_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvxor_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vxor_v(x, y);
  #else
    return x^y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_andnot(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbicq_u32(y, x);
  #elif defined(__AVX512F__)
    return ~x&y;
  #elif defined(__AVX2__)
    return _mm256_andnot_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_andnot_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvandn_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vandn_v(x, y);
  #else
    return ~x&y;
  #endif
}
static MAG_AINLINE int mag_vmask32_any(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxvq_u32(x) != 0;
  #elif defined(__AVX512F__)
    return x != 0;
  #elif defined(__AVX2__)
    return _mm256_movemask_ps(_mm256_castsi256_ps(x)) != 0;
  #elif defined(__SSE2__)
    return _mm_movemask_ps(_mm_castsi128_ps(x)) != 0;
  #elif defined(__loongarch_asx)
    mag_alignas(32) uint32_t t[8];
    __lasx_xvst(x, t, 0);
    return (t[0]|t[1]|t[2]|t[3]|t[4]|t[5]|t[6]|t[7]) != 0;
  #elif defined(__loongarch_sx)
    mag_alignas(16) uint32_t t[4];
    __lsx_vst(x, t, 0);
    return (t[0]|t[1]|t[2]|t[3]) != 0;
  #else
    return x != 0;
  #endif
}
static MAG_AINLINE int mag_vmask32_all(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminvq_u32(x) == ~0u;
  #elif defined(__AVX512F__)
    return x == 0xffff;
  #elif defined(__AVX2__)
    return _mm256_movemask_ps(_mm256_castsi256_ps(x)) == 0xff;
  #elif defined(__SSE2__)
    return _mm_movemask_ps(_mm_castsi128_ps(x)) == 0xf;
  #elif defined(__loongarch_asx)
    mag_alignas(32) uint32_t t[8];
    __lasx_xvst(x, t, 0);
    return (t[0]&t[1]&t[2]&t[3]&t[4]&t[5]&t[6]&t[7]) != 0;
  #elif defined(__loongarch_sx)
    mag_alignas(16) uint32_t t[4];
    __lsx_vst(x, t, 0);
    return (t[0]&t[1]&t[2]&t[3]) == ~0u;
  #else
    return x == ~0u;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(0.f);
  #elif defined(__AVX512F__)
    return _mm512_setzero_ps();
  #elif defined(__AVX2__)
      return _mm256_setzero_ps();
  #elif defined(__SSE2__)
    return _mm_setzero_ps();
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvreplgr2vr_w(0);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vreplgr2vr_w(0);
  #else
    return 0.f;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_set1_ps(x);
  #elif defined(__AVX2__)
    return _mm256_set1_ps(x);
  #elif defined(__SSE2__)
    return _mm_set1_ps(x);
  #elif defined(__loongarch_asx)
    uint32_t u32;
    memcpy(&u32, &x, sizeof u32);
    return (__m256)__lasx_xvreplgr2vr_w(u32);
  #elif defined(__loongarch_sx)
    uint32_t u32;
    memcpy(&u32, &x, sizeof u32);
    return (__m128)__lsx_vreplgr2vr_w(u32);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(*p);
  #elif defined(__AVX512F__)
    return _mm512_set1_ps(*p);
  #elif defined(__AVX2__)
    return _mm256_set1_ps(*p);
  #elif defined(__SSE2__)
    return _mm_set1_ps(*p);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvldrepl_w(p, 0);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vldrepl_w(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_f32(p);
  #elif defined(__AVX512F__)
    return _mm512_load_ps(p);
  #elif defined(__AVX2__)
    return _mm256_load_ps(p);
  #elif defined(__SSE2__)
    return _mm_load_ps(p);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvld(p, 0);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vld(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_f32(p);
  #elif defined(__AVX512F__)
    return _mm512_loadu_ps(p);
  #elif defined(__AVX2__)
    return _mm256_loadu_ps(p);
  #elif defined(__SSE2__)
    return _mm_loadu_ps(p);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvld(p, 0);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vld(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) float tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_f32(tmp);
  #elif defined(__AVX512F__)
    return _mm512_maskz_loadu_ps((__mmask16)((1u<<n)-1u), p);
  #elif defined(__AVX2__)
    mag_alignas(32) float tmp[8] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return _mm256_load_ps(tmp);
  #elif defined(__SSE2__)
    mag_alignas(16) float tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return _mm_load_ps(tmp);
  #elif defined(__loongarch_asx)
    mag_alignas(32) float tmp[8] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return (__m256)__lasx_xvld(tmp, 0);
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return (__m128)__lsx_vld(tmp, 0);
  #else
    return n ? *p : 0;
  #endif
}
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_f32(p, v);
  #elif defined(__AVX512F__)
    _mm512_store_ps(p, v);
  #elif defined(__AVX2__)
    _mm256_store_ps(p, v);
  #elif defined(__SSE2__)
    _mm_store_ps(p, v);
  #elif defined(__loongarch_asx)
    __lasx_xvst((__m256i)v, p, 0);
  #elif defined(__loongarch_sx)
    __lsx_vst((__m128i)v, p, 0);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_f32(p, v);
  #elif defined(__AVX512F__)
    _mm512_storeu_ps(p, v);
  #elif defined(__AVX2__)
    _mm256_storeu_ps(p, v);
  #elif defined(__SSE2__)
    _mm_storeu_ps(p, v);
  #elif defined(__loongarch_asx)
    __lasx_xvst((__m256i)v, p, 0);
  #elif defined(__loongarch_sx)
    __lsx_vst((__m128i)v, p, 0);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) float tmp[4];
    vst1q_f32(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__AVX512F__)
    _mm512_mask_storeu_ps(p, (__mmask16)((1u<<n)-1u), v);
  #elif defined(__AVX2__)
    mag_alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__SSE2__)
    mag_alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__loongarch_asx)
    mag_alignas(32) float tmp[8];
    __lasx_xvst((__m256i)v, tmp, 0);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4];
    __lsx_vst((__m128i)v, tmp, 0);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #else
    if (n) *p = v;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpeq(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vceqq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_EQ_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmpeq_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_ceq_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_ceq_s(x, y);
  #else
    return x==y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpne(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(vceqq_f32(x, y));
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_NEQ_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmpneq_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_cne_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_cne_s(x, y);
  #else
    return x!=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmplt(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcltq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_LT_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmplt_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_clt_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_clt_s(x, y);
  #else
    return x<y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmple(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcleq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_LE_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmple_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_cle_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_cle_s(x, y);
  #else
    return x<=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpgt(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgtq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_GT_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmpgt_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_clt_s(y, x);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_clt_s(y, x);
  #else
    return x>y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpge(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgeq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_GE_OQ));
  #elif defined(__SSE2__)
    return _mm_castps_si128(_mm_cmpge_ps(x, y));
  #elif defined(__loongarch_asx)
    return __lasx_xvfcmp_cle_s(y, x);
  #elif defined(__loongarch_sx)
    return __lsx_vfcmp_cle_s(y, x);
  #else
    return x>=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_blend(mag_vmask32_t m, mag_vf32_t t, mag_vf32_t f) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbslq_f32(m, t, f);
  #elif defined(__AVX512F__)
    return _mm512_mask_blend_ps(m, f, t);
  #elif defined(__AVX2__)
    return _mm256_blendv_ps(f, t, _mm256_castsi256_ps(m));
  #elif defined(__SSE4_1__)
    return _mm_blendv_ps(f, t, _mm_castsi128_ps(m));
  #elif defined(__SSE2__)
    __m128 mask = _mm_castsi128_ps(m);
    return _mm_or_ps(_mm_and_ps(mask, t), _mm_andnot_ps(mask, f));
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvbitsel_v((__m256i)f, (__m256i)t, m);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vbitsel_v((__m128i)f, (__m128i)t, m);
  #else
    uint32_t tb, fb;
    memcpy(&tb, &t, sizeof(tb));
    memcpy(&fb, &f, sizeof(fb));
    uint32_t rb = ((m&tb)|(~m&fb));
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_add_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_add_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_add_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfadd_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfadd_s(x, y);
  #else
    return x+y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vsubq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_sub_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_sub_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_sub_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfsub_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfsub_s(x, y);
  #else
    return x-y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_mul_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_mul_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_mul_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfmul_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfmul_s(x, y);
  #else
    return x*y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdivq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_div_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_div_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_div_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfdiv_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfdiv_s(x, y);
  #else
    return x/y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_min_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_min_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_min_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfmin_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfmin_s(x, y);
  #else
    return x<y ? x : y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_max_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_max_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_max_ps(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvfmax_s(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vfmax_s(x, y);
  #else
    return x>y ? x : y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vfmaq_f32(z, x, y);
  #elif defined(__AVX512F__)
    return _mm512_fmadd_ps(x, y, z);
  #elif defined(__AVX2__)
    #ifdef __FMA__
      return _mm256_fmadd_ps(x, y, z);
    #else
      return _mm256_add_ps(_mm256_mul_ps(x, y), z);
    #endif
  #elif defined(__SSE2__)
    #ifdef __FMA__
      return _mm_fmadd_ps(x, y, z);
    #else
      return _mm_add_ps(_mm_mul_ps(x, y), z);
    #endif
  #elif defined(__loongarch_asx)
    return __lasx_xvfmadd_s(x, y, z);
  #elif defined(__loongarch_sx)
    return __lsx_vfmadd_s(x, y, z);
  #else
    return x*y + z;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vfmsq_f32(z, x, y);
  #elif defined(__AVX512F__)
    return _mm512_fnmadd_ps(x, y, z);
  #elif defined(__AVX2__)
    #ifdef __FMA__
      return _mm256_fnmadd_ps(x, y, z);
    #else
      return _mm256_sub_ps(z, _mm256_mul_ps(x, y));
    #endif
  #elif defined(__SSE2__)
    #ifdef __FMA__
      return _mm_fnmadd_ps(x, y, z);
    #else
      return _mm_sub_ps(z, _mm_mul_ps(x, y));
    #endif
  #elif defined(__loongarch_asx)
    return __lasx_xvfnmsub_s(x, y, z);
  #elif defined(__loongarch_sx)
    return __lsx_vfnmsub_s(x, y, z);
  #else
    return z - x*y;
  #endif
}
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddvq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_reduce_add_ps(x);
  #elif defined(__AVX2__)
    __m128 acc = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    acc = _mm_hadd_ps(acc, acc);
    acc = _mm_hadd_ps(acc, acc);
    return _mm_cvtss_f32(acc);
  #elif defined(__SSE3__)
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
  #elif defined(__SSE2__)
    __m128 shuf = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, x);
    x = _mm_add_ss(x, shuf);
    return _mm_cvtss_f32(x);
  #elif defined(__loongarch_asx)
    mag_alignas(32) float t[8];
    __lasx_xvst((__m256i)x, t, 0);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
  #elif defined(__loongarch_sx)
    mag_alignas(16) float t[4];
    __lsx_vst((__m128i)x, t, 0);
    return t[0]+t[1]+t[2]+t[3];
  #else
    return x;
  #endif
}
static MAG_AINLINE float mag_vf32_reduce_max(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxvq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_reduce_max_ps(x);
  #elif defined(__AVX2__)
    __m128 acc = _mm_max_ps(
      _mm256_castps256_ps128(x),
      _mm256_extractf128_ps(x, 1)
    );
    __m128 shuf = _mm_movehdup_ps(acc);
    acc = _mm_max_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, acc);
    acc = _mm_max_ss(acc, shuf);
    return _mm_cvtss_f32(acc);
  #elif defined(__SSE2__)
    __m128 shuf = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_max_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, x);
    x = _mm_max_ss(x, shuf);
    return _mm_cvtss_f32(x);
  #elif defined(__loongarch_asx)
    mag_alignas(32) float t[8];
    __lasx_xvst((__m256i)x, t, 0);
    float a = mag_xmax(mag_xmax(t[0], t[1]), mag_xmax(t[2], t[3]));
    float b = mag_xmax(mag_xmax(t[4], t[5]), mag_xmax(t[6], t[7]));
    return mag_xmax(a, b);
  #elif defined(__loongarch_sx)
    mag_alignas(16) float t[4];
    __lsx_vst((__m128i)x, t, 0);
    float a = mag_xmax(t[0], t[1]);
    float b = mag_xmax(t[2], t[3]);
    return mag_xmax(a, b);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vabsq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
  #elif defined(__AVX2__)
    return _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
  #elif defined(__SSE2__)
    return _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));
  #else
    return fabsf(x);
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_rcp_approx(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vrecpeq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_rcp14_ps(x);
  #elif defined(__AVX2__)
    return _mm256_rcp_ps(x);
  #elif defined(__SSE2__)
    return _mm_rcp_ps(x);
  #elif defined(__loongarch_asx)
    return __lasx_xvfrecip_s(x);
  #elif defined(__loongarch_sx)
    return __lsx_vfrecip_s(x);
  #else
    return 1.f/x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_rcp_refine_step(mag_vf32_t x, mag_vf32_t r) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_f32(vrecpsq_f32(x, r), r);
  #else
    mag_vf32_t refined = mag_vf32_mul(r, mag_vf32_fnmadd(x, r, mag_vf32_splat(2.f)));
    return mag_vf32_blend(mag_vf32_cmpeq(refined, refined), refined, r);
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_and(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_and_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_and_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_and_ps(x, y);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvand_v((__m256i)x, (__m256i)y);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vand_v((__m128i)x, (__m128i)y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb&yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_sqrt(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vsqrtq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_sqrt_ps(x);
  #elif defined(__AVX2__)
    return _mm256_sqrt_ps(x);
  #elif defined(__SSE2__)
    return _mm_sqrt_ps(x);
  #elif defined(__loongarch_asx)
    return __lasx_xvfsqrt_s(x);
  #elif defined(__loongarch_sx)
    return __lsx_vfsqrt_s(x);
  #else
    return sqrtf(x);
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_or(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_or_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_or_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_or_ps(x, y);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvor_v((__m256i)x, (__m256i)y);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vor_v((__m128i)x, (__m128i)y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb|yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_xor(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_xor_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_xor_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_xor_ps(x, y);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvxor_v((__m256i)x, (__m256i)y);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vxor_v((__m128i)x, (__m128i)y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb^yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_not(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(x)));
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), _mm512_set1_epi32(-1)));
  #elif defined(__AVX2__)
    return _mm256_xor_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
  #elif defined(__SSE2__)
    return _mm_xor_ps(x, _mm_castsi128_ps(_mm_set1_epi32(-1)));
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvxor_v((__m256i)x, __lasx_xvreplgr2vr_w(-1));
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vxor_v((__m128i)x, __lsx_vreplgr2vr_w(-1));
  #else
    uint32_t xb;
    memcpy(&xb, &x, sizeof(xb));
    xb = ~xb;
    mag_vf32_t r;
    memcpy(&r, &xb, sizeof(xb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_andnot(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));
  #elif defined(__AVX512F__)
    return _mm512_andnot_ps(x, y);
  #elif defined(__AVX2__)
    return _mm256_andnot_ps(x, y);
  #elif defined(__SSE2__)
    return _mm_andnot_ps(x, y);
  #elif defined(__loongarch_asx)
    return (__m256)__lasx_xvandn_v((__m256i)x, (__m256i)y);
  #elif defined(__loongarch_sx)
    return (__m128)__lsx_vandn_v((__m128i)x, (__m128i)y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = ~xb&yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_f16(const mag_float16_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvt_f32_f16(vld1_f16((const __fp16 *)p));
  #elif defined(__AVX512F__) && defined(__F16C__)
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
  #elif defined(__AVX512F__)
    mag_alignas(64) float tmp[16];
    for (int i=0; i < 16; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return _mm512_load_ps(tmp);
  #elif defined(__AVX2__) && defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
  #elif defined(__AVX2__)
    mag_alignas(32) float tmp[8];
    for (int i=0; i < 8; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return _mm256_load_ps(tmp);
  #elif defined(__SSE2__)
    mag_alignas(16) float tmp[4];
    for (int i=0; i < 4; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return _mm_load_ps(tmp);
  #elif defined(__loongarch_asx)
    mag_alignas(32) float tmp[8];
    for (int i=0; i < 8; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return (__m256)__lasx_xvld(tmp, 0);
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4];
    for (int i=0; i < 4; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return (__m128)__lsx_vld(tmp, 0);
  #else
    return mag_float16_to_float32(*p);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_f16(mag_float16_t *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1_f16((__fp16 *)p, vcvt_f16_f32(v));
  #elif defined(__AVX512F__) && defined(__F16C__)
    _mm256_storeu_si256((__m256i *)p, _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
  #elif defined(__AVX512F__)
    mag_alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    for (int i=0; i < 16; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #elif defined(__AVX2__) && defined(__F16C__)
    _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
  #elif defined(__AVX2__)
    mag_alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (int i=0; i < 8; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #elif defined(__SSE2__)
    mag_alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    for (int i=0; i < 4; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #elif defined(__loongarch_asx)
    mag_alignas(32) float tmp[8];
    __lasx_xvst((__m256i)v, tmp, 0);
    for (int i=0; i < 8; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4];
    __lsx_vst((__m128i)v, tmp, 0);
    for (int i=0; i < 4; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #else
    *p = mag_float32_to_float16(v);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_masked_f16(mag_float16_t *p, mag_vf32_t v, int n) {
  mag_float16_t tmp[MAG_VF32_LANES];
  mag_vf32_storeu_f16(tmp, v);
  for (int i=0; i < n; ++i) p[i] = tmp[i];
}

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_bf16(const mag_bfloat16_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint16x4_t h = vld1_u16((const uint16_t *)p);
    uint32x4_t u = vshll_n_u16(h, 16);
    return vreinterpretq_f32_u32(u);
  #elif defined(__AVX512F__)
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    __m512i u = _mm512_cvtepu16_epi32(h);
    u = _mm512_slli_epi32(u, 16);
    return _mm512_castsi512_ps(u);
  #elif defined(__AVX2__)
    __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    u = _mm256_slli_epi32(u, 16);
    return _mm256_castsi256_ps(u);
  #elif defined(__SSE2__)
    __m128i h = _mm_loadl_epi64((const __m128i *)p);
    __m128i z = _mm_setzero_si128();
    __m128i u = _mm_unpacklo_epi16(h, z);
    u = _mm_slli_epi32(u, 16);
    return _mm_castsi128_ps(u);
  #elif defined(__loongarch_asx)
    __m256i h = __lasx_xvldi(0);
    __asm__ __volatile__(
      "xvinsgr2vr.d %u[h], %[q0], 0\n"
      "xvinsgr2vr.d %u[h], %[q1], 2\n"
      : [h]"+f"(h)
      : [q0]"r"(*((uint64_t *)p+0)),
        [q1]"r"(*((uint64_t *)p+1))
      :
    );
  __m256i u = __lasx_xvilvl_h(__lasx_xvldi(0), h);
  return (__m256)__lasx_xvslli_w(u, 16);
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4];
    for (int i=0; i < 4; ++i)
      tmp[i] = mag_bfloat16_to_float32(p[i]);
    return (__m128)__lsx_vld(tmp, 0);
  #else
    return mag_bfloat16_to_float32(*p);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_bf16(mag_bfloat16_t *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint16x4_t h = vmovn_u32(vshrq_n_u32(u, 16));
    vst1_u16((uint16_t *)p, h);
  #elif defined(__AVX512F__) && defined(__AVX512BF16__)
    __m256bh h = _mm512_cvtneps_pbh(v);
    _mm256_storeu_si256((__m256i *)p, (__m256i)h);
  #elif defined(__AVX512F__)
    __m512i u = _mm512_castps_si512(v);
    u = _mm512_srli_epi32(u, 16);
    __m256i h = _mm512_cvtepi32_epi16(u);
    _mm256_storeu_si256((__m256i *)p, h);
  #elif defined(__AVX2__)
    __m256i u = _mm256_castps_si256(v);
    u = _mm256_srli_epi32(u, 16);
    __m128i h = _mm_packus_epi32(_mm256_castsi256_si128(u), _mm256_extracti128_si256(u, 1));
    _mm_storeu_si128((__m128i *)p, h);
  #elif defined(__SSE4_1__)
    __m128i u = _mm_castps_si128(v);
    u = _mm_srli_epi32(u, 16);
    __m128i h = _mm_packus_epi32(u, u);
    _mm_storel_epi64((__m128i *)p, h);
  #elif defined(__SSE2__)
    __m128i u = _mm_castps_si128(v);
    u = _mm_srli_epi32(u, 16);
    __m128i a = _mm_shufflelo_epi16(u, _MM_SHUFFLE(2, 0, 2, 0));
    __m128i b = _mm_shufflehi_epi16(u, _MM_SHUFFLE(2, 0, 2, 0));
    b = _mm_srli_si128(b, 8);
    __m128i h = _mm_unpacklo_epi32(a, b);
    _mm_storel_epi64((__m128i *)p, h);
  #elif defined(__loongarch_asx)
    __m256i u = __lasx_xvsrli_w((__m256i)v, 16);
    __m256i h = __lasx_xvpickev_h(u, u);
    uint64_t q0, q1;
    __asm__ __volatile__(
      "xvpickve2gr.d %[q0], %u[h], 0\n\t"
      "xvpickve2gr.d %[q1], %u[h], 2\n\t"
      : [q0] "=r" (q0),
        [q1] "=r" (q1)
      : [h] "f" (h)
      : 
    );
    *((uint64_t *)p+0) = q0;
    *((uint64_t *)p+1) = q1;
  #elif defined(__loongarch_sx)
    mag_alignas(16) float tmp[4];
    __lsx_vst((__m128i)v, tmp, 0);
    for (int i=0; i < 4; ++i)
      p[i] = mag_float32_to_bfloat16(tmp[i]);
  #else
    *p = mag_float32_to_bfloat16(v);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_masked_bf16(mag_bfloat16_t *p, mag_vf32_t v, int n) {
  mag_bfloat16_t tmp[MAG_VF32_LANES];
  mag_vf32_storeu_bf16(tmp, v);
  for (int i=0; i < n; ++i) p[i] = tmp[i];
}

static MAG_AINLINE mag_vbf16_t mag_vbf16_loadu(const mag_bfloat16_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_u16((const uint16_t *)p);
  #elif defined(__AVX512F__)
    return (mag_vbf16_t)_mm512_loadu_si512((const void *)p);
  #elif defined(__AVX2__)
    return _mm256_loadu_si256((const __m256i *)p);
  #elif defined(__SSE2__)
    return _mm_loadu_si128((const __m128i *)p);
  #elif defined(__loongarch_asx)
    #error "TODO"
  #elif defined(__loongarch_sx)
    #error "TODO"
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vbf16_t mag_vbf16_broadcast_pair(const mag_bfloat16_t *p) {
  uint32_t pb;
  memcpy(&pb, p, sizeof pb);
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_u16_u32(vdupq_n_u32(pb));
  #elif defined(__AVX512F__)
    return (mag_vbf16_t)_mm512_set1_epi32((int)pb);
  #elif defined(__AVX2__)
    return _mm256_set1_epi32((int)pb);
  #elif defined(__SSE2__)
    return _mm_set1_epi32((int)pb);
  #elif defined(__loongarch_asx)
    #error "TODO"
  #elif defined(__loongarch_sx)
    #error "TODO"
  #else
    #error "TODO"
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_dpbf16(mag_vf32_t acc, mag_vbf16_t x, mag_vbf16_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_vf32_t xlo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x), 16));
    mag_vf32_t xhi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x), 16));
    mag_vf32_t ylo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(y), 16));
    mag_vf32_t yhi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(y), 16));
    return vfmaq_f32(vfmaq_f32(acc, xlo, ylo), xhi, yhi);
  #elif defined(__AVX512F__) && defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(acc, x, y);
  #elif defined(__AVX512F__)
    mag_vf32_t xlo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(x)), 16));
    mag_vf32_t xhi = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(x, 1)), 16));
    mag_vf32_t ylo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(y)), 16));
    mag_vf32_t yhi = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(y, 1)), 16));
    return _mm512_fmadd_ps(xhi, yhi, _mm512_fmadd_ps(xlo, ylo, acc));
  #elif defined(__AVX2__)
    mag_vf32_t xlo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(x)), 16));
    mag_vf32_t xhi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(x, 1)), 16));
    mag_vf32_t ylo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(y)),16));
    mag_vf32_t yhi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(y, 1)), 16));
    #ifdef __FMA__
      return _mm256_fmadd_ps(xhi, yhi, _mm256_fmadd_ps(xlo, ylo, acc));
    #else
      return _mm256_add_ps(acc, _mm256_add_ps(_mm256_mul_ps(alo, blo), _mm256_mul_ps(ahi, bhi)));
    #endif
  #elif defined(__SSE2__)
    __m128i z = _mm_setzero_si128();
    mag_vf32_t xlo = _mm_castsi128_ps(_mm_unpacklo_epi16(z, x));
    mag_vf32_t xhi = _mm_castsi128_ps(_mm_unpackhi_epi16(z, x));
    mag_vf32_t ylo = _mm_castsi128_ps(_mm_unpacklo_epi16(z, y));
    mag_vf32_t yhi = _mm_castsi128_ps(_mm_unpackhi_epi16(z, y));
    #ifdef __FMA__
      return _mm_fmadd_ps(xhi, yhi, _mm_fmadd_ps(xlo, ylo, acc));
    #else
      return _mm_add_ps(acc, _mm_add_ps(_mm_mul_ps(xlo, ylo), _mm_mul_ps(xhi, yhi)));
    #endif
  #elif defined(__loongarch_asx)
    #error "TODO"
  #elif defined(__loongarch_sx)
    #error "TODO"
  #else
    #error "TODO"
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_float8_e4m3fn(const mag_float8_e4m3fn_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint32x4_t x = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8((const uint8_t *)p))));
    uint32x4_t sign = vshlq_n_u32(vandq_u32(x, vdupq_n_u32(0x80)), 24);
    uint32x4_t abs = vandq_u32(x, vdupq_n_u32(0x7f));
    uint32x4_t exp = vshrq_n_u32(abs, 3);
    uint32x4_t mant = vandq_u32(abs, vdupq_n_u32(0x07));
    uint32x4_t norm = vorrq_u32(sign, vorrq_u32(vshlq_n_u32(vaddq_u32(exp, vdupq_n_u32(120)), 23), vshlq_n_u32(mant, 20)));
    uint32x4_t sub_bits = vorrq_u32(vreinterpretq_u32_f32(vmulq_f32(vcvtq_f32_u32(mant), vdupq_n_f32(0x1p-9f))), sign);
    return vreinterpretq_f32_u32(vbslq_u32(vceqq_u32(abs, vdupq_n_u32(0)), sign, vbslq_u32(vmvnq_u32(vceqq_u32(exp, vdupq_n_u32(0))), norm, sub_bits)));
  #elif defined(__AVX512F__)
    __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p));
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(0x80)), 24);
    __m512i abs = _mm512_and_si512(x, _mm512_set1_epi32(0x7f));
    __m512i exp = _mm512_srli_epi32(abs, 3);
    __m512i mant = _mm512_and_si512(abs, _mm512_set1_epi32(0x07));
    __m512i norm = _mm512_or_si512(sign, _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(120)), 23), _mm512_slli_epi32(mant, 20)));
    __m512i sub_bits = _mm512_or_si512(_mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(mant), _mm512_set1_ps(0x1p-9f))), sign);
    __m512i bits = _mm512_mask_blend_epi32(_mm512_cmpneq_epi32_mask(exp, _mm512_setzero_si512()), sub_bits, norm);
    bits = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(abs, _mm512_setzero_si512()), bits, sign);
    return _mm512_castsi512_ps(bits);
  #elif defined(__AVX2__)
    __m256i x = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x80)), 24);
    __m256i abs = _mm256_and_si256(x, _mm256_set1_epi32(0x7f));
    __m256i exp = _mm256_srli_epi32(abs, 3);
    __m256i mant = _mm256_and_si256(abs, _mm256_set1_epi32(0x07));
    __m256i norm = _mm256_or_si256(sign, _mm256_or_si256( _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23), _mm256_slli_epi32(mant, 20)));
    __m256i sub_bits = _mm256_or_si256(_mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(mant), _mm256_set1_ps(0x1p-9f))), sign);
    return _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_blendv_epi8(norm, sub_bits, _mm256_cmpeq_epi32(exp, _mm256_setzero_si256())), sign, _mm256_cmpeq_epi32(abs, _mm256_setzero_si256())));
  #elif defined(__SSE4_1__)
    __m128i x = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t *)p));
    __m128i sign = _mm_slli_epi32(_mm_and_si128(x, _mm_set1_epi32(0x80)), 24);
    __m128i abs = _mm_and_si128(x, _mm_set1_epi32(0x7f));
    __m128i exp = _mm_srli_epi32(abs, 3);
    __m128i mant = _mm_and_si128(abs, _mm_set1_epi32(0x07));
    __m128i norm = _mm_or_si128(sign, _mm_or_si128( _mm_slli_epi32(_mm_add_epi32(exp, _mm_set1_epi32(120)), 23), _mm_slli_epi32(mant, 20)));
    __m128i sub_bits = _mm_or_si128(_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(mant), _mm_set1_ps(0x1p-9f))), sign);
    return _mm_castsi128_ps(_mm_blendv_epi8(_mm_blendv_epi8(norm, sub_bits, _mm_cmpeq_epi32(exp, _mm_setzero_si128())), sign, _mm_cmpeq_epi32(abs, _mm_setzero_si128())));
  #else
    mag_vf32_t r;
    for (int i=0; i < MAG_VF32_LANES; ++i)
      ((float *)&r)[i] = mag_float8_e4m3fn_to_float32(p[i]);
    return r;
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_float8_e4m3fn(mag_float8_e4m3fn_t *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint32x4_t b = vreinterpretq_u32_f32(v);
    uint32x4_t sign = vandq_u32(b, vdupq_n_u32(0x80000000u));
    uint32x4_t abs = veorq_u32(b, sign);
    float32x4_t denorm_f = vaddq_f32(vreinterpretq_f32_u32(abs), vreinterpretq_f32_u32(vdupq_n_u32(0x46800000u)));
    uint32x4_t norm_r = vminq_u32(vshrq_n_u32(vaddq_u32(vaddq_u32(abs, vdupq_n_u32(-1006108673)), vandq_u32(vshrq_n_u32(abs, 20), vdupq_n_u32(1))), 20), vdupq_n_u32(0x7e));
    uint32x4_t sat_r = vbslq_u32(vcgtq_u32(abs, vdupq_n_u32(0x7f800000u)), vdupq_n_u32(0x7f), vdupq_n_u32(0x7e));
    norm_r = vbslq_u32(vcltq_u32(abs, vdupq_n_u32(0x3c800000u)),  vsubq_u32(vreinterpretq_u32_f32(denorm_f), vdupq_n_u32(0x46800000u)), norm_r);
    norm_r = vbslq_u32(vcgeq_u32(abs, vdupq_n_u32(0x43f00000u)), sat_r, norm_r);
    uint32x4_t r = vorrq_u32(norm_r, vshrq_n_u32(sign, 24));
    uint16x4_t r16 = vmovn_u32(r);
    uint8x8_t r8 = vmovn_u16(vcombine_u16(r16, vdup_n_u16(0)));
    vst1_lane_u32((uint32_t *)p, vreinterpret_u32_u8(r8), 0);
  #elif defined(__AVX512F__)
    __m512i b = _mm512_castps_si512(v);
    __m512i sign = _mm512_and_si512(b, _mm512_set1_epi32(0x80000000u));
    __m512i abs = _mm512_xor_si512(b, sign);
    __m512 denorm_f = _mm512_add_ps(_mm512_castsi512_ps(abs), _mm512_castsi512_ps(_mm512_set1_epi32(0x46800000u)));
    __m512i denorm_r = _mm512_sub_epi32(_mm512_castps_si512(denorm_f), _mm512_set1_epi32(0x46800000u));
    __m512i mant_lsb = _mm512_and_si512(_mm512_srli_epi32(abs, 20), _mm512_set1_epi32(1));
    __m512i nb = _mm512_add_epi32(abs, _mm512_set1_epi32(-1006108673));
    __m512i norm_r = _mm512_min_epu32(_mm512_srli_epi32(_mm512_add_epi32(nb, mant_lsb), 20), _mm512_set1_epi32(0x7e));
    __m512i sat_r = _mm512_mask_blend_epi32(_mm512_cmpgt_epu32_mask(abs, _mm512_set1_epi32(0x7f800000u)), _mm512_set1_epi32(0x7e), _mm512_set1_epi32(0x7f));
    norm_r = _mm512_mask_blend_epi32(_mm512_cmplt_epu32_mask(abs, _mm512_set1_epi32(0x3c800000u)), norm_r, denorm_r);
    norm_r = _mm512_mask_blend_epi32(_mm512_cmpge_epu32_mask(abs, _mm512_set1_epi32(0x43f00000u)), norm_r, sat_r);
    _mm_storeu_si128((__m128i *)p, _mm512_cvtepi32_epi8(_mm512_or_si512(norm_r, _mm512_srli_epi32(sign, 24))));
  #elif defined(__AVX2__)
    __m256i b = _mm256_castps_si256(v);
    __m256i sign = _mm256_and_si256(b, _mm256_set1_epi32(0x80000000u));
    __m256i abs = _mm256_xor_si256(b, sign);
    __m256 denorm_f = _mm256_add_ps(_mm256_castsi256_ps(abs), _mm256_castsi256_ps(_mm256_set1_epi32(0x46800000u)));
    __m256i denorm_r = _mm256_sub_epi32(_mm256_castps_si256(denorm_f), _mm256_set1_epi32(0x46800000u));
    __m256i mant_lsb = _mm256_and_si256(_mm256_srli_epi32(abs, 20), _mm256_set1_epi32(1));
    __m256i nb = _mm256_add_epi32(abs, _mm256_set1_epi32(-1006108673));
    __m256i norm_r = _mm256_min_epu32(_mm256_srli_epi32(_mm256_add_epi32(nb, mant_lsb), 20), _mm256_set1_epi32(0x7e));
    __m256i sat_r = _mm256_blendv_epi8(_mm256_set1_epi32(0x7e), _mm256_set1_epi32(0x7f), _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x7f800000u)));
    norm_r = _mm256_blendv_epi8(norm_r, denorm_r, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x3c800000u), abs));
    norm_r = _mm256_blendv_epi8(norm_r, sat_r, _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x43efffffu)));
    __m256i r32 = _mm256_or_si256(norm_r, _mm256_srli_epi32(sign, 24));
    _mm_storel_epi64((__m128i *)p, _mm_packus_epi16(_mm_packus_epi32(_mm256_castsi256_si128(r32), _mm256_extracti128_si256(r32, 1)), _mm_setzero_si128()));
  #elif defined(__SSE4_1__)
    __m128i b = _mm_castps_si128(v);
    __m128i sign = _mm_and_si128(b, _mm_set1_epi32((int)0x80000000u));
    __m128i abs = _mm_xor_si128(b, sign);
    __m128 denorm_f = _mm_add_ps(_mm_castsi128_ps(abs), _mm_castsi128_ps(_mm_set1_epi32(0x46800000u)));
    __m128i denorm_r = _mm_sub_epi32(_mm_castps_si128(denorm_f), _mm_set1_epi32(0x46800000u));
    __m128i mant_lsb = _mm_and_si128(_mm_srli_epi32(abs, 20), _mm_set1_epi32(1));
    __m128i norm_r = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(abs, _mm_set1_epi32(-1006108673)), mant_lsb), 20);
    norm_r = _mm_blendv_epi8(_mm_min_epu32(norm_r, _mm_set1_epi32(0x7e)), denorm_r, _mm_cmplt_epi32(abs, _mm_set1_epi32(0x3c800000)));
    norm_r = _mm_blendv_epi8(norm_r, _mm_blendv_epi8(_mm_set1_epi32(0x7e), _mm_set1_epi32(0x7f), _mm_cmpgt_epi32(abs, _mm_set1_epi32(0x7f800000))), _mm_cmpgt_epi32(abs, _mm_set1_epi32(0x43efffff)));
    *(uint32_t *)p = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_or_si128(norm_r, _mm_srli_epi32(sign, 24)), _mm_setzero_si128()), _mm_setzero_si128()));
  #else
    for (int i=0; i < MAG_VF32_LANES; ++i) {
      p[i] = mag_float32_to_float8_e4m3fn(((const float *)&v)[i]);
    }
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_masked_float8_e4m3fn(mag_float8_e4m3fn_t *p, mag_vf32_t v, int n) {
  mag_float8_e4m3fn_t tmp[MAG_VF32_LANES];
  mag_vf32_storeu_float8_e4m3fn(tmp, v);
  for (int i=0; i < n; ++i) p[i] = tmp[i];
}

/* i32 vector */
static MAG_AINLINE mag_vi32_t mag_vi32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(0);
  #elif defined(__AVX512F__)
    return _mm512_setzero_si512();
  #elif defined(__AVX2__)
    return _mm256_setzero_si256();
  #elif defined(__SSE2__)
    return _mm_setzero_si128();
  #elif defined(__loongarch_asx)
    return __lasx_xvreplgr2vr_w(0);
  #elif defined(__loongarch_sx)
    return __lsx_vreplgr2vr_w(0);
  #else
    return 0;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_splat(int32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_set1_epi32(x);
  #elif defined(__AVX2__)
    return _mm256_set1_epi32(x);
  #elif defined(__SSE2__)
    return _mm_set1_epi32(x);
  #elif defined(__loongarch_asx)
    return __lasx_xvreplgr2vr_w(x);
  #elif defined(__loongarch_sx)
    return __lsx_vreplgr2vr_w(x);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_broadcast(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(*p);
  #elif defined(__AVX512F__)
    return _mm512_set1_epi32(*p);
  #elif defined(__AVX2__)
    return _mm256_set1_epi32(*p);
  #elif defined(__SSE2__)
    return _mm_set1_epi32(*p);
  #elif defined(__loongarch_asx)
    return __lasx_xvldrepl_w(p, 0);
  #elif defined(__loongarch_sx)
    return __lsx_vldrepl_w(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loada(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_s32(p);
  #elif defined(__AVX512F__)
    return _mm512_load_epi32(p);
  #elif defined(__AVX2__)
    return _mm256_load_si256((const __m256i *)p);
  #elif defined(__SSE2__)
    return _mm_load_si128((const __m128i *)p);
  #elif defined(__loongarch_asx)
    return __lasx_xvld(p, 0);
  #elif defined(__loongarch_sx)
    return __lsx_vld(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loadu(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_s32(p);
  #elif defined(__AVX512F__)
    return _mm512_loadu_epi32(p);
  #elif defined(__AVX2__)
    return _mm256_loadu_si256((const __m256i *)p);
  #elif defined(__SSE2__)
    return _mm_loadu_si128((const __m128i *)p);
  #elif defined(__loongarch_asx)
    return __lasx_xvld(p, 0);
  #elif defined(__loongarch_sx)
    return __lsx_vld(p, 0);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loadu_masked(const int32_t *p, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) int32_t tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_s32(tmp);
  #elif defined(__AVX512F__)
    return _mm512_maskz_loadu_epi32((__mmask16)((1u<<n)-1u), p);
  #elif defined(__AVX2__)
    mag_alignas(32) int32_t tmp[8] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return _mm256_load_si256((const __m256i *)tmp);
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return _mm_load_si128((const __m128i *)tmp);
  #elif defined(__loongarch_asx)
    mag_alignas(32) int32_t tmp[8] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return __lasx_xvld(tmp, 0);
  #elif defined(__loongarch_sx)
    mag_alignas(16) int32_t tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return __lsx_vld(tmp, 0);
  #else
    return n ? *p : 0;
  #endif
}
static MAG_AINLINE void mag_vi32_storea(int32_t *p, mag_vi32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_s32(p, v);
  #elif defined(__AVX512F__)
    _mm512_store_epi32(p, v);
  #elif defined(__AVX2__)
    _mm256_store_si256((__m256i *)p, v);
  #elif defined(__SSE2__)
    _mm_store_si128((__m128i *)p, v);
  #elif defined(__loongarch_asx)
    __lasx_xvstx(v, p, 0);
  #elif defined(__loongarch_sx)
    __lsx_vst(v, p, 0);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vi32_storeu(int32_t *p, mag_vi32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_s32(p, v);
  #elif defined(__AVX512F__)
    _mm512_storeu_epi32(p, v);
  #elif defined(__AVX2__)
    _mm256_storeu_si256((__m256i *)p, v);
  #elif defined(__SSE2__)
    _mm_storeu_si128((__m128i *)p, v);
  #elif defined(__loongarch_asx)
    __lasx_xvstx(v, p, 0);
  #elif defined(__loongarch_sx)
    __lsx_vst(v, p, 0);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vi32_storeu_masked(int32_t *p, mag_vi32_t v, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) int32_t tmp[4];
    vst1q_s32(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__AVX512F__)
    _mm512_mask_storeu_epi32(p, (__mmask16)((1u<<n)-1u), v);
  #elif defined(__AVX2__)
    mag_alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i *)tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t tmp[4];
    _mm_store_si128((__m128i *)tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__loongarch_asx)
    mag_alignas(32) int32_t tmp[8];
    __lasx_xvst(v, tmp, 0);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__loongarch_sx)
    mag_alignas(16) int32_t tmp[4];
    __lsx_vst(v, tmp, 0);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #else
    if (n) *p = v;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpeq(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vceqq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_EQ);
  #elif defined(__AVX2__)
    return _mm256_cmpeq_epi32(x, y);
  #elif defined(__SSE2__)
    return _mm_cmpeq_epi32(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvseq_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vseq_w(x, y);
  #else
    return x == y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpne(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(vceqq_s32(x, y));
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NE);
  #elif defined(__AVX2__)
    return _mm256_xor_si256(_mm256_cmpeq_epi32(x, y), _mm256_set1_epi32(-1));
  #elif defined(__SSE2__)
    return _mm_xor_si128(_mm_cmpeq_epi32(x, y), _mm_set1_epi32(-1));
  #elif defined(__loongarch_asx)
    return __lasx_xvxor_v(__lasx_xvseq_w(x, y), __lasx_xvreplgr2vr_w(-1));
  #elif defined(__loongarch_sx)
    return __lsx_vxor_v(__lsx_vseq_w(x, y), __lsx_vreplgr2vr_w(-1));
  #else
    return x != y ? ~0u : 0u;
  #endif
}

static MAG_AINLINE mag_vmask32_t mag_vi32_cmpgt(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgtq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_GT);
  #elif defined(__AVX2__)
    return _mm256_cmpgt_epi32(x, y);
  #elif defined(__SSE4_1__)
    return _mm_cmpgt_epi32(x, y);
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] > ay[i] ? -1 : 0;
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvslt_w(y, x);
  #elif defined(__loongarch_sx)
    return __lsx_vslt_w(y, x);
  #else
    return x > y ? ~0u : 0u;
  #endif
}

static MAG_AINLINE mag_vmask32_t mag_vi32_cmplt(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcltq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LT);
  #elif defined(__AVX2__)
    return _mm256_cmpgt_epi32(y, x);
  #elif defined(__SSE4_1__)
    return _mm_cmpgt_epi32(y, x);
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] < ay[i] ? -1 : 0;
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvslt_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vslt_w(x, y);
  #else
    return x < y ? ~0u : 0u;
  #endif
}

static MAG_AINLINE mag_vmask32_t mag_vi32_cmpge(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgeq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_GE);
  #elif defined(__AVX2__)
    return _mm256_xor_si256(_mm256_cmpgt_epi32(y, x), _mm256_set1_epi32(-1));
  #elif defined(__SSE4_1__)
    return _mm_xor_si128(_mm_cmpgt_epi32(y, x), _mm_set1_epi32(-1));
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] >= ay[i] ? -1 : 0;
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvsle_w(y, x);
  #elif defined(__loongarch_sx)
    return __lsx_vsle_w(y, x);
  #else
    return x >= y ? ~0u : 0u;
  #endif
}

static MAG_AINLINE mag_vmask32_t mag_vi32_cmple(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcleq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LE);
  #elif defined(__AVX2__)
    return _mm256_xor_si256(_mm256_cmpgt_epi32(x, y), _mm256_set1_epi32(-1));
  #elif defined(__SSE4_1__)
    return _mm_xor_si128(_mm_cmpgt_epi32(x, y), _mm_set1_epi32(-1));
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] <= ay[i] ? -1 : 0;
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvsle_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vsle_w(x, y);
  #else
    return x <= y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_blend(mag_vmask32_t m, mag_vi32_t t, mag_vi32_t f) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbslq_s32(m, t, f);
  #elif defined(__AVX512F__)
    return _mm512_mask_blend_epi32(m, f, t);
  #elif defined(__AVX2__)
    return _mm256_or_si256(_mm256_and_si256(m, t), _mm256_andnot_si256(m, f));
  #elif defined(__SSE2__)
    return _mm_or_si128(_mm_and_si128(m, t), _mm_andnot_si128(m, f));
  #elif defined(__loongarch_asx)
    return __lasx_xvbitsel_v(f, t, m);
  #elif defined(__loongarch_sx)
  return __lsx_vbitsel_v(f, t, m);
  #else
    return (m&t)|(~m&f);
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_add(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_add_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_add_epi32(x, y);
  #elif defined(__SSE2__)
    return _mm_add_epi32(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvadd_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vadd_w(x, y);
  #else
    return x+y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_sub(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vsubq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_sub_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_sub_epi32(x, y);
  #elif defined(__SSE2__)
    return _mm_sub_epi32(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvsub_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vsub_w(x, y);
  #else
    return x-y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_mul(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_mullo_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_mullo_epi32(x, y);
  #elif defined(__SSE4_1__)
    return _mm_mullo_epi32(x, y);
  #elif defined(__SSE2__)
    __m128i prod02 = _mm_mul_epu32(x, y);
    __m128i prod13 = _mm_mul_epu32(_mm_srli_si128(x, 4), _mm_srli_si128(y, 4));
    return _mm_unpacklo_epi64(_mm_unpacklo_epi32(prod02, prod13), _mm_unpackhi_epi32(prod02, prod13));
  #elif defined(__loongarch_asx)
    return __lasx_xvmul_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vmul_w(x, y);
  #else
    return x*y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_mulhi_u32(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint64x2_t p0 = vmull_u32(vget_low_u32(vreinterpretq_u32_s32(x)), vget_low_u32(vreinterpretq_u32_s32(y)));
    uint64x2_t p1 = vmull_u32(vget_high_u32(vreinterpretq_u32_s32(x)), vget_high_u32(vreinterpretq_u32_s32(y)));
    uint32x2_t h0 = vmovn_u64(vshrq_n_u64(p0, 32));
    uint32x2_t h1 = vmovn_u64(vshrq_n_u64(p1, 32));
    return vreinterpretq_s32_u32(vcombine_u32(h0, h1));
  #elif defined(__AVX512F__)
    __m512i even = _mm512_mul_epu32(x, y);
    __m512i xo = _mm512_srli_epi64(x, 32);
    __m512i yo = _mm512_srli_epi64(y, 32);
    __m512i odd = _mm512_mul_epu32(xo, yo);
    __m512i even_hi = _mm512_srli_epi64(even, 32);
    __m512i odd_hi  = _mm512_slli_epi64(_mm512_srli_epi64(odd, 32), 32);
    return _mm512_or_si512(even_hi, odd_hi);
  #elif defined(__AVX2__)
    __m256i even = _mm256_mul_epu32(x, y);
    __m256i xo = _mm256_srli_epi64(x, 32);
    __m256i yo = _mm256_srli_epi64(y, 32);
    __m256i odd = _mm256_mul_epu32(xo, yo);
    __m256i even_hi = _mm256_srli_epi64(even, 32);
    __m256i odd_hi  = _mm256_slli_epi64(_mm256_srli_epi64(odd, 32), 32);
    return _mm256_or_si256(even_hi, odd_hi);
  #elif defined(__SSE2__)
    __m128i even = _mm_mul_epu32(x, y);
    __m128i xo = _mm_srli_epi64(x, 32);
    __m128i yo = _mm_srli_epi64(y, 32);
    __m128i odd = _mm_mul_epu32(xo, yo);
    __m128i even_hi = _mm_srli_epi64(even, 32);
    __m128i odd_hi  = _mm_slli_epi64(_mm_srli_epi64(odd, 32), 32);
    return _mm_or_si128(even_hi, odd_hi);
  #else
    return (uint32_t)(((uint64_t)(uint32_t)x*(uint64_t)(uint32_t)y)>>32);
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_and(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_and_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_and_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_and_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvand_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vand_v(x, y);
  #else
    return x&y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_or(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vorrq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_or_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_or_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_or_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvor_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vor_v(x, y);
  #else
    return x|y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_xor(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_xor_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_xor_si256(x, y);
  #elif defined(__SSE2__)
    return _mm_xor_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvxor_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vxor_v(x, y);
  #else
    return x^y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_andnot(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vbicq_u32(vreinterpretq_u32_s32(y), vreinterpretq_u32_s32(x)));
  #elif defined(__AVX512F__)
    return _mm512_andnot_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_andnot_si256(x, y);
  #elif defined(__SSE2__)
      return _mm_andnot_si128(x, y);
  #elif defined(__loongarch_asx)
    return __lasx_xvandn_v(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vandn_v(x, y);
  #else
    return ~x&y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_not(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vmvnq_u32(vreinterpretq_u32_s32(x)));
  #elif defined(__AVX512F__)
    return _mm512_xor_si512(x, _mm512_set1_epi32(-1));
  #elif defined(__AVX2__)
    return _mm256_xor_si256(x, _mm256_set1_epi32(-1));
  #elif defined(__SSE2__)
      return _mm_xor_si128(x, _mm_set1_epi32(-1));
  #elif defined(__loongarch_asx)
    return __lasx_xvxor_v(x, __lasx_xvreplgr2vr_w(-1));
  #elif defined(__loongarch_sx)
    return __lsx_vxor_v(x, __lsx_vreplgr2vr_w(-1));
  #else
    return ~x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_slli(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(n)));
  #elif defined(__AVX512F__)
    return _mm512_slli_epi32(x, n);
  #elif defined(__AVX2__)
    return _mm256_slli_epi32(x, n);
  #elif defined(__SSE2__)
    return _mm_slli_epi32(x, n);
  #elif defined(__loongarch_asx)
    return __lasx_xvsll_w(x, __lasx_xvreplgr2vr_w(n));
  #elif defined(__loongarch_sx)
    return __lsx_vsll_w(x, __lsx_vreplgr2vr_w(n));
  #else
    return x<<n;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_srli(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(-n)));
  #elif defined(__AVX512F__)
    return _mm512_srli_epi32(x, n);
  #elif defined(__AVX2__)
    return _mm256_srli_epi32(x, n);
  #elif defined(__SSE2__)
    return _mm_srli_epi32(x, n);
  #elif defined(__loongarch_asx)
    return __lasx_xvsrl_w(x, __lasx_xvreplgr2vr_w(n));
  #elif defined(__loongarch_sx)
    return __lsx_vsrl_w(x, __lsx_vreplgr2vr_w(n));
  #else
    return (mag_vi32_t)((uint32_t)x >> n);
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_srai(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vshlq_s32(x, vdupq_n_s32(-n));
  #elif defined(__AVX512F__)
    return _mm512_srai_epi32(x, n);
  #elif defined(__AVX2__)
    return _mm256_srai_epi32(x, n);
  #elif defined(__SSE2__)
    return _mm_srai_epi32(x, n);
  #elif defined(__loongarch_asx)
    return __lasx_xvsra_w(x, __lasx_xvreplgr2vr_w(n));
  #elif defined(__loongarch_sx)
    return __lsx_vsra_w(x, __lsx_vreplgr2vr_w(n));
  #else
    return x>>n;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_min(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_min_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_min_epi32(x, y);
  #elif defined(__SSE4_1__)
    return _mm_min_epi32(x, y);
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] < ay[i] ? ax[i] : ay[i];
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvmin_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vmin_w(x, y);
  #else
    return x<y ? x : y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_max(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_max_epi32(x, y);
  #elif defined(__AVX2__)
    return _mm256_max_epi32(x, y);
  #elif defined(__SSE4_1__)
    return _mm_max_epi32(x, y);
  #elif defined(__SSE2__)
    mag_alignas(16) int32_t ax[4], ay[4], r[4];
    _mm_store_si128((__m128i *)ax, x);
    _mm_store_si128((__m128i *)ay, y);
    for (int i=0; i < 4; ++i) r[i] = ax[i] > ay[i] ? ax[i] : ay[i];
    return _mm_load_si128((const __m128i *)r);
  #elif defined(__loongarch_asx)
    return __lasx_xvmax_w(x, y);
  #elif defined(__loongarch_sx)
    return __lsx_vmax_w(x, y);
  #else
    return x>y ? x : y;
  #endif
}
static MAG_AINLINE int32_t mag_vi32_reduce_add(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddvq_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_reduce_add_epi32(x);
  #elif defined(__AVX2__)
    __m128i acc = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    acc = _mm_hadd_epi32(acc, acc);
    acc = _mm_hadd_epi32(acc, acc);
    return _mm_cvtsi128_si32(acc);
  #elif defined(__SSE2__)
    __m128i acc = _mm_add_epi32(x, _mm_srli_si128(x, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
  #elif defined(__loongarch_asx)
    mag_alignas(32) uint32_t t[8];
    __lasx_xvst(x, t, 0);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
  #elif defined(__loongarch_sx)
    mag_alignas(16) uint32_t t[4];
    __lsx_vst(x, t, 0);
    return t[0]+t[1]+t[2]+t[3];
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_reinterpret_from_vi32(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(x);
  #elif defined(__AVX2__)
    return _mm256_castsi256_ps(x);
  #elif defined(__SSE2__)
    return _mm_castsi128_ps(x);
  #elif defined(__loongarch_asx)
    return (__m256)x;
  #elif defined(__loongarch_sx)
    return (__m128)x;
  #else
    mag_vf32_t r;
    memcpy(&r, &x, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_reinterpret_from_vf32(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_castps_si512(x);
  #elif defined(__AVX2__)
    return _mm256_castps_si256(x);
  #elif defined(__SSE2__)
    return _mm_castps_si128(x);
  #elif defined(__loongarch_asx)
    return (__m256i)x;
  #elif defined(__loongarch_sx)
    return (__m128i)x;
  #else
    mag_vi32_t r;
    memcpy(&r, &x, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vi32_to_f32(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvtq_f32_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_cvtepi32_ps(x);
  #elif defined(__AVX2__)
    return _mm256_cvtepi32_ps(x);
  #elif defined(__SSE2__)
    return _mm_cvtepi32_ps(x);
  #elif defined(__loongarch_asx)
    return __lasx_xvffint_s_w(x);
  #elif defined(__loongarch_sx)
    return __lsx_vffint_s_w(x);
  #else
    return (mag_vf32_t)x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vf32_trunc_to_vi32(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvtq_s32_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_cvttps_epi32(x);
  #elif defined(__AVX2__)
    return _mm256_cvttps_epi32(x);
  #elif defined(__SSE2__)
    return _mm_cvttps_epi32(x);
  #elif defined(__loongarch_asx)
    return __lasx_xvftintrz_w_s(x);
  #elif defined(__loongarch_sx)
    return __lsx_vftintrz_w_s(x);
  #else
    return (mag_vi32_t)x;
  #endif
}

static MAG_AINLINE mag_vi32_t mag_vi32_from_vmask32(mag_vmask32_t m) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(m);
  #elif defined(__AVX512F__)
    return _mm512_maskz_set1_epi32(m, -1);
  #elif defined(__AVX2__)
    return m;
  #elif defined(__SSE2__)
    return m;
  #elif defined(__loongarch_asx)
    return m;
  #elif defined(__loongarch_sx)
    return m;
  #else
    return (mag_vi32_t)m;
  #endif
}

static MAG_AINLINE mag_vi32_t mag_vi32_iota(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return (mag_vi32_t){0,1,2,3};
  #elif defined(__AVX512F__)
    return _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  #elif defined(__AVX2__)
    return _mm256_setr_epi32(0,1,2,3,4,5,6,7);
  #elif defined(__SSE2__)
    return _mm_setr_epi32(0,1,2,3);
  #elif defined(__loongarch_asx)
  #error "TODO"
  #elif defined(__loongarch_sx)
  #error "TODO"
  #else
    return 0;
  #endif
}

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

#ifdef __cplusplus
}
#endif

#endif
