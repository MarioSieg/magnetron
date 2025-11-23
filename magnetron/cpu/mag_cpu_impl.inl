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

#include "mag_cpu.h"

#include <core/mag_tensor.h>
#include <core/mag_cpuid.h>
#include <core/mag_alloc.h>
#include <core/mag_coords.h>
#include <core/mag_coords_iter.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

#include <float.h>
#include <math.h>

#define mag_cvt_nop(x) (x)
#define mag_cvt_i642bool(x) (!!(x))
#define mag_cvt_i642i32(x) ((int32_t)(x))

#define mag_cmd_in(i) (payload->cmd->in[(i)])
#define mag_cmd_out(i) (payload->cmd->out[(i)])
#define mag_cmd_attr(i) (payload->cmd->attrs[(i)])

/* Uniform names for macro expansion */
typedef uint8_t mag_u8_t;
typedef int8_t mag_i8_t;
typedef uint16_t mag_u16_t;
typedef int16_t mag_i16_t;
typedef uint32_t mag_u32_t;
typedef int32_t mag_i32_t;
typedef uint64_t mag_u64_t;
typedef int64_t mag_i64_t;
typedef uint8_t mag_bool_t;

#define mag_e8m23p(t) ((const mag_e8m23_t*)mag_tensor_get_data_ptr(t))
#define mag_e8m23p_mut(t) ((mag_e8m23_t*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p(t) ((const mag_e5m10_t*)mag_tensor_get_data_ptr(t))
#define mag_e5m10p_mut(t) ((mag_e5m10_t*)mag_tensor_get_data_ptr(t))
#define mag_boolp(t) ((const mag_bool_t*)mag_tensor_get_data_ptr(t))
#define mag_boolp_mut(t) ((mag_bool_t*)mag_tensor_get_data_ptr(t))
#define mag_u8p(t) ((const uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_u8p_mut(t) ((uint8_t*)mag_tensor_get_data_ptr(t))
#define mag_i8p(t) ((const int8_t*)mag_tensor_get_data_ptr(t))
#define mag_i8p_mut(t) ((int8_t*)mag_tensor_get_data_ptr(t))
#define mag_u16p(t) ((const uint16_t*)mag_tensor_get_data_ptr(t))
#define mag_u16p_mut(t) ((uint16_t*)mag_tensor_get_data_ptr(t))
#define mag_i16p(t) ((const int16_t*)mag_tensor_get_data_ptr(t))
#define mag_i16p_mut(t) ((int16_t*)mag_tensor_get_data_ptr(t))
#define mag_u32p(t) ((const uint32_t*)mag_tensor_get_data_ptr(t))
#define mag_u32p_mut(t) ((uint32_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p(t) ((const int32_t*)mag_tensor_get_data_ptr(t))
#define mag_i32p_mut(t) ((int32_t*)mag_tensor_get_data_ptr(t))
#define mag_u64p(t) ((const uint64_t*)mag_tensor_get_data_ptr(t))
#define mag_u64p_mut(t) ((uint64_t*)mag_tensor_get_data_ptr(t))
#define mag_i64p(t) ((const int64_t*)mag_tensor_get_data_ptr(t))
#define mag_i64p_mut(t) ((int64_t*)mag_tensor_get_data_ptr(t))

#define MAG_MM_SCRATCH_ALIGN MAG_DESTRUCTIVE_INTERFERENCE_SIZE

typedef struct mag_scratch_buf_t {
    void *top;
    size_t cap;
} mag_scratch_buf_t;

static MAG_THREAD_LOCAL mag_scratch_buf_t mag_tls_scratch = {0};

static void *mag_sb_acquire(size_t size) {
    mag_scratch_buf_t *sb = &mag_tls_scratch;
    if (size <= sb->cap) return sb->top; /* Enough space allocated */
    sb->top = (*mag_alloc)(sb->top, size, MAG_MM_SCRATCH_ALIGN); /* Reallocate */
    sb->cap = size;
    void *p = sb->top;
#ifndef _MSC_VER
    p = __builtin_assume_aligned(p, MAG_MM_SCRATCH_ALIGN);
#endif
    return p;
}

static void mag_sb_release(void) {
    mag_scratch_buf_t *sb = &mag_tls_scratch;
    if (sb->top) (*mag_alloc)(sb->top, 0, MAG_MM_SCRATCH_ALIGN);
    sb->top = NULL;
    sb->cap = 0;
}

#ifdef __AVX512F__ /* Vector register width in bytes */
#define MAG_VREG_WIDTH 64
#elif defined(__AVX__)
#define MAG_VREG_WIDTH 32
#elif defined(__SSE2__)
#define MAG_VREG_WIDTH 16
#elif defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON))
#define MAG_VREG_WIDTH 16
#else
#define MAG_VREG_WIDTH 16
#endif

#if defined(_MSC_VER)
typedef uint16_t __fp16; /* MSVC does not support __fp16. */
#ifdef __AVX2__ /*MSVC does not define FMA and F16C with AVX 2*/
#define __FMA__ 1
#define __F16C__ 1
#endif
#endif

static MAG_AINLINE mag_e5m10_t mag_e8m23_cvt_e5m10(mag_e8m23_t x) {
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
        mag_e8m23_t f;
    } castor;
    mag_e8m23_t base = fabs(x)*0x1.0p+112f*0x1.0p-110f;
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
    return (mag_e5m10_t) {
        .bits=r
    };
}

static MAG_AINLINE mag_e8m23_t mag_e5m10_cvt_e8m23(mag_e5m10_t x) {
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
        mag_e8m23_t f;
    } castor;
    uint32_t w = (uint32_t)x.bits<<16;
    uint32_t sign = w & 0x80000000u;
    uint32_t two_w = w+w;
    uint32_t offs = 0xe0u<<23;
    uint32_t t1 = (two_w>>4) + offs;
    uint32_t t2 = (two_w>>17) | (126u<<23);
    castor.u = t1;
    mag_e8m23_t norm_x = castor.f*0x1.0p-112f;
    castor.u = t2;
    mag_e8m23_t denorm_x = castor.f-0.5f;
    uint32_t denorm_cutoff = 1u<<27;
    uint32_t r = sign | (two_w < denorm_cutoff
                         ? (castor.f = denorm_x, castor.u)
                         : (castor.f = norm_x, castor.u));
    castor.u = r;
    return castor.f;
#endif
}

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

static float32x4_t mag_simd_exp_e8m23(float32x4_t x) {
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

static float32x4_t mag_simd_tanh_e8m23(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.f);
    float32x4_t m1 = vdupq_n_f32(-1.f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t m2 = vdupq_n_f32(-2.0f);
    float32x4_t a = vmulq_f32(m2, x);
    float32x4_t b = mag_simd_exp_e8m23(a);
    float32x4_t c = vaddq_f32(one, b);
    float32x4_t inv = vrecpeq_f32(c);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    return vaddq_f32(m1, vmulq_f32(two, inv));
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

static __m512 mag_simd_exp_e8m23(__m512 x) {
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

static __m512 mag_simd_tanh_e8m23(__m512 x) {
    __m512 one = _mm512_set1_ps(1.f);
    __m512 neg_one = _mm512_set1_ps(-1.f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    __m512 a = _mm512_mul_ps(neg_two, x);
    __m512 b = mag_simd_exp_e8m23(a);
    __m512 c = _mm512_add_ps(one, b);
    __m512 inv = _mm512_rcp14_ps(c);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    return _mm512_fmadd_ps(two, inv, neg_one);
}

#elif defined(__AVX2__) && defined(__FMA__)

#define mag_m256ps_K(T, name, x) static const mag_alignas(32) T mag_m256_##name[8] = {(x),(x),(x),(x),(x),(x),(x),(x)}

mag_m256ps_K(mag_e8m23_t, 1, 1.0f);
mag_m256ps_K(mag_e8m23_t, 0p5, 0.5f);
mag_m256ps_K(int32_t, min_norm_pos, 0x00800000);
mag_m256ps_K(int32_t, mant_mask, 0x7f800000);
mag_m256ps_K(int32_t, inv_mant_mask, ~0x7f800000);
mag_m256ps_K(int32_t, sign_mask, 0x80000000);
mag_m256ps_K(int32_t, inv_sign_mask, ~0x80000000);
mag_m256ps_K(int32_t, 0x7f, 0x7f);

static __m256 mag_simd_exp_e8m23(__m256 x) {
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

static __m256 mag_simd_tanh_e8m23(__m256 x) {
    __m256 one = _mm256_set1_ps(1.f);
    __m256 neg_one = _mm256_set1_ps(-1.f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    __m256 a = _mm256_mul_ps(neg_two, x);
    __m256 b = mag_simd_exp_e8m23(a);
    __m256 c = _mm256_add_ps(one, b);
    __m256 inv = _mm256_rcp_ps(c);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    return _mm256_fmadd_ps(two, inv, neg_one);
}

static __m256 mag_simd_log_e8m23(__m256 x) {
    mag_m256ps_K(mag_e8m23_t, poly_SQRTHF, 0.707106781186547524);
    mag_m256ps_K(mag_e8m23_t, poly_log_p0, 7.0376836292e-2);
    mag_m256ps_K(mag_e8m23_t, poly_log_p1, -1.1514610310e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p2, 1.1676998740e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p3, -1.2420140846e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p4, +1.4249322787e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p5, -1.6668057665e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p6, +2.0000714765e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p7, -2.4999993993e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_p8, +3.3333331174e-1);
    mag_m256ps_K(mag_e8m23_t, poly_log_q1, -2.12194440e-4);
    mag_m256ps_K(mag_e8m23_t, poly_log_q2, 0.693359375);
    __m256i imm0;
    __m256 one = *(__m256 *)mag_m256_1;
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    x = _mm256_max_ps(x, *(__m256 *)mag_m256_min_norm_pos);
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, *(__m256 *)mag_m256_inv_mant_mask);
    x = _mm256_or_ps(x, *(__m256 *)mag_m256_0p5);
    imm0 = _mm256_sub_epi32(imm0, *(__m256i *)mag_m256_0x7f);
    __m256 e = _mm256_cvtepi32_ps(imm0);
    e = _mm256_add_ps(e, one);
    __m256 mask = _mm256_cmp_ps(x, *(__m256 *)mag_m256_poly_SQRTHF, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);
    __m256 z = _mm256_mul_ps(x,x);
    __m256 y = *(__m256 *)mag_m256_poly_log_p0;
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p1);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p2);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p3);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p4);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p5);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p6);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p7);
    y = _mm256_fmadd_ps(y, x, *(__m256 *)mag_m256_poly_log_p8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    tmp = _mm256_mul_ps(e, *(__m256 *)mag_m256_poly_log_q1);
    y = _mm256_add_ps(y, tmp);
    tmp = _mm256_mul_ps(z, *(__m256 *)mag_m256_0p5);
    y = _mm256_sub_ps(y, tmp);
    tmp = _mm256_mul_ps(e, *(__m256 *)mag_m256_poly_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

static void mag_simd_sincos_e8m23(__m256 x, __m256 *s, __m256 *c) {
    mag_m256ps_K(mag_e8m23_t, minus_cephes_DP1, -0.78515625);
    mag_m256ps_K(mag_e8m23_t, minus_cephes_DP2, -2.4187564849853515625e-4);
    mag_m256ps_K(mag_e8m23_t, minus_cephes_DP3, -3.77489497744594108e-8);
    mag_m256ps_K(mag_e8m23_t, sincof_p0, -1.9515295891e-4);
    mag_m256ps_K(mag_e8m23_t, sincof_p1,  8.3321608736e-3);
    mag_m256ps_K(mag_e8m23_t, sincof_p2, -1.6666654611e-1);
    mag_m256ps_K(mag_e8m23_t, coscof_p0,  2.443315711809948e-005);
    mag_m256ps_K(mag_e8m23_t, coscof_p1, -1.388731625493765e-003);
    mag_m256ps_K(mag_e8m23_t, coscof_p2,  4.166664568298827e-002);
    mag_m256ps_K(mag_e8m23_t, cephes_FOPI, 1.27323954473516);
    __m256i v2 = _mm256_set1_epi32(2);
    __m256i v4 = _mm256_set1_epi32(4);
    __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    __m256i imm0, imm2, imm4;
    sign_bit_sin = x;
    x = _mm256_and_ps(x, *(__m256 *)mag_m256_inv_sign_mask);
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256 *)mag_m256_sign_mask);
    y = _mm256_mul_ps(x, *(__m256 *)mag_m256_cephes_FOPI);
    imm2 = _mm256_cvttps_epi32(y);
    imm2 = _mm256_add_epi32(imm2, _mm256_set1_epi32(1));
    imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;
    imm0 = _mm256_and_si256(imm2, v4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    imm2 = _mm256_and_si256(imm2, v2);
    imm2 = _mm256_cmpeq_epi32(imm2, _mm256_setzero_si256());
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    x = _mm256_fmadd_ps(y, *(__m256 *)mag_m256_minus_cephes_DP1, x);
    x = _mm256_fmadd_ps(y, *(__m256 *)mag_m256_minus_cephes_DP2, x);
    x = _mm256_fmadd_ps(y, *(__m256 *)mag_m256_minus_cephes_DP3, x);
    imm4 = _mm256_sub_epi32(imm4, v2);
    imm4 = _mm256_andnot_si256(imm4, v4);
    imm4 = _mm256_slli_epi32(imm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
    __m256 z = _mm256_mul_ps(x,x);
    y = *(__m256 *)mag_m256_coscof_p0;
    y = _mm256_fmadd_ps(y, z, *(__m256 *)mag_m256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(__m256 *)mag_m256_coscof_p2);
    __m256 t = _mm256_mul_ps(y, _mm256_mul_ps(z, z));
    y = _mm256_fnmadd_ps(*(__m256 *)mag_m256_0p5, z, t);
    y = _mm256_add_ps(y, *(__m256 *)mag_m256_1);
    __m256 y2 = *(__m256 *)mag_m256_sincof_p0;
    y2 = _mm256_fmadd_ps(y2, z, *(__m256 *)mag_m256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(__m256 *)mag_m256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);
    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

#elif defined(__SSE2__)

static __m128 mag_simd_exp_e8m23(__m128 x) {
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

static __m128 mag_simd_tanh_e8m23(__m128 x) {
    __m128 one = _mm_set1_ps(1.f);
    __m128 neg_one = _mm_set1_ps(-1.f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 neg_two = _mm_set1_ps(-2.0f);
    __m128 a = _mm_mul_ps(neg_two, x);
    __m128 b = mag_simd_exp_e8m23(a);
    __m128 c = _mm_add_ps(one, b);
    __m128 inv = _mm_rcp_ps(c);
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm_add_ps(neg_one, _mm_mul_ps(two, inv));
}

#endif

static void MAG_HOTPROC mag_vcast_e8m23_e5m10(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e5m10_t *o = xo;
    const mag_e8m23_t *x = xx;
    int64_t i=0;
#ifdef __ARM_NEON
    for (; i+3 < numel; i += 4) {
        float32x4_t v = vld1q_f32(x+i);
        vst1_f16((__fp16 *)o+i, vcvt_f16_f32(v));
    }
#elif defined(__F16C__)
#ifdef __AVX512F__
    for (; i+15 < numel; i += 16) {
        __m512 xv = _mm512_loadu_ps(x+i);
        __m256i yv = _mm512_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(o+i), yv);
    }
#endif
    for (; i+7 < numel; i += 8) {
        __m256 xv = _mm256_loadu_ps(x+i);
        __m128i yv = _mm256_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(o+i), yv);
    }
    for (; i+3 < numel; i += 4) {
        __m128 xv = _mm_loadu_ps(x+i);
        __m128i yv = _mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(o+i), yv);
    }
#endif
    for (; i < numel; ++i) /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(x[i]);
}
static void MAG_HOTPROC mag_vcast_e5m10_e8m23(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e8m23_t *o = xo;
    const mag_e5m10_t *x = xx;
    int64_t i=0;
#ifdef __ARM_NEON
    for (; i+3 < numel; i += 4) {
        float16x4_t v = vld1_f16((const __fp16 *)x+i);
        vst1q_f32(o+i, vcvt_f32_f16(v));
    }
#elif defined(__F16C__)
#ifdef __AVX512F__
    for (; i+15 < numel; i += 16) {
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x+i));
        __m512 yv = _mm512_cvtph_ps(xv);
        _mm512_storeu_ps(o+i, yv);
    }
#endif
    for (; i+7 < numel; i += 8) {
        __m128i xv = _mm_loadu_si128((const __m128i *)(x+i));
        __m256 yv = _mm256_cvtph_ps(xv);
        _mm256_storeu_ps(o+i, yv);
    }
    for (; i+3 < numel; i += 4) {
        __m128i xv = _mm_loadl_epi64((const __m128i *)(x+i));
        __m128 yv = _mm_cvtph_ps(xv);
        _mm_storeu_ps(o+i, yv);
    }
#endif
    for (; i < numel; ++i) /* Scalar drain loop */
        o[i] = mag_e5m10_cvt_e8m23(x[i]);
}

static void MAG_HOTPROC mag_vcast_e8m23_i32(int64_t numel, void *restrict xo, const void *restrict xx) {
    int32_t *o = xo;
    const mag_e8m23_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (int32_t)x[i];
}
static void MAG_HOTPROC mag_vcast_i32_e8m23(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e8m23_t *o = xo;
    const int32_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (mag_e8m23_t)x[i];
}

static void MAG_HOTPROC mag_vcast_e8m23_bool(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_bool_t *o = xo;
    const mag_e8m23_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (mag_bool_t)(x[i] != .0f);
}
static void MAG_HOTPROC mag_vcast_bool_e8m23(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e8m23_t *o = xo;
    const mag_bool_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vcast_i32_bool(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_bool_t *o = xo;
    const int32_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (mag_bool_t)(x[i] != 0);
}
static void MAG_HOTPROC mag_vcast_bool_i32(int64_t numel, void *restrict xo, const void *restrict xx) {
    int32_t *o = xo;
    const mag_bool_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? 1 : 0;
}

static void MAG_HOTPROC mag_vcast_e5m10_i32(int64_t numel, void *restrict xo, const void *restrict xx) {
    int32_t *o = xo;
    const mag_e5m10_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (int32_t)mag_e5m10_cvt_e8m23(x[i]);
}
static void MAG_HOTPROC mag_vcast_i32_e5m10(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e5m10_t *o = xo;
    const int32_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10((mag_e8m23_t)x[i]);
}

static void MAG_HOTPROC mag_vcast_e5m10_bool(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_bool_t *o = xo;
    const mag_e5m10_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (mag_bool_t)(x[i].bits != 0);
}
static void MAG_HOTPROC mag_vcast_bool_e5m10(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_e5m10_t *o = xo;
    const mag_bool_t *x = xx;
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

#include "mag_cpu_impl_rand.inl"

static mag_e8m23_t MAG_HOTPROC mag_vdot_e8m23(int64_t numel, const mag_e8m23_t *restrict x, const mag_e8m23_t *restrict y) {
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
    mag_e8m23_t sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
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
    mag_e8m23_t sum = _mm512_reduce_add_ps(*acc);
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
    mag_e8m23_t sum = _mm_cvtss_f32(v0);
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
    mag_e8m23_t sum = _mm_cvtss_f32(*acc);
#else
    __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(*acc, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    mag_e8m23_t sum = _mm_cvtss_f32(sums);
#endif
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#else
    mag_e11m52_t r = 0.0;
    for (int64_t i=0; i < numel; ++i) r += (mag_e11m52_t)x[i]*(mag_e11m52_t)y[i];
    return (mag_e8m23_t)r;
#endif
}

static mag_e5m10_t MAG_HOTPROC mag_vdot_e5m10(int64_t numel, const mag_e5m10_t *restrict x, const mag_e5m10_t *restrict y) {
    mag_e8m23_t r = .0f;
    for (int64_t i=0; i < numel; ++i) /* TODO: Optimize with SIMD */
        r += mag_e5m10_cvt_e8m23(x[i])*mag_e5m10_cvt_e8m23(y[i]);
    return mag_e8m23_cvt_e5m10(r);
}

static void MAG_HOTPROC mag_vfill_e8m23(int64_t numel, mag_e8m23_t *o, mag_e8m23_t x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x;
}

static void MAG_HOTPROC mag_vacc_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] += x[i];
}

static void MAG_HOTPROC mag_vadd_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
#ifdef MAG_ACCELERATE
    vDSP_vadd(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y[i];
#endif
}

static void MAG_HOTPROC mag_vadd_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i+7 < numel; i += 8) {
        float16x8_t va = vld1q_f16((const __fp16 *)x+i);
        float16x8_t vb = vld1q_f16((const __fp16 *)y+i);
        float16x8_t r = vaddq_f16(va, vb);
        vst1q_f16((__fp16 *)o+i, r);
    }
    for (; i+3 < numel; i += 4) {
        float16x4_t va = vld1_f16((const __fp16 *)x+i);
        float16x4_t vb = vld1_f16((const __fp16 *)y+i);
        float16x4_t r = vadd_f16(va, vb);
        vst1_f16((__fp16 *)o+i, r);
    }
#else
    for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
        float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)x+i));
        float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)y+i));
        float32x4_t r = vaddq_f32(va_f32, vb_f32);
        vst1_f16((__fp16 *)o+i, vcvt_f16_f32(r));
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
        __m256i xph = _mm256_loadu_si256((const __m256i *)(x+i));
        __m256i yph = _mm256_loadu_si256((const __m256i *)(y+i));
        __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
        __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
        __m512 rps = _mm512_add_ps(xps, yps);
        _mm256_storeu_si256((__m256i *)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
    }
#elif defined(__AVX__) && defined(__F16C__)
    for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
        __m128i xph = _mm_loadu_si128((const __m128i *)(x+i));
        __m128i yph = _mm_loadu_si128((const __m128i *)(y+i));
        __m256 xps = _mm256_cvtph_ps(xph);
        __m256 yps = _mm256_cvtph_ps(yph);
        __m256 sum = _mm256_add_ps(xps, yps);
        _mm_storeu_si128((__m128i *)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vsub_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
#ifdef MAG_ACCELERATE
    vDSP_vsub(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y[i];
#endif
}

static void MAG_HOTPROC mag_vsub_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i+7 < numel; i += 8) {
        float16x8_t va = vld1q_f16((const __fp16 *)x+i);
        float16x8_t vb = vld1q_f16((const __fp16 *)y+i);
        float16x8_t r = vsubq_f16(va, vb);
        vst1q_f16((__fp16 *)o+i, r);
    }
    for (; i+3 < numel; i += 4) {
        float16x4_t va = vld1_f16((const __fp16 *)x+i);
        float16x4_t vb = vld1_f16((const __fp16 *)y+i);
        float16x4_t r = vsub_f16(va, vb);
        vst1_f16((__fp16 *)o+i, r);
    }
#else
    for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
        float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)x+i));
        float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)y+i));
        float32x4_t r = vsubq_f32(va_f32, vb_f32);
        vst1_f16((__fp16 *)o+i, vcvt_f16_f32(r));
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
        __m256i xph = _mm256_loadu_si256((const __m256i *)(x+i));
        __m256i yph = _mm256_loadu_si256((const __m256i *)(y+i));
        __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
        __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
        __m512 rps = _mm512_sub_ps(xps, yps);
        _mm256_storeu_si256((__m256i *)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
    }
#elif defined(__AVX__) && defined(__F16C__)
    for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
        __m128i xph = _mm_loadu_si128((const __m128i *)(x+i));
        __m128i yph = _mm_loadu_si128((const __m128i *)(y+i));
        __m256 xps = _mm256_cvtph_ps(xph);
        __m256 yps = _mm256_cvtph_ps(yph);
        __m256 sum = _mm256_sub_ps(xps, yps);
        _mm_storeu_si128((__m128i *)(o+i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vmul_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
#ifdef MAG_ACCELERATE
    vDSP_vmul(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y[i];
#endif
}

static void MAG_HOTPROC mag_vmul_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i+7 < numel; i += 8) {
        float16x8_t va = vld1q_f16((const __fp16 *)x+i);
        float16x8_t vb = vld1q_f16((const __fp16 *)y+i);
        float16x8_t r = vmulq_f16(va, vb);
        vst1q_f16((__fp16 *)o+i, r);
    }
    for (; i+3 < numel; i += 4) {
        float16x4_t va = vld1_f16((const __fp16 *)x+i);
        float16x4_t vb = vld1_f16((const __fp16 *)y+i);
        float16x4_t r = vmul_f16(va, vb);
        vst1_f16((__fp16 *)o+i, r);
    }
#else
    for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
        float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)x+i));
        float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)y+i));
        float32x4_t r = vmulq_f32(va_f32, vb_f32);
        vst1_f16((__fp16 *)o+i, vcvt_f16_f32(r));
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
        __m256i xph = _mm256_loadu_si256((const __m256i *)(x+i));
        __m256i yph = _mm256_loadu_si256((const __m256i *)(y+i));
        __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
        __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
        __m512 rps = _mm512_mul_ps(xps, yps);
        _mm256_storeu_si256((__m256i *)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
    }
#elif defined(__AVX__) && defined(__F16C__)
    for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
        __m128i xph = _mm_loadu_si128((const __m128i *)(x+i));
        __m128i yph = _mm_loadu_si128((const __m128i *)(y+i));
        __m256 xps = _mm256_cvtph_ps(xph);
        __m256 yps = _mm256_cvtph_ps(yph);
        __m256 sum = _mm256_mul_ps(xps, yps);
        _mm_storeu_si128((__m128i *)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i])*mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vdiv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
#ifdef MAG_ACCELERATE
    vDSP_vdiv(y, 1, x, 1, o, 1, numel);
#else
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y[i];
#endif
}

static void MAG_HOTPROC mag_vdiv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i+7 < numel; i += 8) {
        float16x8_t va = vld1q_f16((const __fp16 *)x+i);
        float16x8_t vb = vld1q_f16((const __fp16 *)y+i);
        float16x8_t r = vdivq_f16(va, vb);
        vst1q_f16((__fp16 *)o+i, r);
    }
    for (; i+3 < numel; i += 4) {
        float16x4_t va = vld1_f16((const __fp16 *)x+i);
        float16x4_t vb = vld1_f16((const __fp16 *)y+i);
        float16x4_t r = vdiv_f16(va, vb);
        vst1_f16((__fp16 *)o+i, r);
    }
#else
    for (; i+3 < numel; i += 4) { /* Load, downcast, compute, upcast, store. */
        float32x4_t va_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)x+i));
        float32x4_t vb_f32 = vcvt_f32_f16(vld1_f16((const __fp16 *)y+i));
        float32x4_t r = vdivq_f32(va_f32, vb_f32);
        vst1_f16((__fp16 *)o+i, vcvt_f16_f32(r));
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
        __m256i xph = _mm256_loadu_si256((const __m256i *)(x+i));
        __m256i yph = _mm256_loadu_si256((const __m256i *)(y+i));
        __m512 xps = _mm512_cvt_roundph_ps(xph, _MM_FROUND_CUR_DIRECTION);
        __m512 yps = _mm512_cvt_roundph_ps(yph, _MM_FROUND_CUR_DIRECTION);
        __m512 rps = _mm512_div_ps(xps, yps);
        _mm256_storeu_si256((__m256i *)(o+i), _mm512_cvtps_ph(rps, _MM_FROUND_CUR_DIRECTION));
    }
#elif defined(__AVX__) && defined(__F16C__)
    for (; i+7 < numel; i += 8) { /* Load, downcast, compute, upcast, store. */
        __m128i xph = _mm_loadu_si128((const __m128i *)(x+i));
        __m128i yph = _mm_loadu_si128((const __m128i *)(y+i));
        __m256 xps = _mm256_cvtph_ps(xph);
        __m256 yps = _mm256_cvtph_ps(yph);
        __m256 sum = _mm256_div_ps(xps, yps);
        _mm_storeu_si128((__m128i *)(o + i), _mm256_cvtps_ph(sum, _MM_FROUND_CUR_DIRECTION));
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / mag_e5m10_cvt_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vmod_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmodf(x[i], y[i]);
}

static void MAG_HOTPROC mag_vmod_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_cvt_e5m10(fmodf(mag_e5m10_cvt_e8m23(x[i]), mag_e5m10_cvt_e8m23(y[i])));
    }
}

static void MAG_HOTPROC mag_vpows_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = powf(x[i], y);
}

static void MAG_HOTPROC mag_vpows_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(powf(mag_e5m10_cvt_e8m23(x[i]), y));
}

static void MAG_HOTPROC mag_vadds_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y;
}

static void MAG_HOTPROC mag_vadds_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) + y);
}

static void MAG_HOTPROC mag_vsubs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y;
}

static void MAG_HOTPROC mag_vsubs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) - y);
}

static void MAG_HOTPROC mag_vmuls_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y;
}

static void MAG_HOTPROC mag_vmuls_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i])*y);
}

static void MAG_HOTPROC mag_vdivs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y;
}

static void MAG_HOTPROC mag_vdivs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(x[i]) / y);
}

static void MAG_HOTPROC mag_vabs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fabsf(x[i]);
}

static void MAG_HOTPROC mag_vabs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(fabsf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsgn_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi > 0.f ? 1.f : xi < 0.f ? -1.f : 0.f;
    }
}

static void MAG_HOTPROC mag_vsgn_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = xi > 0.f ? MAG_E5M10_ONE : xi < 0.f ? MAG_E5M10_NEG_ONE : MAG_E5M10_ZERO;
    }
}

static void MAG_HOTPROC mag_vneg_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = -x[i];
}

static void MAG_HOTPROC mag_vneg_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(-mag_e5m10_cvt_e8m23(x[i]));
}

static void MAG_HOTPROC mag_vlog_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = logf(x[i]);
}

static void MAG_HOTPROC mag_vlog_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(logf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog10_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log10f(x[i]);
}

static void MAG_HOTPROC mag_vlog10_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(log10f(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog1p_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log1pf(x[i]);
}

static void MAG_HOTPROC mag_vlog1p_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(log1pf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog2_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log2f(x[i]);
}

static void MAG_HOTPROC mag_vlog2_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(log2f(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsqr_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi*xi;
    }
}

static void MAG_HOTPROC mag_vsqr_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(xi*xi);
    }
}

static void MAG_HOTPROC mag_vsqrt_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sqrtf(x[i]);
}

static void MAG_HOTPROC mag_vsqrt_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(sqrtf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsin_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinf(x[i]);
}

static void MAG_HOTPROC mag_vsin_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(sinf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vcos_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = cosf(x[i]);
}

static void MAG_HOTPROC mag_vcos_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(cosf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtan_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanf(x[i]);
}

static void MAG_HOTPROC mag_vtan_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(tanf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vasin_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinf(x[i]);
}

static void MAG_HOTPROC mag_vasin_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(asinf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vacos_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acosf(x[i]);
}

static void MAG_HOTPROC mag_vacos_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(acosf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vatan_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanf(x[i]);
}

static void MAG_HOTPROC mag_vatan_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(atanf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsinh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinhf(x[i]);
}

static void MAG_HOTPROC mag_vsinh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(sinhf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vcosh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = coshf(x[i]);
}

static void MAG_HOTPROC mag_vcosh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(coshf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtanh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanhf(x[i]);
}

static void MAG_HOTPROC mag_vtanh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(tanhf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vasinh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinhf(x[i]);
}

static void MAG_HOTPROC mag_vasinh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(asinhf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vacosh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acoshf(x[i]);
}

static void MAG_HOTPROC mag_vacosh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(acoshf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vatanh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanhf(x[i]);
}

static void MAG_HOTPROC mag_vatanh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(atanhf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vstep_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.0f ? 1.f : 0.0f;
}

static void MAG_HOTPROC mag_vstep_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > 0.0f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_verf_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erff(x[i]);
}

static void MAG_HOTPROC mag_verf_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(erff(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_verfc_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erfcf(x[i]);
}

static void MAG_HOTPROC mag_verfc_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(erfcf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexp_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expf(x[i]);
}

static void MAG_HOTPROC mag_vexp_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(expf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexp2_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = exp2f(x[i]);
}

static void MAG_HOTPROC mag_vexp2_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(exp2f(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexpm1_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expm1f(x[i]);
}

static void MAG_HOTPROC mag_vexpm1_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(expm1f(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vfloor_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = floorf(x[i]);
}

static void MAG_HOTPROC mag_vfloor_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(floorf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vceil_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = ceilf(x[i]);
}

static void MAG_HOTPROC mag_vceil_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(ceilf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vround_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = nearbyintf(x[i]);
}

static void MAG_HOTPROC mag_vround_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(nearbyintf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtrunc_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = truncf(x[i]);
}

static void MAG_HOTPROC mag_vtrunc_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(truncf(mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsoftmax_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    mag_vexp_e8m23(numel, o, x);
}

static void MAG_HOTPROC mag_vsoftmax_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    mag_vexp_e5m10(numel, o, x);
}

static void MAG_HOTPROC mag_vsigmoid_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = 1.f/(1.f + expf(-x[i]));
}

static void MAG_HOTPROC mag_vsigmoid_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(1.f/(1.f + expf(-mag_e5m10_cvt_e8m23(x[i]))));
}

static void MAG_HOTPROC mag_vsigmoid_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t sig = 1.f/(1.f + expf(-x[i]));
        o[i] = sig*(1.f-sig);
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t sig = 1.f/(1.f + expf(-mag_e5m10_cvt_e8m23(x[i])));
        o[i] = mag_e8m23_cvt_e5m10(sig*(1.f-sig));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fminf(1.f, fmaxf(0.0f, (x[i] + 3.0f)/6.0f));
}

static void MAG_HOTPROC mag_vhard_sigmoid_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10( fminf(1.f, fmaxf(0.0f, (mag_e5m10_cvt_e8m23(x[i]) + 3.0f)/6.0f)));
}

static void MAG_HOTPROC mag_vsilu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi*(1.f/(1.f + expf(-xi)));
    }
}

static void MAG_HOTPROC mag_vsilu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(xi*(1.f/(1.f + expf(-xi))));
    }
}

static void MAG_HOTPROC mag_vsilu_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        mag_e8m23_t sig = 1.f/(1.f + expf(-xi));
        o[i] = sig + xi*sig;
    }
}

static void MAG_HOTPROC mag_vsilu_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        mag_e8m23_t sig = 1.f/(1.f + expf(-xi));
        o[i] = mag_e8m23_cvt_e5m10(sig + xi*sig);
    }
}

static void MAG_HOTPROC mag_vtanh_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t th = tanhf(x[i]);
        o[i] = 1.f - th*th;
    }
}

static void MAG_HOTPROC mag_vtanh_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t th = tanhf(mag_e5m10_cvt_e8m23(x[i]));
        o[i] = mag_e8m23_cvt_e5m10(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vrelu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmaxf(0.f, x[i]);
}

static void MAG_HOTPROC mag_vrelu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_cvt_e5m10(fmaxf(0.f, mag_e5m10_cvt_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vrelu_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.f ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vrelu_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > 0.f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_vgelu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = .5f*xi*(1.f+erff(xi*MAG_INVSQRT2));
    }
}

static void MAG_HOTPROC mag_vgelu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(.5f*xi*(1.f+erff(xi*MAG_INVSQRT2)));
    }
}

static void MAG_HOTPROC mag_vgelu_approx_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    int64_t i=0;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256 coeff = _mm256_set1_ps(MAG_INVSQRT2);
    __m256 coeff2 = _mm256_set1_ps(MAG_GELU_COEFF);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    for (; i+7 < numel; i += 8) {
        __m256 xi = _mm256_loadu_ps(x+i);
        __m256 xi3 = _mm256_mul_ps(xi, _mm256_mul_ps(xi, xi));
        __m256 tan1 = _mm256_add_ps(one, mag_simd_tanh_e8m23(_mm256_mul_ps(coeff, _mm256_add_ps(xi, _mm256_mul_ps(coeff2, xi3)))));
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(xi, tan1), half);
        _mm256_storeu_ps(o+i, r);
    }
#endif
    for (; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = 0.5f*xi*(1.0f+tanhf(MAG_INVSQRT2*(xi+MAG_GELU_COEFF*xi*xi*xi)));
    }
}

static void MAG_HOTPROC mag_vgelu_approx_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        o[i] = mag_e8m23_cvt_e5m10(0.5f*xi*(1.0f+tanhf(MAG_INVSQRT2*(xi+MAG_GELU_COEFF*xi*xi*xi))));
    }
}


static void MAG_HOTPROC mag_vgelu_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        mag_e8m23_t th = tanhf(xi);
        o[i] = .5f*(1.f + th) + .5f*xi*(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vgelu_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_cvt_e8m23(x[i]);
        mag_e8m23_t th = tanhf(xi);
        o[i] = mag_e8m23_cvt_e5m10(.5f*(1.f + th) + .5f*xi*(1.f - th*th));
    }
}

static mag_e11m52_t MAG_HOTPROC mag_vsum_f64_e8m23(int64_t numel, const mag_e8m23_t *x) {
#ifdef MAG_ACCELERATE
    mag_e8m23_t sum;
    vDSP_sve(x, 1, &sum, numel);
    return (mag_e11m52_t)sum;
#else
    mag_e11m52_t sum = 0.0;
    for (int64_t i=0; i < numel; ++i)
        sum += (mag_e11m52_t)x[i];
    return sum;
#endif
}

static mag_e11m52_t MAG_HOTPROC mag_vsum_f64_e5m10(int64_t numel, const mag_e5m10_t *x) {
    mag_e11m52_t sum = 0.0;
    for (int64_t i=0; i < numel; ++i)
        sum += mag_e5m10_cvt_e8m23(x[i]);
    return sum;
}

static mag_e8m23_t MAG_HOTPROC mag_vmin_e8m23(int64_t numel, const mag_e8m23_t *x) {
    mag_e8m23_t min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, x[i]);
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmin_e5m10(int64_t numel, const mag_e5m10_t *x) {
    mag_e8m23_t min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, mag_e5m10_cvt_e8m23(x[i]));
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmax_e8m23(int64_t numel, const mag_e8m23_t *x) {
    mag_e8m23_t min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, x[i]);
    return min;
}

static mag_e8m23_t MAG_HOTPROC mag_vmax_e5m10(int64_t numel, const mag_e5m10_t *x) {
    mag_e8m23_t min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, mag_e5m10_cvt_e8m23(x[i]));
    return min;
}

#define mag_impl_vecop_int(T) \
    static void mag_vadd_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]+y[i]; \
    } \
    static void mag_vsub_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]-y[i]; \
    } \
    static void mag_vmul_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]*y[i]; \
    } \
    static void mag_vdiv_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]/y[i]; \
    } \
    static void mag_vmod_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]%y[i]; \
    } \
    static void mag_vand_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]&y[i]; \
    } \
    static void mag_vor_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]|y[i]; \
    } \
    static void mag_vxor_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]^y[i]; \
    } \
    static void mag_vshl_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]<<(y[i]&((sizeof(mag_##T##_t)<<3)-1)); \
    } \
    static void mag_vshr_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]>>(y[i]&((sizeof(mag_##T##_t)<<3)-1)); \
    } \
    static void mag_vnot_##T(int64_t numel, mag_##T##_t *o, const mag_##T##_t *x) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = ~x[i]; \
    } \
    static void MAG_HOTPROC mag_veq_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]==y[i]; \
    } \
    static void MAG_HOTPROC mag_vne_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]!=y[i]; \
    } \
    static void MAG_HOTPROC mag_vlt_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]<y[i]; \
    } \
    static void MAG_HOTPROC mag_vgt_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]>y[i]; \
    } \
    static void MAG_HOTPROC mag_vle_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]<=y[i]; \
    } \
    static void MAG_HOTPROC mag_vge_##T(int64_t numel, mag_bool_t *o, const mag_##T##_t *x, const mag_##T##_t *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]>=y[i]; \
    }

mag_impl_vecop_int(bool)
mag_impl_vecop_int(u8)
mag_impl_vecop_int(i8)
mag_impl_vecop_int(u16)
mag_impl_vecop_int(i16)
mag_impl_vecop_int(u32)
mag_impl_vecop_int(i32)
mag_impl_vecop_int(u64)
mag_impl_vecop_int(i64)

#undef mag_impl_vecop_int

static void MAG_HOTPROC mag_veq_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] == y[i];
    }
}

static void MAG_HOTPROC mag_veq_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits == y[i].bits;
}

static void MAG_HOTPROC mag_vne_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] != y[i];
}

static void MAG_HOTPROC mag_vne_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits != y[i].bits;
}

static void MAG_HOTPROC mag_vle_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] <= y[i];
}

static void MAG_HOTPROC mag_vle_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) <= mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vge_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >= y[i];
}

static void MAG_HOTPROC mag_vge_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) >= mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vlt_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] < y[i];
}

static void MAG_HOTPROC mag_vlt_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) < mag_e5m10_cvt_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vgt_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > y[i];
}

static void MAG_HOTPROC mag_vgt_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_cvt_e8m23(x[i]) > mag_e5m10_cvt_e8m23(y[i]);
}

static void mag_nop(const mag_kernel_payload_t *payload) {
    (void)payload;
}

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

#define mag_gen_stub_fill(T, G, UT, CVT) \
    static MAG_HOTPROC void mag_fill_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_in(0); \
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

#define mag_gen_stub_masked_fill(T, G, UT, CVT) \
    static MAG_HOTPROC void mag_masked_fill_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_in(0); \
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

#define mag_gen_stub_fill_rand(D, T, TS, UT) \
    static MAG_HOTPROC void mag_fill_rand_##D##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_in(0); \
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

#define mag_gen_stub_fill_arange(T, CVT) \
    static MAG_HOTPROC void mag_fill_arange_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_in(0); \
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

typedef struct mag_discrete_sample_pair_t {
    mag_e8m23_t score;
    int64_t idx;
} mag_discrete_sample_pair_t;

static int mag_discrete_sample_pair_cmp(const void *a, const void *b) {
    const mag_discrete_sample_pair_t *A = a;
    const mag_discrete_sample_pair_t *B = b;
    return A->score < B->score ? 1 : A->score > B->score ? -1 : 0;
}

#define mag_gen_stub_multinomial(T, CVT) \
    static void MAG_HOTPROC mag_multinomial_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_assert2(r->dtype == MAG_DTYPE_I64); \
        mag_i64_t *br = mag_i64p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        int64_t num_samples = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        mag_philox4x32_stream_t *rng = payload->prng; \
        int64_t K = x->coords.shape[x->coords.rank-1]; \
        if (mag_unlikely(K <= 0)) return; \
        int64_t B = x->numel / K; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (B + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, B); \
        for (int64_t b=ra; b < rb; ++b) { \
            const mag_##T##_t *w = bx + b*K; \
            mag_i64_t *o = br + b*num_samples; \
            mag_e8m23_t sumw = .0f; \
            int64_t nnz = 0; \
            for (int64_t i=0; i < K; ++i) { \
                mag_e8m23_t wi = CVT(w[i]); \
                if (!isfinite(wi) || wi <= .0f) wi = .0f; \
                sumw += wi; \
                if (wi > .0f) ++nnz; \
            } \
            if (!(sumw > .0f) || nnz == 0) { \
                for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
                continue; \
            } \
            int64_t k = num_samples; \
            if (k > nnz) k = nnz; \
            if (mag_unlikely(k <= 0)) { \
                for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
                continue; \
            } \
            mag_discrete_sample_pair_t *arr = (mag_discrete_sample_pair_t*)alloca(nnz*sizeof(*arr)); \
            int64_t m=0; \
            for (int64_t i=0; i < K; ++i) { \
                mag_e8m23_t wi = CVT(w[i]); \
                if (mag_unlikely(!isfinite(wi) || wi <= .0f)) continue; \
                mag_e8m23_t u = mag_philox4x32_next_e8m23(rng); \
                mag_e8m23_t g = -logf(-logf(u)); \
                arr[m].score = logf(wi) + g; \
                arr[m].idx = i; \
                ++m; \
            } \
            qsort(arr, (size_t)m, sizeof(*arr), mag_discrete_sample_pair_cmp); \
            for (int64_t s=0; s < k; ++s) o[s] = arr[s].idx; \
            for (int64_t s=k; s < num_samples; ++s) o[s] = -1; \
        } \
    }

#define mag_gen_stub_cat(T) \
    static MAG_HOTPROC void mag_cat_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const int64_t dim = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        const int64_t n = payload->cmd->num_in; \
        mag_assert2(r && n > 0); \
        mag_assert2(dim >= 0 && dim < r->coords.rank); \
        \
        int64_t R = r->coords.rank; \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        mag_assert2(mag_tensor_is_contiguous(r)); \
        \
        int64_t inner_block = 1; \
        for (int64_t d = dim+1; d < R; ++d) inner_block *= r->coords.shape[d]; \
        int64_t outer_count = 1; \
        for (int64_t d=0; d < dim; ++d) outer_count *= r->coords.shape[d]; \
        \
        int64_t mult[MAG_MAX_DIMS]; \
        for (int64_t d = 0; d < dim; ++d) { \
            int64_t m = 1; \
            for (int64_t k = d + 1; k < dim; ++k) m *= r->coords.shape[k]; \
            mult[d] = m; \
        } \
        \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (outer_count + tc - 1)/tc; \
        int64_t oa = ti*chunk; \
        int64_t ob = mag_xmin(oa + chunk, outer_count); \
        \
        for (int64_t p=oa; p < ob; ++p) { \
            int64_t idx_prefix[MAG_MAX_DIMS]; \
            int64_t rtmp = p; \
            for (int64_t d = 0; d < dim; ++d) { \
                int64_t q = !mult[d] ? 0 : rtmp/mult[d]; \
                if (mult[d] != 0) rtmp = rtmp%mult[d]; \
                idx_prefix[d] = q; \
            } \
            \
            int64_t moff = 0; \
            for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->coords.strides[d]; \
            int64_t cur = 0; \
            \
            for (int64_t i=0; i < n; ++i) { \
                const mag_tensor_t *x = mag_cmd_in(i); \
                int64_t smoff=0; \
                for (int64_t d=0; d < dim; ++d) \
                    smoff += idx_prefix[d]*x->coords.strides[d]; \
                int64_t cl = x->coords.shape[dim]; \
                int64_t numel = cl*inner_block; \
                int64_t oel = moff + cur*r->coords.strides[dim]; \
                int64_t sel = smoff; \
                const mag_##T##_t *restrict bx = mag_##T##p(x); \
                const uint8_t *restrict src_ptr = (const uint8_t*)(bx+sel); \
                uint8_t *restrict dst_ptr = (uint8_t*)(br+oel); \
                mag_bnd_chk(bx + sel, bx, mag_tensor_get_data_size(x)); \
                mag_bnd_chk(br + oel, br, mag_tensor_get_data_size(r)); \
                memcpy(dst_ptr, src_ptr, (size_t)numel*sizeof(mag_##T##_t)); \
                cur += cl; \
            } \
        } \
    }

#define mag_gen_stub_tri_mask(T, S, Z, CMP) \
    static void MAG_HOTPROC mag_tri##S##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        int64_t diag = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        int64_t total = r->numel; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        int64_t cols = r->coords.shape[r->coords.rank-1]; \
        int64_t rows = r->coords.shape[r->coords.rank-2]; \
        int64_t mat = rows*cols; \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t inner = i % mat; \
            int64_t row = inner / cols; \
            int64_t col = inner - row*cols; \
            int64_t ri, xi; \
            mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = ((col-row) CMP diag) ? bx[xi] : (Z); \
        }  \
    }

#define mag_gen_stub_binop(T, FUNC, OP, CVT, RCVT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        const mag_##T##_t *by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##T(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = RCVT(CVT(bx[xi]) OP CVT(by[yi])); \
        } \
    }


static int mag_cmp_i64(const void *a, const void *b) {
    int64_t da = *(const int64_t *)a, db = *(const int64_t *)b;
    return (da > db) - (da < db);
}

#define mag_cpu_impl_reduce_axes(T, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        int64_t nd = x->coords.rank; \
        int64_t rank = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        int64_t axes[MAG_MAX_DIMS]; \
        for (int64_t i=0; i<rank; ++i){ \
            int64_t a=mag_op_attr_unwrap_i64(mag_cmd_attr(2+i)); \
            if (a<0) a += nd; \
            axes[i]=a; \
        } \
        qsort(axes, (size_t)rank, sizeof(int64_t), &mag_cmp_i64); \
        int64_t rr=0; for(int64_t i=0;i<rank;++i) if(i==0||axes[i]!=axes[i-1]) axes[rr++]=axes[i]; \
        rank=rr; \
        int64_t keep_axes[MAG_MAX_DIMS]; \
        int64_t nk=0; \
        for (int64_t d=0; d < nd; ++d) { \
            bool red = false; \
            for (int64_t k=0; k < rank; ++k) \
                if (axes[k]==d) { red = true; break; } \
            if (!red) keep_axes[nk++] = d; \
        } \
        int64_t out_numel = r->numel; \
        int64_t red_prod = 1; (void)red_prod; \
        for (int64_t k=0; k < rank; ++k) \
            red_prod *= x->coords.shape[axes[k]]; \
        for (int64_t oi=0; oi<out_numel; ++oi) { \
            int64_t rem = oi, off = 0; \
            for (int64_t j=nk-1; j >= 0; --j) { \
                int64_t ax = keep_axes[j]; \
                int64_t sz = x->coords.shape[ax]; \
                int64_t idx = (sz > 1) ? rem % sz : 0; \
                if (sz > 1) rem /= sz; \
                off += idx*x->coords.strides[ax]; \
            } \
            ACC_T acc = (INIT_EXPR); \
            int64_t ctr[MAG_MAX_DIMS] = {0}; \
            int64_t cur = off; \
            for (;;) { \
                int64_t roff = cur; \
                mag_bnd_chk(bx+roff, bx, mag_tensor_get_data_size(x)); \
                { UPDATE_STMT } \
                int64_t k=0; \
                for (; k < rank; ++k) { \
                    int64_t ax = axes[k]; \
                    if (++ctr[k] < x->coords.shape[ax]) { cur += x->coords.strides[ax]; break; } \
                    cur -= x->coords.strides[ax]*(x->coords.shape[ax]-1); \
                    ctr[k]=0; \
                } \
                if (k == rank) break; \
            } \
            mag_##T##_t *outp = br + oi; \
            { FINAL_STMT } \
        } \
    }


#define mag_gen_stub_repeat_back(T, Z, CVT, RCVT) \
    static void MAG_HOTPROC mag_repeat_back_##T(const mag_kernel_payload_t *payload) { \
        if (payload->thread_idx != 0) return; \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        mag_coords_iter_t cr, cx; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        for (int64_t i=0; i < r->numel; ++i) { \
            int64_t ri = mag_coords_iter_to_offset(&cr, i); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = (Z); \
        } \
        for (int64_t i=0; i < x->numel; ++i) { \
            int64_t xi = mag_coords_iter_to_offset(&cx, i); \
            int64_t ri = mag_coords_iter_repeat(&cr, &cx, i); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = RCVT(CVT(br[ri]) + CVT(bx[xi])); \
        } \
    }

#define mag_gen_stub_cmp(FUNC, T, OP, CVT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        mag_bool_t *br = mag_boolp_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        const mag_##T##_t *by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##T(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = CVT(bx[xi]) OP CVT(by[yi]); \
        } \
    }

#define mag_gen_stub_gather(T) \
    static MAG_HOTPROC void mag_gather_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *src = mag_cmd_in(0); \
        const mag_tensor_t *index = mag_cmd_in(1); \
        mag_assert2(index->dtype == MAG_DTYPE_I64); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(src); \
        const mag_i64_t *bi = mag_i64p(index); \
        int64_t axis = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        if (axis < 0) axis += src->coords.rank; \
        mag_assert2(axis >= 0 && axis < src->coords.rank); \
        mag_assert2(index->coords.rank >= 1); \
        int64_t ax = src->coords.shape[axis]; \
        int64_t on = r->numel; \
        int64_t oc[MAG_MAX_DIMS]; \
        int64_t sc[MAG_MAX_DIMS]; \
        bool full = true; \
        for (int64_t d = 0; d < src->coords.rank; ++d) { \
            if (d == axis) continue; \
            if (index->coords.shape[d] != src->coords.shape[d]) { \
                full = false; \
                break; \
            } \
        } \
        mag_coords_iter_t ci; \
        mag_coords_iter_init(&ci, &index->coords); \
        for (int64_t flat=0; flat < on; ++flat) { \
            int64_t tmp = flat; \
            for (int64_t d=r->coords.rank-1; d >= 0; --d) { \
                oc[d] = tmp % r->coords.shape[d]; \
                tmp /= r->coords.shape[d]; \
            } \
            int64_t gather_idx; \
            if (full) { \
                int64_t index_offset = mag_coords_iter_to_offset(&ci, flat); \
                gather_idx = bi[index_offset]; \
            } else if (index->coords.rank == 1) { \
                int64_t idx_pos = oc[axis]; \
                mag_assert2(idx_pos >= 0 && idx_pos < index->coords.shape[0]); \
                int64_t index_offset = idx_pos*index->coords.strides[0]; \
                gather_idx = bi[index_offset]; \
            } else { \
                int64_t idx_coords[MAG_MAX_DIMS]; \
                for (int64_t i=0; i < index->coords.rank; ++i) idx_coords[i] = oc[axis+i]; \
                int64_t index_offset = 0; \
                for (int64_t d=0; d < index->coords.rank; ++d) index_offset += idx_coords[d]*index->coords.strides[d]; \
                gather_idx = bi[index_offset]; \
            } \
            if (gather_idx < 0) gather_idx += ax; \
            mag_assert2(gather_idx >= 0 && gather_idx < ax); \
            if (full) { \
                for (int64_t d=0; d < src->coords.rank; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
            } else if (index->coords.rank == 1) { \
                for (int64_t d=0; d < src->coords.rank; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
            } else { \
                for (int64_t d=0; d < axis; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
                for (int64_t d=axis+1; d < src->coords.rank; ++d) sc[d] = oc[index->coords.rank+d-1]; \
            } \
            int64_t src_offset = 0, dest_offset = 0; \
            for (int64_t d=0; d < src->coords.rank; ++d) src_offset += sc[d]*src->coords.strides[d]; \
            for (int64_t d=0; d < r->coords.rank; ++d) dest_offset += oc[d]*r->coords.strides[d]; \
            mag_bnd_chk(bx + src_offset, bx, mag_tensor_get_data_size(src)); \
            mag_bnd_chk(br + dest_offset, br, mag_tensor_get_data_size(r)); \
            br[dest_offset] = bx[src_offset]; \
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

#define mag_G(x) (x)                    /* Get scalar value */
#define mag_G_underlying(x) (x.bits)    /* Get underlying storage scalar */

mag_gen_stub_fill(e8m23, mag_G, e8m23, mag_cvt_nop)
mag_gen_stub_fill(e5m10, mag_G_underlying, e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_fill(bool, mag_G, i64, mag_cvt_i642bool)
mag_gen_stub_fill(u8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(i32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_fill(u64, mag_G, i64, mag_cvt_nop)
mag_gen_stub_fill(i64, mag_G, i64, mag_cvt_nop)

mag_gen_stub_masked_fill(e8m23, mag_G, e8m23, mag_cvt_nop)
mag_gen_stub_masked_fill(e5m10, mag_G_underlying, e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_masked_fill(bool, mag_G, i64, mag_cvt_i642bool)
mag_gen_stub_masked_fill(u8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i8, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i16, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(i32, mag_G, i64, mag_cvt_i642i32)
mag_gen_stub_masked_fill(u64, mag_G, i64, mag_cvt_nop)
mag_gen_stub_masked_fill(i64, mag_G, i64, mag_cvt_nop)

mag_gen_stub_fill_rand(uniform, e8m23, e8m23, e8m23)
mag_gen_stub_fill_rand(uniform, e5m10, e8m23, e8m23)
mag_gen_stub_fill_rand(uniform, i32, i32, i64)
mag_gen_stub_fill_rand(normal, e8m23, e8m23, e8m23)
mag_gen_stub_fill_rand(normal, e5m10, e8m23, e8m23)

mag_gen_stub_fill_arange(e8m23, mag_cvt_nop)
mag_gen_stub_fill_arange(e5m10, mag_e8m23_cvt_e5m10)
mag_gen_stub_fill_arange(u8, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i8, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u16, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i16, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u32, mag_cvt_i642i32)
mag_gen_stub_fill_arange(i32, mag_cvt_i642i32)
mag_gen_stub_fill_arange(u64, mag_cvt_nop)
mag_gen_stub_fill_arange(i64, mag_cvt_nop)

static MAG_HOTPROC void mag_fill_rand_bernoulli_bool(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_in(0);
    mag_e8m23_t p = mag_op_attr_unwrap_e8m23(mag_cmd_attr(0));
    mag_bool_t *b_r = mag_boolp_mut(r);
    int64_t numel = r->numel;
    mag_vrand_bernoulli_bool(payload->prng, numel, b_r, p);
}

#undef mag_G
#undef mag_G_underlying

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
        mag_e8m23_t max_val = mag_e5m10_cvt_e8m23(row_in[0]);  /* Max val is computed for numerical stability */
        for (int64_t i=1; i < last_dim; ++i) {
            mag_e8m23_t fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
            if (fp32_row > max_val) {
                mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
                max_val = fp32_row;
            }
        }
        mag_e8m23_t sum = 0.0f;
        for (int64_t i=0; i < last_dim; ++i) {
            mag_bnd_chk(row_in+i, bx, mag_tensor_get_data_size(x));
            mag_bnd_chk(row_out+i, br, mag_tensor_get_data_size(r));
            mag_e8m23_t fp32_row = mag_e5m10_cvt_e8m23(row_in[i]);
            mag_e8m23_t exp = expf(fp32_row - max_val);
            row_out[i] = mag_e8m23_cvt_e5m10(exp); /* -max for numerical stability */
            sum += exp;
        }
        for (int64_t i=0; i < last_dim; ++i) {
            row_out[i] = mag_e8m23_cvt_e5m10(mag_e5m10_cvt_e8m23(row_out[i]) / sum);
        }
    }
}

mag_gen_stub_multinomial(e8m23, mag_cvt_nop)
mag_gen_stub_multinomial(e5m10, mag_e5m10_cvt_e8m23)

mag_gen_stub_cat(e8m23)
mag_gen_stub_cat(e5m10)
mag_gen_stub_cat(bool)
mag_gen_stub_cat(u8)
mag_gen_stub_cat(i8)
mag_gen_stub_cat(u16)
mag_gen_stub_cat(i16)
mag_gen_stub_cat(u32)
mag_gen_stub_cat(i32)
mag_gen_stub_cat(u64)
mag_gen_stub_cat(i64)

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

mag_gen_stub_tri_mask(e8m23, l, 0.f, <=)
mag_gen_stub_tri_mask(e5m10, l, MAG_E5M10_ZERO, <=)
mag_gen_stub_tri_mask(bool, l, 0, <=)
mag_gen_stub_tri_mask(u8, l, 0, <=)
mag_gen_stub_tri_mask(i8, l, 0, <=)
mag_gen_stub_tri_mask(u16, l, 0, <=)
mag_gen_stub_tri_mask(i16, l, 0, <=)
mag_gen_stub_tri_mask(u32, l, 0, <=)
mag_gen_stub_tri_mask(i32, l, 0, <=)
mag_gen_stub_tri_mask(u64, l, 0, <=)
mag_gen_stub_tri_mask(i64, l, 0, <=)
mag_gen_stub_tri_mask(e8m23, u, 0.f, >=)
mag_gen_stub_tri_mask(e5m10, u, MAG_E5M10_ZERO, >=)
mag_gen_stub_tri_mask(bool, u, 0, >=)
mag_gen_stub_tri_mask(u8, u, 0, >=)
mag_gen_stub_tri_mask(i8, u, 0, >=)
mag_gen_stub_tri_mask(u16, u, 0, >=)
mag_gen_stub_tri_mask(i16, u, 0, >=)
mag_gen_stub_tri_mask(u32, u, 0, >=)
mag_gen_stub_tri_mask(i32, u, 0, >=)
mag_gen_stub_tri_mask(u64, u, 0, >=)
mag_gen_stub_tri_mask(i64, u, 0, >=)

#define mag_cvt_nop(x) (x)

mag_gen_stub_binop(e8m23, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, add, +, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, sub, -, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, mul, *, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, div, /, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, div, /, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(u8, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, shr, >>, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(bool, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(bool, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(bool, xor, ^, mag_cvt_nop, mag_cvt_nop)

mag_cpu_impl_reduce_axes( \
                          e8m23, sum, mag_e11m52_t, 0.0, \
                          acc += (mag_e11m52_t)bx[roff];, \
                          *outp = (mag_e8m23_t)acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, sum, mag_e8m23_t, 0.0f, \
                          acc += mag_e5m10_cvt_e8m23(bx[roff]);, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, mean, mag_e11m52_t, 0.0, \
                          acc += (mag_e11m52_t)bx[roff];, \
                          acc /= (mag_e11m52_t)red_prod; *outp = (mag_e8m23_t)acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, mean, mag_e8m23_t, 0.0f, \
                          acc += mag_e5m10_cvt_e8m23(bx[roff]);, \
                          acc /= (mag_e8m23_t)red_prod; *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, mag_e5m10_cvt_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, mag_e5m10_cvt_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

mag_gen_stub_repeat_back(e8m23, .0f, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_repeat_back(e5m10, MAG_E5M10_ZERO, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)

mag_gen_stub_gather(bool)
mag_gen_stub_gather(e8m23)
mag_gen_stub_gather(e5m10)
mag_gen_stub_gather(u8)
mag_gen_stub_gather(i8)
mag_gen_stub_gather(u16)
mag_gen_stub_gather(i16)
mag_gen_stub_gather(u32)
mag_gen_stub_gather(i32)
mag_gen_stub_gather(u64)
mag_gen_stub_gather(i64)

mag_gen_stub_cmp(eq, e8m23, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, e5m10, ==, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(eq, u8,  ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i8,  ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u64, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i64, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, bool, ==, mag_cvt_nop)
mag_gen_stub_cmp(ne, e8m23, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, e5m10, !=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(ne, u8,  !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i8,  !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u64, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i64, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, bool, !=, mag_cvt_nop)

mag_gen_stub_cmp(lt, e8m23, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, e5m10, <, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(lt, u8,  <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i8,  <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u64, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i64, <, mag_cvt_nop)
mag_gen_stub_cmp(gt, e8m23, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, e5m10, >, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(gt, u8,  >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i8,  >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u64, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i64, >, mag_cvt_nop)
mag_gen_stub_cmp(le, e8m23, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, e5m10, <=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(le, u8,  <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i8,  <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u64, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i64, <=, mag_cvt_nop)
mag_gen_stub_cmp(ge, e8m23, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, e5m10, >=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(ge, u8,  >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i8,  >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u64, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i64, >=, mag_cvt_nop)

static int64_t mag_offset_rmn(const mag_tensor_t *t, int64_t flat, int64_t i, int64_t j) {
    int64_t ra = t->coords.rank;
    const int64_t *restrict td = t->coords.shape;
    const int64_t *restrict ts = t->coords.strides;
    if (mag_likely(ra <= 3)) { /* Fast path */
        switch (ra) {
        case 1:
            return i*ts[0];
        case 2:
            return i*ts[0] + j*ts[1];
        case 3:
            return flat*ts[0] + i*ts[1] + j*ts[2];
        default:
            mag_panic("invalid rank: %" PRIi64, ra);
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

static MAG_HOTPROC mag_e5m10_t *mag_mm_pack_x_e5m10(mag_e5m10_t *xbuf, int64_t M, int64_t K, int64_t xb, const mag_tensor_t *x, const mag_e5m10_t *px) {
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

static MAG_HOTPROC mag_e5m10_t *mag_mm_pack_y_e5m10(mag_e5m10_t *ybuf, int64_t K, int64_t N, int64_t yb, const mag_tensor_t *y, const mag_e5m10_t *py) {
    if (y->coords.rank == 1) {
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

#define MAG_PREFETCH_SPAN 8
#define MAG_PF_GROUP 8
#define MAG_PFDIST_B_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PFDIST_B_L2 (MAG_PREFETCH_SPAN*12)
#define MAG_PFDIST_A_L1 (MAG_PREFETCH_SPAN*2)
#define MAG_PFDIST_A_L2 (MAG_PREFETCH_SPAN*10)

#if defined(__GNUC__) || defined(__clang__)
#define mag_prefetchw(addr) __builtin_prefetch((addr), 1, 3)
#else
#define mag_prefetchw(addr) ((void)0)
#endif

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#ifdef __ARM_FEATURE_FMA
#define mag_vfmadd_e8m23(acc, a, b) vfmaq_f32((acc), (a), (b))
#else
#define mag_vfmadd_e8m23(acc, a, b) vmlaq_f32((acc), (a), (b))
#endif
#define mag_prefetcht0(p) __builtin_prefetch((const char*)(p), 0, 3)
#define mag_prefetcht1(p) __builtin_prefetch((const char*)(p), 0, 2)
#else
#define mag_prefetcht0(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#define mag_prefetcht1(p) _mm_prefetch((const char*)(p), _MM_HINT_T1)
#endif

static MAG_AINLINE void mag_mm_tile_8x8_e8m23(int64_t kc, const mag_e8m23_t *restrict a, ptrdiff_t lda, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, ptrdiff_t ldc, bool acc) {
#ifdef __AVX512F__
    __m512 C01, C23, C45, C67;
    if (acc) {
        __m256 c0 = _mm256_loadu_ps(c + 0*ldc);
        __m256 c1 = _mm256_loadu_ps(c + 1*ldc);
        __m256 c2 = _mm256_loadu_ps(c + 2*ldc);
        __m256 c3 = _mm256_loadu_ps(c + 3*ldc);
        __m256 c4 = _mm256_loadu_ps(c + 4*ldc);
        __m256 c5 = _mm256_loadu_ps(c + 5*ldc);
        __m256 c6 = _mm256_loadu_ps(c + 6*ldc);
        __m256 c7 = _mm256_loadu_ps(c + 7*ldc);
        C01 = _mm512_insertf32x8(_mm512_castps256_ps512(c0), c1, 1);
        C23 = _mm512_insertf32x8(_mm512_castps256_ps512(c2), c3, 1);
        C45 = _mm512_insertf32x8(_mm512_castps256_ps512(c4), c5, 1);
        C67 = _mm512_insertf32x8(_mm512_castps256_ps512(c6), c7, 1);
    } else {
        __m512 z = _mm512_setzero_ps();
        C01 = z;
        C23 = z;
        C45 = z;
        C67 = z;
    }
    __m512 P01e = _mm512_setzero_ps();
    __m512 P23e = _mm512_setzero_ps();
    __m512 P45e = _mm512_setzero_ps();
    __m512 P67e = _mm512_setzero_ps();
    __m512 P01o = _mm512_setzero_ps();
    __m512 P23o = _mm512_setzero_ps();
    __m512 P45o = _mm512_setzero_ps();
    __m512 P67o = _mm512_setzero_ps();
#define mag_plat_idx_pair(lo,hi) _mm512_set_epi32((hi),(hi),(hi),(hi),(hi),(hi),(hi),(hi),(lo),(lo),(lo),(lo),(lo),(lo),(lo),(lo))
    __m512i i01 = mag_plat_idx_pair(0,1);
    __m512i i23 = mag_plat_idx_pair(2,3);
    __m512i i45 = mag_plat_idx_pair(4,5);
    __m512i i67 = mag_plat_idx_pair(6,7);
#undef mag_plat_idx_pair
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + ((k + MAG_PFDIST_A_L1)<<3));
            mag_prefetcht1(a + ((k + MAG_PFDIST_A_L2)<<3));
        }
        __m256 b0_256 = _mm256_loadu_ps(b + (k + 0)  *ldb);
#ifdef __AVX512DQ__
        __m512 b0 = _mm512_broadcast_f32x8(b0_256);
#else
        __m512 b0 = _mm512_insertf32x8(_mm512_castps256_ps512(b0_256), b0_256, 1);
#endif
        __m256 av0 = _mm256_loadu_ps(a + k*8);
        __m512 Adup0 = _mm512_broadcast_f32x8(av0);
        __m512 A01_0 = _mm512_permutexvar_ps(i01, Adup0);
        __m512 A23_0 = _mm512_permutexvar_ps(i23, Adup0);
        __m512 A45_0 = _mm512_permutexvar_ps(i45, Adup0);
        __m512 A67_0 = _mm512_permutexvar_ps(i67, Adup0);
        P01e = _mm512_fmadd_ps(A01_0, b0, P01e);
        P23e = _mm512_fmadd_ps(A23_0, b0, P23e);
        P45e = _mm512_fmadd_ps(A45_0, b0, P45e);
        P67e = _mm512_fmadd_ps(A67_0, b0, P67e);
        __m256 b1_256 = _mm256_loadu_ps(b + (k + 1)*ldb);
#ifdef __AVX512DQ__
        __m512 b1 = _mm512_broadcast_f32x8(b1_256);
#else
        __m512 b1 = _mm512_insertf32x8(_mm512_castps256_ps512(b1_256), b1_256, 1);
#endif
        __m256 av1 = _mm256_loadu_ps(a + (k + 1)*8);
        __m512 Adup1 = _mm512_broadcast_f32x8(av1);
        __m512 A01_1 = _mm512_permutexvar_ps(i01, Adup1);
        __m512 A23_1 = _mm512_permutexvar_ps(i23, Adup1);
        __m512 A45_1 = _mm512_permutexvar_ps(i45, Adup1);
        __m512 A67_1 = _mm512_permutexvar_ps(i67, Adup1);
        P01o = _mm512_fmadd_ps(A01_1, b1, P01o);
        P23o = _mm512_fmadd_ps(A23_1, b1, P23o);
        P45o = _mm512_fmadd_ps(A45_1, b1, P45o);
        P67o = _mm512_fmadd_ps(A67_1, b1, P67o);
        __m256 b2_256 = _mm256_loadu_ps(b + (k + 2)*ldb);
#ifdef __AVX512DQ__
        __m512 b2 = _mm512_broadcast_f32x8(b2_256);
#else
        __m512 b2 = _mm512_insertf32x8(_mm512_castps256_ps512(b2_256), b2_256, 1);
#endif
        __m256 av2 = _mm256_loadu_ps(a + (k + 2)*8);
        __m512 Adup2 = _mm512_broadcast_f32x8(av2);
        __m512 A01_2 = _mm512_permutexvar_ps(i01, Adup2);
        __m512 A23_2 = _mm512_permutexvar_ps(i23, Adup2);
        __m512 A45_2 = _mm512_permutexvar_ps(i45, Adup2);
        __m512 A67_2 = _mm512_permutexvar_ps(i67, Adup2);
        P01e = _mm512_fmadd_ps(A01_2, b2, P01e);
        P23e = _mm512_fmadd_ps(A23_2, b2, P23e);
        P45e = _mm512_fmadd_ps(A45_2, b2, P45e);
        P67e = _mm512_fmadd_ps(A67_2, b2, P67e);
        __m256 b3_256 = _mm256_loadu_ps(b + (k + 3)*ldb);
#ifdef __AVX512DQ__
        __m512 b3 = _mm512_broadcast_f32x8(b3_256);
#else
        __m512 b3 = _mm512_insertf32x8(_mm512_castps256_ps512(b3_256), b3_256, 1);
#endif
        __m256 av3 = _mm256_loadu_ps(a + (k + 3)*8);
        __m512 Adup3 = _mm512_broadcast_f32x8(av3);
        __m512 A01_3 = _mm512_permutexvar_ps(i01, Adup3);
        __m512 A23_3 = _mm512_permutexvar_ps(i23, Adup3);
        __m512 A45_3 = _mm512_permutexvar_ps(i45, Adup3);
        __m512 A67_3 = _mm512_permutexvar_ps(i67, Adup3);
        P01o = _mm512_fmadd_ps(A01_3, b3, P01o);
        P23o = _mm512_fmadd_ps(A23_3, b3, P23o);
        P45o = _mm512_fmadd_ps(A45_3, b3, P45o);
        P67o = _mm512_fmadd_ps(A67_3, b3, P67o);
    }
    for (; k < kc; ++k) {
        __m256 bk_256 = _mm256_loadu_ps(b + k*ldb);
#ifdef __AVX512DQ__
        __m512 bk = _mm512_broadcast_f32x8(bk_256);
#else
        __m512 bk = _mm512_insertf32x8(_mm512_castps256_ps512(bk_256), bk_256, 1);
#endif
        __m256 av = _mm256_loadu_ps(a + k  *8);
        __m512 Adup = _mm512_broadcast_f32x8(av);
        __m512 A01 = _mm512_permutexvar_ps(i01, Adup);
        __m512 A23 = _mm512_permutexvar_ps(i23, Adup);
        __m512 A45 = _mm512_permutexvar_ps(i45, Adup);
        __m512 A67 = _mm512_permutexvar_ps(i67, Adup);
        if (k & 1) {
            P01o = _mm512_fmadd_ps(A01, bk, P01o);
            P23o = _mm512_fmadd_ps(A23, bk, P23o);
            P45o = _mm512_fmadd_ps(A45, bk, P45o);
            P67o = _mm512_fmadd_ps(A67, bk, P67o);
        } else {
            P01e = _mm512_fmadd_ps(A01, bk, P01e);
            P23e = _mm512_fmadd_ps(A23, bk, P23e);
            P45e = _mm512_fmadd_ps(A45, bk, P45e);
            P67e = _mm512_fmadd_ps(A67, bk, P67e);
        }
    }
    C01 = _mm512_add_ps(C01, _mm512_add_ps(P01e, P01o));
    C23 = _mm512_add_ps(C23, _mm512_add_ps(P23e, P23o));
    C45 = _mm512_add_ps(C45, _mm512_add_ps(P45e, P45o));
    C67 = _mm512_add_ps(C67, _mm512_add_ps(P67e, P67o));
    _mm256_storeu_ps(c + 0*ldc, _mm512_extractf32x8_ps(C01, 0));
    _mm256_storeu_ps(c + 1*ldc, _mm512_extractf32x8_ps(C01, 1));
    _mm256_storeu_ps(c + 2*ldc, _mm512_extractf32x8_ps(C23, 0));
    _mm256_storeu_ps(c + 3*ldc, _mm512_extractf32x8_ps(C23, 1));
    _mm256_storeu_ps(c + 4*ldc, _mm512_extractf32x8_ps(C45, 0));
    _mm256_storeu_ps(c + 5*ldc, _mm512_extractf32x8_ps(C45, 1));
    _mm256_storeu_ps(c + 6*ldc, _mm512_extractf32x8_ps(C67, 0));
    _mm256_storeu_ps(c + 7*ldc, _mm512_extractf32x8_ps(C67, 1));
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 C[8];
    if (acc) {
#pragma GCC unroll 8
        for (int r = 0; r < 8; ++r)
            C[r] = _mm256_loadu_ps(c + r*ldc);
    } else {
        __m256 z = _mm256_setzero_ps();
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r)
            C[r] = z;
    }
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if ((k & (MAG_PF_GROUP - 1)) == 0) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (int64_t)((k + MAG_PFDIST_A_L1)<<3));
            mag_prefetcht1(a + (int64_t)((k + MAG_PFDIST_A_L2)<<3));
        }
        __m256 B0 = _mm256_loadu_ps(b + (k + 0)*ldb);
        __m256 B1 = _mm256_loadu_ps(b + (k + 1)*ldb);
        __m256 B2 = _mm256_loadu_ps(b + (k + 2)*ldb);
        __m256 B3 = _mm256_loadu_ps(b + (k + 3)*ldb);
        const mag_e8m23_t *a0 = a + (k + 0)*8;
        const mag_e8m23_t *a1 = a + (k + 1)*8;
        const mag_e8m23_t *a2 = a + (k + 2)*8;
        const mag_e8m23_t *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m256 A;
            A = _mm256_broadcast_ss(a0 + r);
            C[r] = _mm256_fmadd_ps(A, B0, C[r]);
            A = _mm256_broadcast_ss(a1 + r);
            C[r] = _mm256_fmadd_ps(A, B1, C[r]);
            A = _mm256_broadcast_ss(a2 + r);
            C[r] = _mm256_fmadd_ps(A, B2, C[r]);
            A = _mm256_broadcast_ss(a3 + r);
            C[r] = _mm256_fmadd_ps(A, B3, C[r]);
        }
    }
    for (; k < kc; ++k) {
        __m256 Bk = _mm256_loadu_ps(b + k*ldb);
        const mag_e8m23_t *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m256 A = _mm256_broadcast_ss(ak + r);
            C[r] = _mm256_fmadd_ps(A, Bk, C[r]);
        }
    }
#pragma GCC unroll 8
    for (int r=0; r < 8; ++r)
        _mm256_storeu_ps(c + r*ldc, C[r]);
#elif defined(__SSE2__)
#define mm_fmadd_ps(a,b,c) _mm_add_ps((c), _mm_mul_ps((a),(b)))
    __m128 C[8][2];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            C[r][0] = _mm_loadu_ps(c + r*ldc + 0);
            C[r][1] = _mm_loadu_ps(c + r*ldc + 4);
        }
    } else {
        __m128 z = _mm_setzero_ps();
#pragma GCC unroll 8
        for (int r = 0; r < 8; ++r) C[r][0] = C[r][1] = z;
    }
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if ((k & (MAG_PF_GROUP - 1)) == 0) {
            _mm_prefetch((const char *)(b + (k + MAG_PFDIST_B_L1)*ldb), _MM_HINT_T0);
            _mm_prefetch((const char *)(b + (k + MAG_PFDIST_B_L2)*ldb), _MM_HINT_T1);
            _mm_prefetch((const char *)(a + (int64_t)((k + MAG_PFDIST_A_L1)*8)), _MM_HINT_T0);
            _mm_prefetch((const char *)(a + (int64_t)((k + MAG_PFDIST_A_L2)*8)), _MM_HINT_T1);
        }
        __m128 B0_0 = _mm_loadu_ps(b + (k + 0)*ldb + 0);
        __m128 B0_1 = _mm_loadu_ps(b + (k + 0)*ldb + 4);
        __m128 B1_0 = _mm_loadu_ps(b + (k + 1)*ldb + 0);
        __m128 B1_1 = _mm_loadu_ps(b + (k + 1)*ldb + 4);
        __m128 B2_0 = _mm_loadu_ps(b + (k + 2)*ldb + 0);
        __m128 B2_1 = _mm_loadu_ps(b + (k + 2)*ldb + 4);
        __m128 B3_0 = _mm_loadu_ps(b + (k + 3)*ldb + 0);
        __m128 B3_1 = _mm_loadu_ps(b + (k + 3)*ldb + 4);
        const mag_e8m23_t *a0 = a + (k + 0)*8;
        const mag_e8m23_t *a1 = a + (k + 1)*8;
        const mag_e8m23_t *a2 = a + (k + 2)*8;
        const mag_e8m23_t *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m128 A;
            A = _mm_set1_ps(a0[r]);
            C[r][0] = mm_fmadd_ps(A, B0_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B0_1, C[r][1]);
            A = _mm_set1_ps(a1[r]);
            C[r][0] = mm_fmadd_ps(A, B1_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B1_1, C[r][1]);
            A = _mm_set1_ps(a2[r]);
            C[r][0] = mm_fmadd_ps(A, B2_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B2_1, C[r][1]);
            A = _mm_set1_ps(a3[r]);
            C[r][0] = mm_fmadd_ps(A, B3_0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B3_1, C[r][1]);
        }
    }
    for (; k < kc; ++k) {
        __m128 B0 = _mm_loadu_ps(b + k*ldb + 0);
        __m128 B1 = _mm_loadu_ps(b + k*ldb + 4);
        const mag_e8m23_t *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            __m128 A = _mm_set1_ps(ak[r]);
            C[r][0] = mm_fmadd_ps(A, B0, C[r][0]);
            C[r][1] = mm_fmadd_ps(A, B1, C[r][1]);
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        _mm_storeu_ps(c + r*ldc + 0, C[r][0]);
        _mm_storeu_ps(c + r*ldc + 4, C[r][1]);
    }
#undef mm_fmadd_ps
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C[8][2];
    if (acc) {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            C[r][0] = vld1q_f32(c + r*ldc + 0);
            C[r][1] = vld1q_f32(c + r*ldc + 4);
        }
    } else {
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r)
            C[r][0] = C[r][1] = vdupq_n_f32(0.f);
    }
    int64_t k=0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            __builtin_prefetch(b + (k + MAG_PFDIST_B_L1)*ldb, 0, 3);
            __builtin_prefetch(b + (k + MAG_PFDIST_B_L2)*ldb, 0, 2);
            __builtin_prefetch(a + (int64_t)((k + MAG_PFDIST_A_L1)<<3), 0, 3);
            __builtin_prefetch(a + (int64_t)((k + MAG_PFDIST_A_L2)<<3), 0, 2);
        }
        float32x4_t B0_0 = vld1q_f32(b + (k + 0)*ldb + 0);
        float32x4_t B0_1 = vld1q_f32(b + (k + 0)*ldb + 4);
        float32x4_t B1_0 = vld1q_f32(b + (k + 1)*ldb + 0);
        float32x4_t B1_1 = vld1q_f32(b + (k + 1)*ldb + 4);
        float32x4_t B2_0 = vld1q_f32(b + (k + 2)*ldb + 0);
        float32x4_t B2_1 = vld1q_f32(b + (k + 2)*ldb + 4);
        float32x4_t B3_0 = vld1q_f32(b + (k + 3)*ldb + 0);
        float32x4_t B3_1 = vld1q_f32(b + (k + 3)*ldb + 4);
        const mag_e8m23_t *a0 = a + (k + 0)*8;
        const mag_e8m23_t *a1 = a + (k + 1)*8;
        const mag_e8m23_t *a2 = a + (k + 2)*8;
        const mag_e8m23_t *a3 = a + (k + 3)*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            float32x4_t A;
            A = vdupq_n_f32(a0[r]);
            C[r][0] = vfmaq_f32(C[r][0], B0_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B0_1, A);
            A = vdupq_n_f32(a1[r]);
            C[r][0] = vfmaq_f32(C[r][0], B1_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B1_1, A);
            A = vdupq_n_f32(a2[r]);
            C[r][0] = vfmaq_f32(C[r][0], B2_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B2_1, A);
            A = vdupq_n_f32(a3[r]);
            C[r][0] = vfmaq_f32(C[r][0], B3_0, A);
            C[r][1] = vfmaq_f32(C[r][1], B3_1, A);
        }
    }
    for (; k < kc; ++k) {
        float32x4_t B0 = vld1q_f32(b + k*ldb + 0);
        float32x4_t B1 = vld1q_f32(b + k*ldb + 4);
        const mag_e8m23_t *ak = a + k*8;
#pragma GCC unroll 8
        for (int r=0; r < 8; ++r) {
            float32x4_t A = vdupq_n_f32(ak[r]);
            C[r][0] = vfmaq_f32(C[r][0], B0, A);
            C[r][1] = vfmaq_f32(C[r][1], B1, A);
        }
    }
#pragma GCC unroll 8
    for (int r=0; r < 8; ++r) {
        vst1q_f32(c + r*ldc + 0, C[r][0]);
        vst1q_f32(c + r*ldc + 4, C[r][1]);
    }
#else
#error "Unsupported architecture"
#endif
}

static MAG_AINLINE void mag_mm_tile_8x16_e8m23(int64_t kc, const mag_e8m23_t *restrict a, ptrdiff_t lda, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x8_e8m23(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x8_e8m23(kc, a, lda, b+8, ldb, c+8, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x32_e8m23(int64_t kc, const mag_e8m23_t *restrict a, ptrdiff_t lda, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x16_e8m23(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x16_e8m23(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_e8m23(int64_t kc, const mag_e8m23_t *restrict a, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, bool acc) {
#ifdef __AVX512F__
    __mmask16 m8 = 0x00ff;
    __m512 C = acc ? _mm512_maskz_loadu_ps(m8, c) : _mm512_setzero_ps();
    __m512 P0 = _mm512_setzero_ps();
    __m512 P1 = _mm512_setzero_ps();
    __m512 P2 = _mm512_setzero_ps();
    __m512 P3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP-1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (k + MAG_PFDIST_A_L1));
            mag_prefetcht1(a + (k + MAG_PFDIST_A_L2));
        }
        __m512 a0 = _mm512_set1_ps(a[k + 0]);
        __m512 a1 = _mm512_set1_ps(a[k + 1]);
        __m512 a2 = _mm512_set1_ps(a[k + 2]);
        __m512 a3 = _mm512_set1_ps(a[k + 3]);
        __m512 b0 = _mm512_maskz_loadu_ps(m8, b + (k + 0)*ldb);
        __m512 b1 = _mm512_maskz_loadu_ps(m8, b + (k + 1)*ldb);
        __m512 b2 = _mm512_maskz_loadu_ps(m8, b + (k + 2)*ldb);
        __m512 b3 = _mm512_maskz_loadu_ps(m8, b + (k + 3)*ldb);
        P0 = _mm512_fmadd_ps(a0, b0, P0);
        P1 = _mm512_fmadd_ps(a1, b1, P1);
        P2 = _mm512_fmadd_ps(a2, b2, P2);
        P3 = _mm512_fmadd_ps(a3, b3, P3);
    }
    C = _mm512_add_ps(C, _mm512_add_ps(_mm512_add_ps(P0, P1), _mm512_add_ps(P2, P3)));
    for (; k < kc; ++k) {
        __m512 ak = _mm512_set1_ps(a[k]);
        __m512 bk = _mm512_maskz_loadu_ps(m8, b + k  *ldb);
        C = _mm512_fmadd_ps(ak, bk, C);
    }
    _mm512_mask_storeu_ps(c, m8, C);
#elif defined(__AVX__) && defined(__FMA__)
    __m256 C0 = acc ? _mm256_loadu_ps(c) : _mm256_setzero_ps();
    for (int64_t k=0; k < kc; ++k) {
        __m256 A = _mm256_broadcast_ss(a + k);
        __m256 B0 = _mm256_loadu_ps(b + k*ldb + 0);
        C0 = _mm256_fmadd_ps(A, B0, C0);
    }
    _mm256_storeu_ps(c, C0);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C0 = acc ? vld1q_f32(c + 0) : vdupq_n_f32(0.0f);
    float32x4_t C1 = acc ? vld1q_f32(c + 4) : vdupq_n_f32(0.0f);
    for (int64_t k = 0; k < kc; ++k) {
        float32x4_t A = vdupq_n_f32(a[k]);
        float32x4_t B0 = vld1q_f32(b + k*ldb + 0);
        float32x4_t B1 = vld1q_f32(b + k*ldb + 4);
        C0 = mag_vfmadd_e8m23(C0, A, B0);
        C1 = mag_vfmadd_e8m23(C1, A, B1);
    }
    vst1q_f32(c + 0, C0);
    vst1q_f32(c + 4, C1);
#else
#pragma GCC unroll 8
    for (int64_t j=0; j < 8; ++j)
        c[j] = acc ? c[j] : 0.f;
    for (int64_t k=0; k < kc; ++k) {
        mag_e8m23_t a0 = a[k];
#pragma GCC unroll 8
        for (int64_t j=0; j < 8; ++j)
            c[j] += a0*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x16_e8m23(int64_t kc, const mag_e8m23_t *restrict a, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, bool acc) {
#ifdef __AVX512F__
    __m512 C = acc ? _mm512_loadu_ps(c) : _mm512_setzero_ps();
    __m512 P0 = _mm512_setzero_ps();
    __m512 P1 = _mm512_setzero_ps();
    __m512 P2 = _mm512_setzero_ps();
    __m512 P3 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k+3 < kc; k += 4) {
        if (!(k & (MAG_PF_GROUP - 1))) {
            mag_prefetcht0(b + (k + MAG_PFDIST_B_L1)*ldb);
            mag_prefetcht1(b + (k + MAG_PFDIST_B_L2)*ldb);
            mag_prefetcht0(a + (k + MAG_PFDIST_A_L1));
            mag_prefetcht1(a + (k + MAG_PFDIST_A_L2));
        }
        __m512 a0 = _mm512_set1_ps(a[k + 0]);
        __m512 a1 = _mm512_set1_ps(a[k + 1]);
        __m512 a2 = _mm512_set1_ps(a[k + 2]);
        __m512 a3 = _mm512_set1_ps(a[k + 3]);
        const mag_e8m23_t *B0p = b + (k + 0)*ldb;
        const mag_e8m23_t *B1p = b + (k + 1)*ldb;
        const mag_e8m23_t *B2p = b + (k + 2)*ldb;
        const mag_e8m23_t *B3p = b + (k + 3)*ldb;
        __m512 B0 = _mm512_loadu_ps(B0p);
        __m512 B1 = _mm512_loadu_ps(B1p);
        __m512 B2 = _mm512_loadu_ps(B2p);
        __m512 B3 = _mm512_loadu_ps(B3p);
        P0 = _mm512_fmadd_ps(a0, B0, P0);
        P1 = _mm512_fmadd_ps(a1, B1, P1);
        P2 = _mm512_fmadd_ps(a2, B2, P2);
        P3 = _mm512_fmadd_ps(a3, B3, P3);
    }
    C = _mm512_add_ps(C, _mm512_add_ps(_mm512_add_ps(P0, P1), _mm512_add_ps(P2, P3)));
    for (; k < kc; ++k) {
        __m512 ak = _mm512_set1_ps(a[k]);
        __m512 bk = _mm512_loadu_ps(b + k  *ldb);
        C = _mm512_fmadd_ps(ak, bk, C);
    }
    _mm512_storeu_ps(c, C);
#elif defined(__AVX__) && defined(__FMA__)
    __m256 C0 = acc ? _mm256_loadu_ps(c) : _mm256_setzero_ps();
    __m256 C1 = acc ? _mm256_loadu_ps(c+8) : _mm256_setzero_ps();
    for (int64_t k=0; k < kc; ++k) {
        __m256 A = _mm256_broadcast_ss(a + k);
        __m256 B0 = _mm256_loadu_ps(b + k*ldb + 0);
        __m256 B1 = _mm256_loadu_ps(b + k*ldb + 8);
        C0 = _mm256_fmadd_ps(A, B0, C0);
        C1 = _mm256_fmadd_ps(A, B1, C1);
    }
    _mm256_storeu_ps(c + 0, C0);
    _mm256_storeu_ps(c + 8, C1);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t C0 = acc ? vld1q_f32(c + 0) : vdupq_n_f32(0.0f);
    float32x4_t C1 = acc ? vld1q_f32(c + 4) : vdupq_n_f32(0.0f);
    float32x4_t C2 = acc ? vld1q_f32(c + 8) : vdupq_n_f32(0.0f);
    float32x4_t C3 = acc ? vld1q_f32(c + 12) : vdupq_n_f32(0.0f);
    for (int64_t k=0; k < kc; ++k) {
        float32x4_t A = vdupq_n_f32(a[k]);
        const mag_e8m23_t *Bk = b + k*ldb;
        C0 = mag_vfmadd_e8m23(C0, A, vld1q_f32(Bk + 0));
        C1 = mag_vfmadd_e8m23(C1, A, vld1q_f32(Bk + 4));
        C2 = mag_vfmadd_e8m23(C2, A, vld1q_f32(Bk + 8));
        C3 = mag_vfmadd_e8m23(C3, A, vld1q_f32(Bk + 12));
    }
    vst1q_f32(c + 0, C0);
    vst1q_f32(c + 4, C1);
    vst1q_f32(c + 8, C2);
    vst1q_f32(c + 12, C3);
#else
#pragma GCC unroll 16
    for (int64_t j=0; j < 16; ++j)
        c[j] = acc ? c[j] : 0.f;
    for (int64_t k=0; k < kc; ++k) {
        mag_e8m23_t a0 = a[k];
#pragma GCC unroll 16
        for (int64_t j=0; j < 16; ++j)
            c[j] += a0*b[k*ldb + j];
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_1x32_e8m23(int64_t kc, const mag_e8m23_t *restrict a, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c,  bool acc) {
    mag_mm_tile_1x16_e8m23(kc, a, b, ldb, c, acc);
    mag_mm_tile_1x16_e8m23(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_e8m23(int64_t kc, int64_t nc, const mag_e8m23_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN, mag_e8m23_t *restrict Bp) {
    if (strideN == 1) {
        for (int64_t k=0; k < kc; ++k) {
            const mag_e8m23_t *src = Bsrc + k*strideK;
#ifdef __AVX512F__
            int64_t j=0;
            mag_e8m23_t *dst = Bp + k*nc;
            for (; j+63 < nc; j += 64) {
                mag_prefetcht0(src + j + 256);
                mag_prefetcht1(src + j + 1024);
                __m512 v0 = _mm512_loadu_ps(src + j + 0);
                __m512 v1 = _mm512_loadu_ps(src + j + 16);
                __m512 v2 = _mm512_loadu_ps(src + j + 32);
                __m512 v3 = _mm512_loadu_ps(src + j + 48);
                _mm512_storeu_ps(dst + j + 0, v0);
                _mm512_storeu_ps(dst + j + 16, v1);
                _mm512_storeu_ps(dst + j + 32, v2);
                _mm512_storeu_ps(dst + j + 48, v3);
            }
            for (; j+31 < nc; j += 32) {
                __m512 v0 = _mm512_loadu_ps(src + j +  0);
                __m512 v1 = _mm512_loadu_ps(src + j + 16);
                _mm512_storeu_ps(dst + j +  0, v0);
                _mm512_storeu_ps(dst + j + 16, v1);
            }
            for (; j+15 < nc; j += 16) {
                __m512 v = _mm512_loadu_ps(src + j);
                _mm512_storeu_ps(dst + j, v);
            }
            if (j < nc) {
                int64_t rem = nc - j;
                __mmask16 m = rem == 16 ? 0xffff : (__mmask16)((1u<<rem)-1);
                __m512 v = _mm512_maskz_loadu_ps(m, src + j);
                _mm512_mask_storeu_ps(dst + j, m, v);
            }
#elif defined(__AVX__)
            int64_t j=0;
            for (; j+31 < nc; j += 32) {
                mag_prefetcht0(src + j + 128);
                mag_prefetcht1(src + j + 512);
                __m256 v0 = _mm256_loadu_ps(src + j + 0);
                __m256 v1 = _mm256_loadu_ps(src + j + 8);
                __m256 v2 = _mm256_loadu_ps(src + j + 16);
                __m256 v3 = _mm256_loadu_ps(src + j + 24);
                _mm256_storeu_ps(Bp + k*nc + j + 0, v0);
                _mm256_storeu_ps(Bp + k*nc + j + 8, v1);
                _mm256_storeu_ps(Bp + k*nc + j + 16, v2);
                _mm256_storeu_ps(Bp + k*nc + j + 24, v3);
            }
            for (; j+15 < nc; j += 16) {
                __m256 v0 = _mm256_loadu_ps(src + j + 0);
                __m256 v1 = _mm256_loadu_ps(src + j + 8);
                _mm256_storeu_ps(Bp + k*nc + j + 0, v0);
                _mm256_storeu_ps(Bp + k*nc + j + 8, v1);
            }
            for (; j+7 < nc; j += 8) {
                __m256 v = _mm256_loadu_ps(src + j);
                _mm256_storeu_ps(Bp + k*nc + j, v);
            }
            for (; j+3 < nc; j += 4) {
                __m128 v = _mm_loadu_ps(src + j);
                _mm_storeu_ps(Bp + k*nc + j, v);
            }
            for (; j < nc; ++j) Bp[k*nc + j] = src[j];
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            int64_t j = 0;
            for (; j+15 < nc; j += 16) {
                vst1q_f32(Bp + k*nc + j + 0, vld1q_f32(src + j + 0));
                vst1q_f32(Bp + k*nc + j + 4, vld1q_f32(src + j + 4));
                vst1q_f32(Bp + k*nc + j + 8, vld1q_f32(src + j + 8));
                vst1q_f32(Bp + k*nc + j + 12, vld1q_f32(src + j + 12));
            }
            for (; j+3 < nc; j += 4)
                vst1q_f32(Bp + k*nc + j, vld1q_f32(src + j));
            for (; j < nc; ++j)
                Bp[k*nc + j] = src[j];
#else
            memcpy(Bp + k*nc, src, nc*sizeof(*Bsrc));
#endif
        }
    } else {
        for (int64_t k=0; k < kc; ++k) {
            const mag_e8m23_t *src = Bsrc + k*strideK;
            for (int64_t j=0; j < nc; ++j)
                Bp[k*nc + j] = src[j*strideN];
        }
    }
}

static MAG_AINLINE void mag_mm_pack_A_mr8_kc_e8m23(int64_t kc, const mag_e8m23_t *restrict Asrc, ptrdiff_t strideK, mag_e8m23_t *restrict Ap) {
    if (strideK == 1) {
#ifdef __AVX512F__
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const mag_e8m23_t *src = Asrc + i*kc;
            mag_e8m23_t *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+63 < kc; k += 64) {
                mag_prefetcht0(src + k + 256);
                mag_prefetcht1(src + k + 1024);
                __m512 v0 = _mm512_loadu_ps(src + k + 0);
                __m512 v1 = _mm512_loadu_ps(src + k + 16);
                __m512 v2 = _mm512_loadu_ps(src + k + 32);
                __m512 v3 = _mm512_loadu_ps(src + k + 48);
                _mm512_storeu_ps(dst + k + 0, v0);
                _mm512_storeu_ps(dst + k + 16, v1);
                _mm512_storeu_ps(dst + k + 32, v2);
                _mm512_storeu_ps(dst + k + 48, v3);
            }
            for (; k+31 < kc; k += 32) {
                __m512 v0 = _mm512_loadu_ps(src + k + 0);
                __m512 v1 = _mm512_loadu_ps(src + k + 16);
                _mm512_storeu_ps(dst + k + 0, v0);
                _mm512_storeu_ps(dst + k + 16, v1);
            }
            for (; k+15 < kc; k += 16) {
                __m512 v = _mm512_loadu_ps(src + k);
                _mm512_storeu_ps(dst + k, v);
            }
            if (k < kc) {
                int64_t rem = kc - k;
                __mmask16 m = (__mmask16)((1u<<rem)-1);
                __m512 v = _mm512_maskz_loadu_ps(m, src + k);
                _mm512_mask_storeu_ps(dst + k, m, v);
            }
        }
#elif defined(__AVX2__)
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const mag_e8m23_t *src = Asrc + i*kc;
            mag_e8m23_t *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+7 < kc; k += 8) {
                __m256 v = _mm256_loadu_ps(src + k);
                _mm256_storeu_ps(dst + k, v);
            }
            for (; k+3 < kc; k += 4) {
                __m128 v = _mm_loadu_ps(src + k);
                _mm_storeu_ps(dst + k, v);
            }
            for (; k < kc; ++k) dst[k] = src[k];
        }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const mag_e8m23_t *src = Asrc + i*kc;
            mag_e8m23_t *dst = Ap + i*kc;
            int64_t k=0;
            for (; k+15 < kc; k += 16) {
                vst1q_f32(dst + k + 0, vld1q_f32(src + k + 0));
                vst1q_f32(dst + k + 4, vld1q_f32(src + k + 4));
                vst1q_f32(dst + k + 8, vld1q_f32(src + k + 8));
                vst1q_f32(dst + k + 12, vld1q_f32(src + k + 12));
            }
            for (; k+3 < kc; k += 4)
                vst1q_f32(dst + k, vld1q_f32(src + k));
            for (; k < kc; ++k)
                dst[k] = src[k];
        }
#else
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i)
            memcpy(Ap + i*kc, Asrc + i*kc, kc*sizeof(*Asrc));
#endif
    } else {
#pragma GCC unroll 8
        for (int i=0; i < 8; ++i) {
            const mag_e8m23_t *src = Asrc + i*strideK*kc; /* start of row i */
            for (int64_t k = 0; k < kc; ++k)
                Ap[i*kc + k] = src[k*strideK];
        }
    }
}

static MAG_AINLINE void mag_mm_pack_B_vec_e8m23(int64_t kc, int64_t nc, const mag_e8m23_t *restrict yvec, mag_e8m23_t *restrict Bp) {
#ifdef __AVX512F__
    for (int64_t k=0; k < kc; ++k) {
        __m512 val = _mm512_set1_ps(yvec[k]);
        mag_e8m23_t *dst = Bp + k*nc;
        int64_t j=0;
        for (; j+63 < nc; j += 64) {
            _mm512_storeu_ps(dst + j + 0, val);
            _mm512_storeu_ps(dst + j + 16, val);
            _mm512_storeu_ps(dst + j + 32, val);
            _mm512_storeu_ps(dst + j + 48, val);
        }
        for (; j+31 < nc; j += 32) {
            _mm512_storeu_ps(dst + j + 0, val);
            _mm512_storeu_ps(dst + j + 16, val);
        }
        for (; j+15 < nc; j += 16) {
            _mm512_storeu_ps(dst + j, val);
        }
        if (j < nc) {
            int64_t rem = nc - j;
            __mmask16 m = (__mmask16)((1u<<rem)-1);
            _mm512_mask_storeu_ps(dst + j, m, val);
        }
    }
#elif defined(__AVX2__)
    for (int64_t k=0; k < kc; ++k) {
        __m256 val = _mm256_broadcast_ss(yvec + k);
        int64_t j = 0;
        for (; j+31 < nc; j += 32) {
            _mm256_storeu_ps(Bp + k*nc + j + 0, val);
            _mm256_storeu_ps(Bp + k*nc + j + 8, val);
            _mm256_storeu_ps(Bp + k*nc + j + 16, val);
            _mm256_storeu_ps(Bp + k*nc + j + 24, val);
        }
        for (; j+15 < nc; j += 16) {
            _mm256_storeu_ps(Bp + k*nc + j, val);
            _mm256_storeu_ps(Bp + k*nc + j + 8, val);
        }
        for (; j+7 < nc; j += 8)
            _mm256_storeu_ps(Bp + k*nc + j, val);
        for (; j+3 < nc; j += 4)
            _mm_storeu_ps(Bp + k*nc + j, _mm256_castps256_ps128(val));
        for (; j < nc; ++j)
            Bp[k*nc + j] = yvec[k];
    }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (int64_t k=0; k < kc; ++k) {
        float32x4_t val = vdupq_n_f32(yvec[k]);
        int64_t j=0;
        for (; j+15 < nc; j += 16) {
            vst1q_f32(Bp + k*nc + j + 0, val);
            vst1q_f32(Bp + k*nc + j + 4, val);
            vst1q_f32(Bp + k*nc + j + 8, val);
            vst1q_f32(Bp + k*nc + j + 12, val);
        }
        for (; j+3 < nc; j += 4)
            vst1q_f32(Bp + k*nc + j, val);
        for (; j < nc; ++j)
            Bp[k*nc + j] = yvec[k];
    }
#else
    for (int64_t k = 0; k < kc; ++k) {
        mag_e8m23_t v = yvec[k];
        for (int64_t j=0; j < nc; ++j)
            Bp[k*nc + j] = v;
    }
#endif
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_e8m23(int64_t kc, int64_t mr, const mag_e8m23_t *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx, mag_e8m23_t *restrict pa) {
    int64_t m8 = mr&~7;
    for (int64_t i=0; i < m8; i += 8) {
        const mag_e8m23_t *p0 = ra + (i+0)*sMx;
        const mag_e8m23_t *p1 = ra + (i+1)*sMx;
        const mag_e8m23_t *p2 = ra + (i+2)*sMx;
        const mag_e8m23_t *p3 = ra + (i+3)*sMx;
        const mag_e8m23_t *p4 = ra + (i+4)*sMx;
        const mag_e8m23_t *p5 = ra + (i+5)*sMx;
        const mag_e8m23_t *p6 = ra + (i+6)*sMx;
        const mag_e8m23_t *p7 = ra + (i+7)*sMx;
        mag_e8m23_t *dst = pa + i*kc;
        int64_t k = 0;
        for (; k+1 < kc; k += 2) {
            if ((k & ((MAG_PF_GROUP<<1) - 1)) == 0) {
                mag_prefetcht0(p0 + (int64_t)MAG_PFDIST_A_L1*sKx);
                mag_prefetcht0(p4 + (int64_t)MAG_PFDIST_A_L1*sKx);
                mag_prefetcht1(p0 + (int64_t)MAG_PFDIST_A_L2*sKx);
                mag_prefetcht1(p4 + (int64_t)MAG_PFDIST_A_L2*sKx);
            }
            mag_e8m23_t s00 = p0[0];
            mag_e8m23_t s10 = p1[0];
            mag_e8m23_t s20 = p2[0];
            mag_e8m23_t s30 = p3[0];
            mag_e8m23_t s40 = p4[0];
            mag_e8m23_t s50 = p5[0];
            mag_e8m23_t s60 = p6[0];
            mag_e8m23_t s70 = p7[0];
            p0 += sKx;
            p1 += sKx;
            p2 += sKx;
            p3 += sKx;
            p4 += sKx;
            p5 += sKx;
            p6 += sKx;
            p7 += sKx;
            mag_e8m23_t s01 = p0[0];
            mag_e8m23_t s11 = p1[0];
            mag_e8m23_t s21 = p2[0];
            mag_e8m23_t s31 = p3[0];
            mag_e8m23_t s41 = p4[0];
            mag_e8m23_t s51 = p5[0];
            mag_e8m23_t s61 = p6[0];
            mag_e8m23_t s71 = p7[0];
            p0 += sKx;
            p1 += sKx;
            p2 += sKx;
            p3 += sKx;
            p4 += sKx;
            p5 += sKx;
            p6 += sKx;
            p7 += sKx;
#if defined(__AVX512F__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            __m256 v1 = _mm256_setr_ps(s01,s11,s21,s31,s41,s51,s61,s71);
            __m512 vv = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v1, 1);
            _mm512_storeu_ps(dst + k*8, vv);
#elif defined(__AVX2__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            __m256 v1 = _mm256_setr_ps(s01,s11,s21,s31,s41,s51,s61,s71);
            _mm256_storeu_ps(dst + (k+0)*8, v0);
            _mm256_storeu_ps(dst + (k+1)*8, v1);
#elif defined(__SSE4_1__) || defined(__SSE2__)
            __m128 v00 = _mm_setr_ps(s00,s10,s20,s30);
            __m128 v01 = _mm_setr_ps(s40,s50,s60,s70);
            __m128 v10 = _mm_setr_ps(s01,s11,s21,s31);
            __m128 v11 = _mm_setr_ps(s41,s51,s61,s71);
            _mm_storeu_ps(dst + (k+0)*8 + 0, v00);
            _mm_storeu_ps(dst + (k+0)*8 + 4, v01);
            _mm_storeu_ps(dst + (k+1)*8 + 0, v10);
            _mm_storeu_ps(dst + (k+1)*8 + 4, v11);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            float32x4_t v00 = vdupq_n_f32(0.f);
            v00 = vsetq_lane_f32(s00, v00, 0);
            v00 = vsetq_lane_f32(s10, v00, 1);
            v00 = vsetq_lane_f32(s20, v00, 2);
            v00 = vsetq_lane_f32(s30, v00, 3);
            float32x4_t v01 = vdupq_n_f32(0.f);
            v01 = vsetq_lane_f32(s40, v01, 0);
            v01 = vsetq_lane_f32(s50, v01, 1);
            v01 = vsetq_lane_f32(s60, v01, 2);
            v01 = vsetq_lane_f32(s70, v01, 3);
            float32x4_t v10 = vdupq_n_f32(0.f);
            v10 = vsetq_lane_f32(s01, v10, 0);
            v10 = vsetq_lane_f32(s11, v10, 1);
            v10 = vsetq_lane_f32(s21, v10, 2);
            v10 = vsetq_lane_f32(s31, v10, 3);
            float32x4_t v11 = vdupq_n_f32(0.f);
            v11 = vsetq_lane_f32(s41, v11, 0);
            v11 = vsetq_lane_f32(s51, v11, 1);
            v11 = vsetq_lane_f32(s61, v11, 2);
            v11 = vsetq_lane_f32(s71, v11, 3);
            vst1q_f32(dst + (k+0)*8 + 0, v00);
            vst1q_f32(dst + (k+0)*8 + 4, v01);
            vst1q_f32(dst + (k+1)*8 + 0, v10);
            vst1q_f32(dst + (k+1)*8 + 4, v11);

#else
            mag_e8m23_t *d0 = dst + (k+0)*8;
            mag_e8m23_t *d1 = dst + (k+1)*8;
            d0[0]=s00;
            d0[1]=s10;
            d0[2]=s20;
            d0[3]=s30;
            d0[4]=s40;
            d0[5]=s50;
            d0[6]=s60;
            d0[7]=s70;
            d1[0]=s01;
            d1[1]=s11;
            d1[2]=s21;
            d1[3]=s31;
            d1[4]=s41;
            d1[5]=s51;
            d1[6]=s61;
            d1[7]=s71;
#endif
        }
        if (k < kc) {
            mag_e8m23_t s00 = p0[0];
            mag_e8m23_t s10 = p1[0];
            mag_e8m23_t s20 = p2[0];
            mag_e8m23_t s30 = p3[0];
            mag_e8m23_t s40 = p4[0];
            mag_e8m23_t s50 = p5[0];
            mag_e8m23_t s60 = p6[0];
            mag_e8m23_t s70 = p7[0];
#if defined(__AVX512F__) || defined(__AVX2__)
            __m256 v0 = _mm256_setr_ps(s00,s10,s20,s30,s40,s50,s60,s70);
            _mm256_storeu_ps(dst + k*8, v0);
#elif defined(__SSE4_1__) || defined(__SSE2__)
            __m128 v00 = _mm_setr_ps(s00,s10,s20,s30);
            __m128 v01 = _mm_setr_ps(s40,s50,s60,s70);
            _mm_storeu_ps(dst + k*8 + 0, v00);
            _mm_storeu_ps(dst + k*8 + 4, v01);
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
            float32x4_t v00 = vdupq_n_f32(0.f);
            v00 = vsetq_lane_f32(s00, v00, 0);
            v00 = vsetq_lane_f32(s10, v00, 1);
            v00 = vsetq_lane_f32(s20, v00, 2);
            v00 = vsetq_lane_f32(s30, v00, 3);
            float32x4_t v01 = vdupq_n_f32(0.f);
            v01 = vsetq_lane_f32(s40, v01, 0);
            v01 = vsetq_lane_f32(s50, v01, 1);
            v01 = vsetq_lane_f32(s60, v01, 2);
            v01 = vsetq_lane_f32(s70, v01, 3);
            vst1q_f32(dst + k*8 + 0, v00);
            vst1q_f32(dst + k*8 + 4, v01);
#else
            mag_e8m23_t *d0 = dst + k*8;
            d0[0]=s00;
            d0[1]=s10;
            d0[2]=s20;
            d0[3]=s30;
            d0[4]=s40;
            d0[5]=s50;
            d0[6]=s60;
            d0[7]=s70;
#endif
        }
    }
    for (int64_t i=m8; i < mr; ++i) {
        const mag_e8m23_t *src = ra + i*sMx;
        mag_e8m23_t *dst = pa + i*kc;
#if defined(__AVX512F__)
        int64_t k = 0;
        for (; k+15 < kc; k += 16) {
            mag_prefetcht0(src + (k + MAG_PFDIST_A_L1)*sKx);
            mag_prefetcht1(src + (k + MAG_PFDIST_A_L2)*sKx);
            __m512 v = _mm512_set_ps(
                           src[(k+15)*sKx], src[(k+14)*sKx], src[(k+13)*sKx], src[(k+12)*sKx],
                           src[(k+11)*sKx], src[(k+10)*sKx], src[(k+9)*sKx], src[(k+8)*sKx],
                           src[(k+7)*sKx], src[(k+6)*sKx], src[(k+5)*sKx], src[(k+4)*sKx],
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm512_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif defined(__AVX2__)
        int64_t k=0;
        for (; k+7 < kc; k += 8) {
            __m256 v = _mm256_set_ps(
                           src[(k+7)*sKx], src[(k+6)*sKx], src[(k+5)*sKx], src[(k+4)*sKx],
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm256_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif defined(__SSE4_1__) || defined(__SSE2__)
        int64_t k = 0;
        for (; k+3 < kc; k += 4) {
            __m128 v = _mm_set_ps(
                           src[(k+3)*sKx], src[(k+2)*sKx], src[(k+1)*sKx], src[(k+0)*sKx]);
            _mm_storeu_ps(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
        int64_t k=0;
        for (; k+3 < kc; k += 4) {
            float32x4_t v;
            v = vsetq_lane_f32(src[(k+0)*sKx], vdupq_n_f32(0.f), 0);
            v = vsetq_lane_f32(src[(k+1)*sKx], v, 1);
            v = vsetq_lane_f32(src[(k+2)*sKx], v, 2);
            v = vsetq_lane_f32(src[(k+3)*sKx], v, 3);
            vst1q_f32(dst + k, v);
        }
        for (; k < kc; ++k) dst[k] = src[k*sKx];
#else
        for (int64_t k=0; k < kc; ++k) dst[k] = src[k*sKx];
#endif
    }
}

static MAG_AINLINE void mag_mv_e8m23(int64_t K, int64_t N, const mag_e8m23_t *restrict A, const mag_e8m23_t *restrict B, int64_t ldb, mag_e8m23_t *restrict C) {
#ifdef __AVX512F__
    int64_t j=0;
    for (; j+127 < N; j += 128) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        __m512 s7 = _mm512_setzero_ps();
        const mag_e8m23_t *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do { \
                    __m512 a = _mm512_set1_ps(A[k + (i)]); \
                    const mag_e8m23_t *bp = brow + (i)*ldb; \
                    s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 0), s0); \
                    s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 16), s1); \
                    s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 32), s2); \
                    s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 48), s3); \
                    s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 64), s4); \
                    s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 80), s5); \
                    s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 96), s6); \
                    s7 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 112), s7); \
                } while (0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=(K&~3); k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 96), s6);
            s7 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 112), s7);
        }
        _mm512_storeu_ps(C + j + 0, s0);
        _mm512_storeu_ps(C + j + 16, s1);
        _mm512_storeu_ps(C + j + 32, s2);
        _mm512_storeu_ps(C + j + 48, s3);
        _mm512_storeu_ps(C + j + 64, s4);
        _mm512_storeu_ps(C + j + 80, s5);
        _mm512_storeu_ps(C + j + 96, s6);
        _mm512_storeu_ps(C + j + 112, s7);
    }
    for (; j+63 < N; j += 64) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        const mag_e8m23_t *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do { \
                    __m512 a = _mm512_set1_ps(A[k + (i)]); \
                    const mag_e8m23_t *bp = brow + (i)*ldb; \
                    s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 0), s0); \
                    s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 16), s1); \
                    s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 32), s2); \
                    s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bp + 48), s3); \
                } while (0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=(K&~3); k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow + 48), s3);
        }

        _mm512_storeu_ps(C + j + 0, s0);
        _mm512_storeu_ps(C + j + 16, s1);
        _mm512_storeu_ps(C + j + 32, s2);
        _mm512_storeu_ps(C + j + 48, s3);
    }
    for (; j+15 < N; j += 16) {
        __m512 s = _mm512_setzero_ps();
        const mag_e8m23_t *restrict brow = B + j;
        for (int64_t k=0; k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            s = _mm512_fmadd_ps(a, _mm512_loadu_ps(brow), s);
        }
        _mm512_storeu_ps(C + j, s);
    }
    if (j < N) {
        int64_t rem = N-j;
        __mmask16 m = rem == 16 ? (__mmask16)0xffff : (__mmask16)((1u<<rem)-1);
        __m512 s = _mm512_setzero_ps();
        const mag_e8m23_t *restrict brow = B + j;
        for (int64_t k=0; k < K; ++k, brow += ldb) {
            __m512 a = _mm512_set1_ps(A[k]);
            __m512 bv = _mm512_maskz_loadu_ps(m, brow);
            s = _mm512_fmadd_ps(a, bv, s);
        }
        _mm512_mask_storeu_ps(C + j, m, s);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    int64_t j = 0;
    for (; j+63 < N; j += 64) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        __m256 s7 = _mm256_setzero_ps();
        const mag_e8m23_t *restrict brow = B + j;
        int64_t kstep = ldb<<2;
        for (int64_t k=0; k+3 < K; k += 4, brow += kstep) {
#define STEP(i) do {                                        \
                    __m256 a = _mm256_broadcast_ss(A + k + i);              \
                    const mag_e8m23_t *restrict bp = brow + i*ldb;          \
                    s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp +  0), s0);  \
                    s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp +  8), s1);  \
                    s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 16), s2);  \
                    s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 24), s3);  \
                    s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 32), s4);  \
                    s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 40), s5);  \
                    s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 48), s6);  \
                    s7 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bp + 56), s7);  \
                } while(0)
            STEP(0);
            STEP(1);
            STEP(2);
            STEP(3);
#undef STEP
        }
        for (int64_t k=K & ~3; k < K; ++k, brow += ldb) {
            __m256 a = _mm256_broadcast_ss(A + k);
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow +  0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow +  8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 48), s6);
            s7 = _mm256_fmadd_ps(a, _mm256_loadu_ps(brow + 56), s7);
        }
        _mm256_storeu_ps(C + j +  0, s0);
        _mm256_storeu_ps(C + j +  8, s1);
        _mm256_storeu_ps(C + j + 16, s2);
        _mm256_storeu_ps(C + j + 24, s3);
        _mm256_storeu_ps(C + j + 32, s4);
        _mm256_storeu_ps(C + j + 40, s5);
        _mm256_storeu_ps(C + j + 48, s6);
        _mm256_storeu_ps(C + j + 56, s7);
    }
    for (; j+7 < N; j += 8) {
        __m256 s = _mm256_setzero_ps();
        const mag_e8m23_t *restrict b = B + j;
        for (int64_t k=0; k < K; ++k, b += ldb)
            s = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), _mm256_loadu_ps(b), s);
        _mm256_storeu_ps(C + j, s);
    }
    for (; j < N; ++j) {
        mag_e8m23_t s = 0.f;
        for (int64_t k=0; k < K; ++k)
            s += A[k]*B[k*ldb + j];
        C[j] = s;
    }
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    int64_t NN = N&-8;
    int64_t j=0;
    for (; j < NN; j += 8) {
        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        for (int64_t k=0; k < K; ++k) {
            float32x4_t b0 = vld1q_f32(B + k*ldb + j + 0);
            float32x4_t b1 = vld1q_f32(B + k*ldb + j + 4);
            float32x4_t a = vdupq_n_f32(A[k]);
            sum0 = mag_vfmadd_e8m23(sum0, a, b0);
            sum1 = mag_vfmadd_e8m23(sum1, a, b1);
        }
        vst1q_f32(C + j + 0, sum0);
        vst1q_f32(C + j + 4, sum1);
    }
    for (; j < N; ++j) {
        mag_e8m23_t sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            sum += A[k]*B[k*ldb + j];
        C[j] = sum;
    }
#else
    for (int64_t j = 0; j < N; ++j) {
        mag_e8m23_t sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            sum += A[k]*B[k*ldb + j];
        C[j] = sum;
    }
#endif
}

static MAG_AINLINE void mag_mm_tile_16x16_e8m23(int64_t kc, const mag_e8m23_t *restrict a, ptrdiff_t lda, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_8x16_e8m23(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_8x16_e8m23(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_e8m23(int64_t kc, const mag_e8m23_t *restrict a, ptrdiff_t lda, const mag_e8m23_t *restrict b, ptrdiff_t ldb, mag_e8m23_t *restrict c, ptrdiff_t ldc, bool acc) {
    mag_mm_tile_16x16_e8m23(kc, a, lda, b, ldb, c, ldc, acc);
    mag_mm_tile_16x16_e8m23(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_HOTPROC void mag_mm_block_e8m23(int64_t kc, int64_t mr, int64_t nr, const mag_e8m23_t *A, int64_t lda, const mag_e8m23_t *B, int64_t ldb, mag_e8m23_t *C, int64_t ldc, bool acc) {
    int64_t j = 0;
    for (; nr-j >= 32; j += 32) {
        int64_t i = 0;
        for (; mr-i >= 16; i += 16) mag_mm_tile_16x32_e8m23(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x32_e8m23 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x32_e8m23 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr-j >= 16; j += 16) {
        int64_t i = 0;
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x16_e8m23 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x16_e8m23 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    for (; nr-j >= 8; j += 8) {
        int64_t i = 0;
        for (; mr-i >= 8; i += 8) mag_mm_tile_8x8_e8m23 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
        for (; i < mr; ++i) mag_mm_tile_1x8_e8m23 (kc, A + i*lda, B + j, ldb, C + i*ldc + j, acc);
    }
    int64_t rem = nr-j;
    if (!rem) return;
    for (int64_t i2=0; i2 < mr; ++i2) {
        const mag_e8m23_t *ap = A + i2*lda;
        mag_e8m23_t *cp = C + i2*ldc + j;
        for (int64_t jj = 0; jj < rem; ++jj) {
            mag_e8m23_t sum = acc ? cp[jj] : 0.f;
            for (int64_t k=0; k < kc; ++k)
                sum += ap[k]*B[k*ldb + (j + jj)];
            cp[jj] = sum;
        }
    }
}


MAG_HOTPROC static void mag_matmul_e8m23(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    const mag_tensor_t *y = mag_cmd_in(1);
    const mag_e8m23_t *bx = mag_e8m23p(x);
    const mag_e8m23_t *by = mag_e8m23p(y);
    mag_e8m23_t *br = mag_e8m23p_mut(r);
    int64_t MR = payload->mm_params.MR;
    int64_t MC = payload->mm_params.MC;
    int64_t KC = payload->mm_params.KC;
    int64_t NC = payload->mm_params.NC;
    int64_t NR = payload->mm_params.NR;
    int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
    int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
    int64_t K = x->coords.shape[x->coords.rank-1];
    int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d)
        batch_total *= r->coords.shape[d];
    if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { /* Detect GEMV */
        int64_t nth = payload->thread_num;
        int64_t tid = payload->thread_idx;
        int64_t j_per_thread = (N + nth - 1) / nth;
        int64_t j0 = tid*j_per_thread;
        int64_t j1 = mag_xmin(N, j0 + j_per_thread);
        for (int64_t batch = 0; batch < batch_total; ++batch) {
            const mag_e8m23_t *A = bx + mag_offset_rmn(x, batch, 0, 0);
            const mag_e8m23_t *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
            mag_e8m23_t *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
            mag_mv_e8m23(K, j1 - j0, A, B, N, C);
        }
        return;
    }
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t tic = (M+MC-1)/MC;
    int64_t tjc = (N+NC-1)/NC;
    int64_t tpb = tic  *tjc;
    int64_t tt = batch_total  *tpb;
    mag_e8m23_t *scratch = mag_sb_acquire(sizeof(*scratch)*(KC*NC + MC*KC));
    mag_e8m23_t *Bp = scratch;
    mag_e8m23_t *Ap = Bp + KC*NC;
    for (;;) {
        int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
        if (tile >= tt) break;
        int64_t batch_idx = tile / tpb;
        int64_t rem = tile % tpb;
        int64_t jc = rem % tjc;
        int64_t ic = rem / tjc;
        int64_t idx_r[MAG_MAX_DIMS] = {0};
        for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
            idx_r[d] = t % r->coords.shape[d];
            t /= r->coords.shape[d];
        }
        int64_t xb_flat = 0;
        for (int64_t d=0; d < bdx; ++d) {
            int64_t rd = bdr - bdx + d;
            xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        int64_t yb_flat = 0;
        for (int64_t d=0; d < bdy; ++d) {
            int64_t rd = bdr - bdy + d;
            yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
        }
        bool yv = y->coords.rank == 1;
        const mag_e8m23_t *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        const mag_e8m23_t *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
        mag_e8m23_t *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
        int64_t i0 = ic*MC;
        int64_t mc = i0+MC <= M ? MC : M-i0;
        int64_t j0 = jc*NC;
        int64_t nc = j0+NC <= N ? NC : N-j0;
        int64_t sMx = x->coords.strides[x->coords.rank-2];
        int64_t sKx = x->coords.strides[x->coords.rank-1];
        int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
        int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = mag_xmin(KC, K - pc);
            if (y->coords.rank == 1) mag_mm_pack_B_vec_e8m23(kc, nc, py_base + pc, Bp);
            else mag_mm_pack_B_kc_nc_e8m23(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
            mag_mm_pack_A_mc_kc_panel8_e8m23(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
            for (int64_t ir=0; ir < mc; ir += MR)
                for (int64_t jr=0; jr < nc; jr += NR)
                    mag_mm_block_e8m23(
                        kc,
                        mag_xmin(MR, mc - ir),
                        mag_xmin(NR, nc - jr),
                        Ap + ir*kc,
                        kc,
                        Bp + jr,
                        nc,
                        pr_base + (i0 + ir)*N + (j0 + jr),
                        N,
                        pc);
        }
    }
}

static MAG_HOTPROC void mag_matmul_e5m10(const mag_kernel_payload_t *payload) {
    if (payload->thread_idx != 0) return;
    mag_tensor_t *r  = mag_cmd_out(0);
    const mag_tensor_t *x  = mag_cmd_in(0);
    const mag_tensor_t *y  = mag_cmd_in(1);
    mag_e5m10_t *br = mag_e5m10p_mut(r);
    const mag_e5m10_t *bx = mag_e5m10p(x);
    const mag_e5m10_t *by = mag_e5m10p(y);
    int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank - 2];
    int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank - 1];
    int64_t K = x->coords.shape[x->coords.rank - 1];
    int64_t bdr = r->coords.rank > 2 ? r->coords.rank-2 : 0;
    int64_t batch_total = 1;
    for (int64_t d=0; d < bdr; ++d) batch_total *= r->coords.shape[d];
    int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    bool x_row = mag_tensor_is_contiguous(x) && x->coords.strides[x->coords.rank-1] == 1;
    mag_e5m10_t *scratch = mag_sb_acquire(sizeof(mag_e5m10_t)*(K*N + (x_row ? 0 : M*K)));
    mag_e5m10_t *xbuf = x_row ? NULL : scratch;
    mag_e5m10_t *ybuf = scratch + (x_row ? 0 : M*K);
    int64_t idx_r[4] = {0};
    for (int64_t b=0; b < batch_total; ++b) {
        int64_t rem = b;
        for (int64_t d = bdr-1; d >= 0; --d) {
            idx_r[d] = rem % r->coords.shape[d];
            rem /= r->coords.shape[d];
        }
        int64_t xb_flat = 0, yb_flat = 0;
        if (bdx) {
            for (int64_t d=0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = x->coords.shape[d] == 1 ? 0 : idx_r[rd];
                xb_flat = xb_flat  *x->coords.shape[d] + idx;
            }
        }
        if (bdy) {
            for (int64_t d=0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = y->coords.shape[d] == 1 ? 0 : idx_r[rd];
                yb_flat = yb_flat  *y->coords.shape[d] + idx;
            }
        }
        const mag_e5m10_t *px = bx + mag_offset_rmn(x, xb_flat, 0, 0);
        mag_e5m10_t *pr = br + mag_offset_rmn(r,b, 0, 0);
        const mag_e5m10_t *restrict A = x_row ? px : mag_mm_pack_x_e5m10(xbuf, M, K, xb_flat, x, bx);
        const mag_e5m10_t *restrict B = mag_mm_pack_y_e5m10(ybuf, K, N, yb_flat, y, by);
        mag_e5m10_t *restrict C = pr;
        for (int64_t i=0; i < M; ++i) {
            const mag_e5m10_t *restrict a_row = A + i*K;
            for (int64_t n=0; n < N; ++n) {
                const mag_e5m10_t *restrict b_col = B + n*K;
                C[i*N + n] = mag_vdot_e5m10(K, b_col, a_row);
            }
        }
    }
}

static void (*const mag_lut_eval_kernels[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_kernel_payload_t *) = {
    [MAG_OP_NOP] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_FILL] = {
        [MAG_DTYPE_E8M23] = &mag_fill_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_e5m10,
        [MAG_DTYPE_BOOL] = &mag_fill_bool,
        [MAG_DTYPE_U8] = &mag_fill_u8,
        [MAG_DTYPE_I8] = &mag_fill_i8,
        [MAG_DTYPE_U16] = &mag_fill_u16,
        [MAG_DTYPE_I16] = &mag_fill_i16,
        [MAG_DTYPE_U32] = &mag_fill_u32,
        [MAG_DTYPE_I32] = &mag_fill_i32,
        [MAG_DTYPE_U64] = &mag_fill_u64,
        [MAG_DTYPE_I64] = &mag_fill_i64,
    },
    [MAG_OP_MASKED_FILL] = {
        [MAG_DTYPE_E8M23] = &mag_masked_fill_e8m23,
        [MAG_DTYPE_E5M10] = &mag_masked_fill_e5m10,
        [MAG_DTYPE_BOOL] = &mag_masked_fill_bool,
        [MAG_DTYPE_U8]  = &mag_masked_fill_u8,
        [MAG_DTYPE_I8]  = &mag_masked_fill_i8,
        [MAG_DTYPE_U16] = &mag_masked_fill_u16,
        [MAG_DTYPE_I16] = &mag_masked_fill_i16,
        [MAG_DTYPE_U32] = &mag_masked_fill_u32,
        [MAG_DTYPE_I32] = &mag_masked_fill_i32,
        [MAG_DTYPE_U64] = &mag_masked_fill_u64,
        [MAG_DTYPE_I64] = &mag_masked_fill_i64,
    },
    [MAG_OP_RAND_UNIFORM] = {
        [MAG_DTYPE_E8M23] = &mag_fill_rand_uniform_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_rand_uniform_e5m10,
        [MAG_DTYPE_I32] = &mag_fill_rand_uniform_i32
    },
    [MAG_OP_RAND_NORMAL] = {
        [MAG_DTYPE_E8M23] = &mag_fill_rand_normal_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_rand_normal_e5m10,
    },
    [MAG_OP_RAND_BERNOULLI] = {
        [MAG_DTYPE_BOOL] = &mag_fill_rand_bernoulli_bool,
    },
    [MAG_OP_ARANGE] = {
        [MAG_DTYPE_E8M23] = &mag_fill_arange_e8m23,
        [MAG_DTYPE_E5M10] = &mag_fill_arange_e5m10,
        [MAG_DTYPE_U8] = &mag_fill_arange_u8,
        [MAG_DTYPE_I8] = &mag_fill_arange_i8,
        [MAG_DTYPE_U16] = &mag_fill_arange_u16,
        [MAG_DTYPE_I16] = &mag_fill_arange_i16,
        [MAG_DTYPE_U32] = &mag_fill_arange_u32,
        [MAG_DTYPE_I32] = &mag_fill_arange_i32,
        [MAG_DTYPE_U64] = &mag_fill_arange_u64,
        [MAG_DTYPE_I64] = &mag_fill_arange_i64,
    },
    [MAG_OP_CLONE] = {
        [MAG_DTYPE_E8M23] = &mag_clone_e8m23,
        [MAG_DTYPE_E5M10] = &mag_clone_e5m10,
        [MAG_DTYPE_BOOL] = &mag_clone_bool,
        [MAG_DTYPE_U8] = &mag_clone_u8,
        [MAG_DTYPE_I8] = &mag_clone_i8,
        [MAG_DTYPE_U16] = &mag_clone_u16,
        [MAG_DTYPE_I16] = &mag_clone_i16,
        [MAG_DTYPE_U32] = &mag_clone_u32,
        [MAG_DTYPE_I32] = &mag_clone_i32,
        [MAG_DTYPE_U64] = &mag_clone_u64,
        [MAG_DTYPE_I64] = &mag_clone_i64,
    },
    [MAG_OP_VIEW] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_TRANSPOSE] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_PERMUTE] = {
        [MAG_DTYPE_E8M23] = &mag_nop,
        [MAG_DTYPE_E5M10] = &mag_nop,
        [MAG_DTYPE_BOOL] = &mag_nop,
        [MAG_DTYPE_U8] = &mag_nop,
        [MAG_DTYPE_I8] = &mag_nop,
        [MAG_DTYPE_U16] = &mag_nop,
        [MAG_DTYPE_I16] = &mag_nop,
        [MAG_DTYPE_U32] = &mag_nop,
        [MAG_DTYPE_I32] = &mag_nop,
        [MAG_DTYPE_U64] = &mag_nop,
        [MAG_DTYPE_I64] = &mag_nop,
    },
    [MAG_OP_MEAN] = {
        [MAG_DTYPE_E8M23] = &mag_mean_e8m23,
        [MAG_DTYPE_E5M10] = &mag_mean_e5m10,
    },
    [MAG_OP_MIN] = {
        [MAG_DTYPE_E8M23] = &mag_min_e8m23,
        [MAG_DTYPE_E5M10] = &mag_min_e5m10,
    },
    [MAG_OP_MAX] = {
        [MAG_DTYPE_E8M23] = &mag_max_e8m23,
        [MAG_DTYPE_E5M10] = &mag_max_e5m10,
    },
    [MAG_OP_SUM] = {
        [MAG_DTYPE_E8M23] = &mag_sum_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sum_e5m10,
    },
    [MAG_OP_ABS] = {
        [MAG_DTYPE_E8M23] = &mag_abs_e8m23,
        [MAG_DTYPE_E5M10] = &mag_abs_e5m10,
    },
    [MAG_OP_SGN] = {
        [MAG_DTYPE_E8M23] = &mag_sgn_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sgn_e5m10,
    },
    [MAG_OP_NEG] = {
        [MAG_DTYPE_E8M23] = &mag_neg_e8m23,
        [MAG_DTYPE_E5M10] = &mag_neg_e5m10,
    },
    [MAG_OP_LOG] = {
        [MAG_DTYPE_E8M23] = &mag_log_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log_e5m10,
    },
    [MAG_OP_LOG10] = {
        [MAG_DTYPE_E8M23] = &mag_log10_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log10_e5m10,
    },
    [MAG_OP_LOG1P] = {
        [MAG_DTYPE_E8M23] = &mag_log1p_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log1p_e5m10,
    },
    [MAG_OP_LOG2] = {
        [MAG_DTYPE_E8M23] = &mag_log2_e8m23,
        [MAG_DTYPE_E5M10] = &mag_log2_e5m10,
    },
    [MAG_OP_SQR] = {
        [MAG_DTYPE_E8M23] = &mag_sqr_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sqr_e5m10,
    },
    [MAG_OP_SQRT] = {
        [MAG_DTYPE_E8M23] = &mag_sqrt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sqrt_e5m10,
    },
    [MAG_OP_SIN] = {
        [MAG_DTYPE_E8M23] = &mag_sin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sin_e5m10,
    },
    [MAG_OP_COS] = {
        [MAG_DTYPE_E8M23] = &mag_cos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cos_e5m10,
    },
    [MAG_OP_TAN] = {
        [MAG_DTYPE_E8M23] = &mag_tan_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tan_e5m10,
    },
    [MAG_OP_SINH] = {
        [MAG_DTYPE_E8M23] = &mag_sinh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sinh_e5m10,
    },
    [MAG_OP_COSH] = {
        [MAG_DTYPE_E8M23] = &mag_cosh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cosh_e5m10,
    },
    [MAG_OP_TANH] = {
        [MAG_DTYPE_E8M23] = &mag_tanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tanh_e5m10,
    },
    [MAG_OP_ASIN] = {
        [MAG_DTYPE_E8M23] = &mag_asin_e8m23,
        [MAG_DTYPE_E5M10] = &mag_asin_e5m10,
    },
    [MAG_OP_ACOS] = {
        [MAG_DTYPE_E8M23] = &mag_acos_e8m23,
        [MAG_DTYPE_E5M10] = &mag_acos_e5m10,
    },
    [MAG_OP_ATAN] = {
        [MAG_DTYPE_E8M23] = &mag_atan_e8m23,
        [MAG_DTYPE_E5M10] = &mag_atan_e5m10,
    },
    [MAG_OP_ASINH] = {
        [MAG_DTYPE_E8M23] = &mag_asinh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_asinh_e5m10,
    },
    [MAG_OP_ACOSH] = {
        [MAG_DTYPE_E8M23] = &mag_acosh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_acosh_e5m10,
    },
    [MAG_OP_ATANH] = {
        [MAG_DTYPE_E8M23] = &mag_atanh_e8m23,
        [MAG_DTYPE_E5M10] = &mag_atanh_e5m10,
    },
    [MAG_OP_STEP] = {
        [MAG_DTYPE_E8M23] = &mag_step_e8m23,
        [MAG_DTYPE_E5M10] = &mag_step_e5m10,
    },
    [MAG_OP_ERF] = {
        [MAG_DTYPE_E8M23] = &mag_erf_e8m23,
        [MAG_DTYPE_E5M10] = &mag_erf_e5m10,
    },
    [MAG_OP_ERFC] = {
        [MAG_DTYPE_E8M23] = &mag_erfc_e8m23,
        [MAG_DTYPE_E5M10] = &mag_erfc_e5m10,
    },
    [MAG_OP_EXP] = {
        [MAG_DTYPE_E8M23] = &mag_exp_e8m23,
        [MAG_DTYPE_E5M10] = &mag_exp_e5m10,
    },
    [MAG_OP_EXP2] = {
        [MAG_DTYPE_E8M23] = &mag_exp2_e8m23,
        [MAG_DTYPE_E5M10] = &mag_exp2_e5m10,
    },
    [MAG_OP_EXPM1] = {
        [MAG_DTYPE_E8M23] = &mag_expm1_e8m23,
        [MAG_DTYPE_E5M10] = &mag_expm1_e5m10,
    },
    [MAG_OP_FLOOR] = {
        [MAG_DTYPE_E8M23] = &mag_floor_e8m23,
        [MAG_DTYPE_E5M10] = &mag_floor_e5m10,
    },
    [MAG_OP_CEIL] = {
        [MAG_DTYPE_E8M23] = &mag_ceil_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ceil_e5m10,
    },
    [MAG_OP_ROUND] = {
        [MAG_DTYPE_E8M23] = &mag_round_e8m23,
        [MAG_DTYPE_E5M10] = &mag_round_e5m10,
    },
    [MAG_OP_TRUNC] = {
        [MAG_DTYPE_E8M23] = &mag_trunc_e8m23,
        [MAG_DTYPE_E5M10] = &mag_trunc_e5m10,
    },
    [MAG_OP_SOFTMAX] = {
        [MAG_DTYPE_E8M23] = &mag_softmax_e8m23,
        [MAG_DTYPE_E5M10] = &mag_softmax_e5m10,
    },
    [MAG_OP_SOFTMAX_DV] = {
        [MAG_DTYPE_E8M23] = &mag_softmax_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_softmax_dv_e5m10,
    },
    [MAG_OP_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sigmoid_e5m10,
    },
    [MAG_OP_SIGMOID_DV] = {
        [MAG_DTYPE_E8M23] = &mag_sigmoid_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sigmoid_dv_e5m10,
    },
    [MAG_OP_HARD_SIGMOID] = {
        [MAG_DTYPE_E8M23] = &mag_hard_sigmoid_e8m23,
        [MAG_DTYPE_E5M10] = &mag_hard_sigmoid_e5m10,
    },
    [MAG_OP_SILU] = {
        [MAG_DTYPE_E8M23] = &mag_silu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_silu_e5m10,
    },
    [MAG_OP_SILU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_silu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_silu_dv_e5m10,
    },
    [MAG_OP_TANH_DV] = {
        [MAG_DTYPE_E8M23] = &mag_tanh_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tanh_dv_e5m10,
    },
    [MAG_OP_RELU] = {
        [MAG_DTYPE_E8M23] = &mag_relu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_relu_e5m10,
    },
    [MAG_OP_RELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_relu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_relu_dv_e5m10,
    },
    [MAG_OP_GELU] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_e5m10,
    },
    [MAG_OP_GELU_APPROX] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_approx_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_approx_e5m10,
    },
    [MAG_OP_GELU_DV] = {
        [MAG_DTYPE_E8M23] = &mag_gelu_dv_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gelu_dv_e5m10,
    },
    [MAG_OP_TRIL] = {
        [MAG_DTYPE_E8M23] = &mag_tril_e8m23,
        [MAG_DTYPE_E5M10] = &mag_tril_e5m10,
        [MAG_DTYPE_BOOL] = &mag_tril_bool,
        [MAG_DTYPE_U8]  = &mag_tril_u8,
        [MAG_DTYPE_I8]  = &mag_tril_i8,
        [MAG_DTYPE_U16] = &mag_tril_u16,
        [MAG_DTYPE_I16] = &mag_tril_i16,
        [MAG_DTYPE_U32] = &mag_tril_u32,
        [MAG_DTYPE_I32] = &mag_tril_i32,
        [MAG_DTYPE_U64] = &mag_tril_u64,
        [MAG_DTYPE_I64] = &mag_tril_i64,
    },
    [MAG_OP_TRIU] = {
        [MAG_DTYPE_E8M23] = &mag_triu_e8m23,
        [MAG_DTYPE_E5M10] = &mag_triu_e5m10,
        [MAG_DTYPE_BOOL] = &mag_triu_bool,
        [MAG_DTYPE_U8]  = &mag_triu_u8,
        [MAG_DTYPE_I8]  = &mag_triu_i8,
        [MAG_DTYPE_U16] = &mag_triu_u16,
        [MAG_DTYPE_I16] = &mag_triu_i16,
        [MAG_DTYPE_U32] = &mag_triu_u32,
        [MAG_DTYPE_I32] = &mag_triu_i32,
        [MAG_DTYPE_U64] = &mag_triu_u64,
        [MAG_DTYPE_I64] = &mag_triu_i64,
    },
    [MAG_OP_MULTINOMIAL] = {
        [MAG_DTYPE_E8M23] = &mag_multinomial_e8m23,
        [MAG_DTYPE_E5M10] = &mag_multinomial_e5m10,
    },
    [MAG_OP_CAT] = {
        [MAG_DTYPE_E8M23] = &mag_cat_e8m23,
        [MAG_DTYPE_E5M10] = &mag_cat_e5m10,
        [MAG_DTYPE_BOOL] = &mag_cat_bool,
        [MAG_DTYPE_U8]  = &mag_cat_u8,
        [MAG_DTYPE_I8]  = &mag_cat_i8,
        [MAG_DTYPE_U16] = &mag_cat_u16,
        [MAG_DTYPE_I16] = &mag_cat_i16,
        [MAG_DTYPE_U32] = &mag_cat_u32,
        [MAG_DTYPE_I32] = &mag_cat_i32,
        [MAG_DTYPE_U64] = &mag_cat_u64,
        [MAG_DTYPE_I64] = &mag_cat_i64,
    },
    [MAG_OP_ADD] = {
        [MAG_DTYPE_E8M23] = &mag_add_e8m23,
        [MAG_DTYPE_E5M10] = &mag_add_e5m10,
        [MAG_DTYPE_U8]  = &mag_add_u8,
        [MAG_DTYPE_I8]  = &mag_add_i8,
        [MAG_DTYPE_U16] = &mag_add_u16,
        [MAG_DTYPE_I16] = &mag_add_i16,
        [MAG_DTYPE_U32] = &mag_add_u32,
        [MAG_DTYPE_I32] = &mag_add_i32,
        [MAG_DTYPE_U64] = &mag_add_u64,
        [MAG_DTYPE_I64] = &mag_add_i64,
    },
    [MAG_OP_SUB] = {
        [MAG_DTYPE_E8M23] = &mag_sub_e8m23,
        [MAG_DTYPE_E5M10] = &mag_sub_e5m10,
        [MAG_DTYPE_U8]  = &mag_sub_u8,
        [MAG_DTYPE_I8]  = &mag_sub_i8,
        [MAG_DTYPE_U16] = &mag_sub_u16,
        [MAG_DTYPE_I16] = &mag_sub_i16,
        [MAG_DTYPE_U32] = &mag_sub_u32,
        [MAG_DTYPE_I32] = &mag_sub_i32,
        [MAG_DTYPE_U64] = &mag_sub_u64,
        [MAG_DTYPE_I64] = &mag_sub_i64,
    },
    [MAG_OP_MUL] = {
        [MAG_DTYPE_E8M23] = &mag_mul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_mul_e5m10,
        [MAG_DTYPE_U8]  = &mag_mul_u8,
        [MAG_DTYPE_I8]  = &mag_mul_i8,
        [MAG_DTYPE_U16] = &mag_mul_u16,
        [MAG_DTYPE_I16] = &mag_mul_i16,
        [MAG_DTYPE_U32] = &mag_mul_u32,
        [MAG_DTYPE_I32] = &mag_mul_i32,
        [MAG_DTYPE_U64] = &mag_mul_u64,
        [MAG_DTYPE_I64] = &mag_mul_i64,
    },
    [MAG_OP_DIV] = {
        [MAG_DTYPE_E8M23] = &mag_div_e8m23,
        [MAG_DTYPE_E5M10] = &mag_div_e5m10,
        [MAG_DTYPE_U8]  = &mag_div_u8,
        [MAG_DTYPE_I8]  = &mag_div_i8,
        [MAG_DTYPE_U16] = &mag_div_u16,
        [MAG_DTYPE_I16] = &mag_div_i16,
        [MAG_DTYPE_U32] = &mag_div_u32,
        [MAG_DTYPE_I32] = &mag_div_i32,
        [MAG_DTYPE_U64] = &mag_div_u64,
        [MAG_DTYPE_I64] = &mag_div_i64,
    },
    [MAG_OP_MATMUL] = {
        [MAG_DTYPE_E8M23] = &mag_matmul_e8m23,
        [MAG_DTYPE_E5M10] = &mag_matmul_e5m10,
    },
    [MAG_OP_REPEAT_BACK] = {
        [MAG_DTYPE_E8M23] = &mag_repeat_back_e8m23,
        [MAG_DTYPE_E5M10] = &mag_repeat_back_e5m10,
    },
    [MAG_OP_GATHER] = {
        [MAG_DTYPE_E8M23] = &mag_gather_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gather_e5m10,
        [MAG_DTYPE_BOOL] = &mag_gather_bool,
        [MAG_DTYPE_U8]  = &mag_gather_u8,
        [MAG_DTYPE_I8]  = &mag_gather_i8,
        [MAG_DTYPE_U16] = &mag_gather_u16,
        [MAG_DTYPE_I16] = &mag_gather_i16,
        [MAG_DTYPE_U32] = &mag_gather_u32,
        [MAG_DTYPE_I32] = &mag_gather_i32,
        [MAG_DTYPE_U64] = &mag_gather_u64,
        [MAG_DTYPE_I64] = &mag_gather_i64,
    },
    [MAG_OP_AND] = {
        [MAG_DTYPE_BOOL] = &mag_and_bool,
        [MAG_DTYPE_U8]  = &mag_and_u8,
        [MAG_DTYPE_I8]  = &mag_and_i8,
        [MAG_DTYPE_U16] = &mag_and_u16,
        [MAG_DTYPE_I16] = &mag_and_i16,
        [MAG_DTYPE_U32] = &mag_and_u32,
        [MAG_DTYPE_I32] = &mag_and_i32,
        [MAG_DTYPE_U64] = &mag_and_u64,
        [MAG_DTYPE_I64] = &mag_and_i64,
    },
    [MAG_OP_OR] = {
        [MAG_DTYPE_BOOL] = &mag_or_bool,
        [MAG_DTYPE_U8]  = &mag_or_u8,
        [MAG_DTYPE_I8]  = &mag_or_i8,
        [MAG_DTYPE_U16] = &mag_or_u16,
        [MAG_DTYPE_I16] = &mag_or_i16,
        [MAG_DTYPE_U32] = &mag_or_u32,
        [MAG_DTYPE_I32] = &mag_or_i32,
        [MAG_DTYPE_U64] = &mag_or_u64,
        [MAG_DTYPE_I64] = &mag_or_i64,
    },
    [MAG_OP_XOR] = {
        [MAG_DTYPE_BOOL] = &mag_xor_bool,
        [MAG_DTYPE_U8]  = &mag_xor_u8,
        [MAG_DTYPE_I8]  = &mag_xor_i8,
        [MAG_DTYPE_U16] = &mag_xor_u16,
        [MAG_DTYPE_I16] = &mag_xor_i16,
        [MAG_DTYPE_U32] = &mag_xor_u32,
        [MAG_DTYPE_I32] = &mag_xor_i32,
        [MAG_DTYPE_U64] = &mag_xor_u64,
        [MAG_DTYPE_I64] = &mag_xor_i64,
    },
    [MAG_OP_NOT] = {
        [MAG_DTYPE_BOOL] = &mag_not_bool,
        [MAG_DTYPE_U8]  = &mag_not_u8,
        [MAG_DTYPE_I8]  = &mag_not_i8,
        [MAG_DTYPE_U16] = &mag_not_u16,
        [MAG_DTYPE_I16] = &mag_not_i16,
        [MAG_DTYPE_U32] = &mag_not_u32,
        [MAG_DTYPE_I32] = &mag_not_i32,
        [MAG_DTYPE_U64] = &mag_not_u64,
        [MAG_DTYPE_I64] = &mag_not_i64,
    },
    [MAG_OP_SHL] = {
        [MAG_DTYPE_U8]  = &mag_shl_u8,
        [MAG_DTYPE_I8]  = &mag_shl_i8,
        [MAG_DTYPE_U16] = &mag_shl_u16,
        [MAG_DTYPE_I16] = &mag_shl_i16,
        [MAG_DTYPE_U32] = &mag_shl_u32,
        [MAG_DTYPE_I32] = &mag_shl_i32,
        [MAG_DTYPE_U64] = &mag_shl_u64,
        [MAG_DTYPE_I64] = &mag_shl_i64,
    },
    [MAG_OP_SHR] = {
        [MAG_DTYPE_U8]  = &mag_shr_u8,
        [MAG_DTYPE_I8]  = &mag_shr_i8,
        [MAG_DTYPE_U16] = &mag_shr_u16,
        [MAG_DTYPE_I16] = &mag_shr_i16,
        [MAG_DTYPE_U32] = &mag_shr_u32,
        [MAG_DTYPE_I32] = &mag_shr_i32,
        [MAG_DTYPE_U64] = &mag_shr_u64,
        [MAG_DTYPE_I64] = &mag_shr_i64,
    },
    [MAG_OP_EQ] = {
        [MAG_DTYPE_E8M23] = &mag_eq_e8m23,
        [MAG_DTYPE_E5M10] = &mag_eq_e5m10,
        [MAG_DTYPE_BOOL] = &mag_eq_bool,
        [MAG_DTYPE_U8]  = &mag_eq_u8,
        [MAG_DTYPE_I8]  = &mag_eq_i8,
        [MAG_DTYPE_U16] = &mag_eq_u16,
        [MAG_DTYPE_I16] = &mag_eq_i16,
        [MAG_DTYPE_U32] = &mag_eq_u32,
        [MAG_DTYPE_I32] = &mag_eq_i32,
        [MAG_DTYPE_U64] = &mag_eq_u64,
        [MAG_DTYPE_I64] = &mag_eq_i64,
    },
    [MAG_OP_NE] = {
        [MAG_DTYPE_E8M23] = &mag_ne_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ne_e5m10,
        [MAG_DTYPE_BOOL] = &mag_ne_bool,
        [MAG_DTYPE_U8]  = &mag_ne_u8,
        [MAG_DTYPE_I8]  = &mag_ne_i8,
        [MAG_DTYPE_U16] = &mag_ne_u16,
        [MAG_DTYPE_I16] = &mag_ne_i16,
        [MAG_DTYPE_U32] = &mag_ne_u32,
        [MAG_DTYPE_I32] = &mag_ne_i32,
        [MAG_DTYPE_U64] = &mag_ne_u64,
        [MAG_DTYPE_I64] = &mag_ne_i64,
    },
    [MAG_OP_LE] = {
        [MAG_DTYPE_E8M23] = &mag_le_e8m23,
        [MAG_DTYPE_E5M10] = &mag_le_e5m10,
        [MAG_DTYPE_U8]  = &mag_le_u8,
        [MAG_DTYPE_I8]  = &mag_le_i8,
        [MAG_DTYPE_U16] = &mag_le_u16,
        [MAG_DTYPE_I16] = &mag_le_i16,
        [MAG_DTYPE_U32] = &mag_le_u32,
        [MAG_DTYPE_I32] = &mag_le_i32,
        [MAG_DTYPE_U64] = &mag_le_u64,
        [MAG_DTYPE_I64] = &mag_le_i64,
    },
    [MAG_OP_GE] = {
        [MAG_DTYPE_E8M23] = &mag_ge_e8m23,
        [MAG_DTYPE_E5M10] = &mag_ge_e5m10,
        [MAG_DTYPE_U8]  = &mag_ge_u8,
        [MAG_DTYPE_I8]  = &mag_ge_i8,
        [MAG_DTYPE_U16] = &mag_ge_u16,
        [MAG_DTYPE_I16] = &mag_ge_i16,
        [MAG_DTYPE_U32] = &mag_ge_u32,
        [MAG_DTYPE_I32] = &mag_ge_i32,
        [MAG_DTYPE_U64] = &mag_ge_u64,
        [MAG_DTYPE_I64] = &mag_ge_i64,
    },
    [MAG_OP_LT] = {
        [MAG_DTYPE_E8M23] = &mag_lt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_lt_e5m10,
        [MAG_DTYPE_U8]  = &mag_lt_u8,
        [MAG_DTYPE_I8]  = &mag_lt_i8,
        [MAG_DTYPE_U16] = &mag_lt_u16,
        [MAG_DTYPE_I16] = &mag_lt_i16,
        [MAG_DTYPE_U32] = &mag_lt_u32,
        [MAG_DTYPE_I32] = &mag_lt_i32,
        [MAG_DTYPE_U64] = &mag_lt_u64,
        [MAG_DTYPE_I64] = &mag_lt_i64,
    },
    [MAG_OP_GT] = {
        [MAG_DTYPE_E8M23] = &mag_gt_e8m23,
        [MAG_DTYPE_E5M10] = &mag_gt_e5m10,
        [MAG_DTYPE_U8]  = &mag_gt_u8,
        [MAG_DTYPE_I8]  = &mag_gt_i8,
        [MAG_DTYPE_U16] = &mag_gt_u16,
        [MAG_DTYPE_I16] = &mag_gt_i16,
        [MAG_DTYPE_U32] = &mag_gt_u32,
        [MAG_DTYPE_I32] = &mag_gt_i32,
        [MAG_DTYPE_U64] = &mag_gt_u64,
        [MAG_DTYPE_I64] = &mag_gt_i64,
    },
};

static void (*const mag_lut_cast_kernels[MAG_DTYPE__NUM][MAG_DTYPE__NUM])(int64_t, void *, const void *) = {
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

static void MAG_HOTPROC mag_vector_cast_stub(size_t nb, const void *src, mag_dtype_t src_t, void *dst, mag_dtype_t dst_t) {
    mag_assert2(dst_t != src_t); /* src and dst types must differ */
    size_t nbs = mag_dtype_meta_of(src_t)->size;
    size_t nbd = mag_dtype_meta_of(dst_t)->size;
    mag_assert2(!((uintptr_t)src&(nbs-1)));     /* src must be aligned */
    mag_assert2(!((uintptr_t)dst&(nbd-1)));     /* dst must be aligned */
    mag_assert2(!(nb&(nbs-1)));                 /* size must be aligned */
    int64_t numel = (int64_t)(nb/nbs);          /* byte -> elems */
    void (*kern)(int64_t, void *, const void *) = mag_lut_cast_kernels[src_t][dst_t];
    mag_assert(kern, "invalid cast dtypes %s -> %s", mag_dtype_meta_of(src_t)->name, mag_dtype_meta_of(dst_t)->name);
    (*kern)(numel, dst, src);
}

static size_t mag_vreg_width(void) {
    return MAG_VREG_WIDTH;
}

static void mag_impl_init(void) {

}

static void mag_impl_deinit(void) {
    mag_sb_release();
}

void MAG_BLAS_SPECIALIZATION(mag_kernel_registry_t *kernels) {
    kernels->init = &mag_impl_init;
    kernels->deinit = &mag_impl_deinit;
    for (int i=0; i < MAG_OP__NUM; ++i) {
        for (int j=0; j < MAG_DTYPE__NUM; ++j) {
            kernels->operators[i][j] = mag_lut_eval_kernels[i][j];
        }
    }
    kernels->vector_cast = &mag_vector_cast_stub;
    kernels->vreg_width = &mag_vreg_width;
}

#ifndef MAG_BLAS_SPECIALIZATION
#error "BLAS specialization undefined"
#endif
#ifndef MAG_BLAS_SPECIALIZATION_FEAT_REQUEST
#error "Feature request routine undefined"
#endif

#if defined(__x86_64__) || defined(_M_X64)
/*
** x86-64 specific feature detection.
** This function is always called, so it must run down to SSE2 at least.
** This means that there should be no fancy instructions or extensions.
** There was a bug where the backend with Intel APX enabled used
** the instruction: pushp  %rbp (d5 08 55) for function prologue, whis is Intel APX and crashes on older CPUs.
** This is why this function should really only return one single integer scalar in the return register, according to the calling convention,
** and NO other code or logic. The function is marked naked to supress the prologue/epilogue generation and associated extension instructions.
*/
mag_amd64_cap_bitset_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST() {
    mag_amd64_cap_bitset_t caps = 0;
#ifdef __SSE__
    caps|=mag_amd64_cap(SSE);
#endif
#ifdef __SSE2__
    caps|=mag_amd64_cap(SSE2);
#endif
#ifdef __SSE3__
    caps|=mag_amd64_cap(SSE3);
#endif
#ifdef __SSSE3__
    caps|=mag_amd64_cap(SSSE3);
#endif
#ifdef __SSE4_1__
    caps|=mag_amd64_cap(SSE41);
#endif
#ifdef __SSE4_2__
    caps|=mag_amd64_cap(SSE42);
#endif
#ifdef __SSE4A__
    caps|=mag_amd64_cap(SSE4A);
#endif

#ifdef __AVX__
    caps|=mag_amd64_cap(AVX);
#endif
#ifdef __FMA__
    caps|=mag_amd64_cap(FMA);
#endif
#ifdef __AVX2__
    caps|=mag_amd64_cap(AVX2);
#endif
#ifdef __F16C__
    caps|=mag_amd64_cap(F16C);
#endif
#ifdef __AVXVNNI__
    caps|=mag_amd64_cap(AVX_VNNI);
#endif
#ifdef __AVXVNNIINT8__
    caps|=mag_amd64_cap(AVX_VNNI_INT8);
#endif
#ifdef __AVXNECONVERT__
    caps|=mag_amd64_cap(AVX_NE_CONVERT);
#endif
#ifdef __AVXIFMA__
    caps|=mag_amd64_cap(AVX_IFMA);
#endif
#ifdef __AVXVNNIINT16__
    caps|=mag_amd64_cap(AVX_VNNI_INT16);
#endif
#ifdef __AVX10__
    caps|=mag_amd64_cap(AVX10);
#endif

#ifdef __AVX512F__
    caps|=mag_amd64_cap(AVX512_F);
#endif
#ifdef __AVX512DQ__
    caps|=mag_amd64_cap(AVX512_DQ);
#endif
#ifdef __AVX512IFMA__
    caps|=mag_amd64_cap(AVX512_IFMA);
#endif
#ifdef __AVX512PF__
    caps|=mag_amd64_cap(AVX512_PF);
#endif
#ifdef __AVX512ER__
    caps|=mag_amd64_cap(AVX512_ER);
#endif
#ifdef __AVX512CD__
    caps|=mag_amd64_cap(AVX512_CD);
#endif
#ifdef __AVX512BW__
    caps|=mag_amd64_cap(AVX512_BW);
#endif
#ifdef __AVX512VL__
    caps|=mag_amd64_cap(AVX512_VL);
#endif
#ifdef __AVX512VBMI__
    caps|=mag_amd64_cap(AVX512_VBMI);
#endif
#ifdef __AVX5124VNNIW__
    caps|=mag_amd64_cap(AVX512_4VNNIW);
#endif
#ifdef __AVX5124FMAPS__
    caps|=mag_amd64_cap(AVX512_4FMAPS);
#endif
#ifdef __AVX512VBMI2__
    caps|=mag_amd64_cap(AVX512_VBMI2);
#endif
#ifdef __AVX512VNNI__
    caps|=mag_amd64_cap(AVX512_VNNI);
#endif
#ifdef __AVX512BITALG__
    caps|=mag_amd64_cap(AVX512_BITALG);
#endif
#ifdef __AVX512VPOPCNTDQ__
    caps|=mag_amd64_cap(AVX512_VPOPCNTDQ);
#endif
#ifdef __AVX512BF16__
    caps|=mag_amd64_cap(AVX512_BF16);
#endif
#ifdef __AVX512VP2INTERSECT__
    caps|=mag_amd64_cap(AVX512_VP2INTERSECT);
#endif
#ifdef __AVX512FP16__
    caps|=mag_amd64_cap(AVX512_FP16);
#endif

#ifdef __AMX_TILE__
    caps|=mag_amd64_cap(AMX_TILE);
#endif
#ifdef __AMX_INT8__
    caps|=mag_amd64_cap(AMX_INT8);
#endif
#ifdef __AMX_BF16__
    caps|=mag_amd64_cap(AMX_BF16);
#endif
#ifdef __AMX_FP16__
    caps|=mag_amd64_cap(AMX_FP16);
#endif
#ifdef __AMX_TRANSPOSE__
    caps|=mag_amd64_cap(AMX_TRANSPOSE);
#endif
#ifdef __AMX_TF32__
    caps|=mag_amd64_cap(AMX_TF32);
#endif
#ifdef __AMX_AVX512__
    caps|=mag_amd64_cap(AMX_AVX512);
#endif
#ifdef __AMX_MOVRS__
    caps|=mag_amd64_cap(AMX_MOVRS);
#endif
#ifdef __AMX_FP8__
    caps|=mag_amd64_cap(AMX_FP8);
#endif


#ifdef __BMI__
    caps|=mag_amd64_cap(BMI1);
#endif
#ifdef __BMI2__
    caps|=mag_amd64_cap(BMI2);
#endif

#ifdef __GFNI__
    caps|=mag_amd64_cap(GFNI);
#endif
#ifdef __APXF__
    caps|=mag_amd64_cap(APX_F);
#endif

    return caps;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

mag_arm64_cap_bitset_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST(void) {
    mag_arm64_cap_bitset_t caps = 0;
#ifdef __ARM_NEON
    caps|=mag_arm64_cap(NEON);
#endif
#ifdef __ARM_FEATURE_DOTPROD
    caps|=mag_arm64_cap(DOTPROD);
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
    caps|=mag_arm64_cap(I8MM);
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    caps|=mag_arm64_cap(F16VECTOR);
    caps|=mag_arm64_cap(F16SCALAR);
    caps|=mag_arm64_cap(F16CVT);
#elif defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    caps|=mag_arm64_cap(F16SCALAR);
    caps|=mag_arm64_cap(F16CVT);
#endif
#ifdef __ARM_FEATURE_BF16
    caps|=mag_arm64_cap(BF16);
#endif
#ifdef __ARM_FEATURE_SVE
    caps|=mag_arm64_cap(SVE);
#endif
#ifdef __ARM_FEATURE_SVE2
    caps|=mag_arm64_cap(SVE2);
#endif
    return caps;
}

#endif
