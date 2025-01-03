/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "magnetron_internal.h"

#include <math.h>
#include <stdio.h>

#if defined(__APPLE__) && defined(MAG_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

typedef struct mag_cpu_thread_local_ctx_t {
    mag_ctx_t* ctx;
    int64_t thread_num;
    int64_t thread_idx;
} mag_cpu_thread_local_ctx_t;

#define mag_f32p(t) ((const float*)(t)->storage.base)
#define mag_f32p_mut(t) ((float*)(t)->storage.base)

#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

static float32x4_t mag_simd_expf(float32x4_t x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0  */
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

static float32x4_t mag_simd_tanh(float32x4_t x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t neg_one = vdupq_n_f32(-1.0f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    float32x4_t a = vmulq_f32(neg_two, x);
    float32x4_t b = mag_simd_expf(a);
    float32x4_t c = vaddq_f32(one, b);
    float32x4_t inv = vrecpeq_f32(c);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
    return vaddq_f32(neg_one, vmulq_f32(two, inv));
}

static void mag_simd_sincos(float32x4_t x, float32x4_t *osin, float32x4_t *ocos) {
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

#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)

static __m512 mag_simd_expf(const __m512 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
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

static __m512 mag_simd_tanh(__m512 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_one = _mm512_set1_ps(-1.0f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    __m512 a = _mm512_mul_ps(neg_two, x);
    __m512 b = mag_simd_expf(a);
    __m512 c = _mm512_add_ps(one, b);
    __m512 inv = _mm512_rcp14_ps(c);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm512_fmadd_ps(two, inv, neg_one);
}

#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)

static __m256 mag_simd_expf(const __m256 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
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

static __m256 mag_simd_tanh(__m256 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    __m256 a = _mm256_mul_ps(neg_two, x);
    __m256 b = mag_simd_expf(a);
    __m256 c = _mm256_add_ps(one, b);
    __m256 inv = _mm256_rcp_ps(c);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm256_fmadd_ps(two, inv, neg_one);
}

#elif MAG_APPROXMATH && defined(__SSE2__)
static __m128 mag_simd_expf(const __m128 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
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

static __m128 mag_simd_tanh(__m128 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    __m128 one = _mm_set1_ps(1.0f);
    __m128 neg_one = _mm_set1_ps(-1.0f);
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

static void mag_simd_sincos(__m128 x, __m128 *osin, __m128 *ocos) {
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
    __m128i tmp = _mm_cmpeq_epi32(emm2, _mm_set1_epi32(4));
    sign_mask_sin = _mm_xor_si128(sign_mask_sin, tmp);
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
    y1 = _mm_add_ps(y1, _mm_set1_ps(1.0f));
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

static void MAG_HOTPROC mag_vadd_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
#ifdef MAG_ACCELERATE
    vDSP_vadd(y, 1, x, 1, o, 1, n);
#else
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] + y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vsub_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
#ifdef MAG_ACCELERATE
    vDSP_vsub(y, 1, x, 1, o, 1, n);
#else
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] - y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vmul_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
#ifdef MAG_ACCELERATE
    vDSP_vmul(y, 1, x, 1, o, 1, n);
#else
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vdiv_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
#ifdef MAG_ACCELERATE
    vDSP_vdiv(y, 1, x, 1, o, 1, n);
#else
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / y[i];
    }
#endif
}

static void MAG_HOTPROC mag_vadds_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] + y;
    }
}

static void MAG_HOTPROC mag_vsubs_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] - y;
    }
}

static void MAG_HOTPROC mag_vmuls_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * y;
    }
}

static void MAG_HOTPROC mag_vdivs_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / y;
    }
}

static float MAG_UNUSED MAG_HOTPROC mag_vdot_f32(
    const int64_t n,
    const float* const x,
    const float* const y
) {
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    const int64_t k = n & -16;
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
    float sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
    for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
    return sum;
#elif defined(__AVX512F__) && defined(__FMA__)
    const int64_t k = n & -64;
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
    float sum = _mm512_reduce_add_ps(*acc);
    for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
    return sum;
#elif defined(__AVX__) && defined(__FMA__)
    const int64_t k = n & -32;
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
    float sum = _mm_cvtss_f32(v0);
    for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
    return sum;
#elif defined(__SSE2__)
    const int64_t k = n & -16;
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
        float sum = _mm_cvtss_f32(*acc);
    #else
        __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(*acc, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);
    #endif
    for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
    return sum;
#else
    double r = 0.0;
    for (int64_t i=0; i < n; ++i) r += (double)x[i] * (double)y[i];
    return (float)r;
#endif
}

static double MAG_HOTPROC mag_vsum_f64_f32( /* Σx. */
    const int64_t n,
    const float* const x
) {
#ifdef MAG_ACCELERATE
    float sum;
    vDSP_sve(x, 1, &sum, n);
    return (double)sum;
#else
    double sum = 0.0;
    for (int64_t i=0; i < n; ++i) {
        sum += (double)x[i];
    }
    return sum;
#endif
}

static float MAG_HOTPROC mag_vmin_f32( /* min x */
    const int64_t n,
    const float* const x
) {
    float min = INFINITY;
    for (int64_t i=0; i < n; ++i) {
        min = fminf(min, x[i]);
    }
    return min;
}

static float MAG_HOTPROC mag_vmax_f32( /* max x */
    const int64_t n,
    const float* const x
) {
    float min = -INFINITY;
    for (int64_t i=0; i < n; ++i) {
        min = fmaxf(min, x[i]);
    }
    return min;
}

static void MAG_HOTPROC mag_vabs_f32( /* o = |x| */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = fabsf(x[i]);
    }
}

static void MAG_HOTPROC mag_vneg_f32( /* o = -x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = -x[i];
    }
}

static void MAG_HOTPROC mag_vlog_f32( /* o = log x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1);
    for (; i+3 < n; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        xi = vmaxq_f32(xi, vdupq_n_f32(0));
        uint32x4_t invalid_mask = vcleq_f32(xi, vdupq_n_f32(0));
        int32x4_t ux = vreinterpretq_s32_f32(xi);
        int32x4_t emm0 = vshrq_n_s32(ux, 23);
        ux = vandq_s32(ux, vdupq_n_s32(~0x7f800000u));
        ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
        xi = vreinterpretq_f32_s32(ux);
        emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
        float32x4_t e = vcvtq_f32_s32(emm0);
        e = vaddq_f32(e, one);
        uint32x4_t mask = vcltq_f32(xi, vdupq_n_f32(0.707106781186547524f));
        float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(xi), mask));
        xi = vsubq_f32(xi, one);
        e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
        xi = vaddq_f32(xi, tmp);
        float32x4_t z = vmulq_f32(xi, xi);
        float32x4_t y = vdupq_n_f32(7.0376836292e-2f);
        y = vmlaq_f32(vdupq_n_f32(-1.1514610310e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(1.1676998740e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-1.2420140846e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(1.4249322787e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-1.6668057665e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(2.0000714765e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(-2.4999993993e-1f), y, xi);
        y = vmlaq_f32(vdupq_n_f32(3.3333331174e-1f), y, xi);
        y = vmulq_f32(y, xi);
        y = vmulq_f32(y, z);
        y = vmlaq_f32(y, e, vdupq_n_f32(-2.12194440e-4f));
        y = vmlsq_f32(y, z, vdupq_n_f32(0.5f));
        xi = vaddq_f32(xi, y);
        xi = vmlaq_f32(xi, e, vdupq_n_f32(0.693359375f));
        xi = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(xi), invalid_mask));
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
#error TODO
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
#error TODO
#elif MAG_APPROXMATH && defined(__SSE2__)
    const __m128 one = _mm_set1_ps(1.0f);
    for (; i+3 < n; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        xi = _mm_max_ps(xi, _mm_set1_ps(0.0f));
        __m128 invalid_mask = _mm_cmple_ps(xi, _mm_set1_ps(0.0f));
        __m128i ux = _mm_castps_si128(xi);
        __m128i emm0 = _mm_srli_epi32(ux, 23);
        ux = _mm_and_si128(ux, _mm_set1_epi32(~0x7f800000u));
        ux = _mm_or_si128(ux, _mm_castps_si128(_mm_set1_ps(0.5f)));
        xi = _mm_castsi128_ps(ux);
        emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
        __m128 e = _mm_cvtepi32_ps(emm0);
        e = _mm_add_ps(e, one);
        __m128 mask = _mm_cmplt_ps(xi, _mm_set1_ps(0.707106781186547524f));
        __m128 tmp = _mm_and_ps(xi, mask);
        xi = _mm_sub_ps(xi, one);
        e = _mm_sub_ps(e, _mm_and_ps(one, mask));
        xi = _mm_add_ps(xi, tmp);
        __m128 z = _mm_mul_ps(xi, xi);
        __m128 y = _mm_set1_ps(7.0376836292e-2f);
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.1514610310e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(1.1676998740e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.2420140846e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(1.4249322787e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-1.6668057665e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(2.0000714765e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(-2.4999993993e-1f));
        y = _mm_add_ps(_mm_mul_ps(y, xi), _mm_set1_ps(3.3333331174e-1f));
        y = _mm_mul_ps(y, xi);
        y = _mm_mul_ps(y, z);
        y = _mm_add_ps(_mm_mul_ps(e, _mm_set1_ps(-2.12194440e-4f)), y);
        y = _mm_sub_ps(y, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
        xi = _mm_add_ps(xi, y);
        xi = _mm_add_ps(_mm_mul_ps(e, _mm_set1_ps(0.693359375f)), xi);
        xi = _mm_or_ps(xi, invalid_mask);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < n; ++i) { /* Process leftovers scalar-wise */
        o[i] = logf(x[i]);
    }
}

static void MAG_HOTPROC mag_vsqr_f32( /* o = x² */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = x[i]*x[i];
}

static void MAG_HOTPROC mag_vsqrt_f32( /* o = √x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = sqrtf(x[i]);
    }
}

static void MAG_HOTPROC mag_vsin_f32( /* o = sin x */
    const int64_t n,
    float* const o,
    const float* const x
) {
       int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < n; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        float32x4_t ocos;
        mag_simd_sincos(xi, &xi, &ocos);
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
#error TODO
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
#error TODO
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < n; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        __m128 ocos;
        mag_simd_sincos(xi, &xi, &ocos);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < n; ++i) { /* Process leftovers scalar-wise */
        o[i] = sinf(x[i]);
    }
}

static void MAG_HOTPROC mag_vcos_f32( /* o = cos x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < n; i += 4) {
        float32x4_t xi = vld1q_f32(x+i);
        float32x4_t osin;
        mag_simd_sincos(xi, &osin, &xi);
        vst1q_f32(o+i, xi);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
#error TODO
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
#error TODO
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < n; i += 4) {
        __m128 xi = _mm_loadu_ps(x+i);
        __m128 osin;
        mag_simd_sincos(xi, &osin, &xi);
        _mm_storeu_ps(o+i, xi);
    }
#endif
    for (; i < n; ++i) { /* Process leftovers scalar-wise */
        o[i] = cosf(x[i]);
    }
}

static void MAG_HOTPROC mag_vstep_f32( /* Heaviside step function. */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] >= 0.0f ? 1.0f : 0.0f;
    }
}

static void MAG_HOTPROC mag_vsoftmax_f32( /* softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < n; i += 4) {
        vst1q_f32(o+i, mag_simd_expf(vld1q_f32(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i+15 < n; i += 16) {
        _mm512_storeu_ps(o+i, mag_simd_expf(_mm512_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    for (; i+7 < n; i += 8) {
        _mm256_storeu_ps(o+i, mag_simd_expf(_mm256_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < n; i += 4) {
        _mm_storeu_ps(o+i, mag_simd_expf(_mm_loadu_ps(x+i)));
    }
#endif
    for (; i < n; ++i) o[i] = expf(x[i]); /* Process leftovers scalar-wise */
}

static void MAG_HOTPROC mag_vsoftmax_dv_f32( /* softmax' = softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    mag_vsoftmax_f32(n, o, x);
}

static void MAG_HOTPROC mag_vsigmoid_f32( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i+3 < n; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t neg_x = vsubq_f32(zero, xx);
        float32x4_t exp_neg_x = mag_simd_expf(neg_x);
        float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
        vst1q_f32(o+i, vdivq_f32(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    for (; i+15 < n; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 neg_x = _mm512_sub_ps(zero, xx);
        __m512 exp_neg_x = mag_simd_expf(neg_x);
        __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
        _mm512_storeu_ps(o+i, _mm512_div_ps(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_setzero_ps();
    for (; i+7 < n; i += 8) {
        __m256 xx = _mm256_loadu_ps(x+i);
        __m256 neg_x = _mm256_sub_ps(zero, xx);
        __m256 exp_neg_x = mag_simd_expf(neg_x);
        __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
        _mm256_storeu_ps(o+i, _mm256_div_ps(one, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_setzero_ps();
    for (; i+3 < n; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 neg_x = _mm_sub_ps(zero, xx);
        __m128 exp_neg_x = mag_simd_expf(neg_x);
        __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
        _mm_storeu_ps(o+i, _mm_div_ps(one, one_plus_exp_neg_x));
    }
#endif
    for (; i < n; ++i) o[i] = 1.0f / (1.0f + expf(-x[i])); /* Process leftovers scalar-wise */
}

static void MAG_HOTPROC mag_vsigmoid_dv_f32( /* σ' : ℝ -> (0, 1), x |-> x * (1-x) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * (1.0f - x[i]);
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_f32( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
    }
}

static void MAG_HOTPROC mag_vsilu_f32( /* silu : ℝ -> ℝ, x |-> x/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i+3 < n; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t neg_x = vsubq_f32(zero, xx);
        float32x4_t exp_neg_x = mag_simd_expf(neg_x);
        float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
        vst1q_f32(o+i, vdivq_f32(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 one = _mm512_set1_ps(1);
    __m512 zero = _mm512_setzero_ps();
    for (; i+15 < n; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 neg_x = _mm512_sub_ps(zero, xx);
        __m512 exp_neg_x = mag_simd_expf(neg_x);
        __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
        _mm512_storeu_ps(o+i, _mm512_div_ps(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 one = _mm256_set1_ps(1);
    __m256 zero = _mm256_setzero_ps();
    for (; i+7 < n; i += 8) {
        const __m256 xx = _mm256_loadu_ps(x+i);
        __m256 neg_x = _mm256_sub_ps(zero, xx);
        __m256 exp_neg_x = mag_simd_expf(neg_x);
        __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
        _mm256_storeu_ps(o+i, _mm256_div_ps(xx, one_plus_exp_neg_x));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_setzero_ps();
    for (; i+3 < n; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 neg_x = _mm_sub_ps(zero, xx);
        __m128 exp_neg_x = mag_simd_expf(neg_x);
        __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
        _mm_storeu_ps(o+i, _mm_div_ps(xx, one_plus_exp_neg_x));
    }
#endif
    for (; i < n; ++i) {
        o[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void MAG_HOTPROC mag_vsilu_dv_f32( /* silu' : ℝ -> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        mag_panic("NYI!");
    }
}

static void MAG_HOTPROC mag_vtanh_f32( /* tanh : ℝ -> (-1, 1), x |-> tanh x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < n; i += 4) {
        vst1q_f32(o+i, mag_simd_tanh(vld1q_f32(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i+15 < n; i += 16) {
        _mm512_storeu_ps(o+i, mag_simd_tanh(_mm512_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    for (; i+7 < n; i += 8) {
        _mm256_storeu_ps(o+i, mag_simd_tanh(_mm256_loadu_ps(x+i)));
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    for (; i+3 < n; i += 4) {
        _mm_storeu_ps(o+i, mag_simd_tanh(_mm_loadu_ps(x+i)));
    }
#endif
    for (; i < n; ++i) {
        o[i] = tanhf(x[i]);
    }
}

static void MAG_HOTPROC mag_vtanh_dv_f32( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float cx = coshf(x[i]);
        o[i] = 1.0f / (cx*cx);
    }
}

static void MAG_HOTPROC mag_vrelu_f32( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = mag_xmax(x[i], 0.0f);
    }
}

static void MAG_HOTPROC mag_vrelu_dv_f32( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] <= 0.0f ? 0.0f : 1.0f; /* relu' is mathematically undefined for x = 0, but we return 0 in this case. */
    }
}

static void MAG_HOTPROC mag_vgelu_f32( /* gelu : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
#if MAG_APPROXMATH && (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t coeff1 = vdupq_n_f32(0.79788456080286535587989211986876f);
    float32x4_t coeff2 = vdupq_n_f32(MAG_GELU_COEFF);
    for (; i+3 < n; i += 4) {
        float32x4_t xx = vld1q_f32(x+i);
        float32x4_t a = vaddq_f32(one, vmulq_f32(coeff2, vmulq_f32(xx, xx)));
        float32x4_t b = vaddq_f32(one, mag_simd_tanh(vmulq_f32(coeff1, vmulq_f32(xx, a))));
        float32x4_t c = vmulq_f32(half, vmulq_f32(xx, b));
        vst1q_f32(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 coeff1 = _mm512_set1_ps(0.79788456080286535587989211986876f);
    __m512 coeff2 = _mm512_set1_ps(MAG_GELU_COEFF);
    for (; i+15 < n; i += 16) {
        __m512 xx = _mm512_loadu_ps(x+i);
        __m512 a = _mm512_fmadd_ps(coeff2, _mm512_mul_ps(xx, xx), one);
        __m512 b = _mm512_add_ps(one, mag_simd_tanh(_mm512_mul_ps(coeff1, _mm512_mul_ps(xx, a))));
        __m512 c = _mm512_mul_ps(half, _mm512_mul_ps(xx, b));
        _mm512_storeu_ps(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__AVX2__) && defined(__FMA__)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 coeff1 = _mm256_set1_ps(0.79788456080286535587989211986876f);
    __m256 coeff2 = _mm256_set1_ps(MAG_GELU_COEFF);
    for (; i+7 < n; i += 8) {
        __m256 xx = _mm256_loadu_ps(x+i);
        __m256 a = _mm256_fmadd_ps(coeff2, _mm256_mul_ps(xx, xx), one);
        __m256 b = _mm256_add_ps(one, mag_simd_tanh(_mm256_mul_ps(coeff1, _mm256_mul_ps(xx, a))));
        __m256 c = _mm256_mul_ps(half, _mm256_mul_ps(xx, b));
        _mm256_storeu_ps(o+i, c);
    }
#elif MAG_APPROXMATH && defined(__SSE2__)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 one = _mm_set1_ps(1.0f);
    __m128 coeff1 = _mm_set1_ps(0.79788456080286535587989211986876f);
    __m128 coeff2 = _mm_set1_ps(MAG_GELU_COEFF);
    for (; i+3 < n; i += 4) {
        __m128 xx = _mm_loadu_ps(x+i);
        __m128 a = _mm_add_ps(one, _mm_mul_ps(coeff2, _mm_mul_ps(xx, xx)));
        __m128 b = _mm_add_ps(one, mag_simd_tanh(_mm_mul_ps(coeff1, _mm_mul_ps(xx, a))));
        __m128 c = _mm_mul_ps(half, _mm_mul_ps(xx, b));
        _mm_storeu_ps(o+i, c);
    }
#endif
    for (; i < n; ++i) {
        o[i] = 0.5f*x[i]*(1.0f + tanhf(0.79788456080286535587989211986876f*x[i]*(1.0f + MAG_GELU_COEFF*x[i]*x[i])));
    }
}

static void MAG_HOTPROC mag_vgelu_dv_f32( /* gelu' : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        mag_panic("NYI"); /* TODO */
    }
}

static void mag_blas_nop(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs
) {
    (void)bci;
    (void)r;
    (void)inputs;
}

static void mag_blas_clone(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs
) {
    const mag_tensor_t* const x = inputs[0];
    mag_assert2(mag_tensor_is_shape_eq(x, r));
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    memcpy(b_r, b_x, mag_tensor_data_size(r));
}

static void MAG_HOTPROC mag_blas_mean_f32( /* Σx/n Arithmetic mean */
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    double sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_f32(
                                x_d0,
                                p_x
                        );
                    }
                }
            }
        }
    }
    sum /= (double)x->numel;
    *b_r = (float)sum;
}

static void MAG_HOTPROC mag_blas_min_f32( /* min x */
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    float min = INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        min = fminf(mag_vmin_f32(x_d0, p_x), min);
                    }
                }
            }
        }
    }
    *b_r = min;
}

static void MAG_HOTPROC mag_blas_max_f32( /* max x */
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    float max = -INFINITY;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        max = fmaxf(mag_vmax_f32(x_d0, p_x), max);
                    }
                }
            }
        }
    }
    *b_r = max;
}

static void MAG_HOTPROC mag_blas_sum_f32( /* Σx/n Arithmetic mean */
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    double sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5;
                        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
                        sum += mag_vsum_f64_f32(x_d0, p_x);
                    }
                }
            }
        }
    }
    *b_r = (float)sum;
}

static void MAG_HOTPROC mag_blas_abs_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vabs_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_neg_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vneg_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_log_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vlog_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_sqr_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsqr_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_sqrt_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsqrt_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_sin_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsin_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_cos_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vcos_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_step_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vstep_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_softmax_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsoftmax_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_softmax_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsoftmax_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsigmoid_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_sigmoid_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsigmoid_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_hard_sigmoid_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vhard_sigmoid_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_silu_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsilu_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_silu_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsilu_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_tanh_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vtanh_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_tanh_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vtanh_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_relu_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vrelu_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_relu_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vrelu_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_gelu_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vgelu_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_gelu_dv_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vgelu_dv_f32(cc, p_r, p_x);
    }
}

static void MAG_HOTPROC mag_blas_add_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const float* const b_y = mag_f32p(y);
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == 1) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            const float* const p_y = b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5;
            mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                mag_bnd_chk(pp_r, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(pp_x, b_x, mag_tensor_data_size(x));
                mag_vadd_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5; /* Compute result ptr. */
                mag_bnd_chk(p_r+i, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(p_x+i, b_x, mag_tensor_data_size(x));
                mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
                p_r[i] = p_x[i] + *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void MAG_HOTPROC mag_blas_sub_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const float* const b_y = mag_f32p(y);
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == 1) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            const float* const p_y = b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5;
            mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                mag_bnd_chk(pp_r, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(pp_x, b_x, mag_tensor_data_size(x));
                mag_vsub_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5; /* Compute result ptr. */
                mag_bnd_chk(p_r+i, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(p_x+i, b_x, mag_tensor_data_size(x));
                mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
                p_r[i] = p_x[i] - *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void MAG_HOTPROC mag_blas_mul_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const float* const b_y = mag_f32p(y);
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == 1) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            const float* const p_y = b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5;
            mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                mag_bnd_chk(pp_r, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(pp_x, b_x, mag_tensor_data_size(x));
                mag_vmul_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5; /* Compute result ptr. */
                mag_bnd_chk(p_r+i, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(p_x+i, b_x, mag_tensor_data_size(x));
                mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
                p_r[i] = p_x[i] * *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void MAG_HOTPROC mag_blas_div_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const float* const b_y = mag_f32p(y);
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == 1) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            const float* const p_y = b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5;
            mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                mag_bnd_chk(pp_r, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(pp_x, b_x, mag_tensor_data_size(x));
                mag_vdiv_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5;
            const float* const p_x = b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5;
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5; /* Compute result ptr. */
                mag_bnd_chk(p_r+i, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(p_x+i, b_x, mag_tensor_data_size(x));
                mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
                p_r[i] = p_x[i] / *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void MAG_HOTPROC mag_blas_adds_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    const float xi = r->op_params->x.f32;
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vadds_f32(cc, p_r, p_x, xi);
    }
}

static void MAG_HOTPROC mag_blas_subs_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    const float xi = r->op_params->x.f32;
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vsubs_f32(cc, p_r, p_x, xi);
    }
}

static void MAG_HOTPROC mag_blas_muls_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    const float xi = r->op_params->x.f32;
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vmuls_f32(cc, p_r, p_x, xi);
    }
}

static void MAG_HOTPROC mag_blas_divs_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const mag_tensor_t* const x = inputs[0];
    const float xi = r->op_params->x.f32;
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_s, strides);
    const int64_t rc = mag_tensor_num_rows(x);
    const int64_t cc = mag_tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = b_r + ri*r_s1;
        const float* const p_x = b_x + ri*x_s1;
        mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
        mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
        mag_vdivs_f32(cc, p_r, p_x, xi);
    }
}

#if 0 /* Naive matrix multiplication, but no broadcasting support. */
static void MAG_HOTPROC mag_blas_matmul_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    for (int64_t i3=0; i3 < r_d3; ++i3) {
        for (int64_t i2=0; i2 < r_d2; ++i2) {
            for (int64_t i1=0; i1 < r_d1; ++i1) {
                for (int64_t i0=0; i0 < r_d1; ++i0) {
                    double sum = 0.0;
                    for (int64_t k=0; k < x_d0; ++k) {
                        const float* const p_x = (const float*)(b_x + k*x_s0 + i0*x_s1 + i2*x_s2 + i3*x_s3);
                        const float* const p_y = (const float*)(b_y + i1*y_s0 + k*y_s1 + i2*y_s2 + i3*y_s3);
                        mag_bnd_chk(p_x, b_x, x->buf_size);
                        mag_bnd_chk(p_y, b_y, y->buf_size);
                        sum += (double)(*p_x**p_y);
                    }
                    float* const p_r = (float*)(b_r + i1*r_s0 + i0*r_s1 + i2*r_s2 + i3*r_s3);
                    mag_bnd_chk(p_r, b_r, r->buf_size);
                    *p_r = (float)sum;
                }
            }
        }
    }
}
#endif

/*
** Matrix multiplication.
** R = A x B
*/
static void MAG_HOTPROC mag_blas_matmul_f32(
    const mag_cpu_thread_local_ctx_t* const bci,
    mag_tensor_t* const r,
    const mag_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const mag_tensor_t* const x = inputs[0];
    const mag_tensor_t* const y = inputs[1];
    float* const b_r = mag_f32p_mut(r);
    const float* const b_x = mag_f32p(x);
    const float* const b_y = mag_f32p(y);
    mag_load_local_storage_group(r, r_d, shape);
    mag_load_local_storage_group(r, r_s, strides);
    mag_load_local_storage_group(x, x_d, shape);
    mag_load_local_storage_group(x, x_s, strides);
    mag_load_local_storage_group(y, y_d, shape);
    mag_load_local_storage_group(y, y_s, strides);
    mag_assert2(x_d2 == 1 && x_d3 == 1);
    mag_assert2(y_d2 == 1 && y_d3 == 1);
    const int64_t rows = x_d0;
    const int64_t cols = x_d1;
    const int64_t inners = y_d1;
    for (int64_t i=0; i < rows; ++i) {
        for (int64_t k=0; k < cols; ++k) {
            const float* const p_x = b_x + x_d1*i + k;
            mag_bnd_chk(p_x, b_x, mag_tensor_data_size(x));
            for (int64_t j = 0; j < inners; ++j) {
                float* const p_r = b_r + r_d1*i + j;
                const float* const p_y = b_y + y_d1*k + j;
                mag_bnd_chk(p_r, b_r, mag_tensor_data_size(r));
                mag_bnd_chk(p_y, b_y, mag_tensor_data_size(y));
                *p_r += *p_x * *p_y;
            }
        }
    }
}

static void (*mag_blas_dispatch_table_forward[MAG_OP__NUM])(const mag_cpu_thread_local_ctx_t*, mag_tensor_t*, const mag_tensor_t**) = {
    [MAG_OP_NOP] = &mag_blas_nop, /* No operation */
    [MAG_OP_CLONE] = &mag_blas_clone,
    [MAG_OP_VIEW] = &mag_blas_nop, /* View is a no-op */
    [MAG_OP_TRANSPOSE] = &mag_blas_nop, /* Transpose is a runtime no-op */
    [MAG_OP_PERMUTE] = &mag_blas_nop, /* Transpose is a runtime no-op */
    [MAG_OP_MEAN] = &mag_blas_mean_f32,
    [MAG_OP_MIN] = &mag_blas_min_f32,
    [MAG_OP_MAX] = &mag_blas_max_f32,
    [MAG_OP_SUM] = &mag_blas_sum_f32,
    [MAG_OP_ABS] = &mag_blas_abs_f32,
    [MAG_OP_NEG] = &mag_blas_neg_f32,
    [MAG_OP_LOG] = &mag_blas_log_f32,
    [MAG_OP_SQR] = &mag_blas_sqr_f32,
    [MAG_OP_SQRT] = &mag_blas_sqrt_f32,
    [MAG_OP_SIN] = &mag_blas_sin_f32,
    [MAG_OP_COS] = &mag_blas_cos_f32,
    [MAG_OP_STEP] = &mag_blas_step_f32,
    [MAG_OP_SOFTMAX] = &mag_blas_softmax_f32,
    [MAG_OP_SOFTMAX_DV] = &mag_blas_softmax_dv_f32,
    [MAG_OP_SIGMOID] = &mag_blas_sigmoid_f32,
    [MAG_OP_SIGMOID_DV] = &mag_blas_sigmoid_dv_f32,
    [MAG_OP_HARD_SIGMOID] = &mag_blas_hard_sigmoid_f32,
    [MAG_OP_SILU] = &mag_blas_silu_f32,
    [MAG_OP_SILU_DV] = &mag_blas_silu_dv_f32,
    [MAG_OP_TANH] = &mag_blas_tanh_f32,
    [MAG_OP_TANH_DV] = &mag_blas_tanh_dv_f32,
    [MAG_OP_RELU] = &mag_blas_relu_f32,
    [MAG_OP_RELU_DV] = &mag_blas_relu_dv_f32,
    [MAG_OP_GELU] = &mag_blas_gelu_f32,
    [MAG_OP_GELU_DV] = &mag_blas_gelu_dv_f32,
    [MAG_OP_ADD] = &mag_blas_add_f32,
    [MAG_OP_SUB] = &mag_blas_sub_f32,
    [MAG_OP_MUL] = &mag_blas_mul_f32,
    [MAG_OP_DIV] = &mag_blas_div_f32,
    [MAG_OP_ADDS] = &mag_blas_adds_f32,
    [MAG_OP_SUBS] = &mag_blas_subs_f32,
    [MAG_OP_MULS] = &mag_blas_muls_f32,
    [MAG_OP_DIVS] = &mag_blas_divs_f32,
    [MAG_OP_MATMUL] = &mag_blas_matmul_f32
};

static void (*mag_blas_dispatch_table_backward[MAG_OP__NUM])(const mag_cpu_thread_local_ctx_t*, mag_tensor_t*, const mag_tensor_t**) = {
    [MAG_OP_NOP] = &mag_blas_nop, /* No operation */
    [MAG_OP_CLONE] = &mag_blas_clone,
    [MAG_OP_VIEW] = &mag_blas_nop, /* View is a no-op */
    [MAG_OP_TRANSPOSE] = &mag_blas_nop, /* Transpose is a runtime no-op */
    [MAG_OP_PERMUTE] = &mag_blas_nop, /* Transpose is a runtime no-op */
    [MAG_OP_MEAN] = &mag_blas_mean_f32,
    [MAG_OP_MIN] = &mag_blas_min_f32,
    [MAG_OP_MAX] = &mag_blas_max_f32,
    [MAG_OP_SUM] = &mag_blas_sum_f32,
    [MAG_OP_ABS] = &mag_blas_abs_f32,
    [MAG_OP_NEG] = &mag_blas_neg_f32,
    [MAG_OP_LOG] = &mag_blas_log_f32,
    [MAG_OP_SQR] = &mag_blas_sqr_f32,
    [MAG_OP_SQRT] = &mag_blas_sqrt_f32,
    [MAG_OP_SIN] = &mag_blas_sin_f32,
    [MAG_OP_COS] = &mag_blas_cos_f32,
    [MAG_OP_STEP] = &mag_blas_step_f32,
    [MAG_OP_SOFTMAX] = &mag_blas_softmax_f32,
    [MAG_OP_SOFTMAX_DV] = &mag_blas_softmax_dv_f32,
    [MAG_OP_SIGMOID] = &mag_blas_sigmoid_f32,
    [MAG_OP_SIGMOID_DV] = &mag_blas_sigmoid_dv_f32,
    [MAG_OP_HARD_SIGMOID] = &mag_blas_hard_sigmoid_f32,
    [MAG_OP_SILU] = &mag_blas_silu_f32,
    [MAG_OP_SILU_DV] = &mag_blas_silu_dv_f32,
    [MAG_OP_TANH] = &mag_blas_tanh_f32,
    [MAG_OP_TANH_DV] = &mag_blas_tanh_dv_f32,
    [MAG_OP_RELU] = &mag_blas_relu_f32,
    [MAG_OP_RELU_DV] = &mag_blas_relu_dv_f32,
    [MAG_OP_GELU] = &mag_blas_gelu_f32,
    [MAG_OP_GELU_DV] = &mag_blas_gelu_dv_f32,
    [MAG_OP_ADD] = &mag_blas_add_f32,
    [MAG_OP_SUB] = &mag_blas_sub_f32,
    [MAG_OP_MUL] = &mag_blas_mul_f32,
    [MAG_OP_DIV] = &mag_blas_div_f32,
    [MAG_OP_ADDS] = &mag_blas_adds_f32,
    [MAG_OP_SUBS] = &mag_blas_subs_f32,
    [MAG_OP_MULS] = &mag_blas_muls_f32,
    [MAG_OP_DIVS] = &mag_blas_divs_f32,
    [MAG_OP_MATMUL] = &mag_blas_matmul_f32
};

static MAG_HOTPROC void mag_cpu_exec_fwd(mag_compute_device_t* dvc, mag_tensor_t* root) {
    mag_cpu_thread_local_ctx_t* bci = (mag_cpu_thread_local_ctx_t*)(dvc+1); // todo
    mag_blas_dispatch_table_forward[root->op](bci, root, (const mag_tensor_t**)root->op_inputs);
}

static MAG_HOTPROC void mag_cpu_exec_bwd(mag_compute_device_t* dvc, mag_tensor_t* root) {
    mag_cpu_thread_local_ctx_t* bci = (mag_cpu_thread_local_ctx_t*)(dvc+1); // todo
    mag_blas_dispatch_table_backward[root->op](bci, root, (const mag_tensor_t**)root->op_inputs);
}

static void mag_cpu_buf_set(mag_storage_buffer_t* sto, size_t offs, uint8_t x) {
    mag_assert2(sto->base+offs <= sto->base+sto->size);
    memset((void*)(sto->base+offs), x, sto->size-offs); /* On CPU just plain old memset with offset. */
}

static void mag_cpu_buf_cpy_host_device(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t n) {
    mag_assert2(sto->base+offs+n <= sto->base+sto->size);
    memcpy((void*)(sto->base+offs), src, n); /* On CPU just plain old memcpy with offset. */
}

static void mag_cpu_buf_cpy_device_host(mag_storage_buffer_t* sto, size_t offs, void* dst, size_t n) {
    mag_assert2(sto->base+offs+n <= sto->base+sto->size);
    memcpy(dst, (void*)(sto->base+offs), n); /* On CPU just plain old memcpy with offset. */
}

static void mag_cpu_alloc_storage(mag_compute_device_t* host, mag_storage_buffer_t* out, size_t size, size_t align) {
    mag_assert2(size);
    void* block = mag_alloc_aligned(size, align);
    memset(block, 0, size);
    *out = (mag_storage_buffer_t){ /* Set up storage buffer. */
        .base = (uintptr_t)block,
        .size = size,
        .alignment = align,
        .host = host,
        .set = &mag_cpu_buf_set,
        .cpy_host_device = &mag_cpu_buf_cpy_host_device,
        .cpy_device_host = &mag_cpu_buf_cpy_device_host
    };
}

static void mag_cpu_free_storage(mag_compute_device_t* dvc, mag_storage_buffer_t* buf) {
    mag_free_aligned((void*)buf->base);
    memset(buf, 0, sizeof(*buf)); /* Set to zero. */
}

static mag_compute_device_t* mag_cpu_init_interface(mag_ctx_t* ctx) {
    mag_compute_device_t* dvc = (*mag_alloc)(NULL, sizeof(*dvc));
    *dvc = (mag_compute_device_t){
        .name = "CPU",
        .impl = NULL,
        .is_async = false,
        .type = MAG_COMPUTE_DEVICE_TYPE_CPU,
        .eager_exec_fwd = &mag_cpu_exec_fwd,
        .eager_exec_bwd = &mag_cpu_exec_bwd,
        .alloc_storage = &mag_cpu_alloc_storage,
        .free_storage = &mag_cpu_free_storage
    };
    snprintf(dvc->name, sizeof(dvc->name), "%s - %s", mag_device_type_get_name(dvc->type), ctx->sys.cpu_name);
    return dvc;
}

static void mag_cpu_release_interface(mag_compute_device_t* ctx) {
    (*mag_alloc)(ctx, 0);
}

mag_compute_device_t* mag_init_device_cpu(mag_ctx_t* ctx, uint32_t num_threads) {
    num_threads = num_threads ? num_threads : mag_xmax(1, ctx->sys.cpu_virtual_cores);
    mag_compute_device_t* dvc = mag_cpu_init_interface(ctx);
    return dvc;
}

void mag_destroy_device_cpu(mag_compute_device_t* dvc) {
    mag_cpu_release_interface(dvc);
}
