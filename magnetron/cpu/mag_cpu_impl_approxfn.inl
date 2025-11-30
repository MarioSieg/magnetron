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

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

static float32x4_t mag_simd_exp_float32(float32x4_t x) {
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

static float32x4_t mag_simd_tanh_float32(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.f);
    float32x4_t m1 = vdupq_n_f32(-1.f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t m2 = vdupq_n_f32(-2.0f);
    float32x4_t a = vmulq_f32(m2, x);
    float32x4_t b = mag_simd_exp_float32(a);
    float32x4_t c = vaddq_f32(one, b);
    float32x4_t inv = vrecpeq_f32(c);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    inv = vmulq_f32(vrecpsq_f32(c, inv), inv);
    return vaddq_f32(m1, vmulq_f32(two, inv));
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

static __m512 mag_simd_exp_float32(__m512 x) {
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

static __m512 mag_simd_tanh_float32(__m512 x) {
    __m512 one = _mm512_set1_ps(1.f);
    __m512 neg_one = _mm512_set1_ps(-1.f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    __m512 a = _mm512_mul_ps(neg_two, x);
    __m512 b = mag_simd_exp_float32(a);
    __m512 c = _mm512_add_ps(one, b);
    __m512 inv = _mm512_rcp14_ps(c);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv);
    return _mm512_fmadd_ps(two, inv, neg_one);
}

#elif defined(__AVX2__) && defined(__FMA__)

#define mag_m256ps_K(T, name, x) static const mag_alignas(32) T mag_m256_##name[8] = {(x),(x),(x),(x),(x),(x),(x),(x)}

mag_m256ps_K(float, 1, 1.0f);
mag_m256ps_K(float, 0p5, 0.5f);
mag_m256ps_K(int32_t, min_norm_pos, 0x00800000);
mag_m256ps_K(int32_t, mant_mask, 0x7f800000);
mag_m256ps_K(int32_t, inv_mant_mask, ~0x7f800000);
mag_m256ps_K(int32_t, sign_mask, 0x80000000);
mag_m256ps_K(int32_t, inv_sign_mask, ~0x80000000);
mag_m256ps_K(int32_t, 0x7f, 0x7f);

static __m256 mag_simd_exp_float32(__m256 x) {
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

static __m256 mag_simd_tanh_float32(__m256 x) {
    __m256 one = _mm256_set1_ps(1.f);
    __m256 neg_one = _mm256_set1_ps(-1.f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    __m256 a = _mm256_mul_ps(neg_two, x);
    __m256 b = mag_simd_exp_float32(a);
    __m256 c = _mm256_add_ps(one, b);
    __m256 inv = _mm256_rcp_ps(c);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv);
    return _mm256_fmadd_ps(two, inv, neg_one);
}

static __m256 mag_simd_log_float32(__m256 x) {
    mag_m256ps_K(float, poly_SQRTHF, 0.707106781186547524);
    mag_m256ps_K(float, poly_log_p0, 7.0376836292e-2);
    mag_m256ps_K(float, poly_log_p1, -1.1514610310e-1);
    mag_m256ps_K(float, poly_log_p2, 1.1676998740e-1);
    mag_m256ps_K(float, poly_log_p3, -1.2420140846e-1);
    mag_m256ps_K(float, poly_log_p4, +1.4249322787e-1);
    mag_m256ps_K(float, poly_log_p5, -1.6668057665e-1);
    mag_m256ps_K(float, poly_log_p6, +2.0000714765e-1);
    mag_m256ps_K(float, poly_log_p7, -2.4999993993e-1);
    mag_m256ps_K(float, poly_log_p8, +3.3333331174e-1);
    mag_m256ps_K(float, poly_log_q1, -2.12194440e-4);
    mag_m256ps_K(float, poly_log_q2, 0.693359375);
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

static void mag_simd_sincos_float32(__m256 x, __m256 *s, __m256 *c) {
    mag_m256ps_K(float, minus_cephes_DP1, -0.78515625);
    mag_m256ps_K(float, minus_cephes_DP2, -2.4187564849853515625e-4);
    mag_m256ps_K(float, minus_cephes_DP3, -3.77489497744594108e-8);
    mag_m256ps_K(float, sincof_p0, -1.9515295891e-4);
    mag_m256ps_K(float, sincof_p1,  8.3321608736e-3);
    mag_m256ps_K(float, sincof_p2, -1.6666654611e-1);
    mag_m256ps_K(float, coscof_p0,  2.443315711809948e-005);
    mag_m256ps_K(float, coscof_p1, -1.388731625493765e-003);
    mag_m256ps_K(float, coscof_p2,  4.166664568298827e-002);
    mag_m256ps_K(float, cephes_FOPI, 1.27323954473516);
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

static __m128 mag_simd_exp_float32(__m128 x) {
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

static __m128 mag_simd_tanh_float32(__m128 x) {
    __m128 one = _mm_set1_ps(1.f);
    __m128 neg_one = _mm_set1_ps(-1.f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 neg_two = _mm_set1_ps(-2.0f);
    __m128 a = _mm_mul_ps(neg_two, x);
    __m128 b = mag_simd_exp_float32(a);
    __m128 c = _mm_add_ps(one, b);
    __m128 inv = _mm_rcp_ps(c);
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
    return _mm_add_ps(neg_one, _mm_mul_ps(two, inv));
}

#endif
