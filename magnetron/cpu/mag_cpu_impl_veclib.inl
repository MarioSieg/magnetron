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
        r += mag_e5m10_to_e8m23(x[i])*mag_e5m10_to_e8m23(y[i]);
    return mag_e8m23_to_e5m10(r);
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
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) + mag_e5m10_to_e8m23(y[i]));
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
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) - mag_e5m10_to_e8m23(y[i]));
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
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i])*mag_e5m10_to_e8m23(y[i]));
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
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) / mag_e5m10_to_e8m23(y[i]));
    }
}

static void MAG_HOTPROC mag_vmod_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmodf(x[i], y[i]);
}

static void MAG_HOTPROC mag_vmod_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    int64_t i=0;
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_e8m23_to_e5m10(fmodf(mag_e5m10_to_e8m23(x[i]), mag_e5m10_to_e8m23(y[i])));
    }
}

static void MAG_HOTPROC mag_vpows_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = powf(x[i], y);
}

static void MAG_HOTPROC mag_vpows_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(powf(mag_e5m10_to_e8m23(x[i]), y));
}

static void MAG_HOTPROC mag_vadds_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y;
}

static void MAG_HOTPROC mag_vadds_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) + y);
}

static void MAG_HOTPROC mag_vsubs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y;
}

static void MAG_HOTPROC mag_vsubs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) - y);
}

static void MAG_HOTPROC mag_vmuls_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y;
}

static void MAG_HOTPROC mag_vmuls_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i])*y);
}

static void MAG_HOTPROC mag_vdivs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y;
}

static void MAG_HOTPROC mag_vdivs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x, mag_e8m23_t y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(mag_e5m10_to_e8m23(x[i]) / y);
}

static void MAG_HOTPROC mag_vabs_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fabsf(x[i]);
}

static void MAG_HOTPROC mag_vabs_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(fabsf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsgn_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi > 0.f ? 1.f : xi < 0.f ? -1.f : 0.f;
    }
}

static void MAG_HOTPROC mag_vsgn_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        o[i] = xi > 0.f ? MAG_E5M10_ONE : xi < 0.f ? MAG_E5M10_NEG_ONE : MAG_E5M10_ZERO;
    }
}

static void MAG_HOTPROC mag_vneg_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = -x[i];
}

static void MAG_HOTPROC mag_vneg_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(-mag_e5m10_to_e8m23(x[i]));
}

static void MAG_HOTPROC mag_vlog_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = logf(x[i]);
}

static void MAG_HOTPROC mag_vlog_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(logf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog10_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log10f(x[i]);
}

static void MAG_HOTPROC mag_vlog10_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(log10f(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog1p_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log1pf(x[i]);
}

static void MAG_HOTPROC mag_vlog1p_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(log1pf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vlog2_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log2f(x[i]);
}

static void MAG_HOTPROC mag_vlog2_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(log2f(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsqr_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi*xi;
    }
}

static void MAG_HOTPROC mag_vsqr_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        o[i] = mag_e8m23_to_e5m10(xi*xi);
    }
}

static void MAG_HOTPROC mag_vsqrt_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sqrtf(x[i]);
}

static void MAG_HOTPROC mag_vsqrt_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(sqrtf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsin_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinf(x[i]);
}

static void MAG_HOTPROC mag_vsin_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(sinf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vcos_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = cosf(x[i]);
}

static void MAG_HOTPROC mag_vcos_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(cosf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtan_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanf(x[i]);
}

static void MAG_HOTPROC mag_vtan_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(tanf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vasin_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinf(x[i]);
}

static void MAG_HOTPROC mag_vasin_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(asinf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vacos_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acosf(x[i]);
}

static void MAG_HOTPROC mag_vacos_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(acosf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vatan_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanf(x[i]);
}

static void MAG_HOTPROC mag_vatan_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(atanf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vsinh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinhf(x[i]);
}

static void MAG_HOTPROC mag_vsinh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(sinhf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vcosh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = coshf(x[i]);
}

static void MAG_HOTPROC mag_vcosh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(coshf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtanh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanhf(x[i]);
}

static void MAG_HOTPROC mag_vtanh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(tanhf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vasinh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinhf(x[i]);
}

static void MAG_HOTPROC mag_vasinh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(asinhf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vacosh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acoshf(x[i]);
}

static void MAG_HOTPROC mag_vacosh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(acoshf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vatanh_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanhf(x[i]);
}

static void MAG_HOTPROC mag_vatanh_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(atanhf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vstep_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.0f ? 1.f : 0.0f;
}

static void MAG_HOTPROC mag_vstep_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_to_e8m23(x[i]) > 0.0f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_verf_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erff(x[i]);
}

static void MAG_HOTPROC mag_verf_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(erff(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_verfc_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erfcf(x[i]);
}

static void MAG_HOTPROC mag_verfc_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(erfcf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexp_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expf(x[i]);
}

static void MAG_HOTPROC mag_vexp_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(expf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexp2_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = exp2f(x[i]);
}

static void MAG_HOTPROC mag_vexp2_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(exp2f(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vexpm1_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expm1f(x[i]);
}

static void MAG_HOTPROC mag_vexpm1_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(expm1f(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vfloor_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = floorf(x[i]);
}

static void MAG_HOTPROC mag_vfloor_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(floorf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vceil_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = ceilf(x[i]);
}

static void MAG_HOTPROC mag_vceil_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(ceilf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vround_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = nearbyintf(x[i]);
}

static void MAG_HOTPROC mag_vround_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(nearbyintf(mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vtrunc_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = truncf(x[i]);
}

static void MAG_HOTPROC mag_vtrunc_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(truncf(mag_e5m10_to_e8m23(x[i])));
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
        o[i] = mag_e8m23_to_e5m10(1.f/(1.f + expf(-mag_e5m10_to_e8m23(x[i]))));
}

static void MAG_HOTPROC mag_vsigmoid_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t sig = 1.f/(1.f + expf(-x[i]));
        o[i] = sig*(1.f-sig);
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t sig = 1.f/(1.f + expf(-mag_e5m10_to_e8m23(x[i])));
        o[i] = mag_e8m23_to_e5m10(sig*(1.f-sig));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fminf(1.f, fmaxf(0.0f, (x[i] + 3.0f)/6.0f));
}

static void MAG_HOTPROC mag_vhard_sigmoid_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10( fminf(1.f, fmaxf(0.0f, (mag_e5m10_to_e8m23(x[i]) + 3.0f)/6.0f)));
}

static void MAG_HOTPROC mag_vsilu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = xi*(1.f/(1.f + expf(-xi)));
    }
}

static void MAG_HOTPROC mag_vsilu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        o[i] = mag_e8m23_to_e5m10(xi*(1.f/(1.f + expf(-xi))));
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
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        mag_e8m23_t sig = 1.f/(1.f + expf(-xi));
        o[i] = mag_e8m23_to_e5m10(sig + xi*sig);
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
        mag_e8m23_t th = tanhf(mag_e5m10_to_e8m23(x[i]));
        o[i] = mag_e8m23_to_e5m10(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vrelu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmaxf(0.f, x[i]);
}

static void MAG_HOTPROC mag_vrelu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e8m23_to_e5m10(fmaxf(0.f, mag_e5m10_to_e8m23(x[i])));
}

static void MAG_HOTPROC mag_vrelu_dv_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.f ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vrelu_dv_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_to_e8m23(x[i]) > 0.f ? MAG_E5M10_ONE : MAG_E5M10_ZERO;
}

static void MAG_HOTPROC mag_vgelu_e8m23(int64_t numel, mag_e8m23_t *o, const mag_e8m23_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = x[i];
        o[i] = .5f*xi*(1.f+erff(xi*MAG_INVSQRT2));
    }
}

static void MAG_HOTPROC mag_vgelu_e5m10(int64_t numel, mag_e5m10_t *o, const mag_e5m10_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        o[i] = mag_e8m23_to_e5m10(.5f*xi*(1.f+erff(xi*MAG_INVSQRT2)));
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
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        o[i] = mag_e8m23_to_e5m10(0.5f*xi*(1.0f+tanhf(MAG_INVSQRT2*(xi+MAG_GELU_COEFF*xi*xi*xi))));
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
        mag_e8m23_t xi = mag_e5m10_to_e8m23(x[i]);
        mag_e8m23_t th = tanhf(xi);
        o[i] = mag_e8m23_to_e5m10(.5f*(1.f + th) + .5f*xi*(1.f - th*th));
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
        sum += mag_e5m10_to_e8m23(x[i]);
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
        min = fminf(min, mag_e5m10_to_e8m23(x[i]));
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
        min = fmaxf(min, mag_e5m10_to_e8m23(x[i]));
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
        o[i] = mag_e5m10_to_e8m23(x[i]) <= mag_e5m10_to_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vge_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >= y[i];
}

static void MAG_HOTPROC mag_vge_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_to_e8m23(x[i]) >= mag_e5m10_to_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vlt_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] < y[i];
}

static void MAG_HOTPROC mag_vlt_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_to_e8m23(x[i]) < mag_e5m10_to_e8m23(y[i]);
}

static void MAG_HOTPROC mag_vgt_e8m23(int64_t numel, mag_bool_t *o, const mag_e8m23_t *x, const mag_e8m23_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > y[i];
}

static void MAG_HOTPROC mag_vgt_e5m10(int64_t numel, mag_bool_t *o, const mag_e5m10_t *x, const mag_e5m10_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_e5m10_to_e8m23(x[i]) > mag_e5m10_to_e8m23(y[i]);
}

static void mag_nop(const mag_kernel_payload_t *payload) {
    (void)payload;
}
