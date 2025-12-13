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

static float MAG_HOTPROC mag_vdot_float32(int64_t numel, const float *restrict x, const float *restrict y) {
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
    float sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
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
    float sum = _mm512_reduce_add_ps(*acc);
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
    float sum = _mm_cvtss_f32(v0);
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
    float sum = _mm_cvtss_f32(*acc);
#else
    __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(*acc, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);
#endif
    for (int64_t i=k; i < numel; ++i) sum += x[i]*y[i]; /* Scalar drain loop */
    return sum;
#else
    double r = 0.0;
    for (int64_t i=0; i < numel; ++i) r += (double)x[i]*(double)y[i];
    return (float)r;
#endif
}

static mag_float16_t MAG_HOTPROC mag_vdot_float16(int64_t numel, const mag_float16_t *restrict x, const mag_float16_t *restrict y) {
    float r = .0f;
    for (int64_t i=0; i < numel; ++i) /* TODO: Optimize with SIMD */
        r += mag_float16_to_float32(x[i])*mag_float16_to_float32(y[i]);
    return mag_float32_to_float16(r);
}

static void MAG_HOTPROC mag_vfill_float32(int64_t numel, float *o, float x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x;
}

static void MAG_HOTPROC mag_vacc_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] += x[i];
}

static void MAG_HOTPROC mag_vadd_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y[i];
}

static void MAG_HOTPROC mag_vadd_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
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
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) + mag_float16_to_float32(y[i]));
    }
}

static void MAG_HOTPROC mag_vsub_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y[i];
}

static void MAG_HOTPROC mag_vsub_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
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
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) - mag_float16_to_float32(y[i]));
    }
}

static void MAG_HOTPROC mag_vmul_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y[i];
}

static void MAG_HOTPROC mag_vmul_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
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
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i])*mag_float16_to_float32(y[i]));
    }
}

static void MAG_HOTPROC mag_vdiv_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y[i];
}

static void MAG_HOTPROC mag_vdiv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
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
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) / mag_float16_to_float32(y[i]));
    }
}

static void MAG_HOTPROC mag_vfloordiv_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_floordivf(x[i], y[i]);
}

static void MAG_HOTPROC mag_vfloordiv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(mag_floordivf(mag_float16_to_float32(x[i]), mag_float16_to_float32(y[i])));
}

static void MAG_HOTPROC mag_vmod_float32(int64_t numel, float *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_remf(x[i], y[i]);
}

static void MAG_HOTPROC mag_vmod_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    int64_t i=0;
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = mag_float32_to_float16(mag_remf(mag_float16_to_float32(x[i]), mag_float16_to_float32(y[i])));
    }
}

static void MAG_HOTPROC mag_vpows_float32(int64_t numel, float *o, const float *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = powf(x[i], y);
}

static void MAG_HOTPROC mag_vpows_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(powf(mag_float16_to_float32(x[i]), y));
}

static void MAG_HOTPROC mag_vadds_float32(int64_t numel, float *o, const float *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] + y;
}

static void MAG_HOTPROC mag_vadds_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) + y);
}

static void MAG_HOTPROC mag_vsubs_float32(int64_t numel, float *o, const float *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] - y;
}

static void MAG_HOTPROC mag_vsubs_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) - y);
}

static void MAG_HOTPROC mag_vmuls_float32(int64_t numel, float *o, const float *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i]*y;
}

static void MAG_HOTPROC mag_vmuls_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i])*y);
}

static void MAG_HOTPROC mag_vdivs_float32(int64_t numel, float *o, const float *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] / y;
}

static void MAG_HOTPROC mag_vdivs_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x, float y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(mag_float16_to_float32(x[i]) / y);
}

static void MAG_HOTPROC mag_vabs_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fabsf(x[i]);
}

static void MAG_HOTPROC mag_vabs_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(fabsf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vsgn_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        o[i] = xi > 0.f ? 1.f : xi < 0.f ? -1.f : 0.f;
    }
}

static void MAG_HOTPROC mag_vsgn_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        o[i] = xi > 0.f ? MAG_FLOAT16_ONE : xi < 0.f ? MAG_FLOAT16_NEG_ONE : MAG_FLOAT16_ZERO;
    }
}

static void MAG_HOTPROC mag_vneg_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = -x[i];
}

static void MAG_HOTPROC mag_vneg_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(-mag_float16_to_float32(x[i]));
}

static void MAG_HOTPROC mag_vlog_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = logf(x[i]);
}

static void MAG_HOTPROC mag_vlog_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(logf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vlog10_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log10f(x[i]);
}

static void MAG_HOTPROC mag_vlog10_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(log10f(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vlog1p_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log1pf(x[i]);
}

static void MAG_HOTPROC mag_vlog1p_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(log1pf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vlog2_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = log2f(x[i]);
}

static void MAG_HOTPROC mag_vlog2_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(log2f(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vsqr_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        o[i] = xi*xi;
    }
}

static void MAG_HOTPROC mag_vsqr_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        o[i] = mag_float32_to_float16(xi*xi);
    }
}

static void MAG_HOTPROC mag_vrcp_float32(int64_t numel, float *o, const float *x) {
    int64_t i = 0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t y0 = vrecpeq_f32(vx);
        float32x4_t two = vdupq_n_f32(2.0f);
        float32x4_t xy = vmulq_f32(vx, y0);
        float32x4_t t = vsubq_f32(two, xy);
        vst1q_f32(o+i, vmulq_f32(y0, t));
    }
#elif defined(__AVX512F__)
    for (; i+15 < numel; i += 16) {
        __m512 vx = _mm512_loadu_ps(x+i);
        __m512 y = _mm512_rcp14_ps(vx);
        __m512 two = _mm512_set1_ps(2.0f);
        __m512 xy = _mm512_mul_ps(vx, y);
        __m512 t = _mm512_sub_ps(two, xy);
        _mm512_storeu_ps(o+i, _mm512_mul_ps(y, t));
    }
#elif defined(__AVX__)
    for (; i+7 < numel; i += 8) {
        __m256 vx = _mm256_loadu_ps(x+i);
        __m256 y = _mm256_rcp_ps(vx);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 xy = _mm256_mul_ps(vx, y);
        __m256 t = _mm256_sub_ps(two, xy);
        _mm256_storeu_ps(o+i, _mm256_mul_ps(y, t));
    }
#elif defined(__SSE__)
    for (; i+3 < numel; i += 4) {
        __m128 vx = _mm_loadu_ps(x+i);
        __m128 y = _mm_rcp_ps(vx);
        __m128 two = _mm_set1_ps(2.0f);
        __m128 xy = _mm_mul_ps(vx, y);
        __m128 t = _mm_sub_ps(two, xy);
        _mm_storeu_ps(o+i, _mm_mul_ps(y, t));
    }
#endif
    for (; i < numel; ++i) /* Scalar drain loop */
        o[i] = 1.f/x[i];
}

static void MAG_HOTPROC mag_vrcp_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = mag_float32_to_float16(1.f/mag_float16_to_float32(x[i]));
    }
}

static void MAG_HOTPROC mag_vsqrt_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sqrtf(x[i]);
}

static void MAG_HOTPROC mag_vsqrt_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(sqrtf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vrsqrt_float32(int64_t numel, float *o, const float *x) {
    int64_t i = 0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    for (; i+3 < numel; i += 4) {
        float32x4_t vx = vld1q_f32(x+i);
        float32x4_t y0 = vrsqrteq_f32(vx);
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t three = vdupq_n_f32(3.0f);
        float32x4_t y0sq = vmulq_f32(y0, y0);
        float32x4_t xy2 = vmulq_f32(vx, y0sq);
        float32x4_t t = vsubq_f32(three, xy2);
        vst1q_f32(o+i, vmulq_f32(y0, vmulq_f32(half, t)));
    }
#elif defined(__AVX512F__)
    for (; i+15 < numel; i += 16) {
        __m512 vx = _mm512_loadu_ps(x+i);
        __m512 y = _mm512_rsqrt14_ps(vx);
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 three = _mm512_set1_ps(3.0f);
        __m512 y2 = _mm512_mul_ps(y, y);
        __m512 xy2 = _mm512_mul_ps(vx, y2);
        __m512 t = _mm512_sub_ps(three, xy2);
        y = _mm512_mul_ps(y, _mm512_mul_ps(half, t));
        _mm512_storeu_ps(o+i, y);
    }
#elif defined(__AVX__)
    for (; i+7 < numel; i += 8) {
        __m256 vx = _mm256_loadu_ps(x+i);
        __m256 y = _mm256_rsqrt_ps(vx);
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 xy2 = _mm256_mul_ps(vx, y2);
        __m256 t = _mm256_sub_ps(three, xy2);
        y = _mm256_mul_ps(y, _mm256_mul_ps(half, t));
        _mm256_storeu_ps(o+i, y);
    }
#elif defined(__SSE__)
    for (; i+3 < numel; i += 4) {
        __m128 vx = _mm_loadu_ps(x+i);
        __m128 y = _mm_rsqrt_ps(vx);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 y2 = _mm_mul_ps(y, y);
        __m128 xy2 = _mm_mul_ps(vx, y2);
        __m128 t = _mm_sub_ps(three, xy2);
        y = _mm_mul_ps(y, _mm_mul_ps(half, t));
        _mm_storeu_ps(o+i, y);
    }
#endif
    for (; i < numel; ++i) { /* Scalar drain loop */
        o[i] = 1.0f / sqrtf(x[i]);
    }
}

static void MAG_HOTPROC mag_vrsqrt_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(1.f/sqrtf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vsin_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinf(x[i]);
}

static void MAG_HOTPROC mag_vsin_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(sinf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vcos_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = cosf(x[i]);
}

static void MAG_HOTPROC mag_vcos_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(cosf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vtan_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanf(x[i]);
}

static void MAG_HOTPROC mag_vtan_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(tanf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vasin_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinf(x[i]);
}

static void MAG_HOTPROC mag_vasin_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(asinf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vacos_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acosf(x[i]);
}

static void MAG_HOTPROC mag_vacos_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(acosf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vatan_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanf(x[i]);
}

static void MAG_HOTPROC mag_vatan_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(atanf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vsinh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = sinhf(x[i]);
}

static void MAG_HOTPROC mag_vsinh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(sinhf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vcosh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = coshf(x[i]);
}

static void MAG_HOTPROC mag_vcosh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(coshf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vtanh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = tanhf(x[i]);
}

static void MAG_HOTPROC mag_vtanh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(tanhf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vasinh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = asinhf(x[i]);
}

static void MAG_HOTPROC mag_vasinh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(asinhf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vacosh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = acoshf(x[i]);
}

static void MAG_HOTPROC mag_vacosh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(acoshf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vatanh_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = atanhf(x[i]);
}

static void MAG_HOTPROC mag_vatanh_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(atanhf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vstep_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.0f ? 1.f : 0.0f;
}

static void MAG_HOTPROC mag_vstep_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) > 0.0f ? MAG_FLOAT16_ONE : MAG_FLOAT16_ZERO;
}

static void MAG_HOTPROC mag_verf_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erff(x[i]);
}

static void MAG_HOTPROC mag_verf_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(erff(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_verfc_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = erfcf(x[i]);
}

static void MAG_HOTPROC mag_verfc_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(erfcf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vexp_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expf(x[i]);
}

static void MAG_HOTPROC mag_vexp_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(expf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vexp2_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = exp2f(x[i]);
}

static void MAG_HOTPROC mag_vexp2_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(exp2f(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vexpm1_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = expm1f(x[i]);
}

static void MAG_HOTPROC mag_vexpm1_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(expm1f(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vfloor_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = floorf(x[i]);
}

static void MAG_HOTPROC mag_vfloor_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(floorf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vceil_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = ceilf(x[i]);
}

static void MAG_HOTPROC mag_vceil_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(ceilf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vround_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = nearbyintf(x[i]);
}

static void MAG_HOTPROC mag_vround_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(nearbyintf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vtrunc_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = truncf(x[i]);
}

static void MAG_HOTPROC mag_vtrunc_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(truncf(mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vsoftmax_dv_float32(int64_t numel, float *o, const float *x) {
    mag_vexp_float32(numel, o, x);
}

static void MAG_HOTPROC mag_vsoftmax_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    mag_vexp_float16(numel, o, x);
}

static void MAG_HOTPROC mag_vsigmoid_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = 1.f/(1.f + expf(-x[i]));
}

static void MAG_HOTPROC mag_vsigmoid_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(1.f/(1.f + expf(-mag_float16_to_float32(x[i]))));
}

static void MAG_HOTPROC mag_vsigmoid_dv_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float sig = 1.f/(1.f + expf(-x[i]));
        o[i] = sig*(1.f-sig);
    }
}

static void MAG_HOTPROC mag_vsigmoid_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float sig = 1.f/(1.f + expf(-mag_float16_to_float32(x[i])));
        o[i] = mag_float32_to_float16(sig*(1.f-sig));
    }
}

static void MAG_HOTPROC mag_vhard_sigmoid_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fminf(1.f, fmaxf(0.0f, (x[i] + 3.0f)/6.0f));
}

static void MAG_HOTPROC mag_vhard_sigmoid_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16( fminf(1.f, fmaxf(0.0f, (mag_float16_to_float32(x[i]) + 3.0f)/6.0f)));
}

static void MAG_HOTPROC mag_vsilu_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        o[i] = xi*(1.f/(1.f + expf(-xi)));
    }
}

static void MAG_HOTPROC mag_vsilu_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        o[i] = mag_float32_to_float16(xi*(1.f/(1.f + expf(-xi))));
    }
}

static void MAG_HOTPROC mag_vsilu_dv_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        float sig = 1.f/(1.f + expf(-xi));
        o[i] = sig + xi*sig;
    }
}

static void MAG_HOTPROC mag_vsilu_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        float sig = 1.f/(1.f + expf(-xi));
        o[i] = mag_float32_to_float16(sig + xi*sig);
    }
}

static void MAG_HOTPROC mag_vtanh_dv_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float th = tanhf(x[i]);
        o[i] = 1.f - th*th;
    }
}

static void MAG_HOTPROC mag_vtanh_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float th = tanhf(mag_float16_to_float32(x[i]));
        o[i] = mag_float32_to_float16(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vrelu_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = fmaxf(0.f, x[i]);
}

static void MAG_HOTPROC mag_vrelu_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float32_to_float16(fmaxf(0.f, mag_float16_to_float32(x[i])));
}

static void MAG_HOTPROC mag_vrelu_dv_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > 0.f ? 1.f : 0.f;
}

static void MAG_HOTPROC mag_vrelu_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) > 0.f ? MAG_FLOAT16_ONE : MAG_FLOAT16_ZERO;
}

static void MAG_HOTPROC mag_vgelu_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        o[i] = .5f*xi*(1.f+erff(xi*MAG_INVSQRT2));
    }
}

static void MAG_HOTPROC mag_vgelu_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        o[i] = mag_float32_to_float16(.5f*xi*(1.f+erff(xi*MAG_INVSQRT2)));
    }
}

static void MAG_HOTPROC mag_vgelu_approx_float32(int64_t numel, float *o, const float *x) {
    int64_t i=0;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256 coeff = _mm256_set1_ps(MAG_INVSQRT2);
    __m256 coeff2 = _mm256_set1_ps(MAG_GELU_COEFF);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    for (; i+7 < numel; i += 8) {
        __m256 xi = _mm256_loadu_ps(x+i);
        __m256 xi3 = _mm256_mul_ps(xi, _mm256_mul_ps(xi, xi));
        __m256 tan1 = _mm256_add_ps(one, mag_simd_tanh_float32(_mm256_mul_ps(coeff, _mm256_add_ps(xi, _mm256_mul_ps(coeff2, xi3)))));
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(xi, tan1), half);
        _mm256_storeu_ps(o+i, r);
    }
#endif
    for (; i < numel; ++i) {
        float xi = x[i];
        o[i] = 0.5f*xi*(1.0f+tanhf(MAG_INVSQRT2*(xi+MAG_GELU_COEFF*xi*xi*xi)));
    }
}

static void MAG_HOTPROC mag_vgelu_approx_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        o[i] = mag_float32_to_float16(0.5f*xi*(1.0f+tanhf(MAG_INVSQRT2*(xi+MAG_GELU_COEFF*xi*xi*xi))));
    }
}


static void MAG_HOTPROC mag_vgelu_dv_float32(int64_t numel, float *o, const float *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = x[i];
        float th = tanhf(xi);
        o[i] = .5f*(1.f + th) + .5f*xi*(1.f - th*th);
    }
}

static void MAG_HOTPROC mag_vgelu_dv_float16(int64_t numel, mag_float16_t *o, const mag_float16_t *x) {
    for (int64_t i=0; i < numel; ++i) {
        float xi = mag_float16_to_float32(x[i]);
        float th = tanhf(xi);
        o[i] = mag_float32_to_float16(.5f*(1.f + th) + .5f*xi*(1.f - th*th));
    }
}

static double MAG_HOTPROC mag_vsum_f64_float32(int64_t numel, const float *x) {
    double sum = 0.0;
    for (int64_t i=0; i < numel; ++i)
        sum += (double)x[i];
    return sum;
}

static double MAG_HOTPROC mag_vsum_f64_float16(int64_t numel, const mag_float16_t *x) {
    double sum = 0.0;
    for (int64_t i=0; i < numel; ++i)
        sum += mag_float16_to_float32(x[i]);
    return sum;
}

static float MAG_HOTPROC mag_vmin_float32(int64_t numel, const float *x) {
    float min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, x[i]);
    return min;
}

static float MAG_HOTPROC mag_vmin_float16(int64_t numel, const mag_float16_t *x) {
    float min = INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fminf(min, mag_float16_to_float32(x[i]));
    return min;
}

static float MAG_HOTPROC mag_vmax_float32(int64_t numel, const float *x) {
    float min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, x[i]);
    return min;
}

static float MAG_HOTPROC mag_vmax_float16(int64_t numel, const mag_float16_t *x) {
    float min = -INFINITY;
    for (int64_t i=0; i < numel; ++i)
        min = fmaxf(min, mag_float16_to_float32(x[i]));
    return min;
}

#define mag_impl_vecop_int(T, TF, SIGNESS) \
    static void mag_vadd_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]+y[i]; \
    } \
    static void mag_vsub_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]-y[i]; \
    } \
    static void mag_vmul_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]*y[i]; \
    } \
    static void mag_vdiv_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]/y[i]; \
    } \
    static void mag_vfloordiv_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = mag_floordiv##SIGNESS(x[i],y[i]); \
    } \
    static void mag_vmod_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = mag_rem##SIGNESS(x[i],y[i]); \
    } \
    static void mag_vand_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]&y[i]; \
    } \
    static void mag_vor_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]|y[i]; \
    } \
    static void mag_vxor_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]^y[i]; \
    } \
    static void mag_vshl_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = mag_shl##SIGNESS(x[i], y[i], sizeof(T)<<3); \
    } \
    static void mag_vshr_##TF(int64_t numel, T *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = mag_shr##SIGNESS(x[i], y[i], sizeof(T)<<3); \
    } \
    static void mag_vnot_##TF(int64_t numel, T *o, const T *x) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = ~x[i]; \
    } \
    static void MAG_HOTPROC mag_veq_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]==y[i]; \
    } \
    static void MAG_HOTPROC mag_vne_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]!=y[i]; \
    } \
    static void MAG_HOTPROC mag_vlt_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]<y[i]; \
    } \
    static void MAG_HOTPROC mag_vgt_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]>y[i]; \
    } \
    static void MAG_HOTPROC mag_vle_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]<=y[i]; \
    } \
    static void MAG_HOTPROC mag_vge_##TF(int64_t numel, uint8_t *o, const T *x, const T *y) { \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = x[i]>=y[i]; \
    }

mag_impl_vecop_int(uint8_t, uint8, u)
mag_impl_vecop_int(int8_t, int8, i)
mag_impl_vecop_int(uint16_t, uint16, u)
mag_impl_vecop_int(int16_t, int16, i)
mag_impl_vecop_int(uint32_t, uint32, u)
mag_impl_vecop_int(int32_t, int32, i)
mag_impl_vecop_int(uint64_t, uint64, u)
mag_impl_vecop_int(int64_t, int64, i)

#undef mag_impl_vecop_int

static void MAG_HOTPROC mag_veq_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i) {
        o[i] = x[i] == y[i];
    }
}

static void MAG_HOTPROC mag_veq_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits == y[i].bits;
}

static void MAG_HOTPROC mag_vne_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] != y[i];
}

static void MAG_HOTPROC mag_vne_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i].bits != y[i].bits;
}

static void MAG_HOTPROC mag_vle_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] <= y[i];
}

static void MAG_HOTPROC mag_vle_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) <= mag_float16_to_float32(y[i]);
}

static void MAG_HOTPROC mag_vge_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] >= y[i];
}

static void MAG_HOTPROC mag_vge_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) >= mag_float16_to_float32(y[i]);
}

static void MAG_HOTPROC mag_vlt_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] < y[i];
}

static void MAG_HOTPROC mag_vlt_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) < mag_float16_to_float32(y[i]);
}

static void MAG_HOTPROC mag_vgt_float32(int64_t numel, uint8_t *o, const float *x, const float *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = x[i] > y[i];
}

static void MAG_HOTPROC mag_vgt_float16(int64_t numel, uint8_t *o, const mag_float16_t *x, const mag_float16_t *y) {
    for (int64_t i=0; i < numel; ++i)
        o[i] = mag_float16_to_float32(x[i]) > mag_float16_to_float32(y[i]);
}

static void mag_nop(const mag_kernel_payload_t *payload) {
    (void)payload;
}
