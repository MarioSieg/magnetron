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

static MAG_AINLINE mag_float16_t mag_float32_to_float16(float x) {
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
    r = mag_float16_from_float32_soft_fp(x).bits;
#endif
    return (mag_float16_t) {
        .bits=r
    };
}

static MAG_AINLINE float mag_float16_to_float32(mag_float16_t x) {
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
    return mag_float16_to_float32_soft_fp(x);
#endif
}

typedef void (mag_vcast_fn_t)(int64_t numel, void *restrict dst, const void *restrict src);

#define mag_cast_fn_builtin(TDst, x) ((TDst)x)
#define mag_cast_fn_float322float16(TDst, x) (mag_float32_to_float16(x))
#define mag_cast_fn_float162float32_upcast(TDst, x) ((TDst)mag_float16_to_float32(x))

#define mag_gen_vcast(TSrc, TDst, F) \
    static void MAG_HOTPROC mag_vcast_##TSrc##_to_##TDst(int64_t numel, void *restrict dst, const void *restrict src) { \
        TDst *restrict o = (TDst *)dst; \
        const TSrc *restrict x = (const TSrc *)src; \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = F(TDst, x[i]); \
    }

#define mag_gen_vcast_to_bool_arith(TSrc) \
    static void MAG_HOTPROC mag_vcast_##TSrc##_to_boolean(int64_t numel, void *restrict dst, const void *restrict src) { \
        uint8_t *restrict o = (uint8_t *)dst; \
        const TSrc *restrict x = (const TSrc *)src; \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = (uint8_t)(x[i]!=(TSrc)0); \
    }

/* Generate all dtype cast perms. TSrc == TDst are unused but available, as the op is delegate to the clone operator before. */
mag_gen_vcast_to_bool_arith(float);

static void MAG_HOTPROC mag_vcast_mag_float16_t_to_boolean(int64_t numel, void *restrict dst, const void *restrict src) {
    uint8_t *restrict o = dst;
    const mag_float16_t *restrict x = src;
    for (int64_t i=0; i < numel; ++i)
        o[i] = (uint8_t)(mag_float16_to_float32(x[i]) != 0.0f);
}

mag_gen_vcast_to_bool_arith(uint8_t);
mag_gen_vcast_to_bool_arith(int8_t);
mag_gen_vcast_to_bool_arith(uint16_t);
mag_gen_vcast_to_bool_arith(int16_t);
mag_gen_vcast_to_bool_arith(uint32_t);
mag_gen_vcast_to_bool_arith(int32_t);
mag_gen_vcast_to_bool_arith(uint64_t);
mag_gen_vcast_to_bool_arith(int64_t);

#undef mag_gen_vcast_to_bool_arith

mag_gen_vcast(float, float, mag_cast_fn_builtin)
/*mag_gen_vcast(float, mag_float16_t, mag_cast_fn_float322float16) - There is a SIMD fast path function for this cast below */
mag_gen_vcast(float, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int64_t, mag_cast_fn_builtin)

/*mag_gen_vcast(mag_float16_t, float, mag_cast_fn_float162float32)  - There is a SIMD fast path function for this cast below */
mag_gen_vcast(mag_float16_t, mag_float16_t, mag_cast_fn_builtin)
mag_gen_vcast(mag_float16_t, uint8_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, int8_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, uint16_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, int16_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, uint32_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, int32_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, uint64_t, mag_cast_fn_float162float32_upcast)
mag_gen_vcast(mag_float16_t, int64_t, mag_cast_fn_float162float32_upcast)

mag_gen_vcast(uint8_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(uint8_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int8_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(int8_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint16_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(uint16_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int16_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(int16_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint32_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(uint32_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int32_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(int32_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint64_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(uint64_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int64_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, mag_float16_t, mag_cast_fn_float322float16)
mag_gen_vcast(int64_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int64_t, mag_cast_fn_builtin)

#undef mag_gen_vcast

/* SIMD fast paths */

static void MAG_HOTPROC mag_vcast_float_to_mag_float16_t(int64_t numel, void *restrict xo, const void *restrict xx) {
    mag_float16_t *o = xo;
    const float *x = xx;
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
        o[i] = mag_float32_to_float16(x[i]);
}

static void MAG_HOTPROC mag_vcast_mag_float16_t_to_float(int64_t numel, void *restrict xo, const void *restrict xx) {
    float *o = xo;
    const mag_float16_t *x = xx;
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
        o[i] = mag_float16_to_float32(x[i]);
}

/* Src -> Dst */
static mag_vcast_fn_t *const mag_cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM] = {
    [MAG_DTYPE_FLOAT32] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_float_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_float_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_float_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_float_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_float_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_float_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_float_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_float_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_float_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_float_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_float_to_int64_t,
    },
    [MAG_DTYPE_FLOAT16] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_mag_float16_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_mag_float16_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_mag_float16_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_mag_float16_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_mag_float16_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_mag_float16_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_mag_float16_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_mag_float16_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_mag_float16_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_mag_float16_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_mag_float16_t_to_int64_t,
    },
    [MAG_DTYPE_BOOLEAN] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_uint8_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_uint8_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_uint8_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_uint8_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_uint8_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_uint8_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_uint8_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_uint8_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_uint8_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_uint8_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_uint8_t_to_int64_t,
    },
    [MAG_DTYPE_UINT8] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_uint8_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_uint8_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_uint8_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_uint8_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_uint8_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_uint8_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_uint8_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_uint8_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_uint8_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_uint8_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_uint8_t_to_int64_t,
    },
    [MAG_DTYPE_INT8] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_int8_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_int8_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_int8_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_int8_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_int8_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_int8_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_int8_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_int8_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_int8_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_int8_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_int8_t_to_int64_t,
    },
    [MAG_DTYPE_UINT16] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_uint16_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_uint16_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_uint16_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_uint16_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_uint16_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_uint16_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_uint16_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_uint16_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_uint16_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_uint16_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_uint16_t_to_int64_t,
    },
    [MAG_DTYPE_INT16] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_int16_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_int16_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_int16_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_int16_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_int16_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_int16_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_int16_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_int16_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_int16_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_int16_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_int16_t_to_int64_t,
    },
    [MAG_DTYPE_UINT32] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_uint32_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_uint32_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_uint32_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_uint32_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_uint32_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_uint32_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_uint32_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_uint32_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_uint32_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_uint32_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_uint32_t_to_int64_t,
    },
    [MAG_DTYPE_INT32] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_int32_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_int32_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_int32_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_int32_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_int32_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_int32_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_int32_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_int32_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_int32_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_int32_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_int32_t_to_int64_t,
    },
    [MAG_DTYPE_UINT64] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_uint64_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_uint64_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_uint64_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_uint64_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_uint64_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_uint64_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_uint64_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_uint64_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_uint64_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_uint64_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_uint64_t_to_int64_t,
    },
    [MAG_DTYPE_INT64] = {
        [MAG_DTYPE_FLOAT32] = mag_vcast_int64_t_to_float,
        [MAG_DTYPE_FLOAT16] = mag_vcast_int64_t_to_mag_float16_t,
        [MAG_DTYPE_BOOLEAN] = mag_vcast_int64_t_to_boolean,
        [MAG_DTYPE_UINT8] = mag_vcast_int64_t_to_uint8_t,
        [MAG_DTYPE_INT8] = mag_vcast_int64_t_to_int8_t,
        [MAG_DTYPE_UINT16] = mag_vcast_int64_t_to_uint16_t,
        [MAG_DTYPE_INT16] = mag_vcast_int64_t_to_int16_t,
        [MAG_DTYPE_UINT32] = mag_vcast_int64_t_to_uint32_t,
        [MAG_DTYPE_INT32] = mag_vcast_int64_t_to_int32_t,
        [MAG_DTYPE_UINT64] = mag_vcast_int64_t_to_uint64_t,
        [MAG_DTYPE_INT64] = mag_vcast_int64_t_to_int64_t,
    },
};

static MAG_HOTPROC void mag_cast_generic(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_dtype_t src = x->dtype;
    mag_dtype_t dst = r->dtype;
    const mag_type_traits_t *msrc = mag_type_trait(src);
    const mag_type_traits_t *mdst = mag_type_trait(dst);
    mag_vcast_fn_t *kernel = mag_cast_table_2D[src][dst];
    mag_assert(kernel, "No kernel found for type cast: %s -> %s", msrc->name, mdst->name);
    uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r);
    const uint8_t *bx = (const uint8_t *)mag_tensor_data_ptr(x);
    int64_t nbs = (int64_t)msrc->size;
    int64_t nbd = (int64_t)mdst->size;
    int64_t total = r->numel;
    int64_t tc = payload->thread_num;
    int64_t ti = payload->thread_idx;
    int64_t chunk = (total + tc - 1)/tc;
    int64_t ra = ti*chunk;
    int64_t rb = mag_xmin(ra + chunk, total);
    if (mag_unlikely(rb <= ra)) return;
    if (mag_full_cont2(r, x)) {
        void *pr = br + ra*nbd;
        const void *px = bx + ra*nbs;
        mag_bnd_chk(px, bx, mag_tensor_numbytes(x));
        mag_bnd_chk(pr, br, mag_tensor_numbytes(r));
        (*kernel)(rb-ra, pr, px);
        return;
    }
    /* We work in byte granularity and compute pointer offsets manually to avoid a generic for this stub function */
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->coords);
    mag_coords_iter_init(&cx, &x->coords);
    for (int64_t i=ra; i < rb; ++i) { /* TODO: Optimize - Slow with the single indirect call for each element */
        int64_t ri, xi;
        mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi);
        void *pr = br + ri*nbd;
        const void *px = bx + xi*nbs;
        mag_bnd_chk(px, bx, mag_tensor_numbytes(x));
        mag_bnd_chk(pr, br, mag_tensor_numbytes(r));
        (*kernel)(1, pr, px);
    }
}
