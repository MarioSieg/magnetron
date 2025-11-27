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

static MAG_AINLINE mag_e5m10_t mag_e8m23_to_e5m10(mag_e8m23_t x) {
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

static MAG_AINLINE mag_e8m23_t mag_e5m10_to_e8m23(mag_e5m10_t x) {
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

typedef void (mag_vcast_fn_t)(int64_t numel, void *restrict dst, const void *restrict src);

#define mag_cast_fn_builtin(TDst, x) ((mag_##TDst##_t)x)
#define mag_cast_fn_e8m232e5m10(TDst, x) (mag_e8m23_to_e5m10(x))
#define mag_cast_fn_e5m102e8m23_upcast(TDst, x) ((mag_##TDst##_t)mag_e5m10_to_e8m23(x))

#define mag_gen_vcast(TSrc, TDst, F) \
    static void MAG_HOTPROC mag_vcast_##TSrc##_##TDst(int64_t numel, void *restrict dst, const void *restrict src) { \
        mag_##TDst##_t *restrict o = (mag_##TDst##_t *)dst; \
        const mag_##TSrc##_t *restrict x = (const mag_##TSrc##_t *)src; \
        for (int64_t i=0; i < numel; ++i) \
            o[i] = F(TDst, x[i]); \
    }

/* Generate all dtype cast perms. TSrc == TDst are unused but available, as the op is delegate to the clone operator before. */

mag_gen_vcast(e8m23, e8m23, mag_cast_fn_builtin)
/*mag_gen_vcast(e8m23, e5m10, mag_cast_fn_e8m232e5m10) - There is a SIMD fast path function for this cast below */
mag_gen_vcast(e8m23, u8, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, i8, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, u16, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, i16, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, u32, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, i32, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, u64, mag_cast_fn_builtin)
mag_gen_vcast(e8m23, i64, mag_cast_fn_builtin)

/*mag_gen_vcast(e5m10, e8m23, mag_cast_fn_e5m102e8m23)  - There is a SIMD fast path function for this cast below */
mag_gen_vcast(e5m10, e5m10, mag_cast_fn_builtin)
mag_gen_vcast(e5m10, u8, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, i8, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, u16, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, i16, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, u32, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, i32, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, u64, mag_cast_fn_e5m102e8m23_upcast)
mag_gen_vcast(e5m10, i64, mag_cast_fn_e5m102e8m23_upcast)

mag_gen_vcast(u8, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(u8, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(u8, u8, mag_cast_fn_builtin)
mag_gen_vcast(u8, i8, mag_cast_fn_builtin)
mag_gen_vcast(u8, u16, mag_cast_fn_builtin)
mag_gen_vcast(u8, i16, mag_cast_fn_builtin)
mag_gen_vcast(u8, u32, mag_cast_fn_builtin)
mag_gen_vcast(u8, i32, mag_cast_fn_builtin)
mag_gen_vcast(u8, u64, mag_cast_fn_builtin)
mag_gen_vcast(u8, i64, mag_cast_fn_builtin)
mag_gen_vcast(i8, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(i8, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(i8, u8, mag_cast_fn_builtin)
mag_gen_vcast(i8, i8, mag_cast_fn_builtin)
mag_gen_vcast(i8, u16, mag_cast_fn_builtin)
mag_gen_vcast(i8, i16, mag_cast_fn_builtin)
mag_gen_vcast(i8, u32, mag_cast_fn_builtin)
mag_gen_vcast(i8, i32, mag_cast_fn_builtin)
mag_gen_vcast(i8, u64, mag_cast_fn_builtin)
mag_gen_vcast(i8, i64, mag_cast_fn_builtin)

mag_gen_vcast(u16, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(u16, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(u16, u8, mag_cast_fn_builtin)
mag_gen_vcast(u16, i8, mag_cast_fn_builtin)
mag_gen_vcast(u16, u16, mag_cast_fn_builtin)
mag_gen_vcast(u16, i16, mag_cast_fn_builtin)
mag_gen_vcast(u16, u32, mag_cast_fn_builtin)
mag_gen_vcast(u16, i32, mag_cast_fn_builtin)
mag_gen_vcast(u16, u64, mag_cast_fn_builtin)
mag_gen_vcast(u16, i64, mag_cast_fn_builtin)
mag_gen_vcast(i16, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(i16, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(i16, u8, mag_cast_fn_builtin)
mag_gen_vcast(i16, i8, mag_cast_fn_builtin)
mag_gen_vcast(i16, u16, mag_cast_fn_builtin)
mag_gen_vcast(i16, i16, mag_cast_fn_builtin)
mag_gen_vcast(i16, u32, mag_cast_fn_builtin)
mag_gen_vcast(i16, i32, mag_cast_fn_builtin)
mag_gen_vcast(i16, u64, mag_cast_fn_builtin)
mag_gen_vcast(i16, i64, mag_cast_fn_builtin)

mag_gen_vcast(u32, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(u32, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(u32, u8, mag_cast_fn_builtin)
mag_gen_vcast(u32, i8, mag_cast_fn_builtin)
mag_gen_vcast(u32, u16, mag_cast_fn_builtin)
mag_gen_vcast(u32, i16, mag_cast_fn_builtin)
mag_gen_vcast(u32, u32, mag_cast_fn_builtin)
mag_gen_vcast(u32, i32, mag_cast_fn_builtin)
mag_gen_vcast(u32, u64, mag_cast_fn_builtin)
mag_gen_vcast(u32, i64, mag_cast_fn_builtin)
mag_gen_vcast(i32, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(i32, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(i32, u8, mag_cast_fn_builtin)
mag_gen_vcast(i32, i8, mag_cast_fn_builtin)
mag_gen_vcast(i32, u16, mag_cast_fn_builtin)
mag_gen_vcast(i32, i16, mag_cast_fn_builtin)
mag_gen_vcast(i32, u32, mag_cast_fn_builtin)
mag_gen_vcast(i32, i32, mag_cast_fn_builtin)
mag_gen_vcast(i32, u64, mag_cast_fn_builtin)
mag_gen_vcast(i32, i64, mag_cast_fn_builtin)

mag_gen_vcast(u64, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(u64, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(u64, u8, mag_cast_fn_builtin)
mag_gen_vcast(u64, i8, mag_cast_fn_builtin)
mag_gen_vcast(u64, u16, mag_cast_fn_builtin)
mag_gen_vcast(u64, i16, mag_cast_fn_builtin)
mag_gen_vcast(u64, u32, mag_cast_fn_builtin)
mag_gen_vcast(u64, i32, mag_cast_fn_builtin)
mag_gen_vcast(u64, u64, mag_cast_fn_builtin)
mag_gen_vcast(u64, i64, mag_cast_fn_builtin)
mag_gen_vcast(i64, e8m23, mag_cast_fn_builtin)
mag_gen_vcast(i64, e5m10, mag_cast_fn_e8m232e5m10)
mag_gen_vcast(i64, u8, mag_cast_fn_builtin)
mag_gen_vcast(i64, i8, mag_cast_fn_builtin)
mag_gen_vcast(i64, u16, mag_cast_fn_builtin)
mag_gen_vcast(i64, i16, mag_cast_fn_builtin)
mag_gen_vcast(i64, u32, mag_cast_fn_builtin)
mag_gen_vcast(i64, i32, mag_cast_fn_builtin)
mag_gen_vcast(i64, u64, mag_cast_fn_builtin)
mag_gen_vcast(i64, i64, mag_cast_fn_builtin)

#undef mag_gen_vcast

/* SIMD fast paths */

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
        o[i] = mag_e8m23_to_e5m10(x[i]);
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
        o[i] = mag_e5m10_to_e8m23(x[i]);
}

/* Src -> Dst */
static mag_vcast_fn_t *const mag_cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM] = {
    [MAG_DTYPE_E8M23] = {
        [MAG_DTYPE_E8M23] = mag_vcast_e8m23_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_e8m23_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_e8m23_u8,   /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_e8m23_u8,
        [MAG_DTYPE_I8]    = mag_vcast_e8m23_i8,
        [MAG_DTYPE_U16]   = mag_vcast_e8m23_u16,
        [MAG_DTYPE_I16]   = mag_vcast_e8m23_i16,
        [MAG_DTYPE_U32]   = mag_vcast_e8m23_u32,
        [MAG_DTYPE_I32]   = mag_vcast_e8m23_i32,
        [MAG_DTYPE_U64]   = mag_vcast_e8m23_u64,
        [MAG_DTYPE_I64]   = mag_vcast_e8m23_i64,
    },
    [MAG_DTYPE_E5M10] = {
        [MAG_DTYPE_E8M23] = mag_vcast_e5m10_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_e5m10_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_e5m10_u8,   /* via float32 -> int, same as u8 */
        [MAG_DTYPE_U8]    = mag_vcast_e5m10_u8,
        [MAG_DTYPE_I8]    = mag_vcast_e5m10_i8,
        [MAG_DTYPE_U16]   = mag_vcast_e5m10_u16,
        [MAG_DTYPE_I16]   = mag_vcast_e5m10_i16,
        [MAG_DTYPE_U32]   = mag_vcast_e5m10_u32,
        [MAG_DTYPE_I32]   = mag_vcast_e5m10_i32,
        [MAG_DTYPE_U64]   = mag_vcast_e5m10_u64,
        [MAG_DTYPE_I64]   = mag_vcast_e5m10_i64,
    },
    [MAG_DTYPE_BOOL] = {
        [MAG_DTYPE_E8M23] = mag_vcast_u8_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_u8_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_u8_u8,
        [MAG_DTYPE_U8]    = mag_vcast_u8_u8,
        [MAG_DTYPE_I8]    = mag_vcast_u8_i8,
        [MAG_DTYPE_U16]   = mag_vcast_u8_u16,
        [MAG_DTYPE_I16]   = mag_vcast_u8_i16,
        [MAG_DTYPE_U32]   = mag_vcast_u8_u32,
        [MAG_DTYPE_I32]   = mag_vcast_u8_i32,
        [MAG_DTYPE_U64]   = mag_vcast_u8_u64,
        [MAG_DTYPE_I64]   = mag_vcast_u8_i64,
    },
    [MAG_DTYPE_U8] = {
        [MAG_DTYPE_E8M23] = mag_vcast_u8_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_u8_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_u8_u8,   /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_u8_u8,
        [MAG_DTYPE_I8]    = mag_vcast_u8_i8,
        [MAG_DTYPE_U16]   = mag_vcast_u8_u16,
        [MAG_DTYPE_I16]   = mag_vcast_u8_i16,
        [MAG_DTYPE_U32]   = mag_vcast_u8_u32,
        [MAG_DTYPE_I32]   = mag_vcast_u8_i32,
        [MAG_DTYPE_U64]   = mag_vcast_u8_u64,
        [MAG_DTYPE_I64]   = mag_vcast_u8_i64,
    },
    [MAG_DTYPE_I8] = {
        [MAG_DTYPE_E8M23] = mag_vcast_i8_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_i8_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_i8_u8,   /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_i8_u8,
        [MAG_DTYPE_I8]    = mag_vcast_i8_i8,
        [MAG_DTYPE_U16]   = mag_vcast_i8_u16,
        [MAG_DTYPE_I16]   = mag_vcast_i8_i16,
        [MAG_DTYPE_U32]   = mag_vcast_i8_u32,
        [MAG_DTYPE_I32]   = mag_vcast_i8_i32,
        [MAG_DTYPE_U64]   = mag_vcast_i8_u64,
        [MAG_DTYPE_I64]   = mag_vcast_i8_i64,
    },
    [MAG_DTYPE_U16] = {
        [MAG_DTYPE_E8M23] = mag_vcast_u16_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_u16_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_u16_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_u16_u8,
        [MAG_DTYPE_I8]    = mag_vcast_u16_i8,
        [MAG_DTYPE_U16]   = mag_vcast_u16_u16,
        [MAG_DTYPE_I16]   = mag_vcast_u16_i16,
        [MAG_DTYPE_U32]   = mag_vcast_u16_u32,
        [MAG_DTYPE_I32]   = mag_vcast_u16_i32,
        [MAG_DTYPE_U64]   = mag_vcast_u16_u64,
        [MAG_DTYPE_I64]   = mag_vcast_u16_i64,
    },
    [MAG_DTYPE_I16] = {
        [MAG_DTYPE_E8M23] = mag_vcast_i16_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_i16_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_i16_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_i16_u8,
        [MAG_DTYPE_I8]    = mag_vcast_i16_i8,
        [MAG_DTYPE_U16]   = mag_vcast_i16_u16,
        [MAG_DTYPE_I16]   = mag_vcast_i16_i16,
        [MAG_DTYPE_U32]   = mag_vcast_i16_u32,
        [MAG_DTYPE_I32]   = mag_vcast_i16_i32,
        [MAG_DTYPE_U64]   = mag_vcast_i16_u64,
        [MAG_DTYPE_I64]   = mag_vcast_i16_i64,
    },
    [MAG_DTYPE_U32] = {
        [MAG_DTYPE_E8M23] = mag_vcast_u32_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_u32_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_u32_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_u32_u8,
        [MAG_DTYPE_I8]    = mag_vcast_u32_i8,
        [MAG_DTYPE_U16]   = mag_vcast_u32_u16,
        [MAG_DTYPE_I16]   = mag_vcast_u32_i16,
        [MAG_DTYPE_U32]   = mag_vcast_u32_u32,
        [MAG_DTYPE_I32]   = mag_vcast_u32_i32,
        [MAG_DTYPE_U64]   = mag_vcast_u32_u64,
        [MAG_DTYPE_I64]   = mag_vcast_u32_i64,
    },
    [MAG_DTYPE_I32] = {
        [MAG_DTYPE_E8M23] = mag_vcast_i32_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_i32_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_i32_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_i32_u8,
        [MAG_DTYPE_I8]    = mag_vcast_i32_i8,
        [MAG_DTYPE_U16]   = mag_vcast_i32_u16,
        [MAG_DTYPE_I16]   = mag_vcast_i32_i16,
        [MAG_DTYPE_U32]   = mag_vcast_i32_u32,
        [MAG_DTYPE_I32]   = mag_vcast_i32_i32,
        [MAG_DTYPE_U64]   = mag_vcast_i32_u64,
        [MAG_DTYPE_I64]   = mag_vcast_i32_i64,
    },
    [MAG_DTYPE_U64] = {
        [MAG_DTYPE_E8M23] = mag_vcast_u64_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_u64_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_u64_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_u64_u8,
        [MAG_DTYPE_I8]    = mag_vcast_u64_i8,
        [MAG_DTYPE_U16]   = mag_vcast_u64_u16,
        [MAG_DTYPE_I16]   = mag_vcast_u64_i16,
        [MAG_DTYPE_U32]   = mag_vcast_u64_u32,
        [MAG_DTYPE_I32]   = mag_vcast_u64_i32,
        [MAG_DTYPE_U64]   = mag_vcast_u64_u64,
        [MAG_DTYPE_I64]   = mag_vcast_u64_i64,
    },
    [MAG_DTYPE_I64] = {
        [MAG_DTYPE_E8M23] = mag_vcast_i64_e8m23,
        [MAG_DTYPE_E5M10] = mag_vcast_i64_e5m10,
        [MAG_DTYPE_BOOL]  = mag_vcast_i64_u8,  /* bool uses u8 kernels */
        [MAG_DTYPE_U8]    = mag_vcast_i64_u8,
        [MAG_DTYPE_I8]    = mag_vcast_i64_i8,
        [MAG_DTYPE_U16]   = mag_vcast_i64_u16,
        [MAG_DTYPE_I16]   = mag_vcast_i64_i16,
        [MAG_DTYPE_U32]   = mag_vcast_i64_u32,
        [MAG_DTYPE_I32]   = mag_vcast_i64_i32,
        [MAG_DTYPE_U64]   = mag_vcast_i64_u64,
        [MAG_DTYPE_I64]   = mag_vcast_i64_i64,
    },
};

static MAG_HOTPROC void mag_cast_generic(const mag_kernel_payload_t *payload) {
    mag_tensor_t *r = mag_cmd_out(0);
    const mag_tensor_t *x = mag_cmd_in(0);
    mag_dtype_t src = x->dtype;
    mag_dtype_t dst = r->dtype;
    const mag_dtype_meta_t *msrc = mag_dtype_meta_of(src);
    const mag_dtype_meta_t *mdst = mag_dtype_meta_of(dst);
    mag_vcast_fn_t *kernel = mag_cast_table_2D[src][dst];
    mag_assert(kernel, "No kernel found for type cast: %s -> %s", msrc->name, mdst->name);
    int64_t numel = r->numel;
    uint8_t *br = mag_u8p_mut(r);
    const uint8_t *bx = mag_u8p(x);
    if (mag_full_cont2(r, x)) {
        (*kernel)(r->numel, br, bx);
        return;
    }
    /* We work in byte granularity and compute pointer offsets manually to avoid a generic for this stub function */
    mag_coords_iter_t cr, cx;
    mag_coords_iter_init(&cr, &r->coords);
    mag_coords_iter_init(&cx, &x->coords);
    int64_t ssrc = (int64_t)msrc->size;
    int64_t sdst = (int64_t)mdst->size;
    for (int64_t i=0; i < numel; ++i) { /* TODO: Optimize - Slow with the single indirect call for each element */
        int64_t ri, xi;
        mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi);
        void *pr = br + ri*sdst;
        const void *px = bx + xi*ssrc;
        mag_bnd_chk(px, bx, mag_tensor_get_data_size(x));
        mag_bnd_chk(pr, br, mag_tensor_get_data_size(r));
        (*kernel)(1, pr, px);
    }
}
