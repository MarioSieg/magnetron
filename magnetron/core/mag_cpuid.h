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

#ifndef MAG_CPUID_H
#define MAG_CPUID_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__) || defined(_M_X64)

#define mag_amd64_capdef(_, __)\
    _(NONE)__\
    _(INTEL)__\
    _(AMD)__\
    \
    _(SSE)__\
    _(SSE2)__\
    _(SSE3)__\
    _(SSSE3)__\
    _(SSE41)__\
    _(SSE42)__\
    _(SSE4A)__\
    \
    _(AVX)__\
    _(FMA)__\
    _(AVX2)__\
    _(F16C)__\
    _(AVX_VNNI)__\
    _(AVX_VNNI_INT8)__\
    _(AVX_NE_CONVERT)__\
    _(AVX_IFMA)__\
    _(AVX_VNNI_INT16)__\
    _(AVX10)__\
    \
    _(AVX512_F)__\
    _(AVX512_DQ)__\
    _(AVX512_IFMA)__\
    _(AVX512_PF)__\
    _(AVX512_ER)__\
    _(AVX512_CD)__\
    _(AVX512_BW)__\
    _(AVX512_VL)__\
    _(AVX512_VBMI)__\
    _(AVX512_4VNNIW)__\
    _(AVX512_4FMAPS)__\
    _(AVX512_VBMI2)__\
    _(AVX512_VNNI)__\
    _(AVX512_BITALG)__\
    _(AVX512_VPOPCNTDQ)__\
    _(AVX512_BF16)__\
    _(AVX512_VP2INTERSECT)__\
    _(AVX512_FP16)__\
    \
    _(AMX_TILE)__\
    _(AMX_INT8)__\
    _(AMX_BF16)__\
    _(AMX_FP16)__\
    _(AMX_TRANSPOSE)__\
    _(AMX_TF32)__\
    _(AMX_AVX512)__\
    _(AMX_MOVRS)__\
    _(AMX_FP8)__\
    \
    _(BMI1)__\
    _(BMI2)__\
    \
    _(OSXSAVE)__\
    _(GFNI)__\
    _(APX_F)__

typedef enum mag_amd64_cap_t {
#define _(name) MAG_AMD64_CAP_##name
    mag_amd64_capdef(_, MAG_SEP)
    MAG_AMD64_CAP__NUM
#undef _
} mag_amd64_cap_t;
typedef uint64_t mag_amd64_cap_bitset_t;
mag_static_assert(MAG_AMD64_CAP__NUM <= sizeof(mag_amd64_cap_bitset_t)<<3); /* Must fit in 64 bits. */
#define mag_amd64_cap_bit(x) (((mag_amd64_cap_bitset_t)1)<<((x)&63))
#define mag_amd64_cap(name) mag_amd64_cap_bit(MAG_AMD64_CAP_##name)

extern const char *const mag_amd64_cpu_cap_names[MAG_AMD64_CAP__NUM]; /* Names of x86-64 CPU capabilities. */
extern void mag_probe_cpu_cache_topology(mag_amd64_cap_bitset_t caps, size_t *ol1, size_t *ol2, size_t *ol3);
extern void mag_probe_cpu_amd64(mag_amd64_cap_bitset_t *o, uint32_t *avx10ver);

#elif defined(__aarch64__) || defined(_M_ARM64) /* ARM 64 specific CPU features. */

#define mag_armd64_capdef(_, __) /* Enumerator */\
    _(NONE)__\
    _(NEON)__\
    _(DOTPROD)__\
    _(I8MM)__\
    _(F16SCALAR)__\
    _(F16VECTOR)__\
    _(F16CVT)__\
    _(BF16)__\
    _(SVE)__\
    _(SVE2)__

typedef enum mag_arm64_cap_t {
#define _(ident) MAG_ARM64_CAP_##ident
    mag_armd64_capdef(_, MAG_SEP)
    MAG_ARM64_CAP__NUM
#undef _
} mag_arm64_cap_t;
typedef uint64_t mag_arm64_cap_bitset_t;
mag_static_assert(MAG_ARM64_CAP__NUM <= sizeof(mag_arm64_cap_bitset_t)<<3); /* Must fit in 64 bits. */
#define mag_arm64_cap_bit(x) (((mag_arm64_cap_bitset_t)1)<<((x)&63))
#define mag_arm64_cap(name) mag_arm64_cap_bit(MAG_ARM64_CAP_##name)

extern const char *const mag_arm64_cpu_cap_names[MAG_ARM64_CAP__NUM];
extern void mag_probe_cpu_arm64(mag_arm64_cap_bitset_t *o, int64_t *sve_width);
extern void mag_probe_cpu_cache_topology(mag_arm64_cap_bitset_t caps, size_t *ol1, size_t *ol2, size_t *ol3);

#endif

#ifdef __cplusplus
}
#endif

#endif
