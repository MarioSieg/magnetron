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

#ifndef MAG_FASTDIVMOD_H
#define MAG_FASTDIVMOD_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* It's usually only worth to use if MAG_FASTDIVMOD_FAST is defined, since this impl requires special intrinsics.
** On platforms or compiler which do not support __int128 or __umulh, fastdivmod is usually slower than the normal instructions.
** So test for the macro and use it conditionally.
*/

static MAG_AINLINE uint32_t mag_mullhi_u32(uint32_t x, uint32_t y) {
    uint64_t xl = x, yl = y;
    uint64_t rl = xl*yl;
    return (uint32_t)(rl>>32);
}

static MAG_AINLINE uint64_t mag_mullhi_u64(uint64_t x, uint64_t y) {
#if defined(_MSC_VER) && (!defined(__clang__) || _MSC_VER > 1930) && (defined(_M_X64) || defined(_M_ARM64))
#define MAG_FASTDIVMOD_FAST /* Fastdivmod is ACTUALLY fast because we have fast compiler intrinsics */
    return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
#define MAG_FASTDIVMOD_FAST /* Fastdivmod is ACTUALLY fast because we have fast compiler intrinsics */
    unsigned __int128 xl = x, yl = y;
    unsigned __int128 rl = xl * yl;
    return (uint64_t)(rl >> 64);
#else
    /* Fastdivmod is slow with this (IPC raise a lot), because we have NO fast compiler intrinsics */
    uint32_t x0 = (uint32_t)(x&~0u);
    uint32_t x1 = (uint32_t)(x>>32);
    uint32_t y0 = (uint32_t)(y&~0u);
    uint32_t y1 = (uint32_t)(y>>32);
    uint32_t x0y0_hi = mag_mullhi_u32(x0, y0);
    uint64_t x0y1 = x0*(uint64_t)y1;
    uint64_t x1y0 = x1*(uint64_t)y0;
    uint64_t x1y1 = x1*(uint64_t)y1;
    uint64_t temp = x1y0 + x0y0_hi;
    uint64_t tlo = temp&~0u;
    uint64_t thi = temp>>32;
    return x1y1 + thi + ((tlo + x0y1)>>32);
#endif
}

typedef struct mag_fastdiv_t {
    uint64_t magic;
    uint8_t flags;
} mag_fastdiv_t;

extern MAG_EXPORT mag_fastdiv_t mag_fastdiv_init(uint64_t d);

static MAG_AINLINE uint64_t mag_fastdiv_eval(uint64_t numer, const mag_fastdiv_t *denom) {
    uint64_t q = mag_mullhi_u64(numer, denom->magic);
    return (((numer-q)>>1)+q)>>denom->flags;
}

#ifdef __cplusplus
}
#endif

#endif
