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

#ifndef MAG_BFLOAT16_H
#define MAG_BFLOAT16_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Google Brain Float 16 */
typedef struct mag_bfloat16_t { uint16_t bits; } mag_bfloat16_t;
mag_static_assert(sizeof(mag_bfloat16_t) == 2);

#define mag_bfloat16c(x) (mag_bfloat16_t){(x)&0xffffu}

#define MAG_BFLOAT16_EPS mag_bfloat16c(0x3c00)
#define MAG_BFLOAT16_INF mag_bfloat16c(0x7f80)
#define MAG_BFLOAT16_MAX mag_bfloat16c(0x7f7f)
#define MAG_BFLOAT16_MAX_SUBNORMAL mag_bfloat16c(0x007f)
#define MAG_BFLOAT16_MIN mag_bfloat16c(0xff7f)
#define MAG_BFLOAT16_MIN_POS mag_bfloat16c(0x0080)
#define MAG_BFLOAT16_MIN_POS_SUBNORMAL mag_bfloat16c(0x0001)
#define MAG_BFLOAT16_NAN mag_bfloat16c(0x7fc0)
#define MAG_BFLOAT16_NEG_INF mag_bfloat16c(0xff80)
#define MAG_BFLOAT16_NEG_ONE mag_bfloat16c(0xbf80)
#define MAG_BFLOAT16_NEG_ZERO mag_bfloat16c(0x8000)
#define MAG_BFLOAT16_ONE mag_bfloat16c(0x3f80)
#define MAG_BFLOAT16_ZERO mag_bfloat16c(0x0000)

/*
** Slow (non-hardware accelerated) conversion routines between float32 and bfloat16.
** These routines do not use any special CPU instructions and work on any platform.
** They are provided as a fallback in case hardware support is not available.
** Magnetron's CPU backend contains optimized versions of these functions using SIMD instructions.
*/
static MAG_AINLINE mag_bfloat16_t mag_bfloat16_from_float32_soft_fp(float x) {
    union { float f32; uint32_t u32; } f32u32 = {.f32=x};
    if ((f32u32.u32 & 0x7fffffffu) > 0x7f800000u)
        return MAG_BFLOAT16_NAN;
    uint32_t bias = 0x7fffu + ((f32u32.u32>>16) & 1u);
    return (mag_bfloat16_t){(uint16_t)((f32u32.u32+bias)>>16)};
}

/*
** Slow (non-hardware accelerated) conversion routines between float32 and bfloat16.
** These routines do not use any special CPU instructions and work on any platform.
** They are provided as a fallback in case hardware support is not available.
** Magnetron's CPU backend contains optimized versions of these functions using SIMD instructions.
*/
static MAG_AINLINE float mag_bfloat16_to_float32_soft_fp(mag_bfloat16_t x) {
    union { float f32; uint32_t u32; } f32u32 = {.u32=x.bits};
    f32u32.u32 <<= 16;
    return f32u32.f32;
}

#ifdef __cplusplus
}
#endif

#endif
