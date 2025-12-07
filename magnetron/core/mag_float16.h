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

#ifndef MAG_FLOAT16_H
#define MAG_FLOAT16_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* IEEE 754 16-bit half precision float. */
typedef struct mag_float16_t { uint16_t bits; } mag_float16_t;

#define msg_float16_pack2(x) (mag_float16_t){(x)&0xffffu}

/* Some fp16 constants. */
#define MAG_FLOAT16_E msg_float16_pack2(0x4170)
#define MAG_FLOAT16_EPS msg_float16_pack2(0x1400)
#define MAG_FLOAT16_INF msg_float16_pack2(0x7c00)
#define MAG_FLOAT16_LN10 msg_float16_pack2(0x409b)
#define MAG_FLOAT16_LN2 msg_float16_pack2(0x398c)
#define MAG_FLOAT16_LOG10_2 msg_float16_pack2(0x34d1)
#define MAG_FLOAT16_LOG10_E msg_float16_pack2(0x36f3)
#define MAG_FLOAT16_LOG2_10 msg_float16_pack2(0x42a5)
#define MAG_FLOAT16_LOG2_E msg_float16_pack2(0x3dc5)
#define MAG_FLOAT16_MAX msg_float16_pack2(0x7bff)
#define MAG_FLOAT16_MAX_SUBNORMAL msg_float16_pack2(0x03ff)
#define MAG_FLOAT16_MIN msg_float16_pack2(0xfbff)
#define MAG_FLOAT16_MIN_POS msg_float16_pack2(0x0400)
#define MAG_FLOAT16_MIN_POS_SUBNORMAL msg_float16_pack2(0x0001)
#define MAG_FLOAT16_NAN msg_float16_pack2(0x7e00)
#define MAG_FLOAT16_NEG_INF msg_float16_pack2(0xfc00)
#define MAG_FLOAT16_NEG_ONE msg_float16_pack2(0xbc00)
#define MAG_FLOAT16_NEG_ZERO msg_float16_pack2(0x8000)
#define MAG_FLOAT16_ONE msg_float16_pack2(0x3c00)
#define MAG_FLOAT16_PI msg_float16_pack2(0x4248)
#define MAG_FLOAT16_SQRT2 msg_float16_pack2(0x3da8)
#define MAG_FLOAT16_ZERO msg_float16_pack2(0x0000)

/*
** Slow (non-hardware accelerated) conversion routines between float32 and float16.
** These routines do not use any special CPU instructions and work on any platform.
** They are provided as a fallback in case hardware support is not available.
** Magnetron's CPU backend contains optimized versions of these functions using SIMD instructions.
*/
static inline mag_float16_t mag_float16_from_float32_soft_fp(float x) {
    union { uint32_t u; float f; } u32f32 = {.f=x};
    float base = fabsf(x)*0x1.0p+112f*0x1.0p-110f;
    uint32_t shl1_w = u32f32.u+u32f32.u;
    uint32_t sign = u32f32.u & 0x80000000u;
    u32f32.u = 0x07800000u + (mag_xmax(0x71000000u, shl1_w & 0xff000000u)>>1);
    u32f32.f = base + u32f32.f;
    uint32_t exp_bits = (u32f32.u>>13) & 0x00007c00u;
    uint32_t mant_bits = u32f32.u & 0x00000fffu;
    uint32_t nonsign = exp_bits + mant_bits;
    return (mag_float16_t){.bits=(uint16_t)((sign>>16)|(shl1_w > 0xff000000 ? 0x7e00 : nonsign))};
}

/*
** Slow (non-hardware accelerated) conversion routines between float32 and float16.
** These routines do not use any special CPU instructions and work on any platform.
** They are provided as a fallback in case hardware support is not available.
** Magnetron's CPU backend contains optimized versions of these functions using SIMD instructions.
*/
static inline float mag_float16_to_float32_soft_fp(mag_float16_t x) {
    uint32_t w = (uint32_t)x.bits<<16;
    uint32_t sign = w & 0x80000000u;
    uint32_t two_w = w+w;
    uint32_t offs = 0xe0u<<23;
    uint32_t t1 = (two_w>>4) + offs;
    uint32_t t2 = (two_w>>17) | (126u<<23);
    union { uint32_t u; float f; } u32f32 = {.u=t1};
    float norm_x = u32f32.f*0x1.0p-112f;
    u32f32.u = t2;
    float denorm_x = u32f32.f-0.5f;
    uint32_t denorm_cutoff = 1u<<27;
    uint32_t r = sign | (two_w < denorm_cutoff ? (u32f32.f = denorm_x, u32f32.u) : (u32f32.f = norm_x, u32f32.u));
    u32f32.u = r;
    return u32f32.f;
}

#ifdef __cplusplus
}
#endif

#endif
