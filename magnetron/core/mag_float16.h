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
extern mag_float16_t mag_float16_from_float32_soft_fp(float x);

/*
** Slow (non-hardware accelerated) conversion routines between float32 and float16.
** These routines do not use any special CPU instructions and work on any platform.
** They are provided as a fallback in case hardware support is not available.
** Magnetron's CPU backend contains optimized versions of these functions using SIMD instructions.
*/
extern float mag_float16_to_float32_soft_fp(mag_float16_t x);

#ifdef __cplusplus
}
#endif

#endif
