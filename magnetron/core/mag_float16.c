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

#include "mag_float16.h"

mag_float16_t mag_float16_from_float32_soft_fp(float x) {
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

float mag_float16_to_float32_soft_fp(mag_float16_t x) {
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
