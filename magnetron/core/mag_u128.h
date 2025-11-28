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

#ifndef MAG_U128_H
#define MAG_U128_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_uint128_t { uint64_t hi; uint64_t lo; } mag_uint128_t;

static inline mag_uint128_t *mag_uint128_add(mag_uint128_t *lhs, uint64_t rhs) {
    uint64_t sum = (lhs->lo + rhs)&(~(uint64_t)0);
    lhs->hi += sum < lhs->lo ? 1 : 0;
    lhs->lo = sum;
    return lhs;
}

static inline uint64_t mag_uint128_mul64(uint32_t x, uint32_t y) { return x*(uint64_t)y; }

static inline mag_uint128_t mag_uint128_mul128(uint64_t x, uint64_t y) {
#ifdef __SIZEOF_INT128__
    unsigned __int128 r = (unsigned __int128)x*(unsigned __int128)y;
    return (mag_uint128_t){(uint64_t)(r>>64), (uint64_t)r};
#else
    uint32_t a = (uint32_t)(x>>32);
    uint32_t b = (uint32_t)x;
    uint32_t c = (uint32_t)(y>>32);
    uint32_t d = (uint32_t)y;
    uint64_t ac = mag_uint128_mul64(a, c);
    uint64_t bc = mag_uint128_mul64(b, c);
    uint64_t ad = mag_uint128_mul64(a, d);
    uint64_t bd = mag_uint128_mul64(b, d);
    uint64_t imm = (bd>>32)+(uint32_t)ad+(uint32_t)bc;
    return (mag_uint128_t){ac+(imm>>32)+(ad>>32)+(bc>>32), (imm<<32)+(uint32_t)(bd)};
#endif
}

static inline uint64_t mag_uint128_mullo128(uint64_t x, uint64_t y) {
#ifdef __SIZEOF_INT128__
    unsigned __int128 r = (unsigned __int128)x*(unsigned __int128)y;
    return (uint64_t)(r>>64);
#else
    uint32_t a = (uint32_t)(x>>32);
    uint32_t b = (uint32_t)x;
    uint32_t c = (uint32_t)(y>>32);
    uint32_t d = (uint32_t)y;
    uint64_t ac = mag_uint128_mul64(a, c);
    uint64_t bc = mag_uint128_mul64(b, c);
    uint64_t ad = mag_uint128_mul64(a, d);
    uint64_t bd = mag_uint128_mul64(b, d);
    uint64_t imm = (bd>>32)+(uint32_t)ad+(uint32_t)bc;
    return ac+(imm>>32)+(ad>>32)+(bc>>32);
#endif
}

static inline mag_uint128_t mag_uint128_mulhi192(uint64_t x, mag_uint128_t y) {
    mag_uint128_t r = mag_uint128_mul128(x, y.hi);
    mag_uint128_add(&r, mag_uint128_mullo128(x, y.lo));
    return r;
}

static inline mag_uint128_t mag_uint128_mullo192(uint64_t x, mag_uint128_t y) {
    uint64_t hi = x*y.hi;
    mag_uint128_t hilo = mag_uint128_mul128(x, y.lo);
    return (mag_uint128_t){(hi+hilo.hi)&(~(uint64_t)0), hilo.lo};
}

static inline uint64_t mag_uint128_mulhi96(uint32_t x, uint64_t y) {
#ifdef __SIZEOF_INT128__
    return mag_uint128_mullo128((uint64_t)x<<32, y);
#else
    uint32_t yh = (uint32_t)(y>>32);
    uint32_t yl = (uint32_t)y;
    uint64_t xyh = mag_uint128_mul64(x, yh);
    uint64_t xyl = mag_uint128_mul64(x, yl);
    return xyh + (xyl>>32);
#endif
}

static inline uint64_t mag_uint128_mullo96(uint32_t x, uint64_t y) {
    return (x*y)&(~(uint64_t)0);
}

#ifdef __cplusplus
}
#endif

#endif
