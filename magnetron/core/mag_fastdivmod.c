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

#include "mag_fastdivmod.h"

static uint64_t mag_u128_div_64(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *r) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t rr;
    __asm__ __volatile__("div %[v]" : "=a"(rr), "=d"(*r) : [v] "r"(d), "a"(lo), "d"(hi));
    return rr;
#else
    uint64_t b = (uint64_t)1<<32;
    if (hi >= d) {
        if (r) *r = ~0ull;
        return ~0ull;
    }
    int s = 63 - (int32_t)mag_fls64(d);
    d <<= s;
    hi <<= s;
    hi |= (lo >> (-s&63)) & (uint64_t)(-(int64_t)s>>63);
    lo <<= s;
    uint32_t n0 = (uint32_t)(lo>>32);
    uint32_t n1 = (uint32_t)(lo&~0u);
    uint32_t d0 = (uint32_t)(d>>32);
    uint32_t d1 = (uint32_t)(d&~0u);
    uint64_t qh = hi / d0;
    uint64_t rh = hi % d0;
    uint64_t c1 = qh*d1;
    uint64_t c2 = rh*b + n0;
    if (c1 > c2) qh -= c1 - c2 > d ? 2 : 1;
    uint32_t q1 = (uint32_t) qh;
    uint64_t rem = hi * b + n0 - q1 * d;
    qh = rem / d0;
    rh = rem % d0;
    c1 = qh*d1;
    c2 = rh*b + n1;
    if (c1 > c2) qh -= c1 - c2 > d ? 2 : 1;
    uint32_t q0 = (uint32_t) qh;
    if (r) *r = (rem*b + n1 - q0*d)>>s;
    return (uint64_t)q1<<32 | q0;
#endif
}

mag_fastdiv_t mag_fastdiv_init(uint64_t d) {
    mag_assert(d != 0, "x/0 is now allowed");
    mag_fastdiv_t result;
    uint32_t fl2d = mag_fls64(d);
    if (!(d & (d-1))) {
        result.magic = 0;
        result.flags = (uint8_t)(fl2d-1);
    } else {
        uint64_t m, rem;
        m = mag_u128_div_64(1ull<<fl2d, 0, d, &rem);
        mag_assert2(rem > 0 && rem < d);
        m += m;
        if (rem+rem >= d || rem+rem < rem) ++m;
        result.magic = 1+m;
        result.flags = (uint8_t) (fl2d|0x40);
    }
    result.flags &= 0x3f;
    return result;
}
