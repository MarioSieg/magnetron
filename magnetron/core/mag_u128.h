/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
#include <core/mag_def.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
#define MAG_HAS_U128
#endif

typedef struct mag_uint128_t {
#ifdef MAG_HAS_U128
  unsigned __int128 v;
#else
  uint64_t hi;
  uint64_t lo;
#endif
} mag_uint128_t;

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_make(uint64_t hi, uint64_t lo) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){((unsigned __int128)hi<<64)|(unsigned __int128)lo};
#else
  return (mag_uint128_t){hi, lo};
#endif
}

MAG_CUDA_DEVICE static inline uint64_t mag_uint128_hi(mag_uint128_t x) {
#ifdef MAG_HAS_U128
  return (uint64_t)(x.v>>64);
#else
  return x.hi;
#endif
}

MAG_CUDA_DEVICE static inline uint64_t mag_uint128_lo(mag_uint128_t x) {
#ifdef MAG_HAS_U128
  return (uint64_t)x.v;
#else
  return x.lo;
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_add(mag_uint128_t lhs, uint64_t rhs) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){lhs.v + (unsigned __int128)rhs};
#else
  uint64_t lo = lhs.lo + rhs;
  uint64_t hi = lhs.hi + (lo < lhs.lo);
  return mag_uint128_make(hi, lo);
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_sub(mag_uint128_t lhs, uint64_t rhs) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){lhs.v - (unsigned __int128)rhs};
#else
  uint64_t lo = lhs.lo - rhs;
  uint64_t hi = lhs.hi - (lhs.lo < rhs);
  return mag_uint128_make(hi, lo);
#endif
}

MAG_CUDA_DEVICE static MAG_CUDA_DEVICE inline mag_uint128_t mag_uint128_mul128(uint64_t x, uint64_t y) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){(unsigned __int128)x*(unsigned __int128)y};
#else
  #ifdef __CUDA_ARCH__
    uint2 a = *(const uint2 *)&x;
    uint2 b = *(const uint2 *)&y;
    uint4 r;
    __asm__ __volatile__(
      "{\n\t"
        "mul.lo.u32 %0, %4, %6;\n\t"
        "mul.hi.u32 %1, %4, %6;\n\t"
        "mad.lo.cc.u32 %1, %4, %7, %1;\n\t"
        "madc.hi.u32 %2, %4, %7, 0;\n\t"
        "mad.lo.cc.u32 %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
        "addc.u32 %3, 0, 0;\n\t"
        "mad.lo.cc.u32 %2, %5, %7, %2;\n\t"
        "madc.hi.u32 %3, %5, %7, %3;\n\t"
      "}\n"
      : "=&r"(r.x), "=&r"(r.y), "=&r"(r.z), "=&r"(r.w)
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y)
      :
    );
    return mag_uint128_make(((uint64_t)r.w<<32)|r.z, ((uint64_t)r.y<<32)|r.x);
  #else
    uint32_t a = (uint32_t)(x>>32);
    uint32_t b = (uint32_t)x;
    uint32_t c = (uint32_t)(y>>32);
    uint32_t d = (uint32_t)y;
    uint64_t ac = a*(uint64_t)c;
    uint64_t bc = b*(uint64_t)c;
    uint64_t ad = a*(uint64_t)d;
    uint64_t bd = b*(uint64_t)d;
    uint64_t imm = (bd>>32) + (uint32_t)ad + (uint32_t)bc;
    return mag_uint128_make(ac + (imm>>32) + (ad>>32) + (bc>>32), (imm<<32) + (uint32_t)bd);
  #endif
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_mul(mag_uint128_t lhs, uint64_t rhs) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){lhs.v * (unsigned __int128)rhs};
#else
  mag_uint128_t lo_mul = mag_uint128_mul128(lhs.lo, rhs);
  return mag_uint128_make(mag_uint128_hi(lo_mul) + lhs.hi*rhs, mag_uint128_lo(lo_mul));
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_div(mag_uint128_t lhs, uint64_t rhs) {
#ifdef MAG_HAS_U128
  return (mag_uint128_t){lhs.v / (unsigned __int128)rhs};
#else
  mag_uint128_t q = mag_uint128_make(0, 0);
  uint64_t r=0;
  for (int i=127; i >= 0; --i) {
    uint64_t bit = i >= 64 ? (lhs.hi>>(i-64))&1 : (lhs.lo>>i)&1;
    bool ge = (r >= ((rhs+1u)>>1));
    r = (r<<1)|bit;
    if (ge || r >= rhs) {
      r -= rhs;
      if (i >= 64) q.hi|=UINT64_C(1)<<(i-64);
      else q.lo|=UINT64_C(1)<<i;
    }
  }
  return q;
#endif
}

MAG_CUDA_DEVICE static inline uint64_t mag_uint128_mullo128(uint64_t x, uint64_t y) {
#ifdef MAG_HAS_U128
  return (uint64_t)(((unsigned __int128)x*(unsigned __int128)y)>>64);
#else
  return mag_uint128_hi(mag_uint128_mul128(x, y));
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_mulhi192(uint64_t x, mag_uint128_t y) {
#ifdef MAG_HAS_U128
  unsigned __int128 r = (unsigned __int128)x*(unsigned __int128)mag_uint128_hi(y);
  r += (unsigned __int128)mag_uint128_mullo128(x, mag_uint128_lo(y));
  return (mag_uint128_t){r};
#else
  return mag_uint128_add(mag_uint128_mul128(x, y.hi), mag_uint128_mullo128(x, y.lo));
#endif
}

MAG_CUDA_DEVICE static inline mag_uint128_t mag_uint128_mullo192(uint64_t x, mag_uint128_t y) {
#ifdef MAG_HAS_U128
  unsigned __int128 lo = (unsigned __int128)x*(unsigned __int128)mag_uint128_lo(y);
  unsigned __int128 hi = (unsigned __int128)(x*mag_uint128_hi(y))<<64;
  return (mag_uint128_t){lo+hi};
#else
  uint64_t hi = x*y.hi;
  mag_uint128_t hilo = mag_uint128_mul128(x, y.lo);
  return mag_uint128_make(hi + hilo.hi, hilo.lo);
#endif
}

MAG_CUDA_DEVICE static inline uint64_t mag_uint128_mulhi96(uint32_t x, uint64_t y) {
#ifdef MAG_HAS_U128
  return (uint64_t)((((unsigned __int128)x<<32)*(unsigned __int128)y)>>64);
#else
  uint32_t yh = (uint32_t)(y>>32);
  uint32_t yl = (uint32_t)y;
  uint64_t xyh = x*(uint64_t)yh;
  uint64_t xyl = x*(uint64_t)yl;
  return xyh + (xyl>>32);
#endif
}

#ifdef __cplusplus
}
#endif

#endif
