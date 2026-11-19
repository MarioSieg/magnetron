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

#ifndef MAG_MATH_H
#define MAG_MATH_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

static bool MAG_AINLINE mag_addov64(int64_t a, int64_t b, int64_t *c) {
  #if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, c);
  #else
    uint64_t r = (uint64_t)a+(uint64_t)b;
    *c = (int64_t)r;
    return ((a^(int64_t)r)&(b^(int64_t)r)) < 0;
  #endif
}

static bool MAG_AINLINE mag_mulov64(int64_t a, int64_t b, int64_t *c) {
  #ifdef _MSC_VER
    #ifdef _M_ARM64
      uint64_t high = __umulh(a, b);
      *c = a*b;
      return high != (*c>>63);
    #else
      int64_t high;
      int64_t low = _mul128(a, b, &high);
      int64_t sign = low>>63;
      *c = low;
      return high != sign;
    #endif
  #else
    return __builtin_mul_overflow(a, b, c);
  #endif
}

static MAG_CUDA_DEVICE MAG_AINLINE uint32_t mag_next_pow2_u32(uint32_t x) {
  if (mag_unlikely(x <= 1)) return 1;
  --x;
  x |= x>>1;
  x |= x>>2;
  x |= x>>4;
  x |= x>>8;
  x |= x>>16;
  return x+1;
}

static MAG_CUDA_DEVICE MAG_AINLINE uint64_t mag_powu(uint64_t x, uint64_t k) {
  if (!k) return 1;
  for (; !(k&1); k >>= 1) x *= x;
  uint64_t y = x;
  if (k>>=1) {
    for (;;) {
      x *= x;
      if (k == 1) break;
      if (k&1) y *= x;
      k >>= 1;
    }
    y *= x;
  }
  return y;
}

static MAG_CUDA_DEVICE MAG_AINLINE int64_t mag_powi(int64_t x, int64_t k) {
  if (!k) return 1;
  if (k < 0) {
    switch (x) {
      case 0: return 0;
      case 1: return 1;
      case -1: return (k&1) ? -1 : 1;
      default: return 0;
    }
  }
  return (int64_t)mag_powu((uint64_t)x, (uint64_t)k);
}

/* Remainder functions that adjust for sign, as in Python and MATLAB. C's % operator does not do this 😿 */
static MAG_CUDA_DEVICE MAG_AINLINE float mag_remf(float x, float y) {
  float r = fmodf(x, y);
  if (r != 0.0f && (r < 0.0f) != (y < 0.0f)) r += y;
  return r;
}

/* Trap free signed integral div */
static MAG_CUDA_DEVICE MAG_AINLINE int64_t mag_remi(int64_t x, int64_t y) {
  if (y == 0) return x;
  if (y == -1) return 0;
  int64_t r = x % y;
  if (r != 0 && (r < 0) != (y < 0)) r += y;
  return r;
}

static MAG_CUDA_DEVICE MAG_AINLINE int64_t mag_divi(int64_t x, int64_t y) { /* Truncating division. */
  if (y == 0) return -1;
  if (y == -1) return (int64_t)(UINT64_C(0)-(uint64_t)x); /* Wrapping */
  return x/y;
}

static MAG_CUDA_DEVICE MAG_AINLINE int64_t mag_floordivi(int64_t x, int64_t y) {
  if (y == 0) return -1;
  if (y == -1) return (int64_t)(UINT64_C(0)-(uint64_t)x);
  int64_t q = x/y;
  if (x%y != 0 && (x < 0) != (y < 0)) --q;
  return q;
}

static MAG_CUDA_DEVICE MAG_AINLINE int64_t mag_sat_cvt_f642i64(double x) {
  if (x!=x) return 0;
  if (x >= 9223372036854775808.0) return INT64_MAX;
  if (x <= -9223372036854775808.0) return INT64_MIN;
  return (int64_t)x;
}

static MAG_CUDA_DEVICE MAG_AINLINE uint64_t mag_sat_cvt_f642u64(double x) {
  if (x < 0.0) return (uint64_t)mag_sat_cvt_f642i64(x);
  if (x!=x) return 0;
  if (x >= 18446744073709551616.0) return UINT64_MAX;
  return (uint64_t)x;
}

#define mag_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define mag_vmax(x, y) (((x)>(y))?(x):(y))
#define mag_vmin(x, y) (((x)<(y))?(x):(y))
#define mag_vclamp(v, lo, hi) ((v)<(lo)?(lo):(v)>(hi)?(hi):(v))
#define mag_align_up(x, al) (((x)+(al)-1)&-(al))
#define mag_align_down(x, al) ((x)&-(al))
#define MAG_TAU 6.283185307179586476925286766559005768394338798f /* τ=2π */
#define MAG_INVSQRT2 0.707106781186547524400844362104849039284835937f /* 1/√2 */
#define MAG_SQRT2OVERPI 0.797884560802865355879892119868763736951717263f

static uint16_t MAG_AINLINE mag_bswap16(uint16_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
  x = __builtin_bswap16(x);
#else
  x = (x&0xff00)>>8 | x&0xff<<8;
#endif
  return x;
}
static uint32_t MAG_AINLINE mag_bswap32(uint32_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
  x = __builtin_bswap32(x);
#else
  x = (x&0xff000000)>>24 |
    (x&0xff0000)>>8 |
    (x&0xff00)<<8 |
    (x&0xff)<<24;
#endif
  return x;
}
static uint64_t MAG_AINLINE mag_bswap64(uint64_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
  x = __builtin_bswap64(x);
#else
  x = (x&0xff00000000000000)>>56 |
    (x&0xff000000000000)>>40 |
    (x&0xff0000000000)>>24 |
    (x&0xff00000000)>>8 |
    (x&0xff000000)<<8 |
    (x&0xff0000)<<24 |
    (x&0xff00)<<40 |
    (x&0xff)<<56;
#endif
  return x;
}

static MAG_CUDA_DEVICE MAG_AINLINE uint32_t mag_mulhilo32(uint32_t x, uint32_t y, uint32_t *hi) {
  #ifdef __CUDA_ARCH__
    *hi = __umulhi(x, y);
    return x*y;
  #else
    uint64_t p = (uint64_t)x*(uint64_t)y;
    *hi = p>>32;
    return (uint32_t)p;
  #endif
}

static MAG_AINLINE uint64_t mag_mulhilo64(uint64_t x, uint64_t y) {
#if defined(_MSC_VER) && (!defined(__clang__) || _MSC_VER > 1930) && (defined(_M_X64) || defined(_M_ARM64))
  return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
  unsigned __int128 xl = x, yl = y;
  unsigned __int128 rl = xl*yl;
  return (uint64_t)(rl>>64);
#else
  /* Fastdivmod is slow with this (IPC raise a lot), because we have NO fast compiler intrinsics */
  uint32_t x0 = (uint32_t)(x&~0u);
  uint32_t x1 = (uint32_t)(x>>32);
  uint32_t y0 = (uint32_t)(y&~0u);
  uint32_t y1 = (uint32_t)(y>>32);
  uint32_t x0y0_hi;
  mag_mulhilo32(x0, y0, &x0y0_hi);
  uint64_t x0y1 = x0*(uint64_t)y1;
  uint64_t x1y0 = x1*(uint64_t)y0;
  uint64_t x1y1 = x1*(uint64_t)y1;
  uint64_t temp = x1y0 + x0y0_hi;
  uint64_t tlo = temp&~0u;
  uint64_t thi = temp>>32;
  return x1y1 + thi + ((tlo + x0y1)>>32);
#endif
}

static MAG_AINLINE int mag_popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll((unsigned long long)x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM 64))
  return __popcnt64(x);
#else
  x -= (x>>1)&0x5555555555555555ull;
  x = (x&0x3333333333333333ull)+((x>>2)&0x3333333333333333ull);
  x = (x+(x>>4))&0x0f0f0f0f0f0f0f0full;
  return (x*0x0101010101010101ull)>>56;
#endif
}

#ifdef __cplusplus
}
#endif

#endif
