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

#ifndef MAG_FLOAT8_E4M3FN_H
#define MAG_FLOAT8_E4M3FN_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 8 bit floating point type - 4 exponent bits + 3 mantissa bits + sign, bias=7 */
typedef struct mag_float8_e4m3fn_t { uint8_t bits; } mag_float8_e4m3fn_t;
mag_static_assert(sizeof(mag_float8_e4m3fn_t) == 1);

#define mag_float8_e4m3fnc(x) (mag_float8_e4m3fn_t){(x)&255}

#define MAG_FLOAT8_E4M3FN_EPS mag_float8_e4m3fnc(0x20)
#define MAG_FLOAT8_E4M3FN_MAX mag_float8_e4m3fnc(0x7e)
#define MAG_FLOAT8_E4M3FN_MAX_SUBNORMAL mag_float8_e4m3fnc(0x07)
#define MAG_FLOAT8_E4M3FN_MIN mag_float8_e4m3fnc(0xfe)
#define MAG_FLOAT8_E4M3FN_MIN_POS mag_float8_e4m3fnc(0x08)
#define MAG_FLOAT8_E4M3FN_MIN_POS_SUBNORMAL mag_float8_e4m3fnc(0x01)
#define MAG_FLOAT8_E4M3FN_NAN mag_float8_e4m3fnc(0x7f)
#define MAG_FLOAT8_E4M3FN_NEG_ONE mag_float8_e4m3fnc(0xb8)
#define MAG_FLOAT8_E4M3FN_NEG_ZERO mag_float8_e4m3fnc(0x80)
#define MAG_FLOAT8_E4M3FN_ONE mag_float8_e4m3fnc(0x38)
#define MAG_FLOAT8_E4M3FN_ZERO mag_float8_e4m3fnc(0x00)

static MAG_AINLINE MAG_CUDA_DEVICE mag_float8_e4m3fn_t mag_float8_e4m3fn_from_float32_soft_fp(float x) {
  uint32_t b;
  memcpy(&b, &x, sizeof(b));
  uint32_t sgn = b&0x80000000u;
  b^=sgn;
  uint8_t r=0;
  if (b >= 0x43f00000u) {
      r = b > 0x7f800000u ? 0x7f : 0x7e;
  } else {
    if (b < 0x3c800000u) {
      uint32_t denorm_mask = 0x46800000;
      float t, dm;
      memcpy(&t, &b, sizeof(t));
      memcpy(&dm, &denorm_mask, sizeof(dm));
      t += dm;
      memcpy(&b, &t, sizeof(t));
      r = (uint8_t)(b-denorm_mask);
    } else {
      uint8_t mant = 1&(b>>20);
      b += ((uint32_t)(7-127)<<23)+0x7ffff;
      b += mant;
      r = (uint8_t)(b>>20);
      r = r == 0x7f ? 0x7e : r;
    }
  }
  r|=(uint8_t)(sgn>>24);
  return (mag_float8_e4m3fn_t){.bits=r};
}

static MAG_AINLINE MAG_CUDA_DEVICE float mag_float8_e4m3fn_to_float32_soft_fp(mag_float8_e4m3fn_t x) {
  /* LUT lookup is actually much slower than scalar conversion on x86-64
  static const uint32_t mag_float8_e4m3fn_lut[256] = {
    0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
    0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
    0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
    0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
    0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
    0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
    0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
    0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
    0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
    0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
    0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
    0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
    0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
    0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
    0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7ff00000,
    0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
    0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
    0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
    0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
    0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
    0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
    0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
    0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
    0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
    0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
    0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
    0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
    0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
    0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
    0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
    0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0xfff00000,
  };
  uint32_t bits = mag_float8_e4m3fn_lut[x.bits];
  float r;
  memcpy(&r, &bits, sizeof r);
  return r;
  */
  uint32_t w = (uint32_t)x.bits<<24;
  uint32_t sgn = w & 0x80000000u;
  uint32_t dat = w & 0x7fffffffu;
  uint32_t renorm;
  #ifdef __CUDA_ARCH__
    renorm = __clz(dat);
  #elif defined(_MSC_VER)
    unsigned long bsr;
    _BitScanReverse(&bsr, (unsigned long)dat);
    renorm = 31^(uint32_t)bsr;
  #else
    renorm = dat ? __builtin_clz(dat) : sizeof(uint32_t)<<3;
  #endif
  renorm = renorm > 4 ? renorm-4 : 0;
  uint32_t r = sgn|((((dat<<renorm>>4)+((0x78-renorm)<<23))|(((int32_t)(dat+0x01000000)>>8) & 0x7f800000))&~((int32_t)(dat-1)>>31));
  float rf;
  memcpy(&rf, &r, sizeof(rf));
  return rf;
}

#ifdef __cplusplus
}
#endif

#endif
