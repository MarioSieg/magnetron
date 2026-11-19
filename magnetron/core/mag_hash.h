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

#ifndef MAG_HASH_H
#define MAG_HASH_H

#include "mag_def.h"
#include "mag_u128.h"

#ifdef __cplusplus
extern "C" {
#endif

extern MAG_EXPORT mag_uint128_t mag_murmur3_128(const void *key, size_t nb, uint32_t seed);
extern MAG_EXPORT uint64_t mag_murmur3_128_red64(const void *key, size_t nb, uint32_t seed);
static uint64_t mag_hmix64(uint64_t x) { /* splitmix64 finalizer (good avalanche too) */
  x^=x>>30;
  x*=0xbf58476d1ce4e5b9ull;
  x^=x>>27;
  x*=0x94d049bb133111ebull;
  x^=x>>31;
  return x;
}

#ifdef __cplusplus
}
#endif

#endif
