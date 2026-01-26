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

#include "mag_hash.h"

static MAG_AINLINE uint64_t mag_rol64(uint64_t x, unsigned r) { return (x<<r) | (x>>(64u-r)); }

static MAG_AINLINE uint64_t mag_fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

static MAG_AINLINE uint64_t mag_loadu64(const void *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

mag_uint128_t mag_murmur3_128(const void *key, size_t nb, uint32_t seed) {
    size_t nblocks = nb>>4;
    uint64_t h1 = seed;
    uint64_t h2 = seed;
    uint64_t c1 = 0x87c37b91114253d5ull;
    uint64_t c2 = 0x4cf5ad432745937full;
    for (size_t i=0; i < nblocks; ++i) {
        uint64_t k1 = mag_loadu64(key + (i<<4) + 0);
        uint64_t k2 = mag_loadu64(key + (i<<4) + 8);
        k1 *= c1;
        k1 = mag_rol64(k1,31);
        k1 *= c2;
        h1 ^= k1;
        h1 = mag_rol64(h1,27);
        h1 += h2;
        h1 = h1*5+0x52dce729;
        k2 *= c2;
        k2 = mag_rol64(k2,33);
        k2 *= c1;
        h2 ^= k2;
        h2 = mag_rol64(h2,31);
        h2 += h1;
        h2 = h2*5+0x38495ab5;
    }
    const uint8_t *tail = (const uint8_t*)key + (nblocks<<4);
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    switch (nb & 0xf) {
        case 15: k2 ^= (uint64_t)tail[14] << 48;
        case 14: k2 ^= (uint64_t)tail[13] << 40;
        case 13: k2 ^= (uint64_t)tail[12] << 32;
        case 12: k2 ^= (uint64_t)tail[11] << 24;
        case 11: k2 ^= (uint64_t)tail[10] << 16;
        case 10: k2 ^= (uint64_t)tail[9] << 8;
        case 9: k2 ^= (uint64_t)tail[8];
            k2 *= c2;
            k2 = mag_rol64(k2,33);
            k2 *= c1;
            h2 ^= k2;
        case 8: k1 ^= (uint64_t)tail[7] << 56;
        case 7: k1 ^= (uint64_t)tail[6] << 48;
        case 6: k1 ^= (uint64_t)tail[5] << 40;
        case 5: k1 ^= (uint64_t)tail[4] << 32;
        case 4: k1 ^= (uint64_t)tail[3] << 24;
        case 3: k1 ^= (uint64_t)tail[2] << 16;
        case 2: k1 ^= (uint64_t)tail[1] << 8;
        case 1: k1 ^= (uint64_t)tail[0];
            k1 *= c1;
            k1 = mag_rol64(k1,31);
            k1 *= c2;
            h1 ^= k1;
    };
    h1 ^= nb;
    h2 ^= nb;
    h1 += h2;
    h2 += h1;
    h1 = mag_fmix64(h1);
    h2 = mag_fmix64(h2);
    h1 += h2;
    h2 += h1;
    return (mag_uint128_t){h1, h2};
}

uint64_t mag_murmur3_128_reduced_64(const void *key, size_t nb, uint32_t seed) {
    mag_uint128_t hash = mag_murmur3_128(key, nb, seed);
    return mag_fmix64(hash.hi ^ hash.lo);
}
