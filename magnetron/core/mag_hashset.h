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

#ifndef MAG_HASHSET_H
#define MAG_HASHSET_H

#include "mag_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed bitset. */
typedef uint64_t mag_bitset64_t;
mag_static_assert(sizeof(mag_bitset64_t) == 8);
#define MAG_BITSET_SHR 6 /* log2(sizeof(mag_bitset32_t)*8) */
#define MAG_BITSET_MASK ((sizeof(mag_bitset64_t)<<3)-1)
#define mag_bitset_size(n) (((n)+MAG_BITSET_MASK)>>MAG_BITSET_SHR)
#define mag_bitset_get(bs, i) (!!((bs)[(i)>>MAG_BITSET_SHR]&(1ull<<((i)&MAG_BITSET_MASK))))
#define mag_bitset_set(bs, i) ((bs)[(i)>>MAG_BITSET_SHR]|=(1ull<<((i)&MAG_BITSET_MASK)))
#define mag_bitset_clear(bs, i) ((bs)[(i)>>MAG_BITSET_SHR]&=~(1ull<<((i)&MAG_BITSET_MASK)))

/* Tensor hashset with linear probing. */
typedef struct mag_hashset_t {
    size_t cap;
    size_t len;
    mag_bitset64_t *used;
    const mag_tensor_t **keys;
} mag_hashset_t;
#define MAG_HASHSET_FULL ((size_t)-1)
#define MAG_HASHSET_DUPLICATE ((size_t)-2)
#define mag_hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

extern void mag_hashset_init(mag_hashset_t *set, size_t cap);
extern bool mag_hashset_reserve(mag_hashset_t *set, size_t min_cap);
extern size_t mag_hashset_lookup(mag_hashset_t *set, const mag_tensor_t *key);
extern bool mag_hashset_contains_key(mag_hashset_t *set, const mag_tensor_t *key);
extern size_t mag_hashset_insert(mag_hashset_t *set, const mag_tensor_t *key);
extern void mag_hashset_reset(mag_hashset_t *set);
extern void mag_hashset_free(mag_hashset_t *set);

#ifdef __cplusplus
}
#endif

#endif
