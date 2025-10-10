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

/* Bitset for 32-bit integers. */
typedef uint32_t mag_bitset_t;
mag_static_assert(sizeof(mag_bitset_t) == 4);
#define mag_bitset_size(n) (((n)+((4<<3)-1))>>5)
#define mag_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define mag_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define mag_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define mag_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

/* Tensor hashset with linear probing. */
typedef struct mag_hashset_t {
    size_t len;
    mag_bitset_t *used;
    const mag_tensor_t **keys;
} mag_hashset_t;
#define MAG_HASHSET_FULL ((size_t)-1)
#define MAG_HASHSET_DUPLICATE ((size_t)-2)
#define MAG_HASHSET_MAX ((size_t)-3) /* Must be last. */
#define mag_hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

extern mag_hashset_t mag_hashset_init(size_t size);
extern size_t mag_hashset_lookup(mag_hashset_t *set, const mag_tensor_t *key);
extern bool mag_hashset_contains_key(mag_hashset_t *set, const mag_tensor_t *key);
extern size_t mag_hashset_insert(mag_hashset_t *set, const mag_tensor_t *key);
extern void mag_hashset_reset(mag_hashset_t *set);
extern void mag_hashset_free(mag_hashset_t *set);

#ifdef __cplusplus
}
#endif

#endif
