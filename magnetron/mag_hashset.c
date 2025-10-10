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

#include "mag_hashset.h"
#include "mag_alloc.h"

static size_t mag_hashset_compute_hash_size(size_t sz) {
    mag_assert2(sz > 0 && sz < MAG_HASHSET_MAX);
    static const size_t prime_lut[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    size_t l = 0;
    size_t r = sizeof(prime_lut)/sizeof(*prime_lut);
    while (l < r) { /* Binary search for the smallest prime > sz. */
        size_t mid = (l+r)>>1;
        if (prime_lut[mid] < sz) l = mid+1;
        else r = mid;
    }
    return l < sizeof(prime_lut)/sizeof(*prime_lut) ? prime_lut[l] : sz|1;
}

mag_hashset_t mag_hashset_init(size_t size) {
    size = mag_hashset_compute_hash_size(size);
    mag_hashset_t set = {
        .len = size,
        .used = (*mag_alloc)(NULL, mag_bitset_size(size)*sizeof(*set.used), 0),
        .keys = (*mag_alloc)(NULL, size*sizeof(*set.keys), 0),
    };
    memset(set.used, 0, mag_bitset_size(size)*sizeof(*set.used));
    return set;
}

size_t mag_hashset_lookup(mag_hashset_t *set, const mag_tensor_t *key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    while (mag_bitset_get(set->used, i) && set->keys[i] != key) { /* Simple linear probe. */
        i = (i+1) % set->len;
        if (i == k) return MAG_HASHSET_FULL; /* Full */
    }
    return i;
}

bool mag_hashset_contains_key(mag_hashset_t *set, const mag_tensor_t *key) {
    size_t i = mag_hashset_lookup(set, key);
    return mag_bitset_get(set->used, i) && i != MAG_HASHSET_FULL;
}

size_t mag_hashset_insert(mag_hashset_t *set, const mag_tensor_t *key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    do { /* Simple linear probing */
        if (!mag_bitset_get(set->used, i)) { /* Insert key. */
            mag_bitset_set(set->used, i);
            set->keys[i] = key;
            return i;
        }
        if (set->keys[i] == key) return MAG_HASHSET_DUPLICATE; /* Key already exists. */
        i = (i+1) % set->len;
    } while (i != k);
    return MAG_HASHSET_FULL; /* Full */
}

void mag_hashset_reset(mag_hashset_t *set) {
    memset(set->used, 0, mag_bitset_size(set->len)*sizeof(*set->used));
}

void mag_hashset_free(mag_hashset_t *set) {
    (*mag_alloc)(set->used, 0, 0);
    (*mag_alloc)(set->keys, 0, 0);
}
