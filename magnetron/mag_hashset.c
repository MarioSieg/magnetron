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

#define MAG_HASHSET_LOAD_NUM 7 /* Load factor numerator */
#define MAG_HASHSET_LOAD_DEN 10 /* Load factor denominator */
#define MAG_HASHSET_GROW_THRESHOLD(n) (((n)*(size_t)MAG_HASHSET_LOAD_NUM)/(size_t)MAG_HASHSET_LOAD_DEN)

static size_t mag_hashset_compute_hash_size(size_t sz) {
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

static void mag_hashset_rehash_grow_to(mag_hashset_t *set, size_t new_len) {
    mag_assert2(new_len > 1);
    size_t nb = mag_bitset_size(new_len)*sizeof(*set->used);
    mag_bitset64_t *bt = (*mag_alloc)(NULL, nb, 0);
    const mag_tensor_t **keys = (*mag_alloc)(NULL, new_len*sizeof(*set->keys), 0);
    memset(bt, 0, nb);
    for (size_t i=0; i < set->cap; ++i) {
        if (!mag_bitset_get(set->used, i)) continue;
        const mag_tensor_t *key = set->keys[i];
        size_t k = mag_hashset_hash_fn(key) % new_len;
        size_t j = k;
        do {
            if (!mag_bitset_get(bt, j)) {
                mag_bitset_set(bt, j);
                keys[j] = key;
                break;
            }
            j = (j+1) % new_len;
        } while (j != k);
    }
    (*mag_alloc)(set->used, 0, 0);
    (*mag_alloc)(set->keys, 0, 0);
    set->used = bt;
    set->keys = keys;
    set->cap = new_len;
}

void mag_hashset_init(mag_hashset_t *set, size_t cap) {
    cap = mag_hashset_compute_hash_size(cap ? cap : 2);
    memset(set, 0, sizeof(*set));
    set->cap = cap;
    set->len = 0;
    set->used = (*mag_alloc)(NULL, mag_bitset_size(cap)*sizeof(*set->used), 0);
    set->keys = (*mag_alloc)(NULL, cap*sizeof(*set->keys), 0);
    memset(set->used, 0, mag_bitset_size(cap)*sizeof(*set->used));
}

bool mag_hashset_reserve(mag_hashset_t *set, size_t min_cap){
    size_t mn = (min_cap*(size_t)MAG_HASHSET_LOAD_DEN + (MAG_HASHSET_LOAD_NUM-1)) / MAG_HASHSET_LOAD_NUM;
    if (mn <= set->cap) return false;
    size_t tn = mag_hashset_compute_hash_size(mn);
    mag_hashset_rehash_grow_to(set, tn);
    return true;
}

size_t mag_hashset_lookup(mag_hashset_t *set, const mag_tensor_t *key) {
    size_t k = mag_hashset_hash_fn(key) % set->cap, i = k;
    while (mag_bitset_get(set->used, i) && set->keys[i] != key) { /* Simple linear probe. */
        i = (i+1) % set->cap;
        if (i == k) return MAG_HASHSET_FULL; /* Full */
    }
    return i;
}

bool mag_hashset_contains_key(mag_hashset_t *set, const mag_tensor_t *key) {
    size_t i = mag_hashset_lookup(set, key);
    return i != MAG_HASHSET_FULL && mag_bitset_get(set->used, i);
}

size_t mag_hashset_insert(mag_hashset_t *set, const mag_tensor_t *key) {
    if (set->len+1 > MAG_HASHSET_GROW_THRESHOLD(set->cap)) {
        size_t t = mag_hashset_compute_hash_size(set->cap);
        mag_hashset_rehash_grow_to(set, t);
    }
    size_t k = mag_hashset_hash_fn(key) % set->cap;
    size_t i = k;
    for (;;) {
        do {
            if (!mag_bitset_get(set->used, i)) {
                mag_bitset_set(set->used, i);
                set->keys[i] = key;
                ++set->len;
                return i;
            }
            if (set->keys[i] == key) return MAG_HASHSET_DUPLICATE;
            i = (i+1) % set->cap;
        } while (i != k);
        /* We circled back, rehash and try again */
        size_t t = mag_hashset_compute_hash_size(set->cap<<1);
        mag_hashset_rehash_grow_to(set, t);
        k = mag_hashset_hash_fn(key) % set->cap;
        i = k;
    }
}

void mag_hashset_reset(mag_hashset_t *set) {
    memset(set->used, 0, mag_bitset_size(set->cap)*sizeof(*set->used));
    set->len = 0;
}

void mag_hashset_free(mag_hashset_t *set) {
    (*mag_alloc)(set->used, 0, 0);
    (*mag_alloc)(set->keys, 0, 0);
    memset(set, 0, sizeof(*set));
}
