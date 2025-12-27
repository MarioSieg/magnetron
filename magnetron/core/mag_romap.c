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

#include "mag_romap.h"
#include "mag_alloc.h"
#include "mag_hash.h"
#include "mag_os.h"

static uint32_t mag_map_seed32(void) {
    uint32_t s = 0;
    if (mag_likely(mag_sec_crypto_entropy(&s, sizeof(s)) && s)) return s;
    s = (uint32_t)rand() ^ ((uint32_t)rand()<<16);
    s ^= ((uint32_t)(uintptr_t)&s>>3) ^ ((uint32_t)(uintptr_t)&mag_sec_crypto_entropy>>3);
    s ^= s>>16;
    s *= 0x7feb352du;
    s ^= s>>15;
    s *= 0x846ca68bu;
    s ^= s>>16;
    return s ? s : 0x9e3779b9u;
}

static void mag_map_resize(mag_map_t *map, size_t newsize) {
    mag_assert2(newsize && !(newsize & (newsize-1)));
    mag_assert2(newsize >= map->nitems);
    mag_bucket_t *oldb = map->buckets;
    size_t oldn = map->size;
    mag_bucket_t *newb;
    if (newsize == 1) {
        memset(&map->init_bucket, 0, sizeof(map->init_bucket));
        newb = &map->init_bucket;
    } else {
        newb = (*mag_alloc)(NULL, newsize*sizeof(*newb), 0);
        memset(newb, 0, newsize*sizeof(*newb));
    }
    map->buckets = newb;
    map->size = newsize;
    map->nitems = 0;
    size_t mask = newsize-1;
    for (size_t i = 0; i < oldn; i++) {
        mag_bucket_t *b = oldb+i;
        if (!b->key) continue;
        mag_bucket_t entry = *b;
        entry.psl = 0;
        size_t idx = entry.hash & mask;
        for (;;) {
            mag_bucket_t *dst = &newb[idx];
            if (!dst->key) {
                *dst = entry;
                ++map->nitems;
                break;
            }
            if (entry.psl > dst->psl) {
                mag_bucket_t tmp = *dst;
                *dst = entry;
                entry = tmp;
            }
            ++entry.psl;
            idx = (idx+1) & mask;
        }
    }
    if (oldb && oldb != &map->init_bucket) (*mag_alloc)(oldb, 0, 0);
}

void mag_map_init(mag_map_t *map, size_t cap, bool clone_keys) {
    mag_assert2(cap >= 1 && !(cap & (cap-1)));
    memset(map, 0, sizeof(*map));
    map->clone_keys = clone_keys;
    map->minsize = cap;
    map->hash_seed = mag_map_seed32();
    mag_map_resize(map, cap);
}

void *mag_map_lookup(mag_map_t *map, const void *key, size_t len) {
    mag_assert2(key && len && len <= UINT32_MAX);
    uint64_t hash = mag_murmur3_128_reduced_64(key, len, map->hash_seed);
    size_t mask = map->size-1;
    size_t idx = hash & mask;
    size_t n = 0;
    for (;;) {
        mag_bucket_t *b = map->buckets+idx;
        if (!b->key || n > b->psl) return NULL;
        if (b->hash == hash && b->len  == len && memcmp(b->key, key, len) == 0) return b->val;
        ++n;
        idx = (idx+1) & mask;
    }
}

void *mag_map_insert(mag_map_t *map, const void *key, size_t len, void *val) {
    mag_assert2(key && len && len <= UINT32_MAX);
    if (map->nitems * 100 >= map->size * 85)
        mag_map_resize(map, map->size<<1);
    uint64_t hash = mag_murmur3_128_reduced_64(key, len, map->hash_seed);
    size_t mask = map->size-1;
    size_t idx = hash & mask;
    mag_bucket_t entry;
    if (map->clone_keys) {
        entry.key = (*mag_alloc)(NULL, len, 0);
        memcpy(entry.key, key, len);
    } else {
        entry.key = (void *)(uintptr_t)key;
    }
    entry.val = val;
    entry.hash = hash;
    entry.len = len;
    entry.psl = 0;
    for (;;) {
        mag_bucket_t *b = map->buckets+idx;
        if (!b->key) {
            *b = entry;
            ++map->nitems;
            return val;
        }
        if (b->hash == hash && b->len  == len && memcmp(b->key, key, len) == 0) {
            if (map->clone_keys) (*mag_alloc)(entry.key, 0, 0);
            return b->val;
        }
        if (entry.psl > b->psl) {
            mag_bucket_t tmp = *b;
            *b = entry;
            entry = tmp;
        }
        ++entry.psl;
        idx = (idx+1) & mask;
    }
}

void *mag_map_erase(mag_map_t *map, const void *key, size_t len) {
    mag_assert2(key && len && len <= UINT32_MAX);
    uint64_t hash = mag_murmur3_128_reduced_64(key, len, map->hash_seed);
    size_t mask = map->size-1;
    size_t idx = hash & mask;
    size_t n = 0;
    for (;;) {
        mag_bucket_t *b = &map->buckets[idx];
        if (!b->key || n > b->psl) return NULL;
        if (b->hash == hash && b->len  == len && memcmp(b->key, key, len) == 0) {
            void *val = b->val;
            if (map->clone_keys) (*mag_alloc)(b->key, 0, 0);
            --map->nitems;
            for (;;) {
                size_t next = (idx+1) & mask;
                mag_bucket_t *nb = map->buckets+next;
                if (!nb->key || nb->psl == 0) {
                    memset(b, 0, sizeof(*b));
                    break;
                }
                --nb->psl;
                *b = *nb;
                b = nb;
                idx = next;
            }

            if (map->size > map->minsize && map->nitems * 100 < map->size * 40) mag_map_resize(map, map->size >> 1);
            return val;
        }
        ++n;
        idx = (idx+1) & mask;
    }
}

void *mag_map_next(mag_map_t *map, size_t *iter, size_t *len, void **val) {
    size_t i = *iter;
    while (i < map->size) {
        mag_bucket_t *b = map->buckets+i;
        ++i;
        if (!b->key) continue;
        *iter = i;
        if (len) *len = (size_t)b->len;
        if (val) *val = b->val;
        return b->key;
    }
    return NULL;
}

void mag_map_free(mag_map_t *map) {
    if (map->clone_keys)
        for (size_t i=0; i < map->size; ++i)
            if (map->buckets[i].key)
                (*mag_alloc)(map->buckets[i].key, 0, 0);
    if (map->buckets && map->buckets != &map->init_bucket)
        (*mag_alloc)(map->buckets, 0, 0);
    memset(map, 0, sizeof(*map));
}
