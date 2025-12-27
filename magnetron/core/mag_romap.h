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

#ifndef MAG_ROMAP_H
#define MAG_ROMAP_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_bucket_t {
    void *key;
    void *val;
    uint64_t hash;
    uint32_t len;
    uint32_t psl;
} mag_bucket_t;

/*
** Robin Hood Hash Map
** When clone_keys = false, the map does not take ownership of the keys.
*/
typedef struct mag_map_t {
    bool clone_keys;
    size_t size;
    size_t nitems;
    size_t minsize;
    mag_bucket_t  *buckets;
    uint32_t hash_seed;
    mag_bucket_t init_bucket;
} mag_map_t;

extern MAG_EXPORT void mag_map_init(mag_map_t *map, size_t cap, bool clone_keys);
extern MAG_EXPORT void *mag_map_lookup(mag_map_t *map, const void *key, size_t len);
extern MAG_EXPORT void *mag_map_insert(mag_map_t *map, const void *key, size_t len, void *val);
extern MAG_EXPORT void *mag_map_erase(mag_map_t *map, const void *key, size_t len);
extern MAG_EXPORT void *mag_map_next(mag_map_t *map, size_t *iter, size_t *len, void **val);
extern MAG_EXPORT void mag_map_free(mag_map_t *map);

#ifdef __cplusplus
}
#endif

#endif
