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

#include "mag_snapshot.h"
#include "mag_alloc.h"
#include "mag_romap.h"

struct mag_snapshot_t {
    mag_context_t *ctx;
    mag_map_t metadata_map;
};

mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx) {
    mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
    memset(snap, 0, sizeof(*snap));
    snap->ctx = ctx;
    mag_map_init(&snap->metadata_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, true);
    return snap;
}

void mag_snapshot_metadata_insert(mag_snapshot_t *snap, const char *key, mag_scalar_t record) {
    mag_scalar_t *cloned_val = (*mag_alloc)(NULL, sizeof(*cloned_val), 0);
    *cloned_val = record;
    mag_map_insert(&snap->metadata_map, key, strlen(key), cloned_val);
}

const mag_scalar_t *mag_snapshot_metadata_lookup(mag_snapshot_t *snap, const char *key) {
    return mag_map_lookup(&snap->metadata_map, key, strlen(key));
}

void mag_snapshot_metadata_erase(mag_snapshot_t *snap, const char *key) {
    mag_map_erase(&snap->metadata_map, key, strlen(key));
}

void mag_snapshot_free(mag_snapshot_t *snap) {
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->metadata_map, &iter, &len, &val)) /* Free cloned metadata records */
        (*mag_alloc)(val, 0, 0);
    mag_map_free(&snap->metadata_map);
    memset(snap, 0, sizeof(*snap));
    (*mag_alloc)(snap, 0, 0);
}
