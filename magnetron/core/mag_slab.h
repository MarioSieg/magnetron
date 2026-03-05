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

#ifndef MAG_POOL_H
#define MAG_POOL_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Memory chunk for intrusive memory pool. */
typedef struct mag_slab_chunk_t mag_slab_chunk_t;
struct mag_slab_chunk_t {
    uint8_t *bot;               /* Bottom (base) of chunk */
    uint8_t *top;               /* Top of chunk, grows downwards towards bottom */
    uint8_t *top_cap;           /* Top capacity of chunk */
    mag_slab_chunk_t *next;     /* Link to next chunk */
};

/* Fixed-size slab allocator with intrusive freelist. */
typedef struct mag_slab_alloc_t {
    size_t block_size;
    size_t block_align;
    size_t blocks_per_chunk;
    mag_slab_chunk_t *chunks;
    mag_slab_chunk_t *chunk_tail;
    void *free_list;
    uint32_t num_freelist_hits;
    uint32_t num_pool_hits;
    uint32_t num_chunks;
    uint32_t num_allocs;
} mag_slab_alloc_t;

extern MAG_EXPORT void mag_slab_init(mag_slab_alloc_t *pool, size_t block_size, size_t block_align, size_t blocks_per_chunk);
extern MAG_EXPORT void *mag_slab_alloc(mag_slab_alloc_t *pool);
extern MAG_EXPORT void mag_slab_free(mag_slab_alloc_t *pool, void *blk);
extern MAG_EXPORT void mag_slab_destroy(mag_slab_alloc_t *pool);
extern MAG_EXPORT void mag_slab_print_info(mag_slab_alloc_t *pool, const char *name);

#ifdef __cplusplus
}
#endif

#endif
