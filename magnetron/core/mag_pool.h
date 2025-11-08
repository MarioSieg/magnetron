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

#ifndef MAG_POOL_H
#define MAG_POOL_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Memory chunk for intrusive memory pool. */
typedef struct mag_pool_chunk_t mag_pool_chunk_t;
struct mag_pool_chunk_t {
    uint8_t *bot;          /* Bottom (base) of chunk */
    uint8_t *top;          /* Top of chunk, grows downwards towards bottom */
    mag_pool_chunk_t *next;   /* Link to next chunk */
};

/* Fast memory allocator for memory blocks of same size. Obtains a memory pool and freelist for fast de/allocation. */
typedef struct mag_fixed_pool_t {
    size_t block_size;                   /* Size of each allocated block */
    size_t block_align;                  /* Alignment requirements of each block. */
    size_t blocks_per_chunk;             /* How many blocks fit in each chunk */
    mag_pool_chunk_t *chunks;      /* Linked list of all chunks */
    mag_pool_chunk_t *chunk_head;  /* Last chunk */
    void *free_list;            /* Intrusive single linked list of free chunks */
    uint64_t num_freelist_hits;          /* Number of cache (free-list) hits */
    uint64_t num_pool_hits;              /* Number of cache (pool) hits */
    uint64_t num_chunks;                 /* Number of used chunks */
    uint64_t num_allocs;                 /* Number of total allocations */
} mag_fixed_pool_t;

extern MAG_EXPORT void mag_fixed_pool_init(mag_fixed_pool_t *pool, size_t block_size, size_t block_align, size_t blocks_per_chunk);
extern MAG_EXPORT void *mag_fixed_pool_alloc_block(mag_fixed_pool_t *pool);
extern MAG_EXPORT void mag_fixed_pool_free_block(mag_fixed_pool_t *pool, void *blk);
extern MAG_EXPORT void mag_fixed_pool_destroy(mag_fixed_pool_t *pool);
extern MAG_EXPORT void mag_fixed_pool_print_info(mag_fixed_pool_t *pool, const char *name);

#ifdef __cplusplus
}
#endif

#endif
