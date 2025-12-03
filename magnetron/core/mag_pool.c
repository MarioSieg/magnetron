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

#include "mag_pool.h"
#include "mag_alloc.h"

/* Allocate a new linear chunk for a fixed pool. */
static mag_pool_chunk_t *mag_fixed_pool_chunk_new(size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    size_t cap = blocks_per_chunk*block_size;
    uintptr_t size = 0;
    mag_pincr((void **)&size, sizeof(mag_pool_chunk_t), __alignof(mag_pool_chunk_t));
    mag_pincr((void **)&size, cap, block_align);
    void *base = (*mag_alloc)(NULL, size, 0), *pos = base;
    mag_pool_chunk_t *chunk = mag_pincr(&pos, sizeof(mag_pool_chunk_t), __alignof(mag_pool_chunk_t));
    uint8_t *bot = mag_pincr(&pos, cap, block_align);
    *chunk = (mag_pool_chunk_t) {
        .bot = bot,
        .top = bot+cap,
        .next = NULL
    };
    return chunk;
}

/* Initialize fixed intrusive pool and allocate start chunk. */
void mag_fixed_pool_init(mag_fixed_pool_t *pool, size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    mag_assert2(blocks_per_chunk);
    block_size = mag_xmax(sizeof(void *), block_size); /* Ensure block size is at least sizeof(void*) to store intrusive free list. */
    mag_pool_chunk_t *chunk = mag_fixed_pool_chunk_new(block_size, block_align, blocks_per_chunk);
    *pool = (mag_fixed_pool_t) {
        .block_size = block_size,
        .block_align = block_align,
        .blocks_per_chunk = blocks_per_chunk,
        .chunks = chunk,
        .chunk_head = chunk,
        .free_list = NULL,
        .num_freelist_hits = 0,
        .num_pool_hits = 0,
        .num_chunks = 1,
        .num_allocs = 0
    };
}

/* Allocate a new fixed block from the pool. Memory is uninitialized. */
void *mag_fixed_pool_alloc_block(mag_fixed_pool_t *pool) {
    ++pool->num_allocs;
    if (mag_likely(pool->free_list)) { /* 1. Try to pop from free_list (fastest path) */
        ++pool->num_freelist_hits;
        void *blk = pool->free_list;
        pool->free_list = *(void **)blk; /* Next free block is stored at block [0..sizeof(void*)-1] */
        return blk;
    }
    mag_pool_chunk_t *chunk = pool->chunk_head;
    mag_assert2(chunk);
    uint8_t *top = chunk->top-pool->block_size;
    if (mag_likely(top >= chunk->bot)) {  /* 2. Allocate from the last pool if possible (fast path) */
        ++pool->num_pool_hits;
        chunk->top = top;
        return top;
    }
    /* 3. Current chunk is exhausted, allocate new (slow path) */
    mag_pool_chunk_t *new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk);
    chunk->next = new_chunk;
    pool->chunk_head = new_chunk;
    new_chunk->top -= pool->block_size;
    ++pool->num_chunks;
    return new_chunk->top;
}

/* Free a fixed block back to the pool. This effectively pushes it into the freelist. */
void mag_fixed_pool_free_block(mag_fixed_pool_t *pool, void *blk) {
    *(void **)blk = pool->free_list;
    pool->free_list = blk;
}

/* Destroy fixed intrusive pool and free all allocated memory. */
void mag_fixed_pool_destroy(mag_fixed_pool_t *pool) {
    mag_pool_chunk_t *chunk = pool->chunks;
    while (chunk) {
        mag_pool_chunk_t *next = chunk->next;
        (*mag_alloc)(chunk, 0, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

/* Print pool information and allocation stats. */
MAG_COLDPROC void mag_fixed_pool_print_info(mag_fixed_pool_t *pool, const char *name) {
    mag_log_info("Fixed Intrusive Pool: %s", name);
    mag_log_info(
        "\tBlock Size: %zu B, Block Align: %zu B, Blocks Per Chunk: %zu B",
        pool->block_size,
        pool->block_align,
        pool->blocks_per_chunk
    );
    mag_log_info(
        "\tChunks: %zu, Allocs: %zu, Freelist Hits: %zu, Num Pool Hits: %zu",
        (size_t)pool->num_chunks,
        (size_t)pool->num_allocs,
        (size_t)pool->num_freelist_hits,
        (size_t)pool->num_pool_hits
    );
    double mem_alloced, pool_mem;
    const char *mem_unit_alloced, *mem_unit_pool;
    mag_humanize_memory_size(pool->num_chunks*pool->blocks_per_chunk*pool->block_size, &mem_alloced, &mem_unit_alloced);
    mag_humanize_memory_size(pool->num_allocs*pool->block_size, &pool_mem, &mem_unit_pool);
    mag_log_info("\t Real Mem Allocated: %.03f %s, Total Pool Mem %.03f %s", mem_alloced, mem_unit_alloced, pool_mem, mem_unit_pool);
}
