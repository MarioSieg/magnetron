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

#include "mag_slab.h"
#include "mag_alloc.h"

static inline bool mag_mulov_sz(size_t a, size_t b, size_t *out) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
    return !!__builtin_mul_overflow(a, b, out);
#endif
#endif
    if (a == 0 || b == 0) { *out = 0; return false; }
    if (a > SIZE_MAX / b) return true;
    *out = a * b;
    return false;
}

static mag_slab_chunk_t *mag_fixed_pool_chunk_new(size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    size_t cap_bytes;
    mag_assert2(!mag_mulov_sz(blocks_per_chunk, block_size, &cap_bytes)); /* overflow */
    uintptr_t total = 0;
    mag_pincr((void **)&total, sizeof(mag_slab_chunk_t), __alignof(mag_slab_chunk_t));
    mag_pincr((void **)&total, cap_bytes, block_align);
    void *base = (*mag_alloc)(NULL, (size_t)total, 0);
    void *pos = base;
    mag_slab_chunk_t *chunk = mag_pincr(&pos, sizeof(mag_slab_chunk_t), __alignof(mag_slab_chunk_t));
    uint8_t *bot = mag_pincr(&pos, cap_bytes, block_align);
    *chunk = (mag_slab_chunk_t) {
        .bot = bot,
        .top = bot + cap_bytes,
        .top_cap = bot + cap_bytes,
        .next = NULL
    };
    return chunk;
}

void mag_slab_init(mag_slab_alloc_t *pool, size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    mag_assert2(pool);
    mag_assert2(blocks_per_chunk);
    block_align = mag_xmax(block_align, __alignof(void *));
    mag_assert2(block_align && !((block_align & (block_align-1))));
    block_size = mag_xmax(block_size, sizeof(void *));
    block_size = (block_size + (block_align-1)) & ~(block_align-1);
    mag_slab_chunk_t *chunk = mag_fixed_pool_chunk_new(block_size, block_align, blocks_per_chunk);
    mag_assert2(chunk);
    memset(pool, 0, sizeof(*pool));
    *pool = (mag_slab_alloc_t) {
        .block_size = block_size,
        .block_align = block_align,
        .blocks_per_chunk = blocks_per_chunk,
        .chunks = chunk,
        .chunk_tail = chunk,
        .free_list = NULL,
        .num_freelist_hits = 0,
        .num_pool_hits = 0,
        .num_chunks = 1,
        .num_allocs = 0
    };
}

void *mag_slab_alloc(mag_slab_alloc_t *pool) {
    mag_assert2(pool);
    ++pool->num_allocs;
    if (mag_likely(pool->free_list)) { /* 1) freelist fast path */
        ++pool->num_freelist_hits;
        void *blk = pool->free_list;
        pool->free_list = *(void **)blk;
        return blk;
    }
    mag_slab_chunk_t *chunk = pool->chunk_tail; /* 2) bump allocate from tail chunk */
    mag_assert2(chunk);
    uint8_t *top = chunk->top-pool->block_size;
    if (mag_likely(top >= chunk->bot)) {
        ++pool->num_pool_hits;
        chunk->top = top;
        return top;
    }
    mag_slab_chunk_t *new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk); /* 3) allocate new chunk */
    mag_assert2(new_chunk);
    pool->chunk_tail->next = new_chunk;
    pool->chunk_tail = new_chunk;
    ++pool->num_chunks;
    new_chunk->top -= pool->block_size;
    return new_chunk->top;
}

#ifdef MAG_DEBUG
static bool mag_fixed_pool_owns_ptr(const mag_slab_alloc_t *pool, const void *p) { /* Debug-only pointer ownership check */
    const mag_slab_chunk_t *c = pool->chunks;
    uintptr_t up = (uintptr_t)p;
    while (c) {
        uintptr_t bot = (uintptr_t)c->bot;
        uintptr_t top = (uintptr_t)c->top_cap;
        if (up >= bot && up < top) return true;
        c = c->next;
    }
    return false;
}
#endif

void mag_slab_free(mag_slab_alloc_t *pool, void *blk) {
    mag_assert2(pool);
    mag_assert2(blk);
#ifdef MAG_DEBUG
    mag_assert(((uintptr_t)blk & (pool->block_align-1)) == 0, "invalid alignment");
    mag_assert(mag_fixed_pool_owns_ptr(pool, blk), "pointer not owned by pool");
#endif
    *(void **)blk = pool->free_list;
    pool->free_list = blk;
}

void mag_slab_destroy(mag_slab_alloc_t *pool) {
    mag_assert2(pool);
    mag_slab_chunk_t *chunk = pool->chunks;
    while (chunk) {
        mag_slab_chunk_t *next = chunk->next;
        (*mag_alloc)(chunk, 0, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

MAG_COLDPROC void mag_slab_print_info(mag_slab_alloc_t *pool, const char *name) {
    mag_log_info("Fixed Slab Pool: %s", name);
    mag_log_info("\tBlock Size: %zu B, Block Align: %zu B, Blocks/Chunk: %zu", pool->block_size, pool->block_align, pool->blocks_per_chunk);
    mag_log_info("\tChunks: %zu, Allocs: %u, Freelist Hits: %u, Pool Hits: %u", (size_t)pool->num_chunks, pool->num_allocs, pool->num_freelist_hits, pool->num_pool_hits);
    size_t capacity_bytes = pool->num_chunks*pool->blocks_per_chunk*pool->block_size;
    double cap_val; const char *cap_unit;
    mag_humanize_memory_size(capacity_bytes, &cap_val, &cap_unit);
    mag_log_info("\tCapacity: %.03f %s", cap_val, cap_unit);
}
