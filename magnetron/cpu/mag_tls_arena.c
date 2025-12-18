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

#include "mag_tls_arena.h"

#include <core/mag_alloc.h>

void mag_scratch_arena_reserve(mag_scratch_arena_t *arena, size_t nb) {
    if (nb <= arena->cap) return;
    size_t nc = arena->cap ? arena->cap : 4096;
    while (nc < nb) nc = nc < 1<<20 ? nc<<1 : nc+(nc>>1);
    nc = (nc + 4095)&~4095;
    arena->base = (*mag_alloc)(arena->base, nc, MAG_MM_SCRATCH_ALIGN);
    #ifndef _MSC_VER
        arena->base = __builtin_assume_aligned(arena->base, MAG_MM_SCRATCH_ALIGN);
    #endif
    arena->cap = nc;
}

size_t mag_scratch_arena_mark(mag_scratch_arena_t *arena) {
    return arena->pos;
}

void mag_scratch_arena_reset(mag_scratch_arena_t *arena, size_t mark) {
    mag_assert2(mark <= arena->pos);
    arena->pos = mark;
}

void *mag_scratch_arena_alloc(mag_scratch_arena_t *arena, size_t nb) {
    size_t pos = (arena->pos+(MAG_MM_SCRATCH_ALIGN-1)) & ~(MAG_MM_SCRATCH_ALIGN-1);
    size_t n = (nb+(MAG_MM_SCRATCH_ALIGN-1)) & ~(MAG_MM_SCRATCH_ALIGN-1);
    size_t end = pos + n;
    if (mag_unlikely(end > arena->cap))
        mag_scratch_arena_reserve(arena, end);
    void *p = arena->base + pos;
    arena->pos = end;
    arena->hi = mag_xmax(end, arena->hi);
    #ifndef _MSC_VER
        p = __builtin_assume_aligned(p, MAG_MM_SCRATCH_ALIGN);
    #endif
    return p;
}

void mag_scratch_arena_clear(mag_scratch_arena_t *arena) {
    arena->pos = 0;
}

void mag_scratch_arena_trim(mag_scratch_arena_t *arena) {
    if (!arena->base) { arena->cap = arena->pos = arena->hi = 0; return; }
    if (arena->keep == 0) {
        arena->pos = 0;
        arena->hi = 0;
        return;
    }
    size_t target = mag_xmin(mag_xmax(arena->hi, 4096), arena->keep);
    target = (target+4095)&~4095;
    if (arena->cap > target) {
        arena->base = (uint8_t *)(*mag_alloc)(arena->base, target, MAG_MM_SCRATCH_ALIGN);
        #ifndef _MSC_VER
            arena->base = __builtin_assume_aligned(arena->base, MAG_MM_SCRATCH_ALIGN);
        #endif
        arena->cap  = target;
    }
    arena->pos = 0;
    arena->hi = 0;
}

void mag_scratch_arena_destroy(mag_scratch_arena_t *arena) {
    if (arena->base)
        (*mag_alloc)(arena->base, 0, MAG_MM_SCRATCH_ALIGN);
    memset(arena, 0, sizeof(*arena));
}
