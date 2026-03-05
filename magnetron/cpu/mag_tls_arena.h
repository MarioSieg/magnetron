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

#ifndef MAG_TLS_ARENA_H
#define MAG_TLS_ARENA_H

#include <core/mag_def.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_MM_SCRATCH_ALIGN MAG_DESTRUCTIVE_INTERFERENCE_SIZE

typedef struct mag_scratch_arena_t {
    uint8_t *base;
    size_t cap;
    size_t pos;
    size_t hi;
    size_t keep;
} mag_scratch_arena_t;

#define MAG_SCRATCH_ARENA_INIT(keep) { NULL, 0, 0, 0, (keep) }

extern void mag_scratch_arena_reserve(mag_scratch_arena_t *arena, size_t nb);
extern size_t mag_scratch_arena_mark(mag_scratch_arena_t *arena);
extern void mag_scratch_arena_reset(mag_scratch_arena_t *arena, size_t mark);
extern void *mag_scratch_arena_alloc(mag_scratch_arena_t *arena, size_t nb);
extern void mag_scratch_arena_clear(mag_scratch_arena_t *arena);
extern void mag_scratch_arena_trim(mag_scratch_arena_t *arena);
extern void mag_scratch_arena_destroy(mag_scratch_arena_t *arena);

#ifdef __cplusplus
}
#endif

#endif
