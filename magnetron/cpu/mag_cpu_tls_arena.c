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

#include "mag_cpu_tls_arena.h"

#include <core/mag_alloc.h>

struct mag_scratch_retired_t {
  mag_scratch_retired_t *next;
  uint8_t *base;
};

static void mag_scratch_arena_free_retired(mag_scratch_arena_t *arena) {
  mag_scratch_retired_t *r = arena->retired;
  while (r) {
    mag_scratch_retired_t *next = r->next;
    (*mag_alloc)(r->base, 0, MAG_MM_SCRATCH_ALIGN);
    (*mag_alloc)(r, 0, 0);
    r = next;
  }
  arena->retired = NULL;
}

static uint8_t *mag_scratch_arena_block_alloc(size_t nc) {
  uint8_t *blk = (*mag_try_alloc)(NULL, nc, MAG_MM_SCRATCH_ALIGN);
  #ifndef _MSC_VER
    if (blk) blk = __builtin_assume_aligned(blk, MAG_MM_SCRATCH_ALIGN);
  #endif
  return blk;
}

bool mag_scratch_arena_reserve(mag_scratch_arena_t *arena, size_t nb) {
  if (nb <= arena->cap) return true;
  size_t nc = arena->cap ? arena->cap : 4096;
  while (nc < nb) nc = nc < 1<<20 ? nc<<1 : nc+(nc>>1);
  nc = (nc + 4095)&~4095;
  uint8_t *grown = mag_scratch_arena_block_alloc(nc);
  if (mag_unlikely(!grown)) return false;
  if (arena->base) {
    if (arena->pos) {
      mag_scratch_retired_t *r = (*mag_try_alloc)(NULL, sizeof(*r), 0);
      if (mag_unlikely(!r)) {
        (*mag_alloc)(grown, 0, MAG_MM_SCRATCH_ALIGN);
        return false;
      }
      r->next = arena->retired;
      r->base = arena->base;
      arena->retired = r;
    } else {
      (*mag_alloc)(arena->base, 0, MAG_MM_SCRATCH_ALIGN);
    }
  }
  arena->base = grown;
  arena->cap = nc;
  return true;
}

size_t mag_scratch_arena_mark(mag_scratch_arena_t *arena) {
  return arena->pos;
}

void mag_scratch_arena_reset(mag_scratch_arena_t *arena, size_t mark) {
  mag_assert2(mark <= arena->pos);
  arena->pos = mark;
  if (!mark && arena->retired) mag_scratch_arena_free_retired(arena);
}

void *mag_scratch_arena_alloc(mag_scratch_arena_t *arena, size_t nb) {
  size_t pos = (arena->pos+(MAG_MM_SCRATCH_ALIGN-1)) & ~(MAG_MM_SCRATCH_ALIGN-1);
  size_t n = (nb+(MAG_MM_SCRATCH_ALIGN-1)) & ~(MAG_MM_SCRATCH_ALIGN-1);
  size_t end = pos + n;
  if (mag_unlikely(end > arena->cap))
    if (mag_unlikely(!mag_scratch_arena_reserve(arena, end))) return NULL;
  void *p = arena->base + pos;
  arena->pos = end;
  arena->hi = mag_vmax(end, arena->hi);
  #ifndef _MSC_VER
    p = __builtin_assume_aligned(p, MAG_MM_SCRATCH_ALIGN);
  #endif
  return p;
}

void mag_scratch_arena_clear(mag_scratch_arena_t *arena) {
  arena->pos = 0;
  if (arena->retired) mag_scratch_arena_free_retired(arena);
}

void mag_scratch_arena_trim(mag_scratch_arena_t *arena) {
  if (arena->retired) mag_scratch_arena_free_retired(arena);
  if (!arena->base) { arena->cap = arena->pos = arena->hi = 0; return; }
  if (arena->keep == 0) {
    arena->pos = 0;
    arena->hi = 0;
    return;
  }
  size_t target = mag_vmin(mag_vmax(arena->hi, 4096), arena->keep);
  target = (target+4095)&~4095;
  if (arena->cap > target) {
    (*mag_alloc)(arena->base, 0, MAG_MM_SCRATCH_ALIGN);
    arena->base = mag_scratch_arena_block_alloc(target);
    arena->cap = arena->base ? target : 0;
  }
  arena->pos = 0;
  arena->hi = 0;
}

void mag_scratch_arena_destroy(mag_scratch_arena_t *arena) {
  if (arena->retired) mag_scratch_arena_free_retired(arena);
  if (arena->base)
    (*mag_alloc)(arena->base, 0, MAG_MM_SCRATCH_ALIGN);
  memset(arena, 0, sizeof(*arena));
}
