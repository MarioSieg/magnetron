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

#include "mag_alloc.h"

#include <mimalloc.h>

static void *mag_alloc_stub(void *blk, size_t size, size_t align) {
    if (mag_unlikely(align <= sizeof(void *))) align = 0;
    mag_assert2(!align || !(align & (align-1)));
    if (!size) {
        mi_free(blk);
        return NULL;
    }
    if (!blk) {
        void *p = align ? mi_malloc_aligned(size, align) : mi_malloc(size);
        if (mag_unlikely(!p)) mag_panic("Failed to allocate %zu bytes", size);
        return p;
    }
    void *p = align ? mi_realloc_aligned(blk, size, align) : mi_realloc(blk, size);
    if (mag_unlikely(!p)) mag_panic("Failed to reallocate %zu bytes", size);
    return p;
}

static void *mag_try_alloc_stub(void *blk, size_t size, size_t align) {
    if (mag_unlikely(align <= sizeof(void *))) align = 0;
    mag_assert2(!align || !(align & (align-1)));
    if (!size) {
        mi_free(blk);
        return NULL;
    }
    if (!blk) {
        void *p = align ? mi_malloc_aligned(size, align) : mi_malloc(size);
        if (mag_unlikely(!p)) return NULL;
        return p;
    }
    void *p = align ? mi_realloc_aligned(blk, size, align) : mi_realloc(blk, size);
    if (mag_unlikely(!p)) return NULL;
    return p;
}

void *(*mag_alloc)(void *, size_t, size_t) = &mag_alloc_stub;
void *(*mag_try_alloc)(void *, size_t, size_t) = &mag_try_alloc_stub;
