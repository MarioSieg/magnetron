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

#include "mag_alloc.h"

#ifdef MAGNETRON_USE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef MAGNETRON_USE_MIMALLOC

static void *mag_alloc_stub(void *blk, size_t size, size_t align) { /* Allocator stub for mimalloc. */
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

#else

#if defined(__GLIBC__) || defined(__linux__)
#include <malloc.h>
#define mag_msize malloc_usable_size
#elif defined(__FreeBSD__)
#include <malloc_np.h>
#define mag_msize malloc_usable_size
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#define mag_msize malloc_size
#elif defined(_WIN32)
#include <malloc.h>
#define mag_msize _msize
#else
#error "Unknown platform"
#endif

static void *mag_alloc_stub(void *blk, size_t size, size_t align) {
    if (mag_unlikely(align <= sizeof(void *))) align = 0;
    if (!size) {
        if (!blk) return NULL;
        free(align ? ((void **)blk)[-1] : blk);
        return NULL;
    }
    if (!blk) goto alloc;
    if (!align) {
        void *new_blk = realloc(blk, size);
        if (!new_blk) mag_panic("Failed to reallocate %zu bytes", size);
        return new_blk;
    } else {
        void *old_base = ((void **)blk)[-1];
        size_t old_size = mag_msize(old_base)-((uintptr_t)blk-(uintptr_t)old_base);
alloc:
        if (!align) {
            void *p = malloc(size);
            if (!p) mag_panic("Failed to allocate %zu bytes", size);
            return p;
        }
        if (align & (align-1) || align < sizeof(void *)) mag_panic("Alignment %zu is not a power of two ≥ sizeof(void*)", align);
        if (size > SIZE_MAX-align-sizeof(void *)) mag_panic("Size/align overflow");
        void *raw = malloc(size+align+sizeof(void *));
        if (!raw) mag_panic("Failed to allocate %zu bytes", size);
        uintptr_t aligned_addr = ((uintptr_t)raw+sizeof(void *)+align-1)&~(uintptr_t)(align-1);
        void *user = (void *)aligned_addr;
        ((void **)user)[-1] = raw;
        if (blk) {
            memcpy(user, blk, old_size < size ? old_size : size);
            free(old_base);
        }
        return user;
    }
}

/* Allocate aligned memory by overallocating. Alignment must be a power of two. */
void *mag_alloc_aligned(size_t size, size_t align) {
    mag_assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    void *p = (*mag_alloc)(NULL, size+sizeof(void *)+align-1, 0);
    uintptr_t pp = ((uintptr_t)p+sizeof(void *)+align-1)&-align;
    ((void **)pp)[-1] = p;
    return (void *)pp;
}

void mag_free_aligned(void *blk) {
    (*mag_alloc)(((void **)blk)[-1], 0, 0);
}

#endif

void *(*mag_alloc)(void *, size_t, size_t) = &mag_alloc_stub;
