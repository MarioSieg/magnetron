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

#ifndef MAG_ALLOC_H
#define MAG_ALLOC_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
** Allocator function. Can be set to custom allocator.
**   ! Never returns NULL, if re/allocation fails, it will abort the program by calling mag_panic().
**   ! Never zero initializes, use manual memset if zeroing is required.
**   ! Set alignment value to 0 for system default alignment, else aligned adress is returned.
**     If alignment value > 0, it must be the same when using the realloc and dealloc mode.
*/
extern MAG_EXPORT void *(*mag_alloc)(void *blk, size_t size, size_t align);
extern MAG_EXPORT void *(*mag_try_alloc)(void *blk, size_t size, size_t align);


#ifdef __cplusplus
}
#endif

#endif
