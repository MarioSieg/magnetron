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

#ifndef MAG_MMAP_RA_H
#define MAG_MMAP_RA_H

#include "mag_mmap.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MMAP readahead prefetch engine */

#define MAG_RA_LOG_CAP 0x8000u
#define MAG_RA_HASH_CAP 0x10000u
#define MAG_RA_QUEUE_CAP 0x1000u
#define MAG_RA_CHUNK 0x100000u
#define MAG_RA_MIN_HINT_BYTES 0x40000u
#define MAG_RA_MAX_ENGINES 4u
#define MAG_RA_DEFAULT_WINDOW_MB 0x400u
extern void mag_mmap_ra_register(const mag_mapped_file_t *mf);
extern void mag_mmap_ra_unregister(const mag_mapped_file_t *mf);
extern MAG_EXPORT void mag_mmap_readahead_hint(const void *ptr, size_t len);
extern MAG_EXPORT void mag_mmap_release_hint(const void *ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif
