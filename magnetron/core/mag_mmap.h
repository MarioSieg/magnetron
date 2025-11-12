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

#ifndef MAG_MMAP_H
#define MAG_MMAP_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MAG_MAP_READ,
    MAG_MAP_WRITE,
    MAG_MAP_READWRITE
} mag_map_mode_t;

typedef struct mag_mapped_file_t {
    uint8_t *map;
    size_t fs;
    bool writable;
#ifdef _WIN32
    void *hfile;
    void *hmap;
#else
    int fd;
#endif
} mag_mapped_file_t;


extern bool mag_map_file(mag_mapped_file_t *o, const char *filename, size_t size, mag_map_mode_t mode);
extern bool mag_unmap_file(mag_mapped_file_t *f);

#ifdef __cplusplus
}
#endif

#endif
