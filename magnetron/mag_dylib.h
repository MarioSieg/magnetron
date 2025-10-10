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

#ifndef MAG_DYLIB_H
#define MAG_DYLIB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void mag_dylib_t;
extern mag_dylib_t *mag_dylib_open(const char *path);
extern void *mag_dylib_sym(mag_dylib_t *lib, const char *sym);
extern void mag_dylib_close(mag_dylib_t *lib);
#ifdef _WIN32
#define MAG_DYLIB_EXT "dll"
#define MAG_DYLIB_PREFIX ""
#elif defined(__APPLE__)
#define MAG_DYLIB_EXT "dylib"
#define MAG_DYLIB_PREFIX "lib"
#else
#define MAG_DYLIB_EXT "so"
#define MAG_DYLIB_PREFIX "lib"
#endif

#ifdef __cplusplus
}
#endif

#endif
