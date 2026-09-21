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

#ifndef MAG_DYLIB_H
#define MAG_DYLIB_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void mag_dylib_t;
/* Exported because a backend may need to load code it produced itself, the way the CPU backend loads
   a compiled fused chain. Core uses these only to load backend modules. */
extern MAG_EXPORT mag_status_t mag_dylib_open(mag_error_t *err, mag_dylib_t **out_lib, const char *path);
extern MAG_EXPORT void *mag_dylib_sym(mag_dylib_t *lib, const char *sym);
extern MAG_EXPORT void mag_dylib_close(mag_dylib_t *lib);
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
