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

#include "mag_dylib.h"
#include "mag_def.h"

#ifdef _WIN32

#else
#include <dlfcn.h>
#endif

mag_dylib_t *mag_dylib_open(const char *path) {
#ifdef _WIN32
#error "TODO: Windows support"
#else
    void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (mag_unlikely(!handle)) {
        return NULL;
    }
    return handle;
#endif
}

void *mag_dylib_sym(mag_dylib_t *lib, const char *sym) {
#ifdef _WIN32
#error "TODO: Windows support"
#else
    return dlsym(lib, sym);
#endif
}

void mag_dylib_close(mag_dylib_t *lib) {
#ifdef _WIN32
#error "TODO: Windows support"
#else
    dlclose(lib);
#endif
}
