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

#include "mag_dylib.h"
#include "mag_def.h"

#ifdef _WIN32

#else
#include <dlfcn.h>
#include <unistd.h>
#endif

mag_status_t mag_dylib_open(mag_error_t *err, mag_dylib_t **out_lib, const char *path) {
  if (mag_unlikely(!out_lib || !path || !*path))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "dylib_open: invalid parameters or path");
#ifdef _WIN32
#error "TODO: Windows support"
#else
  if (mag_unlikely(access(path, F_OK)))
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "dylib_open: file does not exist: %s", path);
  void *handle = dlopen(path, RTLD_LAZY|RTLD_LOCAL);
  if (mag_unlikely(!handle))
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "dylib_open: failed to open file: %s", dlerror());
  *out_lib = handle;
  return MAG_STATUS_OK;
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
