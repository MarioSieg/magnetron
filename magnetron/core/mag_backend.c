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

#include "mag_backend.h"
#include "mag_dylib.h"
#include "mag_os.h"
#include "mag_alloc.h"
#include "mag_hash.h"

#include <ctype.h>
#include <sys/stat.h>

typedef struct mag_backend_module_t {
  mag_dylib_t *handle;
  mag_backend_t *backend;
  size_t fname_hash; /* Hash of the filename to identify the module */
  MAG_BACKEND_SYM_FN_ABI_COOKIE *fn_abi_cookie;
  MAG_BACKEND_SYM_FN_INIT *fn_init;
  MAG_BACKEND_SYM_FN_SHUTDOWN *fn_shutdown;
} mag_backend_module_t;

static mag_status_t mag_backend_module_dlym(mag_error_t *err, mag_dylib_t *handle, const char *sym, const char *fname, void **fn) {
  *fn = mag_dylib_sym(handle, sym);
  if (mag_unlikely(!*fn)) {
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_STATE, "backend: shared library '%s' is missing required symbol '%s' (incompatible or corrupt backend module).", fname, sym);
  }
  return MAG_OK;
}

static mag_status_t mag_backend_module_load(mag_error_t *err, mag_backend_module_t **out_mod, const char *file, mag_context_t *ctx) {
  *out_mod = NULL;
  mag_log_info("Loading backend module: '%s'", file);
  if (mag_unlikely(!mag_utf8_validate((const uint8_t *)file, strlen(file))))
    return mag_set_error(err, MAG_ERR_PARAM, "backend: module path is not valid UTF-8.");
  mag_status_t status;
  mag_dylib_t *handle = NULL;
  status = mag_dylib_open(err, &handle, file); /* Open the dynamic library */
  if (mag_iserr(status)) return status;
  /* Try to get function pointers to the required symbols */
  void *fn_abi_cookie = NULL;
  void *fn_init = NULL;
  void *fn_shutdown = NULL;
  status = mag_backend_module_dlym(err, handle, MAG_BACKEND_SYM_NAME_ABI_COOKIE, file, &fn_abi_cookie);
  if (mag_iserr(status)) return status;
  status = mag_backend_module_dlym(err, handle, MAG_BACKEND_SYM_NAME_INIT, file, &fn_init);
  if (mag_iserr(status)) return status;
  status = mag_backend_module_dlym(err, handle, MAG_BACKEND_SYM_NAME_SHUTDOWN, file, &fn_shutdown);
  if (mag_iserr(status)) return status;

  /* Check ABI cookie, then init backend */
  uint32_t abi_cookie = (*(MAG_BACKEND_SYM_FN_ABI_COOKIE*)fn_abi_cookie)(); /* Call the function to get the ABI version */
  uint32_t curr_cookie = mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
  if (mag_unlikely(abi_cookie != curr_cookie)) {
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_STATE, "backend: shared library '%s' has an incompatible ABI version (got 0x%08x, expected 0x%08x). Rebuild the backend against this magnetron version.", file, abi_cookie, curr_cookie);
  }

  /* Init backend */
  mag_backend_t *backend=NULL;
  status = (*(MAG_BACKEND_SYM_FN_INIT*)(fn_init))(err, &backend, ctx);
  if (mag_iserr(status)) return status;

  /* Verify vtable */
  bool vtab = true;
  mag_assert2(MAG_BACKEND_MODULE_ABI_VER == 2); /* Ensure this code is updated if ABI changes */
  for (size_t i=0; i < MAG_BACKEND_VTABLE_SIZE; ++i)
    vtab &= !!*(void **)((uint8_t *)backend + MAG_BACKEND_VTABLE_START + i*sizeof(void *));
  if (mag_unlikely(!vtab)) {
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_STATE, "backend: shared library '%s' is missing required vtable function pointers (incompatible backend ABI).", file);
  }
  /* Verify runtime versions match */
  uint32_t backend_ver = (*backend->runtime_version)(backend);
  if (mag_unlikely(backend_ver != MAG_VERSION)) {
    uint32_t b_maj = mag_ver_major(backend_ver), b_min = mag_ver_minor(backend_ver), b_pat = mag_ver_patch(backend_ver);
    uint32_t rt_maj = mag_ver_major(MAG_VERSION), rt_min = mag_ver_minor(MAG_VERSION), rt_pat = mag_ver_patch(MAG_VERSION);
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_STATE, "backend: shared library '%s' has an incompatible runtime version (got %d.%d.%d, expected %d.%d.%d).", file, b_maj, b_min, b_pat, rt_maj, rt_min, rt_pat);
  }

  /* Invoke init hook */
  if (mag_iserr((*backend->init)(err, backend, ctx))) {
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_STATE, "backend: shared library '%s' init hook failed (device driver/runtime may be missing or the device failed to initialize).", file);
  }

  /* Create backend module */
  mag_backend_module_t *module = (*mag_try_alloc)(NULL, sizeof(*module), 0);
  if (mag_unlikely(!module)) {
    (void)(*backend->shutdown)(NULL, backend);
    (*(MAG_BACKEND_SYM_FN_SHUTDOWN*)fn_shutdown)(NULL, backend);
    mag_dylib_close(handle);
    return mag_set_error(err, MAG_ERR_OOM, "backend: failed to allocate backend module for '%s'.", file);
  }
  memset(module, 0, sizeof(*module));
  *module = (mag_backend_module_t) {
    .handle = handle,
    .backend = backend,
    .fname_hash = mag_murmur3_128_reduced_64(file, strlen(file), 0),
    .fn_abi_cookie = fn_abi_cookie,
    .fn_init = fn_init,
    .fn_shutdown = fn_shutdown
  };
  *out_mod = module;
  return MAG_OK;
}

static mag_status_t mag_backend_module_shutdown(mag_error_t *err, mag_backend_module_t *mod) {
  mag_status_t status = MAG_OK;
  if (!mod) return status;
  if (mod->backend) {
    if (mod->backend->shutdown) {
      mag_status_t stat = (*mod->backend->shutdown)(err, mod->backend);
      if (mag_iserr(stat) && !mag_iserr(status))
        status = stat;
    }
    if (mod->fn_shutdown) {
      mag_status_t st = (*mod->fn_shutdown)(err, mod->backend);
      if (mag_iserr(st) && !mag_iserr(status))
        status = st;
    }
    mod->backend = NULL;
  }
  if (mod->handle) {
    mag_dylib_close(mod->handle);
    mod->handle = NULL;
  }
  (*mag_alloc)(mod, 0, 0);
  return status;
}

struct mag_backend_registry_t {
  mag_context_t *ctx;
  mag_backend_module_t *backends[MAG_BACKEND_TYPE__COUNT];
  size_t backends_num;
  size_t backends_cap;
};

const char *mag_backend_type_to_str(mag_backend_type_t type) {
  static const char *data[] = {
#define _(name, id, required) [MAG_BACKEND_TYPE_##name] = #id,
  mag_backenddef(_)
#undef _
  };
  return data[type];
}

bool mag_backend_type_is_required(mag_backend_type_t type) {
  static const bool data[] = {
#define _(name, id, required) [MAG_BACKEND_TYPE_##name] = required,
    mag_backenddef(_)
  #undef _
    };
  return data[type];
}

bool mag_backend_type_has_device_ordinals(mag_backend_type_t type) {
  return type != MAG_BACKEND_TYPE_CPU;
}

void mag_device_id_to_str(mag_device_id_t id, char(*buf)[32]) {
  if (id.is_virtual) snprintf(*buf, sizeof(*buf), "virtual");
  else if (!mag_backend_type_has_device_ordinals(id.type)) snprintf(*buf, sizeof(*buf), "%s", mag_backend_type_to_str(id.type));
  else snprintf(*buf, sizeof(*buf), "%s:%u", mag_backend_type_to_str(id.type), id.device_ordinal);
}

bool mag_device_id_eq(mag_device_id_t a, mag_device_id_t b) {
  return a.is_virtual == b.is_virtual && a.type == b.type && a.device_ordinal == b.device_ordinal;
}

mag_status_t mag_backend_registry_init(mag_error_t *err, mag_context_t *ctx, mag_backend_registry_t **out_reg) {
  *out_reg = NULL;
  mag_backend_registry_t *reg = (*mag_try_alloc)(NULL, sizeof(*reg), 0);
  if (mag_unlikely(!reg))
    return mag_set_error(err, MAG_ERR_OOM, "backend: failed to allocate backend registry (%zu bytes).", sizeof(*reg));
  memset(reg, 0, sizeof(*reg));
  reg->ctx = ctx;
  mag_status_t status = MAG_OK;
  char *modpath = mag_current_module_path();
  if (mag_unlikely(!modpath)) {
    status = mag_set_error(err, MAG_ERR_STATE, "backend: failed to determine the magnetron module path needed to locate backend libraries.");
    goto error;
  }
  char *module_dir, *file;
  mag_path_split_dir_inplace(modpath, &module_dir, &file);
  mag_log_info("Module search path: '%s'", module_dir);
  /* Try to load all backends */
  char pathbuf[1024] = {0};
  for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) {
    snprintf(pathbuf, sizeof(pathbuf), "%s/%smagnetron_%s.%s", module_dir, MAG_DYLIB_PREFIX, mag_backend_type_to_str(type), MAG_DYLIB_EXT);
    mag_backend_module_t *mod = NULL;
    mag_status_t status2 = mag_backend_module_load(err, &mod, pathbuf, reg->ctx);
    if (mag_iserr(status2)) {
      if (mag_backend_type_is_required(type)) { /* A required backend (e.g. CPU) failed: propagate its detailed reason to the caller. */
        status = status2; /* 'err' already holds the specific failure message. */
        goto error;
      }
      mag_log_info("Optional backend '%s' not available: %s", mag_backend_type_to_str(type), err ? err->message : "(not present)"); /* Non-fatal. */
      if (err) err->code = MAG_OK; /* Consume the non-fatal failure: clear the sticky error so a later real error can be recorded and the returned 'err' stays clean. */
      continue;
    }
    reg->backends[type] = mod;
    ++reg->backends_num;
  }
  if (mag_unlikely(!reg->backends_num)) { /* Defensive: no usable backend at all. */
    status = mag_set_error(err, MAG_ERR_STATE,
      "backend: no magnetron compute backends could be loaded.\n"
      "Backends are loaded as shared libraries next to libmagnetron_core, but none were located or usable.\n"
      "Ensure at least one backend (e.g. magnetron_cpu) is installed alongside libmagnetron_core,\n"
      "and that the file is named %smagnetron_<backend>.%s.", MAG_DYLIB_PREFIX, MAG_DYLIB_EXT);
    goto error;
  }
  (*mag_alloc)(modpath, 0, 0);
  /* Print short overview of loaded backends */
  for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) {
    if (reg->backends[type]) {
      mag_backend_t *bck = reg->backends[type]->backend;
      mag_log_info("Loaded backend: %s (Version %d.%d.%d, %u Device%s, Best Device: '%s')",
        (*bck->id)(bck),
        mag_ver_major((*bck->runtime_version)(bck)),
        mag_ver_minor((*bck->runtime_version)(bck)),
        mag_ver_patch((*bck->runtime_version)(bck)),
        (*bck->num_devices)(bck),
        (*bck->num_devices)(bck) == 1 ? "" : "s",
        (*bck->num_devices)(bck) > 0 ? bck->get_device(bck, (*bck->best_device_id)(bck))->physical_device_name : "N/A"
      );
    }
  }
  *out_reg = reg;
  return MAG_OK;
error:
  if (modpath) (*mag_alloc)(modpath, 0, 0);
  for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) /* Shut down any backends already loaded before the failure. */
    if (reg->backends[type])
      mag_backend_module_shutdown(NULL, reg->backends[type]);
  (*mag_alloc)(reg, 0, 0);
  return status;
}

mag_backend_t *mag_backend_registry_get_backend(mag_backend_registry_t *reg, mag_backend_type_t type) {
  if (mag_unlikely(type >= MAG_BACKEND_TYPE__COUNT)) return NULL;
  mag_backend_module_t *mod = reg->backends[type];
  if (mag_unlikely(!mod)) return NULL;
  return mod->backend;
}

/*
** Search by the device's own ordinal instead of indexing with it: get_device() is index-addressed and a
** backend only publishes the devices that initialized successfully, so one skipped device (unsupported
** architecture, prohibited compute mode) would otherwise shift every later ordinal down and 'cuda:N' would
** stop meaning GPU N. Device counts are tiny, so the scan is free.
*/
bool mag_backend_registry_lookup_device_id(mag_backend_registry_t *reg, mag_device_id_t id, mag_backend_t **out_bck, mag_device_t **out_dvc) {
  mag_backend_t *bck = mag_backend_registry_get_backend(reg, id.type);
  if (mag_unlikely(!bck)) return false;
  uint32_t nd = (*bck->num_devices)(bck);
  for (uint32_t i=0; i < nd; ++i) {
    mag_device_t *dvc = (*bck->get_device)(bck, i);
    if (!dvc || dvc->id.device_ordinal != id.device_ordinal) continue;
    if (out_bck) *out_bck = bck;
    if (out_dvc) *out_dvc = dvc;
    return true;
  }
  return false;
}

bool mag_backend_registry_best_device(mag_backend_registry_t *reg, mag_backend_type_t type, mag_backend_t **out_bck, mag_device_t **out_dvc) {
  mag_backend_t *bck = mag_backend_registry_get_backend(reg, type);
  if (mag_unlikely(!bck)) return false;
  if (mag_unlikely(!(*bck->num_devices)(bck))) return false;
  mag_device_t *dvc = (*bck->get_device)(bck, (*bck->best_device_id)(bck)); /* Backend-chosen index, not an ordinal. */
  if (mag_unlikely(!dvc)) return false;
  if (out_bck) *out_bck = bck;
  if (out_dvc) *out_dvc = dvc;
  return true;
}

void mag_backend_registry_iter_devices(mag_backend_registry_t *reg, void(*callback)(mag_backend_t *bck, mag_device_t *dvc, void *usr), void *usr) {
  for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) {
    if (!reg->backends[type]) continue;
    mag_backend_t *backend = reg->backends[type]->backend;
    if (!backend) continue;
    uint32_t nd = (*backend->num_devices)(backend);
    for (uint32_t i=0; i<nd; ++i) {
      mag_device_t *dvc = backend->get_device(backend, i);
      if (!dvc) continue;
      (*callback)(backend, dvc, usr);
    }
  }
}

mag_status_t mag_backend_registry_shutdown(mag_error_t *err, mag_backend_registry_t *reg) {
  for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) {
    if (reg->backends[type]) {
      mag_status_t status = mag_backend_module_shutdown(err, reg->backends[type]);
      if (mag_iserr(status)) {
        (*mag_alloc)(reg, 0, 0);
        return status;
      }
    }
  }
  (*mag_alloc)(reg, 0, 0);
  return MAG_OK;
}
