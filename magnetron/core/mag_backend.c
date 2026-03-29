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

typedef struct mag_backend_module_t {
    mag_dylib_t* handle;
    mag_backend_t* backend;
    size_t fname_hash; /* Hash of the filename to identify the module */
    MAG_BACKEND_SYM_FN_ABI_COOKIE *fn_abi_cookie;
    MAG_BACKEND_SYM_FN_INIT *fn_init;
    MAG_BACKEND_SYM_FN_SHUTDOWN *fn_shutdown;
} mag_backend_module_t;

static bool mag_backend_module_dlym(mag_dylib_t* handle, const char *sym, const char *fname, void **fn) {
    *fn = mag_dylib_sym(handle, sym);
    if (mag_unlikely(!*fn)) {
        mag_log_error("Failed to find symbol '%s' in backend library file: %s", sym, fname);
        mag_dylib_close(handle);
        return false;
    }
    return true;
}

static mag_backend_module_t *mag_backend_module_load(const char *file, mag_context_t *ctx) {
    mag_log_info("Loading backend module: '%s'", file);
    mag_assert(mag_utf8_validate((const uint8_t *)file, strlen(file)), "Path is not valid UTF-8");
    mag_dylib_t* handle = mag_dylib_open(file); /* Open the dynamic library */
    if (mag_unlikely(!handle)) {
        return NULL;
    }
    /* Try to get function pointers to the required symbols */
    void *fn_abi_cookie = NULL;
    void *fn_init = NULL;
    void *fn_shutdown = NULL;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_ABI_COOKIE, file, &fn_abi_cookie))) return NULL;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_INIT, file, &fn_init))) return NULL;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_SHUTDOWN, file, &fn_shutdown))) return NULL;

    /* Check ABI cookie, then init backend */
    uint32_t abi_cookie = (*(MAG_BACKEND_SYM_FN_ABI_COOKIE*)fn_abi_cookie)(); /* Call the function to get the ABI version */
    uint32_t curr_cookie = mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
    if (mag_unlikely(abi_cookie != curr_cookie)) {
        mag_log_error("Backend library file '%s' has incompatible ABI version (got 0x%08x, expected 0x%08x)", file, abi_cookie, curr_cookie);
        mag_dylib_close(handle);
        return NULL;
    }

    /* Init backend */
    mag_backend_t *backend = (*(MAG_BACKEND_SYM_FN_INIT*)fn_init)(ctx); /* Call the function to initialize the backend */
    if (mag_unlikely(!backend)) {
        mag_log_error("Backend library file '%s' failed to initialize interface", file);
        mag_dylib_close(handle);
        return NULL;
    }

    /* Check that all function pointers are provided */
    bool fn_ok = true;
    mag_assert2(MAG_BACKEND_MODULE_ABI_VER == 1); /* Ensure this code is updated if ABI changes */
    mag_assert2(MAG_BACKEND_VTABLE_SIZE == 8); /* Ensure this code is updated if vtable size changes */
    fn_ok &= !!backend->init;
    fn_ok &= !!backend->shutdown;
    fn_ok &= !!backend->backend_version;
    fn_ok &= !!backend->runtime_version;
    fn_ok &= !!backend->id;
    fn_ok &= !!backend->num_devices;
    fn_ok &= !!backend->best_device_id;
    fn_ok &= !!backend->get_device;
    if (mag_unlikely(!fn_ok)) {
        mag_log_error("Backend struct from file '%s' is missing required function pointers", file);
        mag_dylib_close(handle);
        return NULL;
    }
    /* Verify runtime versions match */
    uint32_t backend_ver = (*backend->runtime_version)(backend);
    if (mag_unlikely(backend_ver != MAG_VERSION)) {
        uint32_t b_maj = mag_ver_major(backend_ver), b_min = mag_ver_minor(backend_ver), b_pat = mag_ver_patch(backend_ver);
        uint32_t rt_maj = mag_ver_major(MAG_VERSION), rt_min = mag_ver_minor(MAG_VERSION), rt_pat = mag_ver_patch(MAG_VERSION);
        mag_log_error("Backend library file '%s' has incompatible runtime version (got %d.%d.%d, expected %d.%d.%d)", file, b_maj, b_min, b_pat, rt_maj, rt_min, rt_pat);
        mag_dylib_close(handle);
        return NULL;
    }

    /* Invoke init hook */
    if (mag_unlikely(!(*backend->init)(backend, ctx))) {
        mag_log_error("Backend library file '%s' init hook failed", file);
        mag_dylib_close(handle);
        return NULL;
    }

    /* Create backend module */
    mag_backend_module_t *module = (*mag_alloc)(NULL, sizeof(*module), 0);
    memset(module, 0, sizeof(*module));
    *module = (mag_backend_module_t) {
        .handle = handle,
        .backend = backend,
        .fname_hash = mag_murmur3_128_reduced_64(file, strlen(file), 0),
        .fn_abi_cookie = fn_abi_cookie,
        .fn_init = fn_init,
        .fn_shutdown = fn_shutdown
    };

    char id[64];
    snprintf(id, sizeof(id), "%s", (*backend->id)(backend));
    for (char *p = id; *p; ++p)
        if (*p >= 'a' && *p <= 'z')
            *p = (char)(*p - ('a' - 'A'));
    return module;
}

static void mag_backend_module_shutdown(mag_backend_module_t *mod) {
    if (!mod) return;
    if (mag_unlikely(!(*mod->backend->shutdown)(mod->backend))) {
        mag_log_error("Backend shutdown hook failed");
    }
    if (mod->fn_shutdown && mod->backend) {
        (*mod->fn_shutdown)(mod->backend);
        mod->backend = NULL;
    }
    if (mod->handle) {
        mag_dylib_close(mod->handle);
        mod->handle = NULL;
    }
    (*mag_alloc)(mod, 0, 0);
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

void mag_device_id_to_str(mag_device_id_t id, char(*buf)[32]) {
    snprintf(*buf, sizeof(*buf), "%s:%u", mag_backend_type_to_str(id.type), id.device_ordinal);
}

bool mag_device_id_eq(mag_device_id_t a, mag_device_id_t b) {
    return a.type == b.type && a.device_ordinal == b.device_ordinal;
}

mag_backend_registry_t *mag_backend_registry_init(mag_context_t *ctx) {
    mag_backend_registry_t *reg = (*mag_alloc)(NULL, sizeof(*reg), 0);
    memset(reg, 0, sizeof(*reg));
    reg->ctx = ctx;
    char *modpath = mag_current_module_path();
    if (mag_unlikely(!modpath)) {
        mag_log_error("Failed to get current module path");
        goto error;
    }
    char *module_dir, *file;
    mag_path_split_dir_inplace(modpath, &module_dir, &file);
    mag_log_info("Module search path: '%s'", module_dir);
    /* Try to load all backends */
    char pathbuf[1024] = {0};
    for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type) {
        snprintf(pathbuf, sizeof(pathbuf), "%s/%smagnetron_%s.%s", module_dir, MAG_DYLIB_PREFIX, mag_backend_type_to_str(type), MAG_DYLIB_EXT);
        mag_backend_module_t *mod = mag_backend_module_load(pathbuf, reg->ctx);
        if (mag_unlikely(!mod)) {
            mag_log_info("Backend module not available. Name: %s, Required: %s", mag_backend_type_to_str(type), mag_backend_type_is_required(type) ? "Yes" : "No");
            if (mag_backend_type_is_required(type)) goto error;
            continue;
        }
        reg->backends[type] = mod;
        ++reg->backends_num;
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
    return reg;
error:
    if (modpath) (*mag_alloc)(modpath, 0, 0);
    if (reg) (*mag_alloc)(reg, 0, 0);
    return NULL;
}

mag_backend_t *mag_backend_registry_get_backend(mag_backend_registry_t *reg, mag_backend_type_t type) {
    if (mag_unlikely(type >= MAG_BACKEND_TYPE__COUNT)) return NULL;
    mag_backend_module_t *mod = reg->backends[type];
    if (mag_unlikely(!mod)) return NULL;
    return mod->backend;
}

bool mag_backend_registry_get_backend_and_device_by_id(mag_backend_registry_t *reg, mag_device_id_t id, mag_backend_t **out_bck, mag_device_t **out_dvc) {
    mag_backend_t *bck = mag_backend_registry_get_backend(reg, id.type);
    if (mag_unlikely(!bck)) return false;
    if (mag_unlikely(id.device_ordinal >= (*bck->num_devices)(bck))) return false;
    mag_device_t *dvc = (*bck->get_device)(bck, id.device_ordinal);
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

void mag_backend_registry_free(mag_backend_registry_t *reg) {
    for (mag_backend_type_t type=MAG_BACKEND_TYPE_CPU; type < MAG_BACKEND_TYPE__COUNT; ++type)
        if (reg->backends[type])
            mag_backend_module_shutdown(reg->backends[type]);
    (*mag_alloc)(reg, 0, 0);
}
