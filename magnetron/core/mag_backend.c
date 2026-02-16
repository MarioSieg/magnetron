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
        mag_log_error("Backend library file '%s' failed to initialize", file);
        mag_dylib_close(handle);
        return NULL;
    }

    /* Check that all function pointers are provided */
    bool fn_ok = true;
    mag_assert2(MAG_BACKEND_MODULE_ABI_VER == 1); /* Ensure this code is updated if ABI changes */
    mag_assert2(MAG_BACKEND_VTABLE_SIZE == 8); /* Ensure this code is updated if vtable size changes */
    fn_ok &= !!backend->backend_version;
    fn_ok &= !!backend->runtime_version;
    fn_ok &= !!backend->score;
    fn_ok &= !!backend->id;
    fn_ok &= !!backend->num_devices;
    fn_ok &= !!backend->best_device_idx;
    fn_ok &= !!backend->init_device;
    fn_ok &= !!backend->destroy_device;
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
    mag_log_info("Initialized backend module '%s' ABI: v.%d, Hash: 0x%" PRIx64 ", Lib: %s", id, mag_abi_cookie_ver(abi_cookie), (uint64_t)module->fname_hash, file);
    return module;
}

static void mag_backend_module_shutdown(mag_backend_module_t *mod) {
    if (!mod) return;
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
    char *module_path;
    mag_backend_module_t **backends;
    size_t backends_num;
    size_t backends_cap;
};

const char * mag_backend_type_to_str(mag_backend_type_t type) {
    static const char *type_strs[] = {
#define _(name, required) [MAG_BACKEND_TYPE_##name] = #name,
    mag_backenddef(_)
#undef _
    };
    return type_strs[type];
}

void mag_device_id_to_str(mag_device_id_t id, char(*buf)[32]) {
    snprintf(*buf, sizeof(*buf), "%s:%u", mag_backend_type_to_str(id.type), id.device_ordinal);
}

bool mag_device_id_parse(mag_device_id_t *id, const char *str) {
    if (mag_unlikely(!id || !str || !*str)) return false;
    const char *sep = strchr(str, ':');
    char name[32];
    size_t n = sep ? (size_t)(sep-str) : strlen(str);
    if (mag_unlikely(!n || n >= sizeof(name))) return false;
    memcpy(name, str, n);
    name[n] = '\0';
    for (char *p=name; *p; ++p)
        if (*p >= 'a' && *p <= 'z') *p&=(char)~0x20;
    mag_backend_type_t found = MAG_BACKEND_TYPE__COUNT;
    for (mag_backend_type_t type=0; type < MAG_BACKEND_TYPE__COUNT; ++type)
        if (strcmp(name, mag_backend_type_to_str(type)) == 0) { found = type; break; }
    if (mag_unlikely(found == MAG_BACKEND_TYPE__COUNT)) return false;
    uint32_t ord=0;
    if (sep) {
        char *end = NULL;
        unsigned long v = strtoul(sep+1, &end, 10);
        if (mag_unlikely(end == sep+1 || *end != '\0' || v > UINT32_MAX)) return false;
        ord = (uint32_t)v;
    }
    id->type = found;
    id->device_ordinal = ord;
    return true;
}

mag_backend_registry_t *mag_backend_registry_init(mag_context_t *ctx) {
    mag_backend_registry_t *reg = (*mag_alloc)(NULL, sizeof(*reg), 0);
    memset(reg, 0, sizeof(*reg));
    reg->ctx = ctx;
    char *modpath = mag_current_module_path();
    mag_assert(modpath && *modpath, "Failed to query current library module path, cannot load backends!");
    char *dir, *file;
    mag_path_split_dir_inplace(modpath, &dir, &file);
    reg->module_path = mag_strdup(dir);
    (*mag_alloc)(modpath, 0, 0);
    return reg;
}

static bool mag_backend_registry_is_backend_loaded(mag_backend_registry_t *reg, size_t fname_hash) {
    for (size_t i=0; i < reg->backends_num; ++i)
        if (reg->backends[i]->fname_hash == fname_hash)
            return true;
    return false;
}

static void mag_backend_registry_register(mag_backend_registry_t *reg, mag_backend_module_t *mod) {
    size_t *len = &reg->backends_num, *cap = &reg->backends_cap; /* Add to registry */
    if (*len == *cap) {
        *cap = *cap ? *cap<<1 : 2;
        reg->backends = (*mag_alloc)(reg->backends, sizeof(*reg->backends)**cap, 0);
    }
    reg->backends[(*len)++] = mod;
}

static void mag_format_dylib_name(char (*o)[1024], const char *basedir, const char *backend_name) {
    snprintf(*o, sizeof(*o), "%s/%smagnetron_%s.%s", basedir, MAG_DYLIB_PREFIX, backend_name, MAG_DYLIB_EXT);
}

static bool mag_backend_registry_try_backend_load(mag_backend_registry_t *reg, const char *file_path) {
    if (mag_backend_registry_is_backend_loaded(reg, mag_murmur3_128_reduced_64(file_path, strlen(file_path), 0))) return false; /* Already loaded (file name hash exists) */
    mag_backend_module_t *mod = mag_backend_module_load(file_path, reg->ctx);
    if (mag_unlikely(!mod)) { /* Attempt to load module */
        return false; /* Failed to load, skip */
    }
    mag_backend_registry_register(reg, mod); /* Register the loaded module */
    return true;
}

static int mag_backend_registry_module_score_sort_callback(const void *pa, const void *pb) {
    mag_backend_module_t **a = (mag_backend_module_t **)pa;
    mag_backend_module_t **b = (mag_backend_module_t **)pb;
    int32_t sa = (int32_t)(*(*a)->backend->score)((*a)->backend);
    int32_t sb = (int32_t)(*(*b)->backend->score)((*b)->backend);
    if (sa != sb) return sb - sa; /* Descending order */
    const char *namea = (*(*a)->backend->id)((*a)->backend);
    const char *nameb = (*(*b)->backend->id)((*b)->backend);
    return namea ? (nameb ? strcmp(namea, nameb) : -1) : nameb ? 1 : 0; /* Ascending order by name if scores are equal */
}

static const char *mag_additional_backend_names[] = { /* Additional backends without CPU, as CPU is always included */
    "cuda"
};

bool mag_backend_registry_load_all_available(mag_backend_registry_t *reg) {
    const char *basedir = reg->module_path;
    char pathbuf[1024] = {0};
    /* Always try to load required CPU backend first */
    mag_format_dylib_name(&pathbuf, basedir, "cpu"); /* TODO: maybe statically link the CPU backend? */
    if (mag_unlikely(!mag_backend_registry_try_backend_load(reg, pathbuf))) {
        mag_log_error("Failed to load required CPU backend module, aborting backend loading.\n");
        return false;
    }
    /* Try to load additional backends */
    for (size_t i=0; i < sizeof(mag_additional_backend_names)/sizeof(mag_additional_backend_names[0]); ++i) {
        mag_format_dylib_name(&pathbuf, basedir, mag_additional_backend_names[i]);
        mag_backend_registry_try_backend_load(reg, pathbuf); /* Ignore failure, as these are optional */
    }
    if (reg->backends_num > 1) /* Sort backend modules by score */
        qsort(reg->backends, reg->backends_num, sizeof(*reg->backends), &mag_backend_registry_module_score_sort_callback);
    return reg->backends_num > 0;
}

mag_backend_t *mag_backend_registry_get_by_device_id(mag_backend_registry_t *reg, mag_device_t **device, const mag_device_id_t *id) {
    for (size_t i=0; i < reg->backends_num; ++i) {
        mag_backend_t *backend = reg->backends[i]->backend;
        size_t num_devices = (*backend->num_devices)(backend);
        for (size_t d=0; d < num_devices; ++d) {
            mag_device_t *dev = (*backend->init_device)(backend, reg->ctx, d);
            if (mag_unlikely(!dev)) continue;
            if (dev->id.type == id->type && dev->id.device_ordinal == id->device_ordinal) { /* Found matching device */
                *device = dev;
                return backend;
            }
            (*backend->destroy_device)(backend, dev);
        }
    }
    return mag_backend_registry_best_backend(reg); /* Fallback to best backend */
}

mag_backend_t *mag_backend_registry_best_backend(mag_backend_registry_t *reg) {
    mag_assert2(reg->backends_num);
    return (*reg->backends)->backend; /* Backends are sorted by score */
}

void mag_backend_registry_free(mag_backend_registry_t *reg) {
    for (size_t i=0; i < reg->backends_num; ++i)
        mag_backend_module_shutdown(reg->backends[i]);
    (*mag_alloc)(reg->backends, 0, 0);
    (*mag_alloc)(reg->module_path, 0, 0);
    (*mag_alloc)(reg, 0, 0);
}
