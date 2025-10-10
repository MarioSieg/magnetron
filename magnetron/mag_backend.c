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

mag_device_desc_t mag_compute_device_desc_cpu(uint32_t thread_count) {
    return (mag_device_desc_t) {
        .type = MAG_DEVICE_TYPE_CPU,
        .cpu_thread_count = thread_count
    };
}

mag_device_desc_t mag_compute_device_desc_cuda(uint32_t cuda_device_id) {
    return (mag_device_desc_t) {
        .type = MAG_DEVICE_TYPE_GPU_CUDA,
        .cpu_thread_count = cuda_device_id
    };
}

const char *mag_device_type_get_name(mag_device_type_t op) {
    static const char *const names[MAG_DEVICE_TYPE__NUM] = {
        [MAG_DEVICE_TYPE_CPU] = "CPU",
        [MAG_DEVICE_TYPE_GPU_CUDA] = "GPU (CUDA)",
    };
    return names[op];
}

static bool mag_is_backend_module_file_candidate(const char *fname) { /* Checks if file is a valid backend library file name. (lib)magnetron_*name*.(so|dylib|dll) */
    if (!fname || !*fname) return false;
    for (const char *p = fname; *p; ++p)
        if (*p == '/' || *p == '\\')
            return false;
    const char *prefix = MAG_DYLIB_PREFIX "magnetron_";
    size_t prefix_len = strlen(prefix);
    size_t fname_len = strlen(fname);
    if (fname_len <= prefix_len) return false;
    const char *dot = strrchr(fname, '.');
    if (!dot) return false;
    if (dot <= fname+prefix_len) return false;
    const char *ext = dot+1;
    if (strcmp(ext, MAG_DYLIB_EXT) != 0) return false;
    if (strncmp(fname, prefix, prefix_len) != 0) return false;
    size_t base_len = (size_t)(dot - fname);
    const char *core_name = MAG_DYLIB_PREFIX "magnetron_core";
    size_t core_len = strlen(core_name);
    if (base_len == core_len && strncmp(fname, core_name, core_len) == 0)
        return false;
    return true;
}

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
    mag_assert(mag_utf8_validate(file, strlen(file)), "Path is not valid UTF-8");
    mag_dylib_t* handle = mag_dylib_open(file); /* Open the dynamic library */
    if (mag_unlikely(!handle)) {
        mag_log_error("Failed to open backend library file: %s", file);
        return NULL;
    }
    /* Try to get function pointers to the required symbols */
    void *fn_abi_cookie = NULL;
    void *fn_init = NULL;
    void *fn_shutdown = NULL;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_ABI_COOKIE, file, &fn_abi_cookie))) return false;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_INIT, file, &fn_init))) return false;
    if (mag_unlikely(!mag_backend_module_dlym(handle, MAG_BACKEND_SYM_NAME_SHUTDOWN, file, &fn_shutdown))) return false;

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
    mag_backend_module_t *module = (*mag_alloc)(NULL, sizeof(*module), 0);
    memset(module, 0, sizeof(*module));
    *module = (mag_backend_module_t) {
        .handle = handle,
        .backend = backend,
        .fname_hash = mag_hash(file, strlen(file), 0),
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
    char **paths;
    size_t paths_num;
    size_t paths_cap;
    mag_backend_module_t **backends;
    size_t backends_num;
    size_t backends_cap;
};

mag_backend_registry_t *mag_backend_registry_init(mag_context_t *ctx) {
    mag_backend_registry_t *reg = (*mag_alloc)(NULL, sizeof(*reg), 0);
    memset(reg, 0, sizeof(*reg));
    reg->ctx = ctx;
    char *modpath = mag_current_module_path();
    if (mag_likely(modpath && *modpath)) {
        char *dir, *file;
        mag_path_split_dir_inplace(modpath, &dir, &file);
        mag_backend_registry_add_search_path(reg, dir);
        (*mag_alloc)(modpath, 0, 0);
    } else {
        mag_log_warn("Failed to get current module path, using backend from env: MAG_BACKEND_PATH\n");
        char *env_path = getenv("MAG_BACKEND_PATH");
        if (env_path && *env_path) mag_backend_registry_add_search_path(reg, env_path);
        else mag_log_warn("Environment variable MAG_BACKEND_PATH is not set, no backend search paths available\n");
    }
    return reg;
}

void mag_backend_registry_add_search_path(mag_backend_registry_t *reg, const char *path) {
    mag_assert(mag_utf8_validate(path, strlen(path)), "Path is not valid UTF-8");
    size_t *len = &reg->paths_num, *cap = &reg->paths_cap;
    if (*len == *cap) {
        *cap = *cap ? *cap<<1 : 2;
        reg->paths = (*mag_alloc)(reg->paths, sizeof(*reg->paths)**cap, 0);
    }
    reg->paths[(*len)++] = mag_strdup(path);
}

void mag_backend_registry_get_search_paths(mag_backend_registry_t *reg, const char ***out_paths, size_t *out_num_paths){
    *out_paths = (const  char **)reg->paths;
    *out_num_paths = reg->paths_num;
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

static void mag_backend_registry_iter_fs_callback(const char *dir, const char *file, void *ud) {
    mag_backend_registry_t *reg = ud;
    if (!mag_is_backend_module_file_candidate(file)) return; /* Name mask check */
    if (mag_backend_registry_is_backend_loaded(reg, mag_hash(file, strlen(file), 0))) return; /* Already loaded (file name hash exists) */
    mag_backend_module_t *mod = mag_backend_module_load(file, reg->ctx);
    if (mag_unlikely(!mod)) { /* Attempt to load module */
        mag_log_error("Failed to load backend from file: %s\n", file);
        return; /* Failed to load, skip */
    }
    mag_backend_registry_register(reg, mod); /* Register the loaded module */
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

bool mag_backend_registry_scan(mag_backend_registry_t *reg) {
    mag_assert2(reg->backends_num == 0);
    for (size_t i=0; i < reg->paths_num; ++i) {
        const char *path = reg->paths[i];
        mag_iter_dir(path, &mag_backend_registry_iter_fs_callback, reg);
    }
    if (reg->backends_num > 1) /* Sort backend modules by score */
        qsort(reg->backends, reg->backends_num, sizeof(*reg->backends), &mag_backend_registry_module_score_sort_callback);
    return reg->backends_num;
}

mag_backend_t *mag_backend_registry_best_backend(mag_backend_registry_t *reg) {
    mag_assert2(reg->backends_num);
    return (*reg->backends)->backend; /* Backends are sorted by score */
}

void mag_backend_registry_free(mag_backend_registry_t *reg) {
    for (size_t i=0; i < reg->backends_num; ++i)
        mag_backend_module_shutdown(reg->backends[i]);
    (*mag_alloc)(reg->backends, 0, 0);
    for (size_t i=0; i < reg->paths_num; ++i)
        (*mag_alloc)(reg->paths[i], 0, 0);
    (*mag_alloc)(reg->paths, 0, 0);
    (*mag_alloc)(reg, 0, 0);
}
