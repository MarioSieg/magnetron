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

#ifndef MAG_BACKEND_H
#define MAG_BACKEND_H

#include "mag_def.h"
#include "mag_rc.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct mag_device_t mag_device_t;

typedef enum mag_transfer_dir_t {
    MAG_TRANSFER_DIR_H2D,   /* Host -> Device. */
    MAG_TRANSFER_DIR_D2H,   /* Device -> Host. */
} mag_transfer_dir_t;

typedef enum mag_storage_flags_t {
    MAG_STORAGE_FLAG_NONE = 0,
    MAG_STORAGE_FLAG_INTRUSIVE = 1<<0,      /* Storage is intrusive (e.g. scalar optimization). */
    MAG_STORAGE_FLAG_ACCESS_W = 1<<1,       /* Write access. */
} mag_storage_flags_t;

/* Buffer interface on a compute device */
typedef struct mag_storage_buffer_t mag_storage_buffer_t;
struct mag_storage_buffer_t {
    MAG_RC_INJECT_HEADER;                   /* RC Control block must be first */

    mag_context_t *ctx;
    union {
        void *impl;                         /* Backend specific storage buffer implementation, if any. */
        uint8_t inline_buf[sizeof(void *)]; /* Inline buffer for small storage optimizations. */
    } aux;                                  /* Auxiliary storage for backend specific data. */
    mag_storage_flags_t flags;              /* Storage buffer flags. */
    uintptr_t base;                         /* Pointer to buffer on device. Might point to GPU or any other device memory. */
    size_t size;                            /* Size of buffer in bytes. */
    size_t alignment;                       /* Alignment of buffer. */
    size_t granularity;                     /* Element size granularity. */
    mag_dtype_t dtype;                      /* Data type of buffer. */
    mag_device_t *device;                   /* Host device. */
};
MAG_RC_OBJECT_IS_VALID(mag_storage_buffer_t);

typedef struct mag_command_t {
    mag_opcode_t op;
    mag_tensor_t **in;
    mag_tensor_t **out;
    uint32_t num_in;
    uint32_t num_out;
    mag_op_attr_t attrs[MAG_MAX_OP_PARAMS];
} mag_command_t;

#define MAG_DEVICEID_MAX 32

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct mag_device_t {
    mag_context_t *ctx;                                             /* Owning context */
    void *impl;                                                     /* Backend specific device implementation, if any. */
    bool is_async;                                                  /* True if the device executes commands asynchronously. */
    char physical_device_name[256];                                 /* Physical device name, (e.g. "RTX 5080", "Threadripper 9980X") */
    char id[MAG_DEVICEID_MAX];                                      /* Unique device ID, (e.g. "cpu:0", "cuda:0", "cuda:1", "tpu:0") */
    void (*submit)(mag_device_t *dvc, const mag_command_t *cmd);    /* Submit a command to the device for execution. */
    void (*alloc_storage)(mag_device_t *dvc, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype);
    void (*manual_seed)(mag_device_t *dvc, uint64_t seed);
};

extern MAG_EXPORT bool mag_device_is(const mag_device_t *dvc, const char *device_id);
extern MAG_EXPORT bool mag_parse_device_id(const char *device_id, char (*out_type)[MAG_DEVICEID_MAX], int *out_idx);

#define MAG_BACKEND_MODULE_ABI_VER 1 /* Changed if the mag_backend_t struct is changed in a non-compatible way. */
typedef struct mag_backend_t mag_backend_t;
struct mag_backend_t {
    uint32_t (*backend_version)(mag_backend_t *bck);
    uint32_t (*runtime_version)(mag_backend_t *bck);
    uint32_t (*score)(mag_backend_t *bck);
    const char *(*id)(mag_backend_t *bck);
    uint32_t (*num_devices)(mag_backend_t *bck);
    uint32_t (*best_device_idx)(mag_backend_t *bck);
    mag_device_t *(*init_device)(mag_backend_t *bck, mag_context_t *ctx, uint32_t idx);
    void(*destroy_device)(mag_backend_t *bck, mag_device_t *dvc);
    void *impl;
};
#define MAG_BACKEND_VTABLE_SIZE 8 /* Number of function pointers in mag_backend_t struct. */

#define mag_backend_cat_name(x,y) x##y
#define mag_backend_sym_fn_name(x) mag_backend_cat_name(x, _t)
#define mag_pack_abi_cookie(a,b,c,ver) (((a)&255)|(((b)&255)<<8)|(((c)&255)<<16)|(((ver)&255)<<24))
#define mag_abi_cookie_ver(x) (((x)>>24)&255)

#define MAG_BACKEND_SYM_ABI_COOKIE mag_backend_module_hook_abi_cookie
#define MAG_BACKEND_SYM_FN_ABI_COOKIE mag_backend_sym_fn_name(MAG_BACKEND_SYM_ABI_COOKIE)
#define MAG_BACKEND_SYM_NAME_ABI_COOKIE MAG_STRINGIZE(MAG_BACKEND_SYM_ABI_COOKIE)
typedef uint32_t (MAG_BACKEND_SYM_FN_ABI_COOKIE)(void);

#define MAG_BACKEND_SYM_INIT mag_backend_module_hook_init
#define MAG_BACKEND_SYM_FN_INIT mag_backend_sym_fn_name(MAG_BACKEND_SYM_INIT)
#define MAG_BACKEND_SYM_NAME_INIT MAG_STRINGIZE(MAG_BACKEND_SYM_INIT)
typedef mag_backend_t *(MAG_BACKEND_SYM_FN_INIT)(mag_context_t *ctx);

#define MAG_BACKEND_SYM_SHUTDOWN mag_backend_module_hook_shutdown
#define MAG_BACKEND_SYM_FN_SHUTDOWN mag_backend_sym_fn_name(MAG_BACKEND_SYM_SHUTDOWN)
#define MAG_BACKEND_SYM_NAME_SHUTDOWN MAG_STRINGIZE(MAG_BACKEND_SYM_SHUTDOWN)
typedef void (MAG_BACKEND_SYM_FN_SHUTDOWN)(mag_backend_t *bck);

#define mag_backend_decl_interface() \
    extern MAG_EXPORT uint32_t MAG_BACKEND_SYM_ABI_COOKIE(void); \
    extern MAG_EXPORT mag_backend_t *MAG_BACKEND_SYM_INIT(mag_context_t *ctx); \
    extern MAG_EXPORT void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *bck)

typedef struct mag_backend_registry_t mag_backend_registry_t;

extern MAG_EXPORT mag_backend_registry_t *mag_backend_registry_init(mag_context_t *ctx);
extern MAG_EXPORT bool mag_backend_registry_load_all_available(mag_backend_registry_t *reg);
extern MAG_EXPORT mag_backend_t *mag_backend_registry_get_by_device_id(mag_backend_registry_t *reg, mag_device_t **device, const char *device_id); /* Get corresponding backend for device ID like cuda:0 or cpu. */
extern MAG_EXPORT mag_backend_t *mag_backend_registry_best_backend(mag_backend_registry_t *reg);
extern MAG_EXPORT void mag_backend_registry_free(mag_backend_registry_t *reg);

#ifdef __cplusplus
}
#endif

#endif
