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

#ifndef MAG_BACKEND_H
#define MAG_BACKEND_H

#include "mag_def.h"
#include "mag_rc.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Name, ID, Required */
#define mag_backenddef(_)\
    _(CPU, cpu, true)\
    _(CUDA, cuda, false)\
    _(CUSTOM, custom, false)\

typedef enum mag_backend_type_t {
#define _(name, id, required) MAG_BACKEND_TYPE_##name,
    mag_backenddef(_)
    MAG_BACKEND_TYPE__COUNT
#undef _
} mag_backend_type_t;
extern const char *mag_backend_type_to_str(mag_backend_type_t type);
extern bool mag_backend_type_is_required(mag_backend_type_t type);

typedef struct mag_device_id_t {
    mag_backend_type_t type;        /* Backend type, (e.g. CPU, CUDA, etc..) */
    uint32_t device_ordinal;        /* Device index for the given backend type, (e.g. 0 for cuda:0). */
} mag_device_id_t;
extern void mag_device_id_to_str(mag_device_id_t id, char (*buf)[32]);
extern bool mag_device_id_parse(mag_device_id_t *id, const char *str);

#define MAG_DEVICE_ID_CPU ((mag_device_id_t){.type=MAG_BACKEND_TYPE_CPU, .device_ordinal=0})

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct mag_device_t mag_device_t;

typedef enum mag_transfer_dir_t {
    MAG_TRANSFER_DIR_H2D,   /* Host -> Device. */
    MAG_TRANSFER_DIR_D2H,   /* Device -> Host. */
} mag_transfer_dir_t;

typedef enum mag_storage_flags_t {
    MAG_STORAGE_FLAG_NONE = 0,
    MAG_STORAGE_FLAG_BORROWED = 1<<0,       /* Storage memory is borrowed and must not be freed. */
    MAG_STORAGE_FLAG_ACCESS_W = 1<<1,       /* Write access. */
} mag_storage_flags_t;

/* Buffer interface on a compute device */
typedef struct mag_storage_buffer_t mag_storage_buffer_t;
struct mag_storage_buffer_t {
    MAG_RC_INJECT_HEADER;                   /* RC Control block must be first */
    mag_context_t *ctx;
    mag_storage_flags_t flags;                                  /* Storage buffer flags. */
    uint32_t alignment;                                         /* Alignment of buffer. */
    uintptr_t base;                                             /* Pointer to buffer on device. Might point to GPU or any other device memory. */
    size_t size;                                                /* Size of buffer in bytes. */
    mag_device_t *device;                                       /* Host device. */
    mag_alignas(MAG_CPU_BUF_INTRUSIVE_CAP) union {
        void *impl;                                             /* Backend specific storage buffer implementation, if any. */
        mag_alignas(MAG_CPU_BUF_INTRUSIVE_CAP) uint8_t intrusive_storage[MAG_CPU_BUF_INTRUSIVE_CAP]; /* (CPU only) Intrusive inline st */
    } aux;
};
MAG_RC_OBJECT_IS_VALID(mag_storage_buffer_t);

mag_static_assert(offsetof(mag_storage_buffer_t, aux) % MAG_CPU_BUF_ALIGN == 0); /* Ensure aux is properly aligned for intrusive storage. */

typedef struct mag_command_t {
    mag_opcode_t op;
    mag_tensor_t **in;
    mag_tensor_t **out;
    uint32_t num_in;
    uint32_t num_out;
    mag_op_attr_t attrs[MAG_MAX_OP_PARAMS];
} mag_command_t;

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct mag_device_t {
    void *impl;                                                             /* Backend specific device implementation, if any. */
    mag_context_t *ctx;                                                     /* Owning context */
    mag_device_id_t id;                                                     /* Device ID, (e.g. cuda:0, cpu, etc..) */
    bool is_async;                                                          /* True if the device executes commands asynchronously. */
    mag_status_t (*submit)(mag_device_t *dvc, const mag_command_t *cmd);    /* Submit a command to the device for execution. */
    mag_status_t (*alloc_storage)(mag_device_t *dvc, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype);
    void (*manual_seed)(mag_device_t *dvc, uint64_t seed);
    char physical_device_name[256];                                 /* Physical device name, (e.g. "RTX 5080", "Threadripper 9980X") */
};

#define MAG_BACKEND_MODULE_ABI_VER 1 /* Changed if the mag_backend_t struct is changed in a non-compatible way. */
typedef struct mag_backend_t mag_backend_t;
struct mag_backend_t {
    void *impl;
    bool (*init)(mag_backend_t *self, mag_context_t *ctx);
    bool (*shutdown)(mag_backend_t *self);
    uint32_t (*backend_version)(mag_backend_t *bck);
    uint32_t (*runtime_version)(mag_backend_t *bck);
    const char *(*id)(mag_backend_t *bck);
    uint32_t (*num_devices)(mag_backend_t *bck);
    uint32_t (*best_device_id)(mag_backend_t *bck);
    mag_device_t *(*get_device)(mag_backend_t *bck, uint32_t idx);
};
#define MAG_BACKEND_VTABLE_SIZE 8 /* Number of function pointers in mag_backend_t struct. */
mag_static_assert((sizeof(mag_backend_t)/sizeof(void *))-1 == MAG_BACKEND_VTABLE_SIZE);

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
extern MAG_EXPORT bool mag_backend_registry_get_backend_and_device_by_id(mag_backend_registry_t *reg, mag_device_id_t id, mag_backend_t **out_bck, mag_device_t **out_dvc);
extern MAG_EXPORT void mag_backend_registry_free(mag_backend_registry_t *reg);

#ifdef __cplusplus
}
#endif

#endif
