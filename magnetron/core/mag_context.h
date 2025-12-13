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

#ifndef MAG_CONTEXT_H
#define MAG_CONTEXT_H

#include "mag_def.h"
#include "mag_pool.h"
#include "mag_machine.h"
#include "mag_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Context specific flags. */
typedef enum mag_context_flags_t {
    MAG_CTX_FLAG_NONE = 0,
    MAG_CTX_FLAG_GRAD_RECORDER = 1<<0,     /* Gradient recording is currently active. */
} mag_context_flags_t;

struct mag_context_t {
    mag_error_t error_status;                   /* Last error status. */
    mag_machine_info_t machine;                 /* Machine information. */
    mag_fixed_pool_t tensor_pool;               /* Tensor header memory pool. */
    mag_fixed_pool_t storage_pool;              /* Storage header memory pool. */
    mag_fixed_pool_t view_meta_pool;            /* View metadata header memory pool. */
    mag_fixed_pool_t au_state_pool;             /* Autodiff state memory pool. */
    mag_context_flags_t flags;                  /* Context flags. */
    uintptr_t tr_id;                            /* Context thread ID. */
    mag_backend_registry_t *backend_registry;   /* Compute backend registry */
    mag_backend_t *backend;                     /* Active compute backend. */
    mag_device_t *device;                       /* Active compute device. */
    void *ud;                                   /* User data. */
    size_t num_alive_tensors;                   /* Total tensor instances allocated. */
    size_t num_alive_storages;                  /* Total storage buffers allocated. */
    size_t num_created_tensors;                 /* Total tensor instances created. */
    size_t storage_bytes_allocated;             /* Total bytes allocated for storage buffers. */
    size_t ops_dispatched;                      /* Total number of dispatched operations. */
#ifdef MAG_DEBUG
    mag_tensor_t *alive_head;                   /* List of alive tensors used for leak detection. */
#endif
};

#ifdef __cplusplus
}
#endif

#endif
