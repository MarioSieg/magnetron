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

#ifndef MAG_CONTEXT_H
#define MAG_CONTEXT_H

#include "mag_def.h"
#include "mag_slab.h"
#include "mag_machine.h"
#include "mag_backend.h"
#include "mag_toposort.h"
#include "mag_threadlib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_tls_state_t {
  mag_dtype_t dtype;
  mag_device_id_t device;
  bool no_grad;
  bool fusing;                            /* This thread is inside a fusion region. */
} mag_tls_state_t;
mag_static_assert(MAG_DTYPE_FLOAT32 == 0);
mag_static_assert(MAG_BACKEND_TYPE_CPU == 0);

extern MAG_THREAD_LOCAL mag_tls_state_t mag_tls_state; /* Thread local partial state. Needs to be TLS instead of context to enable cross-thread API invocation. */

typedef struct mag_rt_telemetry_t {
  mag_atomic64_t num_alive_tensors;           /* Total tensor instances allocated. */
  mag_atomic64_t num_alive_storages;          /* Total storage buffers allocated. */
  mag_atomic64_t num_created_tensors;         /* Total tensor instances created. */
  mag_atomic64_t storage_bytes_allocated;     /* Total bytes allocated for storage buffers. */
  mag_atomic64_t ops_dispatched;              /* Total number of dispatched operations. */
} mag_rt_telemetry_t;

struct mag_context_t {
  mag_machine_info_t machine;                 /* Machine information. */
  mag_rt_telemetry_t telemetry;               /* Runtime telemetry */
  mag_slab_alloc_t tensor_slab;               /* Tensor headers. */
  mag_slab_alloc_t storage_slab;              /* Storage headers. */
  mag_slab_alloc_t view_meta_slab;            /* View metadata headers. */
  mag_slab_alloc_t au_state_slab;             /* Autodiff states. */
  mag_slab_alloc_t au_state_op_params_slab;   /* Autodiff state op params slab allocator */
  mag_backend_registry_t *backend_registry;   /* Compute backend registry */
  struct mag_fuse_tape_t *fuse_tape;          /* Operators recorded inside a fusion region. NULL until one opens. */
  mag_lock_t fuse_state_lock;                 /* Guards fusion-region ownership and lazy tape allocation. */
  mag_atomic64_t topo_traversal_epoch;        /* Epoch counter for topological traversal of the computation graph */
#ifdef MAG_DEBUG
  mag_lock_t leak_lock;                       /* Guards alive_head. */
  mag_tensor_t *alive_head;                   /* List of alive tensors used for leak detection. */
#endif
};

#ifdef __cplusplus
}
#endif

#endif
