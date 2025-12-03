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

#ifndef MAG_TENSOR_H
#define MAG_TENSOR_H

#include "mag_def.h"
#include "mag_rc.h"
#include "mag_coords.h"
#include "mag_operator.h"
#include "mag_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

    /* Tensor specific flags. */
typedef enum mag_tensor_flags_t {
    MAG_TFLAG_NONE = 0,
    MAG_TFLAG_IS_VIEW = 1<<0,           /* Tensor is a view. */
    MAG_TFLAG_IS_GRAD = 1<<1,           /* Tensor is a gradient. */
    MAG_TFLAG_REQUIRES_GRAD = 1<<2,     /* Tensor requires gradient. */

    MAG_TFLAG_LEN = 4                   /* Number of flags. */
} mag_tensor_flags_t;
mag_static_assert(MAG_TFLAG_LEN <= 8); /* Must fit in one byte */

/* Metadata for view tensors */
typedef struct mag_view_meta_t {
    MAG_RC_INJECT_HEADER; /* RC Control block must be first */

    mag_tensor_t *base;
    uint32_t version_snapshot;
} mag_view_meta_t;
MAG_RC_OBJECT_IS_VALID(mag_view_meta_t);

extern mag_view_meta_t *mag_view_meta_alloc(mag_tensor_t *base);

/*
** Reference counted tensor header. Stores shape, strides, gradient and other metadata.
** The actual data buffer is compute-device specific and can be only accessed via the storage buffer.
** A tensor can be a view, which references the storage buffer of another tensor, but views have their own header too.
*/
struct mag_tensor_t {
    MAG_RC_INJECT_HEADER;                           /* RC Control block must be first */

    mag_context_t  *ctx;                            /* Host context. */
    mag_coords_t coords;                     /* Coords */
    mag_dtype_t dtype : 8;                          /* Data type of the tensor. */
    mag_tensor_flags_t flags : 8;                   /* Tensor flags. */
    mag_storage_buffer_t *storage;                  /* Storage buffer. */
    int64_t numel;                                  /* Number of elements in the tensor. */
    int64_t storage_offset;                         /* Offset in elements in the storage buffer for views. */
    mag_view_meta_t *view_meta;                     /* View metadata, if this is a view. */
    mag_au_state_t *au_state;                       /* Autodiff state, if gradient recording is active. */
    uint64_t version;                               /* Version of the tensor. Used for views to detect changes in the base tensor. */
#ifdef MAG_DEBUG
    mag_tensor_t *alive_next;                       /* Next alive tensor used for leak detection. */
#endif
};
MAG_RC_OBJECT_IS_VALID(mag_tensor_t);

extern MAG_EXPORT bool mag_full_cont2(const mag_tensor_t *a, const mag_tensor_t *b);
extern MAG_EXPORT bool mag_full_cont3(const mag_tensor_t *a, const mag_tensor_t *b, const mag_tensor_t *c);

#ifdef MAG_DEBUG
extern void mag_leak_detector_enqueue(mag_tensor_t *t);
extern void mag_leak_detector_dequeue(mag_tensor_t *t);
extern MAG_COLDPROC void mag_leak_detector_dump_results(mag_context_t *ctx);
#endif

#ifdef __cplusplus
}
#endif

#endif
