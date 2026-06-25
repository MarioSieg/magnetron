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

#include "mag_tensor.h"
#include "mag_context.h"
#include "mag_slab.h"
#include "mag_alloc.h"
#include "mag_autodiff.h"
#include "mag_coords_iter.h"

static mag_status_t mag_view_meta_dtor(void *p) {
  mag_view_meta_t *vm = p;
  mag_context_t *ctx = vm->base->ctx;
  if (vm->base->view_meta == vm)
    vm->base->view_meta = NULL;
  mag_rc_decref(vm->base);
  mag_slab_free(&ctx->view_meta_slab, vm);
  return MAG_STATUS_OK;
}

mag_view_meta_t *mag_view_meta_alloc(mag_tensor_t *base) {
  mag_view_meta_t *vm = mag_slab_alloc(&base->ctx->view_meta_slab);
  mag_rc_init_object(vm, &mag_view_meta_dtor);
  vm->base = base;
  mag_rc_incref(base);
  vm->version_snapshot = base->version;
  return vm;
}

static mag_status_t mag_tensor_dtor(void *self); /* Destructor forward declaration. */

static mag_tensor_t *mag_tensor_init_header(mag_context_t *ctx, mag_dtype_t type, int64_t rank, int64_t numel) {
  mag_tensor_t *hdr = mag_slab_alloc(&ctx->tensor_slab); /* Allocate tensor header. */
  memset(hdr, 0, sizeof(*hdr));
  *hdr = (mag_tensor_t) { /* Initialize tensor header. */
    .ctx = ctx,
    .coords = {.rank=rank},
    .dtype = type,
    .storage = NULL,
    .numel = numel,
    .flags = MAG_TFLAG_NONE,
    .storage_offset = 0,
    .view_meta = NULL,
    .au_state = NULL,
    .version = 0,
  };
  mag_rc_init_object(hdr, &mag_tensor_dtor);
#ifdef MAG_DEBUG
  hdr->alive_next = NULL;
  mag_leak_detector_enqueue(hdr);
#endif
  ++ctx->telemetry.num_alive_tensors; /* Increase tensor count in context. */
  return hdr;
}

static void mag_tensor_free_header(mag_tensor_t *t) {
  mag_context_t *ctx = t->ctx;
#ifdef MAG_DEBUG
  mag_leak_detector_dequeue(t); /* Pop from alive list */
  memset(t, 0, sizeof(*t));
#endif
  mag_slab_free(&ctx->tensor_slab, t);
}

/* Create a new tensor. The must be created on the same thread as the context. */
mag_status_t mag_tensor_init(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_context_t *ctx,
  mag_storage_buffer_t *storage,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_device_id_t device
) {
  *out = NULL;
  mag_contract(err, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "tensor: must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  mag_contract(err, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "tensor: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  if (rank > 0) mag_contract(err, ERR_INVALID_PARAM, {}, shape != NULL, "tensor: shape must not be NULL when rank > 0.");
  int64_t el = (int64_t)mag_type_trait(type)->size;
  int64_t numel = 1;
  for (int64_t i=0; i < rank; ++i) {
    mag_contract(err, ERR_INVALID_DIM, {}, shape[i] > 0, "tensor: all shape dimensions must be > 0, but shape[%" PRIi64 "] = %" PRIi64 ".", i, shape[i]);
    mag_contract(err, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i], numel, &numel), "tensor: element count overflowed at dim %" PRIi64 " (size %" PRIi64 ").", i, shape[i]);
  }
  int64_t numbytes;
  mag_contract(err, ERR_DIM_OVERFLOW, {}, !mag_mulov64(numel, el, &numbytes), "tensor: byte size overflowed (numel=%" PRIi64 ", element size=%" PRIi64 ").", numel, el);
  mag_device_t *target_device=NULL;
  bool has_rquested_device = mag_backend_registry_get_backend_and_device_by_id(ctx->backend_registry, device, NULL, &target_device);
  char device_name[32];
  if (mag_unlikely(!has_rquested_device)) mag_device_id_to_str(device, &device_name);
  mag_contract(err, ERR_INVALID_DEVICE, {}, has_rquested_device, "tensor: device '%s' is not available; the backend may not be enabled.", device_name);
  mag_tensor_t *tensor = mag_tensor_init_header(ctx, type, rank, numel); /* Alloc tensor header. */
  if (!storage) {
    mag_status_t (*allocator)(mag_device_t *, mag_error_t *, mag_storage_buffer_t **, size_t) = target_device->alloc_storage;
    mag_try_or((*allocator)(target_device, err, &tensor->storage, numbytes), {
      mag_tensor_free_header(tensor);
    });
  } else {
    mag_contract(err, ERR_INVALID_PARAM, { mag_tensor_free_header(tensor); }, storage->device == target_device, "tensor: storage device mismatch (tensor is on '%s' but storage is on '%s').", mag_backend_type_to_str(target_device->id.type), mag_backend_type_to_str(storage->device->id.type));
    mag_contract(err, ERR_INVALID_PARAM, { mag_tensor_free_header(tensor); }, storage->size >= (size_t)numbytes, "tensor: provided storage is too small (need %" PRIi64 " bytes, have %zu).", numbytes, storage->size);
    mag_contract(err, ERR_INVALID_PARAM, { mag_tensor_free_header(tensor); }, storage->base != 0 || storage->size == 0, "tensor: provided storage has a NULL base pointer.");
    tensor->storage = storage;
    mag_rc_incref(storage); /* Retain provided storage */
  }
  ctx->telemetry.storage_bytes_allocated += numbytes;
  for (int i=0; i < MAG_MAX_DIMS; ++i)  {
    tensor->coords.shape[i] = shape && i < rank ? shape[i] : 1;
    tensor->coords.strides[i] = 1;
  }
  if (rank > 0) {
    tensor->coords.strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
      mag_contract(err, ERR_DIM_OVERFLOW, { mag_tensor_free_header(tensor); *out = NULL; }, !mag_mulov64(tensor->coords.strides[i+1], tensor->coords.shape[i+1], tensor->coords.strides+i), "tensor: stride computation overflowed at dim %" PRIi64 ".", i);
    }
  }
  ++ctx->telemetry.num_created_tensors;
  *out = tensor;
  return MAG_STATUS_OK;
}

/* Create a new tensor. The must be created on the same thread as the context. */
mag_status_t mag_empty(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  return mag_tensor_init(err, out, ctx, NULL, type, rank, shape, device);
}

mag_status_t mag_as_strided(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
  *out = NULL;
  mag_contract(err, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "as_strided: tensor must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  mag_contract(err, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "as_strided: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  mag_contract(err, ERR_INVALID_INDEX, {}, offset >= 0, "as_strided: storage offset must be non-negative, but got %" PRIi64 ".", offset);
  if (rank > 0) mag_contract(err, ERR_INVALID_PARAM, {}, shape && strides, "as_strided: shape and strides must not be NULL when rank > 0.");
  int64_t last = offset;
  int64_t numel = 1;
  for (int64_t i=0; i < rank; ++i) {
    mag_contract(
      err, ERR_INVALID_DIM, {},
      shape[i] > 0 && strides[i] >= 0,
      "as_strided: invalid shape/stride at dim %" PRIi64
      " (shape=%" PRIi64 ", stride=%" PRIi64
      "); dimensions must be > 0 and strides must be non-negative.",
      i, shape[i], strides[i]
    );
    int64_t span;
    mag_contract(
      err, ERR_DIM_OVERFLOW, {},
      !mag_mulov64(shape[i]-1, strides[i], &span),
      "as_strided: stride span overflowed at dim %" PRIi64 ".",
      i
    );
    mag_contract(
      err, ERR_DIM_OVERFLOW, {},
      !mag_mulov64(shape[i], numel, &numel),
      "as_strided: element count overflowed at dim %" PRIi64
      " (size %" PRIi64 ").",
      i, shape[i]
    );
    last += span;
  }
  int64_t numel_end = (int64_t)(base->storage->size/mag_type_trait(base->dtype)->size);
  mag_contract(err, ERR_OUT_OF_BOUNDS, {}, last < numel_end, "as_strided: view exceeds base tensor storage (end index %" PRIi64 " >= storage capacity %" PRIi64 ").", last, numel_end);
  mag_tensor_t *tensor = mag_tensor_init_header(ctx, base->dtype, rank, numel); /* Alloc tensor header. */
  for (int i=0; i < MAG_MAX_DIMS; ++i) {
    tensor->coords.shape[i] = i < rank && shape ? shape[i] : 1;
    tensor->coords.strides[i] = i < rank && strides ? strides[i] : 1;
  }
  tensor->storage = base->storage;
  mag_rc_incref(base->storage); /* Retain base storage */
  tensor->storage_offset = offset;
  tensor->version = base->version;
  if (!(base->flags & MAG_TFLAG_IS_VIEW)) /* first view */
    tensor->view_meta = mag_view_meta_alloc(base);
  else {
    tensor->view_meta = base->view_meta;
    mag_rc_incref(tensor->view_meta); /* Retain view meta */
  }
  tensor->flags = base->flags | MAG_TFLAG_IS_VIEW; /* Set view flag */
  *out = tensor;
  return MAG_STATUS_OK;
}

mag_status_t mag_broadcast_to(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape) {
  int64_t old_rank = x->coords.rank;
  const int64_t *old_shape = x->coords.shape;
  const int64_t *old_strides = x->coords.strides;
  int64_t new_strides[MAG_MAX_DIMS];
  mag_contract(err, ERR_INVALID_RANK, {}, rank >= old_rank, "broadcast_to: target rank %" PRIi64 " must be >= source rank %" PRIi64 ".", rank, old_rank);
  for (int64_t i=0; i < rank; ++i) {
    int64_t new_ax = rank-1-i;
    int64_t old_ax = old_rank-1-i;
    int64_t new_dim = shape[new_ax];
    if (old_ax < 0) {
      new_strides[new_ax] = 0;
      continue;
    }
    int64_t old_dim = old_shape[old_ax];
    int64_t old_stride = old_strides[old_ax];
    mag_contract(err, ERR_INVALID_RANK, {}, old_dim == new_dim || old_dim == 1, "broadcast_to: cannot broadcast dim of size %" PRIi64 " to %" PRIi64 "; only size-1 dims are broadcastable.", old_dim, new_dim);
    new_strides[new_ax] = old_dim == new_dim ? old_stride : 0;
  }
  return mag_as_strided(
    err,
    out,
    mag_tensor_context(x),
    x,
    rank,
    shape,
    new_strides,
    (int64_t)mag_tensor_data_offset(x)
  );
}

mag_status_t mag_expand(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape) {
  int64_t old_rank = x->coords.rank;
  const int64_t *old_shape = x->coords.shape;
  mag_contract(
    err, ERR_INVALID_RANK, {},
    rank >= old_rank,
    "expand: target rank %" PRIi64 " must be >= source rank %" PRIi64 ".",
    rank, old_rank
  );
  int64_t resolved[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) {
    int64_t new_ax = rank-1-i;
    int64_t old_ax = old_rank-1-i;
    int64_t dim = shape[new_ax];
    if (dim == -1) {
      mag_contract(
        err, ERR_INVALID_PARAM, {},
        old_ax >= 0,
        "expand: -1 is not allowed for a newly prepended dimension."
      );
      resolved[new_ax] = old_shape[old_ax];
    } else {
      mag_contract(
        err, ERR_INVALID_PARAM, {},
        dim >= 0,
        "expand: invalid dimension size %" PRIi64 ".",
        dim
      );
      resolved[new_ax] = dim;
    }
  }
  return mag_broadcast_to(err, out, x, rank, resolved);
}

static mag_status_t mag_tensor_dtor(void *self) {
  mag_tensor_t *t = self;
  mag_context_t *ctx = t->ctx;
  mag_assert(ctx->telemetry.num_alive_tensors > 0, "tensor: double free detected on tensor %p.", t);
  --ctx->telemetry.num_alive_tensors;
  if (t->view_meta) {
    mag_rc_decref(t->view_meta);
    t->view_meta = NULL;
  }
  if (t->au_state) {
    mag_rc_decref(t->au_state);
    t->au_state = NULL;
  }
  mag_rc_decref(t->storage);
  mag_tensor_free_header(t);
  return MAG_STATUS_OK;
}

typedef struct mag_borrow_cookie_t {
  void (*fn)(void *);
  void *usr;
} mag_borrow_cookie_t;

static mag_status_t mag_borrowed_storage_dtor(void *self) {
  mag_storage_buffer_t *buf = self;
  mag_context_t *ctx = buf->ctx;
  mag_assert(ctx->telemetry.num_alive_storages > 0, "tensor: double free detected on storage buffer.");
  --ctx->telemetry.num_alive_storages;
  mag_borrow_cookie_t *cookie = buf->aux.impl;
  if (cookie) {
    if (cookie->fn) (*cookie->fn)(cookie->usr);
    (*mag_alloc)(cookie, 0, 0);
  }
  mag_slab_free(&ctx->storage_slab, buf);
  return MAG_STATUS_OK;
}

mag_status_t mag_borrow_cpu_buffer(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_context_t *ctx,
  void *data,
  size_t num_bytes,
  mag_dtype_t dtype,
  int64_t rank,
  const int64_t *shape,
  bool is_writeable,
  void (*release_cb)(void *usr),
  void *usr
) {
  *out = NULL;
  mag_contract(err, ERR_INVALID_PARAM, {}, release_cb != NULL, "borrow_cpu_buffer: release callback must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, data != NULL, "borrow_cpu_buffer: data pointer must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, num_bytes > 0, "borrow_cpu_buffer: num_bytes must be > 0.");
  mag_contract(err, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "borrow_cpu_buffer: tensor must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  mag_contract(err, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "borrow_cpu_buffer: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  if (rank > 0) mag_contract(err, ERR_INVALID_PARAM, {}, shape != NULL, "borrow_cpu_buffer: shape must not be NULL when rank > 0.");
  int64_t dts = (int64_t)mag_type_trait(dtype)->size;
  int64_t numel=1;
  for (int64_t i=0; i < rank; ++i) {
    mag_contract(err, ERR_INVALID_DIM, {}, shape[i] > 0, "borrow_cpu_buffer: all shape dimensions must be > 0, but shape[%" PRIi64 "] = %" PRIi64 ".", i, shape[i]);
    mag_contract(err, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i], numel, &numel), "borrow_cpu_buffer: element count overflowed at dim %" PRIi64 " (size %" PRIi64 ").", i, shape[i]);
  }
  int64_t need_bytes;
  mag_contract(err, ERR_DIM_OVERFLOW, {}, !mag_mulov64(numel, dts, &need_bytes), "borrow_cpu_buffer: byte size overflowed (numel=%" PRIi64 ", element size=%" PRIi64 ").", numel, dts);
  mag_contract(err, ERR_INVALID_PARAM, {}, (size_t)need_bytes <= num_bytes, "borrow_cpu_buffer: buffer is too small (need at least %zu bytes, but got %zu).", (size_t)need_bytes, num_bytes);
  mag_borrow_cookie_t *cookie = (*mag_alloc)(NULL, sizeof(*cookie), 0);
  cookie->fn = release_cb;
  cookie->usr = usr;
  mag_device_t *cpu_device = NULL;
  mag_contract(err, ERR_INVALID_DEVICE, { (*mag_alloc)(cookie, 0, 0); }, mag_backend_registry_get_backend_and_device_by_id(ctx->backend_registry, mag_device(CPU, 0), NULL, &cpu_device), "borrow_cpu_buffer: CPU backend is not available.");
  mag_storage_flags_t flags = MAG_STORAGE_FLAG_BORROWED|MAG_STORAGE_FLAG_HOST_VISIBLE;
  if (is_writeable) flags |= MAG_STORAGE_FLAG_ACCESS_W;
  mag_storage_buffer_t *buf = mag_slab_alloc(&ctx->storage_slab);
  *buf = (mag_storage_buffer_t) {
    .ctx=ctx,
    .flags=flags,
    .base=(uintptr_t)data,
    .size=num_bytes,
    .alignment=MAG_CPU_BUF_ALIGN, /* TODO: check that data is actually aligned to this */
    .device=cpu_device,
  };
  buf->aux.impl = cookie;
  mag_rc_init_object(buf, &mag_borrowed_storage_dtor);
  ++ctx->telemetry.num_alive_storages;
  mag_tensor_t *tensor = NULL;
  mag_status_t stat = mag_tensor_init(err, &tensor, ctx, buf, dtype, rank, shape, mag_device(CPU, 0));
  mag_rc_decref(buf);
  if (mag_iserr(stat)) {
    *out = NULL;
    return stat;
  }
  *out = tensor;
  return MAG_STATUS_OK;
}

size_t mag_tensor_numbytes(const mag_tensor_t *t) {
  return t->storage->size;
}
int64_t mag_tensor_numel(const mag_tensor_t *tensor) {
  return tensor->numel;
}

void mag_tensor_detach_inplace(mag_tensor_t *target) {
  if (target->au_state) {
    target->au_state->op = MAG_OP_NOP; /* Detach from operations */
    memset(target->au_state->op_inputs, 0, sizeof(target->au_state->op_inputs)); /* Clear op inputs */
    memset(target->au_state->op_attrs, 0, sizeof(target->au_state->op_attrs));
  }
}

mag_tensor_t *mag_tensor_detach(mag_tensor_t *tensor) {
  mag_tensor_detach_inplace(tensor);
  return tensor;
}

int64_t mag_tensor_rank(const mag_tensor_t *tensor) {
  return tensor->coords.rank;
}

const int64_t *mag_tensor_shape_ptr(const mag_tensor_t *tensor) {
  return tensor->coords.shape;
}

const int64_t *mag_tensor_strides_ptr(const mag_tensor_t *tensor) {
  return tensor->coords.strides;
}

mag_dtype_t mag_tensor_type(const mag_tensor_t *tensor) {
  return tensor->dtype;
}

size_t mag_tensor_data_offset(const mag_tensor_t *tensor) {
  return (size_t)tensor->storage_offset*mag_type_trait(tensor->dtype)->size; /* Return offset in bytes */
}

uintptr_t mag_tensor_data_ptr(const mag_tensor_t *tensor) {
  return tensor->storage->base+mag_tensor_data_offset(tensor);
}

uintptr_t mag_tensor_data_ptr_mut(const mag_tensor_t *tensor) {
  mag_assert(tensor->storage->flags & MAG_STORAGE_FLAG_ACCESS_W, "tensor: storage is read-only.");
  return mag_tensor_data_ptr(tensor);
}

uintptr_t mag_tensor_data_storage_ptr(const mag_tensor_t *tensor) {
  return tensor->storage->base;
}

uintptr_t mag_tensor_data_storage_ptr_mut(const mag_tensor_t *tensor) {
  mag_assert(tensor->storage->flags & MAG_STORAGE_FLAG_ACCESS_W, "tensor: storage is read-only.");
  return mag_tensor_data_storage_ptr(tensor);
}

mag_device_id_t mag_tensor_device_id(const mag_tensor_t *tensor) {
  return tensor->storage->device->id;
}

mag_status_t mag_tensor_copy_data(mag_error_t *err, mag_tensor_t *tensor, void **out_buf, size_t *out_size_bytes) {
  *out_buf = NULL;
  *out_size_bytes = 0;
  mag_tensor_t *host = NULL;
  mag_try(mag_transfer(err, &host, tensor, mag_device(CPU, 0)));
  mag_tensor_t *cont = NULL;
  mag_try_or(mag_contiguous(err, &cont, host), { mag_tensor_decref(host); });
  size_t size = mag_tensor_numbytes(cont);
  mag_assert2(size);
  void *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
  const void *src = (const void *)mag_tensor_data_ptr(cont);
  memcpy(dst, src, size);
  mag_tensor_decref(cont);
  mag_tensor_decref(host);
  *out_buf = dst;
  *out_size_bytes = size;
  return MAG_STATUS_OK;
}

void mag_tensor_copy_data_free(void *ret_val) {
  (*mag_alloc)(ret_val, 0, 0);
}

mag_status_t mag_tensor_item(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t *out_value) {
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->numel == 1, "item: can only be called on a single-element tensor, but this tensor has %" PRIi64 " elements.", tensor->numel);
  mag_contract(err, ERR_INVALID_PARAM, {}, out_value != NULL, "item: output value pointer must not be NULL.");
  mag_tensor_t *host = NULL;
  mag_try(mag_transfer(err, &host, tensor, mag_device(CPU, 0)));
  mag_status_t stat;
  mag_tensor_t *scalar = NULL;
  if (host->coords.rank == 0) scalar = host;
  else mag_try_or(mag_view(err, &scalar, host, NULL, 0), { mag_tensor_decref(host); });
  mag_dtype_t dt = scalar->dtype;
  mag_dtype_mask_t mask = mag_dtype_bit(dt);
  mag_scalar_t res;
  if (mask & MAG_DTYPE_MASK_FP) {
    mag_tensor_t *wide = scalar;
    if (dt != MAG_DTYPE_FLOAT32) {
      stat = mag_cast(err, &wide, scalar, MAG_DTYPE_FLOAT32);
      mag_tensor_decref(scalar);
      if (mag_iserr(stat)) {
        if (scalar != host) mag_tensor_decref(host);
        return stat;
      }
    }
    res = mag_scalar_from_f64(*(const float *)mag_tensor_data_ptr(wide));
    mag_tensor_decref(wide);
    if (scalar != host) mag_tensor_decref(host);
    *out_value = res;
    return MAG_STATUS_OK;
  }
  if (mask & MAG_DTYPE_MASK_SINT) {
    mag_tensor_t *wide = scalar;
    if (dt != MAG_DTYPE_INT64) {
      stat = mag_cast(err, &wide, scalar, MAG_DTYPE_INT64);
      mag_tensor_decref(scalar);
      if (mag_iserr(stat)) {
        if (scalar != host) mag_tensor_decref(host);
        return stat;
      }
    }
    res = mag_scalar_from_i64(*(const int64_t *)mag_tensor_data_ptr(wide));
    mag_tensor_decref(wide);
    if (scalar != host) mag_tensor_decref(host);
    *out_value = res;
    return MAG_STATUS_OK;
  }
  if ((mask & MAG_DTYPE_MASK_UINT) || dt == MAG_DTYPE_BOOLEAN) {
    mag_tensor_t *wide = scalar;
    if (dt != MAG_DTYPE_UINT64) {
      stat = mag_cast(err, &wide, scalar, MAG_DTYPE_UINT64);
      mag_tensor_decref(scalar);
      if (mag_iserr(stat)) {
        if (scalar != host) mag_tensor_decref(host);
        return stat;
      }
    }
    res = mag_scalar_from_u64(*(const uint64_t *)mag_tensor_data_ptr(wide));
    mag_tensor_decref(wide);
    if (scalar != host) mag_tensor_decref(host);
    *out_value = res;
    return MAG_STATUS_OK;
  }
  mag_tensor_decref(scalar);
  if (scalar != host) mag_tensor_decref(host);
  mag_contract(err, ERR_INVALID_PARAM, {}, false, "item: does not support dtype %s.", mag_type_trait(dt)->name);
  return MAG_STATUS_ERR_INVALID_PARAM;
}

mag_context_t *mag_tensor_context(const mag_tensor_t *tensor) {
  return tensor->ctx;
}

bool mag_tensor_is_view(const mag_tensor_t *tensor) {
  return tensor->flags & MAG_TFLAG_IS_VIEW;
}

bool mag_tensor_is_floating_point_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_FP;
}

bool mag_tensor_is_integral_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_INTEGRAL;
}

bool mag_tensor_is_integer_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_INTEGER;
}

bool mag_tensor_is_unsigned_integer_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_UINT;
}

bool mag_tensor_is_signed_integer_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_SINT;
}

bool mag_tensor_is_numeric_typed(const mag_tensor_t *tensor) {
  return mag_dtype_bit(tensor->dtype) & MAG_DTYPE_MASK_NUMERIC;
}

bool mag_tensor_is_shape_eq(const mag_tensor_t *x, const mag_tensor_t *y) {
  return mag_coords_shape_cmp(&x->coords, &y->coords);
}

bool mag_tensor_are_strides_eq(const mag_tensor_t *x, const mag_tensor_t *y) {
  return mag_coords_strides_cmp(&x->coords, &y->coords);
}

bool mag_tensor_can_broadcast(const mag_tensor_t *small, const mag_tensor_t *big) {
  return mag_coords_can_broadcast(&small->coords, &big->coords);
}

bool mag_tensor_is_transposed(const mag_tensor_t *tensor) {
  return mag_coords_transposed(&tensor->coords);
}

bool mag_tensor_is_permuted(const mag_tensor_t *tensor) {
  return mag_coords_permuted(&tensor->coords);
}

bool mag_tensor_is_contiguous(const mag_tensor_t *tensor) {
  return mag_coords_contiguous(&tensor->coords);
}

bool mag_tensor_can_view(const mag_tensor_t *tensor, const int64_t *dims, int64_t rank) {
  int64_t tmp[MAG_MAX_DIMS];
  return mag_isok(mag_solve_view_strides(NULL, &tmp, tensor->coords.shape, tensor->coords.strides, tensor->coords.rank, dims, rank));
}

void mag_tensor_incref(mag_tensor_t *tensor) {
  mag_rc_incref(tensor);
}

bool mag_tensor_decref(mag_tensor_t *tensor) {
  return mag_rc_decref(tensor);
}

bool mag_tensor_is_cpu(mag_tensor_t *tensor) {
  return mag_device_id_eq(tensor->storage->device->id, mag_device(CPU, 0));
}

bool mag_all_shapes_equal_and_contig(const mag_tensor_t **tensors, size_t n) {
  if (mag_unlikely(!tensors || !n)) return false;
  const mag_tensor_t *t0 = *tensors;
  if (mag_unlikely(!t0)) return false;
  if (!mag_tensor_is_contiguous(t0)) return false;
  const int64_t rank = t0->coords.rank;
  const int64_t *shape0 = t0->coords.shape;
  for (size_t i=1; i < n; ++i) {
    const mag_tensor_t *t = tensors[i];
    if (mag_unlikely(!t)) return false;
    if (!mag_tensor_is_contiguous(t)) return false;
    if (t->coords.rank != rank) return false;
    const int64_t *shape = t->coords.shape;
    for (int64_t d = 0; d < rank; ++d) {
      if (shape[d] != shape0[d])
        return false;
    }
  }
  return true;
}

#ifdef MAG_DEBUG

void mag_leak_detector_enqueue(mag_tensor_t *t) {
  mag_context_t *ctx = t->ctx;
  t->alive_next = ctx->alive_head;
  ctx->alive_head = t;
}

void mag_leak_detector_dequeue(mag_tensor_t *t) {
  mag_context_t *ctx = t->ctx;
  for (mag_tensor_t **p = &ctx->alive_head; *p; p = &(*p)->alive_next) {
    if (*p == t) {
      *p = t->alive_next;
      break;
    }
  }
}

MAG_COLDPROC void mag_leak_detector_dump_results(mag_context_t *ctx) {
  for (mag_tensor_t *leaked = ctx->alive_head; leaked; leaked = leaked->alive_next) {
    char shape[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape, &leaked->coords.shape, leaked->coords.rank);
    fprintf(
      stderr,
      MAG_CC_RED "[magnetron] " MAG_CC_RESET "Leaked tensor: %p, Shape: %s\n",
      leaked,
      shape
    );
  }
  fflush(stderr);
}

#endif
