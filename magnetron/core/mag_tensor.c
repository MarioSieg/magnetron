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
#include "mag_op_dispatch.h"

static mag_status_t mag_view_meta_dtor(void *p) {
  mag_view_meta_t *vm = p;
  mag_context_t *ctx = vm->base->ctx;
  if (vm->base->view_meta == vm)
    vm->base->view_meta = NULL;
  mag_rc_decref(vm->base);
  mag_slab_free(&ctx->view_meta_slab, vm);
  return MAG_OK;
}

mag_view_meta_t *mag_view_meta_alloc(mag_tensor_t *base) {
  mag_view_meta_t *vm = mag_slab_alloc(&base->ctx->view_meta_slab);
  if (mag_unlikely(!vm)) return NULL;
  mag_rc_init_object(vm, &mag_view_meta_dtor);
  vm->base = base;
  mag_rc_incref(base);
  vm->version_snapshot = base->version;
  return vm;
}

static mag_status_t mag_tensor_dtor(void *self);

mag_tensor_t *mag_tensor_init_header(
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  int64_t numel,
  mag_device_t *device,
  mag_storage_buffer_t *storage
) {
  mag_tensor_t *hdr = mag_slab_alloc(&ctx->tensor_slab);
  if (mag_unlikely(!hdr)) return NULL;
  memset(hdr, 0, sizeof(*hdr));
  *hdr = (mag_tensor_t) {
    .ctx = ctx,
    .coords = {.rank=rank},
    .dtype = type,
    .storage = storage,
    .device = device,
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
  ++ctx->telemetry.num_alive_tensors;
  return hdr;
}

static void mag_tensor_free_header(mag_tensor_t *t) {
  mag_context_t *ctx = t->ctx;
#ifdef MAG_DEBUG
  mag_leak_detector_dequeue(t);
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
  if (mag_unlikely(mag_thread_id() != ctx->tr_id))
    return mag_set_error(err, MAG_ERR_THREAD, "tensor: must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
    return mag_set_error(err, MAG_ERR_RANK, "tensor: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  if (rank > 0 && !shape)
      return mag_set_error(err, MAG_ERR_PARAM, "tensor: shape must not be NULL when rank > 0.");
  int64_t el = (int64_t)mag_type_trait(type)->size;
  int64_t numel=1;
  for (int64_t i=0; i < rank; ++i) {
    if (mag_unlikely(shape[i] < 0))
      return mag_set_error(err, MAG_ERR_DIM, "tensor: all shape dimensions must be >= 0, but shape[%" PRIi64 "] = %" PRIi64 ".", i, shape[i]);
    if (mag_unlikely(mag_mulov64(shape[i], numel, &numel)))
      return mag_set_error(err, MAG_ERR_DIM, "tensor: element count overflowed at dim %" PRIi64 " (size %" PRIi64 ").", i, shape[i]);
  }
  int64_t numby=0;
  if (mag_unlikely(mag_mulov64(numel, el, &numby)))
    return mag_set_error(err, MAG_ERR_DIM, "tensor: byte size overflowed (numel=%" PRIi64 ", element size=%" PRIi64 ").", numel, el);
  mag_device_t *target_device = NULL;
  if (mag_unlikely(!mag_backend_registry_lookup_device_id(ctx->backend_registry, device, NULL, &target_device))) {
    char device_name[32];
    mag_device_id_to_str(device, &device_name);
    return mag_set_error(err, MAG_ERR_DEVICE, "tensor: device '%s' is not available; the backend may not be enabled.", device_name);
  }
  mag_status_t status = MAG_OK;
  if (!storage) {
    mag_status_t (*allocator)(mag_error_t *, mag_device_t *, mag_storage_buffer_t **, size_t) = target_device->alloc_storage;
    status = (*allocator)(err, target_device, &storage, numby);
    if (mag_iserr(status)) return status;
  } else {
    if (mag_unlikely(storage->device != target_device))
      return mag_set_error(err, MAG_ERR_PARAM, "tensor: storage device mismatch (tensor is on '%s' but storage is on '%s').", mag_backend_type_to_str(target_device->id.type), mag_backend_type_to_str(storage->device->id.type));
    if (mag_unlikely(storage->size < (size_t)numby))
      return mag_set_error(err, MAG_ERR_PARAM, "tensor: provided storage is too small (need %" PRIi64 " bytes, have %zu).", numby, storage->size);
    if (mag_unlikely(!(storage->base != 0 || storage->size == 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "tensor: provided storage has a NULL base pointer.");
    mag_rc_incref(storage);
  }
  mag_tensor_t *tensor = mag_tensor_init_header(ctx, type, rank, numel, target_device, storage);
  if (mag_unlikely(!tensor)) {
    status = mag_set_error(err, MAG_ERR_OOM, "tensor: failed to allocate tensor header.");
    goto cleanup;
  }
  ctx->telemetry.storage_bytes_allocated += numby;
  for (int i=0; i < MAG_MAX_DIMS; ++i) {
    tensor->coords.shape[i] = shape && i < rank ? shape[i] : 1;
    tensor->coords.strides[i] = 1;
  }
  if (rank > 0) {
    tensor->coords.strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
      if (mag_unlikely(mag_mulov64(tensor->coords.strides[i+1], tensor->coords.shape[i+1], tensor->coords.strides+i))) {
        status = mag_set_error(err, MAG_ERR_DIM, "tensor: stride computation overflowed at dim %" PRIi64 ".", i);
        goto cleanup;
      }
    }
  }
  ++ctx->telemetry.num_created_tensors;
  *out = tensor;
  return MAG_OK;
cleanup:
  mag_tensor_free_header(tensor);
  return status;
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
  return MAG_OK;
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
  return MAG_OK;
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
  if (mag_unlikely(release_cb == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "borrow_cpu_buffer: release callback must not be NULL.");
  if (mag_unlikely(data == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "borrow_cpu_buffer: data pointer must not be NULL.");
  if (mag_unlikely(!(num_bytes > 0)))
    return mag_set_error(err, MAG_ERR_PARAM, "borrow_cpu_buffer: num_bytes must be > 0.");
  if (mag_unlikely(mag_thread_id() != ctx->tr_id))
    return mag_set_error(err, MAG_ERR_THREAD, "borrow_cpu_buffer: tensor must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
    return mag_set_error(err, MAG_ERR_RANK, "borrow_cpu_buffer: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  if (rank > 0 && !shape)
      return mag_set_error(err, MAG_ERR_PARAM, "borrow_cpu_buffer: shape must not be NULL when rank > 0.");
  int64_t dts = (int64_t)mag_type_trait(dtype)->size;
  int64_t numel = 1;
  for (int64_t i=0; i < rank; ++i) {
    if (mag_unlikely(!(shape[i] >= 0)))
      return mag_set_error(err, MAG_ERR_DIM, "borrow_cpu_buffer: all shape dimensions must be >= 0, but shape[%" PRIi64 "] = %" PRIi64 ".", i, shape[i]);
    if (mag_unlikely(mag_mulov64(shape[i], numel, &numel)))
      return mag_set_error(err, MAG_ERR_DIM, "borrow_cpu_buffer: element count overflowed at dim %" PRIi64 " (size %" PRIi64 ").", i, shape[i]);
  }
  int64_t need_bytes;
  if (mag_unlikely(mag_mulov64(numel, dts, &need_bytes)))
    return mag_set_error(err, MAG_ERR_DIM, "borrow_cpu_buffer: byte size overflowed (numel=%" PRIi64 ", element size=%" PRIi64 ").", numel, dts);
  if (mag_unlikely(!((size_t)need_bytes <= num_bytes)))
    return mag_set_error(err, MAG_ERR_PARAM, "borrow_cpu_buffer: buffer is too small (need at least %zu bytes, but got %zu).", (size_t)need_bytes, num_bytes);
  mag_status_t status = MAG_OK;
  mag_borrow_cookie_t *cookie = (*mag_try_alloc)(NULL, sizeof(*cookie), 0);
  if (mag_unlikely(!cookie))
    return mag_set_error(err, MAG_ERR_OOM, "borrow_cpu_buffer: failed to allocate borrow cookie.");
  cookie->fn = release_cb;
  cookie->usr = usr;
  mag_device_t *cpu_device = NULL;
  if (mag_unlikely(!mag_backend_registry_lookup_device_id(ctx->backend_registry, mag_device(CPU, 0), NULL, &cpu_device))) {
    status = mag_set_error(err, MAG_ERR_DEVICE, "borrow_cpu_buffer: CPU backend is not available.");
    goto cleanup;
  }
  {
    mag_storage_flags_t flags = MAG_STORAGE_FLAG_BORROWED|MAG_STORAGE_FLAG_HOST_VISIBLE;
    if (is_writeable) flags |= MAG_STORAGE_FLAG_ACCESS_W;
    mag_storage_buffer_t *buf = mag_slab_alloc(&ctx->storage_slab);
    if (mag_unlikely(!buf)) {
      status = mag_set_error(err, MAG_ERR_OOM, "borrow_cpu_buffer: failed to allocate storage buffer header.");
      goto cleanup;
    }
    *buf = (mag_storage_buffer_t) {
      .ctx=ctx,
      .flags=flags,
      .base=(uintptr_t)data,
      .size=num_bytes,
      .alignment=MAG_CPU_BUF_ALIGN, /* TODO: check that data is actually aligned to this */
      .device=cpu_device,
    };
    buf->aux.impl = cookie;
    cookie = NULL;
    mag_rc_init_object(buf, &mag_borrowed_storage_dtor);
    ++ctx->telemetry.num_alive_storages;
    mag_tensor_t *tensor = NULL;
    status = mag_tensor_init(err, &tensor, ctx, buf, dtype, rank, shape, mag_device(CPU, 0));
    mag_rc_decref(buf);
    if (mag_iserr(status)) {
      *out = NULL;
      return status;
    }
    *out = tensor;
    return MAG_OK;
  }
cleanup:
  if (cookie) (*mag_alloc)(cookie, 0, 0);
  return status;
}

size_t mag_tensor_numbytes(const mag_tensor_t *t) {
  return t->storage->size;
}
int64_t mag_tensor_numel(const mag_tensor_t *tensor) {
  return tensor->numel;
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
  return tensor->device->id;
}

mag_status_t mag_tensor_copy_data(mag_error_t *err, mag_tensor_t *tensor, void **out_buf, size_t *out_size_bytes) {
  *out_buf = NULL;
  *out_size_bytes = 0;
  mag_status_t status = MAG_OK;
  mag_tensor_t *host = NULL;
  mag_tensor_t *cont = NULL;

  status = mag_transfer(err, &host, tensor, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;

  status = mag_contiguous(err, &cont, host);
  if (mag_iserr(status))
    goto cleanup;

  {
    size_t size = mag_tensor_numbytes(cont);
    if (mag_unlikely(!size)) {
      status = mag_set_error(err, MAG_ERR_STATE, "copy_data: tensor has zero size; nothing to copy.");
      goto cleanup;
    }
    void *dst = (*mag_try_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    if (mag_unlikely(!dst)) {
      status = mag_set_error(err, MAG_ERR_OOM, "copy_data: failed to allocate %zu bytes for host copy.", size);
      goto cleanup;
    }
    const void *src = (const void *)mag_tensor_data_ptr(cont);
    memcpy(dst, src, size);
    mag_tensor_decref(cont);
    mag_tensor_decref(host);
    *out_buf = dst;
    *out_size_bytes = size;
    return MAG_OK;
  }

cleanup:
  if (cont) mag_tensor_decref(cont);
  if (host) mag_tensor_decref(host);
  return status;
}

void mag_tensor_copy_data_free(void *ret_val) {
  (*mag_alloc)(ret_val, 0, 0);
}

mag_status_t mag_tensor_item(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t *out_value) {
  if (mag_unlikely(tensor->numel != 1))
    return mag_set_error(err, MAG_ERR_PARAM, "item: can only be called on a single-element tensor, but this tensor has %" PRIi64 " elements.", tensor->numel);
  if (mag_unlikely(out_value == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "item: output value pointer must not be NULL.");
  mag_status_t status = MAG_OK;
  mag_tensor_t *host = NULL;
  mag_tensor_t *scalar = NULL;
  mag_tensor_t *wide = NULL;
  bool scalar_is_host = false;
  bool wide_is_scalar = false;
  status = mag_transfer(err, &host, tensor, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;
  if (host->coords.rank == 0) {
    scalar = host;
    scalar_is_host = true;
  } else {
    status = mag_view(err, &scalar, host, NULL, 0);
    if (mag_iserr(status))
      goto cleanup;
  }
  {
    mag_dtype_t dt = scalar->dtype;
    mag_dtype_mask_t mask = mag_dtype_bit(dt);
    mag_scalar_t res;
    if (mask & MAG_DTYPE_MASK_FP) {
      if (dt != MAG_DTYPE_FLOAT32) {
        status = mag_cast(err, &wide, scalar, MAG_DTYPE_FLOAT32);
        if (mag_iserr(status))
          goto cleanup;
      } else {
        wide = scalar;
        wide_is_scalar = true;
      }
      res = mag_scalar_from_float64(*(const float *)mag_tensor_data_ptr(wide));
      if (!wide_is_scalar) mag_tensor_decref(wide);
      if (!scalar_is_host) mag_tensor_decref(scalar);
      mag_tensor_decref(host);
      *out_value = res;
      return MAG_OK;
    }
    if (mask & MAG_DTYPE_MASK_SINT) {
      if (dt != MAG_DTYPE_INT64) {
        status = mag_cast(err, &wide, scalar, MAG_DTYPE_INT64);
        if (mag_iserr(status))
          goto cleanup;
      } else {
        wide = scalar;
        wide_is_scalar = true;
      }
      res = mag_scalar_from_int64(*(const int64_t *)mag_tensor_data_ptr(wide));
      if (!wide_is_scalar) mag_tensor_decref(wide);
      if (!scalar_is_host) mag_tensor_decref(scalar);
      mag_tensor_decref(host);
      *out_value = res;
      return MAG_OK;
    }
    if ((mask & MAG_DTYPE_MASK_UINT) || dt == MAG_DTYPE_BOOLEAN) {
      if (dt != MAG_DTYPE_UINT64) {
        status = mag_cast(err, &wide, scalar, MAG_DTYPE_UINT64);
        if (mag_iserr(status))
          goto cleanup;
      } else {
        wide = scalar;
        wide_is_scalar = true;
      }
      res = mag_scalar_from_uint64(*(const uint64_t *)mag_tensor_data_ptr(wide));
      if (!wide_is_scalar) mag_tensor_decref(wide);
      if (!scalar_is_host) mag_tensor_decref(scalar);
      mag_tensor_decref(host);
      *out_value = res;
      return MAG_OK;
    }
    status = mag_set_error(err, MAG_ERR_PARAM, "item: does not support dtype %s.", mag_type_trait(dt)->name);
  }
cleanup:
  if (wide && !wide_is_scalar) mag_tensor_decref(wide);
  if (scalar && !scalar_is_host) mag_tensor_decref(scalar);
  if (host) mag_tensor_decref(host);
  return status;
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
  return mag_device_id_eq(tensor->device->id, mag_device(CPU, 0));
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
    if (t->coords.rank != rank) return false;
    if (!mag_tensor_is_contiguous(t)) return false;
    const int64_t *shape = t->coords.shape;
    for (int64_t dim=0; dim < rank; ++dim) {
      if (shape[dim] != shape0[dim])
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
