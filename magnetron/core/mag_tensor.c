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

#include "mag_tensor.h"
#include "mag_context.h"
#include "mag_pool.h"
#include "mag_alloc.h"
#include "mag_autodiff.h"
#include "mag_coords_iter.h"

static void mag_view_meta_dtor(void *p) {
    mag_view_meta_t *vm = p;
    mag_context_t *ctx = vm->base->ctx;
    if (vm->base->view_meta == vm)
        vm->base->view_meta = NULL;
    mag_rc_decref(vm->base);
    mag_fixed_pool_free_block(&ctx->view_meta_pool, vm);
}

mag_view_meta_t *mag_view_meta_alloc(mag_tensor_t *base) {
    mag_view_meta_t *vm = mag_fixed_pool_alloc_block(&base->ctx->view_meta_pool);
    mag_rc_init_object(vm, &mag_view_meta_dtor);
    vm->base = base;
    mag_rc_incref(base);
    vm->version_snapshot = base->version;
    return vm;
}

static void mag_tensor_dtor(void *self); /* Destructor forward declaration. */

static mag_tensor_t *mag_tensor_init_header(mag_context_t *ctx, mag_dtype_t type, int64_t rank, int64_t numel) {
    mag_tensor_t *hdr = mag_fixed_pool_alloc_block(&ctx->tensor_pool); /* Allocate tensor header. */
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
    ++ctx->num_alive_tensors; /* Increase tensor count in context. */
    return hdr;
}

static void mag_tensor_free_header(mag_tensor_t *t) {
    mag_context_t *ctx = t->ctx;
#ifdef MAG_DEBUG
    mag_leak_detector_dequeue(t); /* Pop from alive list */
    memset(t, 0, sizeof(*t));
#endif
    mag_fixed_pool_free_block(&ctx->tensor_pool, t);
}

/* Create a new tensor. The must be created on the same thread as the context. */
mag_status_t mag_empty(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    *out = NULL;
    mag_contract(ctx, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", (uint64_t)mag_thread_id(), (uint64_t)ctx->tr_id);
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Rank must be within [0, %d]", MAG_MAX_DIMS);
    if (rank > 0) mag_contract(ctx, ERR_INVALID_PARAM, {}, shape != NULL, "Shape must not be NULL if rank > 0");
    int64_t dts = (int64_t)mag_type_trait(type)->size;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) {
        mag_contract(ctx, ERR_INVALID_DIM, {}, shape[i] > 0, "All shape dimensions must be > 0, but shape[% " PRIi64 "] = %" PRIi64, i, shape[i]);
        mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i], numel, &numel), "Dim prod overflowed: dim[%" PRIi64 "] = %" PRIi64, i, shape[i]);
    }
    int64_t numbytes;
    mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(numel, dts, &numbytes), "Total size overflowed: numel = %" PRIi64 ", dtype size = %" PRIi64, numel, dts);
    mag_tensor_t *tensor = mag_tensor_init_header(ctx, type, rank, numel); /* Alloc tensor header. */
    mag_device_t *dvc = ctx->device;
    void (*allocator)(mag_device_t *, mag_storage_buffer_t **, size_t, mag_dtype_t) = dvc->alloc_storage;
    ctx->storage_bytes_allocated += numbytes;
    (*allocator)(dvc, &tensor->storage, numbytes, type);
    for (int i=0; i < MAG_MAX_DIMS; ++i)  {
        tensor->coords.shape[i] = shape && i < rank ? shape[i] : 1;
        tensor->coords.strides[i] = 1;
    }
    if (rank > 0) {
        tensor->coords.strides[rank-1] = 1;
        for (int64_t i=rank-2; i >= 0; --i) {
            mag_contract(ctx, ERR_DIM_OVERFLOW, { mag_tensor_free_header(tensor); *out = NULL; }, !mag_mulov64(tensor->coords.strides[i+1], tensor->coords.shape[i+1], tensor->coords.strides+i), "Stride overflowed at dim[%" PRIi64 "]", i);
        }
    }
    ++ctx->num_created_tensors;
    *out = tensor;
    return MAG_STATUS_OK;
}

mag_status_t mag_as_strided(mag_tensor_t **out, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
    *out = NULL;
    mag_contract(ctx, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", (uint64_t)mag_thread_id(), (uint64_t)ctx->tr_id);
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Rank must be within [0, %d]", MAG_MAX_DIMS);
    mag_contract(ctx, ERR_INVALID_INDEX, {}, offset >= 0, "Offset must be non-negative, but is: %" PRIi64, offset);
    if (rank > 0) mag_contract(ctx, ERR_INVALID_PARAM, {}, shape && strides, "shape/strides cannot be NULL if rank > 0");
    int64_t last = offset;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) {
        mag_contract(ctx, ERR_INVALID_DIM, {}, shape[i] > 0 && (shape[i] == 1 ? strides[i] >= 0 : strides[i] > 0), "All shape dimensions must be > 0 and strides must be positive for non-singleton dims, but shape[% " PRIi64 "] = %" PRIi64 ", strides[%" PRIi64 "] = %" PRIi64, i, shape[i], i, strides[i]);
        int64_t span;
        mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i]-1, strides[i], &span), "Span overflowed at dim[%" PRIi64 "]", i);
        mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i], numel, &numel), "Dim prod overflowed: dim[%" PRIi64 "] = %" PRIi64, i, shape[i]);
        last += span;
    }
    int64_t numel_end = (int64_t)base->storage->size/base->storage->granularity;
    mag_contract(ctx, ERR_OUT_OF_BOUNDS, {}, last < numel_end, "View exceeds base tensor storage bounds: view end = %" PRIi64 ", base storage numel = %" PRIi64, last, numel_end);
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

static void mag_tensor_dtor(void *self) {
    mag_tensor_t *t = self;
    mag_context_t *ctx = t->ctx;
    mag_assert(ctx->num_alive_tensors > 0, "Double free detected on tensor %p", t);
    --ctx->num_alive_tensors;
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
    return (size_t)tensor->storage_offset*tensor->storage->granularity; /* Return offset in bytes */
}

uintptr_t mag_tensor_data_ptr(const mag_tensor_t *tensor) {
    return tensor->storage->base+mag_tensor_data_offset(tensor);
}

uintptr_t mag_tensor_data_ptr_mut(const mag_tensor_t *tensor) {
    mag_assert(tensor->storage->flags & MAG_STORAGE_FLAG_ACCESS_W, "Tensor data storage is not writable");
    return mag_tensor_data_ptr(tensor);
}

uintptr_t mag_tensor_data_storage_ptr(const mag_tensor_t *tensor) {
    return tensor->storage->base;
}

uintptr_t mag_tensor_data_storage_ptr_mut(const mag_tensor_t *tensor) {
    mag_assert(tensor->storage->flags & MAG_STORAGE_FLAG_ACCESS_W, "Tensor data storage is not writable");
    return mag_tensor_data_storage_ptr(tensor);
}

void *mag_tensor_copy_data(mag_tensor_t *tensor) {
    mag_assert2(mag_device_is(tensor->storage->device, "cpu"));
    mag_tensor_t *cont;
    mag_status_t stat = mag_contiguous(&cont, tensor);
    if (mag_iserr(stat)) return NULL;
    size_t size = mag_tensor_numbytes(cont);
    mag_assert2(size);
    void *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    const void *src = (const void *)mag_tensor_data_ptr(cont);
    memcpy(dst, src, size);
    mag_rc_decref(cont);
    return dst;
}

void mag_tensor_copy_data_free(void *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_status_t mag_tensor_item(mag_tensor_t *tensor, mag_scalar_t *out_value) {
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_device_is(tensor->storage->device, "cpu"), "item() is only supported for CPU tensors");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->numel == 1, "item() can only be called on single element (scalar) tensors, but tensor has %" PRIi64 " elements", tensor->numel);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, out_value != NULL, "Output value must not be NULL");
    mag_status_t stat;
    mag_tensor_t *scalar = NULL;
    if (tensor->coords.rank == 0) {
        mag_tensor_incref(tensor);
        scalar = tensor;
    } else {
        stat = mag_view(&scalar, tensor, NULL, 0);
        if (mag_iserr(stat))
            return stat;
    }
    mag_dtype_t dt = scalar->dtype;
    mag_dtype_mask_t mask = mag_dtype_bit(dt);
    mag_scalar_t res;
    if (mask & MAG_DTYPE_MASK_FP) {
        mag_tensor_t *wide = scalar;
        if (dt != MAG_DTYPE_FLOAT32) {
            stat = mag_cast(&wide, scalar, MAG_DTYPE_FLOAT32);
            mag_tensor_decref(scalar);
            if (mag_iserr(stat)) return stat;
        }
        res = mag_scalar_float(*(const float *)mag_tensor_data_ptr(wide));
        mag_tensor_decref(wide);
        *out_value = res;
        return MAG_STATUS_OK;
    }
    if (mask & MAG_DTYPE_MASK_SINT) {
        mag_tensor_t *wide = scalar;
        if (dt != MAG_DTYPE_INT64) {
            stat = mag_cast(&wide, scalar, MAG_DTYPE_INT64);
            mag_tensor_decref(scalar);
            if (mag_iserr(stat)) return stat;
        }
        res = mag_scalar_int(*(const int64_t *)mag_tensor_data_ptr(wide));
        mag_tensor_decref(wide);
        *out_value = res;
        return MAG_STATUS_OK;
    }
    if ((mask & MAG_DTYPE_MASK_UINT) || dt == MAG_DTYPE_BOOLEAN) {
        mag_tensor_t *wide = scalar;
        if (dt != MAG_DTYPE_UINT64) {
            stat = mag_cast(&wide, scalar, MAG_DTYPE_UINT64);
            mag_tensor_decref(scalar);
            if (mag_iserr(stat)) return stat;
        }
        res = mag_scalar_uint(*(const uint64_t *)mag_tensor_data_ptr(wide));
        mag_tensor_decref(wide);
        *out_value = res;
        return MAG_STATUS_OK;
    }
    mag_tensor_decref(scalar);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, false, "Unsupported dtype %s", mag_type_trait(dt)->name);
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

bool mag_full_cont2(const mag_tensor_t *a, const mag_tensor_t *b) {
    return a->numel == b->numel && mag_tensor_is_contiguous(a) && mag_tensor_is_contiguous(b);
}

bool mag_full_cont3(const mag_tensor_t *a, const mag_tensor_t *b, const mag_tensor_t *c) {
    return a->numel == b->numel && a->numel == c->numel && mag_tensor_is_contiguous(a) && mag_tensor_is_contiguous(b) && mag_tensor_is_contiguous(c);
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
    return mag_solve_view_strides(&tmp, tensor->coords.shape, tensor->coords.strides, tensor->coords.rank, dims, rank);
}

void mag_tensor_incref(mag_tensor_t *tensor) {
    mag_rc_incref(tensor);
}

bool mag_tensor_decref(mag_tensor_t *tensor) {
    return mag_rc_decref(tensor);
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
