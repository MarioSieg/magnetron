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
#include "mag_sstream.h"
#include "mag_autodiff.h"
#include "mag_shape.h"

static void mag_view_meta_dtor(void *p) {
    mag_view_meta_t *vm = p;
    mag_context_t *ctx = vm->base->ctx;
    if (vm->base->view_meta == vm)
        vm->base->view_meta = NULL;
    mag_rc_control_decref(&vm->base->rc_control);
    mag_fixed_pool_free_block(&ctx->view_meta_pool, vm);
}

mag_view_meta_t *mag_view_meta_alloc(mag_tensor_t *base) {
    mag_view_meta_t *vm = mag_fixed_pool_alloc_block(&base->ctx->view_meta_pool);
    vm->rc = mag_rc_control_init(vm, &mag_view_meta_dtor);
    vm->base = base;
    mag_rc_control_incref(&base->rc_control);
    vm->version_snapshot = base->version;
    return vm;
}

static void mag_tensor_dtor(void *self); /* Destructor forward declaration. */

static mag_tensor_t *mag_tensor_init_header(mag_context_t *ctx, mag_dtype_t type, int64_t rank, int64_t numel) {
    mag_tensor_t *hdr = mag_fixed_pool_alloc_block(&ctx->tensor_pool); /* Allocate tensor header. */
    memset(hdr, 0, sizeof(*hdr));
    *hdr = (mag_tensor_t) { /* Initialize tensor header. */
        .ctx = ctx,
        .rc_control = mag_rc_control_init(hdr, &mag_tensor_dtor), /* Initialize reference counter. */
        .rank = rank,
        .shape = {0},
        .strides = {0},
        .dtype = type,
        .storage = NULL,
        .numel = numel,
        .flags = MAG_TFLAG_NONE,
        .storage_offset = 0,
        .view_meta = NULL,
        .au_state = NULL,
        .version = 0,
    };
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
mag_status_t mag_tensor_new(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    *out = NULL;
    mag_contract(ctx, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", (uint64_t)mag_thread_id(), (uint64_t)ctx->tr_id);
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
    int64_t dts = (int64_t)mag_dtype_meta_of(type)->size;
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
        tensor->shape[i] = i < rank ? shape[i] : 1;
        tensor->strides[i] = 1;
    }
    /* Compute contiguous row-major strides and check for overflow. */
    tensor->strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
        mag_contract(ctx, ERR_DIM_OVERFLOW, { mag_tensor_free_header(tensor); *out = NULL; }, !mag_mulov64(tensor->strides[i+1], tensor->shape[i+1], tensor->strides+i), "Stride overflowed at dim[%" PRIi64 "]", i);
    }
    ++ctx->num_created_tensors;
    *out = tensor;
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_as_strided(mag_tensor_t **out, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
    *out = NULL;
    mag_contract(ctx, ERR_THREAD_MISMATCH, {}, mag_thread_id() == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", (uint64_t)mag_thread_id(), (uint64_t)ctx->tr_id);
    mag_contract(ctx, ERR_INVALID_RANK, {}, shape && rank > 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
    mag_contract(ctx, ERR_INVALID_INDEX, {}, offset >= 0, "Offset must be non-negative, but is: %" PRIi64, offset);
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
        tensor->shape[i] = i < rank ? shape[i] : 1;
        tensor->strides[i] = i < rank ? strides[i] : 1;
    }
    tensor->storage = base->storage;
    mag_rc_control_incref(&base->storage->rc_control); /* Retain base storage */
    tensor->storage_offset = offset;
    tensor->version = base->version;
    if (!(base->flags & MAG_TFLAG_IS_VIEW)) /* first view */
        tensor->view_meta = mag_view_meta_alloc(base);
    else {
        tensor->view_meta = base->view_meta;
        mag_rc_control_incref(&tensor->view_meta->rc); /* Retain view meta */
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
        mag_rc_control_decref(&t->view_meta->rc);
        t->view_meta = NULL;
    }
    if (t->au_state) {
        mag_rc_control_decref(&t->au_state->rc);
        t->au_state = NULL;
    }
    mag_rc_control_decref(&t->storage->rc_control);
    mag_tensor_free_header(t);
}

mag_status_t mag_tensor_empty(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    return mag_tensor_new(out, ctx, type, rank, shape);
}

mag_status_t mag_tensor_empty_like(mag_tensor_t **out, mag_tensor_t *isomorph) {
    return mag_tensor_new(out, isomorph->ctx, isomorph->dtype, isomorph->rank, isomorph->shape);
}

mag_status_t mag_tensor_empty_scalar(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type) {
    return mag_tensor_empty(out, ctx, type, 1, (int64_t[1]){1});
}

mag_status_t mag_tensor_scalar(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty_scalar(out, ctx, type);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_full(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty(out, ctx, type, rank, shape);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_full_like(mag_tensor_t **out, mag_tensor_t *isomorph, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty_like(out, isomorph);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

int64_t mag_tensor_get_data_size(const mag_tensor_t *t) {
    return t->storage->size;
}
int64_t mag_tensor_get_numel(const mag_tensor_t *t) {
    return t->numel;
}

void mag_tensor_incref(mag_tensor_t *t) { /* Increase reference count of the tensor. */
    mag_rc_control_incref(&t->rc_control);
}

bool mag_tensor_decref(mag_tensor_t *t) { /* Decrease reference count of the tensor. */
    if (mag_unlikely(!t)) return false;
    return mag_rc_control_decref(&t->rc_control);
}

void mag_tensor_detach_inplace(mag_tensor_t *target) {
    if (target->au_state) {
        target->au_state->op = MAG_OP_NOP; /* Detach from operations */
        memset(target->au_state->op_inputs, 0, sizeof(target->au_state->op_inputs)); /* Clear op inputs */
        memset(target->au_state->op_params, 0, sizeof(target->au_state->op_params));
    }
}

mag_tensor_t *mag_tensor_detach(mag_tensor_t *t) {
    mag_tensor_detach_inplace(t);
    return t;
}

int64_t mag_tensor_get_rank(const mag_tensor_t *t) {
    return t->rank;
}
const int64_t *mag_tensor_get_shape(const mag_tensor_t *t) {
    return t->shape;
}
const int64_t *mag_tensor_get_strides(const mag_tensor_t *t) {
    return t->strides;
}
mag_dtype_t mag_tensor_get_dtype(const mag_tensor_t *t) {
    return t->dtype;
}
size_t mag_tensor_get_data_offset(const mag_tensor_t *t) {
    return (size_t)t->storage_offset*t->storage->granularity; /* Return offset in bytes */
}
void *mag_tensor_get_data_ptr(const mag_tensor_t *t) {
    return (void *)(t->storage->base + mag_tensor_get_data_offset(t));
}
void *mag_tensor_get_storage_base_ptr(const mag_tensor_t *t) {
    return (void *)t->storage->base;
}

void *mag_tensor_get_raw_data_as_bytes(mag_tensor_t *t) {
    mag_tensor_t *cont;
    mag_status_t stat = mag_contiguous(&cont, t);
    if (mag_iserr(stat)) return NULL;
    size_t size = mag_tensor_get_data_size(cont);
    mag_assert2(size);
    void *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t *sto = cont->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(cont), dst, size);
    mag_tensor_decref(cont);
    return dst;
}

void mag_tensor_get_raw_data_as_bytes_free(void *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_e8m23_t *mag_tensor_get_data_as_floats(mag_tensor_t *t) {
    mag_tensor_t *cont;
    mag_status_t stat = mag_contiguous(&cont, t);
    if (mag_iserr(stat)) return NULL;
    mag_assert(mag_tensor_is_floating_point_typed(cont), "Tensor must be a floating point tensor, but has dtype: %s", mag_dtype_meta_of(t->dtype)->name);
    size_t size = cont->numel*sizeof(mag_e8m23_t);
    mag_assert2(size);
    mag_e8m23_t *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t *sto = cont->storage;
    if (cont->dtype == MAG_DTYPE_E8M23) (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(cont), dst, size);
    else (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(cont), dst, size, MAG_DTYPE_E8M23);
    mag_tensor_decref(cont);
    return dst;
}

void mag_tensor_get_data_as_floats_free(mag_e8m23_t *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_e8m23_t mag_tensor_get_item_float(const mag_tensor_t *t) {
    mag_storage_buffer_t *sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

int32_t mag_tensor_get_item_int(const mag_tensor_t *t) {
    mag_storage_buffer_t *sto = t->storage;
    int32_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_I32);
    return val;
}

bool mag_tensor_get_item_bool(const mag_tensor_t *t) {
    mag_storage_buffer_t *sto = t->storage;
    uint8_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_BOOL);
    return !!val;
}

/*
** Load all 6 elements of a 6-element array into local storage.
** Used for compute kernels to help the compiler to hold shape and stride values inside registers.
*/
#define mag_load_local_storage_group_arr(arr, prefix) \
    const int64_t prefix##0 = (arr)[0]; \
    const int64_t prefix##1 = (arr)[1]; \
    const int64_t prefix##2 = (arr)[2]; \
    const int64_t prefix##3 = (arr)[3]; \
    const int64_t prefix##4 = (arr)[4]; \
    const int64_t prefix##5 = (arr)[5]; \
    (void)prefix##0; \
    (void)prefix##1; \
    (void)prefix##2; \
    (void)prefix##3; \
    (void)prefix##4; \
    (void)prefix##5

#define mag_load_local_storage_group(xk, prefix, var) mag_load_local_storage_group_arr((xk)->var, prefix)

/* Compute dot product of 6 integers. Used to compute offsets in 6-dimensional index space. */
#define mag_address_dotprod6(x,y) ((x##0*y##0)+(x##1*y##1)+(x##2*y##2)+(x##3*y##3)+(x##4*y##4)+(x##5*y##5))

mag_e8m23_t mag_tensor_subscript_get_multi(mag_tensor_t *t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_storage_buffer_t *sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t) + sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

void mag_tensor_subscript_set_multi(mag_tensor_t *t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, mag_e8m23_t val) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_storage_buffer_t *sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, mag_tensor_get_data_offset(t) + sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val), MAG_DTYPE_E8M23);
}

static MAG_AINLINE void mag_tensor_unravel_index(const mag_tensor_t *t, int64_t v_idx, int64_t(*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

mag_e8m23_t mag_tensor_subscript_get_flattened(mag_tensor_t *t, int64_t idx) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        return mag_tensor_subscript_get_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    mag_storage_buffer_t *sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t) + sto->granularity*(size_t)idx, &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

void mag_tensor_subscript_set_flattened(mag_tensor_t *t, int64_t idx, mag_e8m23_t val) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        mag_tensor_subscript_set_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], val);
        return;
    }
    mag_storage_buffer_t *sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, mag_tensor_get_data_offset(t) + sto->granularity*(size_t)idx, &val, sizeof(val), MAG_DTYPE_E8M23);
}

static void mag_fmt_single_elem(mag_sstream_t *ss, const void *buf, size_t i, mag_dtype_t dtype) {
    switch (dtype) {
    case MAG_DTYPE_E8M23:
    case MAG_DTYPE_E5M10:
        mag_sstream_append(ss, "%g", (mag_e11m52_t)((const mag_e8m23_t *)buf)[i]);
        return;
    case MAG_DTYPE_BOOL:
        mag_sstream_append(ss, "%s", ((const uint8_t *)buf)[i] ? "True" : "False");
        return;
    case MAG_DTYPE_I32:
        mag_sstream_append(ss, "%" PRIi32, ((const int32_t *)buf)[i]);
        return;
    default:
        mag_panic("DType formatting not implemented: %d", dtype);
    }
}

static void mag_tensor_fmt_recursive(
    mag_sstream_t *ss,
    const void *buf,
    mag_dtype_t dtype,
    const int64_t *shape,
    const int64_t *strides,
    int64_t rank,
    int depth,
    int64_t moff,
    size_t pad
) {
    if (depth == rank) { /* scalar leaf */
        mag_fmt_single_elem(ss, buf, moff, dtype);
        return;
    }
    mag_sstream_putc(ss, '[');
    for (int64_t i=0; i < shape[depth]; ++i) {
        mag_tensor_fmt_recursive(ss, buf, dtype, shape, strides, rank, depth+1, moff + i*strides[depth], pad); /* Recurse down */
        if (i != shape[depth]-1) { /* separator */
            mag_sstream_putc(ss, ',');
            if (rank-depth > 1) { /* newline + indent for outer dims */
                mag_sstream_append(ss, "\n%*s", pad, ""); /* indent */
                for (int j=0; j <= depth; ++j)
                    mag_sstream_putc(ss, ' ');
            } else { /* simple space for last dim */
                mag_sstream_putc(ss, ' ');
            }
        }
    }
    mag_sstream_putc(ss, ']');
}

char *mag_tensor_to_string(mag_tensor_t *t, bool with_header, size_t from_start_count, size_t from_end_count) {
    if (!from_end_count) from_end_count = UINT64_MAX;
    void *buf = NULL;
    if (mag_tensor_is_floating_point_typed(t)) /* For all float types we want a (maybe converted) fp32 buffer for easy formatting. */
        buf = mag_tensor_get_data_as_floats(t);
    else /* Integral types can be formated easily */
        buf = mag_tensor_get_raw_data_as_bytes(t);
    mag_sstream_t ss;
    mag_sstream_init(&ss);
    const char *prefix = "Tensor(";
    size_t pad = strlen(prefix);
    mag_sstream_append(&ss, prefix);
    mag_tensor_fmt_recursive(&ss, buf, t->dtype, t->shape, t->strides, t->rank, 0, 0, pad); /* Recursive format */
    mag_sstream_append(&ss, ", dtype=%s, device=%s)", mag_dtype_meta_of(t->dtype)->name, t->ctx->device->id);
    /* Free allocated buffer */
    if (mag_tensor_is_floating_point_typed(t)) mag_tensor_get_data_as_floats_free(buf);
    else mag_tensor_get_raw_data_as_bytes_free(buf);
    return ss.buf; /* Return the string, must be freed with mag_tensor_to_string_free_data. */
}

void mag_tensor_to_string_free_data(char *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_context_t *mag_tensor_get_ctx(const mag_tensor_t *t) {
    return t->ctx;
}

bool mag_tensor_is_view(const mag_tensor_t *t) {
    return t->flags & MAG_TFLAG_IS_VIEW;
}

bool mag_tensor_is_floating_point_typed(const mag_tensor_t *t) {
    return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_FP;
}

bool mag_tensor_is_integral_typed(const mag_tensor_t *t) {
    return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_INTEGRAL;
}

bool mag_tensor_is_integer_typed(const mag_tensor_t *t) {
    return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_INTEGER;
}

bool mag_tensor_is_numeric_typed(const mag_tensor_t *t) {
    return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_NUMERIC;
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
        mag_fmt_shape(&shape, &leaked->shape, leaked->rank);
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
