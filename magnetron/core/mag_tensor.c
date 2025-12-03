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
#include "mag_fmt.h"

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
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
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
        tensor->coords.shape[i] = i < rank ? shape[i] : 1;
        tensor->coords.strides[i] = 1;
    }
    /* Compute contiguous row-major strides and check for overflow. */
    tensor->coords.strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
        mag_contract(ctx, ERR_DIM_OVERFLOW, { mag_tensor_free_header(tensor); *out = NULL; }, !mag_mulov64(tensor->coords.strides[i+1], tensor->coords.shape[i+1], tensor->coords.strides+i), "Stride overflowed at dim[%" PRIi64 "]", i);
    }
    ++ctx->num_created_tensors;
    *out = tensor;
    return MAG_STATUS_OK;
}

mag_status_t mag_as_strided(mag_tensor_t **out, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
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
        tensor->coords.shape[i] = i < rank ? shape[i] : 1;
        tensor->coords.strides[i] = i < rank ? strides[i] : 1;
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

int64_t mag_tensor_numbytes(const mag_tensor_t *t) {
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
    mag_tensor_t *cont;
    mag_status_t stat = mag_contiguous(&cont, tensor);
    if (mag_iserr(stat)) return NULL;
    size_t size = mag_tensor_numbytes(cont);
    mag_assert2(size);
    void *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t *sto = cont->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(cont), dst, size);
    mag_rc_decref(cont);
    return dst;
}

void mag_tensor_copy_data_free(void *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

float *mag_tensor_copy_float_data(mag_tensor_t *tensor) {
    mag_tensor_t *cont;
    mag_status_t stat = mag_contiguous(&cont, tensor);
    if (mag_iserr(stat)) return NULL;
    mag_assert(mag_tensor_is_floating_point_typed(cont), "Tensor must be a floating point tensor, but has dtype: %s", mag_type_trait(tensor->dtype)->name);
    size_t size = cont->numel*sizeof(float);
    mag_assert2(size);
    float *dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_storage_buffer_t *sto = cont->storage;
    if (cont->dtype == MAG_DTYPE_FLOAT32) (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(cont), dst, size);
    else (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(cont), dst, size, MAG_DTYPE_FLOAT32);
    mag_rc_decref(cont);
    return dst;
}

void mag_tensor_copy_float_data_free(float *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

float mag_tensor_item_float(const mag_tensor_t *tensor) {
    mag_storage_buffer_t *sto = tensor->storage;
    float val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(tensor), &val, sizeof(val), MAG_DTYPE_FLOAT32);
    return val;
}

int64_t mag_tensor_item_int(const mag_tensor_t *tensor) {
    mag_storage_buffer_t *sto = tensor->storage;
    int64_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(tensor), &val, sizeof(val), MAG_DTYPE_INT64);
    return val;
}

bool mag_tensor_item_bool(const mag_tensor_t *tensor) {
    mag_storage_buffer_t *sto = tensor->storage;
    uint8_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_data_offset(tensor), &val, sizeof(val), MAG_DTYPE_BOOLEAN);
    return !!val;
}

static void mag_fmt_single_elem(mag_sstream_t *ss, const void *buf, size_t i, mag_dtype_t dtype) {
    char fmt[MAG_FMT_BUF_MAX] = {0};
    char *e = NULL;
    switch (dtype) {
        case MAG_DTYPE_FLOAT32:
        case MAG_DTYPE_FLOAT16: e = mag_fmt_e11m52(fmt, ((const float *)buf)[i], MAG_FMT_G5); break;
        case MAG_DTYPE_BOOLEAN: mag_sstream_append(ss, "%s", ((const uint8_t *)buf)[i] ? "True" : "False"); return;
        case MAG_DTYPE_UINT8:  e = mag_fmt_uint64(fmt, ((const uint8_t *)buf)[i]); break;
        case MAG_DTYPE_INT8:  e = mag_fmt_int64(fmt, ((const int8_t *)buf)[i]); break;
        case MAG_DTYPE_UINT16: e = mag_fmt_uint64(fmt, ((const uint16_t *)buf)[i]); break;
        case MAG_DTYPE_INT16: e = mag_fmt_int64(fmt, ((const int16_t *)buf)[i]); break;
        case MAG_DTYPE_UINT32: e = mag_fmt_uint64(fmt, ((const uint32_t *)buf)[i]); break;
        case MAG_DTYPE_INT32: e = mag_fmt_int64(fmt, ((const int32_t *)buf)[i]); break;
        case MAG_DTYPE_UINT64: e = mag_fmt_uint64(fmt, ((const uint64_t *)buf)[i]); break;
        case MAG_DTYPE_INT64: e = mag_fmt_int64(fmt, ((const int64_t *)buf)[i]); break;
        default: mag_panic("Unknown dtype for formatting: %d", dtype); return;
    }
    mag_assert2(e);
    *e = '\0';
    ptrdiff_t n = e-fmt;
    if (mag_likely(n > 0))
        mag_sstream_append_strn(ss, fmt, e-fmt);
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

char *mag_tensor_to_string(mag_tensor_t *tensor, bool with_header, size_t from_start_count, size_t from_end_count) {
    if (!from_end_count) from_end_count = UINT64_MAX;
    void *buf = NULL;
    if (mag_tensor_is_floating_point_typed(tensor)) /* For all float types we want a (maybe converted) fp32 buffer for easy formatting. */
        buf = mag_tensor_copy_float_data(tensor);
    else /* Integral types can be formated easily */
        buf = mag_tensor_copy_data(tensor);
    mag_sstream_t ss;
    mag_sstream_init(&ss);
    const char *prefix = "Tensor(";
    size_t pad = strlen(prefix);
    mag_sstream_append(&ss, prefix);
    mag_tensor_fmt_recursive(&ss, buf, tensor->dtype, tensor->coords.shape, tensor->coords.strides, tensor->coords.rank, 0, 0, pad); /* Recursive format */
    mag_sstream_putc(&ss, ')');
    /* Free allocated buffer */
    if (mag_tensor_is_floating_point_typed(tensor)) mag_tensor_copy_float_data_free(buf);
    else mag_tensor_copy_data_free(buf);
    return ss.buf; /* Return the string, must be freed with mag_tensor_to_string_free_data. */
}

void mag_tensor_to_string_free_data(char *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
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
    return a->numel == b->numel && a->numel == c->numel &&
           mag_tensor_is_contiguous(a) &&
           mag_tensor_is_contiguous(b) &&
           mag_tensor_is_contiguous(c);
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
